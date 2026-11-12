"""Проверки переноса из ноутбука и файлового интерфейса; unittest, без pytest."""

import ast
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from naps_calo.config import Settings
from naps_calo.models import preprocess, statistics, encode_membership
from naps_calo.inference import ResCNNNAPSPredictor, predict_logits
from naps_calo.data import load_dictionaries, build_splits, file_hash
from naps_calo.sessions import load_session


class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.weights = ROOT / "weights/NAPS_Calo_ResCNN_weights.pt"
        cls.checkpoint = torch.load(cls.weights, map_location="cpu", weights_only=True)
        cls.predictor = ResCNNNAPSPredictor(cls.checkpoint)
        cls.X, cls.ids = load_session(ROOT / "data/sessions/session_demo.npz")
        # Только тест воспроизводит определения из исходных ячеек, чтобы сравнить
        # перенесённый код с независимым оригиналом. Рабочий проект не исполняет ipynb.
        notebook = json.loads(
            (ROOT / "sources/original_notebook.ipynb").read_text(encoding="utf-8")
        )
        namespace = dict(
            np=np,
            torch=torch,
            nn=nn,
            math=math,
            pd=pd,
            Path=Path,
            expit=expit,
            DEVICE=torch.device("cpu"),
        )
        exec("".join(notebook["cells"][2]["source"]), namespace)
        tree = ast.parse("".join(notebook["cells"][4]["source"]))
        definitions = [
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        ]
        exec(
            compile(
                ast.Module(body=definitions, type_ignores=[]),
                "<original-cell-5>",
                "exec",
            ),
            namespace,
        )
        cls.original = namespace
        cls.original_predictor = namespace["ResCNNNAPSPredictor"](
            cls.checkpoint, device="cpu"
        )

    def test_preprocessing_matches_original(self):
        original = self.original["preprocess"](self.X, self.checkpoint["preprocess"])
        migrated = preprocess(self.X, self.checkpoint["preprocess"])
        for left, right in zip(original, migrated):
            np.testing.assert_array_equal(left, right)

    def test_predictions_match_original(self):
        p_old, code_old = self.original_predictor.predict(self.X, batch_size=7)
        p_new, code_new = self.predictor.predict(self.X, batch_size=7)
        np.testing.assert_array_equal(p_old, p_new)
        np.testing.assert_array_equal(code_old, code_new)
        logits = predict_logits(
            self.predictor.model, self.X, self.predictor.scaler, batch_size=7
        )
        np.testing.assert_allclose(expit(logits), p_new, rtol=0, atol=0)
        sets = (
            statistics(logits, self.predictor.prior)[:, None, :]
            > self.predictor.thresholds
        )
        np.testing.assert_array_equal(encode_membership(sets), code_new)

    def test_membership_is_nested(self):
        _, code = self.predictor.predict(self.X)
        np.testing.assert_array_equal(code[:, 1:] & code[:, :-1], code[:, 1:])
        self.assertTrue(np.isin(code, [0, 1, 2, 3]).all())

    def test_export_preserves_identifiers(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "submission.csv"
            exported = self.predictor.export(self.X, path, self.ids)
            reread = pd.read_csv(path)
            self.assertEqual(list(reread), ["event_id", "H_0.05", "H_0.10", "H_0.20"])
            np.testing.assert_array_equal(reread.event_id, self.ids)
            pd.testing.assert_frame_equal(reread, exported, check_dtype=False)

    def test_empty_session(self):
        X = np.empty((0, 4, 8, 8), np.float32)
        p, code = self.predictor.predict(X)
        self.assertEqual(p.shape, (0,))
        self.assertEqual(code.shape, (0, 3))

    def test_invalid_energies_fail(self):
        for value in (-1, np.nan, np.inf):
            with self.subTest(value=value):
                X = self.X[:1].copy()
                X[0, 0, 0, 0] = value
                with self.assertRaisesRegex(ValueError, "неотрицательные"):
                    self.predictor.predict(X)

    def test_bad_dimensions_and_event_ids_fail(self):
        with self.assertRaises(ValueError):
            self.predictor.predict(np.zeros((2, 8, 8)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            np.savez(path, X=self.X[:2], event_id=[1])
            with self.assertRaisesRegex(ValueError, "event_id"):
                load_session(path)

    def test_session_dictionary_form_and_single_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "input.npz"
            np.savez(
                path,
                values=self.X[:6].reshape(2, 3, 4, 8, 8),
                event_id=self.ids[:6].reshape(2, 3),
            )
            raw, ids = load_session(path)
            np.testing.assert_array_equal(raw, self.X[:6])
            np.testing.assert_array_equal(ids, self.ids[:6])
            np.savez(path, X=self.X[0])
            raw, ids = load_session(path)
            self.assertEqual(raw.shape, (1, 4, 8, 8))

    def test_group_splits_and_manifest_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            settings = Settings(
                mode="train",
                data_dir=folder,
                cache_dir=folder / "cache",
                output_dir=folder / "outputs",
                device_name="cpu",
            ).initialize()
            n = 120
            conditions = np.column_stack(
                [np.geomspace(1, 100, n), np.linspace(0.7, 1.3, n)]
            )
            keys = np.array([[label, E, g] for E, g in conditions for label in (0, 1)])
            values = np.ones((len(keys), 2, 4, 8, 8), dtype=np.float32)
            path = folder / "calo_dictionary_test.npz"
            np.savez_compressed(path, keys=keys, values=values)
            data = load_dictionaries(settings, folder)
            indices = build_splits(settings, data, {})
            self.assertEqual(sum(map(len, indices.values())), len(data["y"]))
            for role, ids in indices.items():
                self.assertEqual(data["y"][ids].mean(), 0.5)
                for other_role, other in indices.items():
                    if other_role != role:
                        self.assertFalse(
                            np.intersect1d(
                                data["groups"][ids], data["groups"][other]
                            ).size
                        )
            with self.assertRaisesRegex(ValueError, "исходные NPZ"):
                load_dictionaries(settings, folder, [{"sha256": "not-the-original"}])
            # Точная копия одного физического файла не должна удваивать бюджет.
            (folder / "calo_dictionary_copy.npz").write_bytes(path.read_bytes())
            data["X"]._mmap.close()
            duplicated = load_dictionaries(settings, folder)
            self.assertEqual(len(duplicated["y"]), len(data["y"]))
            duplicated["X"]._mmap.close()

    def test_launcher_from_another_working_directory(self):
        before = file_hash(self.weights)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "main.py"),
                    "--check",
                    "--device",
                    "cpu",
                    "--output-dir",
                    str(Path(tmp) / "outputs"),
                    "--cache-dir",
                    str(Path(tmp) / "cache"),
                ],
                cwd=tmp,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=90,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("inference OK", result.stdout)
        self.assertEqual(before, file_hash(self.weights))


if __name__ == "__main__":
    unittest.main(verbosity=2)
