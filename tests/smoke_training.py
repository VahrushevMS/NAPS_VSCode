"""Полный короткий прогон трёх режимов на синтетических данных, НЕ оценка качества.

Запуск: python tests/smoke_training.py
Нужен CatBoost. Временные файлы удаляются после успешного завершения.
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from naps_calo.data import file_hash


def make_simulations(folder):
    rng = np.random.default_rng(77)
    E_edges = np.geomspace(1, 100, 7)
    g_edges = np.linspace(0.7, 1.3, 5)
    coordinates = np.linspace(-1, 1, 8)
    xx, yy = np.meshgrid(coordinates, coordinates)
    keys, values = [], []
    # 384 независимых условия, 12 событий каждого класса: 9 216 событий.
    for ie in range(6):
        for ig in range(4):
            for _ in range(16):
                E = float(
                    np.exp(rng.uniform(np.log(E_edges[ie]), np.log(E_edges[ie + 1])))
                )
                g = float(rng.uniform(g_edges[ig], g_edges[ig + 1]))
                for label in (0, 1):
                    keys.append([label, E, g])
                    width = 0.20 if label == 0 else 0.60
                    profile = np.exp(-(xx**2 + yy**2) / (2 * width**2))
                    layer_weights = np.array(
                        [0.25, 0.5, 0.2, 0.05] if label == 0 else [0.05, 0.15, 0.4, 0.4]
                    )
                    block = profile[None, None] * layer_weights[None, :, None, None]
                    block = block * rng.lognormal(0, 0.2, (12, 4, 8, 8))
                    block[rng.random(block.shape) < 0.05] = 0
                    block *= (E * g / block.sum(axis=(1, 2, 3)))[:, None, None, None]
                    values.append(block.astype(np.float32))
    path = folder / "calo_dictionary_smoke.npz"
    np.savez_compressed(path, keys=np.asarray(keys), values=np.stack(values))
    return path


def main():
    start = time.monotonic()
    original_hash = file_hash(ROOT / "weights/NAPS_Calo_ResCNN_weights.pt")
    with tempfile.TemporaryDirectory(prefix="naps_training_smoke_") as temp:
        folder = Path(temp)
        make_simulations(folder)
        outputs = folder / "outputs"
        common = [
            "--data-dir",
            str(folder),
            "--cache-dir",
            str(folder / "cache"),
            "--device",
            "cpu",
            "--max-epochs",
            "2",
            "--w-iterations",
            "20",
            "--debug",
        ]

        def run(role, flags):
            out = outputs / role
            command = [
                sys.executable,
                str(ROOT / "main.py"),
                *common,
                "--output-dir",
                str(out),
                *flags,
            ]
            print(f"\n=== Synthetic smoke: {role} ===", flush=True)
            # Файловый лог не забивает терминал большими таблицами.
            log = folder / f"{role}.log"
            with log.open("w", encoding="utf-8") as stream:
                result = subprocess.run(
                    command,
                    cwd=folder,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    timeout=600,
                    text=True,
                    encoding="utf-8",
                )
            if result.returncode:
                print(log.read_text(encoding="utf-8"))
                raise RuntimeError(f"Не прошёл {role}, exit={result.returncode}")
            print(log.read_text(encoding="utf-8")[-2200:], flush=True)
            return out

        train = run("train", ["--mode", "train"])
        trained_weights = train / "NAPS_Calo_ResCNN_gamma0.pt"
        checkpoint = torch.load(trained_weights, map_location="cpu", weights_only=True)
        assert len(checkpoint["training_history"]) == 2
        assert checkpoint["naps"]["gamma"] == 0
        assert checkpoint["saved_test_metrics"]
        assert checkpoint["provenance"]["optimizer"] == "Adam"
        # Независимость всех событий и всех условий между частями.
        parts = [
            indices.numpy() for indices in checkpoint["source_split_indices"].values()
        ]
        all_ids = np.concatenate(parts)
        assert len(all_ids) == 9216 and len(np.unique(all_ids)) == len(all_ids)
        submission = pd.read_csv(train / "submission_session_demo.csv")
        assert len(submission) == 32
        assert np.isin(submission.iloc[:, 1:].to_numpy(), [0, 1, 2, 3]).all()

        evaluate = run(
            "evaluate",
            [
                "--mode",
                "pretrained",
                "--evaluate-data",
                "--weights",
                str(trained_weights),
            ],
        )
        train_metrics = pd.read_csv(train / "test_metrics.csv")
        evaluated_metrics = pd.read_csv(evaluate / "test_metrics.csv")
        pd.testing.assert_frame_equal(
            train_metrics, evaluated_metrics, rtol=1e-7, atol=1e-9
        )
        pd.testing.assert_frame_equal(
            pd.read_csv(train / "classifier_metrics.csv"),
            pd.read_csv(evaluate / "classifier_metrics.csv"),
            rtol=1e-7,
            atol=1e-9,
        )
        # Перекалибровка меняет пороги, но не обученные веса сети.
        recalibrate = run(
            "recalibrate", ["--mode", "recalibrate", "--weights", str(trained_weights)]
        )
        new = torch.load(
            recalibrate / "NAPS_Calo_ResCNN_gamma0.pt",
            map_location="cpu",
            weights_only=True,
        )
        for key in checkpoint["state_dict"]:
            torch.testing.assert_close(
                new["state_dict"][key], checkpoint["state_dict"][key], rtol=0, atol=0
            )
        # Сохранённый checkpoint запускается и без тренировочных данных.
        saved = run(
            "saved", ["--mode", "pretrained", "--weights", str(trained_weights)]
        )
        assert (
            json.loads((saved / "run_metadata.json").read_text())["metric_source"]
            == "saved_checkpoint_test"
        )
        assert all(
            (train / name).is_file()
            for name in [
                "W_0.cbm",
                "W_1.cbm",
                "guard.csv",
                "thresholds.csv",
                "figures/coverage_power.png",
            ]
        )
    assert file_hash(ROOT / "weights/NAPS_Calo_ResCNN_weights.pt") == original_hash
    print(
        f"\nSMOKE OK: train, held-out evaluation, recalibrate, saved checkpoint; {time.monotonic() - start:.1f} s."
    )
    print("Искусственные данные использовались только для проверки работы кода.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
