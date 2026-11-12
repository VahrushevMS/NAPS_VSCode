"""Ячейки 4, 6, 8, 10-13: явное управление этапами проекта."""

import copy
import importlib.util
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import load_dictionaries, build_splits, file_hash
from .inference import ResCNNNAPSPredictor, predict_logits
from .models import CaloResCNN
from .evaluation import evaluate
from .reporting import print_frame
from .sessions import export_sessions


def load_checkpoint(path, device):
    if not path.is_file():
        raise FileNotFoundError(
            f"Нет файла весов: {path}. Сохраните папку weights из архива"
        )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("Нужен checkpoint формата NAPS_Calo_ResCNN_weights.pt")
    required = {
        "config",
        "state_dict",
        "preprocess",
        "train_prior_1",
        "naps",
        "data_manifest",
        "source_split_indices",
    }
    if not required <= set(checkpoint):
        raise ValueError(
            f"В checkpoint отсутствуют поля: {sorted(required - set(checkpoint))}"
        )
    model = CaloResCNN(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return checkpoint, model


def train_new_checkpoint(settings, source, data, indices):
    from .training import fit_preprocessing, train_rescnn

    config = copy.deepcopy(source["config"])
    scaler = fit_preprocessing(data["X"], indices["train"])
    model, history, state = train_rescnn(settings, data, indices, config, scaler)
    checkpoint = {
        "format_version": 1,
        "model_name": "ResCNN",
        "config": config,
        "seed": settings.seed,
        "state_dict": state,
        "train_prior_1": float(data["y"][indices["train"]].mean()),
        "preprocess": scaler,
        "data_manifest": data["manifest"],
        "source_split_indices": {
            key: torch.from_numpy(value.copy()) for key, value in indices.items()
        },
        "training_history": history.to_dict("records"),
        "saved_test_metrics": [],
        "saved_classifier_metrics": [],
        "saved_audit_summary": [],
        "naps": None,
        "budget": {"physical_used": len(data["y"]), "new_physical_simulations": 0},
        "provenance": {
            "optimizer": "Adam",
            "weights_retrained_during_export": True,
            "source": "VS Code version of supplied notebook",
        },
    }
    history.to_csv(settings.output_dir / "training_history.csv", index=False)
    return checkpoint, model, history


def recalibrate_checkpoint(settings, checkpoint, model, data, indices):
    from .calibration import (
        fit_rejection_probabilities,
        worst_case_cutoffs,
        calibrate_margins,
    )

    prior, scaler = checkpoint["train_prior_1"], checkpoint["preprocess"]
    print("\nОценка сети на w_cal...", flush=True)
    w_logits = predict_logits(
        model, data["X"], scaler, settings.batch_size, indices["w_cal"]
    )
    w_models, w_bounds, validation = fit_rejection_probabilities(
        settings, data, indices, prior, w_logits
    )
    print_frame("Качество моделей W", validation)
    validation.to_csv(settings.output_dir / "W_validation.csv", index=False)
    print("\nПоиск худших nuisance-условий...", flush=True)
    raw_cutoffs, worst_cells, _ = worst_case_cutoffs(settings, w_models, w_bounds)
    guard_logits = predict_logits(
        model, data["X"], scaler, settings.batch_size, indices["guard"]
    )
    margins, guard_table = calibrate_margins(
        settings, data, prior, guard_logits, raw_cutoffs, indices["guard"]
    )
    guard_table.to_csv(settings.output_dir / "guard.csv", index=False)
    adjusted = raw_cutoffs - margins
    thresholds = np.minimum.accumulate(adjusted[::-1], axis=0)[::-1].copy()
    checkpoint["naps"] = {
        "alphas": torch.from_numpy(settings.alphas.copy()),
        "gamma": settings.recalibration_gamma,
        "confidence_set": "full_nuisance_domain",
        "nuisance_domain": {"E": [1.0, 100.0], "g": [0.7, 1.3]},
        "raw_cutoffs": torch.from_numpy(raw_cutoffs),
        "margins": torch.from_numpy(margins),
        "thresholds": torch.from_numpy(thresholds),
        "worst_cells_normalized": torch.from_numpy(worst_cells),
        "continuum_certified": False,
        "coverage_scope": "bin-average under current simulation design",
        "calibration_source": "new gamma=0 rejection probability + independent bin guard",
    }
    checkpoint["source_split_indices"] = {
        key: torch.from_numpy(value.copy()) for key, value in indices.items()
    }
    for key in [
        "saved_test_metrics",
        "saved_classifier_metrics",
        "saved_audit_summary",
    ]:
        checkpoint[key] = []
    return checkpoint


def threshold_table(checkpoint, predictor):
    rows = []
    worst = checkpoint["naps"]["worst_cells_normalized"].cpu().numpy()
    for ia, alpha in enumerate(predictor.alphas):
        for label in (0, 1):
            cell = worst[ia, label]
            rows.append(
                {
                    "alpha": float(alpha),
                    "y": label,
                    "raw_cutoff": float(checkpoint["naps"]["raw_cutoffs"][ia, label]),
                    "margin": float(checkpoint["naps"]["margins"][ia, label]),
                    "final_cutoff": float(predictor.thresholds[ia, label]),
                    "E_low": float(100 ** cell[0, 0]),
                    "E_high": float(100 ** cell[1, 0]),
                    "g_low": float(0.7 + 0.6 * cell[0, 1]),
                    "g_high": float(0.7 + 0.6 * cell[1, 1]),
                }
            )
    return pd.DataFrame(rows)


def run_pipeline(settings, session_paths=None, check_only=False):
    print(
        f"NAPS / ResCNN | режим={settings.mode} | устройство={settings.device}",
        flush=True,
    )
    source, model = load_checkpoint(settings.weights_path, settings.device)
    checkpoint = copy.deepcopy(source)
    if check_only:
        predictor = ResCNNNAPSPredictor(checkpoint, device=str(settings.device))
        probability, codes = predictor.predict(np.zeros((1, 4, 8, 8), dtype=np.float32))
        if not np.isfinite(probability).all() or codes.shape != (
            1,
            len(predictor.alphas),
        ):
            raise AssertionError("Не прошла проверка загрузки и inference")
        print(
            f"Загрузка и inference OK. Параметров: {sum(p.numel() for p in model.parameters())}."
        )
        return {"check": "ok"}
    if (
        settings.mode in {"train", "recalibrate"}
        and importlib.util.find_spec("catboost") is None
    ):
        raise ModuleNotFoundError(
            "Для этого режима нужен CatBoost. Запустите python bootstrap.py --with-training --install-only"
        )
    data = indices = None
    if settings.mode != "pretrained" or settings.evaluate_data:
        expected = None if settings.mode == "train" else checkpoint["data_manifest"]
        print(f"\nЧтение симуляций: {settings.data_dir}", flush=True)
        data = load_dictionaries(settings, settings.data_dir, expected)
        if data is None:
            raise FileNotFoundError(
                f"Нет calo_dictionary*.npz в {settings.data_dir}. Для обычного вывода запустите --mode pretrained без --evaluate-data"
            )
        indices = build_splits(settings, data, checkpoint)
        split = pd.DataFrame(
            [
                {
                    "role": key,
                    "events": len(ids),
                    "conditions": len(np.unique(data["groups"][ids])),
                }
                for key, ids in indices.items()
            ]
        )
        print_frame("Разбиение данных", split)
        split.to_csv(settings.output_dir / "split.csv", index=False)
        np.savez_compressed(settings.output_dir / "split_indices.npz", **indices)
        print(
            {
                "physical_events": len(data["y"]),
                "new_simulations": 0,
                "remaining_budget": settings.max_budget - len(data["y"]),
            },
            flush=True,
        )
    history = pd.DataFrame(checkpoint.get("training_history", []))
    if settings.mode == "train":
        checkpoint, model, history = train_new_checkpoint(
            settings, checkpoint, data, indices
        )
    if settings.mode in {"train", "recalibrate"}:
        checkpoint = recalibrate_checkpoint(settings, checkpoint, model, data, indices)
        # Save an intermediate usable checkpoint before test evaluation.
        torch.save(checkpoint, settings.output_dir / "NAPS_Calo_ResCNN_gamma0.pt")
    predictor = ResCNNNAPSPredictor(checkpoint, device=str(settings.device))
    settings.alphas = predictor.alphas.copy()
    thresholds = threshold_table(checkpoint, predictor)
    thresholds.to_csv(settings.output_dir / "thresholds.csv", index=False)
    print_frame("Пороги NAPS", thresholds)
    print(
        {
            "active_gamma": checkpoint["naps"]["gamma"],
            "confidence_set": checkpoint["naps"]["confidence_set"],
            "continuum_certified": checkpoint["naps"]["continuum_certified"],
            "coverage_scope": checkpoint["naps"]["coverage_scope"],
        },
        flush=True,
    )
    test_metrics, classifier, audit, summary = evaluate(
        settings, checkpoint, model, checkpoint["preprocess"], data, indices, predictor
    )
    if data is None:
        print(
            "\nПоказаны сохранённые тестовые результаты из файла весов. Новый test не запускался.",
            flush=True,
        )
    print_frame("Классификатор", classifier)
    print_frame("NAPS: coverage и power", test_metrics.round(6))
    print_frame("Аудит покрытия", summary.round(6))
    if not test_metrics.empty:
        reference = test_metrics.loc[test_metrics["scenario"] == "Reference"]
        if not reference.empty:
            print(
                {
                    "mean_power_over_alphas": float(reference["power"].mean()),
                    "evaluation_source": reference["evaluation_source"].iloc[0],
                },
                flush=True,
            )
    if settings.make_plots:
        from .plotting import save_plots

        save_plots(settings, test_metrics, history, audit)
    sessions = export_sessions(settings, predictor, session_paths)
    print_frame("Готовые CSV сессий", sessions)
    if sessions.empty:
        print(
            f"Поместите session*.npz в {settings.session_dir} или укажите --session путь_к_файлу.npz",
            flush=True,
        )
    if settings.mode in {"train", "recalibrate"}:
        # Store NEW metrics only with a NEW checkpoint; supplied weights stay untouched.
        checkpoint["saved_test_metrics"] = test_metrics.to_dict("records")
        checkpoint["saved_classifier_metrics"] = classifier.to_dict("records")
        checkpoint["saved_audit_summary"] = summary.to_dict("records")
        checkpoint["metrics_provenance"] = {
            "evaluation_source": "current_held_out_data",
            "mode": settings.mode,
        }
        torch.save(checkpoint, settings.output_dir / "NAPS_Calo_ResCNN_gamma0.pt")
    metadata = {
        "mode": settings.mode,
        "device": str(settings.device),
        "weights": str(settings.weights_path),
        "weights_sha256": file_hash(settings.weights_path),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "metric_source": "current_held_out_data"
        if data is not None
        else (
            "saved_checkpoint_test"
            if checkpoint.get("metrics_provenance")
            else "saved_original_test"
        ),
        "input_session_count": len(sessions),
        "alphas": predictor.alphas.tolist(),
        "gamma": float(checkpoint["naps"]["gamma"]),
        "continuum_certified": bool(checkpoint["naps"]["continuum_certified"]),
    }
    (settings.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nГотово. Результаты: {settings.output_dir}", flush=True)
    return metadata
