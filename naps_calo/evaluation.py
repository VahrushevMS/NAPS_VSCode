"""Ячейка 11: новый held-out test или сохранённые исходные результаты."""

import numpy as np
import pandas as pd
from scipy.special import expit

from .inference import predict_logits
from .metrics import classifier_metrics, membership_metrics, conditional_audit
from .models import statistics, encode_membership


def evaluate(settings, checkpoint, model, scaler, data, indices, predictor):
    audit = pd.DataFrame()
    audit_summary = pd.DataFrame()
    if data is not None:
        test_ids = indices["test"]
        y = data["y"][test_ids]
        logits = predict_logits(
            model, data["X"], scaler, batch_size=settings.batch_size, indices=test_ids
        )
        membership = (
            statistics(logits, checkpoint["train_prior_1"])[:, None, :]
            > predictor.thresholds[None, :, :]
        )
        rows = []
        for scenario, mask in [
            ("Reference", np.ones(len(y), dtype=bool)),
            ("Low energy E<=5", data["E"][test_ids] <= 5),
        ]:
            if not mask.any():
                continue
            for ia, alpha in enumerate(predictor.alphas):
                rows.append(
                    {
                        "evaluation_source": "current_held_out_data",
                        "scenario": scenario,
                        "alpha": float(alpha),
                        **membership_metrics(y[mask], membership[mask, ia]),
                    }
                )
        test_metrics = pd.DataFrame(rows)
        classifier = pd.DataFrame(
            [
                {
                    "evaluation_source": "current_held_out_data",
                    **classifier_metrics(y, logits),
                }
            ]
        )
        # Use the active checkpoint levels for the audit as well as inference.
        settings.alphas = predictor.alphas.copy()
        audit = conditional_audit(
            settings,
            y,
            membership,
            data["E"][test_ids],
            data["g"][test_ids],
            data["groups"][test_ids],
        )
        if not audit.empty:
            audit_summary = (
                audit.groupby(["family", "alpha"])
                .agg(
                    min_coverage=("coverage", "min"),
                    min_gap=("gap", "min"),
                    binomial_failures=(
                        "holm_binomial",
                        lambda p: int((p < settings.audit_significance).sum()),
                    ),
                    chernoff_failures=(
                        "holm_chernoff",
                        lambda p: int((p < settings.audit_significance).sum()),
                    ),
                )
                .reset_index()
            )
        sample_ids = test_ids[: min(512, len(test_ids))]
        probability, codes = predictor.predict(
            data["X"][sample_ids], settings.batch_size
        )
        if not np.array_equal(codes, encode_membership(membership[: len(sample_ids)])):
            raise AssertionError("Predictor отличается от test pipeline")
        if not np.allclose(
            probability, expit(logits[: len(sample_ids)]), rtol=1e-5, atol=1e-6
        ):
            raise AssertionError("Вероятности predictor отличаются от test pipeline")
        predictor.export(
            data["X"][sample_ids],
            settings.output_dir / "example_submission.csv",
            event_ids=sample_ids,
            batch_size=settings.batch_size,
        )
    else:
        saved_source = (
            "saved_checkpoint_test"
            if checkpoint.get("metrics_provenance")
            else "saved_original_test"
        )
        test_metrics = pd.DataFrame(checkpoint.get("saved_test_metrics", []))
        if not test_metrics.empty:
            test_metrics["evaluation_source"] = saved_source
            if {"ambiguous", "empty"} <= set(test_metrics):
                test_metrics["singleton"] = (
                    1 - test_metrics["ambiguous"] - test_metrics["empty"]
                )
            test_metrics = test_metrics.drop(columns=["pipeline"], errors="ignore")
        classifier = pd.DataFrame(checkpoint.get("saved_classifier_metrics", []))
        if not classifier.empty:
            classifier["evaluation_source"] = saved_source
        audit_summary = pd.DataFrame(checkpoint.get("saved_audit_summary", [])).drop(
            columns=["pipeline"], errors="ignore"
        )
        if not audit_summary.empty:
            audit_summary["evaluation_source"] = saved_source
    test_metrics.to_csv(settings.output_dir / "test_metrics.csv", index=False)
    classifier.to_csv(settings.output_dir / "classifier_metrics.csv", index=False)
    audit.to_csv(settings.output_dir / "conditional_coverage.csv", index=False)
    audit_summary.to_csv(settings.output_dir / "audit_summary.csv", index=False)
    return test_metrics, classifier, audit, audit_summary
