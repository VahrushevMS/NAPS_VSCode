"""Ячейка 7: метрики, KL и поправка Holm."""

import math
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import binomtest
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
from .data import nuisance_bins


def membership_metrics(y, membership):
    y = np.asarray(y, dtype=np.int64)
    m = np.asarray(membership, dtype=bool)
    if len(y) == 0 or m.shape != (len(y), 2) or (not np.isin(y, [0, 1]).all()):
        raise ValueError("Нужны непустые y и membership формы (N,2)")
    covered = m[np.arange(len(y)), y]
    powered = ~m[np.arange(len(y)), 1 - y]
    sizes = m.sum(axis=1)
    class_powers = [
        powered[y == label].mean() for label in (0, 1) if np.any(y == label)
    ]
    result = {
        "n": len(y),
        "coverage": float(covered.mean()),
        "power": float(np.mean(class_powers)),
        "empty": float((sizes == 0).mean()),
        "ambiguous": float((sizes == 2).mean()),
        "singleton": float((sizes == 1).mean()),
        "correct_singleton": float((covered & (sizes == 1)).mean()),
        "mean_size": float(sizes.mean()),
    }
    for label in (0, 1):
        mask = y == label
        result[f"coverage_{label}"] = (
            float(covered[mask].mean()) if mask.any() else np.nan
        )
        result[f"power_{label}"] = float(powered[mask].mean()) if mask.any() else np.nan
    return result


def classifier_metrics(y, logits):
    p = np.clip(expit(logits), 1e-15, 1 - 1e-15)
    return {
        "bce": float(log_loss(y, p, labels=[0, 1])),
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else np.nan,
        "accuracy": float(accuracy_score(y, p >= 0.5)),
    }


def bernoulli_kl(p, q):
    if p == q:
        return 0.0
    a = 0.0 if p == 0 else p * math.log(p / q)
    b = 0.0 if p == 1 else (1 - p) * math.log((1 - p) / (1 - q))
    return a + b


def kl_upper(k, n, delta):
    if n == 0 or k == n:
        return 1.0
    p = k / n
    target = math.log(1 / delta) / n
    lo, hi = (p, 1.0)
    for _ in range(60):
        mid = (lo + hi) / 2
        if mid == 1 or bernoulli_kl(p, mid) > target:
            hi = mid
        else:
            lo = mid
    return hi


def holm_adjust(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum(
        1, np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order])
    )
    out = np.empty_like(p)
    out[order] = adjusted
    return out


def conditional_audit(settings, y, membership, E, g, groups):
    bin_id = nuisance_bins(settings, E, g)
    results = []
    for family, grouping in (
        ("bins", bin_id),
        ("exact_conditions", np.asarray(groups)),
    ):
        rows = []
        for ia, alpha in enumerate(settings.alphas):
            covered = membership[np.arange(len(y)), ia, y]
            powered = ~membership[np.arange(len(y)), ia, 1 - y]
            for label in (0, 1):
                for group in np.unique(grouping[y == label]):
                    mask = (y == label) & (grouping == group)
                    n, k = (int(mask.sum()), int(covered[mask].sum()))
                    if n < settings.audit_min_n:
                        continue
                    coverage, nominal = (k / n, 1 - alpha)
                    p_kl = (
                        1.0
                        if coverage >= nominal
                        else math.exp(-n * bernoulli_kl(coverage, nominal))
                    )
                    rows.append(
                        {
                            "family": family,
                            "y": label,
                            "group": int(group),
                            "alpha": float(alpha),
                            "n": n,
                            "covered": k,
                            "coverage": coverage,
                            "power": float(powered[mask].mean()),
                            "nominal": nominal,
                            "gap": coverage - nominal,
                            "E_mean": float(np.mean(E[mask])),
                            "g_mean": float(np.mean(g[mask])),
                            "p_binomial": float(
                                binomtest(k, n, nominal, alternative="less").pvalue
                            ),
                            "p_chernoff": p_kl,
                        }
                    )
        if rows:
            frame = pd.DataFrame(rows)
            frame["holm_binomial"] = holm_adjust(frame["p_binomial"])
            frame["holm_chernoff"] = holm_adjust(frame["p_chernoff"])
            results.extend(frame.to_dict("records"))
    return pd.DataFrame(results)
