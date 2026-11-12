"""Ячейки 9-10: условная W, инверсия и независимый guard."""

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from .models import statistics
from .data import split_condition_groups
from .metrics import kl_upper


def compressed_cdf_rows(data, ids, stats, label, thresholds):
    blocks, targets, weights = ([], [], [])
    for gid in np.unique(data["groups"][ids]):
        mask = (data["groups"][ids] == gid) & (data["y"][ids] == label)
        if not mask.any():
            continue
        scores = np.sort(stats[mask, label])
        counts = np.searchsorted(scores, thresholds, side="right")
        nu = data["nu"][ids[np.flatnonzero(mask)[0]]]
        rows = np.column_stack([np.tile(nu, (len(thresholds), 1)), thresholds]).astype(
            np.float32
        )
        for target, weight in ((0, len(scores) - counts), (1, counts)):
            keep = weight > 0
            if keep.any():
                endpoint_weight = np.where(
                    (thresholds[keep] == thresholds[0])
                    | (thresholds[keep] == thresholds[-1]),
                    8.0,
                    1.0,
                )
                blocks.append(rows[keep])
                targets.append(np.full(keep.sum(), target))
                weights.append(weight[keep] * endpoint_weight)
    return (np.concatenate(blocks), np.concatenate(targets), np.concatenate(weights))


def fit_rejection_probabilities(settings, data, indices, train_prior_1, logits):
    from catboost import CatBoostClassifier, Pool

    ids = indices["w_cal"]
    groups = np.unique(data["groups"][ids])
    fit_groups, val_groups = split_condition_groups(
        settings,
        groups,
        max(1, len(groups) // 5),
        data["conditions"],
        settings.seed + 200,
    )
    fit_mask = np.isin(data["groups"][ids], fit_groups)
    stats = statistics(logits, train_prior_1)
    models, bounds, rows = ({}, {}, [])
    for label in (0, 1):
        true_scores = stats[fit_mask & (data["y"][ids] == label), label]
        levels = np.unique(
            np.r_[np.linspace(0.001, 0.25, 40), np.linspace(0.3, 0.99, 15)]
        )
        thresholds = np.unique(
            np.r_[
                np.quantile(true_scores, levels),
                np.linspace(true_scores.min() - 2, true_scores.max() + 2, 16),
            ].astype(np.float32)
        )
        fx, fy, fw = compressed_cdf_rows(
            data, ids[fit_mask], stats[fit_mask], label, thresholds
        )
        vx, vy, vw = compressed_cdf_rows(
            data, ids[~fit_mask], stats[~fit_mask], label, thresholds
        )
        model = CatBoostClassifier(
            iterations=settings.w_iterations,
            depth=4,
            l2_leaf_reg=10,
            learning_rate=0.05,
            border_count=settings.border_count,
            loss_function="Logloss",
            monotone_constraints=[0, 0, 1],
            random_seed=settings.seed + label,
            thread_count=settings.threads,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(
            Pool(fx, fy, weight=fw),
            eval_set=Pool(vx, vy, weight=vw),
            early_stopping_rounds=70,
            use_best_model=True,
        )
        models[label] = model
        bounds[label] = (float(true_scores.min() - 20), float(true_scores.max() + 20))
        rows.append(
            {
                "y": label,
                "trees": model.tree_count_,
                "validation_logloss": float(
                    log_loss(vy, model.predict_proba(vx)[:, 1], sample_weight=vw)
                ),
            }
        )
        model.save_model(str(settings.output_dir / f"W_{label}.cbm"))
    return (models, bounds, pd.DataFrame(rows))


def rejection_probability(model, nuisance, threshold):
    points = np.asarray(nuisance, dtype=np.float32).reshape(-1, 2)
    thresholds = np.broadcast_to(threshold, (len(points),))
    return model.predict_proba(
        np.column_stack([points, thresholds]).astype(np.float32)
    )[:, 1]


def inverse_rejection_probability(
    model, points, probability, lower, upper, iterations=32
):
    points = np.asarray(points, dtype=np.float32)
    if not 0 < probability < 1:
        raise ValueError("Уровень вероятности должен лежать между 0 и 1")
    lo, hi = (np.full(len(points), lower), np.full(len(points), upper))
    valid = (rejection_probability(model, points, lo) <= probability) & (
        rejection_probability(model, points, hi) > probability
    )
    for _ in range(iterations):
        mid = (lo + hi) / 2
        safe = rejection_probability(model, points, mid) <= probability
        lo = np.where(safe, mid, lo)
        hi = np.where(safe, hi, mid)
    cutoff = np.nextafter(lo.astype(np.float32), np.float32(-np.inf)).astype(np.float64)
    valid &= rejection_probability(model, points, cutoff) <= probability
    return np.where(valid, cutoff, -np.inf)


def nuisance_partition(model):
    axes = []
    for feature in (0, 1):
        borders = np.unique(
            np.asarray(model.get_borders().get(feature, []), dtype=np.float32)
        )
        borders = borders[(borders >= 0) & (borders < 1)]
        hi = np.r_[borders, np.float32(1)].astype(np.float32)
        valid_lo = np.r_[np.float32(0), np.nextafter(borders, np.float32(np.inf))]
        outer_lo = np.r_[np.float32(0), borders]
        keep = valid_lo <= hi
        axes.append((outer_lo[keep], hi[keep], valid_lo[keep]))
    i, j = np.meshgrid(
        np.arange(len(axes[0][0])), np.arange(len(axes[1][0])), indexing="ij"
    )
    lo = np.column_stack([axes[0][0][i.ravel()], axes[1][0][j.ravel()]])
    hi = np.column_stack([axes[0][1][i.ravel()], axes[1][1][j.ravel()]])
    representative = np.column_stack(
        [axes[0][2][i.ravel()], axes[1][2][j.ravel()]]
    ).astype(np.float32)
    if len(representative) > 100000:
        raise RuntimeError("Слишком много nuisance-ячеек")
    return (lo, hi, representative)


def worst_case_cutoffs(settings, models, bounds):
    raw, worst, basis = (
        np.empty((len(settings.alphas), 2)),
        np.empty((len(settings.alphas), 2, 2, 2)),
        {},
    )
    for label in (0, 1):
        lo, hi, points = nuisance_partition(models[label])
        cuts = np.column_stack(
            [
                inverse_rejection_probability(
                    models[label], points, float(alpha), *bounds[label]
                )
                for alpha in settings.alphas
            ]
        )
        if not np.all(cuts[:, 1:] >= cuts[:, :-1]):
            raise RuntimeError(
                "Обратная rejection probability должна возрастать по alpha"
            )
        raw[:, label] = cuts.min(axis=0)
        for ia in range(len(settings.alphas)):
            index = int(cuts[:, ia].argmin())
            worst[ia, label] = np.stack([lo[index], hi[index]])
        basis[label] = {"lo": lo, "hi": hi, "cutoffs": cuts}
        np.savez_compressed(
            settings.output_dir / f"nuisance_cells_{label}.npz", **basis[label]
        )
    return (raw, worst, basis)


def calibrate_margins(settings, data, train_prior_1, logits, raw_cutoffs, ids):
    stats = statistics(logits, train_prior_1)
    family = 2 * settings.n_bins * len(settings.alphas) * (len(settings.margins) - 1)
    delta = settings.guard_delta / family
    margins = np.full((len(settings.alphas), 2), np.inf)
    rows = []
    for ia, alpha in enumerate(settings.alphas):
        for label in (0, 1):
            for margin in settings.margins[:-1]:
                rejected = stats[:, label] <= raw_cutoffs[ia, label] - margin
                upper = 0.0
                for bin_id in range(settings.n_bins):
                    mask = (data["y"][ids] == label) & (data["bins"][ids] == bin_id)
                    bound = (
                        0.0
                        if np.isneginf(raw_cutoffs[ia, label])
                        else kl_upper(int(rejected[mask].sum()), int(mask.sum()), delta)
                    )
                    upper = max(upper, bound)
                rows.append(
                    {
                        "alpha": float(alpha),
                        "y": label,
                        "margin": float(margin),
                        "max_bin_ucb": upper,
                    }
                )
                if upper <= alpha and np.isinf(margins[ia, label]):
                    margins[ia, label] = margin
    return (margins, pd.DataFrame(rows))
