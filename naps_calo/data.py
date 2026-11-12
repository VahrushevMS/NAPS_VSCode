"""Ячейка 6: словари симуляций и непересекающиеся условия."""

import hashlib
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split


def build_splits(settings, data, checkpoint):
    """Вторая часть ячейки 6; условия (E,g) остаются целыми."""
    if settings.mode == "pretrained":
        indices = {
            key: value.cpu().numpy()
            for key, value in checkpoint["source_split_indices"].items()
        }
    elif settings.mode == "recalibrate":
        saved = {
            key: value.cpu().numpy()
            for key, value in checkpoint["source_split_indices"].items()
        }
        if {"s_cal", "select", "s_guard"} <= set(saved):
            indices = {
                "train": saved["train"],
                "val": saved["val"],
                "test": saved["test"],
                "w_cal": np.sort(
                    np.concatenate([saved["w_cal"], saved["s_cal"], saved["select"]])
                ),
                "guard": np.sort(np.concatenate([saved["guard"], saved["s_guard"]])),
            }
        else:
            indices = saved
    else:
        remaining = np.arange(len(data["conditions"]))
        roles = {}
        for offset, (name, fraction) in enumerate(
            [("test", 0.10), ("guard", 0.19), ("w_cal", 0.17), ("val", 0.04)]
        ):
            count = int(round(fraction * len(data["conditions"])))
            if count < 1 or count >= len(remaining):
                raise ValueError(
                    "Недостаточно условий (E,g) для независимых частей данных"
                )
            remaining, chosen = split_condition_groups(
                settings, remaining, count, data["conditions"], settings.seed + offset
            )
            roles[name] = chosen
        roles["train"] = remaining
        indices = {
            key: np.flatnonzero(np.isin(data["groups"], groups))
            for key, groups in roles.items()
        }
    if not {"train", "val", "test", "w_cal", "guard"} <= set(indices):
        raise ValueError(
            "Checkpoint не содержит обязательные части train/val/test/w_cal/guard"
        )
    size = len(data["y"])
    for role, ids in indices.items():
        if (
            ids.ndim != 1
            or not np.issubdtype(ids.dtype, np.integer)
            or np.any(ids < 0)
            or np.any(ids >= size)
        ):
            raise ValueError(f"Некорректные индексы части {role}")
    all_ids = np.concatenate(list(indices.values()))
    if len(all_ids) != size or len(np.unique(all_ids)) != size:
        raise ValueError("Части должны быть непересекающимися и покрывать все события")
    assigned_groups = [np.unique(data["groups"][ids]) for ids in indices.values()]
    if sum(map(len, assigned_groups)) != len(
        np.unique(np.concatenate(assigned_groups))
    ):
        raise ValueError("Повторы одного условия (E,g) попали в разные части")
    return indices


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def nuisance_bins(settings, E, g):
    ie = np.clip(np.searchsorted(settings.energy_edges, E, side="right") - 1, 0, 5)
    ig = np.clip(np.searchsorted(settings.gain_edges, g, side="right") - 1, 0, 3)
    return ie * 4 + ig


def normalize_nuisance(E, g):
    return np.column_stack(
        [np.log(E) / np.log(100.0), (np.asarray(g) - 0.7) / 0.6]
    ).astype(np.float32)


def load_dictionaries(settings, data_dir, expected_manifest=None):
    paths = sorted(Path(data_dir).glob("calo_dictionary*.npz"))
    if not paths:
        return None
    unique = {}
    for path in paths:
        unique.setdefault(file_hash(path), path)
    if expected_manifest is not None:
        expected = [row["sha256"] for row in expected_manifest]
        if set(unique) != set(expected):
            raise ValueError(
                "Для готовых весов нужны исходные NPZ из data_manifest; иные файлы допустимы в MODE='train'"
            )
        ordered = [(digest, unique[digest]) for digest in expected]
    else:
        ordered = list(unique.items())
    specifications, keys, manifest = ([], [], [])
    for digest, path in ordered:
        with np.load(path, allow_pickle=False) as d:
            if not {"keys", "values"} <= set(d.files):
                raise ValueError(f"Нужны keys и values: {path.name}")
            k = np.asarray(d["keys"], dtype=np.float64)
            v = d["values"]
            if (
                k.ndim != 2
                or k.shape[1] != 3
                or v.ndim != 5
                or (len(v) != len(k))
                or (v.shape[2:] != (4, 8, 8))
                or len(k) == 0
                or v.shape[1] == 0
            ):
                raise ValueError(f"Некорректные формы массивов: {path.name}")
            if not np.isfinite(k).all() or not np.isfinite(v).all() or np.any(v < 0):
                raise ValueError(f"Некорректные энергии/параметры: {path.name}")
            if not np.isin(k[:, 0], [0, 1]).all() or np.any(
                (k[:, 1] < 1) | (k[:, 1] > 100) | (k[:, 2] < 0.7) | (k[:, 2] > 1.3)
            ):
                raise ValueError(f"Параметры вне допустимых диапазонов: {path.name}")
            specifications.append((path, len(k), v.shape[1]))
            keys.append(k)
            manifest.append(
                {
                    "file": path.name,
                    "sha256": digest,
                    "events": int(len(k) * v.shape[1]),
                }
            )
    keys = np.concatenate(keys)
    if len(np.unique(keys, axis=0)) != len(keys):
        raise ValueError(
            "Повторные ключи (y,E,g); независимость событий не установлена"
        )
    conditions, key_groups = np.unique(keys[:, 1:], axis=0, return_inverse=True)
    if not np.all(np.bincount(key_groups) == 2):
        raise ValueError("Нужны оба класса на каждом условии (E,g)")
    n = sum((row["events"] for row in manifest))
    if n > settings.max_budget:
        raise ValueError(f"Бюджет превышен: {n} > {settings.max_budget}")
    raw = np.lib.format.open_memmap(
        settings.cache_dir / "raw.npy", mode="w+", dtype="float32", shape=(n, 4, 8, 8)
    )
    labels, energies, gains, groups = ([], [], [], [])
    event_offset = key_offset = 0
    for path, n_keys, repeats in specifications:
        k = keys[key_offset : key_offset + n_keys]
        with np.load(path, allow_pickle=False) as d:
            raw[event_offset : event_offset + n_keys * repeats] = d["values"].reshape(
                -1, 4, 8, 8
            )
        labels.append(np.repeat(k[:, 0].astype(np.int64), repeats))
        energies.append(np.repeat(k[:, 1], repeats))
        gains.append(np.repeat(k[:, 2], repeats))
        groups.append(np.repeat(key_groups[key_offset : key_offset + n_keys], repeats))
        event_offset += n_keys * repeats
        key_offset += n_keys
    raw.flush()
    y, E, g, group = map(np.concatenate, (labels, energies, gains, groups))
    counts = np.bincount(2 * group + y, minlength=2 * len(conditions)).reshape(-1, 2)
    if not np.all(counts[:, 0] == counts[:, 1]):
        raise ValueError("Нужны одинаковые числа повторов обоих классов на условие")
    return {
        "X": raw,
        "y": y,
        "E": E,
        "g": g,
        "groups": group,
        "conditions": conditions,
        "nu": normalize_nuisance(E, g),
        "bins": nuisance_bins(settings, E, g),
        "manifest": manifest,
    }


def split_condition_groups(settings, gids, count, conditions, seed):
    strata = nuisance_bins(settings, conditions[gids, 0], conditions[gids, 1])
    counts = np.unique(strata, return_counts=True)[1]
    use_strata = (
        strata
        if min(count, len(gids) - count) >= len(counts) and counts.min() >= 2
        else None
    )
    return train_test_split(
        gids, test_size=count, random_state=seed, stratify=use_strata
    )
