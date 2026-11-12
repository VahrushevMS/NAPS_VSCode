"""Ячейка 13: сессии новых наблюдений и CSV."""

from pathlib import Path
import numpy as np
import pandas as pd
from .models import validate_observations


def load_session(path):
    with np.load(path, allow_pickle=False) as data:
        key = "X" if "X" in data.files else "values" if "values" in data.files else None
        if key is None:
            raise ValueError(f"В {Path(path).name} нет массива X или values")
        raw = np.asarray(data[key], dtype=np.float32)
        if raw.shape == (4, 8, 8):
            raw = raw[None]
        elif raw.ndim == 5 and raw.shape[2:] == (4, 8, 8):
            raw = raw.reshape(-1, 4, 8, 8)
        observations = validate_observations(raw)
        event_ids = (
            np.asarray(data["event_id"]).reshape(-1)
            if "event_id" in data.files
            else np.arange(len(raw))
        )
    if len(event_ids) != len(observations):
        raise ValueError(f"Число event_id не равно числу событий в {Path(path).name}")
    return observations, event_ids


def export_sessions(settings, predictor, explicit_paths=None):
    paths = (
        [Path(p) for p in explicit_paths]
        if explicit_paths
        else sorted(settings.session_dir.glob("session*.npz"))
    )
    outputs = []
    destinations = set()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Нет файла сессии: {path}")
        observations, event_ids = load_session(path)
        destination = settings.output_dir / f"submission_{path.stem}.csv"
        if destination in destinations:
            raise ValueError(
                "Сессии с одинаковыми именами дадут одинаковый CSV. Переименуйте один из файлов"
            )
        destinations.add(destination)
        frame = predictor.export(
            observations,
            destination,
            event_ids=event_ids,
            batch_size=settings.batch_size,
        )
        outputs.append(
            {"session": path.name, "events": len(frame), "submission": str(destination)}
        )
        print(f"\nСессия: {path.name}; событий: {len(frame)}", flush=True)
        if path.name == "session_demo.npz":
            print(
                "Демонстрационные искусственные события: проверка запуска, без оценки качества модели.",
                flush=True,
            )
        print(frame.head().to_string(index=False), flush=True)
    result = pd.DataFrame(outputs, columns=["session", "events", "submission"])
    result.to_csv(settings.output_dir / "session_outputs.csv", index=False)
    return result
