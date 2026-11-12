"""Искусственные события для проверки запуска; не заменяют физические симуляции."""

from pathlib import Path
import numpy as np


def make_demo(destination=None):
    path = (
        Path(destination)
        if destination
        else Path(__file__).resolve().parents[1] / "data/sessions/session_demo.npz"
    )
    rng = np.random.default_rng(1203)
    coordinates = np.linspace(-1, 1, 8)
    xx, yy = np.meshgrid(coordinates, coordinates)
    maps = []
    for i in range(32):
        center = rng.uniform(-0.35, 0.35, size=2)
        width = rng.uniform(0.16, 0.6)
        profile = np.exp(
            -((xx - center[0]) ** 2 + (yy - center[1]) ** 2) / (2 * width**2)
        )
        layers = rng.dirichlet([2, 4, 3, 1])
        event = (
            layers[:, None, None] * profile[None] * rng.lognormal(0, 0.25, (4, 8, 8))
        )
        event[rng.random(event.shape) < 0.12] = 0
        event *= np.geomspace(1, 100, 32)[i] / event.sum()
        maps.append(event.astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, X=np.stack(maps), event_id=np.array([f"demo_{i:03d}" for i in range(32)])
    )
    print(path)
    return path


if __name__ == "__main__":
    make_demo()
