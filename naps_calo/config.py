"""Ячейки 1-2: пути, параметры и вычислительное устройство."""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value, default):
    path = Path(value).expanduser() if value else default
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


@dataclass
class Settings:
    mode: str = "pretrained"
    weights_path: Path = PROJECT_ROOT / "weights/NAPS_Calo_ResCNN_weights.pt"
    data_dir: Path = PROJECT_ROOT / "data/simulations"
    session_dir: Path = PROJECT_ROOT / "data/sessions"
    output_dir: Path = PROJECT_ROOT / "outputs"
    cache_dir: Path = PROJECT_ROOT / ".cache/naps"
    device_name: str = "auto"
    seed: int = 20261109
    max_epochs: int = 40
    patience: int = 10
    w_iterations: int = 800
    batch_size: int = 2048
    threads: int = min(2, os.cpu_count() or 1)
    evaluate_data: bool = False
    make_plots: bool = True
    max_budget: int = 1_000_000
    guard_delta: float = 0.005
    audit_min_n: int = 50
    audit_significance: float = 0.05
    border_count: int = 32
    n_bins: int = 24
    recalibration_gamma: float = 0.0
    alphas: np.ndarray = field(
        default_factory=lambda: np.array([0.05, 0.10, 0.20], dtype=np.float64)
    )
    energy_edges: np.ndarray = field(default_factory=lambda: np.geomspace(1, 100, 7))
    gain_edges: np.ndarray = field(default_factory=lambda: np.linspace(0.7, 1.3, 5))
    margins: np.ndarray = field(
        default_factory=lambda: np.array(
            [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4, 6, 10, np.inf]
        )
    )
    device: torch.device = field(init=False)
    use_amp: bool = field(init=False)

    def initialize(self):
        if self.mode not in {"pretrained", "recalibrate", "train"}:
            raise ValueError("Режим: pretrained, recalibrate или train")
        if (
            min(
                self.max_epochs,
                self.patience,
                self.w_iterations,
                self.batch_size,
                self.threads,
            )
            < 1
        ):
            raise ValueError(
                "Число эпох, patience, итераций, потоков и размер пакета должны быть положительными"
            )
        if self.device_name not in {"auto", "cpu", "cuda"}:
            raise ValueError("Устройство: auto, cpu или cuda")
        if self.device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError(
                "CUDA недоступна. Запустите с --device cpu или установите подходящую GPU-версию PyTorch"
            )
        name = (
            "cuda"
            if self.device_name == "auto" and torch.cuda.is_available()
            else self.device_name
        )
        self.device = torch.device("cpu" if name == "auto" else name)
        self.use_amp = self.device.type == "cuda"
        for folder in (self.output_dir, self.cache_dir, self.output_dir / "figures"):
            folder.mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(self.threads)
        seed_everything(self.seed)
        return self
