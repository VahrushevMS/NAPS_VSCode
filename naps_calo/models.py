"""Ячейка 3: признаки, ResCNN и статистики NAPS."""

import math
import numpy as np
import torch
from torch import nn

COORD = np.linspace(-1, 1, 8, dtype=np.float32)

GRID_X, GRID_Y = np.meshgrid(COORD, COORD)


def validate_observations(raw):
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 4 or x.shape[1:] != (4, 8, 8):
        raise ValueError("X должен иметь форму (N,4,8,8)")
    if not np.isfinite(x).all() or np.any(x < 0):
        raise ValueError("X должен содержать конечные неотрицательные энергии")
    return x


def event_features(raw):
    x = validate_observations(raw)
    layer = x.sum(axis=(2, 3))
    total = layer.sum(axis=1)
    denominator = np.maximum(total, 1e-08)
    fractions = layer / denominator[:, None]
    image = x.sum(axis=1)
    cx = (image * GRID_X).sum(axis=(1, 2)) / denominator
    cy = (image * GRID_Y).sum(axis=(1, 2)) / denominator
    wx = np.sqrt(
        np.maximum((image * GRID_X**2).sum(axis=(1, 2)) / denominator - cx**2, 0)
    )
    wy = np.sqrt(
        np.maximum((image * GRID_Y**2).sum(axis=(1, 2)) / denominator - cy**2, 0)
    )
    z = np.linspace(0, 1, 4, dtype=np.float32)
    cz = (fractions * z).sum(axis=1)
    wz = np.sqrt(np.maximum((fractions * z**2).sum(axis=1) - cz**2, 0))
    return np.column_stack(
        [
            np.log1p(total),
            np.log1p(layer),
            fractions,
            (x > 0).mean(axis=(2, 3)),
            np.log1p(x.max(axis=(1, 2, 3))),
            cx,
            cy,
            wx,
            wy,
            cz,
            wz,
        ]
    ).astype(np.float32)


def numpy_scaler(preprocessing):
    return {
        key: value.detach().cpu().numpy()
        if isinstance(value, torch.Tensor)
        else np.asarray(value)
        for key, value in preprocessing.items()
    }


def preprocess(raw, scaler, features=None):
    x = validate_observations(raw)
    s = numpy_scaler(scaler)
    logmaps = (np.log1p(x) - s["image_mean"][None, :, None, None]) / s["image_std"][
        None, :, None, None
    ]
    relative = x / np.maximum(x.sum(axis=(1, 2, 3)), 1e-08)[:, None, None, None]
    maps = np.concatenate([logmaps, (x > 0).astype(np.float32), relative], axis=1)
    f = (
        event_features(x)
        if features is None
        else np.asarray(features, dtype=np.float32)
    )
    f = (f - s["feature_mean"]) / s["feature_std"]
    return (maps.astype(np.float32), f.astype(np.float32))


class ResidualBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width),
            nn.SiLU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(width, width, 3, padding=1, bias=False),
            nn.GroupNorm(4, width),
        )

    def forward(self, x):
        return nn.functional.silu(x + self.layers(x))


class CaloResCNN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if (
            config["arch"] != "rescnn"
            or not config["use_shape"]
            or (not config["use_features"])
        ):
            raise ValueError("Этот Colab поддерживает только выбранную ResCNN")
        width, dropout = (config["width"], config["dropout"])
        self.encoder = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            nn.SiLU(),
            *[ResidualBlock(width, dropout) for _ in range(config["blocks"])],
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
        )
        self.feature_encoder = nn.Sequential(
            nn.Linear(20, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU()
        )
        self.shared = nn.Sequential(
            nn.Linear(width * 16 + 32, 128), nn.SiLU(), nn.Dropout(dropout)
        )
        self.class_head = nn.Linear(128, 1)
        self.nuisance_head = nn.Linear(128, 8)

    def forward(self, x, features):
        h = torch.cat([self.encoder(x), self.feature_encoder(features)], dim=1)
        h = self.shared(h)
        nuisance = self.nuisance_head(h).reshape(-1, 2, 4)
        mu = torch.sigmoid(nuisance[:, :, :2])
        scale = 0.025 + nn.functional.softplus(nuisance[:, :, 2:])
        return (self.class_head(h).squeeze(1), mu, scale)


def statistics(logits, train_prior_1):
    if not 0 < train_prior_1 < 1:
        raise ValueError("Обучающая доля класса 1 должна лежать между 0 и 1")
    prior_logit = math.log(train_prior_1 / (1 - train_prior_1))
    log_bf = np.asarray(logits, dtype=np.float64) - prior_logit
    return np.column_stack([-log_bf, log_bf])


def encode_membership(membership):
    m = np.asarray(membership, dtype=bool)
    return m[..., 0].astype(np.uint8) + 2 * m[..., 1].astype(np.uint8)
