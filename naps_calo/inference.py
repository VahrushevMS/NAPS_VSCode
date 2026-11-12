"""Ячейка 5: пакетный вывод и экспорт множеств."""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from .models import CaloResCNN, preprocess, numpy_scaler, statistics, encode_membership


@torch.inference_mode()
def predict_logits(model, raw, scaler, batch_size=2048, indices=None):
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")
    if np.ndim(raw) != 4 or np.shape(raw)[1:] != (4, 8, 8):
        raise ValueError("X должен иметь форму (N,4,8,8)")
    size = len(raw) if indices is None else len(indices)
    if size == 0:
        return np.empty(0, dtype=np.float64)
    model.eval()
    logits = []
    for start in range(0, size, batch_size):
        batch = (
            raw[start : start + batch_size]
            if indices is None
            else raw[np.asarray(indices[start : start + batch_size])]
        )
        maps, features = preprocess(batch, scaler)
        score, _, _ = model(
            torch.from_numpy(maps).to(next(model.parameters()).device),
            torch.from_numpy(features).to(next(model.parameters()).device),
        )
        logits.append(score.float().cpu().numpy().astype(np.float64))
    return np.concatenate(logits)


class ResCNNNAPSPredictor:
    def __init__(self, checkpoint, device="cpu"):
        checkpoint = (
            torch.load(checkpoint, map_location="cpu", weights_only=True)
            if isinstance(checkpoint, (str, Path))
            else checkpoint
        )
        self.device = torch.device(device)
        self.model = CaloResCNN(checkpoint["config"]).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model.eval()
        self.scaler = numpy_scaler(checkpoint["preprocess"])
        self.prior = float(checkpoint["train_prior_1"])
        self.alphas = checkpoint["naps"]["alphas"].cpu().numpy()
        self.thresholds = checkpoint["naps"]["thresholds"].cpu().numpy()
        if (
            self.alphas.ndim != 1
            or len(self.alphas) == 0
            or not np.isfinite(self.alphas).all()
            or not np.all((self.alphas > 0) & (self.alphas < 1))
            or not np.all(np.diff(self.alphas) > 0)
        ):
            raise ValueError(
                "Уровни alpha должны строго возрастать и лежать между 0 и 1"
            )
        if (
            self.thresholds.shape != (len(self.alphas), 2)
            or np.isnan(self.thresholds).any()
        ):
            raise ValueError("Некорректные пороги NAPS")
        if not np.all(self.thresholds[1:] >= self.thresholds[:-1]):
            raise ValueError("Пороги нарушают вложенность prediction sets")

    @torch.inference_mode()
    def predict(self, X, batch_size=2048):
        if batch_size < 1:
            raise ValueError("batch_size должен быть положительным")
        if np.ndim(X) != 4 or np.shape(X)[1:] != (4, 8, 8):
            raise ValueError("X должен иметь форму (N,4,8,8)")
        if len(X) == 0:
            return (np.empty(0), np.empty((0, len(self.alphas)), dtype=np.uint8))
        logits = []
        for start in range(0, len(X), batch_size):
            maps, features = preprocess(X[start : start + batch_size], self.scaler)
            l, _, _ = self.model(
                torch.from_numpy(maps).to(self.device),
                torch.from_numpy(features).to(self.device),
            )
            logits.append(l.float().cpu().numpy().astype(np.float64))
        logits = np.concatenate(logits)
        membership = (
            statistics(logits, self.prior)[:, None, :] > self.thresholds[None, :, :]
        )
        return (expit(logits), encode_membership(membership))

    def export(self, X, path, event_ids=None, batch_size=2048):
        _, codes = self.predict(X, batch_size)
        ids = np.arange(len(X)) if event_ids is None else np.asarray(event_ids)
        if ids.ndim != 1 or len(ids) != len(X):
            raise ValueError("Количество event_id не совпадает с количеством X")
        frame = pd.DataFrame({"event_id": ids})
        for ia, alpha in enumerate(self.alphas):
            frame[f"H_{alpha:.2f}"] = codes[:, ia]
        frame.to_csv(path, index=False)
        return frame
