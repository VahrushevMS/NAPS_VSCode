"""Ячейка 8: нормализация и обучение новой ResCNN."""

import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from .config import seed_everything
from .models import CaloResCNN, preprocess, event_features
from .inference import predict_logits
from .metrics import classifier_metrics


def fit_preprocessing(raw, train_ids):
    sum_x, sum_x2 = (np.zeros(4), np.zeros(4))
    sum_f, sum_f2 = (np.zeros(20), np.zeros(20))
    pixels = events = 0
    for start in range(0, len(train_ids), 8192):
        x = raw[train_ids[start : start + 8192]]
        z = np.log1p(x).astype(np.float64)
        f = event_features(x).astype(np.float64)
        sum_x += z.sum(axis=(0, 2, 3))
        sum_x2 += (z * z).sum(axis=(0, 2, 3))
        sum_f += f.sum(axis=0)
        sum_f2 += (f * f).sum(axis=0)
        pixels += len(x) * 64
        events += len(x)
    image_mean = sum_x / pixels
    feature_mean = sum_f / events
    scaler = {
        "image_mean": image_mean.astype(np.float32),
        "image_std": np.sqrt(np.maximum(sum_x2 / pixels - image_mean**2, 1e-12)).astype(
            np.float32
        ),
        "feature_mean": feature_mean.astype(np.float32),
        "feature_std": np.maximum(
            np.sqrt(np.maximum(sum_f2 / events - feature_mean**2, 0)), 1e-06
        ).astype(np.float32),
    }
    return {key: torch.from_numpy(value) for key, value in scaler.items()}


class IndexDataset(Dataset):
    def __init__(self, indices):
        self.indices = np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        return int(self.indices[i])


def make_loader(settings, data, config, indices, scaler, seed):

    def collate(batch):
        ids = np.asarray(batch, dtype=np.int64)
        maps, f = preprocess(data["X"][ids], scaler)
        return (
            torch.from_numpy(maps),
            torch.from_numpy(f),
            torch.from_numpy(data["y"][ids]),
            torch.from_numpy(data["nu"][ids]),
        )

    return DataLoader(
        IndexDataset(indices),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        collate_fn=collate,
        pin_memory=settings.device.type == "cuda",
        generator=torch.Generator().manual_seed(seed),
    )


def train_rescnn(settings, data, indices, config, scaler):
    seed_everything(settings.seed)
    model = CaloResCNN(config).to(settings.device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=settings.max_epochs
    )
    amp_scaler = torch.amp.GradScaler("cuda", enabled=settings.use_amp)
    loader = make_loader(
        settings, data, config, indices["train"], scaler, settings.seed
    )
    best_bce, best_state, stale, history = (np.inf, None, 0, [])
    for epoch in range(1, settings.max_epochs + 1):
        start = time.time()
        model.train()
        bce_sum = aux_sum = 0.0
        for maps, features, labels, nuisance_target in loader:
            maps, features = (maps.to(settings.device), features.to(settings.device))
            labels, nuisance_target = (
                labels.to(settings.device),
                nuisance_target.to(settings.device),
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=settings.device.type, enabled=settings.use_amp
            ):
                logits, mu, scale = model(maps, features)
                bce = nn.functional.binary_cross_entropy_with_logits(
                    logits, labels.float()
                )
                pos = torch.arange(len(labels), device=settings.device)
                my, sy = (mu[pos, labels], scale[pos, labels])
                auxiliary = (
                    0.5 * ((nuisance_target - my) / sy) ** 2 + torch.log(sy)
                ).mean()
                loss = bce + config["aux_weight"] * auxiliary
            if not torch.isfinite(loss):
                raise FloatingPointError("Неконечная функция потерь")
            amp_scaler.scale(loss).backward()
            amp_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            amp_scaler.step(optimizer)
            amp_scaler.update()
            bce_sum += float(bce.detach()) * len(labels)
            aux_sum += float(auxiliary.detach()) * len(labels)
        scheduler.step()
        val_logits = predict_logits(model, data["X"], scaler, indices=indices["val"])
        metrics = classifier_metrics(data["y"][indices["val"]], val_logits)
        row = {
            "epoch": epoch,
            "train_bce": bce_sum / len(indices["train"]),
            "train_aux": aux_sum / len(indices["train"]),
            "val_bce": metrics["bce"],
            "val_auc": metrics["auc"],
            "seconds": time.time() - start,
        }
        history.append(row)
        if metrics["bce"] < best_bce - 1e-05:
            best_bce = metrics["bce"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
            torch.save(
                {
                    "config": config,
                    "state_dict": best_state,
                    "preprocess": scaler,
                    "epoch": epoch,
                },
                settings.output_dir / "best_training_state.pt",
            )
        else:
            stale += 1
        if (
            epoch == 1
            or epoch % 5 == 0
            or stale >= settings.patience
            or (epoch == settings.max_epochs)
        ):
            print(
                f"ResCNN epoch {epoch}: val_BCE={metrics['bce']:.6f}, val_AUC={metrics['auc']:.6f}",
                flush=True,
            )
        if stale >= settings.patience:
            break
    if best_state is None:
        raise RuntimeError("Не получено ни одного валидного checkpoint")
    model.load_state_dict(best_state)
    return (model.eval(), pd.DataFrame(history), best_state)
