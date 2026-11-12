"""Ячейка 12: графики в PNG без интерактивного окна."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def save_plots(settings, test_metrics, history, audit):
    if not test_metrics.empty:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for scenario, frame in test_metrics.groupby("scenario", sort=False):
            frame = frame.sort_values("alpha")
            axes[0].plot(frame["alpha"], frame["coverage"], "o-", label=scenario)
            axes[1].plot(frame["alpha"], frame["power"], "o-", label=scenario)
        axes[0].plot(settings.alphas, 1 - settings.alphas, "k--", label="1-alpha")
        ref = test_metrics[test_metrics["scenario"] == "Reference"].sort_values("alpha")
        axes[2].plot(ref["alpha"], ref["ambiguous"], "o-", label="ambiguous")
        axes[2].plot(ref["alpha"], ref["singleton"], "s-", label="singleton")
        axes[2].plot(ref["alpha"], ref["empty"], "^-", label="empty")
        for ax, title in zip(axes, ["Coverage", "Balanced power", "Prediction sets"]):
            ax.set(xlabel="alpha", title=title, ylim=(0, 1.02))
            ax.set_xticks(settings.alphas)
            ax.grid(alpha=0.3)
            ax.legend()
        fig.tight_layout()
        fig.savefig(settings.output_dir / "figures/coverage_power.png", dpi=160)
        plt.close(fig)
    if not history.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(history["epoch"], history["train_bce"], label="train BCE")
        ax.plot(history["epoch"], history["val_bce"], label="validation BCE")
        ax.set(xlabel="epoch", ylabel="BCE", title="ResCNN")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(settings.output_dir / "figures/training.png", dpi=160)
        plt.close(fig)
    if not audit.empty:
        for metric in ("coverage", "power"):
            frame = audit[audit["family"] == "bins"].copy()
            frame["E_bin"] = frame["group"] // 4
            frame["g_bin"] = frame["group"] % 4
            fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
            for label in (0, 1):
                for ia, alpha in enumerate(settings.alphas):
                    part = frame[(frame["y"] == label) & (frame["alpha"] == alpha)]
                    matrix = part.pivot(
                        index="E_bin", columns="g_bin", values=metric
                    ).reindex(index=range(6), columns=range(4))
                    image = axes[label, ia].imshow(
                        matrix.to_numpy(),
                        origin="lower",
                        vmin=0,
                        vmax=1,
                        aspect="auto",
                        cmap="viridis",
                    )
                    axes[label, ia].set(
                        title=f"{metric}: y={label}, alpha={alpha}",
                        xlabel="g bin",
                        ylabel="E bin",
                    )
                    axes[label, ia].set_xticks(
                        range(4),
                        [
                            f"{settings.gain_edges[j]:.2f}-{settings.gain_edges[j + 1]:.2f}"
                            for j in range(4)
                        ],
                        rotation=45,
                    )
                    axes[label, ia].set_yticks(
                        range(6),
                        [
                            f"{settings.energy_edges[j]:.1f}-{settings.energy_edges[j + 1]:.1f}"
                            for j in range(6)
                        ],
                    )
            fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8)
            fig.savefig(
                settings.output_dir / f"figures/conditional_{metric}.png", dpi=160
            )
            plt.close(fig)
