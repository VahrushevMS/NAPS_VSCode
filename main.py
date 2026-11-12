

import argparse
import importlib.util
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("PYTHONUTF8", "1")


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="ResCNN и NAPS: вывод, test, обучение и перекалибровка"
    )
    p.add_argument(
        "--mode",
        choices=["pretrained", "recalibrate", "train"],
        default=os.environ.get("NAPS_MODE", "pretrained"),
    )
    p.add_argument("--weights", default=os.environ.get("NAPS_WEIGHTS_PATH"))
    p.add_argument("--data-dir", default=os.environ.get("NAPS_DATA_DIR"))
    p.add_argument("--session-dir")
    p.add_argument(
        "--session",
        action="append",
        help="Один NPZ; можно повторить --session несколько раз",
    )
    p.add_argument("--output-dir", default=os.environ.get("NAPS_OUTPUT_DIR"))
    p.add_argument("--cache-dir", default=os.environ.get("NAPS_CACHE_DIR"))
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument(
        "--max-epochs", type=int, default=int(os.environ.get("NAPS_MAX_EPOCHS", "40"))
    )
    p.add_argument("--patience", type=int, default=10)
    p.add_argument(
        "--w-iterations",
        type=int,
        default=int(os.environ.get("NAPS_W_ITERATIONS", "800")),
    )
    p.add_argument("--threads", type=int, default=min(2, os.cpu_count() or 1))
    p.add_argument(
        "--evaluate-data",
        action="store_true",
        help="В pretrained заново оценить test на всех исходных NPZ",
    )
    p.add_argument("--no-plots", action="store_true")
    p.add_argument(
        "--check",
        action="store_true",
        help="Только проверить зависимости, веса и один inference",
    )
    p.add_argument(
        "--debug", action="store_true", help="Показать полный traceback при ошибке"
    )
    return p.parse_args(argv)


def main(argv=None):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    modules = ["numpy", "pandas", "torch", "scipy", "sklearn", "matplotlib"]
    missing = [name for name in modules if importlib.util.find_spec(name) is None]
    if missing:
        print("Не установлены библиотеки: " + ", ".join(missing), file=sys.stderr)
        print(
            "Запустите START_WINDOWS.bat или python bootstrap.py из папки проекта.",
            file=sys.stderr,
        )
        return 2
    try:
        from naps_calo.config import Settings, PROJECT_ROOT, project_path
        from naps_calo.pipeline import run_pipeline

        settings = Settings(
            mode=args.mode,
            weights_path=(
                PROJECT_ROOT / "weights" / "NAPS_Calo_ResCNN_weights.pt"
            ),
            data_dir=project_path(args.data_dir, PROJECT_ROOT / "data/simulations"),
            session_dir=project_path(args.session_dir, PROJECT_ROOT / "data/sessions"),
            output_dir=project_path(args.output_dir, PROJECT_ROOT / "outputs"),
            cache_dir=project_path(args.cache_dir, PROJECT_ROOT / ".cache/naps"),
            device_name=args.device,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            w_iterations=args.w_iterations,
            threads=args.threads,
            evaluate_data=args.evaluate_data,
            make_plots=not args.no_plots,
        ).initialize()
        paths = (
            [project_path(path, PROJECT_ROOT) for path in args.session]
            if args.session
            else None
        )
        run_pipeline(settings, paths, check_only=args.check)
    except KeyboardInterrupt:
        print("\nЗапуск остановлен пользователем.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"\nОшибка: {error}", file=sys.stderr)
        if args.debug:
            traceback.print_exc()
        else:
            print("Для полного сообщения добавьте --debug.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
