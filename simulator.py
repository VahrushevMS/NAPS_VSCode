import io
import getpass
import json
import os
from pathlib import Path
import requests
import numpy as np
import pandas as pd

SAVE_DIR = r"C:\Users\m12g\.vscode\NAPS_VSCode_fixed\NAPS_VSCode\data\simulations"
TOKEN = "UcWTCTg8j75cNWNoUbCwcd8n"
BASE_URL = "https://dam-reader-disclosure-climb.trycloudflare.com"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}
def tarin_simulatoun():
    
    SEED = 44
    N_CONDITIONS = 3249
    N_PER_CLASS = 100
    BATCH_SIZE = 10000
    TOTAL = N_CONDITIONS * 2 * N_PER_CLASS
    assert TOTAL == 649800
    assert BATCH_SIZE % (2 * N_PER_CLASS) == 0


    def make_plan():
        rng = np.random.default_rng(SEED)
        n_uniform = N_CONDITIONS // 2
        n_loguniform = N_CONDITIONS - n_uniform
        E_conditions = rng.uniform(1.0, 100.0, N_CONDITIONS),   
        rng.shuffle(E_conditions)
        g_conditions = rng.uniform(0.7, 1.3, N_CONDITIONS)

        boundary = rng.choice(N_CONDITIONS, size=round(N_CONDITIONS * 0.1), replace=False)
        energy_edges, gain_edges = boundary[:len(boundary)//2], boundary[len(boundary)//2:]
        E_conditions[energy_edges] = rng.choice([1.0, 100.0], len(energy_edges))
        g_conditions[gain_edges] = rng.choice([0.7, 1.3], len(gain_edges))

        y = np.tile(np.repeat([0, 1], N_PER_CLASS), N_CONDITIONS).astype(np.int64)
        E = np.repeat(E_conditions, 2 * N_PER_CLASS)
        g = np.repeat(g_conditions, 2 * N_PER_CLASS)
        assert np.bincount(y).tolist() == [324900, 324900]
        assert len(set(zip(E_conditions, g_conditions))) == N_CONDITIONS
        return y, E, g


    def run_simulation(token, save_dir=SAVE_DIR, session=None):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        session = requests.Session() if session is None else session
        headers = {"Authorization": f"Bearer {token.strip()}"}
        plan_file = save_dir / "parameter_plan_649800.npz"

        # При перезапуске используем тот же план, а не новые случайные параметры.
        y, E, g = make_plan()
        if plan_file.exists():
            with np.load(plan_file, allow_pickle=False) as old:
                if not all(np.array_equal(old[k], v) for k, v in zip(["y", "E", "g"], [y, E, g])):
                    raise RuntimeError("Сохранённый план отличается. Выберите другую папку.")
        else:
            np.savez_compressed(plan_file, y=y, E=E, g=g)

        # Папка с .pending означает неопределённый исход прошлого POST.
        # Нельзя повторять его автоматически: бюджет мог уже списаться.
        pending = list(save_dir.glob("*.pending"))
        if pending:
            raise RuntimeError(
                f"Незавершённый запрос: {pending[0].name}. Сверьте /api/status. "
                "Не удаляйте маркер и не повторяйте запрос без проверки бюджета."
            )

        data_dict = {}
        for start in range(0, TOTAL, BATCH_SIZE):
            stop = min(start + BATCH_SIZE, TOTAL)
            batch_file = save_dir / f"batch_649800_{start:06d}_{stop:06d}.npz"
            marker = batch_file.with_suffix(".pending")

            if batch_file.exists():
                with np.load(batch_file, allow_pickle=False) as saved:
                    X_batch = saved["X"].copy()
                    if not all(np.array_equal(saved[k], v[start:stop])
                            for k, v in zip(["y", "E", "g"], [y, E, g])):
                        raise RuntimeError(f"Метаданные не совпадают: {batch_file.name}")
                print("Из кэша:", batch_file.name)
            else:
                params = {
                    "y": y[start:stop].tolist(),
                    "E": E[start:stop].tolist(),
                    "g": g[start:stop].tolist(),
                    "n": stop - start,
                    "format": "npz",
                }
                marker.write_text(json.dumps({"start": start, "stop": stop}), encoding="utf8")
                # POST не имеет автоматических повторов. Маркер остаётся при ошибке.
                response = session.post(
                    f"{BASE_URL}/api/simulate", headers=headers, json=params, timeout=240,
                )
                if response.status_code != 200:
                    raise RuntimeError(
                        f"Ошибка API {response.status_code}. Маркер сохранён; "
                        "сверьте статус перед повтором."
                    )
                with np.load(io.BytesIO(response.content), allow_pickle=False) as received:
                    X_batch = received["X"].copy()

            if X_batch.shape != (stop-start, 4, 8, 8):
                raise RuntimeError(f"Неожиданная форма X: {X_batch.shape}")
            if not np.isfinite(X_batch).all() or (X_batch < 0).any():
                raise RuntimeError("Некорректные значения энергии в X.")

            if not batch_file.exists():
                # Сначала сохраняем результат, затем снимаем маркер.
                temporary_file = batch_file.with_suffix(".tmp.npz")
                np.savez_compressed(temporary_file, X=X_batch,
                                    y=y[start:stop], E=E[start:stop], g=g[start:stop])
                os.replace(temporary_file, batch_file)
                marker.unlink()
                print(f"Получено: {stop:,} / {TOTAL:,}")

            # Каждая партия содержит целые группы по 100 событий.
            for local in range(0, len(X_batch), N_PER_CLASS):
                idx = start + local
                key = (int(y[idx]), float(E[idx]), float(g[idx]))
                if key in data_dict:
                    raise RuntimeError("Повторный ключ: план содержит дубликаты условий.")
                data_dict[key] = X_batch[local:local+N_PER_CLASS].copy()

        assert len(data_dict) == 6498
        assert all(a.shape == (100, 4, 8, 8) for a in data_dict.values())
        assert sum(len(a) for a in data_dict.values()) == 649800

        # Сохраняем словарь без pickle: keys[i] соответствует values[i].
        # Форма values: (6498,100,4,8,8). После чтения восстанавливаем Python dict.
        keys = np.array(list(data_dict), dtype=np.float64)
        values = np.stack(list(data_dict.values()))
        filename = save_dir / "calo_dictionary_649800.npz"
        temporary_file = filename.with_suffix(".tmp.npz")
        np.savez_compressed(temporary_file, keys=keys, values=values)
        os.replace(temporary_file, filename)
        print("Число ключей:", len(data_dict))
        print("Событий каждого класса:", np.bincount(y))
        print("Сохранено:", filename)
        return data_dict


    data_dict = run_simulation(TOKEN)
    first_key = next(iter(data_dict))


from pathlib import Path

def test_simulatoun():
    TOTAL_EVENTS = 30_000
    BATCH_SIZE = 1_000

    X_parts = []
    event_id_parts = []
    offset = 0

    with requests.Session() as session:
        session.headers.update(HEADERS)

        while offset < TOTAL_EVENTS:
            limit = min(BATCH_SIZE, TOTAL_EVENTS - offset)

            response = session.get(
                f"{BASE_URL}/api/observe/dev",
                params={
                    "offset": offset,
                    "limit": limit,
                    "format": "npz",
                },
                timeout=120,
            )

            if response.status_code != 200:
                raise RuntimeError(
                    f"Ошибка {response.status_code} при offset={offset}: "
                    f"{response.text[:500]}"
                )

            # Проверяем, достаточно ли событий в сеансе.
            total_header = response.headers.get("X-Total-Events")
            if total_header is not None and int(total_header) < TOTAL_EVENTS:
                raise RuntimeError(
                    f"В сеансе только {total_header} событий, "
                    f"а запрошено {TOTAL_EVENTS}."
                )

            with np.load(io.BytesIO(response.content), allow_pickle=False) as data:
                X_batch = data["X"].copy()
                ids_batch = data["event_id"].copy()

            n_received = len(X_batch)

            if X_batch.shape != (n_received, 4, 8, 8):
                raise RuntimeError(f"Неожиданная форма X: {X_batch.shape}")

            if ids_batch.shape != (n_received,):
                raise RuntimeError(
                    f"Неожиданная форма event_id: {ids_batch.shape}"
                )

            if n_received == 0 or n_received > limit:
                raise RuntimeError(
                    f"При offset={offset} получено {n_received} событий "
                    f"при запрошенных {limit}."
                )

            X_parts.append(X_batch)
            event_id_parts.append(ids_batch)

            offset += n_received
            print(f"Загружено {offset:,} / {TOTAL_EVENTS:,}")

    X = np.concatenate(X_parts, axis=0)
    event_id = np.concatenate(event_id_parts, axis=0)

    if len(np.unique(event_id)) != TOTAL_EVENTS:
        raise RuntimeError("Обнаружены повторяющиеся идентификаторы событий.")

    print("\nФорма X:", X.shape)               # (30000, 4, 8, 8)
    print("Форма event_id:", event_id.shape)   # (30000,)
    print(f"Размер X в памяти: {X.nbytes / 1024**2:.1f} МБ")

    
    save_dir = Path(SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True) 

    output_path = save_dir / "run_a_all_30000.npz"

    np.savez_compressed(
        output_path,
        X=X,
        event_id=event_id,
    )

    print(f"Данные сохранены в {output_path}")

PREDICTOR = None


def load_predictor(weights_path=None, device="cpu"):
    from pathlib import Path
    from naps_calo.inference import ResCNNNAPSPredictor

    project_dir = Path(__file__).resolve().parent

    default_path = (
        project_dir / "weights" / "NAPS_Calo_ResCNN_weights.pt"
    )

    path = Path(
        default_path if weights_path is None else weights_path
    ).expanduser()

    if not path.is_absolute():
        path = project_dir / path

    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден файл весов: {path}\n"
            "Укажите его путь через --weights."
        )

    return ResCNNNAPSPredictor(path, device=device)


def pred(unlabeled_path=None, weights_path=None, output_path=None,
         device="cpu", batch_size=2048):
    
    

    global PREDICTOR
    project_dir = Path(__file__).resolve().parent

    def resolve_path(value, default):
        path = Path(default if value is None else value).expanduser()
        return path if path.is_absolute() else project_dir / path

    input_path = resolve_path(
        unlabeled_path,
        r"C:\Users\m12g\.vscode\NAPS_VSCode_fixed\NAPS_VSCode\data\simulations\run_a_all_30000.npz",
    )
    csv_path = resolve_path(
        output_path,
        r"C:\Users\m12g\.vscode\NAPS_VSCode_fixed\NAPS_VSCode\data\simulations\naps_predictions_run_a.csv",
    )
    if not input_path.is_file():
        raise FileNotFoundError(f"Не найден датасет: {input_path}")
    if batch_size < 1:
        raise ValueError("batch_size должен быть положительным")

    with np.load(input_path, allow_pickle=False) as data:
        X_new = np.asarray(data["X"], dtype=np.float32)
        event_ids = (
            np.asarray(data["event_id"])
            if "event_id" in data.files else np.arange(len(X_new))
        )

    if X_new.ndim != 4 or X_new.shape[1:] != (4, 8, 8):
        raise ValueError(f"Ожидается X формы (N,4,8,8), получено {X_new.shape}")
    if not np.isfinite(X_new).all() or np.any(X_new < 0):
        raise ValueError("X должен содержать конечные неотрицательные энергии")
    if event_ids.ndim != 1 or len(event_ids) != len(X_new):
        raise ValueError("event_id должен быть одномерным массивом длины N")

    print("Загружено событий:", len(X_new))
    # Объект создаётся до вызова predict; notebook-переменные не нужны.
    PREDICTOR = load_predictor(weights_path, device=device)
    probabilities, H_codes = PREDICTOR.predict(X_new, batch_size=batch_size)

    results = pd.DataFrame({"event_id": event_ids, "P_gamma": probabilities})
    # Порядок и уровни alpha берём из загруженного checkpoint.
    for index, alpha in enumerate(PREDICTOR.alphas):
        results[f"H_{alpha:.2f}"] = H_codes[:, index]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv_path, index=False)
    print(results.head(20).to_string(index=False))
    print("Результаты сохранены:", csv_path)
    return results


def to_json():
    df = pd.read_csv(r"C:\Users\m12g\.vscode\NAPS_VSCode_fixed\NAPS_VSCode\data\simulations\naps_predictions_run_a.csv")

    payload = {
        "event_id": df["event_id"].astype(str).tolist(),
        "sets": {
            "0.05": df["H_0.05"].astype(int).tolist(),
            "0.10": df["H_0.10"].astype(int).tolist(),
            "0.20": df["H_0.20"].astype(int).tolist()
        }
    }

    with open(r"C:\Users\m12g\.vscode\NAPS_VSCode_fixed\NAPS_VSCode\data\simulations\run_a_prediction_sets.json", "w") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(json.dumps({
        "event_id": payload["event_id"][:3],
        "sets": {k: v[:3] for k, v in payload["sets"].items()}
    }, indent=2))


def send_res():
    from getpass import getpass

    URL = "https://dam-reader-disclosure-climb.trycloudflare.com/api/submit/run_a"


    token = TOKEN

    if token.lower().startswith("bearer "):
        token = token[7:].strip()

    with open("/content/run_a_prediction_sets.json", "r", encoding="utf-8") as f:
        payload = json.load(f)

    response = requests.post(
        URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        },
        json=payload,
        timeout=180
    )

    print("HTTP status:", response.status_code)

    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2)[:30000])
    except ValueError:
        print(response.text[:10000])

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Предсказания ResCNN + NAPS")
    parser.add_argument("--input", help="NPZ с X и необязательным event_id")
    parser.add_argument("--weights", help="Checkpoint с сетью и порогами NAPS")
    parser.add_argument("--output", help="Путь к выходному CSV")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=2048)
    args = parser.parse_args()
    #pred(args.input, args.weights, args.output, args.device, args.batch_size)
    #to_json()
