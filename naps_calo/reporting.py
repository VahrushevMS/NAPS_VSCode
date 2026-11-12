"""Вывод таблиц в терминал вместо IPython.display."""


def print_frame(title, frame):
    print(f"\n{title}", flush=True)
    if frame.empty:
        print("Нет записей.", flush=True)
    else:
        print(frame.to_string(index=False), flush=True)
