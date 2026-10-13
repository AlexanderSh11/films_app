import time
from contextlib import contextmanager


@contextmanager
def timer(label="Блок"):
    """Контекстный менеджер для замера времени блока кода"""
    time_start = time.time()
    yield
    print(f"[timer] {label}: {(time.time() - time_start):.3f} сек.")
