from collections.abc import Callable

import numpy as np


def get_even_part(x: Callable, t: np.ndarray) -> np.ndarray:
    """
    Return the even part of a given signal
    """
    return (1 / 2) * (x(t) + (x(-t)))


def get_odd_part(x: Callable, t: np.ndarray) -> np.ndarray:
    """
    Return the odd part of a give signal
    """
    return (1 / 2) * (x(t) - x(-t))


def bkwd_diff(x: Callable, n: np.ndarray) -> np.ndarray:
    return x(n) - x(n - 1)


def avg_pow_periodic(x: Callable, n: np.ndarray) -> float:
    return np.mean(np.abs(x(n)) ** 2)
