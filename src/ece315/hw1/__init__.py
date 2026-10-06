from timeit import default_timer as timer

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    t = np.arange(-10, 11, step=0.25)

    s = timer()
    for title, fn in (("x(t)", x), ("y(t)", y), ("y~(t)", y_tilde)):
        plt.figure()
        plt.title(title)
        plt.scatter(t, fn(t))
        plt.grid()
        plt.tight_layout()
    e = timer()

    print(e - s)
    plt.show()


# Problem 2a
def x(t: np.ndarray) -> np.ndarray:
    ret = np.copy(t)
    period = 3

    return ret


# Problem 2c
def y(t: np.ndarray) -> np.ndarray:
    ret = np.copy(t)

    return ret


# Problem 2e
def y_tilde(t: np.ndarray) -> np.ndarray:
    ret = np.copy(t)

    return ret
