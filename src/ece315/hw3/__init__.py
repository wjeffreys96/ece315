from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt

from ece315.plottable import Plottable

plt.style.use("seaborn-v0_8-muted")


def main() -> None:
    n = np.arange(-10, 11)

    plt.figure()

    y = y_p1(n)

    plt.stem(2 * n, y)
    plt.grid()
    plt.ylim((-10, 10))

    # Create a list of plottable functions
    plots = [
        Plottable(
            title="$y_{p1}[n]$",
            fn=lambda n: y_p1(n),
        )
    ]

    # For each function in plots, create a figure, and plot the function of n on it
    for p in plots:
        p.plot(plt)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


def h_p1(n: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(n)
    ret[n == 1] = 2
    ret[n == 2] = -3
    ret[n == 3] = -1
    return ret


def x_p1(n: np.ndarray) -> np.ndarray:
    ret = np.zeros_like(n)
    ret[n == -2] = 1
    ret[n == -1] = 3
    ret[n == 0] = 2
    ret[n == 1] = 1
    return ret


def y_p1(n: np.ndarray) -> np.ndarray:
    ret = np.convolve(x_p1(n), h_p1(n), "same")
    return ret


if __name__ == "__main__":
    main()
