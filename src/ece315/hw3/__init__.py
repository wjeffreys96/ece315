import numpy as np
from matplotlib import pyplot as plt

from ece315.plottable import Plottable

plt.style.use("seaborn-v0_8-muted")


def main() -> None:
    # Create a list of plottable functions
    plots = [
        Plottable(
            title="$y_{p1}[n]$",
            fn=y_p1,
        ),
        Plottable(
            title="$h_{p2}[n]$",
            fn=h_p2,
            domain=np.arange(-25, 26),
            xlim=(-10, 25),
            xstep=2,
            kind="dtx",
        ),
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
    return np.convolve(x_p1(n), h_p1(n), "same")


def h_p2(n: np.ndarray) -> np.ndarray:
    u = np.heaviside(n, 1)
    u_2 = np.heaviside(n - 2, 1)
    a = (1 - 1j) / 2
    b = (1 + 1j) / 2
    h_hat = lambda n: a * (b**n) + b * (a**n)
    return -2 * h_hat(n) * u + 3 * h_hat(n - 2) * u_2


if __name__ == "__main__":
    main()
