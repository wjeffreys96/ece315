from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt

from ece315.plottable import Plottable

XP3A_PERIOD = 63
XP3C_PERIOD = 672
XP3A_DOMAIN = np.arange(0, XP3A_PERIOD)
XP3C_DOMAIN = np.arange(0, XP3C_PERIOD, step=5)
XP4_DOMAIN = np.arange(0, 55)

plt.style.use("seaborn-v0_8-muted")


def main() -> None:
    # Create a list of plottable functions
    plots = [
        Plottable(
            fn=x1,
            title="$x_1[n]$",
            ylim=(-2, 20),
        ),
        Plottable(
            fn=x2,
            title="$x_2[n]$",
            ylim=(-2, 20),
        ),
        Plottable(
            fn=x3,
            title="$x_3[n]$",
            ylim=(-2, 20),
        ),
        Plottable(
            fn=lambda n: bkwd_diff(x1, n),
            title="$y_1[n]$",
        ),
        Plottable(
            fn=y2,
            title="$y_2[n]$",
        ),
        Plottable(
            fn=lambda n: bkwd_diff(x3, n),
            title="$y_3[n]$",
        ),
        Plottable(
            fn=lambda n: np.cumsum(y2(n)),
            title="$\\tilde{x}[n]$",
            ylim=(-1, 20),
        ),
        Plottable(
            fn=xp3a,
            title="$x(n)$ Problem 3 Part a",
            domain=XP3A_DOMAIN,
            xlim=(0, XP3A_PERIOD),
            ylim=(-5, 5),
            xstep=10,
        ),
        Plottable(
            fn=xp3b,
            title="$x(n)$ Problem 3 Part b",
            domain=np.arange(0, 25),
            xlim=(-1, 25),
            ylim=(-5, 5),
            xstep=2,
        ),
        Plottable(
            fn=xp3c,
            kind="dtx",
            title="$x_{p3c}(n)$",
            domain=XP3C_DOMAIN,
            xlim=(0, XP3C_PERIOD),
            ylim=(-5, 5),
            xstep=50,
        ),
        Plottable(
            fn=xp3d,
            kind="dtx",
            title="$x_{p3d}(n)$",
            domain=np.arange(-15, 15),
            xlim=(-15, 15),
            ylim=(-25, 25),
            xstep=2,
            ystep=2,
        ),
        Plottable(
            fn=x1p4,
            title="$x_{1p4}(n)$",
            ylim=(-5, 5),
            xlim=(float(np.min(XP4_DOMAIN)) - 1, float(np.max(XP4_DOMAIN) + 1)),
            xstep=2,
            domain=XP4_DOMAIN,
        ),
        Plottable(
            fn=x2p4,
            title="$x_{2p4}(n)$",
            ylim=(-5, 5),
            xlim=(float(np.min(XP4_DOMAIN)) - 1, float(np.max(XP4_DOMAIN) + 1)),
            xstep=2,
            domain=XP4_DOMAIN,
        ),
    ]

    # For each function in plots, create a figure, and plot the function of n on it
    for p in plots:
        p.plot(plt)

    print(
        f"Average power of xp3a: {avg_pow_periodic(xp3a, np.arange(0, XP3A_PERIOD + 1))} square peanut butter and jelly sandwiches"
    )
    print(
        f"Average power of xp3c: {avg_pow_periodic(xp3c, XP3C_DOMAIN)} square peanut butter and jelly sandwiches"
    )
    n = np.arange(0, 55)
    print(f"(Problem 4) x1 == x2: {np.allclose(x1p4(n), x2p4(n))}")

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


def bkwd_diff(x: Callable, n: np.ndarray) -> np.ndarray:
    return x(n) - x(n - 1)


def avg_pow_periodic(x: Callable, n: np.ndarray) -> float:
    return np.mean(np.abs(x(n)) ** 2)


def x1(n: np.ndarray) -> np.ndarray:
    boundary = (n >= -4) & (n <= 4)
    return np.piecewise(
        n,
        [
            boundary,
            ~boundary,
        ],
        [
            lambda t: -1 * np.power(t, 2) + 16,
            lambda t: 0,
        ],
    )


def x2(n: np.ndarray) -> np.ndarray:
    boundary = (n >= -4) & (n <= 4)
    return np.piecewise(
        n,
        [
            boundary,
            ~boundary,
        ],
        [
            lambda n: -1 * np.power(n, 2) + 18,
            lambda n: 2,
        ],
    )


def x3(n: np.ndarray) -> np.ndarray:
    boundary = (n >= -4) & (n <= 4)
    return np.piecewise(
        n,
        [
            boundary,
            ~boundary,
        ],
        [
            lambda n: -1 * np.power(n, 2) + 18,
            lambda n: 0,
        ],
    )


def y2(n: np.ndarray) -> np.ndarray:
    return bkwd_diff(x2, n)


def xp3a(n: np.ndarray) -> np.ndarray:
    return (8 / 3) * np.sin(((8 * np.pi * n) / 63) + ((3 * np.pi) / 5))


def xp3b(n: np.ndarray) -> np.ndarray:
    return (8 / 3) * np.sin(((24 * n) / 63) + ((3 * np.pi) / 5))


def xp3c(n: np.ndarray) -> np.ndarray:
    return 4 * np.exp(1j * ((n * (5 * np.pi) / 336) - (np.pi / 6)))


def xp3d(n: np.ndarray) -> np.ndarray:
    return 3 * np.exp(n * (((-5 * np.pi) / 112) + (7j * np.pi / 100)))


def x1p4(n: np.ndarray) -> np.ndarray:
    return (5 / 2) * np.cos(((452 * np.pi * n) / 55) + (np.pi / 4))


def x2p4(n: np.ndarray) -> np.ndarray:
    return (5 / 2) * np.cos(((12 * np.pi * n) / 55) + (np.pi / 4))


if __name__ == "__main__":
    main()
