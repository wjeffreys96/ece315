from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt

XP3A_PERIOD = 63
XP3C_PERIOD = 672
XP3A_DOMAIN = np.arange(0, XP3A_PERIOD)
XP3C_DOMAIN = np.arange(0, XP3C_PERIOD)


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
            xstep=10,
        ),
        Plottable(
            fn=lambda n: np.abs(xp3c(n)),
            title="Magnitude of $x(n)$ for Problem 3 Part c",
            domain=np.arange(0, 11),
            xlim=(-1, 11),
        ),
        Plottable(
            fn=lambda n: np.angle(xp3c(n)),
            title="Phase Angle of $x(n)$ for Problem 3 Part c",
            domain=XP3C_DOMAIN,
            xlim=(0, XP3C_PERIOD),
            xstep=50,
        ),
        Plottable(
            fn=lambda n: np.real(xp3c(n)),
            title="Real Part of $x(n)$ for Problem 3 Part c",
            domain=XP3C_DOMAIN,
            xlim=(0, XP3C_PERIOD),
            xstep=50,
        ),
        Plottable(
            fn=lambda n: np.imag(xp3c(n)),
            title="Imaginary Part of $x(n)$ for Problem 3 Part c",
            domain=XP3C_DOMAIN,
            xlim=(0, XP3C_PERIOD),
            xstep=50,
        ),
    ]

    # For each function in plots, create a figure, and plot the function of n on it
    for p in plots:
        p.stem(plt)

    print(f"Average power of xp3a: {avg_pow_periodic(xp3a, XP3A_DOMAIN)}")
    print(f"Average power of xp3c: {avg_pow_periodic(xp3c, XP3C_DOMAIN)}")

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


def xp3c(n: np.ndarray) -> np.ndarray:
    return 4 * np.exp(1j * ((n * (5 * np.pi) / 336) - (np.pi / 6)))


# Dataclass to hold information about a function and its plot attributes
@dataclass()
class Plottable:
    fn: Callable
    title: str
    xlim: tuple[float, float] = (-10, 10)
    ylim: tuple[float, float] = (-10, 10)
    xstep: int = 1
    ystep: int = 1
    domain: np.ndarray | None = None

    def _set_styles(self, plt) -> None:
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.xticks(np.arange(self.xlim[0], self.xlim[1] + 1, step=self.xstep))
        plt.yticks(np.arange(self.ylim[0], self.ylim[1] + 1, step=self.ystep))
        plt.title(self.title)
        plt.grid()

    def plot(self, plt) -> None:
        t = (
            self.domain
            if self.domain is not None
            else np.linspace(-10, 10, endpoint=True)
        )
        plt.figure()
        self._set_styles(plt)
        plt.plot(t, self.fn(t))

    def stem(self, plt) -> None:
        n = self.domain if self.domain is not None else np.arange(-10, 11)
        plt.figure()
        self._set_styles(plt)
        plt.stem(n, self.fn(n))


if __name__ == "__main__":
    main()
