from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    # Create a numpy array containing the t values
    n = np.arange(-10, 11)

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
    ]

    # For each function in plots, create a figure, and plot the function of t on it
    for p in plots:
        p.stem(plt, n)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


def bkwd_diff(x: Callable, n: np.ndarray) -> np.ndarray:
    return x(n) - x(n - 1)


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


# Dataclass to hold information about a function and its plot attributes
@dataclass()
class Plottable:
    fn: Callable
    title: str
    xlim: tuple[float, float] = (-10, 10)
    ylim: tuple[float, float] = (-10, 10)
    step: int = 1

    def _set_styles(self, plt) -> None:
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.xticks(np.arange(self.xlim[0], self.xlim[1] + 1, step=self.step))
        plt.yticks(np.arange(self.ylim[0], self.ylim[1] + 1, step=self.step))
        plt.title(self.title)
        plt.grid()

    def plot(self, plt, t: np.ndarray) -> None:
        plt.figure()
        self._set_styles(plt)
        plt.plot(t, self.fn(t))

    def stem(self, plt, n: np.ndarray) -> None:
        plt.figure()
        self._set_styles(plt)
        plt.stem(n, self.fn(n))


if __name__ == "__main__":
    main()
