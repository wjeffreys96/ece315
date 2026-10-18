from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    # Create a numpy array containing the t values
    t = np.linspace(-8, 8, endpoint=True, num=1000)

    # Create a list of plottable functions
    plots = [
        Plottable(
            fn=x,
            title="$x(t)$",
            xlim=(-9, 9),
            ylim=(-4, 4),
            xticks=np.arange(-9, 10),
            yticks=np.arange(-4, 5),
        ),
        Plottable(
            fn=y,
            title="$y(t)$",
            xlim=(-9, 9),
            ylim=(-7, 7),
            xticks=np.arange(-9, 10),
            yticks=np.arange(-7, 8),
        ),
        Plottable(
            fn=y_tilde,
            title="$\\tilde{y}(t)$",
            xlim=(-9, 9),
            ylim=(-7, 7),
            xticks=np.arange(-9, 10),
            yticks=np.arange(-7, 8),
        ),
        Plottable(
            fn=xp1,
            title="$x_{p1}(t)$",
            xlim=(-6, 6),
            ylim=(-3, 3),
            xticks=np.arange(-6, 7),
            yticks=np.arange(-3, 4),
        ),
        Plottable(
            fn=lambda t: get_even_part(xp1, t),
            title="$x_{p1even}(t)$",
            xlim=(-6, 6),
            ylim=(-3, 3),
            xticks=np.arange(-6, 7),
            yticks=np.arange(-3, 4),
        ),
        Plottable(
            fn=lambda t: get_odd_part(xp1, t),
            title="$x_{p1odd}(t)$",
            xlim=(-6, 6),
            ylim=(-3, 3),
            xticks=np.arange(-6, 7),
            yticks=np.arange(-3, 4),
        ),
        Plottable(
            fn=lambda t: get_odd_part(xp1, t) + get_even_part(xp1, t),
            title="$x_{p1odd}(t) + x_{p1even}(t)$",
            xlim=(-6, 6),
            ylim=(-3, 3),
            xticks=np.arange(-6, 7),
            yticks=np.arange(-3, 4),
        ),
        Plottable(
            fn=xp5b,
            title="$x_{p5b}(t)$",
            xlim=(-10, 10),
            ylim=(-10, 10),
            xticks=np.arange(-10, 10),
            yticks=np.arange(-10, 10),
        ),
        Plottable(
            fn=lambda t: get_even_part(xp5b, t),
            title="$x_{p5b,even}(t)$",
            xlim=(-10, 10),
            ylim=(-10, 10),
            xticks=np.arange(-10, 10),
            yticks=np.arange(-10, 11),
        ),
        Plottable(
            fn=lambda t: get_odd_part(xp5b, t),
            title="$x_{p5b,odd}(t)$",
            xlim=(-10, 10),
            ylim=(-10, 10),
            xticks=np.arange(-10, 10),
            yticks=np.arange(-10, 11),
        ),
        Plottable(
            fn=lambda t: get_odd_part(xp5b, t) + get_even_part(xp5b, t),
            title="$x_{p5b,odd}(t) + x_{p5b,even}(t)$",
            xlim=(-10, 10),
            ylim=(-10, 10),
            xticks=np.arange(-10, 10),
            yticks=np.arange(-10, 11),
        ),
    ]

    # For each function in plots, create a figure, and plot the function of t on it
    for p in plots:
        p.plot(plt, t)

    plt.show()


# Dataclass to hold information about a function and its plot attributes
@dataclass()
class Plottable:
    fn: Callable
    title: str
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    xticks: np.ndarray
    yticks: np.ndarray

    def plot(self, plt, t: np.npdarray):
        plt.figure()
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.xticks(self.xticks)
        plt.yticks(self.yticks)
        plt.title(self.title)
        plt.grid()
        plt.plot(t, self.fn(t))


# Problem 2a
def x(t: np.ndarray) -> np.ndarray:
    """
    Use the modulo operator to create a repeating cycle from -1 to 2.
    Then mask off the important sections of the period and apply the
    piecewise linear functions to them. Returns the resulting array
    """
    period = 3

    # create a periodic cycle from t, and align it to [-1, 2], starting at 1
    cycle = np.mod((t + 1), period) - 1

    # Mask off the two portions of the period that each linear function applies to
    one_third_period_msk = (cycle >= 1) & (cycle < 2)
    remaining_period_msk = ~one_third_period_msk

    # Apply the piecewise functions to the t values for the 2 sections
    cycle[one_third_period_msk] *= 3  # slope
    cycle[one_third_period_msk] += -5  # intercept

    cycle[remaining_period_msk] *= -2  # 2nd line segment slope

    return cycle


# Problem 2c
def y(t: np.ndarray) -> np.ndarray:
    """
    Apply the following transformations to x(t):
    - Time shift by 1/2 in the positive direction
    - Time stretch by 3/4
    - Amplitude scale by 3
    """
    return 3 * x(0.75 * (t - 0.5))


# Problem 2e
def y_tilde(t: np.ndarray) -> np.ndarray:
    """
    Interchange the time shift and time stretch transformations applied to x(t) in y(t)
    """
    return 3 * x((0.75 * t) - 0.5)


# Problem 5
def xp1(t: np.ndarray) -> np.ndarray:
    """
    Return the piecewise signal shown in problem 1
    """
    return np.piecewise(
        t,  # the input domain
        [  # the conditions:
            t <= -4,
            ((-4 < t) & (t < 1)),
            ((1 <= t) & (t <= 3)),
            (t > 3),
        ],
        [  # scalars/functions to be applied to each condition's range:
            0,
            lambda t: (3 / 5) * t + (7 / 5),
            lambda t: (-2) * t + 4,
            0,
        ],
    )


def xp5b(t: np.ndarray) -> np.ndarray:
    """
    Return the signal defined in problem 5 part b
    """
    return (np.power(t, 2) - 2 * t - 3) / (2 * t - 3)


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
