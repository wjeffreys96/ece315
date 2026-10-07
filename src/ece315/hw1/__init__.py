from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    # Create a numpy array containing the t values
    t = np.arange(-8, 8.25, step=0.25)

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
    ]

    # For each function in plots, create a figure, and plot the function of t on it
    for p in plots:
        plt.figure()
        plt.xlim(p.xlim)
        plt.ylim(p.ylim)
        plt.xticks(p.xticks)
        plt.yticks(p.yticks)
        plt.title(p.title)
        plt.grid()
        plt.plot(t, p.fn(t))

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


# Problem 2a
def x(t: np.ndarray) -> np.ndarray:
    """
    Use the modulo operator to create a repeating cycle from -1 to 2.
    Then mask off the important sections of the period and apply the
    piecewise linear functions to them. Returns the resulting array
    """
    period = 3

    # create a periodic cycle from t, and align it to [-1, 2], starting at 1
    cycle = (t + 1) % period - 1

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
