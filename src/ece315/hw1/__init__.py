import numpy as np
from matplotlib import pyplot as plt


def main() -> None:
    # Create a numpy array containing the t values
    t = np.arange(-8, 9, step=0.25)

    # Create a dict where the key is the title of the graph,
    # and the value is a pointer to the function to be called on each t
    funcs = {"$x(t)$": x, "$y(t)$": y, "$\\tilde{y}(t)$": y_tilde}

    # For each function in funcs, create a figure, and plot the function of t on it
    for title, fn in funcs.items():
        plt.figure()
        plt.title(title)
        plt.grid()
        plt.plot(t, fn(t))

    plt.show()


# Problem 2a
def x(t: np.ndarray) -> np.ndarray:
    """
    Use the modulo operator to create a repeating cycle from -1 to 2.
    Then mask off the important sections of the period and apply the
    piecewise linear functions to them. Returns the resulting array
    """
    period = 3

    # Create the repeating cycle and shift its window
    # For the remaining_period_msk portion, it needs to start at -1
    cycle = (t + 1) % period - 1

    # Mask off the two portions of the period that each linear function applies to
    first_third_period_msk = (cycle >= 1) & (cycle < 2)
    remaining_period_msk = ~first_third_period_msk

    cycle[first_third_period_msk] *= 3  # apply the slope
    cycle[first_third_period_msk] += -5  # and the intercept

    cycle[remaining_period_msk] *= -2

    return cycle


# Problem 2c
def y(t: np.ndarray) -> np.ndarray:
    ret = np.copy(t)

    return ret


# Problem 2e
def y_tilde(t: np.ndarray) -> np.ndarray:
    ret = np.copy(t)

    return ret
