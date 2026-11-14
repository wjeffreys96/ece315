from collections.abc import Callable

import numpy as np
from matplotlib import pyplot as plt

from ece315.plottable import Plottable

plt.style.use("seaborn-v0_8-muted")


def main() -> None:
    # Create a list of plottable functions
    plots = []

    # For each function in plots, create a figure, and plot the function of n on it
    for p in plots:
        p.plot(plt)

    try:
        plt.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
