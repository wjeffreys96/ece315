from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np


# Dataclass to hold information about a function and its plot attributes
@dataclass()
class Plottable:
    fn: Callable
    title: str
    kind: Literal["ct", "dt", "ctx", "dtx"] = "dt"
    xlim: tuple[float, float] = (-10, 10)
    ylim: tuple[float, float] = (-10, 10)
    xstep: int | None = None
    ystep: int | None = None
    domain: np.ndarray | None = None

    def plot(self, plt) -> None:
        match self.kind:
            case "ct":
                self.plot_ct(plt)
            case "dt":
                self.plot_dt(plt)
            case "ctx":
                self.implot_ct(plt)
            case "dtx":
                self.implot_dt(plt)

    def _set_styles(self, plt) -> None:
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.xticks(np.arange(self.xlim[0], self.xlim[1] + 1, step=self.xstep))
        plt.yticks(np.arange(self.ylim[0], self.ylim[1] + 1, step=self.ystep))
        plt.title(self.title)
        plt.grid()

    @staticmethod
    def _pad(lo: float, hi: float, margin: float = 0.05) -> tuple[float, float]:
        span = hi - lo
        pad = margin * span if span > 0 else 1.0
        return lo - pad, hi + pad

    def _set_imstyles(self, plt, lo: float, hi: float) -> None:
        plt.xlim(self._pad(*self.xlim))
        plt.ylim(self._pad(lo, hi))
        plt.xticks(np.arange(self.xlim[0], self.xlim[1] + 1, step=self.xstep))
        plt.grid()
        plt.legend()

    def plot_ct(self, plt) -> None:
        t = (
            self.domain
            if self.domain is not None
            else np.linspace(-10, 10, endpoint=True)
        )
        plt.figure()
        self._set_styles(plt)
        plt.plot(t, self.fn(t))

    def implot_ct(self, plt) -> None:
        raise NotImplementedError

    def _2func_stem(
        self,
        plt,
        f1: np.ndarray,
        f2: np.ndarray,
        n: np.ndarray,
        f1_label: str,
        f2_label,
    ) -> None:
        plt.figure()
        lo = min(np.min(f1), np.min(f2))
        hi = max(np.max(f1), np.max(f2))
        plt.title(f"{f1_label} and {f2_label} of {self.title}")
        plt.stem(n, f1, linefmt=":", markerfmt="o", label=f1_label)
        plt.stem(n, f2, linefmt="--", markerfmt="s", label=f2_label)
        self._set_imstyles(plt, lo, hi)

    def implot_dt(self, plt) -> None:
        n = self.domain if self.domain is not None else np.arange(-10, 11)
        xn = self.fn(n)
        self._2func_stem(plt, np.abs(xn), np.angle(xn), n, "Magnitude", "Phase")
        self._2func_stem(
            plt, np.real(xn), np.imag(xn), n, "Real Part", "Imaginary Part"
        )

    def plot_dt(self, plt) -> None:
        n = self.domain if self.domain is not None else np.arange(-10, 11)
        plt.figure()
        self._set_styles(plt)
        plt.stem(n, self.fn(n))
