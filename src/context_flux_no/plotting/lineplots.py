import matplotlib.pyplot as plt
import numpy as np
from jaxtyping import ArrayLike, Float


def plot_line_and_band(
    ax: plt.Axes,
    x: Float[ArrayLike, " N"],
    y: Float[ArrayLike, " N"],
    y_widths: Float[ArrayLike, " N"]
    | tuple[Float[ArrayLike, " N"], Float[ArrayLike, " N"]],
    color: str = "royalblue",
    alpha: float = 1.0,
    alpha_band: float = 0.3,
    **plot_kwargs,
) -> plt.Axes:
    ax.plot(x, y, color=color, alpha=alpha, **plot_kwargs)
    if isinstance(y_widths, tuple):
        width_lower, width_upper = y_widths
    else:
        width_lower, width_upper = y_widths, y_widths

    ax.fill_between(x, y - width_lower, y + width_upper, color=color, alpha=alpha_band)
    return ax


def plot_mean_and_std(
    ax: plt.Axes,
    x: Float[ArrayLike, " N"],
    y: Float[ArrayLike, "B N"],
    color: str = "royalblue",
    alpha: float = 1.0,
    alpha_band: float = 0.3,
    **plot_kwargs,
) -> plt.Axes:
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)
    ax = plot_line_and_band(
        ax,
        x,
        y_mean,
        y_std,
        color=color,
        alpha=alpha,
        alpha_band=alpha_band,
        **plot_kwargs,
    )
    return ax
