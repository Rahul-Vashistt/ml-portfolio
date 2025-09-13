from typing import Union, Sequence, Optional, Mapping, Literal, Tuple, Any
from matplotlib.colors import Colormap
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR)

'''For Bivariate analysis'''

DataVar = Union[str, Sequence[str], pd.Series, np.ndarray]

def scatter(
        df: pd.DataFrame = None,
        x: DataVar = None,
        y: DataVar = None,
        min_inns: int = None,
        min_mat: int = None,
        hue: Optional[DataVar] = None,
        size: Optional[DataVar] = None,
        style: Optional[DataVar] = None,
        palette: Optional[Union[str, Sequence[str], Mapping[object, object], Colormap]] = None,
        legend: Union[Literal['auto'], Literal['brief'], Literal['full'], bool] = 'auto',
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[float, float] = (10,6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        fig_kwargs: Optional[Mapping[str, Any]] = None,
        scatter_kwargs: Optional[Mapping[str, Any]] = None
) -> plt.Axes:
    """
    Create a fully customizable scatter plot using Seaborn's `scatterplot`.

    Parameters
    ----------
    df : pandas.DataFrame, default=None
        The input DataFrame containing the data to plot.
    x, y : str, sequence, or array-like, default=None
        Variables to plot on the x- and y-axes. If strings, they must be column
        names present in `df`. If sequence/array-like, they should be aligned
        with the rows of `df`.
    min_inns : int, optional
        If provided and the column 'Inns' exists in `df`, filters rows to those
        with Inns >= `min_inns` before plotting.
    min_mat : int, optional
        If provided and the column 'Mat' exists in `df`, filters rows to those
        with Mat >= `min_mat` before plotting.
    hue : str, sequence, or array-like, optional
        Grouping variable that will produce points with different colors.
        Can be a column name in `df` or an array-like of the same length.
    size : str, sequence, or array-like, optional
        Grouping variable that will vary the marker sizes.
    style : str, sequence, or array-like, optional
        Grouping variable that will vary the marker styles.
    palette : str, sequence, mapping, or matplotlib.colors.Colormap, optional
        Colors to use for the different levels of the `hue` variable.
    legend : {'auto', 'brief', 'full'} or bool, default='auto'
        How to draw the legend; see Seaborn documentation for details.
    ax : matplotlib.axes.Axes, optional
        Existing Matplotlib Axes to draw the plot onto. If None, a new figure
        and axes are created with size `figsize`.
    figsize : tuple of float, default=(10, 6)
        Figure size (width, height) in inches when creating a new figure.
    title : str, optional
        Plot title. If None, defaults to "<x> vs <y>".
    xlabel : str, optional
        Label for the x-axis. If None, defaults to `x` (stringified).
    ylabel : str, optional
        Label for the y-axis. If None, defaults to `y` (stringified).
    fig_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `matplotlib.pyplot.figure`
        when creating a new figure (e.g., dpi, facecolor).
    scatter_kwargs : mapping of str to Any, optional
        Additional keyword arguments forwarded to `seaborn.scatterplot`
        (e.g., s, alpha, markers, edgecolor).

    Returns
    -------
    matplotlib.axes.Axes
        The Matplotlib Axes containing the scatter plot.

    Raises
    ------
    ValueError
        If `df` is None or empty, or if either `x` or `y` is not provided.
    KeyError
        If `x` or `y` is a string and not found in `df` columns.

    Notes
    -----
    - Applies optional row filtering using 'Inns' and/or 'Mat' columns if
      `min_inns`/`min_mat` are provided and those columns exist.
    - When `ax` is None, a new figure is created with `figsize`; otherwise the
      plot is drawn on the provided axes.
    - Use `scatter_kwargs` to fine-tune Seaborn aesthetics such as marker size,
      transparency, and edge colors.
    """
    
    # Validation
    if df is None or df.shape[0] == 0:
        raise ValueError("DataFrame is empty or None.")
    if x is None or y is None:
        raise ValueError("Both x and y must be provided.")
    # validate provided columns if str
    if isinstance(x, str) and x not in df.columns:
        raise KeyError(f"Column '{x}' not in df")
    if isinstance(y, str) and y not in df.columns:
        raise KeyError(f"Column '{y}' not in df")

    # Threshold filter
    # operate on a copy and filter safely if columns exist
    d = df.copy()
    if min_inns is not None and "Inns" in d.columns:
        d = d[d["Inns"] >= min_inns]
    if min_mat is not None and "Mat" in d.columns:
        d = d[d["Mat"] >= min_mat]

    # Figure params
    fig_params = {"figsize": figsize}
    if fig_kwargs:
        fig_params.update(fig_kwargs)
    if ax is None:
        plt.figure(**fig_params)

    # Scatter params
    scatter_params = {
        "hue": hue,
        "size": size,
        "style": style,
        "palette": palette,
        "legend": legend,
        "ax": ax
    }
    if scatter_kwargs:
        scatter_params.update(scatter_kwargs)

    # Create scatterplot
    ax = sns.scatterplot(data=d, x=x, y=y, **scatter_params)

    # Labels
    ax.set_title(title or f"{x} vs {y}")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)

    plt.tight_layout()
    plt.show()
    return ax