from typing import Union, Sequence, Optional, Mapping, Literal, Tuple, Any
from matplotlib.colors import Colormap
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR)


def hist(
    df: pd.DataFrame,
    col: Union[str, Sequence[str]],
    bins: Union[int, Sequence[float], str] = 30,
    log: bool = False,
    figsize: Tuple[float, float] = (10, 6),
    kde: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: str = "count",
    histplot_kwargs: Optional[Mapping[str, Any]] = None,
    fig_kwargs: Optional[Mapping[str, Any]] = None
):
    """
    Create a fully customizable histogram using Seaborn's `histplot`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to plot.
    col : str or sequence of str
        Column name(s) to plot. If multiple columns are provided, their data
        is concatenated for plotting.
    bins : int, sequence of float, or str, default=30
        Number of bins, bin edges, or a binning strategy name supported by
        Matplotlib (e.g., 'auto', 'fd', 'sturges').
    log : bool, default=False
        If True, sets the y-axis to a logarithmic scale.
    figsize : tuple of float, default=(10, 6)
        Figure size (width, height) in inches.
    kde : bool, default=False
        If True, overlays a kernel density estimate on the histogram.
    title : str, optional
        Plot title. If None, defaults to "<col> distribution".
    xlabel : str, optional
        Label for the x-axis. If None, defaults to `col`.
    ylabel : str, default="count"
        Label for the y-axis.
    histplot_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `seaborn.histplot`.
    fig_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `matplotlib.pyplot.figure`.

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure containing the histogram.

    Notes
    -----
    - `fig_kwargs` is ignored if you have an existing figure context.
    - `histplot_kwargs` can include arguments such as `color`, `alpha`,
      `stat`, `multiple`, etc.
    """

    # Validation of data and axis
    try:
        if df is None:
            logging.error("Missing required parameter 'df'.")
            raise ValueError("df is None")  # raise concrete error
        elif df.shape[0] == 0:
            logging.exception("Data can't be empty")
            raise ValueError("df has no rows")  # actually raise
        elif col is None:
            # use first numeric column
            first_numeric = df.select_dtypes(include=['number']).columns
            if len(first_numeric) == 0:
                raise ValueError("No numeric column found for histogram")
            series = df[first_numeric].dropna()
            col_name = first_numeric
        else:
            # handle sequence of columns by concatenation
            if isinstance(col, (list, tuple, pd.Index, np.ndarray)):
                missing = [c for c in col if c not in df.columns]
                if missing:
                    raise KeyError(f"Columns not in df: {missing}")
                series = pd.concat([df[c] for c in col], ignore_index=True).dropna()
                col_name = ", ".join(map(str, col))
            else:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not in df")
                series = df[col].dropna()
                col_name = str(col)
    except Exception as e:
        logging.exception(f'Unexpected error during validation of data. {e}')
        raise
    
    # Merge user figure kwargs
    fig_params = {"figsize": figsize}
    if fig_kwargs:
        fig_params.update(fig_kwargs)
    
    plt.figure(**fig_params)
    
    # Merge user histplot kwargs
    hist_params = {
        "bins": bins,
        "kde": kde
    }
    if histplot_kwargs:
        hist_params.update(histplot_kwargs)
    
    sns.histplot(series, **hist_params)  

    if log:
        plt.yscale("log")

    plt.title(title or f"{col_name} distribution")  
    plt.xlabel(xlabel or col_name)                  
    plt.ylabel(ylabel)
    plt.tight_layout()  
    plt.show()