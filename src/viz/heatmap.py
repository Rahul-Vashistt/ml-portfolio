from typing import Union, Sequence, Optional, Mapping, Literal, Tuple, Any
from matplotlib.colors import Colormap
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from viz import corr_table

logging.basicConfig(level=logging.ERROR)


'''For Multivariate analysis'''

def heatMap(
        df: pd.DataFrame,
        method: str = 'pearson',
        drop_boolean: bool = True,
        cmap: str = 'coolwarm',
        annot: bool = True,
        fmt: str = ".2f",
        figsize: Tuple[float, float] = (10,8),
        fig_kwargs: Optional[Mapping[str, Any]] = None,
        heatmap_kwargs: Optional[Mapping[str, Any]] = None
):
    """
    Create a correlation heatmap from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data.
    method : {'pearson', 'spearman'}, default='pearson'
        The correlation method to use for computing the correlation matrix.
    drop_boolean : bool, default=True
        Whether to drop boolean columns before computing correlations.
    cmap : str, default='coolwarm'
        Colormap to use for the heatmap.
    annot : bool, default=True
        Whether to display the correlation coefficient values inside the cells.
    fmt : str, default='.2f'
        String formatting code for annotations (e.g., '.2f' for 2 decimal places).
    figsize : tuple of float, default=(10, 8)
        Figure size (width, height) in inches.
    fig_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `matplotlib.pyplot.figure`.
    heatmap_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `seaborn.heatmap`.

    Returns
    -------
    None
        Displays the heatmap plot.

    Raises
    ------
    ValueError
        If the input DataFrame has no numeric columns or if the method is invalid.
    TypeError
        If `method` is not a string.

    Notes
    -----
    - Uses `corr_table` to compute the correlation matrix.
    - The `heatmap_kwargs` dictionary can be used to customize Seaborn's heatmap
      parameters, such as `linewidths`, `linecolor`, `square`, etc.
    """
     
    corr = corr_table(df, method=method, drop_boolean=drop_boolean, cmap=cmap)

    fig_params = {"figsize": figsize}
    if fig_kwargs:
        fig_params.update(fig_kwargs)

    plt.figure(**fig_params)
    
    heatmap_params = {
        "annot": annot,
        "cmap": cmap,
        "fmt": fmt
    }
    if heatmap_kwargs:
        heatmap_params.update(heatmap_kwargs)

    sns.heatmap(corr, **heatmap_params)
    plt.title(f"{method.title()} Correlation Heatmap")
    plt.tight_layout()
    plt.show()