from typing import Union, Sequence, Optional, Mapping, Literal, Tuple, Any
from matplotlib.colors import Colormap
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR)

def box(
        df: pd.DataFrame,
        col: Union[str, Sequence[str]],
        orient: str = 'v',
        figsize: Tuple[float, float] = (8,6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        boxplot_kwargs: Optional[Mapping[str, Any]] = None,
        fig_kwargs: Optional[Mapping[str, Any]] = None 
):
    """
    Create a fully customizable boxplot using Seaborn's `boxplot`.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing the data to plot.
    col : str or sequence of str
        Column name(s) to plot. If multiple columns are provided, their data
        is concatenated for plotting.
    orient : {'v', 'h'}, default='v'
        Orientation of the boxplot. 'v' plots vertical boxes (y-axis values),
        'h' plots horizontal boxes (x-axis values).
    figsize : tuple of float, default=(8, 6)
        Figure size (width, height) in inches.
    title : str, optional
        Plot title. If None, defaults to "<col> distribution".
    xlabel : str, optional
        Label for the x-axis. Defaults to `col` if horizontal orientation.
    ylabel : str, optional
        Label for the y-axis. Defaults to `col` if vertical orientation.
    boxplot_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `seaborn.boxplot`.
    fig_kwargs : mapping of str to Any, optional
        Additional keyword arguments passed to `matplotlib.pyplot.figure`.

    Returns
    -------
    matplotlib.figure.Figure
        The created Matplotlib figure containing the boxplot.

    Raises
    ------
    TypeError
        If `orient` is not a string.
    ValueError
        If `orient` is not 'v' or 'h'.

    Notes
    -----
    - Orientation validation is strict; invalid values raise exceptions.
    - `boxplot_kwargs` can include arguments such as `color`, `width`,
      `whis`, `fliersize`, etc.
    """
    
    # Validation of data and axis
    try:
        if df is None:
            logging.error("Missing required parameter 'df'.")
            raise ValueError("df is None")  # raise error
        elif df.shape[0] == 0:
            logging.exception("Data can't be empty")
            raise ValueError("df has no rows")  # raise error
        elif col is None:
            # use first numeric column
            first_numeric = df.select_dtypes(include=['number']).columns
            if len(first_numeric) == 0:
                raise ValueError("No numeric column found for boxplot")
            data = df[first_numeric].dropna()
            col_name = first_numeric
        else:
            # handle list/sequence via melt for multiple boxes
            if isinstance(col, (list, tuple, pd.Index, np.ndarray)):
                missing = [c for c in col if c not in df.columns]
                if missing:
                    raise KeyError(f"Columns not in df: {missing}")
                data = df[list(col)].melt(var_name="variable", value_name="value").dropna()
                col_name = ", ".join(map(str, col))
            else:
                if col not in df.columns:
                    raise KeyError(f"Column '{col}' not in df")
                data = df[col].dropna()
                col_name = str(col)
    except Exception as e:
        logging.exception(f'Unexpected error during validation of data. {e}')
        raise
    
    # Validate Orientation
    try:
        if isinstance(orient, str):
            if orient.lower() not in ['v', 'h']:
                raise ValueError(f"{orient} invalid value. Choose either 'v' or 'h' only.")
            orient = orient.lower()
        else:
            raise TypeError("Value must be a string.")
    except (TypeError, ValueError) as e:
        logging.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logging.exception("Unexpected error during orientation validation")
        raise
    
    # Merge user figure kwargs
    fig_params = {"figsize" : figsize}
    if fig_kwargs:
        fig_params.update(fig_kwargs)

    plt.figure(**fig_params)

    # Merge user boxplot kwargs
    box_params = {"orient": orient}
    if boxplot_kwargs:
        box_params.update(boxplot_kwargs)

    # plot logic for single vs multiple columns
    if isinstance(data, pd.DataFrame) and {'variable', 'value'}.issubset(data.columns):
        if orient=='v':
            sns.boxplot(x='variable', y='value', data=data, **box_params)
        else:
            sns.boxplot(y='variable', x='value', data=data, **box_params)
    else:
        if orient=='v':
            sns.boxplot(y=data, **box_params)
        else:
            sns.boxplot(x=data, **box_params)

    plt.title(title or f'{col_name} distribution')
    plt.xlabel(xlabel or (col_name if orient=='h' else ""))
    plt.ylabel(ylabel or (col_name if orient=='v' else ""))
    plt.tight_layout()  # FIX: layout
    plt.show()