from typing import Union, Sequence, Optional, Mapping, Literal, Tuple, Any
from matplotlib.colors import Colormap
import logging
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.ERROR)

# # Pretty section headers (optional helper, does not change loop logic)
# def _section_header(title: str, level: int = 1):
#     bar = ('=' if level == 1 else '-') * max(10, len(title) + (8 if level == 1 else 6))
#     prefix = '===' if level == 1 else '--'
#     print(f"\n{bar}\n{prefix} {title} {prefix}\n{bar}")


'''For Bivariate analysis'''

DataVar = Union[str, Sequence[str], pd.Series, np.ndarray]

def bar(
        df: pd.DataFrame,
        x: Optional[DataVar] = None,
        y: Optional[DataVar] = None,
        min_inns: int = None,
        min_mat: int = None,
        orient: str = 'v',
        hue: Optional[DataVar] = None,
        palette: Optional[Union[str, Sequence[str], Mapping[Any, Any]]] = None,
        order: Optional[Sequence] = None,
        hue_order: Optional[Sequence] = None,
        figsize: Tuple[float, float] = (10,6),
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        annotate: bool = True,
        fig_kwargs: Optional[Mapping[str, Any]] = None,
        barplot_kwargs: Optional[Mapping[str, Any]] = None
) -> plt.Axes:
    """
    Fully customizable barplot wrapper for sns.barplot with optional annotations.

    Parameters
    ----------
    df : DataFrame
        Input data.
    x, y : str, Series, or sequence
        Variables for barplot axes. Interpretation depends on `orient`.
    min_inns : int
        Variable to set the threshold for innings.
    min_mat : int
        Variable to set the threshold for matches.
    orient : {'v', 'h'}
        Plot orientation: 'v' = vertical bars, 'h' = horizontal bars.
    hue : str, Series, or sequence, optional
        Grouping variable for bars.
    palette : str, sequence, or mapping, optional
        Colors for different groups.
    order, hue_order : sequence, optional
        Order of categories.
    figsize : tuple, default=(10, 6)
        Figure size.
    title, xlabel, ylabel : str, optional
        Plot labels.
    annotate : bool, default=True
        If True, annotate each bar with its height/value.
    fig_kwargs : dict, optional
        Extra kwargs for plt.figure.
    barplot_kwargs : dict, optional
        Extra kwargs for sns.barplot.

    Returns
    -------
    matplotlib.axes.Axes
    """

    # Validation of data and axis
    try:
        if df is None:
            logging.error("Missing required parameter 'df'.")
            raise ValueError("df is None")  # raise error
        elif df.shape[0] == 0:
            logging.exception("Data can't be empty")
            raise ValueError("df has no rows")  # raise error
        elif x is None:
            if not df.shape >=2:
                logging.exception(f'{df} must have 2 or more columns.')
                raise ValueError("df must have >=2 columns")  # raise error
            # avoid df and df[-1]; choose first/last column names robustly
            first = df.columns
            last = df.columns[-1]
            x = df[first].dropna()
            if y is None:
                y = df[last].dropna()
            else:
                if isinstance(y, str):
                    y = df[y].dropna()
        else:
            # if x is str, use column; if Series/array, leave as-is
            if isinstance(x, str):
                if x not in df.columns:
                    raise KeyError(f"Column '{x}' not in df")
                x = df[x].dropna()
            if y is None:
                last = df.columns[-1]
                y = df[last].dropna()
            else:
                if isinstance(y, str):
                    if y not in df.columns:
                        raise KeyError(f"Column '{y}' not in df")
                    y = df[y].dropna()
    except Exception as e:
        logging.exception('Unexpected error during validation of data.')
        raise

    # Apply threshold and update values
    # filter a copy, do not attempt to index by Series as column names
    d = df.copy()
    if min_inns is not None and 'Inns' in d.columns:
        d = d[d['Inns'] >= min_inns]
    if min_mat is not None and 'Mat' in d.columns:
        d = d[d['Mat'] >= min_mat]
    # Note: x and y passed to seaborn can be names or arrays; keep consistency:
    # If x/y were strings above, they were replaced with Series; to keep seaborn happy with data=d,
    # convert them back to column names if applicable.
    x_arg = x if not isinstance(x, pd.Series) else x.name if x.name in d.columns else x
    y_arg = y if not isinstance(y, pd.Series) else y.name if y.name in d.columns else y
    
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
    fig_params = {"figsize": figsize}
    if fig_kwargs:
        fig_params.update(fig_kwargs)

    plt.figure(**fig_params)

    # Merge user barplot kwargs
    bar_params = {
        "data": d,
        "x": x_arg,
        "y": y_arg,
        "hue": hue,
        "palette": palette,
        "order": order,
        "hue_order": hue_order,
        "orient": orient
    }
    if barplot_kwargs:
        bar_params.update(barplot_kwargs)

    ax = sns.barplot(**bar_params)

    # Annotate bars
    if annotate:
        for p in ax.patches:
            if orient == 'v':
                height = p.get_height()
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom',
                    xytext=(0,3),
                    textcoords="offset points"
                )
            else:  # horizontal
                width = p.get_width()
                ax.annotate(
                    f"{width:.2f}",
                    (width, p.get_y() + p.get_height() / 2.),
                    ha='left', va='center',
                    xytext=(3,0),
                    textcoords="offset points"
                )

    # Labels
    # titles/labels must be strings, not Series
    x_label = x_arg if isinstance(x_arg, str) else (x_arg.name if isinstance(x_arg, pd.Series) and x_arg.name else "x")
    y_label = y_arg if isinstance(y_arg, str) else (y_arg.name if isinstance(y_arg, pd.Series) and y_arg.name else "y")
    ax.set_title(title or f"{y_label} by {x_label}" if orient=='h' else f"{x_label} by {y_label}")
    ax.set_xlabel(xlabel or (x_label if orient=="h" else y_label))
    ax.set_ylabel(ylabel or (y_label if orient=="h" else x_label))
    plt.tight_layout()
    plt.show()

    return ax