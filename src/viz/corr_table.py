import logging
import pandas as pd

logging.basicConfig(level=logging.ERROR)

'''For Multivariate analysis'''

def corrTable(
        df: pd.DataFrame = None,
        method: str = 'pearson',
        drop_boolean: bool = True,
        cmap: str = 'coolwarm'
) -> pd.DataFrame:
    """
    Compute and display a correlation table for numeric columns in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame, default=None
        The input DataFrame. Must contain at least one numeric column after filtering.
    method : {'pearson', 'spearman'}, default='pearson'
        The correlation method to use:
        - 'pearson': Standard correlation coefficient.
        - 'spearman': Rank correlation coefficient.
    drop_boolean : bool, default=True
        Whether to drop boolean columns before computing the correlation matrix.
    cmap : str, default='coolwarm'
        Colormap used for the background gradient in the styled output.

    Returns
    -------
    pandas.DataFrame
        The correlation matrix as a DataFrame.

    Raises
    ------
    ValueError
        If `df` is None, if there are no numeric columns after filtering,
        or if the correlation method is invalid.
    TypeError
        If `method` is not a string.

    Notes
    -----
    - The function displays a styled correlation matrix using `pandas.DataFrame.style`.
    - Boolean columns are excluded from correlation calculations if `drop_boolean=True`.
    """

    if df is None:
        raise ValueError("df cannot be None.")

    # Drop boolean columns if requested
    if drop_boolean:
        df = df.drop(columns=df.select_dtypes(include=["bool"]).columns, errors='ignore')

    # Keep only numeric columns
    cols = df.select_dtypes(include=['number'])

    # Shape check
    if cols.shape[1] == 0:
        logging.error("DataFrame has no numeric columns.")
        raise ValueError("DataFrame has no numeric columns.")

    # Method check
    if not isinstance(method, str):
        raise TypeError("Invalid type for method.")

    method = method.lower()
    if method not in ['pearson', 'spearman']:
        logging.error(f"{method} is invalid. Choose from 'pearson' or 'spearman'.")
        raise ValueError(f"Invalid method: {method}")

    corr = cols.corr(method=method)

    # Display styled correlation if available
    try:
        from IPython.display import display
        display(corr.style.background_gradient(cmap=cmap))
    except Exception:
        pass

    return corr
