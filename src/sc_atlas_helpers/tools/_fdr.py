import pandas as pd
import statsmodels.stats.multitest
from typing import Optional


def fdr_correction(
    df: pd.DataFrame,
    pvalue_col: str = "pvalue",
    key_added: str = "fdr",
    inplace: bool = False,
) -> pd.DataFrame:
    """Adjust p-values in a data frame with test results using FDR correction.

    Parameters
    ----------
    df
        The input DataFrame containing the test results.
    pvalue_col
        The column name containing the p-values. Default is "pvalue".
    key_added
        The column name to store the FDR-adjusted p-values. Default is "fdr".
    inplace
        If True, modifies the input DataFrame in place. If False, returns a new DataFrame. Default is False.
    """
    if not inplace:
        df = df.copy()

    # Assuming that fdrcorrection returns a tuple (array of adjusted p-values, rejected)
    adjusted_pvalues = statsmodels.stats.multitest.fdrcorrection(df[pvalue_col].values)[1]
    df[key_added] = adjusted_pvalues

    if not inplace:
        return df