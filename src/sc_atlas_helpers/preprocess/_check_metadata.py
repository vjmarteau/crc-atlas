from typing import Dict, List, Optional

import warnings
import pandas as pd
import anndata


def validate_obs(
    adata_obs: pd.DataFrame,
    ref_meta_dict: Dict[str, Dict[str, List]],
    keys_to_ignore: Optional[List[str]] = None,
) -> None:
    """
    Validates the metadata information in the `.obs` attribute of an AnnData object or a Pandas DataFrame against a
    reference metadata dictionary.

    Parameters
    ----------
    adata_obs : pd.DataFrame
        Pandas DataFrame or `.obs` attribute of an AnnData object to be validated.
    ref_meta_dict : dict
        Reference metadata dictionary.
    keys_to_ignore : list[str], optional
        List of keys to ignore during validation. Defaults to None.

    Raises
    ------
    ValueError: If missing columns or invalid values are found in the metadata.

    Returns
    -------
    None

    Note:
    -----
    This function validates the metadata information in the `.obs` attribute of an AnnData object or a Pandas DataFrame
    against a reference metadata dictionary `ref_meta_dict`. The `ref_meta_dict` should be a dictionary where the keys
    represent the metadata columns to be validated, and the values are dictionaries with a 'values' key containing a list
    of allowed values for that metadata column. The function raises a `ValueError` if any missing columns or invalid values
    are found in the metadata columns of the AnnData object. The `keys_to_ignore` parameter can be used to specify a
    list of keys that should be ignored during validation. If missing columns are found, the error message will include
    the missing columns in the order of the input dictionary. If invalid values are found, the error message will
    include the invalid values for the corresponding column.
    """
    if keys_to_ignore is None:
        keys_to_ignore = []

    # Check if all expected keys are present in adata_obs.columns
    expected_cols = [k for k in ref_meta_dict.keys() if k not in keys_to_ignore]
    missing_cols = [c for c in expected_cols if c not in adata_obs.columns]
    if missing_cols:
        missing_cols_str = ", ".join(
            missing_col for missing_col in expected_cols if missing_col in missing_cols
        )
        raise ValueError(f"Missing columns: {missing_cols_str}")

    # Check if keys are present as columns and verify values if present (except keys_to_ignore)
    for key, value in ref_meta_dict.items():
        if key in keys_to_ignore:
            continue  # Skip keys to ignore

        # Raise a warning if the column contains only NaN values
        if adata_obs[key].dropna().empty:
            warnings.warn(f"Column '{key}' contains only missing values")

        if key not in adata_obs.columns:
            raise ValueError(f"Missing columns: {key}")

        # Verify type of corresponding column
        column_type = adata_obs[key].dtype
        expected_type = value.get("type", None)

        if expected_type is not None and column_type != expected_type:
            offending_value = adata_obs[key][
                adata_obs[key].apply(lambda x: type(x) != expected_type)
            ].iloc[0]
            raise ValueError(
                f"Unexpected data type found in column '{key}'. Expected '{expected_type}', but found '{offending_value}'."
            )

        # Verify values in corresponding column
        allowed_values = value.get("values", None)
        column_values = adata_obs[key].unique()

        if "min" in value and "max" in value:
            min_value = value["min"]
            max_value = value["max"]
            invalid_values = [
                val for val in column_values if not (min_value <= val <= max_value)
            ]
            invalid_values = [
                val for val in invalid_values if not (pd.isna(val) or val == "nan")
            ]  # Add check for non-NA values
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in column '{key}': {invalid_values}"
                )
        elif allowed_values is not None:
            invalid_values = [val for val in column_values if val not in allowed_values]
            invalid_values = [
                val for val in invalid_values if not (pd.isna(val) or val == "nan")
            ]  # Add check for non-NA values
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in column '{key}': {invalid_values}"
                )

        # Verify values in corresponding column
        allowed_values = value.get("values", None)
        column_values = adata_obs[key].unique()

        if "min" in value and "max" in value:
            min_value = value["min"]
            max_value = value["max"]
            invalid_values = [
                val for val in column_values if not (min_value <= val <= max_value)
            ]
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in column '{key}': {invalid_values}"
                )
        elif allowed_values is not None:
            invalid_values = [
                val
                for val in column_values
                if val not in allowed_values and not (pd.isna(val) or val == "nan")
            ]
            if invalid_values:
                raise ValueError(
                    f"Invalid values found in column '{key}': {invalid_values}"
                )


def search_dict(
    my_dict: dict, columns: List[str], search: Optional[List[str]] = None
) -> dict:
    """
    Searches a nested dictionary for specified keys in each of the columns.
    """
    values = {}
    for column in columns:
        if column in my_dict:
            column_values = {}
            for key, value in my_dict[column].items():
                if not search or key in search:
                    if key in ["values", "description", "type"]:
                        column_values[key] = value
            values[column] = column_values
    return values


def tnm_tumor_stage(adata_obs: pd.DataFrame, group: str) -> pd.Series:
    """
    Add a new "tumor_stage_TNM" column to the input DataFrame based on the T, N, and M
    stages, using the criteria defined by the American Joint Committee on Cancer.
    See: https://www.cancer.net/cancer-types/colorectal-cancer/stages#:~:text=T4a%3A%20The%20tumor%20has%20grown,to%20other%20organs%20or%20structures.
    """
    df = adata_obs.groupby(group).first()
    # create a new empty list to store the stage values for each row
    stage = []

    # loop through each row in the DataFrame
    for i, row in df.iterrows():
        # get the values of the tumor, node, and metastasis columns for the current row
        tumor = row["tumor_stage_TNM_T"]
        node = row["tumor_stage_TNM_N"]
        metastasis = row["tumor_stage_TNM_M"]

        # consider M0 and MX as equivalent
        if metastasis == "Mx":
            metastasis = "M0"

        # consider N1, N2, M1, and T4 to be equivalent to their counterparts with an appended "a"
        if node == "N1":
            node = "N1a"
        elif node == "N2":
            node = "N2a"
        if metastasis == "M1":
            metastasis = "M1a"
        elif tumor == "T4":
            tumor = "T4a"

        # apply the stage criteria and append the corresponding stage to the stage list
        if (
            (tumor in ["T1", "T1a", "T1b", "T1c", "T2", "T2a", "T2b", "T2c"])
            and (node == "N0")
            and (metastasis == "M0")
        ):
            stage.append("I")
        elif (tumor == "T3") and (node == "N0") and (metastasis == "M0"):
            stage.append("IIA")
        elif (tumor == "T4a") and (node == "N0") and (metastasis == "M0"):
            stage.append("IIB")
        elif (tumor == "T4b") and (node == "N0") and (metastasis == "M0"):
            stage.append("IIC")
        elif (
            (tumor in ["T1", "T1a", "T1b", "T1c", "T2", "T2a", "T2b", "T2c"])
            and (node in ["N1a", "N1b", "N1c"])
            and (metastasis == "M0")
        ) or (
            (tumor in ["T1", "T1a", "T1b", "T1c"])
            and (node == "N2a")
            and (metastasis == "M0")
        ):
            stage.append("IIIA")
        elif (
            (
                (tumor in ["T3", "T4a"])
                and (node in ["N1", "N1a", "N1b", "N1c"])
                and (metastasis == "M0")
            )
            or ((tumor in ["T2", "T3"]) and (node == "N2a") and (metastasis == "M0"))
            or ((tumor in ["T1", "T2"]) and (node == "N2b") and (metastasis == "M0"))
        ):
            stage.append("IIIB")
        elif (
            ((tumor == "T4a") and (node == "N2a") and (metastasis == "M0"))
            or ((tumor in ["T3", "T4a"]) and (node == "N2b") and (metastasis == "M0"))
            or (
                (tumor == "T4b")
                and (node in ["N1", "N1a", "N1b", "N1c", "N2", "N2a", "N2b"])
                and (metastasis == "M0")
            )
        ):
            stage.append("IIIC")
        elif (tumor != "Tx") and (node != "Nx") and (metastasis == "M1a"):
            stage.append("IVA")
        elif (tumor != "Tx") and (node != "Nx") and (metastasis == "M1b"):
            stage.append("IVB")
        elif (tumor != "Tx") and (node != "Nx") and (metastasis == "M1c"):
            stage.append("IVC")
        else:
            stage.append("nan")

    # add the stage list as a new "stage" column to the DataFrame
    df = pd.concat([df, pd.Series(stage, index=df.index, name="tumor_stage_TNM")], axis=1)
    return adata_obs[group].map(df["tumor_stage_TNM"].to_dict())


def add_matched_samples_column(adata_obs: pd.DataFrame, group: str) -> pd.Series:
    """
    Looks if values "tumor" and "normal" are present in the column "sample_type" per patient to return yes/no for matched_samples
    """
    # Group the DataFrame by patient_id and check if the patient has both "tumor" and "normal" in their sample_type column
    matched_samples = adata_obs.groupby(group)["sample_type"].apply(lambda x: "yes" if set(x) == set(["tumor", "normal"]) else "no")
    # Create a dictionary from the resulting boolean series
    matched_samples_dict = matched_samples.to_dict()
    # Map the dictionary onto the DataFrame and return the updated DataFrame
    return adata_obs[group].map(matched_samples_dict)
