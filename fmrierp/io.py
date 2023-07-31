import numpy as np
import pandas as pd


def extract_onsets_from_events(
    df: pd.DataFrame, trial_type: str, target_col: str = "trial_type"
) -> np.array:
    """Extracts onsets from the "trial_type" column of an important events.tsv.

    Parameters
    ----------
    df : pd.DataFrame
        Imported events.tsv.
    trial_type : str
        An event in the trial_type columns.
    target_col : str
        If a column other than trial_type should be chosen for selection.

    Returns
    -------
    np.array
        Array of onsets.

    Raises
    ------
    ValueError
        Error if "trial_type" does not match any of the events.
    """

    onsets = df.query(f"{target_col} == @trial_type")["onset"]

    onsets = onsets.values

    if len(onsets) == 0:
        raise ValueError(
            f"There are 0 events of {trial_type} in the events.tsv. Please check inputs."
        )

    return onsets
