from typing import List

import numpy as np
import pandas as pd
from nibabel.nifti1 import Nifti1Image


def round_onsets(onsets: np.array, t_r: float) -> np.array:
    """Rounds onsets to TR indices and transforms them to int.

    Parameters
    ----------
    onsets : np.array
        array of event onsets.
    t_r : float
        repetition time of the sequence

    Returns
    -------
    np.array
        array of onsets as TR index.
    """

    onsets = onsets.copy() / t_r
    onsets = np.round(onsets).astype(int)

    return onsets


def create_nifti_masker(
    mask_img: Nifti1Image,
    t_r: float,
    smoothing_fwhm: float = None,
    standardize: str = "psc",
    detrend: bool = True,
    high_pass: float = None,
):
    """Creates a nifti masker to extract fmri data using a mask.

    Parameters
    ----------
    mask_img : Nifti1Image
        _description_
    t_r : float
        _description_
    smoothing_fwhm : float, optional
        _description_, by default None
    standardize : str, optional
        _description_, by default 'psc'
    detrend : bool, optional
        _description_, by default True
    high_pass : float, optional
        _description_, by default None

    Returns
    -------
    _type_
        _description_
    """
    from nilearn.maskers import NiftiMasker

    masker = NiftiMasker(
        mask_img,
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm,
        standardize=standardize,
        detrend=detrend,
        high_pass=high_pass,
    )

    return masker


def extract_timeseries(fmri_data, masker):
    data = masker.fit_transform(fmri_data)

    data = data.mean(1)
    return data


def create_extraction_window(window: List, t_r: float) -> np.array:
    """Converts the input window, defined in s, to indices.

    Parameters
    ----------
    window : List
        window in seconds
    t_r : float
        Repetition time of signal

    Returns
    -------
    np.array
        Rounded window

    Raises
    ------
    ValueError
        Checks if window has the correct size
    """

    if len(window) != 2:
        raise ValueError("window has to be a list of length 2")

    window = np.array(window)
    window = window / t_r  # TR as sampling rate
    window = window.astype(int)
    return window


def extract_windows(data, onsets, window):

    window_array = list()
    rejected = []
    for ons in onsets:
        # Check if the window covers a full peak.
        if (ons - window[0]) < 0 or (ons + window[1] + 1) > len(data):
            rejected.append(ons)
        else:
            window_array.append(data[ons + window[0] : ons + window[1] + 1])

    window_array = np.array(window_array)

    print(f"Onsets, with no full window {rejected}")
    return window_array, rejected
