import os

import numpy as np
import pandas as pd

from .erp import (
    create_extraction_window,
    create_nifti_masker,
    extract_timeseries,
    extract_windows,
    round_onsets,
)
from .io import extract_onsets_from_events
from .plotting import plot_erps


def extract_erps(
    events, data, event_names, t_r, target_col="trial_type", window=[0, 15]
):

    window = create_extraction_window(window, t_r)
    t = (np.arange(window[0], window[1] + 1)) * t_r
    col_names = [f"{i:4.2f}" for i in t]

    window_dfs = []

    for ev in event_names:
        onsets = extract_onsets_from_events(events, ev, target_col)
        onsets = round_onsets(onsets, t_r)
        windowed_data = extract_windows(data, onsets, window)[0]
        tmp_df = pd.DataFrame(windowed_data, columns=col_names)
        tmp_df["trial_type"] = ev
        window_dfs.append(tmp_df)

    window_dfs = pd.concat(window_dfs).reset_index()

    return window_dfs, col_names


def get_fmri_data(
    fmri,
    mask,
    t_r,
    smoothing_fwhm: float = None,
    standardize: str = "psc",
    detrend: bool = True,
    high_pass: float = None,
):

    # TODO: Some safetys around loading and mask dimensions (at least resampling the mask).
    nifti_masker = create_nifti_masker(
        mask,
        t_r,
        smoothing_fwhm=smoothing_fwhm,
        standardize=standardize,
        detrend=detrend,
        high_pass=high_pass,
    )

    extracted = extract_timeseries(fmri, nifti_masker)

    return extracted


def simple_workflow(
    event_file,
    fmri_file,
    mask_file,
    out_dir,
    event_names,
    t_r,
    mask_name="",
    window=[0, 15],
    target_col="trial_type",
    smoothing_fwhm: float = None,
    standardize: str = "psc",
    detrend: bool = True,
    high_pass: float = None,
):

    events = pd.read_csv(event_file, sep="\t")
    fmri = get_fmri_data(
        fmri_file,
        mask_file,
        t_r=t_r,
        smoothing_fwhm=smoothing_fwhm,
        standardize=standardize,
        detrend=detrend,
        high_pass=high_pass,
    )

    df, col_names = extract_erps(
        events,
        fmri,
        event_names=event_names,
        target_col=target_col,
        window=window,
        t_r=t_r,
    )

    df.to_csv(os.path.join(out_dir, f"windows_{mask_name}.tsv"), index=None, sep="\t")

    fig, _ = plot_erps(df, event_names=event_names, cols=col_names)

    fig.savefig(os.path.join(out_dir, f"erps_{mask_name}.pdf"), dpi=600)
