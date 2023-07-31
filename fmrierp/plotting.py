from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def plot_erps(
    df: pd.DataFrame,
    event_names: List[str],
    cols: List,
    cmap: str = "Set2",
    plot_traces: bool = True,
):
    from matplotlib import colormaps

    cmap = colormaps.get_cmap(cmap)

    fig, axes = plt.subplots(1, 1)

    mean_response = df.groupby("trial_type").mean().reset_index()

    for n, ev in enumerate(event_names):
        axes.plot(
            mean_response.query("trial_type == @ev")[cols].values.T, color=cmap(n)
        )

    axes.legend(event_names)

    if plot_traces:
        for n, ev in enumerate(event_names):
            tmpdf = df.query("trial_type == @ev")[cols]
            for jj in tmpdf.iterrows():
                axes.plot(jj[1].T, color=cmap(n), alpha=0.05)

    return fig, axes
