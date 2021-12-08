import multiprocessing
import sys
import time
import pandas as pd
import numpy as np
import public

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Collection
from typing import List
from itertools import combinations
from fastcore.imports import in_notebook


if in_notebook():
    from tqdm import tqdm_notebook as tqdm
else:
    from tqdm import tqdm


@public.add
def parallel(
    func, arr: Collection, max_workers: int = None, show_progress: bool = True
):
    """
    NOTE: This code was adapted from the ``parallel`` function
        within Fastai's Fastcore library. Key differences include
        returning a list with order preserved.

    Run a function on a collection (list, set etc) of items
    :param func: The function to run
    :param arr: The collection to run on
    :param max_workers: How many workers to use. Will use
        multiprocessing.cpu_count() if this is not provided
    :return: a list of the results
    """
    if show_progress:
        progress_bar = tqdm(arr, smoothing=0, file=sys.stdout)
    results = []
    max_workers = multiprocessing.cpu_count() if max_workers is None else max_workers
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_to_index = {ex.submit(func, o): i for i, o in enumerate(arr)}
        for future in as_completed(future_to_index):
            results.append((future_to_index[future], future.result()))
            if show_progress:
                progress_bar.update()
    results.sort()

    # Complete the progress bar if not complete
    if show_progress:
        for n in range(progress_bar.n, len(list(arr))):
            time.sleep(0.1)
            progress_bar.update()
    return [result for i, result in results]


@public.add
def column_indexes(df: pd.DataFrame, cols: List[str]):
    """

    :param df: The dataframe
    :param cols: a list of column names
    :return: The column indexes of the column names
    """
    return [df.columns.get_loc(col) for col in cols if col in df]


def format_date(date_str, dateformat="%b%d"):
    date = pd.to_datetime(date_str)
    return datetime.strftime(date, dateformat)


@public.add
def compute_divergence_crosstabs(
    data, datecol=None, format=None, show_progress=True, divergence=None
):
    """Compute the divergence crosstabs.

    :param data: The data to compute the divergences on
    :param datecol: The column representing the date. If None, will
        use the index, if the index is a datetimeindex
    :param format: A function applied to datecol values for formatting
        e.g. ``format_date``
    :param show_progress: Whether the progress bar will be shown
    :param divergence: The divergence function to use
    """
    if datecol is None:
        datecol = data.index
    dates, subsets = zip(*data.groupby(datecol))
    dates = list(dates)
    subsets = (subset.drop(columns=[datecol]) for subset in subsets)

    return compute_divergence_crosstabs_split(
        subsets, dates, format, show_progress, divergence
    )


@public.add
def compute_divergence_crosstabs_split(
    subsets, dates, format=None, show_progress=True, divergence=None
):
    """Compute the divergence crosstabs.

    :param subsets: The data to compute the divergences on
    :param dates: The list of dates for the subsets
    :param format: A function applied to datecol values for formatting
        e.g. ``format_date``
    :param show_progress: Whether the progress bar will be shown
    :param divergence: The divergence function to use
    """

    # Create a divergence matrix
    divergences = np.zeros((len(dates), len(dates)))
    if not divergence:
        from mvtk.supervisor.divergence import calc_tv

        divergence = calc_tv

    def compute_divergence(args):
        return divergence(*args)

    for (i, j), v in zip(
        combinations(range(len(dates)), 2),
        parallel(
            compute_divergence, combinations(subsets, 2), show_progress=show_progress
        ),
    ):
        divergences[i, j] = divergences[j, i] = v
    if format is None:
        formatted = dates
    else:
        formatted = [format(d) for d in dates]
    return pd.DataFrame(divergences, columns=formatted, index=formatted)


@public.add
def plot_divergence_crosstabs_3d(divergences):
    """Plot the divergences in 3d.

    :params divergences: The list of divergences
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    keys = list(divergences.keys())
    indexes = range(len(keys))

    for i in indexes:
        y = [x[1] for x in list(divergences[keys[i]].items())]
        ax.bar(indexes, y, i, zdir="y", alpha=0.8)

    ax.set(xticks=indexes, xticklabels=keys, yticks=indexes, yticklabels=keys)

    return fig


@public.add
def split(x, train_ratio=0.5, nprng=np.random.RandomState(0)):
    i = int(len(x) * train_ratio)
    if hasattr(x, "shape"):
        idx = np.arange(x.shape[0])
        nprng.shuffle(idx)
        x = x[idx]
    else:
        nprng.shuffle(x)
    return x[:i], x[i:]
