import public
import numpy
import pandas

from scipy.stats import entropy
from sklearn.feature_selection import mutual_info_classif


def binarize(data, t):
    y_true, y_pred = data.values.T
    return y_true > t, y_pred


@public.add
def monotonicity(y_true, y_pred, weights=None):
    r"""Generalizes ROC AUC by computing
    :math:`P\left(\frac{\Delta\mathrm{y_pred}}{\Delta\mathrm{y_true}} >
    0\right)`, the probability incrementing ``y_true`` increases ``y_pred`` for
    a randomly chosen pair of instances. This reduces to ROC AUC when
    ``y_true`` has two unique values. Adapted from Algorithm 2 in `Fawcett, T.
    (2006). An introduction to ROC analysis. Pattern Recognition Letters,
    27(8), 861-874.
    <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_

    Args:
        y_true (list-like): Ground truth ordinal values
        y_pred (list-like): Predicted ordinal values
        weights (list-like): Sample weights. Will be normalized to one
            across each unique values of ``y_true``. If ``None`` (default) all
            samples are weighed equally.

    Returns:
       Float between 0 and 1. 0 indicates 100% chance of ``y_pred``
       decreasing upon incrementing ``y_true`` up to its next
       highest value in the dataset. 1 being a 100% chance of
       ``y_pred`` increasing for the same scenario. 0.5 would be 50%
       chance of either.
    """
    if weights is None:
        weights = numpy.ones(len(y_true))
    unique = numpy.unique(y_true)
    n = len(unique) - 1
    true_lookup = {u: i + 1 for i, u in enumerate(unique)}
    idx = numpy.argsort(-y_pred)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    weights = weights[idx]
    # fp, fp_prev, tp, tp_prev, auc
    data = numpy.zeros((5, n))
    prev_pred = numpy.full(n, numpy.nan)
    for true, pred, weight in zip(y_true, y_pred, weights):
        i = true_lookup[true]
        j = max(i - 2, 0)
        mask = pred != prev_pred[j:i]
        data[4, j:i][mask] += trap(*data[:4, j:i][:, mask])
        data[1:4:2, j:i][:, mask] = data[:4:2, j:i][:, mask]
        prev_pred[j:i] = pred
        i -= 1
        if i:
            data[2, j] += weight
        if i < n:
            data[0, i] += weight
    data[4] += trap(*data[:4])
    return numpy.sum(data[4]) / 2 / data[0].dot(data[2])


def trap(x2, x1, y2, y1):
    return (x2 - x1) * (y2 + y1)


@public.add
def rank_auc(y_true, y_pred, weights=None):
    r"""Generalizes ROC AUC by computing probability that two randomly chosen
    data points would be ranked consistently with ground truth labels. This
    reduces to ROC AUC when ``y_true`` has two unique values.
    Adapted from Algorithm 2 in `Fawcett, T. (2006). An introduction
    to ROC analysis. Pattern Recognition Letters, 27(8), 861-874.
    <https://www.sciencedirect.com/science/article/pii/S016786550500303X>`_

    Args:
        y_true (list-like): Ground truth ordinal values
        y_pred (list-like): Predicted ordinal values
        weights (list-like): Sample weights. Will be normalized to one
            across each unique values of ``y_true``. If ``None`` (default) all
            samples are weighed equally.

    Returns:
       Float between 0 and 1. 0 indicates 100% chance of ``y_pred``
       matching order of ``y_true``. 1 being a 100% chance of
       ``y_pred`` having the opposite order of ``y_true``. 0.5 would be 50%
       chance of either.
    """
    if weights is None:
        weights = numpy.ones(len(y_true))
    unique = numpy.unique(y_true)
    n = len(unique) - 1
    true_lookup = {u: i + 1 for i, u in enumerate(unique)}
    idx = numpy.argsort(-y_pred)
    y_true = y_true[idx]
    y_pred = y_pred[idx]
    weights = weights[idx]
    # fp, fp_prev, tp, tp_prev, auc
    data = numpy.zeros((5, n))
    prev_pred = numpy.full(n, numpy.nan)
    for true, pred, weight in zip(y_true, y_pred, weights):
        i = true_lookup[true]
        mask = pred != prev_pred[:i]
        data[4, :i][mask] += trap(*data[:4, :i][:, mask])
        data[1:4:2, :i][:, mask] = data[:4:2, :i][:, mask]
        prev_pred[:i] = pred
        i -= 1
        data[2, :i] += weight
        if i < n:
            data[0, i] += weight
    data[4] += trap(*data[:4])
    return numpy.sum(data[4]) / 2 / data[0].dot(data[2])


@public.add
def normalized_mutual_info(X, y, **kwargs):
    """Thin wrapper around `sklearn's mutual information
    <https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html>`_.
    This normalizes the result to 0-1 scale. ``y`` is assumed categorical.
    """
    _, counts = numpy.unique(y, return_counts=True)
    return pandas.Series(
        dict(
            zip(
                X.columns,
                mutual_info_classif(X, y, **kwargs) / entropy(counts / counts.sum()),
            )
        )
    )
