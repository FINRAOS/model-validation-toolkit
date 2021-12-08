import itertools
import numpy

from mvtk import metrics


def test_rank_auc():
    nprng = numpy.random.RandomState(0)
    S = 32
    y_true, y_pred = nprng.randint(0, 5, S), nprng.uniform(size=S).round(1)
    N = 0
    auc = 0
    for (true1, pred1), (true2, pred2) in itertools.product(
        zip(y_true, y_pred), repeat=2
    ):
        if true1 > true2:
            if pred1 == pred2:
                auc += 0.5
            else:
                auc += pred1 > pred2
            N += 1
    auc /= N
    assert metrics.rank_auc(y_true, y_pred) == auc


def test_monotonicity():
    nprng = numpy.random.RandomState(0)
    S = 32
    y_true, y_pred = nprng.randint(0, 5, S), nprng.uniform(size=S).round(1)
    N = 0
    auc = 0
    for (true1, pred1), (true2, pred2) in itertools.product(
        zip(y_true, y_pred), repeat=2
    ):
        if true1 - true2 == 1:
            if pred1 == pred2:
                auc += 0.5
            else:
                auc += pred1 > pred2
            N += 1
    auc /= N
    assert metrics.monotonicity(y_true, y_pred) == auc


def weighted_roc_auc(y_test, y_pred, weights):
    def process(stuff):
        (pos, w_p), (neg, w_n) = stuff
        p = w_p * w_n
        return p * (0.5 if pos == neg else pos > neg), p

    mask = y_test == 1
    positives, w_pos = y_pred[mask], weights[mask]
    negatives, w_neg = y_pred[~mask], weights[~mask]
    numerator, denominator = map(
        sum,
        zip(
            *map(
                process, itertools.product(zip(positives, w_pos), zip(negatives, w_neg))
            )
        ),
    )

    return numerator / denominator


def test_weighted_roc_auc():
    nprng = numpy.random.RandomState(0)
    S = 32
    y_true, y_pred, weights = (
        nprng.randint(0, 2, S),
        nprng.uniform(size=S).round(1),
        nprng.uniform(size=S),
    )
    assert (
        abs(
            weighted_roc_auc(y_true, y_pred, weights)
            - metrics.rank_auc(y_true, y_pred, weights)
        )
        < 2 ** -32
    )
