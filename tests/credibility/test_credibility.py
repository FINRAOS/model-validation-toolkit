import numpy
import itertools
import pandas

from mvtk import credibility


def test_value_error():
    try:
        credibility.credible_interval(0, 0, prior=(0, 0))
    except ValueError:
        return
    raise Exception("Expected ValueError")


def test_equivalence():
    assert credibility.credible_interval(0, 0) == credibility.credible_interval(
        1, 1, prior=(0, 0)
    )


def test_prob_greater_cmp():
    nprng = numpy.random.RandomState(0)
    prior_sample_size = 10**6
    for N in range(2, 8):
        for prior1, prior2 in itertools.product(
            itertools.product(range(1, 3), repeat=2), repeat=2
        ):
            df = pandas.DataFrame()
            p1 = nprng.beta(*prior1, size=prior_sample_size)
            df["positives1"] = nprng.binomial(N, p1)
            p2 = nprng.beta(*prior2, size=prior_sample_size)
            df["positives2"] = nprng.binomial(N, p2)
            df["target"] = p1 > p2
            for (p1, p2), subset in df.groupby(["positives1", "positives2"]):
                p = subset["target"].mean()
                q = credibility.prob_greater_cmp(
                    p1, N - p1, p2, N - p2, prior1=prior1, prior2=prior2, err=10**-5
                )
                assert abs(q - p) < 0.05
