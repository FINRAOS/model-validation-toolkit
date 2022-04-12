import numpy
import mvtk.supervisor.divergence as divergence

from functools import partial


def mutually_exclusive_support_tester(metric, num_features=4, eps=0.1):
    data1 = numpy.ones((4, num_features))
    data1[:, :2] = 0
    data2 = 1 - data1
    assert numpy.isclose(metric([data1], [data1]), 0, atol=eps)
    assert numpy.isclose(metric([data2], [data2]), 0, atol=eps)
    assert numpy.isclose(metric([data1], [data2]), 1, atol=eps)


def get_batches(nprng, batch_size, n=2):
    """Pick a random binomial distribution Sample batch_size samples from
    it."""
    choices = numpy.arange(n)
    x = []
    alpha = nprng.rand(n)
    alpha /= alpha.sum()
    for d in range(batch_size):
        choice = nprng.choice(choices, p=alpha)
        z = numpy.zeros_like(choices)
        z[choice] = 1
        x.append(z)
    x = numpy.asarray(x).reshape(batch_size, n)
    return x, alpha


def divergence_tester(
    approximate_metric, analytical_metric, batch_sizes=[256] * 8, thresh=0.85
):
    nprng = numpy.random.RandomState(0)
    batches, alphas = zip(*map(partial(get_batches, nprng), batch_sizes))
    assert (
        numpy.corrcoef(
            numpy.asarray([analytical_metric(alphas, alpha) for alpha in alphas]),
            divergence.utils.get_drift_series(
                approximate_metric, batches, [[batch] for batch in batches]
            ),
        )[0, 1]
        > thresh
    )


def gaussian_test(approximate_metric, dim=1, N=1024, thresh=0.05):
    nprng = numpy.random.RandomState(0)
    m = approximate_metric(*nprng.normal(size=(2, 1, N, dim)))
    assert m < thresh
    assert m >= 0


def test_hl_gaussian():
    for dim in range(1, 4):
        gaussian_test(partial(divergence.calc_hl, train_test_split=0.5), dim)


def test_tv_gaussian():
    for dim in range(1, 4):
        gaussian_test(partial(divergence.calc_tv, train_test_split=0.5), dim)
        gaussian_test(
            partial(divergence.calc_tv_knn, k=64 * 2**dim),
            dim,
            N=1024 * 2**dim,
            thresh=0.1,
        )


def test_js_gaussian():
    for dim in range(1, 4):
        gaussian_test(partial(divergence.calc_js, train_test_split=0.5), dim)


def test_em_gaussian():
    for dim in range(1, 4):
        gaussian_test(
            partial(divergence.calc_em, train_test_split=0.5), dim, thresh=0.11
        )


def test_js_by_corr():
    def kl(alpha1, alpha2):
        return numpy.sum(alpha1 * numpy.log2(alpha1 / alpha2))

    def js(alpha1, alpha2):
        mean = alpha1 + alpha2
        mean /= 2
        ret = kl(alpha1, mean) + kl(alpha2, mean)
        return ret / 2

    divergence_tester(
        lambda *x: numpy.sqrt(divergence.calc_js_mle(*x)), lambda *x: numpy.sqrt(js(*x))
    )
    divergence_tester(
        lambda *x: numpy.sqrt(divergence.calc_js(*x)), lambda *x: numpy.sqrt(js(*x))
    )


def test_js_by_support():
    mutually_exclusive_support_tester(divergence.calc_js_mle)
    mutually_exclusive_support_tester(divergence.calc_js)


def test_hl_by_corr():
    def hl(alpha1, alpha2):
        return numpy.sqrt(numpy.sum((numpy.sqrt(alpha1) - numpy.sqrt(alpha2)) ** 2) / 2)

    divergence_tester(divergence.calc_hl_mle, hl)
    divergence_tester(divergence.calc_hl, hl)


def test_hl_by_support():
    mutually_exclusive_support_tester(divergence.calc_hl)
    mutually_exclusive_support_tester(divergence.calc_hl_mle)


def test_tv_by_corr():
    def tv(alpha1, alpha2):
        return numpy.abs(alpha1 - alpha2).sum() / 2

    divergence_tester(divergence.calc_tv_mle, tv)
    divergence_tester(divergence.calc_tv, tv)


def test_tv_by_support():
    mutually_exclusive_support_tester(divergence.calc_tv_mle)
    mutually_exclusive_support_tester(divergence.calc_tv)


def test_em_by_support():
    for num_features in range(1, 3):
        data1 = numpy.zeros((4, num_features))
        data2 = 1 - data1
        eps = 0.125
        assert numpy.isclose(divergence.calc_em([data1], [data1]), 0, atol=eps)
        assert numpy.isclose(divergence.calc_em([data2], [data2]), 0, atol=eps)
        assert numpy.isclose(divergence.calc_em([data1], [data2]), 1, atol=eps)
        assert numpy.isclose(divergence.calc_em([data1], [2 * data2]), 2, atol=eps)


def test_calc_tv_lower_bound():
    a = numpy.asarray([0, 1, 0, 0, 1])
    b = numpy.asarray([0.01, 0.98, 0.03, 0.04, 0.99])
    log_loss = divergence.metrics.balanced_binary_cross_entropy(a, b)
    tv = divergence.metrics.calc_tv_lower_bound(log_loss)
    assert tv < 1 and tv > 0
