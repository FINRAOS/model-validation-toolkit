__all__ = []

import itertools
import jax
import numpy
import scipy
import public

from .nn import Approximator, NormalizedLinear
from .generators import fdiv_data_stream, js_data_stream
from collections import Counter
from functools import partial
from jax.experimental import optimizers
from mvtk.supervisor.utils import parallel, split
from .utils import arrayify


@public.add
def calc_div_variational(data_stream, loss, model_generator=Approximator, summary=""):
    r"""Calculate an :math:`f`-divergence or integral probability metric using
    a variational of hybrid variational estimator.

    Variational estimates will generally (but, not always, thanks to the
    training proceedure!) be a lower bound on the true value

    Args:
        data_stream (generator): The data stream generator.
        loss (function): Loss function that takes as arguments the model
            outputs. Returns a scalar.
        model_generator: A function that takes a Jax ``PRNGKey`` the number
            of dimensions of the support and returns a `Jax model
            <https://jax.readthedocs.io/en/latest/jax.experimental.stax.html>`_ to
            be used for variational approximations. The function this model is
            trained to approximate is sometimes known as the *witness
            function*--especially when dealing with `integral probability metrics
            <http://www.gatsby.ucl.ac.uk/~gretton/papers/montreal19.pdf>`_.
            Specifically, the function returns a tuple that contains the initial
            parameters and a function that maps those parameters and the model
            inputs to the model outputs. Defaults to
            :meth:`supervisor.divergence.Approximator`.
        summary (string): Summary of divergence to appear in docstring
    Returns:
        function for computing divergence"""

    def calc_div(
        *sample_distributions,
        categorical_columns=tuple(),
        model_generator_kwargs={},
        loss_kwargs={},
        nprng=None,
        batch_size=16,
        num_batches=128,
        num_epochs=4,
        effective_sample_size=None,
        train_test_split=0.75,
        step_size=0.0125
    ):
        r"""

        Args:
            *sample_distributions (list): Sample distributions. A numpy array,
                pandas dataframe, or pandas series or a list of numpy arrays,
                dataframes or series. If it is a list then will sample from each in the
                list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
                [batch6, batch7]]`` Assuming ``batch1`` came from distribution
                :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
                simulate a system in which a latent `N=3` sided die role that
                determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
                p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
                The outer most list is typically a singleton.
            model_generator_kwargs (optional): Dictionary of optional kwargs to pass to
                model_generator. ``width`` and ``depth`` are useful. See
                :meth:`supervisor.divergence.Approximator` for more details.
            loss_kwargs (optional): Dictionary of optional kwargs to pass to
                loss function. ``weights`` is commonly used for reweighting
                expectations. See `hybrid estimation
                <user_guide.rst#hybrid-estimation>`__ for details.
            categorical_columns (optional): List of indices of columns which should
                be treated as categorical.
            nprng (optional): Numpy ``RandomState``
            batch_size (int): mini batch size. Defaults to 16.
            num_batches (int): number of batches per epoch. Defaults to 128.
            num_epochs (int): number of epochs to train for. Defaults to 4.
            effective_sample_size (optional): Size of subsample over which Epoch
                losses are computed. This determines how large a sample a divergence is
                computed over.
            train_test_split (optional): If not None, specifies the
                proportion of samples devoted to training as opposed to
                validation. If None, no split is used. Defaults to 0.75.
            step_size (float): step size for Adam optimizer

        Returns:
            Estimate of divergence."""
        if nprng is None:
            nprng = numpy.random.RandomState(1)
        sample_distributions = tuple(map(arrayify, sample_distributions))
        if train_test_split is None:
            training_samples = validation_samples = sample_distributions
        else:
            training_samples, validation_samples = zip(
                *(
                    zip(
                        *(
                            split(sample, train_ratio=train_test_split, nprng=nprng)
                            for sample in sample_distribution
                        )
                    )
                    for sample_distribution in sample_distributions
                )
            )
        mini_batches = data_stream(
            nprng, batch_size, training_samples, categorical_columns=categorical_columns
        )
        if effective_sample_size is None:
            effective_sample_size = num_batches * batch_size
        large_batches = data_stream(
            nprng,
            effective_sample_size,
            validation_samples,
            categorical_columns=categorical_columns,
        )
        input_size = next(next(mini_batches)[0].values().__iter__()).shape[1]
        init_params, approximator = model_generator(
            input_size, **model_generator_kwargs
        )
        key_to_index = (
            {
                key: index
                for index, key in enumerate(
                    get_density_estimators(categorical_columns, *sample_distributions)
                )
            }
            if categorical_columns
            else {tuple(): 0}
        )
        opt_init, opt_update, get_params = optimizers.adam(step_size)
        opt_state = opt_init({key: init_params for key in key_to_index})

        def _loss(params, batch):
            return loss(
                (
                    jax.numpy.vstack(
                        approximator(params[key], sample)
                        for key, sample in samples.items()
                    )
                    for samples in batch
                ),
                **loss_kwargs
            )

        @jax.jit
        def update(i, opt_state, batch):
            params = get_params(opt_state)
            return opt_update(i, jax.grad(_loss)(params, batch), opt_state)

        itercount = itertools.count()
        best_loss = numpy.inf
        for epoch in range(num_epochs):
            for _ in range(num_batches):
                opt_state = update(next(itercount), opt_state, next(mini_batches))
            params = get_params(opt_state)
            epoch_loss = _loss(params, next(large_batches))
            if numpy.isnan(epoch_loss):
                raise ValueError(
                    """The loss is NaN.
                Make sure floating point arithmetic makes sense for your data."""
                )
            if epoch_loss < best_loss:
                # track the best expected loss over all epochs this is where a
                # biased sample could push estimate of the expected loss higher
                # than the true :math:`f`-divergence I think the trade off for
                # fast convergence given a low epoch number is well worth it.
                best_loss = epoch_loss
        return numpy.clip(-best_loss, 0, numpy.inf)  # clip just above zero

    calc_div.__doc__ = summary + calc_div.__doc__
    return calc_div


@public.add
def fdiv_loss(convex_conjugate):
    """General template for :math:`f`-divergence losses given convex conjugate.

    Args:
        convex_conjugate: The convex conjugate of the function, :math:`f`.
    """

    def loss(batch, weights=(1, 1)):
        r"""Args:
        batch: pair of minibatches drawn from each sample
        weights: Provides an alternative means of reweighting minibatches.
            See `hybrid estimation
        <user_guide.rst#hybrid-estimation>`__ for details."""
        input1, input2 = batch
        batch_loss = (
            convex_conjugate(input2).mean() * weights[1] - input1.mean() * weights[0]
        )
        # print(batch_loss)
        return batch_loss

    return loss


@public.add
def ipm_loss(batch):
    """Integral probability metric loss."""
    input1, input2 = batch
    batch_loss = input2.mean() - input1.mean()
    return -jax.numpy.abs(batch_loss)


calc_tv = calc_div_variational(
    fdiv_data_stream,
    fdiv_loss(lambda x: x),
    model_generator=partial(Approximator, activation=lambda x: jax.numpy.tanh(x) / 2),
    summary=r"""Total variation - :math:`f`-divergence form

    :math:`\frac{1}{2}\int dx \vert p\left(x\right) - q\left(x\right) \vert =
    \sup\limits_{f : \|f\|_\infty \le \frac{1}{2}} \mathbb{E}_{x \sim
    p}\left[f(x)\right] - \mathbb{E}_{x^\prime \sim q}\left[f(x^\prime)\right]`

    https://arxiv.org/abs/1606.00709""",
)
__all__.append("calc_tv")

calc_em = calc_div_variational(
    fdiv_data_stream,
    ipm_loss,
    model_generator=partial(Approximator, linear=NormalizedLinear, residual=False),
    summary=r"""Wasserstein-1 (Earth Mover's) metric

    :math:`\int dxdx^\prime d(x, x^\prime)\gamma(x, x^\prime)`

    , with

    :math:`d(x, x^\prime)=\|x - x^\prime\|_1`

    subject to constraints

    :math:`\int dx^\prime\gamma(x, x^\prime) = p(x)`

    :math:`\int dx\gamma(x, x^\prime) = q(x^\prime)`

    Via Kantorovich-Rubinstein duality, this is equivalent to

    :math:`\sup\limits_{f \in \mathcal{F}} \vert \mathbb{E}_{x \sim
    p}\left[f(x)\right] - \mathbb{E}_{x^\prime \sim q}\left[f(x^\prime)\right]
    \vert`

    , with
    :math:`\mathcal{F} = \{f: \|f(x) - f(x^\prime)\|_1 \le \|x - x^\prime\|_1 \}`

    According to `Joel Tropp's thesis section 4.3.1
    <http://users.cms.caltech.edu/~jtropp/papers/Tro04-Topics-Sparse.pdf>`_,
    the operator norm of a linear transformation from an :math:`L^1` metric
    space to an :math:`L^1` metric space is bounded above by the :math:`L^1`
    norms of its columns. This is realized by normalzing the weight columns
    with an :math:`L^1` norm and excluding residual connections before applying
    them.""",
)
__all__.append("calc_em")

calc_js = calc_div_variational(
    js_data_stream,
    fdiv_loss(lambda y: (jax.numpy.log(-1 / (y * numpy.log(2))) - 1) / numpy.log(2)),
    model_generator=partial(Approximator, activation=lambda y: -jax.numpy.exp(y)),
    summary=r"""Jensen-Shannon divergence calculator

    :math:`f(x) = -\log_2(x)`

    :math:`f^{*}(y) = \sup\limits_x \left[xy - f(x)\right]`

    :math:`\frac{d}{dx}\left[xy - f(x)\right] = 0`

    :math:`x = \frac{-1}{y\log(2)}`

    :math:`f^{*}(y) = -\frac{\log\left(-y\log(2)\right) + 1}{\log(2)}`

    Note that the domain of this function (when assumed to be real valued) is
    naturally :math:`y < 0`.""",
)
__all__.append("calc_js")

calc_hl = calc_div_variational(
    fdiv_data_stream,
    fdiv_loss(lambda y: -1 / (4 * y) - 1),
    model_generator=partial(Approximator, activation=lambda y: -abs(y)),
    summary=r"""Hellinger distance calculator

    :math:`f(x) = 1 - \sqrt{x}`

    :math:`f^{*}(y) = \sup\limits_x\left[xy - f(x)\right]`

    :math:`\frac{d}{dx}\left[xy - f(x)\right] = 0`

    :math:`x = \frac{1}{4y ^ 2}`

    :math:`f^{*}(y) = \frac{1}{2\vert y \vert} + \frac{1}{4y} - 1`

    Since the `Fenchel–Moreau theorem
    <https://en.wikipedia.org/wiki/Fenchel%E2%80%93Moreau_theorem>`_ requires
    the convex conjugate to be lower semicontinuous for bicongugacy to hold, we
    take :math:`y < 0`.

    This in turn simplifies the expression of :math:`f^{*}` to

    :math:`f^{*}(y) = -\frac{1}{4y} - 1`""",
)
__all__.append("calc_hl")


@public.add
def histogram(data):
    histogram = {}
    N = len(data)
    for key, count in Counter(map(tuple, data)).items():
        histogram[key] = count / N
    return histogram


def join_keys(dictionaries):
    keys = set()
    for dictionary in dictionaries:
        keys |= dictionary.keys()
    return keys


@public.add
def average_histograms(histograms):
    avg = {}
    N = len(tuple(histograms))
    for key in join_keys(histograms):
        p = 0
        for histogram in histograms:
            if key in histogram:
                p += histogram[key]
            avg[key] = p / N
    return avg


@public.add
def cat_histograms(histograms):
    histogram = {}
    for key in join_keys(histograms):
        histogram[key] = tuple(
            histogram[key] if key in histogram else 0 for histogram in histograms
        )
    return histogram


@public.add
def get_density_estimators(categorical_columns, *sample_distributions):
    return cat_histograms(
        tuple(
            average_histograms(
                tuple(
                    histogram(samples[:, categorical_columns])
                    for samples in sample_distribution
                )
            )
            for sample_distribution in sample_distributions
        )
    )


@public.add
def metric_from_density(metric, *densities):
    return metric(*numpy.asarray(densities).T)


@public.add
def calc_mle(metric, *sample_distributions):
    sample_distributions = tuple(map(arrayify, sample_distributions))
    categorical_columns = numpy.arange(sample_distributions[0][0].shape[1], dtype="int")
    densities = get_density_estimators(
        categorical_columns, *sample_distributions
    ).values()
    return metric_from_density(metric, *densities)


@public.add
def calc_hl_density(density_p, density_q):
    r"""Hellinger distance calculated from histograms.

    Hellinger distance is defined as

    :math:`\sqrt{\frac{1}{2}\sum\limits_{x\in\mathcal{X}}\left(\sqrt{p(x)} -
    \sqrt{q(x)}\right)^2}`.

    Args:
        density_p (list): probability mass function of p
        density_q (list): probability mass function of q"""
    return numpy.sqrt(((numpy.sqrt(density_p) - numpy.sqrt(density_q)) ** 2).sum() / 2)


@public.add
def calc_hl_mle(sample_distribution_p, sample_distribution_q):
    r"""Hellinger distance calculated via histogram based density estimators.

    Hellinger distance is defined as

    :math:`\sqrt{\frac{1}{2}\sum\limits_{x\in\mathcal{X}}\left(\sqrt{p(x)} -
    \sqrt{q(x)}\right)^2}`.

    Args:
        sample_distribution_p (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton.
        sample_distribution_q (list):"""
    return calc_mle(calc_hl_density, sample_distribution_p, sample_distribution_q)


@public.add
def calc_tv_density(density_p, density_q):
    r"""Total variation calculated from histograms.

    For two distributions, :math:`p` and :math:`q` defined over the same
    probability space, :math:`\mathcal{X}`, the total variation is defined as

    :math:`\frac{1}{2}\sum\limits_{x\in\mathcal{X}}\vert p(x) - q(x)\vert`.

    Args:
        density_p (list): probability mass function of p
        density_q (list): probability mass function of q"""
    return numpy.abs(density_p - density_q).sum() / 2


@public.add
def calc_tv_mle(sample_distribution_p, sample_distribution_q):
    r"""Total variation calculated via histogram based density estimators. All
    columns are assumed to be categorical.

    For two distributions, :math:`p` and :math:`q` defined over the same
    probability space, `\mathcal{X}`, the total variation is defined as

    :math:`\frac{1}{2}\sum\limits_{x\in\mathcal{X}}\vert p(x) - q(x)\vert`.

    Args:
        sample_distribution_p (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton.
        sample_distribution_q (list):"""
    return calc_mle(calc_tv_density, sample_distribution_p, sample_distribution_q)


@public.add
def calc_kl_density(density_p, density_q):
    r"""Kullback–Leibler (KL) divergence calculated from histograms.

    For two distributions, :math:`p` and :math:`q` defined over the same
        probability space, `\mathcal{X}`, the total variation is defined as

    :math:`\sum\limits_{x\in\mathcal{X}}p(x)\log\left(\frac{p(x)}{q(x)}\right)`.

    Args:
        density_p (list): probability mass function of :math:`p`
        density_q (list): probability mass function of :math:`q`"""
    return numpy.log((density_p / density_q) ** density_p).sum()


@public.add
def calc_kl_mle(sample_distribution_p, sample_distribution_q):
    r"""Kullback–Leibler (KL) divergence calculated via histogram based density estimators.

    For two distributions, :math:`p` and :math:`q` defined over the same
    probability space, `\mathcal{X}`, the KL divergence is defined as

    :math:`\sum\limits_{x\in\mathcal{X}}p(x)\log\left(\frac{p(x)}{q(x)}\right)`.

    Args:
        sample_distribution_p (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton.
        sample_distribution_q (list):"""
    return calc_mle(calc_kl_density, sample_distribution_p, sample_distribution_q)


@public.add
def calc_js_density(*densities):
    r"""Jensen-Shannon divergence calculated from histograms.

    For two distributions, :math:`p` and :math:`q` defined over the same
    probability space, `\mathcal{X}`, the Jensen-Shannon divergence is defined
    as the average of the KL divergences between each probability mass function
    and the average of all probability mass functions being compared. This is
    well defined for more than two probability masses, and will be zero when
    all probability masses have disjoint support and 1 when they are all
    identical and the KL divergences are taken using a logarithmic base equal
    to the number of probability masses being compared. Typically, there will
    be only two probability mass functions, and the logarithmic base is
    therefore taken to be 2.

    Args:
        *densities (list): probability mass functions"""
    n = len(densities)
    mean = sum(densities) / n
    return sum(calc_kl_density(density, mean) for density in densities) / (
        n * numpy.log(n)
    )


@public.add
def calc_js_mle(*sample_distributions):
    r"""Jensen-Shannon divergences calculated via histogram based density estimators.

    For two distributions, :math:`p` and :math:`q` defined over the same
    probability space, `\mathcal{X}`, the Jensen-Shannon divergence is defined
    as the average of the KL divergences between each probability mass function
    and the average of all probability mass functions being compared. This is
    well defined for more than two probability masses, and will be zero when
    all probability masses have disjoint support and 1 when they are all
    identical and the KL divergences are taken using a logarithmic base equal
    to the number of probability masses being compared. Typically, there will
    be only two probability mass functions, and the logarithmic base is
    therefore taken to be 2.

    Args:
        *sample_distributions (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton."""
    return calc_mle(calc_js_density, *sample_distributions)


@public.add
def cal_div_knn(
    divergence,
    sample_distribution_p,
    sample_distribution_q,
    bias=lambda N, k: 0,
    num_samples=2048,
    categorical_columns=tuple(),
    nprng=numpy.random.RandomState(0),
    k=128,
):
    r""":math:`f`-divergence from knn density estimators

    Args:
        divergence: :math:`f` that defines the :math:`f`-divergence.
        sample_distribution_p (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton.
        sample_distribution_q (list):
        bias (function): function of the number of samples and number of
            nearest neighbors that compensates for expected bias of plugin
            estimator.
        num_samples (int, optional): Number of subsamples to take
            from each distribution on which to construct kdtrees and
            otherwise make computations. Defaults to 2046.
        k (int, optional): Number of nearest neighbors. As a rule of
            thumb, you should multiply this by two with every dimension
            past one. Defaults to 128."""
    sample_distribution_p = arrayify(sample_distribution_p)
    sample_distribution_q = arrayify(sample_distribution_q)
    p, q = next(
        fdiv_data_stream(
            nprng,
            num_samples,
            (sample_distribution_p, sample_distribution_q),
            categorical_columns=categorical_columns,
        )
    )

    def knn_ratio(ptree, qtree, x):
        d = max(qtree.query(x, k=k)[0])
        n = len(ptree.query_ball_point(x, d))
        return divergence(n / (k + 1))

    numerator = 0
    denominator = 0
    for key, conditional in q.items():
        denominator += len(conditional)
        if key not in p:
            continue
        qtree = scipy.spatial.cKDTree(conditional)
        ptree = scipy.spatial.cKDTree(p[key])
        numerator += numpy.sum(parallel(partial(knn_ratio, ptree, qtree), qtree.data))
    return max(0, numerator / denominator - bias(num_samples, k))


@public.add
def calc_tv_knn(sample_distribution_p, sample_distribution_q, **kwargs):
    r"""Total variation from knn density estimators

    Args:
        divergence: :math:`f` that defines the :math:`f`-divergence.
        sample_distribution_p (list): A numpy array,
            pandas dataframe, or pandas series or a list of numpy arrays,
            dataframes or series. If it is a list then will sample from each in the
            list For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 + p_7}{2}`.
            The outer most list is typically a singleton.
        sample_distribution_q (list):
        bias (function): function of the number of samples and number of
            nearest neighbors that compensates for expected bias of plugin
            estimator.
        num_samples (int, optional): Number of subsamples to take
            from each distribution on which to construct kdtrees and
            otherwise make computations. Defaults to 2046.
        k (int, optional): Number of nearest neighbors. As a rule of
            thumb, you should multiply this by two with every dimension
            past one. Defaults to 128."""

    def bias(N, k):
        def integral_no_p(p):
            return (
                (1 - p) ** (-k + N) * p ** k
                - N * scipy.special.betainc(k, 1 - k + N, p)
            ) / (k - N)

        def integral_with_p(p):
            return scipy.special.betainc(k + 1, N - k, p)

        r0 = (k - 1) / N
        p_less = (
            integral_no_p(r0)
            - integral_with_p(r0)
            - (integral_no_p(0) - integral_with_p(0))
        )
        p_greater = (
            integral_with_p(1)
            - integral_no_p(1)
            - (integral_with_p(r0) - integral_no_p(r0))
        )
        return p_less + p_greater

    return cal_div_knn(
        lambda r: abs(1 - r),
        sample_distribution_p,
        sample_distribution_q,
        bias=bias,
        **kwargs
    )


@public.add
def balanced_binary_cross_entropy(y_true, y_pred):
    r"""Compute cross entropy loss while compensating for class imbalance

    Args:
        y_true (array): Ground truth, binary or soft labels.
        y_pred (array): Array of model scores."""
    P = y_true.sum()
    N = len(y_true) - P
    return (
        scipy.special.rel_entr(y_true, y_pred).sum() / P
        + scipy.special.rel_entr(1 - y_true, 1 - y_pred).sum() / N
    ) / 2


@public.add
def calc_tv_lower_bound(log_loss):
    r"""Lower bound of total variation. A model (not provided) must be trained
    to classify data as belonging to one of two datasets using log loss,
    ideally compensating for class imbalance during training. This function
    will compute an lower bound on the total variation of the two datasets the
    model was trained to distinguish using the loss from the validation set.

    Args:
        log_loss (float): Binary cross entropy loss with class imbalance
            compensated."""

    js0 = 1 - log_loss / numpy.log(2)
    return max(0, js0)
