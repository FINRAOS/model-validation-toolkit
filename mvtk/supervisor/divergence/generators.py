import numpy
import public

from collections import defaultdict
from functools import reduce


@public.add
def js_data_stream(
    nprng, batch_size, sample_distributions, categorical_columns=tuple()
):
    r"""Data stream generator for Jensen-Shannon divergence of N distributions.
    Jensen-Shannon divergence measures the information of knowing which of
    those N distributions a sample will be drawn from before it is drawn. So if
    we rolled a fair N sided die to determine which distribution we will draw a
    sample from, JS divergence reports how many bits of information will be
    revealed from the die. This scenario is ultimately simulated in this
    function. However, in real life, we may only have examples of samples from
    each distribution we wish to compare. In the most general case, each
    distribution we wish to compare is represented by M samples of samples
    (with potentially different sizes) from M similar distributions whose
    average is most interesting. Just as we might simulate sampling from a
    single distribution by randomly sampling a batch of examples with
    replacement, we can effectively sample from an average of distributions by
    randomly sampling each batch (which may be representative of a single
    distribution), then randomly sampling elements of the chosen batch. This
    can ultimately be thought of a more data efficient means to the same end as
    downsampling large batch sizes.

    Args:
        nprng: Numpy ``RandomState`` used to generate random samples
            batch_size: size of batch
        *sample_distributions: list of lists of samples to compare.
            For example, ``[[batch1, batch2, batch3], [batch4, batch5],
            [batch6, batch7]]`` Assuming ``batch1`` came from distribution
            :math:`p_1`, ``batch2`` from :math:`p_2`, etc, this function will
            simulate a system in which a latent `N=3` sided die role that
            determines whether to draw a sample from :math:`\frac{p_1 + p_2 +
            p_3}{3}`, :math:`\frac{p_4 + p_5}{2}`, or :math:`\frac{p_6 +
            p_7}{2}`.
        categorical_columns (tuple): list or tuple of column indices that are
            considered categorical.

    Returns:
        The output of this function will be two samples of size batch_size with
        samples, :math:`x`, drawn from batch_size roles, :math:`z`, of our
        :math:`N` sided die. Following the example above for which :math:`N=3`,
        the first of these two output samples will be of the form :math:`(x,
        z)`, where x is the sample drawn and z is the die roll. The second of
        these two samples will be of the form :math:`(x, z^{\prime})` where x
        is the same sample as before, but :math:`z^\prime` is a new set of
        otherwise unrelated roles of the same :math:`N=3` sided die."""

    def process_sample_distributions(sample_distributions):
        z = []
        out = []
        for idx, count in zip(
            *numpy.unique(
                nprng.randint(0, len(sample_distributions), size=batch_size),
                return_counts=True,
            )
        ):
            sample_distribution = sample_distributions[idx]
            out.extend(
                [
                    sample_distribution[i][
                        nprng.randint(0, len(sample_distribution[i]))
                    ]
                    for i in nprng.randint(0, len(sample_distribution), size=count)
                ]
            )
            z.extend([idx] * count)
        sample_distribution = numpy.asarray(out)
        catted1 = numpy.concatenate(
            (sample_distribution, numpy.asarray(z)[:, numpy.newaxis]), axis=1
        )
        z = nprng.randint(0, len(sample_distributions), size=batch_size)
        catted2 = numpy.concatenate((sample_distribution, z[:, numpy.newaxis]), axis=1)
        return numpy.asarray((catted2, catted1))

    while True:
        yield groupby(
            categorical_columns, *process_sample_distributions(sample_distributions)
        )


@public.add
def fdiv_data_stream(
    nprng, batch_size, sample_distributions, categorical_columns=tuple()
):
    r"""Data stream generator for f-divergence.

    Args:
        nprng: Numpy ``RandomState`` used to generate random samples
        batch_size: size of batch
        sample_distributions: list of lists of samples to compare for each
            partition of the data. For example, ``[[batch1, batch2, batch3],
            [batch4, batch5], [batch6, batch7]]``
        categorical_columns (tuple): list or tuple of column indices that are
            considered categorical.

    Returns:
        The output of this function will be ``N`` samples of size
        ``batch_size``, where ``N = len(sample_distributions)`` Following the
        example above, assuming ``batch1`` came from distribution p_1,
        ``batch2`` from :math:`p_2`, etc, This function will output a tuple of
        ``N = 3`` samples of size ``batch_size``, where ``batch1`` is sampled
        from :math:`\frac{p_1 + p_2 + p_3}{3}`, ``batch2`` is sampled from
        :math:`\frac{p_4 + p_5}{2}`, and ``batch3`` is sampled from
        :math:`\frac{p_6 + p_7}{2}`."""

    def process_sample_distributions(sample_distributions):
        return numpy.asarray(
            [
                [
                    sample_distribution[i][
                        nprng.randint(0, len(sample_distribution[i]))
                    ]
                    for i in nprng.randint(0, len(sample_distribution), size=batch_size)
                ]
                for sample_distribution in sample_distributions
                if len(sample_distribution)
            ]
        )

    while True:
        yield groupby(
            categorical_columns, *process_sample_distributions(sample_distributions)
        )


def groupby(categorical_columns, *samples):
    r"""Group samples by unique values found in a subset of columns
    Args:
        categorical_columns: List of indices of columns which should be
            treated as categorical.
        *samples: A set of samples drawn from distinct distributions.
            Each distribution is assumed to be defined on the same probability
            space, so it would make sense to compare a sample drawn from one
            distribution to a sample drawn from another.

    Returns:
        tuple of dicts that each map unique combinations of
        ``categorical_columns`` to a subset of samples from the
        ``sample_distributions`` that have these values in their
        ``categorical_columns``. ``categorical_columns`` are omitted from
        the values of these dicts."""
    if not categorical_columns:
        return [{tuple(): sample.astype("float")} for sample in samples]
    # the complement of categorical_columns is assumed to be numeric
    numerical_columns = [
        i for i in range(samples[0].shape[1]) if i not in categorical_columns
    ]

    def grouper(accum, element):
        accum[tuple(element[categorical_columns])].append(element[numerical_columns])
        return accum

    return tuple(
        {
            key: numpy.asarray(value, dtype="float")
            for key, value in reduce(grouper, sample, defaultdict(list)).items()
        }
        for sample in samples
    )
