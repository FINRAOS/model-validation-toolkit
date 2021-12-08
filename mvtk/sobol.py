import numpy
import public


def choose(x, N, nprng=None):
    if nprng is None:
        nprng = numpy.random.RandomState(0)
    return x[nprng.choice(numpy.arange(len(x), dtype="int"), N)]


@public.add
def sobol(model, data, N=None, nprng=None):
    """Total and first order Sobol sensitivity indices.
    https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis.

    Args:
        model (function): Maps data to scores
        data (ndarray): Data matrix. Each row is a sample vector.
        N (int): sample size for monte carlo estimate of sobol
            indices. Should be less than or equal to the number of rows
            of data. If None, entire dataset is used.
        nprng (RandomState): Optional numpy RandomState.
    returns:
        Total and first order Sobol sensitivity indices. Each index
        is expressed as an array of length equal to the number of
        features in the supplied data matrix.
    """
    if nprng is None:
        nprng = numpy.random.RandomState(0)
    if N is None:
        A = data.copy()
        B = data.copy()
        nprng.shuffle(A)
        nprng.shuffle(B)
        N = len(data)
    elif N > len(data):
        raise ValueError("Sample size must be less than or equal to size of dataset")
    else:
        A, B = (choose(data, N, nprng) for _ in range(2))
    d = data.shape[1]
    total = []
    first_order = []
    for i in range(d):
        C = A[:, i].copy()
        A[:, i] = B[:, i]
        diff = model(A)
        A[:, i] = C
        diff -= model(A)
        first_order.append(model(B).dot(diff) / N)
        total.append(diff.dot(diff) / (2 * N))
    variance_y = model(numpy.vstack((A, B))).std() ** 2
    total = numpy.asarray(total) / variance_y
    first_order = numpy.asarray(first_order) / variance_y
    return total, first_order
