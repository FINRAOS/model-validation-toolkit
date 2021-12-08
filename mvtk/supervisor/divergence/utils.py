import itertools
import numpy
import scipy
import public

from mvtk.supervisor.utils import parallel


@public.add
def get_drift_series(metric, baseline, test):
    return numpy.asarray(parallel(lambda x: metric(x, baseline), test))


@public.add
def get_distance_matrix(metric, sample_distributions):
    distance_matrix = numpy.zeros((len(sample_distributions),) * 2)
    for index, d in parallel(
        lambda x: (x[0], metric(x[1][0], x[1][1])),
        [
            list(zip(*x))
            for x in itertools.combinations(enumerate(sample_distributions), 2)
        ],
        show_progress=True,
    ):
        distance_matrix[index] = d
    distance_matrix += distance_matrix.T
    return distance_matrix


@public.add
def sparse_wrapper(v):
    class _SparseWrapper(type(v)):
        def __getitem__(self, i):
            ret = super().__getitem__(i)
            if isinstance(i, int):
                return ret.toarray()[0]
            return ret

        def __len__(self):
            return self.shape[0]

    return _SparseWrapper(v)


def to_array_like(v):
    if hasattr(v, "values"):
        return v.values
    if isinstance(v, scipy.sparse.spmatrix):
        return sparse_wrapper(v)
    return v


@public.add
def arrayify(item):
    """Convert the value to at least dim 3. If is dataframe it converts it to a
    list of values.

    :param item: ndarray or a list of ndarray, or a dataframe, a series or a
        list of dataframes or series
    :return: a list of dataframes/series or array of dim 3
    """
    if hasattr(item, "shape"):
        ret = to_array_like(item)
        if len(ret.shape) == 2:
            return [ret]
        if len(ret.shape) == 1:
            return numpy.atleast_3d(ret)
    return list(map(to_array_like, item))
