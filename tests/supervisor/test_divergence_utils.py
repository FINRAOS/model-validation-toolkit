import numpy as np
import pandas as pd
import scipy

from mvtk.supervisor.divergence.utils import arrayify


def test_arrayify_dataframes():
    df_a = pd.DataFrame({"a": list(range(4))})
    assert (
        df_a.shape == arrayify(df_a)[0].shape
    ), "Dataframe shape is same after arrayify"
    assert (
        df_a.shape == arrayify([df_a])[0].shape
    ), "Dataframe shape is same after arrayify"
    assert isinstance(arrayify([df_a])[0], np.ndarray)
    assert isinstance(arrayify(df_a)[0], np.ndarray)


def test_arrayify_numpy():
    ones = np.ones((2, 4))
    ones_lst = arrayify(ones)
    assert (
        ones.shape == ones_lst[0].shape
    ), "Shape should be same after arrayify_as_array"
    ones_lst2 = arrayify([ones])
    assert (
        ones_lst[0].shape == ones_lst2[0].shape
    ), "Shape should be same after arrayify_as_array"
    ones_lst3 = arrayify([ones, ones])
    assert (
        ones_lst[0].shape == ones_lst3[0].shape
    ), "Shape should be same after arrayify_as_array"


def test_arrayify_csr():
    ones = scipy.sparse.csr_matrix(np.ones((2, 4)))
    ones_lst = arrayify(ones)
    assert (
        ones.shape == ones_lst[0].shape
    ), "Shape should be same after arrayify_as_array"
    ones_lst2 = arrayify([ones])
    assert (
        ones_lst[0].shape == ones_lst2[0].shape
    ), "Shape should be same after arrayify_as_array"
    ones_lst3 = arrayify([ones, ones])
    assert (
        ones_lst[0].shape == ones_lst3[0].shape
    ), "Shape should be same after arrayify_as_array"
