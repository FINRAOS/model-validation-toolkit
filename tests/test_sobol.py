import numpy

from mvtk import sobol


def test_sobol():
    nprng = numpy.random.RandomState(0)
    data = nprng.uniform(size=(1000000, 4))
    coefficients = numpy.arange(1, 5)

    def model(x):
        return x.dot(coefficients)

    first_order, total = sobol.sobol(model, data)
    variance = model(data).std() ** 2
    V = coefficients**2 / 12
    assert numpy.allclose(first_order * variance, V, rtol=0.01)
    assert numpy.allclose(total.sum(), 1, rtol=0.01)
    assert numpy.allclose(total, first_order, rtol=0.01)
