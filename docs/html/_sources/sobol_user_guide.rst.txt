###########
Sobol User Guide
###########

**********
Motivation
**********

`Sensitivity analysis <https://en.wikipedia.org/wiki/Sensitivity_analysis>`_ is
concerned with the degree to which uncertainty in the output of a model can be
attributed to uncertainty in its inputs :cite:`saltelli2008global`. Variance
based sensitivity analysis, commonly known as `sobol sensitivity analysis
<https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_ seeks to
answer this question by attributing the variance of the output to variances in
one or more inputs. This breakdown is known as a sobol indices and are typically measured
in one of two ways: *first-order* indices and *total-effect* indices.
:cite:`sobol2001global`.  

The first-order sobol index with respect to some feature is given by averaging
the output of the model over all other values of all other features and
computing the variance of the result while varying the feature in question.
This is normalized by dividing by the total variance of the output measured by
varying all feature values :cite:`im1993sensitivity`. Their sum is between 0 and 1. The total-effect index is computed by first computing the variance of the
model output with respect to the feature in question, and then computing the
expectation of the result over values of all other
features. This is again normalized by the variance
of the output of the model across all features.
These will sum to a number greater than
or equal to 1. Both are discussed in more detail
here
`https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis
<https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_.

.. currentmodule:: sobol

:meth:`sobol` takes a model and dataset, and runs a
monte carlo simulation as described in the above
link to compute the first and total order sobol
indices. Each index is expressed as a one
dimensional array of length equal to the number of
features in the supplied data matrix. The model is
assumed to be a function that outputs one scalar
for each row of the data matrix.

.. code-block:: python
    
    import numpy
    from mvtk import sobol

    nprng = numpy.random.RandomState(0)

    data = nprng.normal(size=(1000, 4)) # 4 features
    model = lambda x: (x ** 2).dot([1, 2, 3, 4])
    total, first_order = sobol.sobol(model, data, N=500)

.. bibliography:: refs.bib
    :cited:
