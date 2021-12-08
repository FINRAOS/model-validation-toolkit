Getting Started
===============

Model Validation Toolkit is an open source library that provides various
tools for model validation, data quality checks, analysis of thresholding,
sensitivity analysis, and interpretable model development. The purpose of this
guide is to illustrate some of the main features that Model Validation Toolkit
provides. Please refer to the README for installation instructions.  

Divergences
----------------------------------------

Model Validation Toolkit provides a fast and accurate means of assessing
large scale statistical differences between datasets. Rather than checking
whether two samples are identical, this check asserts that they are similar in
a statistical sense and can be used for data quality checks and concept drift
detection.

.. code-block:: python
    
    import numpy
    from mvtk.supervisor.divergence import calc_tv

    nprng = numpy.random.RandomState(0)

    train = nprng.uniform(size=(1000, 4)) # 4 features
    val = nprng.uniform(size=(1000, 4)) # 4 features

    # Close to 0 is similar; close to 1 is different
    print(calc_tv(train, val))

See the :doc:`user guide <supervisor_user_guide>` for more information.

Credibility
----------------------------------------

.. currentmodule:: mvtk.credibility

Model Validation Toolkit provides a lightweight suite to assess credibility
of model performance given a finite sample. Whether your validation set has
several dozen or million records, you can quantify your confidence in
performance using this module. For example, if a model correctly identifies 8
of 10 images, its empirical accuracy is 80%. However, that does not mean we
should be confident the accuracy could turn out to be lower if we had more
data. We would obviously be more confident in this assessment if it identified
800 of 1000 images, but how much more so? With a few assumptions and
:meth:`prob_below`, we can estimate the probability that the true accuracy
would be less than 70% if we had more data.

.. code-block:: python
    
    from mvtk.credibility import prob_below
    print(prob_below(8, 2, 0.7))

See the :doc:`user guide <credibility_user_guide>` for more information.

Thresholding
----------------------------------------

Model Validation Toolkit provides a module for determining and
dynamically seta nd sample thresholds for binary classifiers that maximize a
utility function. The general idea is to intelligently reassess false and true
negative rates in a production system. See the :doc:`user guide
<interprenet_user_guide>` for more information.

Sobol
----------------------------------------

.. currentmodule:: sobol

Model Validation Toolkit provides a lightweight module for `sobol
sensitivity analysis
<https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis>`_. This can
be used to assess and quantify uncertainty of model outputs with respect to
model inputs. The module currently supports first order and total sobol
indexes--both which are computed and reported using :meth:`sobol`.

.. code-block:: python
    
    import numpy
    from mvtk import sobol

    nprng = numpy.random.RandomState(0)

    data = nprng.normal(size=(1000, 4)) # 4 features
    model = lambda x: (x ** 2).dot([1, 2, 3, 4])
    total, first_order = sobol.sobol(model, data, N=500)

See the :doc:`user guide
<sobol_user_guide>` for more information.
