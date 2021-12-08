###########
Interprenet User Guide
###########

**********
Motivation
**********

Neural networks are generally difficult to interpret. While there
are tools that can help to interpret certain types of neural
networks such as image classifiers and language models,
interpretation of neural networks that simply ingest tabular data
and return a scalar value is generally limited to various measures of feature
importance. This can be problematic as what makes a feature "important" can
vary between use cases.

Rather than interpret a neural network as a black box, we seek to constrain
neural network in ways we consider useful and interpretable. In particular, The
`interprenet <interprenet.html>`_ module currently have two such constraints
implemented:

* Monotonicity
* Lipschitz constraint

`Monotonic functions <https://en.wikipedia.org/wiki/Monotonic_function>`_
either always increase or decrease with their arguments but never both. This is
often an expected relationship between features and the model output. For
example, we may believe that increasing blood pressure increases risk of
cardiovascular disease. The exact relationship is not known, but we may believe
that it is monotonic.

`Lipschitz constraints
<https://en.wikipedia.org/wiki/Lipschitz_continuity>`_ constrain the
maximum rate of change of the model. This can make the model arbitrarily robust
`against adversarial perturbations
<http://karpathy.github.io/2015/03/30/breaking-convnets>`_
:cite:`anil2019sorting`.


How?
====

All constraints are currently implemented as weight constraints. While
arbitrary weights are stored within each linear layer, the weights are
transformed before application so the network can satisfy is prescribed
constraints. Changes are backpropagated through this transformation.
Monotonic increasing neural networks are implemented by taking the absolute
value of weight matrices before applying them. When paired with a monotonically
increasing activation (such as ReLU, Sigmoid, or Tanh), this ensures the
gradient of the output with respect to any features is positive. This is
sufficient to ensure monotonicity with respect to the features.

Lipschitz constraints are enforced by dividing each weight vector by
its :math:`L^\infty` norm as described in :cite:`anil2019sorting`. This
constrains the :math:`L^\infty`-:math:`L^\infty` `operator norm
<https://en.wikipedia.org/wiki/Operator_norm>`_
of the weight matrix :cite:`tropp2004topics`. Constraining the
:math:`L^\infty`-:math:`L^\infty` operator norm of the weight
matrix ensures every element of the jacobian of the linear layers is less than
or equal to :math:`1`. Meanwhile, using activation functions with Lipschitz
constants of :math:`1` ensure the entire network is constrained to never have a
slope greater than :math:`1` for any of its features.

**********
Different Constraints on Different Features
**********

.. currentmodule:: mvtk.interprenet

:meth:`constrained_model` generates a neural network with one set of
constraints per feature. Constraints currently available are:

- :meth:`identity` (for no constraint)
- :meth:`monotonic_constraint`
- :meth:`lipschitz_constraint`

Features are grouped by the set of constraints applied to them, and
different constrained neural networks are generated for each group
of features. The outputs of those neural networks are concatenated
and fed into a final neural network constrained using all
constraints applied to all features. Since constraints on weight
matrices compose, they can be applied as a series of transformations
on the weights before application.

.. figure:: images/interprenet.png
    :width: 500px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center

    4 features with Lipschitz constraints and 4 features wtih
    monotonic constraints are fed to their respectively constrained
    neural networks. Intermediate outputs are concatenated and fed into a neural
    network with monotonic and lipschitz constraints.

We use the Sort function as a nonlinear activation as described in
:cite:`anil2019sorting`. The jacobian of this matrix is always a
permutation matrix, which retains any Lipschitz and monotonicity
constraints.

**********
Preprocessing
**********

Thus far, we have left out two important detail: How to constrain
the Lipschitz constant to be something other than :math:`1`, and how
to create monotonically decreasing networks. Both are a simple
matter of preprocessing. The ``preprocess`` argument (defaulting to
``identity``), specifies a function to be applied to the feature
vector before passing it to the neural network. For decreasing
monotonic constraints, multiply the respective features by
:math:`-1`. For a Lipschitz constant of :math:`L`, multiply the
respective features by :math:`L`.

.. topic:: Tutorials:

    * :doc:`Interprenet <notebooks/interprenet/Interprenet>`

.. bibliography:: refs.bib
    :cited:
