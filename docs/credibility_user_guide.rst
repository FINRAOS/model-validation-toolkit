###########
Credibility User Guide
###########

**********
Motivation
**********

Let's say we are training a model for medical diagnoses. Missing false negatives
is important and we have a hard requirement that a model's recall (proportion
of positive instances identified) must not fall below 70%. If someone validates
a model and reports a recall of 80%, are we clear? Well, maybe. It turns out
this data scientist had a validation set with 5 positive instances. The model
correctly identified 4 of them, giving it a recall of 80%. Would you trust
that? Of course not! You say that a larger sample size is needed. "How many do we
need?" they ask. This module will help answer that question.

How?
====

There's two schools of thought for this problem. The `frequentist
<https://en.wikipedia.org/wiki/Frequentist_probability>`_ and the
`Bayesian <https://en.wikipedia.org/wiki/Bayesian_probability>`_ approaches.
In practice they tend to give similar results. Going back to our 5 sample
validation set, the frequentist would be concerned with how much our recall
would be expected to vary from one 5 sample hold out set to another. They would
want the hold out set to be large enough that you would not expect much change
in the estimated recall from one hold out set to another. The Bayesian approach
seeks to directly identify the probability that the recall would be lower than
70% if the validation set were infinitely large. We believe this is a better
representation of the problem at hand, and designed the library around this
Bayesian approach.


***********
Beta Distributions
***********

Probability of Low Performance
=================

.. currentmodule:: mvtk.credibility

If you flip a coin 100 times, and it comes up heads 99 times, would you suspect
a biased coin? Probably. What about if you flipped it 5 times and saw 4 heads.
This is much less strange. Determining the bias of a coin embodies the core
principles behind determining whether many performance metrics are unacceptably
low.

If the coin *is* biased, how biased is it? In general, we'd say there's some
probability distribution over all possible biases. We would generally use a
`beta distribution <https://en.wikipedia.org/wiki/Beta_distribution>`_ to
model this distribution for good reasons. This distribution has two free
parameters: the number of heads and the number of tails. However, we generally
offset both of those numbers by 1 so the distribution for observed flips is
:math:`B(1, 1)` (with :math:`B` representing our beta distribution as a
function of heads and tails plus respective offsets), which as it turns out is
exactly a uniform distribution over all possible biases. In this sense, we can
express total uncertainty before taking measurements. The beta distribution
becomes more concentrated around the empirical proportion of heads as you take
more and more measurements. If, we were reasonably certain of a 60% bias, we
might offset the number of heads with a 6 and the number of tails with a 4.
Then we would start to expect an unbiased coin after observing 2 tails. This
offset is called the *prior* in Bayesian inference, and represents our
understanding before making any observations.  

.. math::
    B(\alpha, \beta)

.. figure:: images/Beta_distribution_pdf.svg
    :width: 800px
    :align: center
    :height: 400px
    :alt: alternate text
    :figclass: align-center

    Beta distribution for different :math:`alpha` (for heads plus offset) and
    :math:`\beta` (tails plus offset).

We integrate the area under :math:`B(\alpha,\beta)` from 0 to
:math:`p` to determine the probability that a coin's bias is less
than :math:`p`. This is effectively how :meth:`prob_below` works.


Credible Intervals
=================

Sometimes you just want a general sense of uncertainty for your sample
estimates. We use :meth:`credible_interval` to compute a `credible interval <https://en.wikipedia.org/wiki/Credible_interval>`_. This will give you the
smallest interval for which there is a `credibility` (kwarg argument that
defaults to :math:`0.5`) chance of the bias being within that region. It will
return a lower bound no less than :math:`0` and an upper bound no greater than :math:`1`.
This is subtly different from frequentist `confidence intervals
<https://en.wikipedia.org/wiki/Confidence_interval>`_. In our 5 sample
example, the latter reports an interval that is expected to contain `p` (often
chosen to be 95%) all such 5 sample estimates of the mean.

**********
Common Metrics
**********
Many performance metrics used for binary
classification follow the same mechanics as the
analysis above. This following is not an exhaustive
list of performance metrics that can be readily
translated into a biased coin scenario in which we
wish to determine heads / (heads + tails).

* Precision: true positive / (true positive + false positive)
* Recall: true positive / (true positive + false negative)
* Accuracy: correctly identified / (correctly identified + incorrectly identified)


ROC AUC
=================

`ROC AUC
<https://en.wikipedia.org/wiki/Receiver_operating_characteristic>`_
is an extremely useful measure for binary classification. Like many
other measures of performance for binary classification, it can be
expressed as a proportion of outcomes. However,
unlike other measures of performance, it does not
make use of a threshold. This ultimately makes it a
ranking metric, as it characterizes the degree to
which positive instances are scored higher than
negative instances. However, like other metrics, it
can be expressed as an empirical measure of a
proportion. Specifically, ROC AUC is the proportion
of pairs of positive and negative examples such
that the positive example is scored higher than the
negative one. This can be expressed as 

.. math::
    \frac{1}{NM}\sum\limits_{n,m}^{N,M} \mathrm{score}(\mathrm{Positive}_n) > \mathrm{score}(\mathrm{Negative}_m)

However, computing the area under the receiver
operating characteristic is a more computationally
efficient means of computing the same quantity.
:meth:`roc_auc_preprocess` will convert a positive and negative
sample count to an associated count of correctly and incorrectly
ranked pairs of positive and negative instances using the ROC AUC
score. This pair of numbers can be used as arguments for
:meth:`prob_below` and :meth:`credible_interval`.

.. topic:: Tutorials:

    * :doc:`Credibility <notebooks/credibility/Credibility>`
