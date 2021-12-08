import numpy
import public

from sklearn.metrics import roc_auc_score
from scipy.stats import beta


@public.add
def credible_interval(positive, negative, credibility=0.5, prior=(1, 1)):
    """What is the shortest interval that contains probability(positive) with
    `credibility`% probability?

    Args:
        positive (int): number of times the first possible outcome has been seen
        negative (int): number of times the second possible outcome has been seen
        credibility (float): The probability that the true p(positive) is
            contained within the reported interval
        prior (tuple): psueodcount for positives and negatives

    returns:
        (lower bound, upper bound)
    """
    distribution = beta(positive + prior[0], negative + prior[1])
    mode = positive / (positive + negative)
    cdf_mode = distribution.cdf(mode)
    cred_2 = credibility / 2
    lower = cdf_mode - cred_2
    true_lower = max(lower, 0)
    excess = true_lower - lower
    upper = cdf_mode + cred_2 + excess
    true_upper = min(upper, 1)
    excess = upper - true_upper
    true_lower -= excess
    assert numpy.isclose((true_upper - true_lower), credibility)
    return distribution.ppf(true_lower), distribution.ppf(true_upper)


@public.add
def prob_below(positive, negative, cutoff, prior=(1, 1)):
    """What is the probability P(positive) is unacceptably low?

    Args:
        positive (int): number of times the positive outcome has been seen
        negative (int): number of times the negative outcome has been seen
        cutoff (float): lowest acceptable value of P(positive)
        prior (tuple): psueodcount for positives and negatives
    returns:
        Probability that P(positive) < cutoff
    """
    return beta(prior[0] + positive, prior[1] + negative).cdf(cutoff)


@public.add
def roc_auc_preprocess(positives, negatives, roc_auc):
    """ROC AUC analysis must be preprocessed using the number of positive and
    negative instances in the entire dataset and the AUC itself.

    Args:
        positives (int): number of positive instances in the dataset
        negatives (int): number of negative instances in the dataset
        roc_auc (float): ROC AUC
    returns:
        (positive, negative) tuple that can be used for `prob_below` and
            `credible_interval`
    """
    unique_combinations = positives * negatives
    # correctly ranked combinations are pairs of positives and negatives
    # instances where the model scored the positive instance higher than the
    # negative instance
    correctly_ranked_combinations = roc_auc * unique_combinations
    # the number of incorrectly ranked combinations is the number of
    # combinations that aren't correctly ranked
    incorrectly_ranked_combinations = (
        unique_combinations - correctly_ranked_combinations
    )
    return correctly_ranked_combinations, incorrectly_ranked_combinations


@public.add
def prob_greater_cmp(
    positive1,
    negative1,
    positive2,
    negative2,
    prior1=(1, 1),
    prior2=(1, 1),
    err=10 ** -5,
):
    """Probability the first set comes from a distribution with a greater
    proportion of positive than the other.

    Args:
        positive1 (int): number of positive instances in the first dataset
        negative1 (int): number of negative instances in the first dataset
        positive1 (int): number of positive instances in the second dataset
        negative1 (int): number of negative instances in the second dataset
        prior1 (tuple): psueodcount for positives and negatives
        prior2 (tuple): psueodcount for positives and negatives
        err (float): upper bound of frequentist sample std from monte carlo simulation.
    """
    nprng = numpy.random.RandomState(0)
    distribution1 = beta(positive1 + prior1[0], negative1 + prior1[1])
    distribution2 = beta(positive2 + prior2[0], negative2 + prior2[1])
    # CLT implies ROC AUC error shrinks like 1/PN
    # for P positives and N negatives
    N = int(1 + 1 / (2 * err))
    sample1 = distribution1.rvs(N, random_state=nprng)
    sample2 = distribution2.rvs(N, random_state=nprng)
    y = numpy.ones(2 * N)
    y[N:] = 0
    return roc_auc_score(y, numpy.concatenate((sample1, sample2)))
