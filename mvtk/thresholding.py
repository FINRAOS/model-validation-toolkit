import public
import bisect
import numpy
import matplotlib.pylab as plt

from functools import reduce


@public.add
def plot_err(scores, utility_mean, utility_err, color=None, label=None, alpha=0.5):
    plt.plot(scores, utility_mean, color=color)
    plt.fill_between(scores, *utility_err, alpha=alpha, color=color, label=label)


@public.add
def expected_utility(utility, data, N=4096, credibility=0.5):
    """Get the utility distribution over possible thresholds.

    Args:
        utility (function): utility function that ingests true/false
            positive/negative rates.
        data (list-like): iterable of list-likes of the form (ground truth,
            score). Feedback is null when an alert is not triggered.
        credibility (float): Credibility level for a credible interval. This
            interval will be centered about the mean and have a `credibility`
            chance of containing the true utility.

    returns:
        tuple of three elements:
        - candidate thresholds
        - mean expected utility
        - upper and lower quantile of estimate of expected utility associated
          with each threshold
    """
    credibility /= 2
    scores, utilities = sample_utilities(utility, data, N=N)
    low = int(N * credibility)
    high = int(N * (1 - credibility))
    utilities = numpy.asarray(utilities)
    utilities.sort(axis=1)
    return scores, utilities.mean(1), numpy.asarray(utilities[:, [low, high]]).T


@public.add
def optimal_threshold(utility, data, N=4096):
    scores, utilities = sample_utilities(utility, data, N=N)
    means = utilities.mean(1)
    idx = means.argmax()
    return scores[idx], means[idx]


@public.add
def sample_utilities(utility, data, N=4096):
    """Get distribution of utilities.

    Args:
        utility (float): utility function that ingests true/false
            positive/negative rates.
        data (list-like): iterable of of iterables of the form (ground truth, score).
            Feedback is null when an alert is not triggered.

    returns: thresholds, utilities
    """
    if not len(data):
        return data, numpy.asarray([])
    nprng = numpy.random.RandomState(0)
    data = numpy.asarray(data)
    num_positives = data[:, 0].sum()
    rates = [1 + num_positives, 1 + len(data) - num_positives, 1, 1]
    utilities = []
    data = data[numpy.argsort(data[:, 1])]
    for ground_truth, score in data:
        update_rates(rates, ground_truth)
        utilities.append(utility(*nprng.dirichlet(rates, size=N).T))
    return data[:, 1], numpy.asarray(utilities)


@public.add
def thompson_sample(utility, data, N=1024, quantile=False):
    scores, utilities = sample_utilities(utility, data, N)
    if quantile:
        return utilities.argmax(axis=0) / (len(utilities) - 1)
    return scores[utilities.argmax(axis=0)]


@public.add
def update_rates(rates, ground_truth):
    rates[0] -= ground_truth
    rates[1] -= not ground_truth
    rates[2] += not ground_truth
    rates[3] += ground_truth


@public.add
class AdaptiveThreshold:
    """Adaptive agent that balances exploration with exploitation with respect
    to setting and adjusting thresholds.

    When exploring, the threshold is 0, effectively letting anything
    through. This produces unbiased data that can then be used to set a
    more optimal threshold in subsequent rounds. The agent seeks to
    balance the opportunity cost of running an experiment with the
    utility gained over subsequent rounds using the information gained
    from this experiment.
    """

    def __init__(self, utility):
        """
        Args:
            utility (function): Function that takes in true/false
                positive/negative rates. Specifically (tp, fp, tn fn) -> float
                representing utility."""

        self.utility = utility
        self.results = []
        self.unbiased_positives = 1
        self.unbiased_negatives = 1
        self.previous_threshold = 0
        self.nprng = numpy.random.RandomState(0)

    def get_best_threshold(self):
        # true positives, false positives, true negatives, false negatives
        rates = [self.unbiased_positives, self.unbiased_negatives, 1, 1]
        experiment_utility = self.utility(*self.nprng.dirichlet(rates))
        hypothetical_rates = [
            self.unbiased_positives - self.last_experiment_outcome,
            self.unbiased_negatives - (1 - self.last_experiment_outcome),
            1,
            1,
        ]
        best_hypothetical_utility = -numpy.inf
        best_utility = -numpy.inf
        for score, ground_truth, idx in self.results:
            update_rates(rates, ground_truth)
            utility = self.utility(*self.nprng.dirichlet(rates))
            if utility > best_utility:
                best_utility = utility
                best_threshold = score
            if idx >= self.last_experiment_idx:
                continue
            update_rates(hypothetical_rates, ground_truth)
            hypothetical_utility = self.utility(
                *self.nprng.dirichlet(hypothetical_rates)
            )
            if hypothetical_utility > best_hypothetical_utility:
                best_hypothetical_utility = hypothetical_utility
                hindsight_utility = utility
        return best_threshold, experiment_utility, best_utility, hindsight_utility

    def __call__(self, ground_truth, score):
        """Args are ignored if previous threshold was not 0. Otherwise, the
        score is added as a potential threhold and ground_truth noted to help
        identify the optimal threshold.

        Args:
            ground_truth (bool)
            score (float)
        """
        idx = len(self.results)
        if self.previous_threshold == 0:
            bisect.insort(self.results, (score, ground_truth, idx))
            self.unbiased_positives += ground_truth
            self.unbiased_negatives += 1 - ground_truth
            self.last_experiment_idx = idx
            self.last_experiment_outcome = ground_truth
        if len(self.results) < 2:
            return self.previous_threshold
        (
            best_threshold,
            experiment_utility,
            best_utility,
            hindsight_utility,
        ) = self.get_best_threshold()
        total_utility_gained = (best_utility - hindsight_utility) * (
            idx - self.last_experiment_idx
        )
        opportunity_cost = hindsight_utility - experiment_utility
        if opportunity_cost <= total_utility_gained:
            self.previous_threshold = 0
        else:
            self.previous_threshold = best_threshold
        return self.previous_threshold


@public.add
def exploration_proportion(thresholds, N):
    exploration = thresholds == 0
    alpha = 1 - 1.0 / N
    return reduce(
        lambda accum, elem: accum + [accum[-1] * alpha + elem * (1 - alpha)],
        exploration[N:],
        [exploration[:N].mean()],
    )
