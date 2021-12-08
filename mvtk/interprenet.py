import jax
import itertools
import public

from jax.experimental import optimizers, stax
from jax._src.nn.initializers import glorot_normal, normal


@public.add
def monotonic_constraint(weights):
    """Monotonicity constraint on weights."""
    return abs(weights)


@public.add
def lipschitz_constraint(weights):
    """Lipschitz constraint on weights.

    https://arxiv.org/abs/1811.05381
    """
    return weights / abs(weights).sum(0)


@public.add
def identity(weights):
    return weights


@public.add
def clip(x, eps=2 ** -16):
    return jax.numpy.clip(x, eps, 1 - eps)


@public.add
def ConstrainedDense(constraint):
    """Layer constructor function for a constrained dense (fully-connected)
    layer.

    Args:
        constraint (function): Transformation to be applied to weights
    """

    def Dense(out_dim, W_init=glorot_normal(), b_init=normal()):
        def init_fun(rng, input_shape):
            output_shape = input_shape[:-1] + (out_dim,)
            k1, k2 = jax.random.split(rng)
            W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
            return output_shape, (W, b)

        def apply_fun(params, inputs, **kwargs):
            W, b = params
            return jax.numpy.dot(inputs, constraint(W)) + b

        return init_fun, apply_fun

    return Dense


@public.add
def SortLayer(axis=-1):
    """Sort layer used for lipschitz preserving nonlinear activation function.

    https://arxiv.org/abs/1811.05381
    """

    def init_fun(rng, input_shape):
        output_shape = input_shape
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return jax.numpy.sort(inputs, axis=axis)

    return init_fun, apply_fun


@public.add
def net(layers, linear, activation):
    return stax.serial(
        *itertools.chain(*((linear(layer), activation()) for layer in layers))
    )


@public.add
def thread(fns):
    def composition(x):
        for fn in fns:
            x = fn(x)
        return x

    return composition


def partialnet(linear, activation):
    def init_fun(rng, input_shape, n_hyper, layers):
        params = []
        apply_funs = []
        for layer in layers:
            rng, layer_rng = jax.random.split(rng)
            layer_init, apply_layer = linear(layer)
            input_shape, layer_params = layer_init(layer_rng, input_shape)
            d1, d2 = input_shape
            d2 *= n_hyper
            input_shape = (d1, d2)
            params.append(layer_params)
            apply_funs.append(apply_layer)
            layer_init, apply_layer = activation()
            input_shape, layer_params = layer_init(layer_rng, input_shape)
            input_shape = (d1, d2)
            params.append(layer_params)
            apply_funs.append(apply_layer)

        def apply_net(multiparams, inputs):
            # (network 1 output), (network 2 output), ...
            for params, fn in zip(zip(*multiparams), apply_funs):
                inputs = jax.numpy.concatenate(
                    tuple(fn(param, inputs) for param in params), axis=1
                )
            return inputs

        return params, apply_net

    return init_fun


@public.add
def sigmoid_clip(inputs, eps=2 ** -16):
    return jax.scipy.special.expit(clip(inputs, eps))


@public.add
def constrained_model(
    constraints,
    get_layers=lambda input_shape: [input_shape + 1] * 2,
    output_shape=1,
    preprocess=identity,
    postprocess=sigmoid_clip,
    rng=jax.random.PRNGKey(0),
):
    """Create a neural network with groups of constraints assigned to each
    feature. Separate constrained neural networks are generated for each group
    of contraints. Each feature is fed into exactly one of these neural
    networks (the one that matches its assigned group of constraints). The
    output of these constrained neural networks are concatenated and fed into
    one final neural network that obeys the union of all constraints applied.

    Args:
        constraints (list): List sets of constraints (one frozenset of constraints
            for each feature)
        get_layers (function): Returns shape of constrained neural network
            given size of input (i.e. the number of features that will be fed into
            it).
        preprocess: Preprocessing function to be applied to feature vector
            before being sent through any neural networks. This can be useful for
            adjusting signs for monotonic neural networks or scales for lipschitz
            ones.
        postprocess: Final activation applied to output of neural network.
        rng: jax PRNGKey
    Returns:
        init_params, model
    """
    union = set()
    groups = {}
    for i, constraint in enumerate(constraints):
        union |= constraint
        if constraint not in groups:
            groups[constraint] = [i]
        else:
            groups[constraint].append(i)
    nets = []
    catted_size = 0
    for constraint, idx in groups.items():
        init_net, apply_net = net(
            get_layers(len(idx)), ConstrainedDense(thread(constraint)), SortLayer
        )
        rng, new_rng = jax.random.split(rng)
        suboutput_shape, params = init_net(new_rng, (-1, len(idx)))
        catted_size += suboutput_shape[1]
        nets.append((params, apply_net))
    params1, apply_nets = zip(*nets)
    init_net, apply_net2 = stax.serial(
        net(get_layers(catted_size), ConstrainedDense(thread(union)), SortLayer),
        ConstrainedDense(thread(union))(output_shape),
    )
    rng, new_rng = jax.random.split(rng)
    output_shape, params2 = init_net(new_rng, (-1, catted_size))
    params = (params1, params2)
    groups = {key: jax.numpy.asarray(value) for key, value in groups.items()}

    def apply_net_pipeline(params, inputs):
        inputs = preprocess(inputs)
        params1, params2 = params
        return postprocess(
            apply_net2(
                params2,
                jax.numpy.concatenate(
                    tuple(
                        apply_net(p, inputs[:, idx])
                        for p, apply_net, idx in zip(
                            params1, apply_nets, groups.values()
                        )
                    ),
                    axis=1,
                ),
            )
        )

    return params, apply_net_pipeline


@public.add
def cross_entropy(y, y_pred):
    return (y * jax.numpy.log(y_pred) + (1 - y) * jax.numpy.log(1 - y_pred)).mean()


@public.add
def parameterized_loss(loss, net):
    def _(params, batch):
        X, y = batch
        return loss(y, net(params, X))

    return _


@public.add
def batch_generator(X, y, balance=False):
    assert len(X) == len(y)

    if balance:
        weights = jax.numpy.empty(len(y))
        p = jax.numpy.mean(y)
        weights = jax.ops.index_update(weights, y == 1, 1 / p)
        weights = jax.ops.index_update(weights, y == 0, 1 / (1 - p))
        weights /= weights.sum()
        weights = jax.numpy.clip(weights, 0, 1)
    else:
        weights = None
    N = len(X)

    def _(batch_size, rng=jax.random.PRNGKey(0), replace=False):
        while True:
            rng, new_rng = jax.random.split(rng)
            idx = jax.random.choice(
                new_rng, N, shape=(batch_size,), p=weights, replace=replace
            )
            yield X[idx], y[idx]

    return _


@public.add
def train(
    train,
    test,
    net,
    metric,
    loss_fn=cross_entropy,
    mini_batch_size=32,
    num_epochs=64,
    step_size=0.01,
    track=1,
):
    """Train interpretable neural network. This routine will check accuracy
    using ``metric`` every ``track`` epochs. The model parameters with the
    highest accuracy are returned.

    Args:
        train (tuple): (X, y), each ``jax.numpy.array`` of type ``float``.
        test (tuple): (X, y), each ``jax.numpy.array`` of type ``float``.
        net (tuple): (init_params, model) a jax model returned by
            ``constrained_model``.
        metric (function): function of two jax arrays: ground truth and
            predictions. Returns ``float`` representing performance metric.
        loss_fn (function): function of two jax arrays: ground truth and
            predictions. Returns ``float`` representing loss.
        mini_batch_size (int): Size of minibatches from train used for
            stochastic gradient descent
        num_epochs (int): Number of epochs to train
        step_size (float): Step size used for stochastic gradient descent
        track (int): Number of epochs between metric checks
    Returns:
        best params
    """

    mini_batches = batch_generator(*train)(mini_batch_size)
    params, apply_net = net
    loss = parameterized_loss(loss_fn, apply_net)

    @jax.jit
    def update(i, opt_state):
        return opt_update(
            i, jax.grad(loss)(get_params(opt_state), next(mini_batches)), opt_state
        )

    opt_init, opt_update, get_params = optimizers.adam(step_size)
    opt_state = opt_init(params)
    best_performance = -jax.numpy.inf
    best_params = params
    for epoch in range(num_epochs):
        opt_state = update(epoch, opt_state)
        if epoch and not epoch % track:
            X, y = test
            params = get_params(opt_state)
            performance = metric(y, apply_net(params, X))
            if performance > best_performance:
                best_performance = performance
                best_params = params
            # print(epoch, best_performance)
    return best_params


@public.add
def plot(
    model,
    data,
    feature,
    N=256,
    n_interp=1024,
    fig=None,
    rng=jax.random.PRNGKey(0),
):
    r"""`Individual Conditional Expectation plot
    (blue) <https://christophm.github.io/interpretable-ml-book/ice.html>`_ and
    `Partial Dependence Plot
    (red) <https://christophm.github.io/interpretable-ml-book/pdp.html>`_.

    Args:
        model (function): function from data (as dataframe) to scores
        feature (string): Feature to examine
        N (int): size of sample from ``data`` to consider for averages and
            conditional expectation plots. Determines number of blue lines and
            sample size for averaging to create red line.
        n_interp (int): Number of values of ``feature`` to evaluate along
            x-axis. Randomly chosen from unique values of this feature within
            ``data``.
        fig: Matplotlib figure. Defaults to ``None``.
        rng (PRNGKey): Jax ``PRNGKey``
    Returns:
        matplolib figure"""
    import matplotlib.pylab as plt
    import matplotlib.pyplot

    if fig is None:
        fig = matplotlib.pyplot.gcf()

    plt.clf()

    plt.title(feature)
    plt.ylabel("Model Score")
    plt.xlabel(feature)

    data = data.sort_values([feature])
    rng, new_rng = jax.random.split(rng)
    all_values_idx = list(
        jax.random.choice(new_rng, len(data), shape=(N,), replace=False)
    )
    all_values = data.values[all_values_idx]
    _, unique_feature_idx = jax.numpy.unique(data[feature].values, return_index=True)
    nunique = len(unique_feature_idx)
    n_interp = min(nunique, n_interp)
    feature_idx = list(
        unique_feature_idx[
            jax.random.choice(rng, nunique, shape=(n_interp,), replace=False).sort()
        ]
    )
    feature_values = data[feature].values[feature_idx]
    rest = jax.numpy.asarray(
        [i for i, column in enumerate(data.columns) if column != feature], dtype="int32"
    )
    all_scores = []
    for replacement in all_values[:, rest]:
        fixed_values = jax.ops.index_update(
            data.values[feature_idx], jax.ops.index[:, rest], replacement
        )
        scores = model(fixed_values)
        all_scores.append(scores)
        plt.plot(feature_values, scores, "b", alpha=0.125)
    plt.plot(feature_values, jax.numpy.asarray(all_scores).mean(0), "r", linewidth=2.0)
    return fig
