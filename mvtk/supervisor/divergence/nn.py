import jax
import public

from jax.experimental import stax
from jax._src.nn.initializers import glorot_normal, normal
from jax.experimental.stax import Dense, FanInSum, FanOut, Identity, Relu, elementwise


def ResBlock(*layers, fan_in=FanInSum, tail=Identity):
    """Split input, feed it through one or more layers in parallel, recombine
    them with a fan-in, apply a trailing layer (i.e. an activation)

    Args:
        *layers: a sequence of layers, each an (init_fun, apply_fun) pair.
        fan_in, optional: a fan-in to recombine the outputs of each layer
        tail, optional: a final layer to apply after recombination


    Returns:
        A new layer, meaning an (init_fun, apply_fun) pair, representing the
        parallel composition of the given sequence of layers fed into fan_in
        and then tail. In particular, the returned layer takes a sequence of
        inputs and returns a sequence of outputs with the same length as the
        argument `layers`.
    """
    return stax.serial(FanOut(len(layers)), stax.parallel(*layers), fan_in, tail)


@public.add
def Approximator(
    input_size,
    depth=3,
    width=None,
    output_size=1,
    linear=Dense,
    residual=True,
    activation=lambda x: x,
    rng=jax.random.PRNGKey(0),
):
    r"""Basic Neural network based function
    :math:`\mathbb{R}^N\rightarrow\mathbb{R}^M` function approximator.

    Args:
        input_size (int): Size of input dimension.
        depth (int, optional): Depth of network. Defaults to ``3``.
        width (int, optional): Width of network. Defaults to ``input_size + 1``.
        output_size (int, optional): Number of outputs. Defaults to ``1``.
        linear (``torch.nn.Module``, optional): Linear layer drop in
            replacement. Defaults to ``jax.experimental.stax.Dense``.
        residual (bool, optional): Turn on ResNet blocks. Defaults to ``True``.
        activation (optional): A map from :math:`(-\infty, \infty)` to an
            appropriate domain (such as the domain of a convex conjugate).
            Defaults to the identity.
        rng (optional): Jax ``PRNGKey`` key. Defaults to `jax.random.PRNGKey(0)``.

    Returns:
        initial parameter values, neural network function
    """
    # input_size + output_size hidden hidden units is sufficient for universal
    # approximation given unconstrained depth even without ResBlocks.
    # https://arxiv.org/abs/1710.112780. With ResBlocks (as used below), only
    # one hidden unit is needed for Relu activation
    # https://arxiv.org/abs/1806.10909.
    if width is None:
        hidden = input_size + 1
    else:
        hidden = width
    if depth > 2:
        layers = [linear(hidden), Relu]
    else:
        layers = []
    for _ in range(depth - 2):
        if residual:
            layers.append(
                ResBlock(stax.serial(linear(hidden), Relu), linear(hidden), tail=Relu)
            )
        else:
            layers.append(linear(hidden))
    layers.append(linear(output_size))
    layers.append(elementwise(activation))
    init_approximator_params, approximator = stax.serial(*layers)
    _, init_params = init_approximator_params(rng, (-1, input_size))
    return init_params, approximator


@public.add
def NormalizedLinear(out_dim, W_init=glorot_normal(), b_init=normal()):
    r"""Linear layer with positive weights with columns that sum to one."""

    def init_fun(rng, input_shape):
        output_shape = input_shape[:-1] + (out_dim,)
        k1, k2 = jax.random.split(rng)
        W, b = W_init(k1, (input_shape[-1], out_dim)), b_init(k2, (out_dim,))
        return output_shape, (W, b)

    def apply_fun(params, inputs, **kwargs):
        W, b = params
        W_normalized = W / jax.numpy.abs(W).sum(0)
        return jax.numpy.dot(inputs, W_normalized) + b

    return init_fun, apply_fun
