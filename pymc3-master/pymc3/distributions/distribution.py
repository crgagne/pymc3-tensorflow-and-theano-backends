import numbers
import numpy as np

#import theano.tensor as tt
#from theano import function
#import theano
from .. import backends_symbolic as S

from ..memoize import memoize
from ..model import Model, get_named_nodes_and_relations, FreeRV, ObservedRV
from ..vartypes import string_types


__all__ = ['DensityDist', 'Distribution', 'Continuous', 'Discrete',
           'NoDistribution', #'TensorType',
            'draw_values']


class _Unpickling(object):
    pass


class Distribution(object):
    """Statistical distribution"""
    def __new__(cls, name, *args, **kwargs):
        if name is _Unpickling:
            return object.__new__(cls)  # for pickle
        try:
            model = Model.get_context()
        except TypeError:
            raise TypeError("No model on context stack, which is needed to "
                            "instantiate distributions. Add variable inside "
                            "a 'with model:' block, or use the '.dist' syntax "
                            "for a standalone distribution.")

        if isinstance(name, string_types):
            data = kwargs.pop('observed', None)
            if isinstance(data, ObservedRV) or isinstance(data, FreeRV):
                raise TypeError("observed needs to be data but got: {}".format(type(data)))
            total_size = kwargs.pop('total_size', None)
            dist = cls.dist(*args, **kwargs)
            return model.Var(name, dist, data, total_size)
        else:
            raise TypeError("Name needs to be a string but got: {}".format(name))

    def __getnewargs__(self):
        return _Unpickling,

    @classmethod
    def dist(cls, *args, **kwargs):
        dist = object.__new__(cls)
        dist.__init__(*args, **kwargs)
        return dist

    def __init__(self, shape, dtype, testval=None, defaults=(),
                 transform=None, broadcastable=None,initial_value=None):
        self.shape = np.atleast_1d(shape)
        if False in (np.floor(self.shape) == self.shape):
            raise TypeError("Expected int elements in shape")
        self.dtype = dtype

        ## CHANGED:
        #self.type = TensorType(self.dtype, self.shape, broadcastable)
        self.type = S.TensorVariableType(self.dtype,self.shape,broadcastable)

        self.testval = testval
        self.defaults = defaults
        self.transform = transform
        self.initial_value=initial_value; ### adding this to work with theano variables

    def default(self):
        return np.asarray(self.get_test_val(self.testval, self.defaults), self.dtype)

    def get_test_val(self, val, defaults):
        if val is None:
            for v in defaults:
                if hasattr(self, v) and np.all(np.isfinite(self.getattr_value(v))):
                    return self.getattr_value(v)
        else:
            return self.getattr_value(val)

        if val is None:
            raise AttributeError("%s has no finite default value to use, "
                                 "checked: %s. Pass testval argument or "
                                 "adjust so value is finite."
                                 % (self, str(defaults)))

    def getattr_value(self, val):
        if isinstance(val, string_types):
            val = getattr(self, val)

        ## CHANGED: Moved get-value-based-on-type logic to the backend
        #if isinstance(val, tt.TensorVariable):
        #    return val.tag.test_value

        #if isinstance(val, tt.TensorConstant):
            #return val.value
        #return val
        return S.get_val(val)


    def _repr_latex_(self, name=None, dist=None):
        """Magic method name for IPython to use for LaTeX formatting."""
        return None

    def logp_nojac(self, *args, **kwargs):
        """Return the logp, but do not include a jacobian term for transforms.

        If we use different parametrizations for the same distribution, we
        need to add the determinant of the jacobian of the transformation
        to make sure the densities still describe the same distribution.
        However, MAP estimates are not invariant with respect to the
        parametrization, we need to exclude the jacobian terms in this case.

        This function should be overwritten in base classes for transformed
        distributions.
        """
        return self.logp(*args, **kwargs)

    def logp_sum(self, *args, **kwargs):
        """Return the sum of the logp values for the given observations.

        Subclasses can use this to improve the speed of logp evaluations
        if only the sum of the logp values is needed.
        """
        ## CHANGED
        #return tt.sum(self.logp(*args, **kwargs))
        return S.tsum(self.logp(*args, **kwargs))

    __latex__ = _repr_latex_


#def TensorType(dtype, shape, broadcastable=None):
#    if broadcastable is None:
#        broadcastable = np.atleast_1d(shape) == 1
#    return tt.TensorType(str(dtype), broadcastable)


class NoDistribution(Distribution):

    def __init__(self, shape, dtype, testval=None, defaults=(),
                 transform=None, parent_dist=None, *args, **kwargs):
        super(NoDistribution, self).__init__(shape=shape, dtype=dtype,
                                             testval=testval, defaults=defaults,
                                             *args, **kwargs)
        self.parent_dist = parent_dist

    def __getattr__(self, name):
        # Do not use __getstate__ and __setstate__ from parent_dist
        # to avoid infinite recursion during unpickling
        if name.startswith('__'):
            raise AttributeError(
                "'NoDistribution' has no attribute '%s'" % name)
        return getattr(self.parent_dist, name)

    def logp(self, x):
        return 0


class Discrete(Distribution):
    """Base class for discrete distributions"""

    def __init__(self, shape=(), dtype=None, defaults=('mode',),
                 *args, **kwargs):
        if dtype is None:

            ## CHANGED: Just asking for float type from backend
            if S.floatx()=='float32':
            #if theano.config.floatX == 'float32':
                dtype = 'int16'
            else:
                dtype = 'int64'
        if dtype != 'int16' and dtype != 'int64':
            raise TypeError('Discrete classes expect dtype to be int16 or int64.')

        if kwargs.get('transform', None) is not None:
            raise ValueError("Transformations for discrete distributions "
                             "are not allowed.")

        super(Discrete, self).__init__(
            shape, dtype, defaults=defaults, *args, **kwargs)


class Continuous(Distribution):
    """Base class for continuous distributions"""

    def __init__(self, shape=(), dtype=None, defaults=('median', 'mean', 'mode'),
                 *args, **kwargs):
        if dtype is None:

            ## CHANGED: asking for float type from backed.
            #dtype = theano.config.floatX
            dtype = S.floatx()

        super(Continuous, self).__init__(
            shape, dtype, defaults=defaults, *args, **kwargs)


class DensityDist(Distribution):
    """Distribution based on a given log density function.

        A distribution with the passed log density function is created.
        Requires a custom random function passed as kwarg `random` to
        enable sampling.

        Example:
        --------
        .. code-block:: python
            with pm.Model():
                mu = pm.Normal('mu',0,1)
                normal_dist = pm.Normal.dist(mu, 1)
                pm.DensityDist('density_dist', normal_dist.logp, observed=np.random.randn(100), random=normal_dist.random)
                trace = pm.sample(100)

    """

    def __init__(self, logp, shape=(), dtype=None, testval=0, random=None, *args, **kwargs):
        if dtype is None:
            dtype = theano.config.floatX
        super(DensityDist, self).__init__(
            shape, dtype, testval, *args, **kwargs)
        self.logp = logp
        self.rand = random

    def random(self, *args, **kwargs):
        if self.rand is not None:
            return self.rand(*args, **kwargs)
        else:
            raise ValueError("Distribution was not passed any random method "
                            "Define a custom random method and pass it as kwarg random")



def draw_values(params, point=None):
    """
    Draw (fix) parameter values. Handles a number of cases:

        1) The parameter is a scalar
        2) The parameter is an *RV

            a) parameter can be fixed to the value in the point
            b) parameter can be fixed by sampling from the *RV
            c) parameter can be fixed using tag.test_value (last resort)

        3) The parameter is a tensor variable/constant. Can be evaluated using
        theano.function, but a variable may contain nodes which

            a) are named parameters in the point
            b) are *RVs with a random method

    """
    # Distribution parameters may be nodes which have named node-inputs
    # specified in the point. Need to find the node-inputs, their
    # parents and children to replace them.
    leaf_nodes = {}
    named_nodes_parents = {}
    named_nodes_children = {}
    for param in params:
        if hasattr(param, 'name'):
            # Get the named nodes under the `param` node
            nn, nnp, nnc = get_named_nodes_and_relations(param)
            leaf_nodes.update(nn)
            # Update the discovered parental relationships
            for k in nnp.keys():
                if k not in named_nodes_parents.keys():
                    named_nodes_parents[k] = nnp[k]
                else:
                    named_nodes_parents[k].update(nnp[k])
            # Update the discovered child relationships
            for k in nnc.keys():
                if k not in named_nodes_children.keys():
                    named_nodes_children[k] = nnc[k]
                else:
                    named_nodes_children[k].update(nnc[k])

    # Init givens and the stack of nodes to try to `_draw_value` from
    givens = {}
    stored = set([])  # Some nodes
    stack = list(leaf_nodes.values())  # A queue would be more appropriate
    while stack:
        next_ = stack.pop(0)
        if next_ in stored:
            # If the node already has a givens value, skip it
            continue
        elif is_shared(next_) or is_constant(next_):
        #elif isinstance(next_, (tt.TensorConstant,
        #                        tt.sharedvar.SharedVariable)):
            # If the node is a theano.tensor.TensorConstant or a
            # theano.tensor.sharedvar.SharedVariable, its value will be
            # available automatically in _compile_theano_function so
            # we can skip it. Furthermore, if this node was treated as a
            # TensorVariable that should be compiled by theano in
            # _compile_theano_function, it would raise a `TypeError:
            # ('Constants not allowed in param list', ...)` for
            # TensorConstant, and a `TypeError: Cannot use a shared
            # variable (...) as explicit input` for SharedVariable.
            stored.add(next_.name)
            continue
        else:
            # If the node does not have a givens value, try to draw it.
            # The named node's children givens values must also be taken
            # into account.
            children = named_nodes_children[next_]
            temp_givens = [givens[k] for k in givens.keys() if k in children]
            try:
                # This may fail for autotransformed RVs, which don't
                # have the random method
                givens[next_.name] = (next_, _draw_value(next_,
                                                         point=point,
                                                         givens=temp_givens))
                stored.add(next_.name)
            except: # XXX: this will accept more errors, so error prone theano.gof.fg.MissingInputError:
                # The node failed, so we must add the node's parents to
                # the stack of nodes to try to draw from. We exclude the
                # nodes in the `params` list.
                stack.extend([node for node in named_nodes_parents[next_]
                              if node is not None and
                              node.name not in stored and
                              node not in params])
    values = []
    for param in params:
        values.append(_draw_value(param, point=point, givens=givens.values()))
    return values


## CHANGED: This is a big change. All this logic is going into the backend.
# Unforunately, I don't think I could preserve the memoizing functionality in backend at the momemt.

# @memoize
# def _compile_theano_function(param, vars, givens=None):
#     """Compile theano function for a given parameter and input variables.
#
#     This function is memoized to avoid repeating costly theano compilations
#     when repeatedly drawing values, which is done when generating posterior
#     predictive samples.
#
#     Parameters
#     ----------
#     param : Model variable from which to draw value
#     vars : Children variables of `param`
#     givens : Variables to be replaced in the Theano graph
#
#     Returns
#     -------
#     A compiled theano function that takes the values of `vars` as input
#         positional args
#     """
#     return function(vars, param, givens=givens,
#                     rebuild_strict=True,
#                     on_unused_input='ignore',
#                     allow_input_downcast=True)


def _draw_value(param, point=None, givens=None):
    """Draw a random value from a distribution or return a constant.

    Parameters
    ----------
    param : number, array like, theano variable or pymc3 random variable
        The value or distribution. Constants or shared variables
        will be converted to an array and returned. Theano variables
        are evaluated. If `param` is a pymc3 random variables, draw
        a new value from it and return that, unless a value is specified
        in `point`.
    point : dict, optional
        A dictionary from pymc3 variable names to their values.
    givens : dict, optional
        A dictionary from theano variables to their values. These values
        are used to evaluate `param` if it is a theano variable.
    """
    if isinstance(param, numbers.Number):
        return param
    elif isinstance(param, np.ndarray):
        return param

    ## CHANGED: Asking for type from backend and getting Value
    # Probably a bit redundent with get_val
    #elif isinstance(param, tt.TensorConstant):
    #    return param.value
    #elif isinstance(param, tt.sharedvar.SharedVariable):
    #    return param.get_value()
    elif S.is_constant(param):
        S.get_val(param)
    elif S.is_shared(param):
        S.get_val(param)

    ## CHANGED: Asking for type from backend and getting Value
    #elif isinstance(param, tt.TensorVariable):
    elif S.is_variable(param):
        if point and hasattr(param, 'model') and param.name in point:
            return point[param.name]
        elif hasattr(param, 'random') and param.random is not None:
            return param.random(point=point, size=None)
        else:
            if givens:
                variables, values = list(zip(*givens))
            else:
                variables = values = []

            ## CHANGED: call the function class defined in the backend.
            #func = _compile_theano_function(param, variables)
            #return func(*values)
            func = S.function(param,variables)
            return(func(*values))

    else:
        raise ValueError('Unexpected type in draw_value: %s' % type(param))


def broadcast_shapes(*args):
    """Return the shape resulting from broadcasting multiple shapes.
    Represents numpy's broadcasting rules.

    Parameters
    ----------
    *args : array-like of int
        Tuples or arrays or lists representing the shapes of arrays to be broadcast.

    Returns
    -------
    Resulting shape or None if broadcasting is not possible.
    """
    x = list(np.atleast_1d(args[0])) if args else ()
    for arg in args[1:]:
        y = list(np.atleast_1d(arg))
        if len(x) < len(y):
            x, y = y, x
        x[-len(y):] = [j if i == 1 else i if j == 1 else i if i == j else 0
                       for i, j in zip(x[-len(y):], y)]
        if not all(x):
            return None
    return tuple(x)


def infer_shape(shape):
    try:
        shape = tuple(shape or ())
    except TypeError:  # If size is an int
        shape = tuple((shape,))
    except ValueError:  # If size is np.array
        shape = tuple(shape)
    return shape


def reshape_sampled(sampled, size, dist_shape):
    dist_shape = infer_shape(dist_shape)
    repeat_shape = infer_shape(size)

    if np.size(sampled) == 1 or repeat_shape or dist_shape:
        return np.reshape(sampled, repeat_shape + dist_shape)
    else:
        return sampled


def replicate_samples(generator, size, repeats, *args, **kwargs):
    n = int(np.prod(repeats))
    if n == 1:
        samples = generator(size=size, *args, **kwargs)
    else:
        samples = np.array([generator(size=size, *args, **kwargs)
                            for _ in range(n)])
        samples = np.reshape(samples, tuple(repeats) + tuple(size))
    return samples


def generate_samples(generator, *args, **kwargs):
    """Generate samples from the distribution of a random variable.

    Parameters
    ----------
    generator : function
        Function to generate the random samples. The function is
        expected take parameters for generating samples and
        a keyword argument `size` which determines the shape
        of the samples.
        The *args and **kwargs (stripped of the keywords below) will be
        passed to the generator function.

    keyword arguments
    ~~~~~~~~~~~~~~~~

    dist_shape : int or tuple of int
        The shape of the random variable (i.e., the shape attribute).
    size : int or tuple of int
        The required shape of the samples.
    broadcast_shape: tuple of int or None
        The shape resulting from the broadcasting of the parameters.
        If not specified it will be inferred from the shape of the
        parameters. This may be required when the parameter shape
        does not determine the shape of a single sample, for example,
        the shape of the probabilities in the Categorical distribution.

    Any remaining *args and **kwargs are passed on to the generator function.
    """
    dist_shape = kwargs.pop('dist_shape', ())
    size = kwargs.pop('size', None)
    broadcast_shape = kwargs.pop('broadcast_shape', None)
    params = args + tuple(kwargs.values())

    if broadcast_shape is None:
        broadcast_shape = broadcast_shapes(*[np.atleast_1d(p).shape for p in params
                                             if not isinstance(p, tuple)])
    if broadcast_shape == ():
        broadcast_shape = (1,)

    args = tuple(p[0] if isinstance(p, tuple) else p for p in args)
    for key in kwargs:
        p = kwargs[key]
        kwargs[key] = p[0] if isinstance(p, tuple) else p

    if np.all(dist_shape[-len(broadcast_shape):] == broadcast_shape):
        prefix_shape = tuple(dist_shape[:-len(broadcast_shape)])
    else:
        prefix_shape = tuple(dist_shape)

    repeat_shape = infer_shape(size)

    if broadcast_shape == (1,) and prefix_shape == ():
        if size is not None:
            samples = generator(size=size, *args, **kwargs)
        else:
            samples = generator(size=1, *args, **kwargs)
    else:
        if size is not None:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        repeat_shape + prefix_shape,
                                        *args, **kwargs)
        else:
            samples = replicate_samples(generator,
                                        broadcast_shape,
                                        prefix_shape,
                                        *args, **kwargs)
    return reshape_sampled(samples, size, dist_shape)
