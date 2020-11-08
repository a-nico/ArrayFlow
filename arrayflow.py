import numpy
from typing import List

is_training: bool = True
precision = numpy.float32


class Node:

    op: 'Operation' = None
    input_nodes: List['Node'] = []
    array: numpy.ndarray = None
    _partial_derivative: numpy.ndarray = None
    _is_reset = True

    def __init__(self, data):
        if not isinstance(data, numpy.ndarray):
            data = numpy.asarray(data, dtype=precision)
        if data.dtype != precision:
            data = data.astype(precision)
        self.array = data

    def _reset_parameter_derivatives(self):
        if not self._is_reset:
            self._partial_derivative.fill(0)
        for node in self.input_nodes:
            node._reset_parameter_derivatives()

    def compute_gradient(self):
        assert self.array.size == 1, "Gradient is only implemented for scalar fields."
        assert is_training
        if self.partial_derivative is None:
            self.partial_derivative = numpy.ones_like(self.array)
        self._reset_parameter_derivatives()
        self._autodiff()

    def _autodiff(self):
        if self.op is not None:
            dldx = self.op.differentiate(self)
            for k, pd in enumerate(dldx):
                self.input_nodes[k].partial_derivative = pd
        for node in self.input_nodes:
            node._autodiff()

    @property
    def partial_derivative(self) -> numpy.ndarray:
        return self._partial_derivative

    @partial_derivative.setter
    def partial_derivative(self, value: numpy.ndarray):
        self._partial_derivative = value

    @property
    def shape(self) -> tuple:
        return self.array.shape

    def __str__(self):
        classname = type(self).__name__
        arraystring = numpy.array2string(self.array, precision=4)
        return f"{classname}\n{arraystring}"

    # Operator Overloads
    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return times(self, other)

    def __rmul__(self, other):
        return times(other, self)

    def __matmul__(self, other):
        return matmul(self, other)

    def __rmatmul__(self, other):
        return matmul(other, self)

    def __truediv__(self, other):
        return div(self, other)

    def __rtruediv__(self, other):
        return div(other, self)

    def __pow__(self, other):
        return pow(self, other)

    def __rpow__(self, other):
        return pow(other, self)

    def __neg__(self):
        return subtract(0, self) # kind of hacky ¯\_(ツ)_/¯


class Constant(Node):

    @property
    def partial_derivative(self) -> None:
        return None

    @partial_derivative.setter
    def partial_derivative(self, value: numpy.ndarray):
        pass

class Parameter(Node):

    def __init__(self, data):
        assert is_training, "Training not enabled, use Constant."
        super().__init__(data)
        self._partial_derivative = numpy.zeros(self.array.shape, dtype=precision)

    @Node.partial_derivative.setter
    def partial_derivative(self, value: numpy.ndarray):
        self._is_reset = False
        if value.ndim == 1 + self._partial_derivative.ndim:
            value = numpy.sum(value, axis=-1)
        self._partial_derivative += value # accumulate


class Operation:

    @classmethod
    def evaluate(cls, *inputs) -> Node:
        inputs = list(inputs)
        for k, n in enumerate(inputs):
            if not isinstance(n, Node):
                inputs[k] = Constant(n)
        x = [n.array for n in inputs]
        y = cls._f(*x)
        output_node = Node(y)
        if is_training:
            output_node.op = cls
            output_node.input_nodes = inputs
        return output_node

    @classmethod
    def differentiate(cls, node: Node) -> List[numpy.ndarray]:
        dldy = node.partial_derivative
        x = [n.array for n in node.input_nodes]
        y = node.array
        return cls._df(dldy, y, *x)

    @staticmethod
    def _f(*x: numpy.ndarray) -> numpy.ndarray:
        raise NotImplementedError("Function not implemented.")

    @staticmethod
    def _df(dldy: numpy.ndarray, y: numpy.ndarray, *x: numpy.ndarray) -> List[numpy.ndarray]:
        raise NotImplementedError("Partial derivative not implemented. Autodiff cannot proceed on this branch.")


# Elementary Functions

class AbsoluteValue(Operation):

    @staticmethod
    def _f(x):
        return numpy.abs(x)

    @staticmethod
    def _df(dldy, y, x):
        gz = (x > 0)
        lz = numpy.logical_not(gz)
        return [dldy * gz - dldy * lz]


class Add(Operation):

    @staticmethod
    def _f(*x):
        return numpy.sum(x, axis=0)

    @staticmethod
    def _df(dldy, y, *x):
        return [dldy] * len(x)


class Concatenate(Operation):

    @staticmethod
    def _f(*x):
        return numpy.concatenate(x)

    @staticmethod
    def _df(dldy, y, *x):
        return numpy.split(dldy, numpy.cumsum([len(arr) for arr in x])[:-1])


class Cosine(Operation):

    @staticmethod
    def _f(x):
        return numpy.cos(x)

    @staticmethod
    def _df(dldy, y, x):
        return [-numpy.sin(x) * dldy]


class CrossCorrelate(Operation):

    @staticmethod
    def _f(s, k):
        return numpy.correlate(s, k)

    @staticmethod
    def _df(dldy, y, s, k):
        dlds = numpy.convolve(k, dldy, mode="full")
        dldk = numpy.correlate(dldy, s, mode="valid")
        return dlds, dldk


class Divide(Operation):

    @staticmethod
    def _f(a, b):
        return a / b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy / b
        dldb = -dldy * a / numpy.square(b)
        return dlda, dldb


class Exp(Operation):

    @staticmethod
    def _f(x):
        return numpy.exp(x)

    @staticmethod
    def _df(dydl, y, x):
        return [y * dydl]


class Expand(Operation):

    @staticmethod
    def _f(x):
        assert x.ndim == 1, "Expand is only for 1d arrays."
        return numpy.expand_dims(x, 1)

    @staticmethod
    def _df(dldy, y, x):
        return [numpy.squeeze(dldy, 1)]


class HyperbolicTangent(Operation):

    @staticmethod
    def _f(x):
        return numpy.tanh(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / numpy.square(numpy.cosh(x))]


class Logarithm(Operation):

    @staticmethod
    def _f(x):
        return numpy.log(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy / x]


class MatrixMultiply(Operation):

    @staticmethod
    def _f(a, b):
        y = numpy.matmul(a, b)
        if y.ndim == 1:
            y = numpy.expand_dims(y, 1)
        return y

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = numpy.matmul(dldy, b.T)
        dldb = numpy.matmul(a.T, dldy)
        return dlda, dldb


class Max(Operation):

    @staticmethod
    def _f(a, b):
        return numpy.maximum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a > b
        dlda = dldy * c
        dldb = dldy * numpy.logical_not(c)
        return dlda, dldb


class MaxPool(Operation):

    @staticmethod
    def _f(x, n):
        n = int(n)
        assert len(x) % n == 0, "Input length must be divisible by n."
        xs = numpy.reshape(x, (-1, n))
        return numpy.max(xs, 1)

    @staticmethod
    def _df(dldy, y, x, _):
        n = len(x) // len(dldy)
        xs = numpy.reshape(x, (-1, n))
        idx = numpy.argmax(xs, axis=1)
        points = idx + numpy.arange(0, len(idx)) * n
        dldx = numpy.zeros(x.shape)
        dldx[points] = dldy
        return [dldx]


class Mean(Operation):

    @staticmethod
    def _f(x):
        return numpy.mean(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * numpy.ones(x.shape) / x.size]


class Min(Operation):

    @staticmethod
    def _f(a, b):
        return numpy.minimum(a, b)

    @staticmethod
    def _df(dldy, y, a, b):
        c = a < b
        dlda = dldy * c
        dldb = dldy * numpy.logical_not(c)
        return dlda, dldb


class Multiply(Operation):

    @staticmethod
    def _f(a, b):
        return a * b

    @staticmethod
    def _df(dldy, y, a, b):
        dlda = dldy * b
        dldb = dldy * a
        return dlda, dldb


class Pow(Operation):

    @staticmethod
    def _f(x, n):
        return numpy.power(x, n)

    @staticmethod
    def _df(dldy, y, x, n):
        return [n * numpy.power(x, n - 1) * dldy]


class Slice(Operation):

    @staticmethod
    def _f(x, start, end):
        start = int(start)
        if numpy.isnan(end):
            return x[start:]
        return x[start: int(end)]

    @staticmethod
    def _df(dldy, y, x, start, end):
        dldx = numpy.zeros_like(x)
        start = int(start)
        if numpy.isnan(end):
            dldx[start:] = dldy
        else:
            dldx[start:int(end)] = dldy
        return [dldx]


class Sine(Operation):

    @staticmethod
    def _f(x):
        return numpy.sin(x)

    @staticmethod
    def _df(dldy, y, x):
        return [numpy.cos(x) * dldy]


class Subtract(Operation):

    @staticmethod
    def _f(a, b):
        return a - b

    @staticmethod
    def _df(dldy, y, a, b):
        return dldy, -dldy


class Sum(Operation):

    @staticmethod
    def _f(x):
        return numpy.sum(x)

    @staticmethod
    def _df(dldy, y, x):
        return [dldy * numpy.ones(x.shape)]


class SquareRoot(Operation):

    @staticmethod
    def _f(x):
        return numpy.sqrt(x)

    @staticmethod
    def _df(dldy, y, x):
        return [.5 * dldy / y]


# Shorthand functions

def abs(x):
    '''Element-wise absolute value'''
    return AbsoluteValue.evaluate(x)

def add(*elements):
    '''Element-wise sum of all elements.'''
    return Add.evaluate(*elements)

def ccor(s, k):
    '''Cross correlation of signal s with kernel k.'''
    return CrossCorrelate.evaluate(s, k)

def concat(*x):
    '''Concate along the first axis'''
    return Concatenate.evaluate(*x)

def cos(x):
    '''Element-wise cosine.'''
    return Cosine.evaluate(x)

def div(a, b):
    '''Element-wise division a / b'''
    return Divide.evaluate(a, b)

def exp(x):
    '''Element-wise exponential.'''
    return Exp.evaluate(x)

def expand(x):
    '''Converts 1 dim array into (n x 1) arrays'''
    return Expand.evaluate(x)

def log(x):
    '''Element-wise natural logarithm'''
    return Logarithm.evaluate(x)

def matmul(a, b):
    '''Matrix multiplication a * b.'''
    return MatrixMultiply.evaluate(a, b)

def max(a, b):
    '''Element-wise maximum.'''
    return Max.evaluate(a, b)

def maxpool(x, n):
    '''1D Max Pooling with span n. Length of x must be divisible by n.'''
    return MaxPool.evaluate(x, n)

def mean(x):
    '''Mean of all elements'''
    return Mean.evaluate(x)

def min(a, b):
    '''Element-wise minimum.'''
    return Min.evaluate(a, b)

def pow(x, n):
    '''Element-wise x raised to the power n.'''
    return Pow.evaluate(x, n)

def slice(x, start, end=None):
    '''Returns x[start: end]. If end is None, it returns x[start:].'''
    return Slice.evaluate(x, start, end)

def sin(x):
    '''Element-wise sine.'''
    return Sine.evaluate(x)

def subtract(a, b):
    '''Element-wise a minus b.'''
    return Subtract.evaluate(a, b)

def sum(x):
    '''Sum of the elements.'''
    return Sum.evaluate(x)

def sqrt(x):
    '''Element-wise square root.'''
    return SquareRoot.evaluate(x)

def tanh(x):
    '''Element-wise hyperbolic tangent.'''
    return HyperbolicTangent.evaluate(x)

def times(a, b):
    '''Element-wise product of a and b.'''
    return Multiply.evaluate(a, b)


class Optimizer:

    def __init__(self, initial_stepsize=1E-5, fixed_step=False, beta=0.05, moving_avg_length=10):
        self.beta = beta
        self.stepsize = initial_stepsize
        self.moving_avg_length = moving_avg_length
        self.bd_2nd = numpy.asarray([-1, 4, -5, 2], dtype=precision) # 2nd derivative backward difference coefficients, O(h^2)
        self.bd_1st = numpy.asarray([-1/3, 3/2, -3, 11/6], dtype=precision) # 1st derivative backward difference coefficients, O(h^3)
        self.parameters, self.grads, self.lt, self.lt_ = None, None, None, None
        self._growth_phase = True
        self.fixed_step = fixed_step

    def update(self, l: Node):
        if self.lt  is None:
            self.lt = numpy.ones(self.moving_avg_length, dtype=precision) * l.array
            self.lt_= numpy.ones(len(self.bd_2nd), dtype=precision) * l.array
        if self.parameters is None:
            self.parameters = set()
            self.find_parameters(l, self.parameters)
        if not self.fixed_step:
            self.lt = self.push(self.lt, l.array)
            self.lt_ = self.push(self.lt_, self.lt.mean())
            d2lt_ = numpy.sum(self.bd_2nd * self.lt_)
            dlt_ = numpy.sum(self.bd_1st * self.lt_)
            if d2lt_ > 0 and dlt_ > 0:
                self.stepsize *= .99
                if self._growth_phase:
                    print("Step size capped at:", self.stepsize)
                    self._growth_phase = False
            elif d2lt_ < 0 and self._growth_phase:
                self.stepsize *= 1.02
        l.compute_gradient()
        if self.grads is None:
            self.grads = {x: x.partial_derivative for x in self.parameters}
        for p in self.parameters:
            g = p.partial_derivative * self.beta + (1 - self.beta) * self.grads[p]
            p.array -= self.stepsize * g
            self.grads[p] = g

    def find_parameters(self, l: Node, params: set):
        for x in l.input_nodes:
            if type(x) is Parameter:
                params.add(x)
            elif type(x) is not Constant:
                self.find_parameters(x, params)

    def push(self, arr, value):
        x = numpy.roll(arr, -1)
        x[-1] = value
        return x

