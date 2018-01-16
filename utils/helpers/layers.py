from cs231n.fast_layers import conv_forward_fast
from cs231n.fast_layers import conv_backward_fast
from cs231n.fast_layers import max_pool_forward_fast
from cs231n.fast_layers import max_pool_backward_fast
from cs231n.layers import *

# Base class for all layers
empty_name_number = 0
params = {}
grads = {}


def get_new_name():
    empty_name_number += 1
    return 'empty-name#{}'.format(empty_name_number)


class Layer(object):
    def __init__(self, name, prev=None):
        if name is None:
            name = get_new_name()
        assert name not in params
        params[name] = True

        self.next = None
        self.initPrev(prev)
        self.name = name

    def initPrev(self, prev):
        self.prev = prev
        if not (prev is None):
            self.prev.setNext(self)

    def setNext(self, next):
        self.next = next

    def forward(self, x):
        (res, self.cache) = self._forward(x)
        if self.next is None:
            return res
        return self.next.forward(res)

    def _forward(self, x):
        print("[{}]: \tCalculating forward using '{}'".format(self.name, x))
        return ("Forward result of " + self.name, "Forward cache of " +
                self.name)

    def backward(self, x):
        res = self._backward(x, self.cache)
        if self.prev is None:
            return
        self.prev.backward(res)

    def _backward(self, x, cache):
        print("[{}]: \tCalculating backward using '{}' and '{}'"
              .format(self.name, x, cache))
        return "Backward result of " + self.name

    def n(val):
        return self.name + '/' + val


# Implementation stub
class Conv(Layer):
    """
    Inputs:
    - size: (fiters count, previous filters count, filter width, filter height)
    """
    def __init__(self, size, conv_param, prev, weight_scale=1e-3, name=None):
        super(Conv, self).__init__(name, prev)
        params[n('w')] = weight_scale * np.random.normal(size=size)
        params[n('b')] = bp.zeros(size[0])
        self.conv_param = conv_param

    def _forward(self, x):
        return conv_forward_fast(x, params[n('w')], params[n('b')],
                                 self.conv_param)

    def _backward(self, x, cache):
        dx, grads[n('w')], grads[n('b')] = conv_backward_fast()
        return dx


class Reshape(Layer):
    def __init__(self, new_shape, prev, name=None):
        super(Reshape, self).__init__(name, prev)
        self.new_shape = new_shape

    def _forward(self, x):
        return (x.reshape(new_shape), x.shape)

    def _backward(self, x, cache):
        dx = x.reshape(cache)
        return dx


class Affine(Layer):
    def __init__(self, size, prev, weight_scale=1e-3, name=None):
        super(Affine, self).__init__(name, prev)
        params[n('w')] = weight_scale * np.random.normal(size=size)
        params[n('b')] = bp.zeros(size[-1])

    def _forward(self, x):
        return affine_forward(x, params[n('w')], params[n('b')])

    def _backward(self, cache, x):
        dx, grads[n('w')], grads[n('b')] = affine_backward(x, cache)
        return dx


class Input(Layer):
    def __init__(self, name=None):
        super(Input, self).__init__(name)

    def _forward(self, x):
        assert type(x) == dict
        for n in x:
            if n != '#input':
                params[n] = x[n]
        return (x['#input'], None)

    def _backward(self, cache, x):
        return x


class SoftmaxLoss(Layer):
    def __init__(self, ground_truth, prev, name=None, loss_name=None):
        super(SoftmaxLoss, self).__init__(name, prev)
        if prev is not None:
            prev.next = None
            self.prev = prev.prev
            prev.prev = self
        if loss_name is None:
            loss_name = n('loss')
        self.cache = None
        self.ground_truth = ground_truth

    def _forward(self, x):
       return (None, None) 

    def _backward(self, cache, x)
        params[loss_name], dx = softmax_loss(x, params[ground_truth])


class Relu(Layer):
    def __init__(self, prev, name=None):
        super(Relu, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class Batchnorm(Layer):
    def __init__(self, prev, name=None):
        super(Batchnorm, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class Dropout(Layer):
    def __init__(self, prev, name=None):
        super(Dropout, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class MaxPool(Layer):
    def __init__(self, prev, name=None):
        super(MaxPool, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class SpatialBatchnorm(Layer):
    def __init__(self, prev, name=None):
        super(SpatialBatchnorm, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class SvmLoss(Layer):
    def __init__(self, prev, name=None):
        super(SvmLoss, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


class L2Norm(Layer):
    def __init__(self, prev, name=None):
        super(L2Norm, self).__init__(name, prev)

    def _forward(self, x):

    def _backward(self, cache, x):


if __name__ == "__main__":
    # Instantiation example
    i1 = Input()
    c1 = Conv(i1)
    s1 = Reshape(c1)
    a1 = Affine(s1)
    l1 = SoftmaxLoss(a1)
    print("Starting forward...")
    i1.forward("input")

    print("\nStarting backward...")
    l1.backward(l1.cache)
