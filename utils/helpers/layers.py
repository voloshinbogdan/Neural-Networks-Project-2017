from cs231n.fast_layers import conv_forward_fast
from cs231n.fast_layers import conv_backward_fast
from cs231n.fast_layers import max_pool_forward_fast
from cs231n.fast_layers import max_pool_backward_fast
from cs231n.layers import *
import numpy_files as nf

# Base class for all layers
empty_name_number = 0
params = {}
grads = {}


def get_new_name():
    global empty_name_number
    empty_name_number += 1
    return 'empty-name#{}'.format(empty_name_number)


def save_network(name):
    nf.save_obj(params, name)


def load_network(name):
    global params
    params = nf.load_obj(name)


class Layer(object):
    def __init__(self, name, prev=None):
        global params
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

    def n(self, val):
        return self.name + '/' + val


# Implementation stub
class Conv(Layer):
    """
    Inputs:
    - size: (fiters count, previous filters count, filter width, filter height)
    - conv_param: A dictionary with the following keys:
        - 'stride': The number of pixels between adjacent receptive fields in
                    the horizontal and vertical directions.
        - 'pad': The number of pixels that will be used to zero-pad the input.

    """
    def __init__(self, size, conv_param, prev, weight_scale=1e-3, name=None):
        super(Conv, self).__init__(name, prev)
        global params
        params[self.n('w')] = weight_scale * np.random.normal(size=size)
        params[self.n('b')] = np.zeros(size[0])
        self.conv_param = conv_param

    def _forward(self, x):
        global params
        return conv_forward_fast(x, params[self.n('w')], params[self.n('b')],
                                 self.conv_param)

    def _backward(self, x, cache):
        global grads
        dx, grads[self.n('w')], grads[self.n('b')] = \
            conv_backward_fast(x, cache)
        return dx


class Reshape(Layer):
    def __init__(self, new_shape, prev, name=None):
        super(Reshape, self).__init__(name, prev)
        self.new_shape = new_shape

    def _forward(self, x):
        shape = x.shape
        return (x.reshape((shape[0],) + self.new_shape), shape)

    def _backward(self, x, cache):
        dx = x.reshape(cache)
        return dx


class Affine(Layer):
    def __init__(self, size, prev, weight_scale=1e-3, name=None):
        super(Affine, self).__init__(name, prev)
        global params
        params[self.n('w')] = weight_scale * np.random.normal(size=size)
        params[self.n('b')] = np.zeros(size[-1])

    def _forward(self, x):
        global params
        return affine_forward(x, params[self.n('w')], params[self.n('b')])

    def _backward(self, x, cache):
        global grads
        dx, grads[self.n('w')], grads[self.n('b')] = affine_backward(x, cache)
        return dx


class Input(Layer):
    def __init__(self, name=None):
        super(Input, self).__init__(name)

    def _forward(self, x):
        assert type(x) == dict
        global params
        for n in x:
            if n != '#input':
                params[n] = x[n]
        return (x['#input'], None)

    def _backward(self, x, cache):
        return x


class SoftmaxLoss(Layer):
    def __init__(self, ground_truth, loss_name, prev, name=None):
        super(SoftmaxLoss, self).__init__(name, prev)
        self.loss_name = loss_name
        self.ground_truth = ground_truth

    def _forward(self, x):
        return (x, x)

    def _backward(self, cache, x):
        global params
        loss, dx = softmax_loss(x, params[self.ground_truth])
        params[self.loss_name] += loss
        return dx


class Relu(Layer):
    def __init__(self, prev, name=None):
        super(Relu, self).__init__(name, prev)

    def _forward(self, x):
        return relu_forward(x)

    def _backward(self, x, cache):
        return relu_backward(x, cache)


class Batchnorm(Layer):
    def __init__(self, size, eps, momentum, mode_name, prev, name=None):
        super(Batchnorm, self).__init__(name, prev)
        global params
        params[self.n('running_mean')] = np.zeros(size)
        params[self.n('running_var')] = np.zeros(size)
        params[self.n('gamma')] = np.ones(size)
        params[self.n('beta')] = np.zeros(size)
        self.eps = eps
        self.momentum = momentum
        self.mode_name = mode_name

    def _forward(self, x):
        global params
        bn_params = {
            'mode': params[self.mode_name],
            'eps': self.eps,
            'momentum': self.momentum,
            'running_mean': params[self.n('running_mean')],
            'running_var': params[self.n('running_var')]}

        res = batchnorm_forward(x, params[self.n('gamma')],
                                params[self.n('beta')], bn_params)
        params[self.n('running_mean')] = bn_params['running_mean']
        params[self.n('running_var')] = bn_params['running_var']

        return res

    def _backward(self, x, cache):
        global grads
        dx, grads[self.n('gamma')], grads[self.n('beta')] = \
            batchnorm_backward(x, cache)
        return dx


class Dropout(Layer):
    def __init__(self, p, mode_name, prev, name=None):
        super(Dropout, self).__init__(name, prev)
        self.mode_name = mode_name
        self.p = p

    def _forward(self, x):
        global params
        return dropout_froward(x, {'p': self.p, 'mode': params['mode_name']})

    def _backward(self, x, cache):
        return dropout_backward(x, cache)


class MaxPool(Layer):
    """
    Inputs:
    - size: (Height, Width)
    """
    def __init__(self, size, stride, prev, name=None):
        super(MaxPool, self).__init__(name, prev)
        self.size = size
        self.stride = stride

    def _forward(self, x):
        return max_pool_forward_fast(
            x, {'pool_height': self.size[0], 'pool_width': self.size[1],
                'stride': self.stride})

    def _backward(self, x, cache):
        return max_pool_backward_fast(x, cache)


class SpatialBatchnorm(Layer):
    def __init__(self, size, eps, momentum, mode_name, prev, name=None):
        super(SpatialBatchnorm, self).__init__(name, prev)
        global params
        params[self.n('running_mean')] = np.zeros(size)
        params[self.n('running_var')] = np.zeros(size)
        params[self.n('gamma')] = np.ones(size)
        params[self.n('beta')] = np.zeros(size)
        self.eps = eps
        self.momentum = momentum
        self.mode_name = mode_name

    def _forward(self, x):
        bn_params = {
            'mode': params[self.mode_name],
            'eps': self.eps,
            'momentum': self.momentum,
            'running_mean': params[self.n('running_mean')],
            'running_var': params[self.n('running_var')]}

        res = spatial_batchnorm_forward(x, params[self.n('gamma')],
                                        params[self.n('beta')], bn_params)
        params[self.n('running_mean')] = bn_params['running_mean']
        params[self.n('running_var')] = bn_params['running_var']

        return res

    def _backward(self, x, cache):
        global grads
        dx, grads[self.n('gamma')], grads[self.n('beta')] = \
            batchnorm_backward(x, cache)
        return dx


class SvmLoss(Layer):
    def __init__(self, ground_truth, loss_name, prev, name=None):
        super(SvmLoss, self).__init__(name, prev)
        self.ground_truth = ground_truth
        self.loss_name = loss_name

    def _forward(self, x):
        return (x, x)

    def _backward(self, x, cache):
        global params
        loss, dx = svm_loss(x, params[self.ground_truth])
        params[self.loss_name] += loss
        return dx


class L2Norm(Layer):
    def __init__(self, reg, loss_name, prev, name=None):
        super(L2Norm, self).__init__(name, prev)
        self.reg = reg
        self.loss_name = loss_name

    def _forward(self, x):
        return (x, None)

    def _backward(self, x, cache):
        global params
        global grads
        keys = np.array(list(params.keys()))

        for key in keys[np.char.endswith(keys, '/w')]:
            grads[key] += self.reg * params[key]
            params[self.loss_name] += \
                0.5 * self.reg * np.sum(params[key] * params[key])

        return x


if __name__ == "__main__":
    # Instantiation example
    i1 = Input()
    c1 = Conv((8, 3, 3, 3), {'stride': 1, 'pad': 1}, i1)
    flat = 8 * 28 * 28
    s1 = Reshape((flat,), c1)
    a1 = Affine((flat, 10), s1)
    l1 = SoftmaxLoss('y', 'loss', a1)
    try:
        load_network('network')
    except IOError:
        pass
    print("Starting forward...")
    x = {
        '#input': np.ones((16, 3, 28, 28)) * 0.1,
        'y': np.zeros(16, dtype=np.int),
        'loss': 0
    }
    res = i1.forward(x)
    print(res)

    print("\nStarting backward...")
    l1.backward(l1.cache)
    save_network('network')
