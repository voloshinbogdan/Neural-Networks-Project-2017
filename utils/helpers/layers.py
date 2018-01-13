# Base class for all layers
class Layer(object):
    def __init__(self, name, prev=None):
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
            return
        self.next.forward(res)

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


# Implementation stub
class Conv(Layer):
    def __init__(self, prev):
        super(Conv, self).__init__("Conv", prev)


class Reshape(Layer):
    def __init__(self, prev):
        super(Reshape, self).__init__("Reshape", prev)


class Affine(Layer):
    def __init__(self, prev):
        super(Affine, self).__init__("Affine", prev)


class Input(Layer):
    def __init__(self):
        super(Input, self).__init__("Input")


class Loss(Layer):
    def __init__(self, prev):
        super(Loss, self).__init__("Loss", prev)


if __name__ == "__main__":
    # Instantiation example
    i1 = Input()
    c1 = Conv(i1)
    s1 = Reshape(c1)
    a1 = Affine(s1)
    l1 = Loss(a1)
    print("Starting forward...")
    i1.forward("input")

    print("\nStarting backward...")
    l1.backward(l1.cache)
