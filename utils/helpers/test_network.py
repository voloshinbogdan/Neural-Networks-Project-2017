from net import *
from cs231n.solver import *
import layers

if __name__ == "__main__":
    # Instantiation example
    i1 = layers.Input()
    c1 = layers.Conv((8, 3, 3, 3), {'stride': 1, 'pad': 1}, i1)
    flat = 8 * 28 * 28
    s1 = layers.Reshape((flat,), c1)
    a1 = layers.Affine((flat, 10), s1)
    l1 = layers.SoftmaxLoss('y', 'loss', a1)
    try:
        layers.load_network('network')
    except IOError:
        pass
    model = NeuralNetwork(i1, l1, 'loss', layers.params, layers.grads)
    print("Starting forward...")
    x = np.ones((16, 3, 28, 28)) * 0.1
    y = np.ones(16, dtype=np.int) * 2
    res = model.loss(x)
    print('--scores--')
    print(res)

    print("\nStarting backward...")
    loss, grads = model.loss(x, y)
    print('--loss--')
    print(loss)
    print('--grads--')
    print(grads)
    layers.save_network('network')

