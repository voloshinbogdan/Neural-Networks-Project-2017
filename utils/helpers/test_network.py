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

    data = {
      'X_train': np.ones((2**10, 3, 28, 28)) * 0.1,
      'y_train': np.ones(2**10, dtype=np.int) * 2,
      'X_val': np.ones((2**3, 3, 28, 28)) * 0.1,
      'y_val': np.ones(2**3, dtype=np.int) * 2
    }
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=64,
                    print_every=10)
    solver.train()
    layers.save_network('network')
