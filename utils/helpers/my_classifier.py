import argparse

import layers
import net
import data_loader
from cs231n.solver import *


def conv_bn_conv_bn_pool2x2(inp_layer, conv_filters, conv_shapes,
                            res_shape, training_name):
    assert conv_shapes[0][1] == conv_shapes[0][2]
    pad1 = conv_shapes[0][1] // 2
    conv1 = layers.Conv((conv_filters[0],) + conv_shapes[0],
                        {'stride': 1, 'pad': pad1}, inp_layer)
    conv1 = layers.SpatialBatchnorm((conv_filters[0],) + res_shape,
                                    training_name, conv1)
    conv1 = layers.Relu(conv1)
    conv1 = layers.Dropout(0.6, training_name, conv1)

    assert conv_shapes[1][0] == conv_shapes[1][1]
    pad2 = conv_shapes[1][1] // 2
    conv2 = layers.Conv((conv_filters[1], conv_filters[0]) + conv_shapes[1],
                        {'stride': 1, 'pad': pad2}, conv1)
    conv2 = layers.SpatialBatchnorm((conv_filters[1],) + res_shape,
                                    training_name, conv2)
    conv2 = layers.Relu(conv2)
    conv2 = layers.Dropout(0.6, training_name, conv2)

    pool = layers.MaxPool((2, 2), 2, conv2)

    return pool


def fc_bn_dropout(inp_layer, size, training_name):
    fc = layers.Affine(size, inp_layer)
    fc = layers.Batchnorm(size[1], training_name, fc)
    fc = layers.Relu(fc)
    fc = layers.Dropout(0.8, training_name, fc)

    return fc


def my_conv_net(n_classes):

    # initialization
    training_name = 'mode'
    loss_name = 'loss'
    ground_truth = 'y'

    input_layer = layers.Input()
    inp = layers.L2Norm(0.01, loss_name, input_layer)

    # Convoluton layers
    conv_count = 3
    res_shape = [(32, 32), (16, 16), (8, 8)]
    conv_filters = [(32, 32), (128, 128), (256, 512)]
    conv_shapes = [((3, 3, 3), (3, 3)), ((32, 3, 3), (3, 3)), ((128, 3, 3), (3, 3))]
    for i in range(0, conv_count):
        inp = conv_bn_conv_bn_pool2x2(
            inp, conv_filters[i], conv_shapes[i], res_shape[i],
            training_name)

    flat = 4 * 4 * 512
    inp = layers.Reshape((flat,), inp)

    # Fully-connected layers
    fc_count = 2
    fc_sizes = [(flat, 2048), (2048, 256)]
    for i in range(0, fc_count):
        inp = fc_bn_dropout(inp, fc_sizes[i], training_name)

    # Last fc layer
    y = layers.Affine((fc_sizes[-1][-1], n_classes), inp)

    loss = layers.SoftmaxLoss(ground_truth, loss_name, y)

    model = net.NeuralNetwork(input_layer, loss, loss_name, ground_truth,
                              training_name, layers.params, layers.grads)

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run learning of my network')
    parser.add_argument('--load_checkpoint', dest='load_checkpoint', type=str,
                        default=None, help='path to checkpoint to load')
    parser.add_argument('--num_epochs', dest='num_epochs', type=int,
                        default=10, help='epochs count')
    args = parser.parse_args()
    load_checkpoint = args.load_checkpoint
    data = data_loader.data_to_img(data_loader.data).transpose((0, 3, 1, 2))
    labels = data_loader.labels
    val_data = data_loader.data_to_img(data_loader.test_data).transpose((0, 3, 1, 2))
    val_labels = data_loader.test_labels
    model = my_conv_net(10)
    data = {
      'X_train': data,
      'y_train': labels,
      'X_val': val_data,
      'y_val': val_labels
    }
    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                      'learning_rate': 5e-4,
                    },
                    lr_decay=0.95,
                    num_epochs=args.num_epochs,
                    batch_size=16,
                    print_every=10,
                    checkpoint_name='checkpoints/new')
    if load_checkpoint is not None:
        model = solver.load_checkpoint(load_checkpoint)
        layers.params = model.params
        layers.grads = model.grads
        solver.optim_config['learning_rate'] = 5e-4
    solver.train()
