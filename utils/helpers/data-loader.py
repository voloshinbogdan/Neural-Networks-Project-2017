import numpy as np


def unpickle(file):
    import _pickle as cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo, encoding='bytes')
    return dict


def data_to_img(data):
    return np.stack(np.split(data, 3, axis=1), axis=2).reshape(-1, 32, 32, 3)


meta = unpickle('./cifar-10-batches-py/batches.meta')
batch = [unpickle('./cifar-10-batches-py/data_batch_{}'.format(i))
         for i in range(1, 6)]
test_batch = unpickle('./cifar-10-batches-py/test_batch')


def get_label_name(l):
    return meta[b'label_names'][l]


data = [batch[i][b'data'] for i in range(0, 5)]
data = np.concatenate(data)
labels = [batch[i][b'labels'] for i in range(0, 5)]
labels = np.concatenate(labels)

test_data = test_batch[b'data']
test_labels = test_batch[b'labels']
