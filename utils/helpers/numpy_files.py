import numpy as np
import pickle

# save / load helpers

# Protocol version 0 is the original ASCII protocol and is backwards compatible
# with earlier versions of Python.
# Protocol version 1 is the old binary format which is also compatible with
# earlier versions of Python.
# Protocol version 2 was introduced in Python 2.3. It provides much more
# efficient pickling of new-style classes.


def save_obj(obj, name, protocol=0):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, protocol)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":

    # create data
    arr = np.array([1.3, 2., 3.])
    dict = {'arr1': arr}

    # write data to file
    fname = 'dict1'
    save_obj(dict, fname)

    # load saved data
    loadedDict = load_obj(fname)

    # print loaded data
    print(loadedDict)
