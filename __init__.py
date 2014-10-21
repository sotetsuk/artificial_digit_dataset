import gzip
import cPickle
import numpy as np


def load_data(digits, shuffle=True):
    """
    load Digital DIgits Data set (d3set)

    Args
    ------
    digits: list ex. [0, 1, 3, 5]
    shuffle: bool

    Return
    ---------
    datasets: tupled data (train_set, valid_set, test_set)

    train_set: (train_set_x, train_set_y)
    valid_set: (valid_set_x, valid_set_y)
    test_set: (test_set_x, test_set_y)

    Example
    -----------
    >>> datasets = load_data(digits=[0, 5, 8])
    >>> train_set, valid_set, test_set = datasets
    >>> train_set_x, train_set_y = train_set
    ... valid_set_x, valid_set_y = valid_set
    ... test_set_x, test_set_y = test_set
    >>> train_set_x.shape
    (15000, 724)

    """

    for d in digits:
        assert d in range(10)

    path_to_d3set = './d3set.pkl.gz'
    seed = 1234
    dim = 28
    n_class = 10
    n_train = 50000
    n_valid = 10000
    n_test = 10000

    with gzip.open(path_to_d3set, 'rb') as gf:
        train_set, valid_set, test_set = cPickle.load(gf)

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    train_set_x_ret = np.zeros((len(digits)*n_train/n_class, dim * dim))
    train_set_y_ret = np.zeros(len(digits)*n_train/n_class)
    for i, d in enumerate(digits):
            train_set_x_ret[i*n_train/n_class:(i+1)*n_train/n_class, :] = train_set_x[d*n_train/n_class:(d+1)*n_train/n_class, :]
            train_set_y_ret[i*n_train/n_class:(i+1)*n_train/n_class] = train_set_y[d*n_train/n_class:(d+1)*n_train/n_class]

    valid_set_x_ret = np.zeros((len(digits)*n_valid/n_class, dim*dim))
    valid_set_y_ret = np.zeros(len(digits)*n_valid/n_class)
    for i, d in enumerate(digits):
            valid_set_x_ret[i*n_valid/n_class:(i+1)*n_valid/n_class, :] = valid_set_x[d*n_valid/n_class:(d+1)*n_valid/n_class, :]
            valid_set_y_ret[i*n_valid/n_class:(i+1)*n_valid/n_class] = valid_set_y[d*n_valid/n_class:(d+1)*n_valid/n_class]

    test_set_x_ret = np.zeros((len(digits)*n_test/n_class, dim*dim))
    test_set_y_ret = np.zeros(len(digits)*n_test/n_class)
    for i, d in enumerate(digits):
            test_set_x_ret[i*n_test/n_class:(i+1)*n_test/n_class, :] = test_set_x[d*n_test/n_class:(d+1)*n_test/n_class, :]
            test_set_y_ret[i*n_test/n_class:(i+1)*n_test/n_class] = test_set_y[d*n_test/n_class:(d+1)*n_test/n_class]

    if shuffle:
        train_shuffled = np.c_[train_set_x_ret, train_set_y_ret]
        valid_shuffled = np.c_[valid_set_x_ret, valid_set_y_ret]
        test_shuffled = np.c_[test_set_x_ret, test_set_y_ret]

        np.random.seed(seed)

        np.random.shuffle(train_shuffled)
        np.random.shuffle(valid_shuffled)
        np.random.shuffle(test_shuffled)

        train_set_x_ret, train_set_y_ret = train_shuffled[:, :dim*dim], train_shuffled[:, dim*dim]
        valid_set_x_ret, valid_set_y_ret = valid_shuffled[:, :dim*dim], valid_shuffled[:, dim*dim]
        test_set_x_ret, test_set_y_ret = test_shuffled[:, :dim*dim], test_shuffled[:, dim*dim]

    return ((train_set_x_ret, train_set_y_ret),
            (valid_set_x_ret, valid_set_y_ret),
            (test_set_x_ret, test_set_y_ret))


def load_img(digits, shuffle=True):
    """
    load Digital DIgits Data set (d3set)

    Args
    ------
    digits: list ex. [0, 1, 3, 5]
    shuffle: bool

    Return
    ---------
    datasets: tupled data (train_set, valid_set, test_set)

    train_set: (train_set_x, train_set_y)
    valid_set: (valid_set_x, valid_set_y)
    test_set: (test_set_x, test_set_y)

    Example
    -----------
    >>> datasets = load_data(digits=[0, 5, 8])
    >>> train_set, valid_set, test_set = datasets
    >>> train_set_x, train_set_y = train_set
    ... valid_set_x, valid_set_y = valid_set
    ... test_set_x, test_set_y = test_set
    >>> train_set_x.shape
    (15000, 28, 28)

    """

    for d in digits:
        assert d in range(10)

    path_to_d3set = './d3set.pkl.gz'
    seed = 1234
    dim = 28
    n_class = 10
    n_train = 50000
    n_valid = 10000
    n_test = 10000

    with gzip.open(path_to_d3set, 'rb') as gf:
        train_set, valid_set, test_set = cPickle.load(gf)

    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    train_set_x_ret = np.zeros((len(digits)*n_train/n_class, dim*dim))
    train_set_y_ret = np.zeros(len(digits)*n_train/n_class)
    for i, d in enumerate(digits):
            train_set_x_ret[i*n_train/n_class:(i+1)*n_train/n_class, :] = train_set_x[d*n_train/n_class:(d+1)*n_train/n_class, :]
            train_set_y_ret[i*n_train/n_class:(i+1)*n_train/n_class] = train_set_y[d*n_train/n_class:(d+1)*n_train/n_class]

    valid_set_x_ret = np.zeros((len(digits)*n_valid/n_class, dim*dim))
    valid_set_y_ret = np.zeros(len(digits)*n_valid/n_class)
    for i, d in enumerate(digits):
            valid_set_x_ret[i*n_valid/n_class:(i+1)*n_valid/n_class, :] = valid_set_x[d*n_valid/n_class:(d+1)*n_valid/n_class, :]
            valid_set_y_ret[i*n_valid/n_class:(i+1)*n_valid/n_class] = valid_set_y[d*n_valid/n_class:(d+1)*n_valid/n_class]

    test_set_x_ret = np.zeros((len(digits)*n_test/n_class, dim*dim))
    test_set_y_ret = np.zeros(len(digits)*n_test/n_class)
    for i, d in enumerate(digits):
            test_set_x_ret[i*n_test/n_class:(i+1)*n_test/n_class, :] = test_set_x[d*n_test/n_class:(d+1)*n_test/n_class, :]
            test_set_y_ret[i*n_test/n_class:(i+1)*n_test/n_class] = test_set_y[d*n_test/n_class:(d+1)*n_test/n_class]

    if shuffle:
        train_shuffled = np.c_[train_set_x_ret, train_set_y_ret]
        valid_shuffled = np.c_[valid_set_x_ret, valid_set_y_ret]
        test_shuffled = np.c_[test_set_x_ret, test_set_y_ret]

        np.random.seed(seed)

        np.random.shuffle(train_shuffled)
        np.random.shuffle(valid_shuffled)
        np.random.shuffle(test_shuffled)

        train_set_x_ret, train_set_y_ret = train_shuffled[:, :dim*dim], train_shuffled[:, dim*dim]
        valid_set_x_ret, valid_set_y_ret = valid_shuffled[:, :dim*dim], valid_shuffled[:, dim*dim]
        test_set_x_ret, test_set_y_ret = test_shuffled[:, :dim*dim], test_shuffled[:, dim*dim]

    return ((train_set_x_ret.reshape((len(digits)*n_train/n_class, dim, dim)), train_set_y_ret),
            (valid_set_x_ret.reshape((len(digits)*n_valid/n_class, dim, dim)), valid_set_y_ret),
            (test_set_x_ret.reshape((len(digits)*n_test/n_class, dim, dim)), test_set_y_ret))
