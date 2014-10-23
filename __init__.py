import os
import gzip
import cPickle
import numpy as np
import scipy.stats as ss


def load_data(digits=[0,1,2,3,4,5,6,7,8,9], shuffle=True):
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

    path_to_d3set = os.path.join(os.path.dirname(__file__), 'd3set.pkl.gz')
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


def load_img(digits=[0,1,2,3,4,5,6,7,8,9], shuffle=True):
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

    path_to_d3set = os.path.join(os.path.dirname(__file__), 'd3set.pkl.gz')
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


def make_data(digits=[0,1,2,3,4,5,6,7,8,9], p=0.25, std=0.1, shuffle=True):
    """
    make data set
    """

    # make parts
    dim = 28

    a = np.zeros((dim, dim))
    b = np.zeros((dim, dim))
    b[3:5, 8:20] = 1.
    c = np.zeros((dim, dim))
    c[5:13, 6:8] = 1.
    d = np.zeros((dim, dim))
    d[13:15, 8:20] = 1.
    e = np.zeros((dim, dim))
    e[5:13, 20:22] = 1.
    f = np.zeros((dim, dim))
    f[15:23, 6:8] = 1.
    g = np.zeros((dim, dim))
    g[23:25, 8:20] = 1.
    h = np.zeros((dim, dim))
    h[15:23, 20:22]  = 1.

    # make digits
    digits = [np.zeros((dim, dim)) for i in range(10)]
    digits[0] = a + b + c + e + f + g + h
    digits[1] = a + e + h
    digits[2] = a + b + e + d + f + g
    digits[3] = a + b + e + d + h + g
    digits[4] = a + c + d + e + h
    digits[5] = a + b + c + d + g + h
    digits[6] = a + b + c + f + d + h + g
    digits[7] = a + b + e + h
    digits[8] = a + b + c + d + e +f + g + h
    digits[9] = a + b + c + d + e + h

    # make dataset
    train_n = 50000
    valid_n = 10000
    test_n = 10000 

    train_set_x = []
    train_set_y = []

    for i in range(10):
        for j in range(train_n/10):
            m = np.array(ss.bernoulli(p).rvs((28, 28)), np.bool)
            t = np.copy(digits[i])
            t[m] -= 1.
            t[t == -1.] = 1.
            t *= 0.5
            t = t + ss.norm(0., std).rvs((dim, dim))
            t[t < 0.] = 0.
        
            train_set_x.append(t)
            train_set_y.append(i)
        
    train_set_x = np.array(train_set_x, np.float32).reshape(train_n, dim*dim)
    train_set_y = np.array(train_set_y, np.int32)


    valid_set_x = []
    valid_set_y = []

    for i in range(10):
        for j in range(valid_n/10):
            m = np.array(ss.bernoulli(p).rvs((28, 28)), np.bool)
            t = np.copy(digits[i])
            t[m] -= 1.
            t[t == -1.] = 1.
            t *= 0.5
            t = t + ss.norm(0., std).rvs((dim, dim))
            t[t < 0.] = 0.
        
            valid_set_x.append(t)
            valid_set_y.append(i)
        
    valid_set_x = np.array(valid_set_x, np.float32).reshape(valid_n, dim*dim)
    valid_set_y = np.array(valid_set_y, np.int32)


    test_set_x = []
    test_set_y = []

    for i in range(10):
        for j in range(test_n/10):
            m = np.array(ss.bernoulli(p).rvs((28, 28)), np.bool)
            t = np.copy(digits[i])
            t[m] -= 1.
            t[t == -1.] = 1.
            t *= 0.5
            t = t + ss.norm(0., std).rvs((dim, dim))
            t[t < 0.] = 0.
        
            test_set_x.append(t)
            test_set_y.append(i)
        
    test_set_x = np.array(test_set_x, np.float32).reshape(test_n, dim*dim)
    test_set_y = np.array(test_set_y, np.int32)

    datasets = ((train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y))
    return datasets
