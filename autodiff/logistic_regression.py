from autodiff_dynamic import Tensor
import numpy as np


def gen_2d_data(n):
    x_data = np.random.random([n, 2])
    y_data = np.ones([n, 1])
    for i in range(n):
        if x_data[i][0] + x_data[i][1] < 1:
            y_data[i][0] = 0
    x_data_with_bias = np.ones([n, 3])
    x_data_with_bias[:, 1:] = x_data
    return x_data_with_bias, y_data


def logistic_prob(_w):
    def wrapper(_x):
        return 1 / (1 + np.exp(-np.sum(_x * _w)))

    return wrapper


def test_accuracy(_w, _X, _Y):
    prob = logistic_prob(_w)
    correct = 0
    total = len(_Y)
    for i in range(len(_Y)):
        x = _X[i]
        y = _Y[i]
        p = prob(x)
        if p >= 0.5 and y == 1.0:
            correct += 1
        elif p < 0.5 and y == 0.0:
            correct += 1
    print("总数：%d, 预测正确：%d" % (total, correct))


def auto_diff_lr():
    N = 100
    X_val, Y_val = gen_2d_data(N)
    w_val = np.ones(3)
    w = Tensor(w_val)
    # plot()
    test_accuracy(w_val, X_val, Y_val)
    alpha = 0.01
    max_iters = 300
    for iteration in range(max_iters):
        acc_L_val = 0
        for i in range(N):
            w.zero_grad()
            x_val = X_val[i]
            y_val = Y_val[i]
            x = Tensor(x_val)
            y = Tensor(y_val)
            h = 1 / (1 + (-(w * x).sum()).exp())
            l = y * h.log() + (1 - y) * (1 - h).log()
            l.backward()
            w.data += alpha * w.grad
            acc_L_val += l.data
        print("iter = %d, likelihood = %s, w = %s" % (iteration, acc_L_val, w_val))
    test_accuracy(w_val, X_val, Y_val)


if __name__ == "__main__":
    auto_diff_lr()
