from autodiff_dynamic import Tensor
import numpy as np


def test_identity():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = x2
    assert np.array_equal(y.data, x2_val)

    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_add_two_vars():
    x1_val = 3 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = x1 + x2
    assert np.array_equal(y.data, x1_val + x2_val)

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_add_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = 5 + x2
    assert np.array_equal(y.data, x2_val + 5)

    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val))


def test_sub_two_vars():
    x1_val = 3 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = x1 - x2
    assert np.array_equal(y.data, x1_val - x2_val)

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, -np.ones_like(x2_val))


def test_sub_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = np.ones(3) - x2
    assert np.array_equal(y.data, np.ones(3) - x2_val)

    y.backward()
    assert np.array_equal(x2.grad, -np.ones_like(x2_val))


def test_sub_by_const_num():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = 3 - x2
    assert np.array_equal(y.data, 3 - x2_val)

    y.backward()
    assert np.array_equal(x2.grad, -np.ones_like(x2_val))


def test_mul_two_vars():
    x1_val = 3 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = x1 * x2
    assert np.array_equal(y.data, x1_val * x2_val)

    y.backward()
    assert np.array_equal(x1.grad, x2_val)
    assert np.array_equal(x2.grad, x1_val)


def test_mul_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = 5 * x2
    assert np.array_equal(y.data, 5 * x2_val)

    y.backward()
    assert np.array_equal(x2.grad, np.ones_like(x2_val) * 5)


def test_div_two_vars():
    x1_val = 3 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = x1 / x2
    assert np.array_equal(y.data, x1_val / x2_val)

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val) / x2_val)
    assert np.array_equal(x2.grad, -x1_val / (x2_val * x2_val))


def test_div_by_const():
    x2_val = 2 * np.ones(3)
    x2 = Tensor(x2_val)

    y = 5 / x2
    assert np.array_equal(y.data, 5 / x2_val)

    y.backward()
    assert np.array_equal(x2.grad, -5 / (x2_val * x2_val))


def test_add_mul_mix_1():
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x1 = Tensor(x1_val, name="x1")
    x2 = Tensor(x2_val, name="x2")
    x3 = Tensor(x3_val, name="x3")

    y = x1 + x2 * x3 * x1
    assert np.array_equal(y.data, x1_val + x2_val * x3_val * x1_val)

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val) + x2_val * x3_val)
    assert np.array_equal(x2.grad, x3_val * x1_val)
    assert np.array_equal(x3.grad, x2_val * x1_val)


def test_add_mul_mix_2():
    x1_val = 1 * np.ones(3)
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x4_val = 4 * np.ones(3)
    x1 = Tensor(x1_val, name="x1")
    x2 = Tensor(x2_val, name="x2")
    x3 = Tensor(x3_val, name="x3")
    x4 = Tensor(x4_val, name="x4")

    y = x1 + x2 * x3 * x4
    assert np.array_equal(y.data, x1_val + x2_val * x3_val * x4_val)

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))
    assert np.array_equal(x2.grad, x3_val * x4_val)
    assert np.array_equal(x3.grad, x2_val * x4_val)
    assert np.array_equal(x4.grad, x2_val * x3_val)


def test_add_mul_mix_3():
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    x2 = Tensor(x2_val, name="x2")
    x3 = Tensor(x3_val, name="x3")

    z = x2 * x2 + x2 + x3 + 3
    y = z * z + x3
    z_val = x2_val * x2_val + x2_val + x3_val + 3
    assert np.array_equal(y.data, z_val * z_val + x3_val)

    y.backward()
    assert np.array_equal(x2.grad, 2 * (x2_val * x2_val + x2_val + x3_val + 3) * (2 * x2_val + 1))
    assert np.array_equal(x3.grad, 2 * (x2_val * x2_val + x2_val + x3_val + 3) + 1)


def test_matmul_two_vars():
    x2_val = np.array([[1, 2], [3, 4], [5, 6]])  # 3x2
    x3_val = np.array([[7, 8, 9], [10, 11, 12]])  # 2x3
    x2 = Tensor(x2_val)
    x3 = Tensor(x3_val)

    y = x2.matmul(x3)
    assert np.array_equal(y.data, np.matmul(x2_val, x3_val))

    y.backward()
    # Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
    assert np.array_equal(x2.grad, np.matmul(np.ones_like(np.matmul(x2_val, x3_val)), x3_val.T))
    assert np.array_equal(x3.grad, np.matmul(x2_val.T, np.ones_like(np.matmul(x2_val, x3_val))))


def test_log_op():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)

    y = x1.log()
    assert np.array_equal(y.data, np.log(x1_val))

    y.backward()
    assert np.array_equal(x1.grad, 1 / x1_val)


def test_log_two_vars():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = (x1 * x2).log()
    assert np.array_equal(y.data, np.log(x1_val * x2_val))

    y.backward()
    assert np.array_equal(x1.grad, 1 / x1_val)
    assert np.array_equal(x2.grad, 1 / x2_val)


def test_exp_op():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)

    y = x1.exp()
    assert np.array_equal(y.data, np.exp(x1_val))

    y.backward()
    assert np.array_equal(x1.grad, np.exp(x1_val))


def test_exp_mix_op():
    x1_val = 2 * np.ones(3)
    x2_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    x2 = Tensor(x2_val)

    y = ((x1 * x2).log() + 1).exp()
    assert np.array_equal(y.data, np.exp(np.log(x1_val * x2_val) + 1))

    y.backward()
    assert np.array_equal(x1.grad, y.data / x1_val)
    assert np.array_equal(x2.grad, y.data / x2_val)


def test_reduce_sum():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)

    y = x1.sum()
    assert np.array_equal(y.data, np.sum(x1_val))

    y.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val))


def test_reduce_sum_mix():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)

    y = x1.sum().exp()
    assert np.array_equal(y.data, np.exp(np.sum(x1_val)))

    y.backward()
    assert np.array_equal(x1.grad, np.exp(np.sum(x1_val)) * np.ones_like(x1_val))

    x1.zero_grad()
    y2 = x1.sum().log()
    assert np.array_equal(y2.data, np.log(np.sum(x1_val)))
    y2.backward()
    assert np.array_equal(x1.grad, np.ones_like(x1_val) / np.sum(x1_val))


def test_mix_all():
    x1_val = 2 * np.ones(3)
    x1 = Tensor(x1_val)

    y = 1 / (1 + (-x1.sum()).exp())
    expected_y_val = 1 / (1 + np.exp(-np.sum(x1_val)))
    assert np.array_equal(y.data, expected_y_val)

    y.backward()
    expected_y_grad = expected_y_val * (1 - expected_y_val) * np.ones_like(x1_val)
    assert np.sum(np.abs(x1.grad - expected_y_grad)) < 1e-10


def test_logistic():
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    w = Tensor(w_val)

    y = 1 / (1 + (-w * x1).sum().exp())
    expected_y_val = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
    assert np.array_equal(y.data, expected_y_val)

    y.backward()
    expected_w_grad = expected_y_val * (1 - expected_y_val) * x1_val
    assert np.sum(np.abs(w.grad - expected_w_grad)) < 1e-7


def test_log_logistic():
    x1_val = 3 * np.ones(3)
    w_val = 3 * np.ones(3)
    x1 = Tensor(x1_val)
    w = Tensor(w_val)

    y = (1 / (1 + (-w * x1).sum().exp())).log()
    logistic = 1 / (1 + np.exp(-np.sum(w_val * x1_val)))
    expected_y_val = np.log(logistic)
    assert np.array_equal(y.data, expected_y_val)

    y.backward()
    expected_w_grad = (1 - logistic) * x1_val
    assert np.sum(np.abs(w.grad - expected_w_grad)) < 1e-7


def test_logistic_loss():
    y_val = 0
    x_val = np.array([2, 3, 4])
    w_val = np.random.random(3)

    x = Tensor(x_val)
    w = Tensor(w_val)
    y = Tensor(y_val)

    h = 1 / (1 + (-w * x).sum().exp())
    l = y * h.log() + (1 - y) * (1 - h).log()
    logistic = 1 / (1 + np.exp(-np.sum(w_val * x_val)))
    expected_l_val = y_val * np.log(logistic) + (1 - y_val) * np.log(1 - logistic)
    assert np.array_equal(l.data, expected_l_val)

    l.backward()
    expected_w_grad = (y_val - logistic) * x_val
    assert np.sum(np.abs(expected_w_grad - w.grad)) < 1e-1


def test_mean():
    x1_val = np.array([1, 2, 3, 4])
    x = Tensor(x1_val)

    y = x.mean()
    assert np.array_equal(y.data, np.mean(x1_val))

    y.backward()
    assert np.array_equal(x.grad, np.ones_like(x1_val) / 4)
