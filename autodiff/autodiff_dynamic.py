import numpy as np
from abc import ABCMeta, abstractmethod


class Tensor(object):
    __array_ufunc__ = None

    def __init__(self, data, from_tensors=None, op=None, grad=None, name=""):
        """
        Parameters
        ----------
        data: 数据
        from_tensors: 计算图中的历史tensors
        op: 运算操作符
        grad: 梯度值

        Returns
        -------

        """
        self.data = data
        self.from_tensors = from_tensors
        self.op = op
        if grad is None:
            grad = np.zeros(self.data.shape) if isinstance(self.data, np.ndarray) else 0
        self.grad = grad
        self.name = name

    def __add__(self, other):
        if isinstance(other, Tensor):
            return add.forward([self, other])
        else:
            return add_with_const.forward([self, other])

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return sub.forward([self, other])
        else:
            return add_with_const.forward([self, -other])

    def __rsub__(self, other):
        return rsub_with_const.forward([self, other])

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return mul.forward([self, other])
        else:
            return mul_with_const.forward([self, other])

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return div.forward([self, other])
        else:
            return mul_with_const.forward([self, 1 / other])

    def __rtruediv__(self, other):
        return rdiv_with_const.forward([self, other])

    def __neg__(self):
        return self.__rsub__(0)

    def matmul(self, other):
        return mul_with_matrix.forward([self, other])

    def sum(self):
        return sum.forward([self])

    def mean(self):
        return sum.forward([self]) / self.data.size

    def log(self):
        return log.forward([self])

    def exp(self):
        return exp.forward([self])

    def backward(self, grad=None):
        # 判断y的梯度是否存在，如果不存在则初始化为y.data形状的1
        if grad is None:
            self.grad = grad = np.ones(self.data.shape) if isinstance(self.data, np.ndarray) else 1
        # 若存在op，则计算梯度，并将梯度回传到from_tensors中的梯度
        if self.op and self.from_tensors:
            grads = self.op.backward(self.from_tensors, grad)
            for i in range(len(grads)):
                self.from_tensors[i].grad += grads[i]
                self.from_tensors[i].backward(grads[i])

    def zero_grad(self):
        self.grad = np.zeros(self.data.shape) if isinstance(self.data, np.ndarray) else 0

    __radd__ = __add__
    __rmul__ = __mul__


class Op(metaclass=ABCMeta):
    @abstractmethod
    def forward(self, from_tensors):
        pass

    @abstractmethod
    def backward(self, from_tensors, grad):
        pass


class AddOp(Op):
    def forward(self, from_tensors):
        return Tensor(
            from_tensors[0].data + from_tensors[1].data,
            from_tensors,
            self,
            name=from_tensors[0].name + "_add_" + from_tensors[1].name,
        )

    def backward(self, from_tensors, grad):
        return [grad, grad]


class AddWithConstOp(Op):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data + from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad]


class SubOp(Op):
    def forward(self, from_tensors):
        return Tensor(
            from_tensors[0].data - from_tensors[1].data,
            from_tensors,
            self,
            name=from_tensors[0].name + "_sub_" + from_tensors[1].name,
        )

    def backward(self, from_tensors, grad):
        return [grad, -grad]


class RSubWithConstOp(Op):
    def forward(self, from_tensors):
        return Tensor(-(from_tensors[0].data - from_tensors[1]), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [-grad]


class MulOp(Op):
    def forward(self, from_tensors):
        return Tensor(
            from_tensors[0].data * from_tensors[1].data,
            from_tensors,
            self,
            name=from_tensors[0].name + "_mul_" + from_tensors[1].name,
        )

    def backward(self, from_tensors, grad):
        return [from_tensors[1].data * grad, from_tensors[0].data * grad]


class MulWithConstOp(Op):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data * from_tensors[1], from_tensors, self)

    def backward(self, from_tensors, grad):
        return [from_tensors[1] * grad]


class MulWithMatrixOp(Op):
    def forward(self, from_tensors):
        return Tensor(np.matmul(from_tensors[0].data, from_tensors[1].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        # Useful formula: if Y=AB, then dA=dY B^T, dB=A^T dY
        return [np.matmul(grad, from_tensors[1].data.T), np.matmul(from_tensors[0].data.T, grad)]


class DivOp(Op):
    def forward(self, from_tensors):
        return Tensor(from_tensors[0].data / from_tensors[1].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [
            grad / from_tensors[1].data,
            -grad * from_tensors[0].data / (from_tensors[1].data * from_tensors[1].data),
        ]


class RDivWithConstOp(Op):
    def forward(self, from_tensors):
        return Tensor(from_tensors[1] / from_tensors[0].data, from_tensors, self)

    def backward(self, from_tensors, grad):
        return [-grad * from_tensors[1] / (from_tensors[0].data * from_tensors[0].data)]


class SumOp(Op):
    def forward(self, from_tensors):
        return Tensor(np.sum(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * np.ones(from_tensors[0].data.shape)]


class ExpOp(Op):
    def forward(self, from_tensors):
        return Tensor(np.exp(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad * np.exp(from_tensors[0].data)]


class LogOp(Op):
    def forward(self, from_tensors):
        return Tensor(np.log(from_tensors[0].data), from_tensors, self)

    def backward(self, from_tensors, grad):
        return [grad / from_tensors[0].data]


add = AddOp()
add_with_const = AddWithConstOp()
sub = SubOp()
rsub_with_const = RSubWithConstOp()
mul = MulOp()
mul_with_const = MulWithConstOp()
mul_with_matrix = MulWithMatrixOp()
div = DivOp()
rdiv_with_const = RDivWithConstOp()
sum = SumOp()
exp = ExpOp()
log = LogOp()
