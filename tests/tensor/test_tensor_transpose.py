from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_transpose:
    def test_1(self):
        x = Tensor([
            [Value(1, requires_grad=True), Value(2, requires_grad=True)],
            [Value(0, requires_grad=True), Value(1, requires_grad=True)],
            [Value(2, requires_grad=True), Value(3, requires_grad=True)]
        ])
        y = x.transpose()
        z = y.sum()

        z.backward()

        assert y.tolist() == [[1, 0, 2], [2, 1, 3]]
        assert y.grad.tolist() == [[1, 1, 1], [1, 1, 1]]
        assert x.grad.tolist() == [[1, 1], [1, 1], [1, 1]]
