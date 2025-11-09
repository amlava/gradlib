from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_matmul:
    def test_1(self):
        x = Tensor([
            [Value(1, requires_grad=True), Value(3, requires_grad=True)],
            [Value(3, requires_grad=True), Value(5, requires_grad=True)]
        ])
        y = Tensor([
            [Value(1, requires_grad=True), Value(0, requires_grad=True), Value(3, requires_grad=True)],
            [Value(1, requires_grad=True), Value(2, requires_grad=True), Value(1, requires_grad=True)]
        ])
        z = x @ y
        v = z.sum()

        v.backward()

        assert z.tolist() == [[4, 6, 6], [8, 10, 14]]
        assert x.grad.tolist() == [[4, 4], [4, 4]]
        assert y.grad.tolist() == [[4, 4, 4], [8, 8, 8]]
