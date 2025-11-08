from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_multiply:
    def test_1(self):
        x = Tensor(Value(2, requires_grad=True))
        y = Tensor(Value(3, requires_grad=True))
        z = x * y

        z.backward()

        assert x.grad.tolist() == 3
        assert y.grad.tolist() == 2

    def test_2(self):
        x = Tensor([Value(i, requires_grad=True) for i in range(5)])
        y = Tensor([Value(i, requires_grad=True) for i in range(5, 10)])
        z = (x * y).sum()

        z.backward()

        assert x.grad.tolist() == list(range(5, 10))
        assert y.grad.tolist() == list(range(5))

    def test_3(self):
        x = Tensor([
            [Value(1, requires_grad=True)],
            [Value(2, requires_grad=True)]
        ])
        y = Tensor([
            [Value(3, requires_grad=True)],
            [Value(4, requires_grad=True)]
        ])
        z = (x * y).sum()

        z.backward()

        assert x.grad.tolist() == [[3], [4]]
        assert y.grad.tolist() == [[1], [2]]
