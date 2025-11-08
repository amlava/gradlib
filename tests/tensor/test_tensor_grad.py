from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_requires_grad:
    def test_1(self):
        x = Tensor(Value(1))
        y = Tensor(Value(1, requires_grad=True))

        assert x.requires_grad is False
        assert y.requires_grad is True

    def test_2(self):
        x = Tensor([Value(1, requires_grad=False), Value(2, requires_grad=False)])
        y = Tensor([Value(1, requires_grad=True), Value(2, requires_grad=False)])
        z = Tensor([Value(1, requires_grad=True), Value(2, requires_grad=True)])

        assert x.requires_grad is False
        assert y.requires_grad is False
        assert z.requires_grad is True

    def test_3(self):
        x = Tensor([[Value(1, requires_grad=False), Value(2, requires_grad=False)],
                    [Value(3, requires_grad=True), Value(4, requires_grad=True)]])
        y = Tensor([[Value(1, requires_grad=True), Value(2, requires_grad=True)],
                    [Value(3, requires_grad=True), Value(4, requires_grad=True)]])

        assert x.requires_grad is False
        assert y.requires_grad is True


class Test_tensor_grad:
    def test_1(self):
        x = Value(1, requires_grad=True)
        y = Value(2, requires_grad=True)
        z = x * y
        t = z + x

        t.backward() # x.grad == Value(3), y.grad == Value(1)

        v = Tensor([x, y])

        assert v.requires_grad is True
        assert x.grad.data == 3
        assert y.grad.data == 1
        assert v.grad.tolist() == [3, 1]
