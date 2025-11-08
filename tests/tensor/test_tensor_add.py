from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_add:
    def test_1(self):
        x = Tensor(Value(1, requires_grad=True))
        y = Tensor(Value(2))
        z = x + y

        z.backward()

        assert z.tolist() == 3
        assert z.grad.tolist() == 1
        assert x.grad.tolist() == 1

    def test_2(self):
        x = Tensor([Value(1, requires_grad=True)])
        y = Tensor([Value(2)])
        z = x + y

        assert z.tolist() == [3]

    def test_3(self):
        x = Tensor([Value(1, requires_grad=True)])
        y = Tensor([Value(2)])
        z = x + y
        t = (z + x).sum()

        t.backward()

        assert t.tolist() == 4
        assert z.grad.tolist() == [1]
        assert x.grad.tolist() == [2]
