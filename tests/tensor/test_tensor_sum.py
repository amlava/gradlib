from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_sum:
    def test_1(self):
        x = Tensor([Value(i) for i in range(1, 4)])
        assert x.sum().tolist() == 6
    
    def test_2(self):
        x = Tensor([[Value(i)] for i in range(1, 4)])
        assert x.sum().tolist() == 6

    def test_3(self):
        x = Value(1, requires_grad=True)
        y = Value(2, requires_grad=True)
        z = Tensor([x, y])
        t = z.sum()

        t.backward()
        
        assert z.grad.tolist() == [1, 1]
