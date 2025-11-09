from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_test:
    def test_1(self):
        x = Tensor([Value(1, requires_grad=True), Value(1, requires_grad=True), Value(1, requires_grad=True)])
        y = Tensor([Value(0, requires_grad=True), Value(2, requires_grad=True), Value(1, requires_grad=True)])
        z = x.dot(y)

        z.backward()
        
        assert z.tolist() == 3
        assert x.grad.tolist() == [0, 2, 1]
        assert y.grad.tolist() == [1, 1, 1]
