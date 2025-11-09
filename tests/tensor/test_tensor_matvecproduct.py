from gradlib.value import Value
from gradlib.tensor import Tensor


class Test_tensor_matvecproduct:
    def test_1(self):
        x = Tensor([
            [Value(1, requires_grad=True), Value(2, requires_grad=True)],
            [Value(0, requires_grad=True), Value(1, requires_grad=True)],
            [Value(3, requires_grad=True), Value(4, requires_grad=True)]
        ])
        y = Tensor([Value(1, requires_grad=True), Value(2, requires_grad=True)])
        z = x @ y

        assert z.tolist() == [5, 2, 11]
