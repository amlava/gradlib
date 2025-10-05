import pytest

from gradlib.value import Value
from gradlib.tensor import Tensor, ShapeError


class Test_tensor_shape:
    def test_1(self):
        x = Tensor(Value(3))
        assert x.shape == tuple()

    def test_2(self):
        x = Tensor([Value(x) for x in range(3)])
        assert x.shape == (3, )

    def test_3(self):
        x = Tensor([[Value(y) for y in range(x, x+3)] for x in range(0, 9, 3)])

        assert x.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
        assert x.shape == (3, 3)

    def test_4(self):
        x = Tensor([[Value(y) for y in range(x, x+3)] for x in range(0, 12, 3)])

        assert x.tolist() == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        assert x.shape == (4, 3)

    def test_5(self):
        with pytest.raises(ShapeError):
            Tensor([Value(0), [Value(1)]])
        with pytest.raises(ShapeError):
            Tensor([[[Value(0), Value(1)]],
                    [[Value(2), Value(3), Value(4)]]])
