from gradlib.value import Value


class Test_value_multiply:
    def test_1(self):
        x = Value(2, requires_grad=True)
        y = Value(3, requires_grad=True)
        z = x * y

        assert z.requires_grad is True
        assert z.data == 6

        z.backward()

        assert z.grad.data == 1
        assert x.grad.data == 3
        assert y.grad.data == 2