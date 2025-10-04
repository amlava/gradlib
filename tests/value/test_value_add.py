from gradlib.value import Value


class Test_value_add:
    def test_1(self):
        x = Value(2, requires_grad=True)
        y = Value(3)
        z = x + y

        assert z.requires_grad is True
        assert z.data == 5

        z.backward()

        assert z.grad.data == 1
        assert x.grad.data == 1
