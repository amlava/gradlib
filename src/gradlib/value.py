from __future__ import annotations

from typing import Optional, Callable, Any


class Value:
    def __init__(self, data, requires_grad: bool = False, children: Optional[list[Value]] = None):
        self._data = data
        self._requires_grad = requires_grad
        self._children = children

        if requires_grad:
            self._grad_func: Callable[[Value], None] = lambda x: None
            self._grad: Value = Value(0)

    @property
    def data(self):
        return self._data
    
    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def children(self) -> Optional[list[Value]]:
        return self._children
    
    @property
    def grad_func(self) -> Callable[[Value], None]:
        assert self._requires_grad, 'Tried to call "grad_func" on Value with requires_grad=False.'
        return self._grad_func

    @grad_func.setter
    def grad_func(self, f: Callable[[Value], None]):
        assert self._requires_grad, 'Tried to set "grad_func" on Value with requires_grad=False.'
        self._grad_func = f

    @property
    def grad(self) -> Value:
        assert self._requires_grad, 'Tried to call "grad" on Value with requires_grad=False.'
        return self._grad

    @grad.setter
    def grad(self, x: Value) -> None:
        assert self._requires_grad, 'Tried to set "grad" on Value with requires_grad=False.'
        self._grad = x

    def backward(self, grad: Optional[Value] = None) -> None:
        assert self._requires_grad, 'Called "backward" on Value with requires_grad=False.'
        if grad is None:
            self.grad = Value(1)
        self.grad_func(Value(1))

    def copy(self) -> Value:
        return value_like(self)

    def __add__(self, other: Value) -> Value:
        return add_values(self, other)

    def __iadd__(self, other: Value) -> Value:
        self._data += other.data 
        return self

    def __mul__(self, other: Value) -> Value:
        return multiply_values(self, other)

    def __imul__(self, other: Value) -> Value:
        self._data *= other.data
        return self

    def __neg__(self) -> Value:
        return Value(-1) * self

    def __truediv__(self, other: Value) -> Value:
        if other.data == 0:
            raise ZeroDivisionError
        return self * value_like(other, data=1/other.data)

    def __floordiv__(self, other: Value) -> Value:
        if other.data == 0:
            raise ZeroDivisionError
        return self * value_like(other, data=1//other.data)

    def __repr__(self) -> str:
        return f'Value(data={self._data}, requires_grad={self._requires_grad}, children={self._children})'


def value_like(
    value: Value,
    data: Optional[Any] = None,
    requires_grad: Optional[bool] = None,
    children: Optional[list] = None
) -> Value:
    data = data if data is not None else value.data
    requires_grad = requires_grad if requires_grad is not None else value.requires_grad
    children = children if children is not None else value.children
    return Value(data, requires_grad, children)

def add_values(x: Value, y: Value) -> Value:
    requires_grad = x.requires_grad or y.requires_grad
    z = Value(x.data + y.data,
              requires_grad,
              [x, y])

    if not requires_grad:
        return z

    def _add_func(grad: Value) -> None:
        if x.requires_grad:
            x.grad += Value(1) * grad
            x.grad_func(x.grad)
        if y.requires_grad:
            y.grad += Value(1) * grad
            y.grad_func(y.grad)

    z.grad_func = _add_func

    return z

def multiply_values(x: Value, y: Value) -> Value:
    requires_grad = x.requires_grad or y.requires_grad
    z = Value(x.data * y.data,
              requires_grad,
              [x, y])

    if not requires_grad:
        return z

    def _multiply_func(grad: Value) -> None:
        if x.requires_grad:
            x.grad += Value(y.data) * grad
            x.grad_func(x.grad)
        if y.requires_grad:
            y.grad += Value(x.data) * grad
            y.grad_func(y.grad)

    z.grad_func = _multiply_func

    return z
