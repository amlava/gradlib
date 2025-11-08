from __future__ import annotations

from typing import TypeAlias, Union, Optional, Iterable, Literal, overload

from gradlib.value import Value


NestedValueList: TypeAlias = Union[Value, list["NestedValueList"]]


class ShapeError(Exception):
    pass


class Tensor:
    def __init__(self, data: NestedValueList):
        self._data = data
        self._shape = self._get_shape(data)
        self._requires_grad = self._all_require_grad(data)

    @property
    def data(self) -> NestedValueList:
        return self._data

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    @property
    def grad(self) -> Tensor:
        assert self._requires_grad, 'Called "grad" on a Tensor with requires_grad=False.'
        return Tensor(self._get_grad(self._data))

    def tolist(self, data: Optional[NestedValueList] = None) -> list:
        data = data if data is not None else self._data
        if isinstance(data, Value):
            return data.data
        return [self.tolist(x) for x in data]

    def _all_require_grad(self, data: NestedValueList) -> bool:
        if isinstance(data, Value):
            return data.requires_grad
        return all(self._all_require_grad(x) for x in data)

    def _all_same(self, l: Iterable) -> bool:
        x0 = next(iter(l))
        return all(x == x0 for x in l)

    def _get_shape(self, data: NestedValueList) -> tuple:
        if isinstance(data, Value):
            return tuple()
        t = () # satisfy type checker rule `unsupported-operator`
        if self._all_same((t := self._get_shape(x) for x in data)):
            return (len(data), ) + t
        msg = 'Tried to construct Tensor out of data with inconsistent shape.'
        raise ShapeError(msg)

    def _get_grad(self, data: NestedValueList) -> NestedValueList:
        if isinstance(data, Value):
            return data.grad
        return [self._get_grad(d) for d in data]

    def __getitem__(self, idx: int) -> Tensor:
        assert len(self.shape) > 0
        return Tensor(self.data[idx])

    def __add__(self, other: Tensor) -> Tensor:
        return add_tensors(self, other)
    
    def __mul__(self, other: Tensor) -> Tensor:
        return multiply_tensors(self, other)

    def backward(self) -> None:
        assert len(self._shape) == 0, 'Called "backward" on Tensor with dimension larger than 0.'
        self._grad = Tensor(Value(1))
        self.data.backward()

    def sum(self) -> Tensor:
        return tensor_sum(self)

    def __repr__(self) -> str:
        return f'Tensor(data={self.tolist()}, shape={self._shape}, requires_grad={self._requires_grad})'


def tensor_sum(x: Tensor) -> Tensor:
    if len(x.shape) == 0:
        return x
    elif len(x.shape) == 1:
        return Tensor(sum(x.data, start=Value(0)))
    return sum([tensor_sum(x[i]) for i in range(x.shape[0])], start=Tensor(Value(0)))

@overload
def add_tensors(x: Tensor, y: Tensor, as_value: Literal[False] = ...) -> Tensor: ...
@overload
def add_tensors(x: Tensor, y: Tensor, as_value: Literal[True]) -> NestedValueList: ...
@overload
def multiply_tensors(x: Tensor, y: Tensor, as_value: Literal[False] = ...) -> Tensor: ...
@overload
def multiply_tensors(x: Tensor, y: Tensor, as_value: Literal[True]) -> NestedValueList: ...

def add_tensors(x: Tensor, y: Tensor, as_value: bool = False) -> Tensor | NestedValueList:
    assert x.shape == y.shape, f'Cannot add Tensors of different shape. Tensor1.shape={x.shape}, Tensor2.shape={y.shape}.'
    if len(x.shape) == 0:
        return x.data + y.data if as_value else Tensor(x.data + y.data)
    data = [add_tensors(x[i], y[i], as_value=True) for i in range(x.shape[0])]
    return data if as_value else Tensor(data)

def multiply_tensors(x: Tensor, y: Tensor, as_value: bool = False) -> Tensor | NestedValueList:
    assert x.shape == y.shape, f'Cannot multiply Tensors of different shape. Tensor1.shape={x.shape}, Tensor2.shape={y.shape}.'
    if len(x.shape) == 0:
        return x.data * y.data if as_value else Tensor(x.data * y.data)
    data = [multiply_tensors(x[i], y[i], as_value=True) for i in range(x.shape[0])]
    return data if as_value else Tensor(data)
