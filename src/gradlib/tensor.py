from __future__ import annotations

from typing import TypeAlias, Union, Optional, Iterable

from gradlib.value import Value


NestedValueList: TypeAlias = Union[Value, list["NestedValueList"]]


class ShapeError(Exception):
    pass


class Tensor:
    def __init__(self, data: NestedValueList):
        self._data = data
        self._shape: tuple = self._get_shape(data)
        self._requires_grad = self._all_require_grad(data)

        if self._requires_grad:
            self._grad = Tensor(self._get_grad(data))

    @property
    def data(self):
        return self._data

    @property
    def requires_grad(self) -> bool:
        return self._requires_grad
    
    @property
    def shape(self) -> tuple:
        return self._shape
    
    def tolist(self, data: Optional[NestedValueList] = None) -> list:
        data = data if data is not None else self.data
        if isinstance(data, Value):
            return data.data

        return [self.tolist(x) for x in data]

    def _all_require_grad(self, data: NestedValueList) -> bool:
        if isinstance(data, Value):
            return data.requires_grad

        return all(self._all_require_grad(x) for x in data)

    def _all_same(self, l: Iterable) -> bool:
        x0 = next(l)
        return all(x == x0 for x in l)

    def _get_shape(self, data: NestedValueList) -> tuple:
        if isinstance(data, Value):
            return tuple()
        if self._all_same((t := self._get_shape(x) for x in data)):
            return (len(data), ) + t

        msg = 'Tried to construct Tensor out of data with inconsistent shape.'
        raise ShapeError(msg)

    def _get_grad(self, data: NestedValueList) -> NestedValueList:
        if isinstance(data, Value):
            return data.grad

        return [self._get_grad(d) for d in data]

    def backward(self) -> None:
        assert len(self._shape) == 0, 'Called "backward" on Tensor with dimension larger than 0.'

        self._grad = Tensor(Value(1))
        self.data.backward()
