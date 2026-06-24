from dataclasses import dataclass
import sys
import time
from typing import NamedTuple

@dataclass(slots=True)
class DataClass:
    a: int
    b: int
    c: int
    d: int
    e: int

class TupleWithName(NamedTuple):
    a: int
    b: int
    c: int
    d: int
    e: int

class SugaredLists:
    __slots__ = ("a", "b", "c", "d", "e")

    def __init__(self, n):
        self.a = [None] * n
        self.b = [None] * n
        self.c = [None] * n
        self.d = [None] * n
        self.e = [None] * n

    def __getitem__(self, i: int) -> "RowView":
        return RowView(self, i)

class RowView:
    __slots__ = ("_table", "_i")

    def __init__(self, table: SugaredLists, i: int):
        self._table = table
        self._i = i

    @property
    def a(self) -> int:
        return self._table.a[self._i]

    @a.setter
    def a(self, value: int) -> None:
        self._table.a[self._i] = value

    @property
    def b(self) -> int:
        return self._table.b[self._i]

    @b.setter
    def b(self, value: int) -> None:
        self._table.b[self._i] = value

    @property
    def c(self) -> int:
        return self._table.c[self._i]

    @c.setter
    def c(self, value: int) -> None:
        self._table.c[self._i] = value

    @property
    def d(self) -> int:
        return self._table.d[self._i]

    @d.setter
    def d(self, value: int) -> None:
        self._table.d[self._i] = value

    @property
    def e(self) -> int:
        return self._table.e[self._i]

    @e.setter
    def e(self, value: int) -> None:
        self._table.e[self._i] = value

def main():
    N = 500_000

    # Classes
    start_time = time.perf_counter()
    classes = [None] * N
    for i in range(N):
        classes[i] = DataClass(i, i+1, i+2, i+3, i+4)
    classes_time = time.perf_counter() - start_time

    # NamedTuple
    start_time = time.perf_counter()
    named_tuples = [None] * N
    for i in range(N):
        named_tuples[i] = TupleWithName(i, i+1, i+2, i+3, i+4)
    named_tuples_time = time.perf_counter() - start_time

    # Tuples
    start_time = time.perf_counter()
    tuples = [None] * N
    for i in range(N):
        tuples[i] = (i, i+1, i+2, i+3, i+4)
    tuples_time = time.perf_counter() - start_time

    # Parallel lists
    start_time = time.perf_counter()
    list_a = [-1] * N
    list_b = [-1] * N
    list_c = [-1] * N
    list_d = [-1] * N
    list_e = [-1] * N
    for i in range(N):
        list_a[i] = i
        list_b[i] = i+1
        list_c[i] = i+2
        list_d[i] = i+3
        list_e[i] = i+4
    lists_time = time.perf_counter() - start_time

    # Sugared lists
    start_time = time.perf_counter()
    sugared_lists = SugaredLists(N)
    for i in range(N):
        sugared_lists[i].a = i
        sugared_lists[i].b = i+1
        sugared_lists[i].c = i+2
        sugared_lists[i].d = i+3
        sugared_lists[i].e = i+4
    sugared_lists_time = time.perf_counter() - start_time

    print(f"{classes_time=}\n{named_tuples_time=}\n{tuples_time=}\n{lists_time=}\n{sugared_lists_time=}")
    return 0

if __name__ == '__main__':
    sys.exit(main())