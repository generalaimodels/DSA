from collections import deque
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class ArrayStack:
    def __init__(self) -> None:
        self._data: List[Any] = []

    def push(self, value: Any) -> None:
        self._data.append(value)

    def pop(self) -> Any:
        if not self._data:
            raise IndexError("pop from empty stack")
        return self._data.pop()

    def peek(self) -> Any:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)

    def clear(self) -> None:
        self._data.clear()


class _SLLNode:
    __slots__ = ("value", "next")

    def __init__(self, value: Any, next: Optional["__class__"] = None) -> None:
        self.value = value
        self.next = next


class LinkedStack:
    def __init__(self) -> None:
        self._head: Optional[_SLLNode] = None
        self._size = 0

    def push(self, value: Any) -> None:
        self._head = _SLLNode(value, self._head)
        self._size += 1

    def pop(self) -> Any:
        if self._head is None:
            raise IndexError("pop from empty stack")
        node = self._head
        self._head = node.next
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise IndexError("peek from empty stack")
        return self._head.value

    def is_empty(self) -> bool:
        return self._head is None

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = None
        self._size = 0


class TwoStacksOneArray:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: List[Any] = [None] * capacity
        self._top1 = -1
        self._top2 = capacity
        self._n = capacity

    def push1(self, value: Any) -> None:
        if self._top1 + 1 == self._top2:
            raise OverflowError("stack overflow")
        self._top1 += 1
        self._data[self._top1] = value

    def push2(self, value: Any) -> None:
        if self._top2 - 1 == self._top1:
            raise OverflowError("stack overflow")
        self._top2 -= 1
        self._data[self._top2] = value

    def pop1(self) -> Any:
        if self._top1 == -1:
            raise IndexError("pop from empty stack1")
        v = self._data[self._top1]
        self._top1 -= 1
        return v

    def pop2(self) -> Any:
        if self._top2 == self._n:
            raise IndexError("pop from empty stack2")
        v = self._data[self._top2]
        self._top2 += 1
        return v

    def peek1(self) -> Any:
        if self._top1 == -1:
            raise IndexError("peek from empty stack1")
        return self._data[self._top1]

    def peek2(self) -> Any:
        if self._top2 == self._n:
            raise IndexError("peek from empty stack2")
        return self._data[self._top2]

    def size1(self) -> int:
        return self._top1 + 1

    def size2(self) -> int:
        return self._n - self._top2


class KStacksOneArray:
    def __init__(self, k: int, capacity: int) -> None:
        if k <= 0 or capacity <= 0 or k > capacity:
            raise ValueError("invalid k or capacity")
        self._k = k
        self._n = capacity
        self._data: List[Any] = [None] * capacity
        self._top: List[int] = [-1] * k
        self._next: List[int] = list(range(1, capacity)) + [-1]
        self._free = 0

    def push(self, stack_index: int, value: Any) -> None:
        if not (0 <= stack_index < self._k):
            raise IndexError("invalid stack index")
        if self._free == -1:
            raise OverflowError("stack overflow")
        i = self._free
        self._free = self._next[i]
        self._data[i] = value
        self._next[i] = self._top[stack_index]
        self._top[stack_index] = i

    def pop(self, stack_index: int) -> Any:
        if not (0 <= stack_index < self._k):
            raise IndexError("invalid stack index")
        i = self._top[stack_index]
        if i == -1:
            raise IndexError("pop from empty stack")
        self._top[stack_index] = self._next[i]
        self._next[i] = self._free
        self._free = i
        return self._data[i]

    def peek(self, stack_index: int) -> Any:
        if not (0 <= stack_index < self._k):
            raise IndexError("invalid stack index")
        i = self._top[stack_index]
        if i == -1:
            raise IndexError("peek from empty stack")
        return self._data[i]

    def is_empty(self, stack_index: int) -> bool:
        if not (0 <= stack_index < self._k):
            raise IndexError("invalid stack index")
        return self._top[stack_index] == -1


class StackPushEfficient:
    def __init__(self) -> None:
        self._q1: Deque[Any] = deque()
        self._q2: Deque[Any] = deque()

    def push(self, value: Any) -> None:
        self._q1.append(value)

    def pop(self) -> Any:
        if not self._q1:
            raise IndexError("pop from empty stack")
        while len(self._q1) > 1:
            self._q2.append(self._q1.popleft())
        v = self._q1.popleft()
        self._q1, self._q2 = self._q2, self._q1
        return v

    def peek(self) -> Any:
        if not self._q1:
            raise IndexError("peek from empty stack")
        while len(self._q1) > 1:
            self._q2.append(self._q1.popleft())
        v = self._q1[0]
        self._q2.append(self._q1.popleft())
        self._q1, self._q2 = self._q2, self._q1
        return v

    def is_empty(self) -> bool:
        return not self._q1

    def size(self) -> int:
        return len(self._q1)


class StackPopEfficient:
    def __init__(self) -> None:
        self._q: Deque[Any] = deque()

    def push(self, value: Any) -> None:
        self._q.append(value)
        for _ in range(len(self._q) - 1):
            self._q.append(self._q.popleft())

    def pop(self) -> Any:
        if not self._q:
            raise IndexError("pop from empty stack")
        return self._q.popleft()

    def peek(self) -> Any:
        if not self._q:
            raise IndexError("peek from empty stack")
        return self._q[0]

    def is_empty(self) -> bool:
        return not self._q

    def size(self) -> int:
        return len(self._q)


class MinStack:
    def __init__(self) -> None:
        self._data: List[int] = []
        self._min: List[int] = []

    def push(self, value: int) -> None:
        self._data.append(value)
        if not self._min or value <= self._min[-1]:
            self._min.append(value)
        else:
            self._min.append(self._min[-1])

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty stack")
        self._min.pop()
        return self._data.pop()

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def get_min(self) -> int:
        if not self._min:
            raise IndexError("min from empty stack")
        return self._min[-1]

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)


class EncodedMinStack:
    def __init__(self) -> None:
        self._data: List[int] = []
        self._min: Optional[int] = None

    def push(self, x: int) -> None:
        if self._min is None:
            self._data.append(x)
            self._min = x
        elif x >= self._min:
            self._data.append(x)
        else:
            self._data.append(2 * x - self._min)
            self._min = x

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty stack")
        t = self._data.pop()
        if t >= self._min:  # type: ignore
            return t
        res = self._min  # type: ignore
        self._min = 2 * self._min - t  # type: ignore
        return res  # type: ignore

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty stack")
        t = self._data[-1]
        if t >= self._min:  # type: ignore
            return t
        return self._min  # type: ignore

    def get_min(self) -> int:
        if self._min is None:
            raise IndexError("min from empty stack")
        return self._min

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)


class MaxStack:
    def __init__(self) -> None:
        self._data: List[int] = []
        self._max: List[int] = []

    def push(self, value: int) -> None:
        self._data.append(value)
        if not self._max or value >= self._max[-1]:
            self._max.append(value)
        else:
            self._max.append(self._max[-1])

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty stack")
        self._max.pop()
        return self._data.pop()

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1]

    def get_max(self) -> int:
        if not self._max:
            raise IndexError("max from empty stack")
        return self._max[-1]

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)


class MinMaxStack:
    def __init__(self) -> None:
        self._data: List[Tuple[int, int, int]] = []

    def push(self, value: int) -> None:
        if not self._data:
            self._data.append((value, value, value))
        else:
            _, mn, mx = self._data[-1]
            mn = value if value < mn else mn
            mx = value if value > mx else mx
            self._data.append((value, mn, mx))

    def pop(self) -> int:
        if not self._data:
            raise IndexError("pop from empty stack")
        v, _, _ = self._data.pop()
        return v

    def peek(self) -> int:
        if not self._data:
            raise IndexError("peek from empty stack")
        return self._data[-1][0]

    def get_min(self) -> int:
        if not self._data:
            raise IndexError("min from empty stack")
        return self._data[-1][1]

    def get_max(self) -> int:
        if not self._data:
            raise IndexError("max from empty stack")
        return self._data[-1][2]

    def is_empty(self) -> bool:
        return not self._data

    def size(self) -> int:
        return len(self._data)


class _DLLNode:
    __slots__ = ("value", "prev", "next")

    def __init__(self, value: Any) -> None:
        self.value = value
        self.prev: Optional["_DLLNode"] = None
        self.next: Optional["_DLLNode"] = None


class MiddleStack:
    def __init__(self) -> None:
        self._head: Optional[_DLLNode] = None
        self._mid: Optional[_DLLNode] = None
        self._size = 0

    def push(self, value: Any) -> None:
        node = _DLLNode(value)
        node.next = self._head
        if self._head:
            self._head.prev = node
        self._head = node
        self._size += 1
        if self._size == 1:
            self._mid = node
        else:
            if self._size % 2 == 1:
                self._mid = self._mid.prev if self._mid and self._mid.prev else self._mid

    def pop(self) -> Any:
        if self._head is None:
            raise IndexError("pop from empty stack")
        node = self._head
        self._head = node.next
        if self._head:
            self._head.prev = None
        self._size -= 1
        if self._size == 0:
            self._mid = None
        else:
            if self._size % 2 == 0:
                self._mid = self._mid.next if self._mid and self._mid.next else self._mid
        return node.value

    def find_middle(self) -> Any:
        if self._mid is None:
            raise IndexError("find from empty stack")
        return self._mid.value

    def delete_middle(self) -> Any:
        if self._mid is None:
            raise IndexError("delete from empty stack")
        node = self._mid
        if node.prev:
            node.prev.next = node.next
        if node.next:
            node.next.prev = node.prev
        if node == self._head:
            self._head = node.next
        self._size -= 1
        if self._size == 0:
            self._mid = None
        else:
            if self._size % 2 == 0:
                self._mid = node.next if node.next else node.prev
            else:
                self._mid = node.prev if node.prev else node.next
        return node.value

    def size(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self._size == 0


class IncrementalStack:
    def __init__(self, capacity: int) -> None:
        self._cap = capacity
        self._data: List[int] = []
        self._inc: List[int] = []

    def push(self, value: int) -> None:
        if len(self._data) == self._cap:
            return
        self._data.append(value)
        self._inc.append(0)

    def pop(self) -> int:
        if not self._data:
            return -1
        idx = len(self._data) - 1
        inc_val = self._inc[idx]
        v = self._data.pop() + inc_val
        self._inc.pop()
        if self._inc:
            self._inc[-1] += inc_val
        return v

    def increment(self, k: int, val: int) -> None:
        if not self._data:
            return
        idx = min(k, len(self._data)) - 1
        if idx >= 0:
            self._inc[idx] += val

    def size(self) -> int:
        return len(self._data)

    def is_empty(self) -> bool:
        return not self._data


class CircularQueue:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self._data: List[Any] = [None] * capacity
        self._cap = capacity
        self._head = 0
        self._tail = 0
        self._size = 0

    def enqueue(self, value: Any) -> None:
        if self._size == self._cap:
            raise OverflowError("queue overflow")
        self._data[self._tail] = value
        self._tail = (self._tail + 1) % self._cap
        self._size += 1

    def dequeue(self) -> Any:
        if self._size == 0:
            raise IndexError("dequeue from empty queue")
        v = self._data[self._head]
        self._data[self._head] = None
        self._head = (self._head + 1) % self._cap
        self._size -= 1
        return v

    def peek(self) -> Any:
        if self._size == 0:
            raise IndexError("peek from empty queue")
        return self._data[self._head]

    def is_empty(self) -> bool:
        return self._size == 0

    def is_full(self) -> bool:
        return self._size == self._cap

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._data = [None] * self._cap
        self._head = 0
        self._tail = 0
        self._size = 0


class LinkedQueue:
    def __init__(self) -> None:
        self._head: Optional[_SLLNode] = None
        self._tail: Optional[_SLLNode] = None
        self._size = 0

    def enqueue(self, value: Any) -> None:
        node = _SLLNode(value)
        if self._tail:
            self._tail.next = node
        else:
            self._head = node
        self._tail = node
        self._size += 1

    def dequeue(self) -> Any:
        if self._head is None:
            raise IndexError("dequeue from empty queue")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._head is None:
            raise IndexError("peek from empty queue")
        return self._head.value

    def is_empty(self) -> bool:
        return self._head is None

    def size(self) -> int:
        return self._size

    def clear(self) -> None:
        self._head = None
        self._tail = None
        self._size = 0


class DequeLL:
    def __init__(self) -> None:
        self._head: Optional[_DLLNode] = None
        self._tail: Optional[_DLLNode] = None
        self._size = 0

    def append(self, value: Any) -> None:
        node = _DLLNode(value)
        if self._tail is None:
            self._head = node
            self._tail = node
        else:
            node.prev = self._tail
            self._tail.next = node
            self._tail = node
        self._size += 1

    def appendleft(self, value: Any) -> None:
        node = _DLLNode(value)
        if self._head is None:
            self._head = node
            self._tail = node
        else:
            node.next = self._head
            self._head.prev = node
            self._head = node
        self._size += 1

    def pop(self) -> Any:
        if self._tail is None:
            raise IndexError("pop from empty deque")
        node = self._tail
        self._tail = node.prev
        if self._tail is None:
            self._head = None
        else:
            self._tail.next = None
        self._size -= 1
        return node.value

    def popleft(self) -> Any:
        if self._head is None:
            raise IndexError("popleft from empty deque")
        node = self._head
        self._head = node.next
        if self._head is None:
            self._tail = None
        else:
            self._head.prev = None
        self._size -= 1
        return node.value

    def peek(self) -> Any:
        if self._tail is None:
            raise IndexError("peek from empty deque")
        return self._tail.value

    def peekleft(self) -> Any:
        if self._head is None:
            raise IndexError("peekleft from empty deque")
        return self._head.value

    def is_empty(self) -> bool:
        return self._size == 0

    def size(self) -> int:
        return self._size


class QueueUsingTwoStacks:
    def __init__(self) -> None:
        self._in: List[Any] = []
        self._out: List[Any] = []

    def enqueue(self, value: Any) -> None:
        self._in.append(value)

    def dequeue(self) -> Any:
        if not self._out:
            while self._in:
                self._out.append(self._in.pop())
        if not self._out:
            raise IndexError("dequeue from empty queue")
        return self._out.pop()

    def peek(self) -> Any:
        if not self._out:
            while self._in:
                self._out.append(self._in.pop())
        if not self._out:
            raise IndexError("peek from empty queue")
        return self._out[-1]

    def is_empty(self) -> bool:
        return not self._in and not self._out

    def size(self) -> int:
        return len(self._in) + len(self._out)

    def clear(self) -> None:
        self._in.clear()
        self._out.clear()


class MinQueue:
    def __init__(self) -> None:
        self._in = MinStack()
        self._out = MinStack()

    def enqueue(self, value: int) -> None:
        self._in.push(value)

    def dequeue(self) -> int:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        if self._out.is_empty():
            while not self._in.is_empty():
                self._out.push(self._in.pop())
        return self._out.pop()

    def get_min(self) -> int:
        if self.is_empty():
            raise IndexError("min from empty queue")
        if self._out.is_empty():
            return self._in.get_min()
        if self._in.is_empty():
            return self._out.get_min()
        a = self._in.get_min()
        b = self._out.get_min()
        return a if a < b else b

    def is_empty(self) -> bool:
        return self._in.size() + self._out.size() == 0

    def size(self) -> int:
        return self._in.size() + self._out.size()


class MaxQueue:
    def __init__(self) -> None:
        self._in = MaxStack()
        self._out = MaxStack()

    def enqueue(self, value: int) -> None:
        self._in.push(value)

    def dequeue(self) -> int:
        if self.is_empty():
            raise IndexError("dequeue from empty queue")
        if self._out.is_empty():
            while not self._in.is_empty():
                self._out.push(self._in.pop())
        return self._out.pop()

    def get_max(self) -> int:
        if self.is_empty():
            raise IndexError("max from empty queue")
        if self._out.is_empty():
            return self._in.get_max()
        if self._in.is_empty():
            return self._out.get_max()
        a = self._in.get_max()
        b = self._out.get_max()
        return a if a > b else b

    def is_empty(self) -> bool:
        return self._in.size() + self._out.size() == 0

    def size(self) -> int:
        return self._in.size() + self._out.size()


class MonotonicQueue:
    def __init__(self, decreasing: bool = True) -> None:
        self._dq: Deque[int] = deque()
        self._decreasing = decreasing

    def push(self, value: int) -> None:
        if self._decreasing:
            while self._dq and self._dq[-1] < value:
                self._dq.pop()
        else:
            while self._dq and self._dq[-1] > value:
                self._dq.pop()
        self._dq.append(value)

    def pop(self, value: int) -> None:
        if self._dq and self._dq[0] == value:
            self._dq.popleft()

    def front(self) -> int:
        if not self._dq:
            raise IndexError("front from empty queue")
        return self._dq[0]

    def is_empty(self) -> bool:
        return not self._dq

    def size(self) -> int:
        return len(self._dq)


class FrontMiddleBackQueue:
    def __init__(self) -> None:
        self._left: Deque[int] = deque()
        self._right: Deque[int] = deque()

    def _balance(self) -> None:
        if len(self._left) > len(self._right) + 1:
            self._right.appendleft(self._left.pop())
        elif len(self._left) < len(self._right):
            self._left.append(self._right.popleft())

    def pushFront(self, val: int) -> None:
        self._left.appendleft(val)
        self._balance()

    def pushMiddle(self, val: int) -> None:
        if len(self._left) > len(self._right):
            self._right.appendleft(self._left.pop())
        self._left.append(val)

    def pushBack(self, val: int) -> None:
        self._right.append(val)
        self._balance()

    def popFront(self) -> int:
        if not self._left and not self._right:
            return -1
        if self._left:
            v = self._left.popleft()
        else:
            v = self._right.popleft()
        self._balance()
        return v

    def popMiddle(self) -> int:
        if not self._left and not self._right:
            return -1
        v = self._left.pop()
        self._balance()
        return v

    def popBack(self) -> int:
        if not self._left and not self._right:
            return -1
        if self._right:
            v = self._right.pop()
        else:
            v = self._left.pop()
        self._balance()
        return v


class RecentCounter:
    def __init__(self) -> None:
        self._q: Deque[int] = deque()

    def ping(self, t: int) -> int:
        self._q.append(t)
        while self._q and self._q[0] < t - 3000:
            self._q.popleft()
        return len(self._q)


class BrowserHistory:
    def __init__(self, homepage: str) -> None:
        self._current = homepage
        self._back: List[str] = []
        self._forward: List[str] = []

    def visit(self, url: str) -> None:
        self._back.append(self._current)
        self._current = url
        self._forward.clear()

    def back(self, steps: int) -> str:
        while steps and self._back:
            self._forward.append(self._current)
            self._current = self._back.pop()
            steps -= 1
        return self._current

    def forward(self, steps: int) -> str:
        while steps and self._forward:
            self._back.append(self._current)
            self._current = self._forward.pop()
            steps -= 1
        return self._current


def is_valid_parentheses(s: str) -> bool:
    stack: List[str] = []
    pairs = {')': '(', ']': '[', '}': '{'}
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack[-1] != pairs[ch]:
                return False
            stack.pop()
    return not stack


def longest_valid_parentheses(s: str) -> int:
    ans = 0
    stack: List[int] = [-1]
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        else:
            stack.pop()
            if not stack:
                stack.append(i)
            else:
                ans = max(ans, i - stack[-1])
    return ans


def next_greater(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [-1] * n
    st: List[int] = []
    for i, v in enumerate(arr):
        while st and arr[st[-1]] < v:
            res[st.pop()] = v
        st.append(i)
    return res


def prev_greater(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [-1] * n
    st: List[int] = []
    for i, v in enumerate(arr):
        while st and arr[st[-1]] <= v:
            st.pop()
        res[i] = arr[st[-1]] if st else -1
        st.append(i)
    return res


def next_smaller(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [-1] * n
    st: List[int] = []
    for i, v in enumerate(arr):
        while st and arr[st[-1]] > v:
            res[st.pop()] = v
        st.append(i)
    return res


def prev_smaller(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [-1] * n
    st: List[int] = []
    for i, v in enumerate(arr):
        while st and arr[st[-1]] >= v:
            st.pop()
        res[i] = arr[st[-1]] if st else -1
        st.append(i)
    return res


def next_greater_circular(arr: List[int]) -> List[int]:
    n = len(arr)
    res = [-1] * n
    st: List[int] = []
    for i in range(2 * n):
        v = arr[i % n]
        while st and arr[st[-1]] < v:
            res[st.pop()] = v
        if i < n:
            st.append(i)
    return res


def stock_span(prices: List[int]) -> List[int]:
    n = len(prices)
    res = [0] * n
    st: List[int] = []
    for i, p in enumerate(prices):
        while st and prices[st[-1]] <= p:
            st.pop()
        res[i] = i - st[-1] if st else i + 1
        st.append(i)
    return res


def daily_temperatures(temps: List[int]) -> List[int]:
    n = len(temps)
    res = [0] * n
    st: List[int] = []
    for i, t in enumerate(temps):
        while st and temps[st[-1]] < t:
            j = st.pop()
            res[j] = i - j
        st.append(i)
    return res


def largest_rectangle_histogram(heights: List[int]) -> int:
    st: List[int] = []
    ans = 0
    heights.append(0)
    for i, h in enumerate(heights):
        while st and heights[st[-1]] > h:
            H = heights[st.pop()]
            L = st[-1] if st else -1
            ans = max(ans, H * (i - L - 1))
        st.append(i)
    heights.pop()
    return ans


def maximal_rectangle(matrix: List[List[str]]) -> int:
    if not matrix or not matrix[0]:
        return 0
    n = len(matrix[0])
    heights = [0] * n
    ans = 0
    for row in matrix:
        for j in range(n):
            heights[j] = heights[j] + 1 if row[j] == '1' else 0
        ans = max(ans, largest_rectangle_histogram(heights))
    return ans


def trap_rain_water(height: List[int]) -> int:
    st: List[int] = []
    water = 0
    for i, h in enumerate(height):
        while st and height[st[-1]] < h:
            mid = st.pop()
            if not st:
                break
            left = st[-1]
            bounded = min(height[left], h) - height[mid]
            width = i - left - 1
            water += bounded * width
        st.append(i)
    return water


def simplify_unix_path(path: str) -> str:
    parts = path.split("/")
    st: List[str] = []
    for p in parts:
        if p == "" or p == ".":
            continue
        if p == "..":
            if st:
                st.pop()
        else:
            st.append(p)
    return "/" + "/".join(st)


def remove_k_digits(num: str, k: int) -> str:
    st: List[str] = []
    for ch in num:
        while k and st and st[-1] > ch:
            st.pop()
            k -= 1
        st.append(ch)
    while k and st:
        st.pop()
        k -= 1
    s = "".join(st).lstrip("0")
    return s if s else "0"


def decode_string(s: str) -> str:
    count_stack: List[int] = []
    str_stack: List[str] = []
    curr = []
    k = 0
    for ch in s:
        if ch.isdigit():
            k = k * 10 + ord(ch) - 48
        elif ch == '[':
            count_stack.append(k)
            str_stack.append("".join(curr))
            curr = []
            k = 0
        elif ch == ']':
            times = count_stack.pop()
            prev = str_stack.pop()
            curr = [prev + "".join(curr) * times]
        else:
            curr.append(ch)
    return "".join(curr)


def evaluate_rpn(tokens: List[str]) -> int:
    st: List[int] = []
    for tok in tokens:
        if tok in {"+", "-", "*", "/"}:
            b = st.pop()
            a = st.pop()
            if tok == "+":
                st.append(a + b)
            elif tok == "-":
                st.append(a - b)
            elif tok == "*":
                st.append(a * b)
            else:
                st.append(int(a / b))
        else:
            st.append(int(tok))
    return st[-1]


def _tokenize(expression: str) -> List[str]:
    tokens: List[str] = []
    i = 0
    n = len(expression)
    prev: Optional[str] = None
    while i < n:
        ch = expression[i]
        if ch.isspace():
            i += 1
            continue
        if ch.isdigit():
            j = i
            while j < n and expression[j].isdigit():
                j += 1
            tokens.append(expression[i:j])
            prev = "num"
            i = j
        elif ch in "+-*/^()":
            if ch == '-' and (prev is None or prev in "+-*/^("):
                tokens.append("u-")
            else:
                tokens.append(ch)
            prev = ch if ch != '(' else "("
            i += 1
        else:
            i += 1
    return tokens


def infix_to_postfix(expression: str) -> List[str]:
    tokens = _tokenize(expression)
    out: List[str] = []
    st: List[str] = []
    prec = {"u-": 5, "^": 4, "*": 3, "/": 3, "+": 2, "-": 2}
    right_assoc = {"^", "u-"}
    for tok in tokens:
        if tok.isdigit():
            out.append(tok)
        elif tok == "(":
            st.append(tok)
        elif tok == ")":
            while st and st[-1] != "(":
                out.append(st.pop())
            if st and st[-1] == "(":
                st.pop()
        else:
            while st and st[-1] != "(" and (prec[st[-1]] > prec[tok] or (prec[st[-1]] == prec[tok] and tok not in right_assoc)):
                out.append(st.pop())
            st.append(tok)
    while st:
        out.append(st.pop())
    return out


def evaluate_infix(expression: str) -> int:
    tokens = _tokenize(expression)
    vals: List[int] = []
    ops: List[str] = []

    def apply() -> None:
        op = ops.pop()
        if op == "u-":
            a = vals.pop()
            vals.append(-a)
            return
        b = vals.pop()
        a = vals.pop()
        if op == "+":
            vals.append(a + b)
        elif op == "-":
            vals.append(a - b)
        elif op == "*":
            vals.append(a * b)
        elif op == "/":
            vals.append(int(a / b))
        elif op == "^":
            vals.append(pow(a, b))

    prec = {"u-": 5, "^": 4, "*": 3, "/": 3, "+": 2, "-": 2}
    right_assoc = {"^", "u-"}
    for tok in tokens:
        if tok.isdigit():
            vals.append(int(tok))
        elif tok == "(":
            ops.append(tok)
        elif tok == ")":
            while ops and ops[-1] != "(":
                apply()
            if ops and ops[-1] == "(":
                ops.pop()
        else:
            while ops and ops[-1] != "(" and (prec[ops[-1]] > prec[tok] or (prec[ops[-1]] == prec[tok] and tok not in right_assoc)):
                apply()
            ops.append(tok)
    while ops:
        apply()
    return vals[-1] if vals else 0


def infix_to_prefix(expression: str) -> List[str]:
    tokens = _tokenize(expression)
    tokens.reverse()
    swapped: List[str] = []
    for tok in tokens:
        if tok == "(":
            swapped.append(")")
        elif tok == ")":
            swapped.append("(")
        else:
            swapped.append(tok)
    postfix = infix_to_postfix(" ".join(swapped))
    postfix.reverse()
    return postfix


def evaluate_prefix(tokens: List[str]) -> int:
    st: List[int] = []
    for tok in reversed(tokens):
        if tok in {"+", "-", "*", "/"}:
            a = st.pop()
            b = st.pop()
            if tok == "+":
                st.append(a + b)
            elif tok == "-":
                st.append(a - b)
            elif tok == "*":
                st.append(a * b)
            else:
                st.append(int(a / b))
        else:
            st.append(int(tok))
    return st[-1]


def sort_stack_using_recursion(stack: List[int]) -> None:
    def insert_sorted(x: int) -> None:
        if not stack or stack[-1] <= x:
            stack.append(x)
            return
        y = stack.pop()
        insert_sorted(x)
        stack.append(y)

    if not stack:
        return
    t = stack.pop()
    sort_stack_using_recursion(stack)
    insert_sorted(t)


def reverse_stack_using_recursion(stack: List[Any]) -> None:
    def insert_bottom(x: Any) -> None:
        if not stack:
            stack.append(x)
            return
        y = stack.pop()
        insert_bottom(x)
        stack.append(y)

    if not stack:
        return
    t = stack.pop()
    reverse_stack_using_recursion(stack)
    insert_bottom(t)


def clone_stack_using_recursion(stack: List[Any]) -> List[Any]:
    aux: List[Any] = []

    def clone() -> Any:
        if not stack:
            return None
        x = stack.pop()
        y = clone()
        aux.append(x)
        return y

    clone()
    res = aux[:]
    while aux:
        stack.append(aux.pop())
    return res


def validate_stack_sequences(pushed: List[int], popped: List[int]) -> bool:
    st: List[int] = []
    j = 0
    for x in pushed:
        st.append(x)
        while st and j < len(popped) and st[-1] == popped[j]:
            st.pop()
            j += 1
    return j == len(popped)


def asteroid_collision(asteroids: List[int]) -> List[int]:
    st: List[int] = []
    for a in asteroids:
        alive = True
        while alive and a < 0 and st and st[-1] > 0:
            if st[-1] < -a:
                st.pop()
                continue
            if st[-1] == -a:
                st.pop()
            alive = False
        if alive:
            st.append(a)
    return st


def smallest_subsequence_distinct(s: str) -> str:
    last: Dict[str, int] = {c: i for i, c in enumerate(s)}
    used: Dict[str, bool] = {}
    st: List[str] = []
    for i, c in enumerate(s):
        if used.get(c, False):
            continue
        while st and st[-1] > c and last[st[-1]] > i:
            used[st.pop()] = False
        st.append(c)
        used[c] = True
    return "".join(st)


def remove_adjacent_duplicates_k(s: str, k: int) -> str:
    st: List[Tuple[str, int]] = []
    for ch in s:
        if st and st[-1][0] == ch:
            ch_, cnt = st.pop()
            cnt += 1
            if cnt < k:
                st.append((ch_, cnt))
        else:
            st.append((ch, 1))
    return "".join(c * cnt for c, cnt in st)


def score_of_parentheses(s: str) -> int:
    st = [0]
    for ch in s:
        if ch == '(':
            st.append(0)
        else:
            v = st.pop()
            st[-1] += max(2 * v, 1)
    return st[-1]


def car_fleet(target: int, position: List[int], speed: List[int]) -> int:
    pairs = sorted(zip(position, speed), reverse=True)
    fleets = 0
    last_time = 0.0
    for pos, spd in pairs:
        time = (target - pos) / spd
        if time > last_time:
            fleets += 1
            last_time = time
    return fleets


def exclusive_time_of_functions(n: int, logs: List[str]) -> List[int]:
    res = [0] * n
    st: List[int] = []
    prev_time = 0
    for log in logs:
        fid_str, typ, t_str = log.split(":")
        fid, t = int(fid_str), int(t_str)
        if typ == "start":
            if st:
                res[st[-1]] += t - prev_time
            st.append(fid)
            prev_time = t
        else:
            res[st.pop()] += t - prev_time + 1
            prev_time = t + 1
    return res


def find_132_pattern(nums: List[int]) -> bool:
    third = float("-inf")
    st: List[int] = []
    for x in reversed(nums):
        if x < third:
            return True
        while st and x > st[-1]:
            third = st.pop()
        st.append(x)
    return False


def sliding_window_max(nums: List[int], k: int) -> List[int]:
    dq: Deque[int] = deque()
    res: List[int] = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] <= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


def sliding_window_min(nums: List[int], k: int) -> List[int]:
    dq: Deque[int] = deque()
    res: List[int] = []
    for i, x in enumerate(nums):
        while dq and dq[0] <= i - k:
            dq.popleft()
        while dq and nums[dq[-1]] >= x:
            dq.pop()
        dq.append(i)
        if i >= k - 1:
            res.append(nums[dq[0]])
    return res


def first_negative_in_window(nums: List[int], k: int) -> List[int]:
    dq: Deque[int] = deque()
    res: List[int] = []
    for i, x in enumerate(nums):
        if x < 0:
            dq.append(i)
        while dq and dq[0] <= i - k:
            dq.popleft()
        if i >= k - 1:
            res.append(nums[dq[0]] if dq else 0)
    return res


def shortest_subarray_at_least_k(nums: List[int], k: int) -> int:
    n = len(nums)
    pref = [0] * (n + 1)
    for i in range(n):
        pref[i + 1] = pref[i] + nums[i]
    dq: Deque[int] = deque()
    ans = n + 1
    for j in range(n + 1):
        while dq and pref[j] - pref[dq[0]] >= k:
            ans = min(ans, j - dq.popleft())
        while dq and pref[j] <= pref[dq[-1]]:
            dq.pop()
        dq.append(j)
    return ans if ans <= n else -1


def topological_sort(num_nodes: int, edges: List[Tuple[int, int]]) -> List[int]:
    adj: List[List[int]] = [[] for _ in range(num_nodes)]
    indeg = [0] * num_nodes
    for u, v in edges:
        adj[u].append(v)
        indeg[v] += 1
    dq: Deque[int] = deque([i for i in range(num_nodes) if indeg[i] == 0])
    order: List[int] = []
    while dq:
        u = dq.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                dq.append(v)
    return order if len(order) == num_nodes else []


def bfs_shortest_path(adj: List[List[int]], source: int) -> List[int]:
    n = len(adj)
    dist = [-1] * n
    dq: Deque[int] = deque([source])
    dist[source] = 0
    while dq:
        u = dq.popleft()
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                dq.append(v)
    return dist


def rotten_oranges(grid: List[List[int]]) -> int:
    m = len(grid)
    n = len(grid[0]) if m else 0
    dq: Deque[Tuple[int, int]] = deque()
    fresh = 0
    for i in range(m):
        for j in range(n):
            if grid[i][j] == 2:
                dq.append((i, j))
            elif grid[i][j] == 1:
                fresh += 1
    minutes = 0
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    while dq and fresh:
        for _ in range(len(dq)):
            i, j = dq.popleft()
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < m and 0 <= nj < n and grid[ni][nj] == 1:
                    grid[ni][nj] = 2
                    fresh -= 1
                    dq.append((ni, nj))
        minutes += 1
    return minutes if fresh == 0 else -1


class MovingAverage:
    def __init__(self, size: int) -> None:
        self._size = size
        self._q: Deque[int] = deque()
        self._sum = 0

    def next(self, val: int) -> float:
        self._q.append(val)
        self._sum += val
        if len(self._q) > self._size:
            self._sum -= self._q.popleft()
        return self._sum / len(self._q)


def sum_of_subarray_mins(arr: List[int]) -> int:
    n = len(arr)
    prev = [-1] * n
    next_ = [n] * n
    st: List[int] = []
    for i in range(n):
        while st and arr[st[-1]] >= arr[i]:
            st.pop()
        prev[i] = st[-1] if st else -1
        st.append(i)
    st.clear()
    for i in range(n):
        while st and arr[i] <= arr[st[-1]]:
            idx = st.pop()
            next_[idx] = i
        st.append(i)
    total = 0
    for i in range(n):
        total += arr[i] * (i - prev[i]) * (next_[i] - i)
    return total


def sum_of_subarray_maxs(arr: List[int]) -> int:
    n = len(arr)
    prev = [-1] * n
    next_ = [n] * n
    st: List[int] = []
    for i in range(n):
        while st and arr[st[-1]] <= arr[i]:
            st.pop()
        prev[i] = st[-1] if st else -1
        st.append(i)
    st.clear()
    for i in range(n):
        while st and arr[i] >= arr[st[-1]]:
            idx = st.pop()
            next_[idx] = i
        st.append(i)
    total = 0
    for i in range(n):
        total += arr[i] * (i - prev[i]) * (next_[i] - i)
    return total


def sum_of_subarray_ranges(arr: List[int]) -> int:
    return sum_of_subarray_maxs(arr) - sum_of_subarray_mins(arr)


def max_of_min_for_window_sizes(arr: List[int]) -> List[int]:
    n = len(arr)
    prev = [-1] * n
    next_ = [n] * n
    st: List[int] = []
    for i in range(n):
        while st and arr[st[-1]] >= arr[i]:
            st.pop()
        prev[i] = st[-1] if st else -1
        st.append(i)
    st.clear()
    for i in range(n):
        while st and arr[i] < arr[st[-1]]:
            idx = st.pop()
            next_[idx] = i
        st.append(i)
    res = [float("-inf")] * (n + 1)
    for i in range(n):
        length = next_[i] - prev[i] - 1
        if arr[i] > res[length]:
            res[length] = arr[i]
    for i in range(n - 1, 0, -1):
        if res[i] < res[i + 1]:
            res[i] = res[i + 1]
    return res[1:]


def min_add_to_make_valid(s: str) -> int:
    bal = 0
    add = 0
    for ch in s:
        if ch == '(':
            bal += 1
        else:
            if bal == 0:
                add += 1
            else:
                bal -= 1
    return add + bal


def min_remove_to_make_valid(s: str) -> str:
    st: List[int] = []
    remove = set()
    for i, ch in enumerate(s):
        if ch == '(':
            st.append(i)
        elif ch == ')':
            if st:
                st.pop()
            else:
                remove.add(i)
    remove.update(st)
    res = []
    for i, ch in enumerate(s):
        if i not in remove:
            res.append(ch)
    return "".join(res)


def redundant_braces(expression: str) -> bool:
    st: List[str] = []
    ops = set("+-*/")
    for ch in expression:
        if ch == ')':
            has_op = False
            while st and st[-1] != '(':
                if st[-1] in ops:
                    has_op = True
                st.pop()
            if st:
                st.pop()
            if not has_op:
                return True
        else:
            st.append(ch)
    return False


def reverse_first_k_queue(arr: List[int], k: int) -> List[int]:
    q = deque(arr)
    st: List[int] = []
    for _ in range(min(k, len(q))):
        st.append(q.popleft())
    while st:
        q.append(st.pop())
    for _ in range(len(q) - k):
        q.append(q.popleft())
    return list(q)


def generate_binary_numbers(n: int) -> List[str]:
    res: List[str] = []
    q: Deque[str] = deque(["1"])
    for _ in range(n):
        s = q.popleft()
        res.append(s)
        q.append(s + "0")
        q.append(s + "1")
    return res


def zero_one_bfs(n: int, edges: List[Tuple[int, int, int]], source: int) -> List[int]:
    adj: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for u, v, w in edges:
        adj[u].append((v, w))
        adj[v].append((u, w))
    dist = [10**18] * n
    dist[source] = 0
    dq: Deque[int] = deque([source])
    while dq:
        u = dq.popleft()
        for v, w in adj[u]:
            nd = dist[u] + w
            if nd < dist[v]:
                dist[v] = nd
                if w == 0:
                    dq.appendleft(v)
                else:
                    dq.append(v)
    return dist


def shortest_path_in_binary_matrix(grid: List[List[int]]) -> int:
    n = len(grid)
    if n == 0 or grid[0][0] == 1 or grid[-1][-1] == 1:
        return -1
    dq: Deque[Tuple[int, int]] = deque([(0, 0)])
    grid[0][0] = 1
    steps = 1
    dirs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while dq:
        for _ in range(len(dq)):
            i, j = dq.popleft()
            if i == n - 1 and j == n - 1:
                return steps
            for di, dj in dirs:
                ni, nj = i + di, j + dj
                if 0 <= ni < n and 0 <= nj < n and grid[ni][nj] == 0:
                    grid[ni][nj] = 1
                    dq.append((ni, nj))
        steps += 1
    return -1


if __name__ == "__main__":
    s = ArrayStack()
    s.push(1)
    s.push(2)
    print("ArrayStack pop", s.pop(), "# expected", 2)
    ls = LinkedStack()
    ls.push("a")
    ls.push("b")
    print("LinkedStack peek", ls.peek(), "# expected", "b")
    ts = TwoStacksOneArray(5)
    ts.push1(1)
    ts.push2(9)
    print("TwoStacksOneArray pop1", ts.pop1(), "# expected", 1)
    ks = KStacksOneArray(3, 5)
    ks.push(0, 10)
    ks.push(1, 20)
    ks.push(2, 30)
    print("KStacksOneArray pop idx1", ks.pop(1), "# expected", 20)
    sp = StackPushEfficient()
    sp.push(5)
    sp.push(7)
    print("StackPushEfficient pop", sp.pop(), "# expected", 7)
    spe = StackPopEfficient()
    spe.push(3)
    spe.push(4)
    print("StackPopEfficient peek", spe.peek(), "# expected", 4)
    ms = MinStack()
    ms.push(3)
    ms.push(5)
    ms.push(2)
    print("MinStack get_min", ms.get_min(), "# expected", 2)
    ems = EncodedMinStack()
    ems.push(5)
    ems.push(2)
    ems.push(4)
    print("EncodedMinStack get_min", ems.get_min(), "# expected", 2)
    mxs = MaxStack()
    mxs.push(1)
    mxs.push(7)
    mxs.push(3)
    print("MaxStack get_max", mxs.get_max(), "# expected", 7)
    mms = MinMaxStack()
    mms.push(4)
    mms.push(2)
    mms.push(9)
    print("MinMaxStack min,max", mms.get_min(), mms.get_max(), "# expected", 2, 9)
    mid = MiddleStack()
    mid.push(1)
    mid.push(2)
    mid.push(3)
    print("MiddleStack find_middle", mid.find_middle(), "# expected", 2)
    print("MiddleStack delete_middle", mid.delete_middle(), "# expected", 2)
    inc = IncrementalStack(5)
    inc.push(1)
    inc.push(2)
    inc.increment(2, 3)
    print("IncrementalStack pop", inc.pop(), "# expected", 5)
    cq = CircularQueue(3)
    cq.enqueue(1)
    cq.enqueue(2)
    print("CircularQueue dequeue", cq.dequeue(), "# expected", 1)
    lq = LinkedQueue()
    lq.enqueue("x")
    lq.enqueue("y")
    print("LinkedQueue peek", lq.peek(), "# expected", "x")
    dqll = DequeLL()
    dqll.append(1)
    dqll.appendleft(0)
    print("DequeLL pop,popleft", dqll.pop(), dqll.popleft(), "# expected", 1, 0)
    q2s = QueueUsingTwoStacks()
    q2s.enqueue(10)
    q2s.enqueue(20)
    print("QueueUsingTwoStacks dequeue", q2s.dequeue(), "# expected", 10)
    minq = MinQueue()
    minq.enqueue(5)
    minq.enqueue(1)
    minq.enqueue(3)
    print("MinQueue get_min", minq.get_min(), "# expected", 1)
    maxq = MaxQueue()
    maxq.enqueue(2)
    maxq.enqueue(9)
    maxq.enqueue(4)
    print("MaxQueue get_max", maxq.get_max(), "# expected", 9)
    fm = FrontMiddleBackQueue()
    fm.pushFront(1)
    fm.pushBack(2)
    fm.pushMiddle(3)
    print("FrontMiddleBackQueue popMiddle", fm.popMiddle(), "# expected", 3)
    rc = RecentCounter()
    print("RecentCounter", rc.ping(1), rc.ping(100), rc.ping(3001), rc.ping(3002), "# expected", 1, 2, 3, 3)
    bh = BrowserHistory("a.com")
    bh.visit("b.com")
    bh.visit("c.com")
    print("BrowserHistory back,forward", bh.back(1), bh.forward(1), "# expected", "b.com", "c.com")
    print("is_valid_parentheses", "()[]{}", is_valid_parentheses("()[]{}"), "# expected", True)
    print("longest_valid_parentheses", ")(()())", longest_valid_parentheses(")(()())"), "# expected", 6)
    arr = [2, 1, 2, 4, 3]
    print("next_greater", arr, next_greater(arr), "# expected", [4, 2, 4, -1, -1])
    print("prev_greater", arr, prev_greater(arr), "# expected", [-1, 2, -1, -1, 4])
    print("next_smaller", arr, next_smaller(arr), "# expected", [1, -1, -1, 3, -1])
    print("prev_smaller", arr, prev_smaller(arr), "# expected", [-1, -1, 1, 2, 2])
    print("next_greater_circular", [1, 2, 1], next_greater_circular([1, 2, 1]), "# expected", [2, -1, 2])
    print("stock_span", [100, 80, 60, 70, 60, 75, 85], stock_span([100, 80, 60, 70, 60, 75, 85]), "# expected", [1, 1, 1, 2, 1, 4, 6])
    print("daily_temperatures", [73, 74, 75, 71, 69, 72, 76, 73], daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]), "# expected", [1, 1, 1, 2, 1, 1, 0, 0])
    print("largest_rectangle_histogram", [2, 1, 5, 6, 2, 3], largest_rectangle_histogram([2, 1, 5, 6, 2, 3]), "# expected", 10)
    matrix = [["1","0","1","0","0"],["1","0","1","1","1"],["1","1","1","1","1"],["1","0","0","1","0"]]
    print("maximal_rectangle", maximal_rectangle(matrix), "# expected", 6)
    print("trap_rain_water", [0,1,0,2,1,0,1,3,2,1,2,1], trap_rain_water([0,1,0,2,1,0,1,3,2,1,2,1]), "# expected", 6)
    print("simplify_unix_path", "/a/./b/../../c/", simplify_unix_path("/a/./b/../../c/"), "# expected", "/c")
    print("remove_k_digits", "1432219", 3, remove_k_digits("1432219", 3), "# expected", "1219")
    print("decode_string", "3[a2[c]]", decode_string("3[a2[c]]"), "# expected", "accaccacc")
    print("evaluate_rpn", ["2", "1", "+", "3", "*"], evaluate_rpn(["2", "1", "+", "3", "*"]), "# expected", 9)
    expr = "3 + 5 * (2 - 4) ^ 2"
    print("infix_to_postfix", expr, infix_to_postfix(expr), "# expected", ['3','5','2','4','-','2','^','*','+'])
    print("evaluate_infix", expr, evaluate_infix(expr), "# expected", 23)
    print("infix_to_prefix", expr, infix_to_prefix(expr), "# expected", ['+','3','*','5','^','-','2','4','2'])
    print("evaluate_prefix", ['+','3','*','5','^','-','2','4','2'], evaluate_prefix(['+','3','*','5','^','-','2','4','2']), "# expected", 23)
    st = [3, 1, 4, 2]
    sort_stack_using_recursion(st)
    print("sort_stack_using_recursion", st, "# expected", [1, 2, 3, 4])
    st2 = [1, 2, 3]
    reverse_stack_using_recursion(st2)
    print("reverse_stack_using_recursion", st2, "# expected", [3, 2, 1])
    st3 = [1, 2, 3]
    cloned = clone_stack_using_recursion(st3)
    print("clone_stack_using_recursion", cloned, st3, "# expected", [1,2,3], [1,2,3])
    print("validate_stack_sequences", [1,2,3,4,5], [4,5,3,2,1], validate_stack_sequences([1,2,3,4,5],[4,5,3,2,1]), "# expected", True)
    print("asteroid_collision", [5,10,-5], asteroid_collision([5,10,-5]), "# expected", [5,10])
    print("smallest_subsequence_distinct", "cbacdcbc", smallest_subsequence_distinct("cbacdcbc"), "# expected", "acdb")
    print("remove_adjacent_duplicates_k", "deeedbbcccbdaa", 3, remove_adjacent_duplicates_k("deeedbbcccbdaa", 3), "# expected", "aa")
    print("score_of_parentheses", "(()(()))", score_of_parentheses("(()(()))"), "# expected", 6)
    print("car_fleet", 12, [10,8,0,5,3], [2,4,1,1,3], car_fleet(12,[10,8,0,5,3],[2,4,1,1,3]), "# expected", 3)
    n = 2
    logs = ["0:start:0","1:start:2","1:end:5","0:end:6"]
    print("exclusive_time_of_functions", n, logs, exclusive_time_of_functions(n, logs), "# expected", [3,4])
    print("find_132_pattern", [3,1,4,2], find_132_pattern([3,1,4,2]), "# expected", True)
    print("sliding_window_max", [1,3,-1,-3,5,3,6,7], 3, sliding_window_max([1,3,-1,-3,5,3,6,7], 3), "# expected", [3,3,5,5,6,7])
    print("sliding_window_min", [1,3,-1,-3,5,3,6,7], 3, sliding_window_min([1,3,-1,-3,5,3,6,7], 3), "# expected", [-1,-3,-3,-3,3])
    print("first_negative_in_window", [12,-1,-7,8,-15,30,16,28], 3, first_negative_in_window([12,-1,-7,8,-15,30,16,28], 3), "# expected", [-1,-1,-7,-15,-15,0])
    print("shortest_subarray_at_least_k", [2,-1,2], 3, shortest_subarray_at_least_k([2,-1,2], 3), "# expected", 3)
    print("topological_sort", 6, [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)], topological_sort(6, [(5,2),(5,0),(4,0),(4,1),(2,3),(3,1)]), "# expected", [4,5,0,2,3,1])
    adj = [[1,2],[2],[3],[]]
    print("bfs_shortest_path", adj, 0, bfs_shortest_path(adj, 0), "# expected", [0,1,1,2])
    grid = [[2,1,1],[1,1,0],[0,1,1]]
    print("rotten_oranges", rotten_oranges(grid), "# expected", 4)
    ma = MovingAverage(3)
    print("MovingAverage", ma.next(1), ma.next(10), ma.next(3), ma.next(5), "# expected", 1.0, 5.5, 4.666666666666667, 6.0)
    arr2 = [3,1,2,4]
    print("sum_of_subarray_mins", arr2, sum_of_subarray_mins(arr2), "# expected", 17)
    print("sum_of_subarray_maxs", arr2, sum_of_subarray_maxs(arr2), "# expected", 30)
    print("sum_of_subarray_ranges", arr2, sum_of_subarray_ranges(arr2), "# expected", 13)
    arr3 = [10,20,30,50,10,70,30]
    print("max_of_min_for_window_sizes", arr3, max_of_min_for_window_sizes(arr3), "# expected", [70,30,20,10,10,10,10])
    print("min_add_to_make_valid", "()))((", min_add_to_make_valid("()))(("), "# expected", 4)
    print("min_remove_to_make_valid", "a)b(c)d", min_remove_to_make_valid("a)b(c)d"), "# expected", "ab(c)d")
    print("redundant_braces", "((a+b))", redundant_braces("((a+b))"), "# expected", True)
    print("reverse_first_k_queue", [1,2,3,4,5], 3, reverse_first_k_queue([1,2,3,4,5], 3), "# expected", [3,2,1,4,5])
    print("generate_binary_numbers", 5, generate_binary_numbers(5), "# expected", ["1","10","11","100","101"])
    edges = [(0,1,0),(1,2,1),(0,2,1),(2,3,0)]
    print("zero_one_bfs", 4, edges, 0, zero_one_bfs(4, edges, 0), "# expected", [0,0,1,1])
    grid2 = [[0,1],[1,0]]
    print("shortest_path_in_binary_matrix", grid2, shortest_path_in_binary_matrix([row[:] for row in grid2]), "# expected", 2)
    mq = MonotonicQueue(decreasing=True)
    seq = [1,3,2,5,4]
    outs = []
    for x in seq:
        mq.push(x)
        outs.append(mq.front())
    mq.pop(1)
    outs.append(mq.front())
    print("MonotonicQueue stream", seq, outs, "# expected", [1,3,3,5,5,5])