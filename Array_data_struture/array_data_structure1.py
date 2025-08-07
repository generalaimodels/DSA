"""
================================================================================
ARRAY DATA STRUCTURE – END-TO-END MASTERCLASS
================================================================================
Author : 
File   : array_master.py
Scope  : Every medium-to-advanced algorithm & pattern that revolves around
         the humble “array” (a.k.a. contiguous block of memory).
Language: Python 3.11+ (PEP-8 compliant, type-hinted, zero external deps)
================================================================================
"""

from __future__ import annotations
import bisect
import heapq
import itertools
import math
import random
from typing import List, Tuple, Dict, Any, Optional, Callable, Iterator

# ------------------------------------------------------------------------------
# 0.  UTILITIES
# ------------------------------------------------------------------------------
def _swap(arr: List[Any], i: int, j: int) -> None:
    """In-place swap two indices."""
    arr[i], arr[j] = arr[j], arr[i]


# ------------------------------------------------------------------------------
# 1.  BASIC OPERATIONS (O(1) & O(n))
# ------------------------------------------------------------------------------
class StaticArray:
    """
    A fixed-capacity array (no resize) to demonstrate raw memory semantics.
    """
    def __init__(self, capacity: int):
        self._data: List[Optional[int]] = [None] * capacity
        self._size = 0

    def set(self, idx: int, val: int) -> None:
        if not 0 <= idx < len(self._data):
            raise IndexError("StaticArray index out of range")
        self._data[idx] = val

    def get(self, idx: int) -> int:
        if not 0 <= idx < len(self._data):
            raise IndexError("StaticArray index out of range")
        return self._data[idx]

    def __repr__(self) -> str:
        return str(self._data)


# ------------------------------------------------------------------------------
# 2.  DYNAMIC ARRAY (PYTHON LIST UNDER THE HOOD)
# ------------------------------------------------------------------------------
class DynamicArray:
    """
    Re-implements Python list growth strategy (geometric resize).
    """
    def __init__(self):
        self._capacity = 1
        self._size = 0
        self._data = self._make_array(self._capacity)

    def _make_array(self, capacity: int) -> List[Optional[int]]:
        return [None] * capacity

    def _resize(self, new_cap: int) -> None:
        new_arr = self._make_array(new_cap)
        for k in range(self._size):
            new_arr[k] = self._data[k]
        self._data = new_arr
        self._capacity = new_cap

    def append(self, val: int) -> None:
        if self._size == self._capacity:
            self._resize(2 * self._capacity)
        self._data[self._size] = val
        self._size += 1

    def pop(self) -> int:
        if self._size == 0:
            raise IndexError("pop from empty DynamicArray")
        val = self._data[self._size - 1]
        self._size -= 1
        # Optional shrink
        if 0 < self._size < self._capacity // 4:
            self._resize(max(1, self._capacity // 2))
        return val

    def __getitem__(self, idx: int) -> int:
        if not 0 <= idx < self._size:
            raise IndexError("index out of range")
        return self._data[idx]

    def __setitem__(self, idx: int, val: int) -> None:
        if not 0 <= idx < self._size:
            raise IndexError("index out of range")
        self._data[idx] = val

    def __len__(self) -> int:
        return self._size

    def __repr__(self) -> str:
        return str([self._data[i] for i in range(self._size)])


# ------------------------------------------------------------------------------
# 3.  SEARCH ALGORITHMS
# ------------------------------------------------------------------------------
def linear_search(arr: List[int], target: int) -> int:
    """
    O(n) worst-case, O(1) space.
    Returns first index of target or -1.
    """
    for idx, val in enumerate(arr):
        if val == target:
            return idx
    return -1


def binary_search(arr: List[int], target: int) -> int:
    """
    Pre-condition: arr is sorted ascending.
    O(log n) time, O(1) space.
    Returns index or -1.
    """
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def lower_bound(arr: List[int], target: int) -> int:
    """
    First index where arr[i] >= target.
    Uses bisect_left.
    """
    return bisect.bisect_left(arr, target)


def upper_bound(arr: List[int], target: int) -> int:
    """
    First index where arr[i] > target.
    Uses bisect_right.
    """
    return bisect.bisect_right(arr, target)


# ------------------------------------------------------------------------------
# 4.  SORTING ALGORITHMS (IN-PLACE)
# ------------------------------------------------------------------------------
def bubble_sort(arr: List[int]) -> None:
    """
    O(n²) worst, O(1) space.
    """
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                _swap(arr, j, j + 1)
                swapped = True
        if not swapped:
            break


def insertion_sort(arr: List[int]) -> None:
    """
    O(n²) worst, O(n) best (nearly sorted), O(1) space.
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key


def selection_sort(arr: List[int]) -> None:
    """
    O(n²) always, O(1) space.
    """
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i + 1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        _swap(arr, i, min_idx)


def merge_sort(arr: List[int]) -> List[int]:
    """
    O(n log n) always, O(n) space.
    Returns new sorted list (not in-place).
    """
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return _merge(left, right)


def _merge(left: List[int], right: List[int]) -> List[int]:
    merged = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            merged.append(left[i])
            i += 1
        else:
            merged.append(right[j])
            j += 1
    merged.extend(left[i:])
    merged.extend(right[j:])
    return merged


def quick_sort(arr: List[int], lo: int = 0, hi: Optional[int] = None) -> None:
    """
    O(n log n) average, O(n²) worst, O(log n) space (recursion).
    In-place.
    """
    if hi is None:
        hi = len(arr) - 1
    if lo < hi:
        p = _partition(arr, lo, hi)
        quick_sort(arr, lo, p - 1)
        quick_sort(arr, p + 1, hi)


def _partition(arr: List[int], lo: int, hi: int) -> int:
    pivot = arr[hi]
    i = lo - 1
    for j in range(lo, hi):
        if arr[j] <= pivot:
            i += 1
            _swap(arr, i, j)
    _swap(arr, i + 1, hi)
    return i + 1


def heap_sort(arr: List[int]) -> None:
    """
    O(n log n) always, O(1) space.
    In-place.
    """
    n = len(arr)
    # Build max-heap
    for i in range(n // 2 - 1, -1, -1):
        _heapify(arr, n, i)
    # Extract elements
    for end in range(n - 1, 0, -1):
        _swap(arr, 0, end)
        _heapify(arr, end, 0)


def _heapify(arr: List[int], n: int, root: int) -> None:
    largest = root
    left = 2 * root + 1
    right = 2 * root + 2
    if left < n and arr[left] > arr[largest]:
        largest = left
    if right < n and arr[right] > arr[largest]:
        largest = right
    if largest != root:
        _swap(arr, root, largest)
        _heapify(arr, n, largest)


# ------------------------------------------------------------------------------
# 5.  TWO-POINTER TECHNIQUES
# ------------------------------------------------------------------------------
def two_sum_sorted(arr: List[int], target: int) -> Tuple[int, int]:
    """
    Sorted array. Find two indices whose values sum to target.
    O(n) time, O(1) space.
    """
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        s = arr[lo] + arr[hi]
        if s == target:
            return lo, hi
        elif s < target:
            lo += 1
        else:
            hi -= 1
    raise ValueError("No pair found")


def three_sum_zero(arr: List[int]) -> List[Tuple[int, int, int]]:
    """
    All unique triplets summing to 0.
    O(n²) time, O(1) extra space (ignoring output).
    """
    arr.sort()
    res = []
    n = len(arr)
    for i in range(n - 2):
        if i > 0 and arr[i] == arr[i - 1]:
            continue
        lo, hi = i + 1, n - 1
        while lo < hi:
            s = arr[i] + arr[lo] + arr[hi]
            if s == 0:
                res.append((arr[i], arr[lo], arr[hi]))
                lo += 1
                hi -= 1
                while lo < hi and arr[lo] == arr[lo - 1]:
                    lo += 1
                while lo < hi and arr[hi] == arr[hi + 1]:
                    hi -= 1
            elif s < 0:
                lo += 1
            else:
                hi -= 1
    return res


# ------------------------------------------------------------------------------
# 6.  SLIDING WINDOW
# ------------------------------------------------------------------------------
def max_sum_subarray_k(arr: List[int], k: int) -> int:
    """
    Maximum sum of any contiguous subarray of length k.
    O(n) time, O(1) space.
    """
    if k > len(arr):
        raise ValueError("k larger than array")
    window = sum(arr[:k])
    max_sum = window
    for i in range(k, len(arr)):
        window += arr[i] - arr[i - k]
        max_sum = max(max_sum, window)
    return max_sum


def longest_substring_k_distinct(s: str, k: int) -> int:
    """
    Longest substring with at most k distinct characters.
    O(n) time, O(k) space.
    """
    from collections import defaultdict
    freq = defaultdict(int)
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        freq[ch] += 1
        while len(freq) > k:
            left_ch = s[left]
            freq[left_ch] -= 1
            if freq[left_ch] == 0:
                del freq[left_ch]
            left += 1
        max_len = max(max_len, right - left + 1)
    return max_len


# ------------------------------------------------------------------------------
# 7.  PREFIX SUM & DIFFERENCE ARRAY
# ------------------------------------------------------------------------------
class PrefixSum:
    """
    Immutable prefix sums for range sum queries.
    """
    def __init__(self, arr: List[int]):
        self.prefix = [0]
        for val in arr:
            self.prefix.append(self.prefix[-1] + val)

    def range_sum(self, l: int, r: int) -> int:
        """
        Sum arr[l..r] inclusive.
        O(1) per query.
        """
        return self.prefix[r + 1] - self.prefix[l]


class DifferenceArray:
    """
    Mutable range add in O(1) per update, O(n) final build.
    """
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.diff = [0] * (self.n + 1)
        for i, val in enumerate(arr):
            self.diff[i] += val
            if i + 1 < self.n:
                self.diff[i + 1] -= val

    def add_range(self, l: int, r: int, delta: int) -> None:
        self.diff[l] += delta
        if r + 1 < self.n:
            self.diff[r + 1] -= delta

    def build(self) -> List[int]:
        res = []
        curr = 0
        for i in range(self.n):
            curr += self.diff[i]
            res.append(curr)
        return res


# ------------------------------------------------------------------------------
# 8.  SPARSE TABLE – STATIC RANGE MINIMUM QUERY
# ------------------------------------------------------------------------------
class SparseTable:
    """
    Preprocessing O(n log n), query O(1).
    Immutable array.
    """
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.log = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log[i] = self.log[i // 2] + 1
        k = self.log[self.n] + 1
        self.st = [[0] * self.n for _ in range(k)]
        for i in range(self.n):
            self.st[0][i] = arr[i]
        j = 1
        while (1 << j) <= self.n:
            i = 0
            while i + (1 << j) <= self.n:
                self.st[j][i] = min(self.st[j - 1][i],
                                    self.st[j - 1][i + (1 << (j - 1))])
                i += 1
            j += 1

    def query_min(self, l: int, r: int) -> int:
        """
        Minimum on [l..r] inclusive.
        """
        length = r - l + 1
        k = self.log[length]
        return min(self.st[k][l], self.st[k][r - (1 << k) + 1])


# ------------------------------------------------------------------------------
# 9.  FENWICK TREE (BINARY INDEXED TREE)
# ------------------------------------------------------------------------------
class FenwickTree:
    """
    Point update & prefix sum in O(log n).
    1-based indexing internally.
    """
    def __init__(self, arr: List[int]):
        self.n = len(arr)
        self.tree = [0] * (self.n + 1)
        for idx, val in enumerate(arr):
            self.add(idx + 1, val)

    def add(self, idx: int, delta: int) -> None:
        while idx <= self.n:
            self.tree[idx] += delta
            idx += idx & -idx

    def prefix(self, idx: int) -> int:
        res = 0
        while idx > 0:
            res += self.tree[idx]
            idx -= idx & -idx
        return res

    def range_sum(self, l: int, r: int) -> int:
        """
        Sum arr[l..r] inclusive (0-based).
        """
        return self.prefix(r + 1) - self.prefix(l)


# ------------------------------------------------------------------------------
# 10. SEGMENT TREE – GENERIC
# ------------------------------------------------------------------------------
class SegmentTree:
    """
    Point update & range query in O(log n).
    Supports any associative operation.
    """
    def __init__(self, arr: List[int], op: Callable[[int, int], int], default: int):
        self.n = len(arr)
        self.op = op
        self.default = default
        self.tree = [default] * (2 * self.n)
        # Build
        for i in range(self.n):
            self.tree[self.n + i] = arr[i]
        for i in range(self.n - 1, 0, -1):
            self.tree[i] = op(self.tree[i * 2], self.tree[i * 2 + 1])

    def update(self, idx: int, val: int) -> None:
        idx += self.n
        self.tree[idx] = val
        while idx > 1:
            idx //= 2
            self.tree[idx] = self.op(self.tree[2 * idx], self.tree[2 * idx + 1])

    def query(self, l: int, r: int) -> int:
        """
        Query [l..r] inclusive.
        """
        l += self.n
        r += self.n
        res = self.default
        while l <= r:
            if l % 2 == 1:
                res = self.op(res, self.tree[l])
                l += 1
            if r % 2 == 0:
                res = self.op(res, self.tree[r])
                r -= 1
            l //= 2
            r //= 2
        return res


# ------------------------------------------------------------------------------
# 11. KADANE – MAXIMUM SUBARRAY
# ------------------------------------------------------------------------------
def kadane(arr: List[int]) -> int:
    """
    Maximum subarray sum.
    O(n) time, O(1) space.
    """
    best = curr = arr[0]
    for val in arr[1:]:
        curr = max(val, curr + val)
        best = max(best, curr)
    return best


# ------------------------------------------------------------------------------
# 12. NEXT PERMUTATION
# ------------------------------------------------------------------------------
def next_permutation(arr: List[int]) -> bool:
    """
    In-place next lexicographically greater permutation.
    Returns False if already last permutation.
    """
    n = len(arr)
    i = n - 2
    while i >= 0 and arr[i] >= arr[i + 1]:
        i -= 1
    if i == -1:
        arr.reverse()
        return False
    j = n - 1
    while arr[j] <= arr[i]:
        j -= 1
    _swap(arr, i, j)
    arr[i + 1:] = reversed(arr[i + 1:])
    return True


# ------------------------------------------------------------------------------
# 13. ROTATION ALGORITHMS
# ------------------------------------------------------------------------------
def rotate_left(arr: List[int], k: int) -> None:
    """
    Rotate array left by k positions in O(n) time, O(1) space.
    """
    n = len(arr)
    k %= n
    if k == 0:
        return
    # Triple reverse
    arr[:k] = reversed(arr[:k])
    arr[k:] = reversed(arr[k:])
    arr.reverse()


# ------------------------------------------------------------------------------
# 14. UNION-FIND (DISJOINT SET) ON ARRAY INDICES
# ------------------------------------------------------------------------------
class UnionFind:
    """
    Path compression + union by rank.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            self.parent[px] = py
        elif self.rank[px] > self.rank[py]:
            self.parent[py] = px
        else:
            self.parent[py] = px
            self.rank[px] += 1


# ------------------------------------------------------------------------------
# 15. MO’S ALGORITHM – OFFLINE SQUARE-ROOT DECOMPOSITION
# ------------------------------------------------------------------------------
class MoAlgorithm:
    """
    Answer range queries offline in O((n+q)√n).
    """
    def __init__(self, arr: List[int]):
        self.arr = arr
        self.n = len(arr)
        self.block_size = int(math.isqrt(self.n)) + 1

    def add(self, idx: int) -> None:
        raise NotImplementedError

    def remove(self, idx: int) -> None:
        raise NotImplementedError

    def get_answer(self) -> int:
        raise NotImplementedError

    def process_queries(self, queries: List[Tuple[int, int]]) -> List[int]:
        # queries are (l, r) inclusive
        indexed = [(l, r, i) for i, (l, r) in enumerate(queries)]
        indexed.sort(key=lambda q: (q[0] // self.block_size,
                                    q[1] if (q[0] // self.block_size) % 2 == 0 else -q[1]))
        curr_l = curr_r = 0
        self.add(0)
        answers = [0] * len(queries)
        for l, r, idx in indexed:
            while curr_l > l:
                curr_l -= 1
                self.add(curr_l)
            while curr_r < r:
                curr_r += 1
                self.add(curr_r)
            while curr_l < l:
                self.remove(curr_l)
                curr_l += 1
            while curr_r > r:
                self.remove(curr_r)
                curr_r -= 1
            answers[idx] = self.get_answer()
        return answers


# ------------------------------------------------------------------------------
# 16. CYCLIC SORT – O(n) FOR 1..n
# ------------------------------------------------------------------------------
def cyclic_sort(arr: List[int]) -> None:
    """
    In-place sort for array containing 1..n exactly once.
    O(n) time, O(1) space.
    """
    i = 0
    while i < len(arr):
        j = arr[i] - 1
        if arr[i] != arr[j]:
            _swap(arr, i, j)
        else:
            i += 1


# ------------------------------------------------------------------------------
# 17. DUTCH NATIONAL FLAG
# ------------------------------------------------------------------------------
def dutch_flag(arr: List[int]) -> None:
    """
    Sort array of 0,1,2 in O(n) time, O(1) space.
    """
    low = mid = 0
    high = len(arr) - 1
    while mid <= high:
        if arr[mid] == 0:
            _swap(arr, low, mid)
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            _swap(arr, mid, high)
            high -= 1


# ------------------------------------------------------------------------------
# 18. MONOTONIC STACK
# ------------------------------------------------------------------------------
def next_greater_element(arr: List[int]) -> List[int]:
    """
    For each element, next greater to the right.
    O(n) time, O(n) space.
    """
    res = [-1] * len(arr)
    st = []
    for i, val in enumerate(arr):
        while st and arr[st[-1]] < val:
            res[st.pop()] = val
        st.append(i)
    return res


# ------------------------------------------------------------------------------
# 19. MONOTONIC QUEUE (SLIDING WINDOW MAXIMUM)
# ------------------------------------------------------------------------------
from collections import deque

def sliding_window_max(arr: List[int], k: int) -> List[int]:
    """
    Maximum in every window of size k.
    O(n) time, O(k) space.
    """
    q = deque()
    res = []
    for i, val in enumerate(arr):
        while q and arr[q[-1]] <= val:
            q.pop()
        q.append(i)
        while q[0] <= i - k:
            q.popleft()
        if i >= k - 1:
            res.append(arr[q[0]])
    return res


# ------------------------------------------------------------------------------
# 20. COUNTING INVERSIONS (MODIFIED MERGE SORT)
# ------------------------------------------------------------------------------
def count_inversions(arr: List[int]) -> int:
    """
    Count inversions in O(n log n).
    """
    def _merge_count(arr: List[int], tmp: List[int], l: int, r: int) -> int:
        if l >= r:
            return 0
        mid = (l + r) // 2
        inv = _merge_count(arr, tmp, l, mid) + _merge_count(arr, tmp, mid + 1, r)
        i, j, k = l, mid + 1, l
        while i <= mid and j <= r:
            if arr[i] <= arr[j]:
                tmp[k] = arr[i]
                i += 1
            else:
                tmp[k] = arr[j]
                inv += mid - i + 1
                j += 1
            k += 1
        while i <= mid:
            tmp[k] = arr[i]
            i += 1
            k += 1
        while j <= r:
            tmp[k] = arr[j]
            j += 1
            k += 1
        for idx in range(l, r + 1):
            arr[idx] = tmp[idx]
        return inv

    tmp = [0] * len(arr)
    return _merge_count(arr, tmp, 0, len(arr) - 1)


# ------------------------------------------------------------------------------
# 21. MAJORITY ELEMENT (BOYER-MOORE VOTING)
# ------------------------------------------------------------------------------
def majority_element(arr: List[int]) -> Optional[int]:
    """
    Element appearing > n/2 times.
    O(n) time, O(1) space.
    """
    candidate = None
    count = 0
    for val in arr:
        if count == 0:
            candidate = val
            count = 1
        elif val == candidate:
            count += 1
        else:
            count -= 1
    # Verify
    if arr.count(candidate) > len(arr) // 2:
        return candidate
    return None


# ------------------------------------------------------------------------------
# 22. FIND MEDIAN OF TWO SORTED ARRAYS
# ------------------------------------------------------------------------------
def median_two_sorted(a: List[int], b: List[int]) -> float:
    """
    O(log(min(m,n))) binary search.
    """
    if len(a) > len(b):
        a, b = b, a
    m, n = len(a), len(b)
    lo, hi = 0, m
    while lo <= hi:
        i = (lo + hi) // 2
        j = (m + n + 1) // 2 - i
        max_left_a = -math.inf if i == 0 else a[i - 1]
        min_right_a = math.inf if i == m else a[i]
        max_left_b = -math.inf if j == 0 else b[j - 1]
        min_right_b = math.inf if j == n else b[j]
        if max_left_a <= min_right_b and max_left_b <= min_right_a:
            if (m + n) % 2 == 0:
                return (max(max_left_a, max_left_b) +
                        min(min_right_a, min_right_b)) / 2
            else:
                return max(max_left_a, max_left_b)
        elif max_left_a > min_right_b:
            hi = i - 1
        else:
            lo = i + 1
    raise ValueError("Input arrays not sorted")


# ------------------------------------------------------------------------------
# 23. KTH LARGEST ELEMENT (QUICKSELECT)
# ------------------------------------------------------------------------------
def quickselect(arr: List[int], k: int) -> int:
    """
    k is 1-based largest.
    O(n) average, O(n²) worst.
    """
    if not 1 <= k <= len(arr):
        raise IndexError("k out of range")
    k = len(arr) - k  # convert to 0-based smallest
    lo, hi = 0, len(arr) - 1
    while True:
        p = _partition(arr, lo, hi)
        if p == k:
            return arr[p]
        elif p < k:
            lo = p + 1
        else:
            hi = p - 1


# ------------------------------------------------------------------------------
# 24. LONGEST INCREASING SUBSEQUENCE (O(n log n))
# ------------------------------------------------------------------------------
def lis(arr: List[int]) -> int:
    """
    Length of longest strictly increasing subsequence.
    O(n log n) time, O(n) space.
    """
    tails = []
    for val in arr:
        idx = bisect.bisect_left(tails, val)
        if idx == len(tails):
            tails.append(val)
        else:
            tails[idx] = val
    return len(tails)


# ------------------------------------------------------------------------------
# 25. BIT MANIPULATION TRICKS ON ARRAYS
# ------------------------------------------------------------------------------
def single_number_xor(arr: List[int]) -> int:
    """
    Element appearing exactly once, others twice.
    O(n) time, O(1) space.
    """
    res = 0
    for val in arr:
        res ^= val
    return res


def find_missing_number(arr: List[int]) -> int:
    """
    Given 0..n with one missing, find it.
    O(n) time, O(1) space.
    """
    n = len(arr)
    expected = n * (n + 1) // 2
    return expected - sum(arr)


# ------------------------------------------------------------------------------
# 26. RUN-LENGTH ENCODING
# ------------------------------------------------------------------------------
def run_length_encode(arr: List[str]) -> List[Tuple[str, int]]:
    """
    Compress consecutive duplicates.
    """
    if not arr:
        return []
    res = []
    curr = arr[0]
    cnt = 1
    for ch in arr[1:]:
        if ch == curr:
            cnt += 1
        else:
            res.append((curr, cnt))
            curr = ch
            cnt = 1
    res.append((curr, cnt))
    return res


# ------------------------------------------------------------------------------
# 27. SUFFIX ARRAY (O(n log n) WITH KASAI)
# ------------------------------------------------------------------------------
def build_suffix_array(s: str) -> List[int]:
    """
    Returns suffix array for string s.
    O(n log n) time, O(n) space.
    """
    n = len(s)
    k = 1
    c = [ord(ch) for ch in s]
    sa = list(range(n))
    ra = c[:]
    while k < n:
        # Pair array
        sa = sorted(sa, key=lambda x: (ra[x], ra[x + k]) if x + k < n else (ra[x],))
        new_ra = [0] * n
        new_ra[sa[0]] = 0
        for i in range(1, n):
            curr = (ra[sa[i]], ra[sa[i] + k]) if sa[i] + k < n else (ra[sa[i]],)
            prev = (ra[sa[i - 1]], ra[sa[i - 1] + k]) if sa[i - 1] + k < n else (ra[sa[i - 1]],)
            new_ra[sa[i]] = new_ra[sa[i - 1]] + (1 if curr > prev else 0)
        ra = new_ra
        if ra[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


# ------------------------------------------------------------------------------
# 28. SELF-TEST / DEMO
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # Basic
    da = DynamicArray()
    for v in [3, 1, 4, 1, 5]:
        da.append(v)
    print("DynamicArray:", da)

    # Sorting
    arr = [5, 2, 9, 1, 5, 6]
    quick_sort(arr)
    print("QuickSort:", arr)

    # Fenwick
    ft = FenwickTree([1, 2, 3, 4, 5])
    print("Fenwick range_sum(1,3):", ft.range_sum(1, 3))

    # Mo’s
    class MoSum(MoAlgorithm):
        def __init__(self, arr):
            super().__init__(arr)
            self.curr_sum = 0

        def add(self, idx):
            self.curr_sum += self.arr[idx]

        def remove(self, idx):
            self.curr_sum -= self.arr[idx]

        def get_answer(self):
            return self.curr_sum

    mo = MoSum([1, 2, 3, 4, 5])
    queries = [(0, 2), (1, 3), (0, 4)]
    print("Mo’s answers:", mo.process_queries(queries))

    # LIS
    print("LIS of [10,9,2,5,3,7,101,18]:", lis([10, 9, 2, 5, 3, 7, 101, 18]))

    # Suffix array
    print("Suffix array of 'banana':", build_suffix_array("banana"))