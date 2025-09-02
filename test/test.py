from typing import Dict, List, Union, Optional, Tuple, Set, Deque
from collections import defaultdict, deque
from heapq import heappush, heappop
import bisect

class AdvancedDSALibrary:
    
    def __init__(self):
        self.trie_root = {}
        self.parent = {}
        self.rank = {}
        self.cache = {}
        self.cache_capacity = 0
        self.cache_order = deque()
    
    def trie_insert(self, word: str) -> None:
        node = self.trie_root
        for char in word:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['#'] = True
    
    def trie_search(self, word: str) -> bool:
        node = self.trie_root
        for char in word:
            if char not in node:
                return False
            node = node[char]
        return '#' in node
    
    def trie_starts_with(self, prefix: str) -> bool:
        node = self.trie_root
        for char in prefix:
            if char not in node:
                return False
            node = node[char]
        return True
    
    def union_find_make_set(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
    
    def union_find_find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.union_find_find(self.parent[x])
        return self.parent[x]
    
    def union_find_union(self, x: int, y: int) -> bool:
        px, py = self.union_find_find(x), self.union_find_find(y)
        if px == py:
            return False
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        return True
    
    def lru_cache_init(self, capacity: int) -> None:
        self.cache.clear()
        self.cache_order.clear()
        self.cache_capacity = capacity
    
    def lru_cache_get(self, key: int) -> int:
        if key in self.cache:
            self.cache_order.remove(key)
            self.cache_order.append(key)
            return self.cache[key]
        return -1
    
    def lru_cache_put(self, key: int, value: int) -> None:
        if key in self.cache:
            self.cache_order.remove(key)
        elif len(self.cache) >= self.cache_capacity:
            oldest = self.cache_order.popleft()
            del self.cache[oldest]
        self.cache[key] = value
        self.cache_order.append(key)
    
    def fenwick_tree_update(self, tree: List[int], i: int, delta: int) -> None:
        while i < len(tree):
            tree[i] += delta
            i += i & (-i)
    
    def fenwick_tree_query(self, tree: List[int], i: int) -> int:
        result = 0
        while i > 0:
            result += tree[i]
            i -= i & (-i)
        return result
    
    def segment_tree_build(self, arr: List[int]) -> List[int]:
        n = len(arr)
        tree = [0] * (4 * n)
        self._segment_tree_build_helper(tree, arr, 1, 0, n - 1)
        return tree
    
    def _segment_tree_build_helper(self, tree: List[int], arr: List[int], v: int, tl: int, tr: int) -> None:
        if tl == tr:
            tree[v] = arr[tl]
        else:
            tm = (tl + tr) // 2
            self._segment_tree_build_helper(tree, arr, 2 * v, tl, tm)
            self._segment_tree_build_helper(tree, arr, 2 * v + 1, tm + 1, tr)
            tree[v] = tree[2 * v] + tree[2 * v + 1]
    
    def segment_tree_query(self, tree: List[int], v: int, tl: int, tr: int, l: int, r: int) -> int:
        if l > r:
            return 0
        if l == tl and r == tr:
            return tree[v]
        tm = (tl + tr) // 2
        return (self.segment_tree_query(tree, 2 * v, tl, tm, l, min(r, tm)) +
                self.segment_tree_query(tree, 2 * v + 1, tm + 1, tr, max(l, tm + 1), r))
    
    def dijkstra_shortest_path(self, graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        pq = [(0, start)]
        visited = set()
        
        while pq:
            current_dist, u = heappop(pq)
            if u in visited:
                continue
            visited.add(u)
            
            for v, weight in graph[u]:
                distance = current_dist + weight
                if distance < distances[v]:
                    distances[v] = distance
                    heappush(pq, (distance, v))
        
        return dict(distances)
    
    def topological_sort_kahn(self, graph: Dict[int, List[int]]) -> List[int]:
        in_degree = defaultdict(int)
        for u in graph:
            for v in graph[u]:
                in_degree[v] += 1
        
        queue = deque([u for u in graph if in_degree[u] == 0])
        result = []
        
        while queue:
            u = queue.popleft()
            result.append(u)
            for v in graph[u]:
                in_degree[v] -= 1
                if in_degree[v] == 0:
                    queue.append(v)
        
        return result
    
    def tarjan_scc(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        result = []
        
        def strongconnect(v):
            index[v] = index_counter[0]
            lowlinks[v] = index_counter[0]
            index_counter[0] += 1
            stack.append(v)
            on_stack[v] = True
            
            for w in graph[v]:
                if w not in index:
                    strongconnect(w)
                    lowlinks[v] = min(lowlinks[v], lowlinks[w])
                elif on_stack[w]:
                    lowlinks[v] = min(lowlinks[v], index[w])
            
            if lowlinks[v] == index[v]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == v:
                        break
                result.append(component)
        
        for v in graph:
            if v not in index:
                strongconnect(v)
        
        return result
    
    def kmp_search(self, text: str, pattern: str) -> List[int]:
        def compute_lps(pattern):
            lps = [0] * len(pattern)
            length = 0
            i = 1
            while i < len(pattern):
                if pattern[i] == pattern[length]:
                    length += 1
                    lps[i] = length
                    i += 1
                else:
                    if length != 0:
                        length = lps[length - 1]
                    else:
                        lps[i] = 0
                        i += 1
            return lps
        
        if not pattern:
            return []
        
        lps = compute_lps(pattern)
        result = []
        i = j = 0
        
        while i < len(text):
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == len(pattern):
                result.append(i - j)
                j = lps[j - 1]
            elif i < len(text) and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return result
    
    def rolling_hash(self, s: str, base: int = 31, mod: int = 10**9 + 7) -> int:
        hash_value = 0
        power = 1
        for char in s:
            hash_value = (hash_value + (ord(char) - ord('a') + 1) * power) % mod
            power = (power * base) % mod
        return hash_value
    
    def suffix_array(self, s: str) -> List[int]:
        n = len(s)
        suffixes = [(s[i:], i) for i in range(n)]
        suffixes.sort()
        return [suffix[1] for suffix in suffixes]
    
    def binary_search_leftmost(self, arr: List[int], target: int) -> int:
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left
    
    def binary_search_rightmost(self, arr: List[int], target: int) -> int:
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left - 1
    
    def quick_select(self, arr: List[int], k: int) -> int:
        def partition(left, right, pivot_index):
            pivot_value = arr[pivot_index]
            arr[pivot_index], arr[right] = arr[right], arr[pivot_index]
            store_index = left
            for i in range(left, right):
                if arr[i] < pivot_value:
                    arr[store_index], arr[i] = arr[i], arr[store_index]
                    store_index += 1
            arr[right], arr[store_index] = arr[store_index], arr[right]
            return store_index
        
        def select(left, right, k_smallest):
            if left == right:
                return arr[left]
            pivot_index = left + (right - left) // 2
            pivot_index = partition(left, right, pivot_index)
            if k_smallest == pivot_index:
                return arr[k_smallest]
            elif k_smallest < pivot_index:
                return select(left, pivot_index - 1, k_smallest)
            else:
                return select(pivot_index + 1, right, k_smallest)
        
        return select(0, len(arr) - 1, k)
    
    def merge_intervals(self, intervals: List[List[int]]) -> List[List[int]]:
        if not intervals:
            return []
        
        intervals.sort(key=lambda x: x[0])
        merged = [intervals[0]]
        
        for current in intervals[1:]:
            if current[0] <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], current[1])
            else:
                merged.append(current)
        
        return merged
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            while dq and dq[0] < i - k + 1:
                dq.popleft()
            
            while dq and nums[dq[-1]] <= nums[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def monotonic_stack_next_greater(self, nums: List[int]) -> List[int]:
        stack = []
        result = [-1] * len(nums)
        
        for i in range(len(nums)):
            while stack and nums[stack[-1]] < nums[i]:
                result[stack.pop()] = nums[i]
            stack.append(i)
        
        return result
    
    def two_pointers_two_sum(self, nums: List[int], target: int) -> List[int]:
        left, right = 0, len(nums) - 1
        
        while left < right:
            current_sum = nums[left] + nums[right]
            if current_sum == target:
                return [left, right]
            elif current_sum < target:
                left += 1
            else:
                right -= 1
        
        return []
    
    def backtrack_permutations(self, nums: List[int]) -> List[List[int]]:
        result = []
        
        def backtrack(path, remaining):
            if not remaining:
                result.append(path[:])
                return
            
            for i in range(len(remaining)):
                path.append(remaining[i])
                backtrack(path, remaining[:i] + remaining[i+1:])
                path.pop()
        
        backtrack([], nums)
        return result
    
    def memoization_fibonacci(self, n: int, memo: Dict[int, int] = None) -> int:
        if memo is None:
            memo = {}
        
        if n in memo:
            return memo[n]
        
        if n <= 1:
            return n
        
        memo[n] = self.memoization_fibonacci(n - 1, memo) + self.memoization_fibonacci(n - 2, memo)
        return memo[n]
    
    def detect_cycle_directed(self, graph: Dict[int, List[int]]) -> bool:
        WHITE, GRAY, BLACK = 0, 1, 2
        color = defaultdict(int)
        
        def dfs(node):
            if color[node] == GRAY:
                return True
            if color[node] == BLACK:
                return False
            
            color[node] = GRAY
            for neighbor in graph[node]:
                if dfs(neighbor):
                    return True
            color[node] = BLACK
            return False
        
        for node in graph:
            if color[node] == WHITE:
                if dfs(node):
                    return True
        
        return False


from transformers import ZeroShotObjectDetectionPipeline

