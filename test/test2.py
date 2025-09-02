from typing import Dict, List, Union, Optional, Tuple, Set, Deque
from collections import defaultdict, deque
from heapq import heappush, heappop, heapify
import bisect

class AdvancedDSAAlgorithms:
    
    def binary_search_range(self, nums: List[int], target: int) -> Tuple[int, int]:
        def find_boundary(is_left: bool) -> int:
            left, right = 0, len(nums) - 1
            boundary = -1
            while left <= right:
                mid = (left + right) // 2
                if nums[mid] == target:
                    boundary = mid
                    if is_left:
                        right = mid - 1
                    else:
                        left = mid + 1
                elif nums[mid] < target:
                    left = mid + 1
                else:
                    right = mid - 1
            return boundary
        return (find_boundary(True), find_boundary(False))
    
    def quick_sort_optimized(self, arr: List[int]) -> List[int]:
        def partition(low: int, high: int) -> int:
            pivot = arr[high]
            i = low - 1
            for j in range(low, high):
                if arr[j] <= pivot:
                    i += 1
                    arr[i], arr[j] = arr[j], arr[i]
            arr[i + 1], arr[high] = arr[high], arr[i + 1]
            return i + 1
        
        def quick_sort_helper(low: int, high: int) -> None:
            if low < high:
                pi = partition(low, high)
                quick_sort_helper(low, pi - 1)
                quick_sort_helper(pi + 1, high)
        
        quick_sort_helper(0, len(arr) - 1)
        return arr
    
    def merge_sort_inplace(self, arr: List[int]) -> List[int]:
        def merge(left: int, mid: int, right: int) -> None:
            temp = arr[left:right + 1]
            i, j, k = 0, mid - left + 1, left
            
            while i <= mid - left and j <= right - left:
                if temp[i] <= temp[j]:
                    arr[k] = temp[i]
                    i += 1
                else:
                    arr[k] = temp[j]
                    j += 1
                k += 1
            
            while i <= mid - left:
                arr[k] = temp[i]
                i += 1
                k += 1
            
            while j <= right - left:
                arr[k] = temp[j]
                j += 1
                k += 1
        
        def merge_sort_helper(left: int, right: int) -> None:
            if left < right:
                mid = (left + right) // 2
                merge_sort_helper(left, mid)
                merge_sort_helper(mid + 1, right)
                merge(left, mid, right)
        
        merge_sort_helper(0, len(arr) - 1)
        return arr
    
    def max_heap_operations(self, operations: List[Tuple[str, int]]) -> List[int]:
        heap = []
        results = []
        
        for op, val in operations:
            if op == "insert":
                heappush(heap, -val)
            elif op == "extract_max":
                if heap:
                    results.append(-heappop(heap))
                else:
                    results.append(-1)
            elif op == "peek_max":
                if heap:
                    results.append(-heap[0])
                else:
                    results.append(-1)
        
        return results
    
    def trie_advanced_operations(self, words: List[str], queries: List[str]) -> Dict[str, Union[bool, List[str]]]:
        class TrieNode:
            def __init__(self):
                self.children = {}
                self.is_end = False
                self.count = 0
        
        root = TrieNode()
        
        for word in words:
            node = root
            for char in word:
                if char not in node.children:
                    node.children[char] = TrieNode()
                node = node.children[char]
                node.count += 1
            node.is_end = True
        
        def search(word: str) -> bool:
            node = root
            for char in word:
                if char not in node.children:
                    return False
                node = node.children[char]
            return node.is_end
        
        def starts_with_prefix(prefix: str) -> List[str]:
            node = root
            for char in prefix:
                if char not in node.children:
                    return []
                node = node.children[char]
            
            result = []
            def dfs(current_node: TrieNode, path: str) -> None:
                if current_node.is_end:
                    result.append(prefix + path)
                for char, child in current_node.children.items():
                    dfs(child, path + char)
            
            dfs(node, "")
            return result
        
        results = {}
        for query in queries:
            if query.startswith("search:"):
                word = query[7:]
                results[query] = search(word)
            elif query.startswith("prefix:"):
                prefix = query[7:]
                results[query] = starts_with_prefix(prefix)
        
        return results
    
    def dijkstra_shortest_path(self, graph: Dict[int, List[Tuple[int, int]]], start: int) -> Dict[int, int]:
        distances = defaultdict(lambda: float('inf'))
        distances[start] = 0
        heap = [(0, start)]
        visited = set()
        
        while heap:
            current_dist, current_node = heappop(heap)
            
            if current_node in visited:
                continue
            
            visited.add(current_node)
            
            for neighbor, weight in graph[current_node]:
                distance = current_dist + weight
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    heappush(heap, (distance, neighbor))
        
        return dict(distances)
    
    def union_find_operations(self, n: int, operations: List[Tuple[str, int, int]]) -> List[bool]:
        parent = list(range(n))
        rank = [0] * n
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> bool:
            px, py = find(x), find(y)
            if px == py:
                return False
            
            if rank[px] < rank[py]:
                parent[px] = py
            elif rank[px] > rank[py]:
                parent[py] = px
            else:
                parent[py] = px
                rank[px] += 1
            return True
        
        results = []
        for op, x, y in operations:
            if op == "union":
                results.append(union(x, y))
            elif op == "connected":
                results.append(find(x) == find(y))
        
        return results
    
    def segment_tree_range_queries(self, arr: List[int], queries: List[Tuple[str, int, int]]) -> List[int]:
        n = len(arr)
        tree = [0] * (4 * n)
        
        def build(node: int, start: int, end: int) -> None:
            if start == end:
                tree[node] = arr[start]
            else:
                mid = (start + end) // 2
                build(2 * node, start, mid)
                build(2 * node + 1, mid + 1, end)
                tree[node] = tree[2 * node] + tree[2 * node + 1]
        
        def update(node: int, start: int, end: int, idx: int, val: int) -> None:
            if start == end:
                tree[node] = val
            else:
                mid = (start + end) // 2
                if idx <= mid:
                    update(2 * node, start, mid, idx, val)
                else:
                    update(2 * node + 1, mid + 1, end, idx, val)
                tree[node] = tree[2 * node] + tree[2 * node + 1]
        
        def query(node: int, start: int, end: int, l: int, r: int) -> int:
            if r < start or end < l:
                return 0
            if l <= start and end <= r:
                return tree[node]
            mid = (start + end) // 2
            return query(2 * node, start, mid, l, r) + query(2 * node + 1, mid + 1, end, l, r)
        
        build(1, 0, n - 1)
        results = []
        
        for op, x, y in queries:
            if op == "update":
                update(1, 0, n - 1, x, y)
            elif op == "sum":
                results.append(query(1, 0, n - 1, x, y))
        
        return results
    
    def lru_cache_implementation(self, capacity: int, operations: List[Tuple[str, int, int]]) -> List[int]:
        class Node:
            def __init__(self, key: int = 0, val: int = 0):
                self.key = key
                self.val = val
                self.prev = None
                self.next = None
        
        cache = {}
        head = Node()
        tail = Node()
        head.next = tail
        tail.prev = head
        
        def add_node(node: Node) -> None:
            node.prev = head
            node.next = head.next
            head.next.prev = node
            head.next = node
        
        def remove_node(node: Node) -> None:
            node.prev.next = node.next
            node.next.prev = node.prev
        
        def move_to_head(node: Node) -> None:
            remove_node(node)
            add_node(node)
        
        def pop_tail() -> Node:
            last_node = tail.prev
            remove_node(last_node)
            return last_node
        
        results = []
        
        for op, key, val in operations:
            if op == "get":
                if key in cache:
                    node = cache[key]
                    move_to_head(node)
                    results.append(node.val)
                else:
                    results.append(-1)
            elif op == "put":
                if key in cache:
                    node = cache[key]
                    node.val = val
                    move_to_head(node)
                else:
                    new_node = Node(key, val)
                    if len(cache) >= capacity:
                        tail_node = pop_tail()
                        del cache[tail_node.key]
                    cache[key] = new_node
                    add_node(new_node)
        
        return results
    
    def sliding_window_maximum(self, nums: List[int], k: int) -> List[int]:
        dq = deque()
        result = []
        
        for i in range(len(nums)):
            while dq and dq[0] < i - k + 1:
                dq.popleft()
            
            while dq and nums[dq[-1]] < nums[i]:
                dq.pop()
            
            dq.append(i)
            
            if i >= k - 1:
                result.append(nums[dq[0]])
        
        return result
    
    def kmp_string_matching(self, text: str, pattern: str) -> List[int]:
        def compute_lps(pattern: str) -> List[int]:
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
        matches = []
        i = j = 0
        
        while i < len(text):
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == len(pattern):
                matches.append(i - j)
                j = lps[j - 1]
            elif i < len(text) and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    def topological_sort_kahn(self, graph: Dict[int, List[int]]) -> List[int]:
        in_degree = defaultdict(int)
        all_nodes = set()
        
        for node in graph:
            all_nodes.add(node)
            for neighbor in graph[node]:
                all_nodes.add(neighbor)
                in_degree[neighbor] += 1
        
        queue = deque([node for node in all_nodes if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in graph.get(node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        return result if len(result) == len(all_nodes) else []
    
    def tarjan_strongly_connected_components(self, graph: Dict[int, List[int]]) -> List[List[int]]:
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        components = []
        
        def strongconnect(node: int) -> None:
            index[node] = index_counter[0]
            lowlinks[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack[node] = True
            
            for neighbor in graph.get(node, []):
                if neighbor not in index:
                    strongconnect(neighbor)
                    lowlinks[node] = min(lowlinks[node], lowlinks[neighbor])
                elif on_stack[neighbor]:
                    lowlinks[node] = min(lowlinks[node], index[neighbor])
            
            if lowlinks[node] == index[node]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                components.append(component)
        
        for node in graph:
            if node not in index:
                strongconnect(node)
        
        return components
    
    def kruskal_minimum_spanning_tree(self, n: int, edges: List[Tuple[int, int, int]]) -> List[Tuple[int, int, int]]:
        parent = list(range(n))
        
        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]
        
        def union(x: int, y: int) -> bool:
            px, py = find(x), find(y)
            if px == py:
                return False
            parent[px] = py
            return True
        
        edges.sort(key=lambda x: x[2])
        mst = []
        
        for u, v, weight in edges:
            if union(u, v):
                mst.append((u, v, weight))
                if len(mst) == n - 1:
                    break
        
        return mst
    
    def floyd_warshall_all_pairs(self, graph: Dict[int, Dict[int, int]]) -> Dict[Tuple[int, int], int]:
        nodes = set()
        for u in graph:
            nodes.add(u)
            for v in graph[u]:
                nodes.add(v)
        
        dist = {}
        for i in nodes:
            for j in nodes:
                if i == j:
                    dist[(i, j)] = 0
                elif j in graph.get(i, {}):
                    dist[(i, j)] = graph[i][j]
                else:
                    dist[(i, j)] = float('inf')
        
        for k in nodes:
            for i in nodes:
                for j in nodes:
                    dist[(i, j)] = min(dist[(i, j)], dist[(i, k)] + dist[(k, j)])
        
        return {k: v for k, v in dist.items() if v != float('inf')}
    
    def knapsack_01_optimized(self, weights: List[int], values: List[int], capacity: int) -> Tuple[int, List[int]]:
        n = len(weights)
        dp = [[0] * (capacity + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for w in range(capacity + 1):
                if weights[i - 1] <= w:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weights[i - 1]] + values[i - 1])
                else:
                    dp[i][w] = dp[i - 1][w]
        
        selected = []
        i, w = n, capacity
        while i > 0 and w > 0:
            if dp[i][w] != dp[i - 1][w]:
                selected.append(i - 1)
                w -= weights[i - 1]
            i -= 1
        
        return dp[n][capacity], selected[::-1]
    
    def longest_common_subsequence_with_path(self, text1: str, text2: str) -> Tuple[int, str]:
        m, n = len(text1), len(text2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if text1[i - 1] == text2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        lcs = []
        i, j = m, n
        while i > 0 and j > 0:
            if text1[i - 1] == text2[j - 1]:
                lcs.append(text1[i - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] > dp[i][j - 1]:
                i -= 1
            else:
                j -= 1
        
        return dp[m][n], ''.join(reversed(lcs))
    
    def fast_matrix_exponentiation(self, matrix: List[List[int]], power: int, mod: int = 10**9 + 7) -> List[List[int]]:
        n = len(matrix)
        
        def multiply_matrices(a: List[List[int]], b: List[List[int]]) -> List[List[int]]:
            result = [[0] * n for _ in range(n)]
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % mod
            return result
        
        def matrix_power(mat: List[List[int]], p: int) -> List[List[int]]:
            if p == 0:
                return [[1 if i == j else 0 for j in range(n)] for i in range(n)]
            
            if p == 1:
                return [row[:] for row in mat]
            
            if p % 2 == 0:
                half_power = matrix_power(mat, p // 2)
                return multiply_matrices(half_power, half_power)
            else:
                return multiply_matrices(mat, matrix_power(mat, p - 1))
        
        return matrix_power(matrix, power)
    
    def advanced_two_pointers_triplets(self, nums: List[int], target: int) -> List[List[int]]:
        nums.sort()
        result = []
        n = len(nums)
        
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            
            left, right = i + 1, n - 1
            
            while left < right:
                current_sum = nums[i] + nums[left] + nums[right]
                
                if current_sum == target:
                    result.append([nums[i], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
        
        return result