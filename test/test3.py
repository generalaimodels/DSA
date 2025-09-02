from typing import TypeVar, Dict, Union, List, Set, Optional, Tuple
from collections import deque, defaultdict
import heapq
import sys

T = TypeVar('T')
Graph = Dict[T, Union[List[T], T]]

def get_neighbors(graph: Graph[T], node: T) -> List[T]:
    neighbors = graph.get(node, [])
    if not isinstance(neighbors, list):
        return [neighbors] if neighbors else []
    return neighbors

def get_all_nodes(graph: Graph[T]) -> Set[T]:
    nodes = set(graph.keys())
    for neighbors in graph.values():
        if isinstance(neighbors, list):
            nodes.update(neighbors)
        elif neighbors:
            nodes.add(neighbors)
    return nodes

def breadth_first_search(graph: Graph[T], start: T) -> List[T]:
    visited = set()
    queue = deque([start])
    traversal_order = []
    while queue:
        node = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        traversal_order.append(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                queue.append(neighbor)
    return traversal_order

def depth_first_search_iterative(graph: Graph[T], start: T) -> List[T]:
    visited = set()
    stack = [start]
    traversal_order = []
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        traversal_order.append(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                stack.append(neighbor)
    return traversal_order

def depth_first_search_recursive(graph: Graph[T], start: T) -> List[T]:
    visited = set()
    traversal_order = []
    def dfs(node: T):
        if node in visited:
            return
        visited.add(node)
        traversal_order.append(node)
        for neighbor in get_neighbors(graph, node):
            dfs(neighbor)
    dfs(start)
    return traversal_order

def has_path_bfs(graph: Graph[T], start: T, target: T) -> bool:
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node == target:
            return True
        if node in visited:
            continue
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                queue.append(neighbor)
    return False

def has_path_dfs(graph: Graph[T], start: T, target: T) -> bool:
    visited = set()
    def dfs(node: T) -> bool:
        if node == target:
            return True
        if node in visited:
            return False
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if dfs(neighbor):
                return True
        return False
    return dfs(start)

def shortest_path_length_unweighted(graph: Graph[T], start: T, target: T) -> int:
    if start == target:
        return 0
    visited = set()
    queue = deque([(start, 0)])
    while queue:
        node, dist = queue.popleft()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor == target:
                return dist + 1
            if neighbor not in visited:
                queue.append((neighbor, dist + 1))
    return -1

def detect_cycle_undirected(graph: Graph[T]) -> bool:
    all_nodes = get_all_nodes(graph)
    visited = set()
    def dfs(node: T, parent: Optional[T]) -> bool:
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor == parent:
                continue
            if neighbor in visited:
                return True
            if dfs(neighbor, node):
                return True
        return False
    for node in all_nodes:
        if node not in visited:
            if dfs(node, None):
                return True
    return False

def detect_cycle_directed(graph: Graph[T]) -> bool:
    all_nodes = get_all_nodes(graph)
    visited = set()
    rec_stack = set()
    def dfs(node: T) -> bool:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node)
        return False
    for node in all_nodes:
        if node not in visited:
            if dfs(node):
                return True
    return False

def topological_sort_kahn(graph: Graph[T]) -> List[T]:
    all_nodes = get_all_nodes(graph)
    indegree = {node: 0 for node in all_nodes}
    for node in graph:
        for neighbor in get_neighbors(graph, node):
            indegree[neighbor] += 1
    queue = deque([node for node in indegree if indegree[node] == 0])
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in get_neighbors(graph, node):
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)
    if len(order) == len(indegree):
        return order
    return []

def topological_sort_dfs(graph: Graph[T]) -> List[T]:
    all_nodes = get_all_nodes(graph)
    visited = set()
    order = []
    def dfs(node: T):
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                dfs(neighbor)
        order.append(node)
    for node in all_nodes:
        if node not in visited:
            dfs(node)
    order.reverse()
    return order

def connected_components(graph: Graph[T]) -> List[List[T]]:
    all_nodes = get_all_nodes(graph)
    visited = set()
    components = []
    for node in all_nodes:
        if node not in visited:
            component = breadth_first_search(graph, node)
            components.append(component)
            visited.update(component)
    return components

def strongly_connected_components_kosaraju(graph: Graph[T]) -> List[List[T]]:
    all_nodes = get_all_nodes(graph)
    visited = set()
    stack = []
    def dfs1(node: T):
        visited.add(node)
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                dfs1(neighbor)
        stack.append(node)
    for node in all_nodes:
        if node not in visited:
            dfs1(node)
    transpose = defaultdict(list)
    for node in all_nodes:
        for neighbor in get_neighbors(graph, node):
            transpose[neighbor].append(node)
    visited = set()
    components = []
    def dfs2(node: T, component: List[T]):
        visited.add(node)
        component.append(node)
        for neighbor in transpose[node]:
            if neighbor not in visited:
                dfs2(neighbor, component)
    while stack:
        node = stack.pop()
        if node not in visited:
            component = []
            dfs2(node, component)
            components.append(component)
    return components

def find_bridges(graph: Graph[T]) -> List[Tuple[T, T]]:
    all_nodes = get_all_nodes(graph)
    visited = set()
    disc = {}
    low = {}
    parent = {}
    time = 0
    bridges = []
    def dfs(node: T):
        nonlocal time
        visited.add(node)
        disc[node] = time
        low[node] = time
        time += 1
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                parent[neighbor] = node
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
                if low[neighbor] > disc[node]:
                    bridges.append((node, neighbor))
            elif neighbor != parent.get(node, None):
                low[node] = min(low[node], disc.get(neighbor, time))
    for node in all_nodes:
        if node not in visited:
            parent[node] = None
            dfs(node)
    return bridges

def find_articulation_points(graph: Graph[T]) -> Set[T]:
    all_nodes = get_all_nodes(graph)
    visited = set()
    disc = {}
    low = {}
    parent = {}
    ap = set()
    time = 0
    def dfs(node: T):
        nonlocal time
        children = 0
        visited.add(node)
        disc[node] = time
        low[node] = time
        time += 1
        for neighbor in get_neighbors(graph, node):
            if neighbor not in visited:
                parent[neighbor] = node
                children += 1
                dfs(neighbor)
                low[node] = min(low[node], low[neighbor])
                if parent.get(node, None) is None and children > 1:
                    ap.add(node)
                if parent.get(node, None) is not None and low[neighbor] >= disc[node]:
                    ap.add(node)
            elif neighbor != parent.get(node, None):
                low[node] = min(low[node], disc.get(neighbor, time))
    for node in all_nodes:
        if node not in visited:
            parent[node] = None
            dfs(node)
    return ap

def is_bipartite(graph: Graph[T]) -> bool:
    all_nodes = get_all_nodes(graph)
    colors = {}
    def bfs(start: T) -> bool:
        queue = deque([start])
        colors[start] = 0
        while queue:
            node = queue.popleft()
            for neighbor in get_neighbors(graph, node):
                if neighbor not in colors:
                    colors[neighbor] = 1 - colors[node]
                    queue.append(neighbor)
                elif colors[neighbor] == colors[node]:
                    return False
        return True
    for node in all_nodes:
        if node not in colors:
            if not bfs(node):
                return False
    return True

def eulerian_path(graph: Graph[T]) -> Optional[List[T]]:
    all_nodes = get_all_nodes(graph)
    temp_graph = defaultdict(list)
    for node in all_nodes:
        temp_graph[node] = get_neighbors(graph, node)[:]
    in_degree = {node: 0 for node in all_nodes}
    out_degree = {node: len(temp_graph[node]) for node in all_nodes}
    for node in temp_graph:
        for neighbor in temp_graph[node]:
            in_degree[neighbor] += 1
    start_nodes = [node for node in all_nodes if out_degree[node] - in_degree.get(node, 0) == 1]
    if not start_nodes:
        start = next(iter(all_nodes)) if all_nodes else None
    else:
        start = start_nodes[0]
    if not start:
        return None
    stack = [start]
    path = []
    while stack:
        node = stack[-1]
        if temp_graph[node]:
            stack.append(temp_graph[node].pop())
        else:
            path.append(stack.pop())
    if any(temp_graph[node] for node in temp_graph):
        return None
    path.reverse()
    return path

def eulerian_circuit(graph: Graph[T]) -> Optional[List[T]]:
    all_nodes = get_all_nodes(graph)
    temp_graph = defaultdict(list)
    for node in all_nodes:
        temp_graph[node] = get_neighbors(graph, node)[:]
    in_degree = {node: 0 for node in all_nodes}
    out_degree = {node: len(temp_graph[node]) for node in all_nodes}
    for node in temp_graph:
        for neighbor in temp_graph[node]:
            in_degree[neighbor] += 1
    if any(in_degree.get(node, 0) != out_degree[node] for node in all_nodes):
        return None
    start = next((node for node in all_nodes if out_degree[node] > 0), next(iter(all_nodes), None))
    if not start:
        return None
    stack = [start]
    path = []
    while stack:
        node = stack[-1]
        if temp_graph[node]:
            stack.append(temp_graph[node].pop())
        else:
            path.append(stack.pop())
    if any(temp_graph[node] for node in temp_graph):
        return None
    path.reverse()
    return path

def transitive_closure(graph: Graph[T]) -> Dict[T, Set[T]]:
    nodes = list(get_all_nodes(graph))
    reach = {node: set([node]) for node in nodes}
    for node in nodes:
        reach[node].update(get_neighbors(graph, node))
    for k in nodes:
        for i in nodes:
            for j in nodes:
                if j in reach[k] and k in reach[i]:
                    reach[i].add(j)
    return reach

def all_simple_paths(graph: Graph[T], start: T, target: T) -> List[List[T]]:
    paths = []
    def backtrack(current: T, path: List[T]):
        if current == target:
            paths.append(path[:])
            return
        for neighbor in get_neighbors(graph, current):
            if neighbor not in path:
                path.append(neighbor)
                backtrack(neighbor, path)
                path.pop()
    backtrack(start, [start])
    return paths

def shortest_path_dijkstra_unweighted(graph: Graph[T], start: T, target: T) -> int:
    all_nodes = get_all_nodes(graph)
    dist = {node: sys.maxsize for node in all_nodes}
    dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, node = heapq.heappop(pq)
        if d > dist[node]:
            continue
        if node == target:
            return d
        for neighbor in get_neighbors(graph, node):
            new_dist = d + 1
            if new_dist < dist.get(neighbor, sys.maxsize):
                dist[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))
    return -1