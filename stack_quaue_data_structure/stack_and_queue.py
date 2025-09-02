
from collections import deque
import heapq
from typing import List, Optional, Any

class ArrayStack:
    def __init__(self, capacity: int = 1000):
        self.data = [None] * capacity
        self.top_index = -1
        self.capacity = capacity
    
    def push(self, item: Any) -> bool:
        if self.top_index >= self.capacity - 1:
            return False
        self.top_index += 1
        self.data[self.top_index] = item
        return True
    
    def pop(self) -> Any:
        if self.top_index < 0:
            return None
        item = self.data[self.top_index]
        self.top_index -= 1
        return item
    
    def peek(self) -> Any:
        return self.data[self.top_index] if self.top_index >= 0 else None
    
    def is_empty(self) -> bool:
        return self.top_index < 0
    
    def size(self) -> int:
        return self.top_index + 1

class ListNode:
    def __init__(self, val: Any = 0, next_node=None):
        self.val = val
        self.next = next_node

class LinkedListStack:
    def __init__(self):
        self.head = None
        self.length = 0
    
    def push(self, item: Any):
        new_node = ListNode(item, self.head)
        self.head = new_node
        self.length += 1
    
    def pop(self) -> Any:
        if not self.head:
            return None
        val = self.head.val
        self.head = self.head.next
        self.length -= 1
        return val
    
    def peek(self) -> Any:
        return self.head.val if self.head else None
    
    def is_empty(self) -> bool:
        return self.head is None
    
    def size(self) -> int:
        return self.length

class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []
    
    def push(self, val: int):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)
    
    def pop(self) -> int:
        if not self.stack:
            return None
        val = self.stack.pop()
        if val == self.min_stack[-1]:
            self.min_stack.pop()
        return val
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def get_min(self) -> int:
        return self.min_stack[-1] if self.min_stack else None

class MaxStack:
    def __init__(self):
        self.stack = []
        self.max_stack = []
    
    def push(self, val: int):
        self.stack.append(val)
        if not self.max_stack or val >= self.max_stack[-1]:
            self.max_stack.append(val)
    
    def pop(self) -> int:
        if not self.stack:
            return None
        val = self.stack.pop()
        if val == self.max_stack[-1]:
            self.max_stack.pop()
        return val
    
    def top(self) -> int:
        return self.stack[-1] if self.stack else None
    
    def get_max(self) -> int:
        return self.max_stack[-1] if self.max_stack else None

class StackWithMiddle:
    def __init__(self):
        self.dll = DoublyLinkedList()
        self.middle = None
        self.count = 0
    
    def push(self, val: int):
        new_node = DLLNode(val)
        if self.count == 0:
            self.dll.head = self.dll.tail = self.middle = new_node
        else:
            self.dll.add_to_head(new_node)
            if self.count % 2 == 0:
                self.middle = self.middle.prev
        self.count += 1
    
    def pop(self) -> int:
        if self.count == 0:
            return None
        val = self.dll.head.val
        self.dll.remove_head()
        if self.count % 2 == 1:
            self.middle = self.middle.next
        self.count -= 1
        return val
    
    def find_middle(self) -> int:
        return self.middle.val if self.middle else None

class DLLNode:
    def __init__(self, val: int):
        self.val = val
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def add_to_head(self, node: DLLNode):
        if not self.head:
            self.head = self.tail = node
        else:
            node.next = self.head
            self.head.prev = node
            self.head = node
    
    def remove_head(self):
        if not self.head:
            return
        if self.head == self.tail:
            self.head = self.tail = None
        else:
            self.head = self.head.next
            self.head.prev = None

class TwoStacksInArray:
    def __init__(self, capacity: int):
        self.data = [None] * capacity
        self.top1 = -1
        self.top2 = capacity
        self.capacity = capacity
    
    def push1(self, val: Any) -> bool:
        if self.top1 < self.top2 - 1:
            self.top1 += 1
            self.data[self.top1] = val
            return True
        return False
    
    def push2(self, val: Any) -> bool:
        if self.top1 < self.top2 - 1:
            self.top2 -= 1
            self.data[self.top2] = val
            return True
        return False
    
    def pop1(self) -> Any:
        if self.top1 >= 0:
            val = self.data[self.top1]
            self.top1 -= 1
            return val
        return None
    
    def pop2(self) -> Any:
        if self.top2 < self.capacity:
            val = self.data[self.top2]
            self.top2 += 1
            return val
        return None

class StackUsingQueue:
    def __init__(self):
        self.q1 = deque()
        self.q2 = deque()
    
    def push(self, val: Any):
        self.q2.append(val)
        while self.q1:
            self.q2.append(self.q1.popleft())
        self.q1, self.q2 = self.q2, self.q1
    
    def pop(self) -> Any:
        return self.q1.popleft() if self.q1 else None
    
    def top(self) -> Any:
        return self.q1[0] if self.q1 else None
    
    def empty(self) -> bool:
        return len(self.q1) == 0

class ExpressionEvaluator:
    @staticmethod
    def precedence(op: str) -> int:
        if op in '+-':
            return 1
        if op in '*/':
            return 2
        if op == '^':
            return 3
        return 0
    
    @staticmethod
    def infix_to_postfix(expression: str) -> str:
        stack = []
        result = []
        
        for char in expression:
            if char.isalnum():
                result.append(char)
            elif char == '(':
                stack.append(char)
            elif char == ')':
                while stack and stack[-1] != '(':
                    result.append(stack.pop())
                stack.pop()
            else:
                while (stack and stack[-1] != '(' and 
                       ExpressionEvaluator.precedence(char) <= ExpressionEvaluator.precedence(stack[-1])):
                    result.append(stack.pop())
                stack.append(char)
        
        while stack:
            result.append(stack.pop())
        
        return ''.join(result)
    
    @staticmethod
    def evaluate_postfix(expression: str) -> int:
        stack = []
        
        for char in expression:
            if char.isdigit():
                stack.append(int(char))
            else:
                b = stack.pop()
                a = stack.pop()
                if char == '+':
                    stack.append(a + b)
                elif char == '-':
                    stack.append(a - b)
                elif char == '*':
                    stack.append(a * b)
                elif char == '/':
                    stack.append(a // b)
        
        return stack[0]

class ParenthesesValidator:
    @staticmethod
    def is_valid(s: str) -> bool:
        stack = []
        mapping = {')': '(', '}': '{', ']': '['}
        
        for char in s:
            if char in mapping:
                if not stack or stack.pop() != mapping[char]:
                    return False
            else:
                stack.append(char)
        
        return not stack

class NextGreaterElement:
    @staticmethod
    def next_greater_elements(nums: List[int]) -> List[int]:
        n = len(nums)
        result = [-1] * n
        stack = []
        
        for i in range(2 * n):
            while stack and nums[stack[-1]] < nums[i % n]:
                result[stack.pop()] = nums[i % n]
            if i < n:
                stack.append(i)
        
        return result

class LargestRectangleHistogram:
    @staticmethod
    def largest_rectangle_area(heights: List[int]) -> int:
        stack = []
        max_area = 0
        index = 0
        
        while index < len(heights):
            if not stack or heights[index] >= heights[stack[-1]]:
                stack.append(index)
                index += 1
            else:
                top = stack.pop()
                width = index if not stack else index - stack[-1] - 1
                area = heights[top] * width
                max_area = max(max_area, area)
        
        while stack:
            top = stack.pop()
            width = index if not stack else index - stack[-1] - 1
            area = heights[top] * width
            max_area = max(max_area, area)
        
        return max_area

class StockSpanProblem:
    def __init__(self):
        self.stack = []
        self.index = 0
    
    def next(self, price: int) -> int:
        while self.stack and self.stack[-1][0] <= price:
            self.stack.pop()
        
        span = self.index + 1 if not self.stack else self.index - self.stack[-1][1]
        self.stack.append((price, self.index))
        self.index += 1
        
        return span

class StackSorter:
    @staticmethod
    def sort_stack(stack: List[int]):
        if not stack:
            return
        
        temp = stack.pop()
        StackSorter.sort_stack(stack)
        StackSorter.sorted_insert(stack, temp)
    
    @staticmethod
    def sorted_insert(stack: List[int], item: int):
        if not stack or item > stack[-1]:
            stack.append(item)
            return
        
        temp = stack.pop()
        StackSorter.sorted_insert(stack, item)
        stack.append(temp)

class ArrayQueue:
    def __init__(self, capacity: int = 1000):
        self.data = [None] * capacity
        self.front = 0
        self.rear = -1
        self.size = 0
        self.capacity = capacity
    
    def enqueue(self, item: Any) -> bool:
        if self.size >= self.capacity:
            return False
        self.rear = (self.rear + 1) % self.capacity
        self.data[self.rear] = item
        self.size += 1
        return True
    
    def dequeue(self) -> Any:
        if self.size == 0:
            return None
        item = self.data[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1
        return item
    
    def peek(self) -> Any:
        return self.data[self.front] if self.size > 0 else None
    
    def is_empty(self) -> bool:
        return self.size == 0
    
    def is_full(self) -> bool:
        return self.size == self.capacity

class LinkedListQueue:
    def __init__(self):
        self.front = None
        self.rear = None
        self.size = 0
    
    def enqueue(self, item: Any):
        new_node = ListNode(item)
        if self.rear is None:
            self.front = self.rear = new_node
        else:
            self.rear.next = new_node
            self.rear = new_node
        self.size += 1
    
    def dequeue(self) -> Any:
        if self.front is None:
            return None
        item = self.front.val
        self.front = self.front.next
        if self.front is None:
            self.rear = None
        self.size -= 1
        return item
    
    def peek(self) -> Any:
        return self.front.val if self.front else None
    
    def is_empty(self) -> bool:
        return self.front is None
    
    def get_size(self) -> int:
        return self.size

class CircularQueue:
    def __init__(self, k: int):
        self.data = [0] * k
        self.head = -1
        self.tail = -1
        self.size = k
    
    def enqueue(self, value: int) -> bool:
        if self.is_full():
            return False
        if self.is_empty():
            self.head = 0
        self.tail = (self.tail + 1) % self.size
        self.data[self.tail] = value
        return True
    
    def dequeue(self) -> bool:
        if self.is_empty():
            return False
        if self.head == self.tail:
            self.head = -1
            self.tail = -1
        else:
            self.head = (self.head + 1) % self.size
        return True
    
    def front(self) -> int:
        return -1 if self.is_empty() else self.data[self.head]
    
    def rear(self) -> int:
        return -1 if self.is_empty() else self.data[self.tail]
    
    def is_empty(self) -> bool:
        return self.head == -1
    
    def is_full(self) -> bool:
        return ((self.tail + 1) % self.size) == self.head

class Deque:
    def __init__(self):
        self.items = []
    
    def add_front(self, item: Any):
        self.items.insert(0, item)
    
    def add_rear(self, item: Any):
        self.items.append(item)
    
    def remove_front(self) -> Any:
        return self.items.pop(0) if self.items else None
    
    def remove_rear(self) -> Any:
        return self.items.pop() if self.items else None
    
    def is_empty(self) -> bool:
        return len(self.items) == 0
    
    def size(self) -> int:
        return len(self.items)

class PriorityQueue:
    def __init__(self, is_min_heap: bool = True):
        self.heap = []
        self.is_min = is_min_heap
    
    def push(self, item: Any, priority: int):
        if self.is_min:
            heapq.heappush(self.heap, (priority, item))
        else:
            heapq.heappush(self.heap, (-priority, item))
    
    def pop(self) -> Any:
        if not self.heap:
            return None
        priority, item = heapq.heappop(self.heap)
        return item
    
    def peek(self) -> Any:
        return self.heap[0][1] if self.heap else None
    
    def is_empty(self) -> bool:
        return len(self.heap) == 0

class QueueUsingStacks:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []
    
    def push(self, x: int):
        self.in_stack.append(x)
    
    def pop(self) -> int:
        self.peek()
        return self.out_stack.pop() if self.out_stack else None
    
    def peek(self) -> int:
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
        return self.out_stack[-1] if self.out_stack else None
    
    def empty(self) -> bool:
        return not self.in_stack and not self.out_stack

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = DLLNode(0)
        self.tail = DLLNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def add_node(self, node: DLLNode):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def remove_node(self, node: DLLNode):
        prev_node = node.prev
        new_node = node.next
        prev_node.next = new_node
        new_node.prev = prev_node
    
    def move_to_head(self, node: DLLNode):
        self.remove_node(node)
        self.add_node(node)
    
    def pop_tail(self) -> DLLNode:
        last_node = self.tail.prev
        self.remove_node(last_node)
        return last_node
    
    def get(self, key: int) -> int:
        node = self.cache.get(key)
        if not node:
            return -1
        self.move_to_head(node)
        return node.val
    
    def put(self, key: int, value: int):
        node = self.cache.get(key)
        if not node:
            new_node = DLLNode(value)
            new_node.key = key
            self.cache[key] = new_node
            self.add_node(new_node)
            if len(self.cache) > self.capacity:
                tail = self.pop_tail()
                del self.cache[tail.key]
        else:
            node.val = value
            self.move_to_head(node)

class SlidingWindowMaximum:
    @staticmethod
    def max_sliding_window(nums: List[int], k: int) -> List[int]:
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

class FirstNonRepeatingCharacter:
    def __init__(self):
        self.queue = deque()
        self.char_count = {}
    
    def first_non_repeating(self, char: str) -> str:
        self.queue.append(char)
        self.char_count[char] = self.char_count.get(char, 0) + 1
        
        while self.queue and self.char_count[self.queue[0]] > 1:
            self.queue.popleft()
        
        return self.queue[0] if self.queue else '#'

class BinaryNumberGenerator:
    @staticmethod
    def generate_binary_numbers(n: int) -> List[str]:
        queue = deque(['1'])
        result = []
        
        for _ in range(n):
            front = queue.popleft()
            result.append(front)
            queue.append(front + '0')
            queue.append(front + '1')
        
        return result

class RottenOranges:
    @staticmethod
    def oranges_rotting(grid: List[List[int]]) -> int:
        if not grid or not grid[0]:
            return 0
        
        rows, cols = len(grid), len(grid[0])
        queue = deque()
        fresh_count = 0
        
        for r in range(rows):
            for c in range(cols):
                if grid[r][c] == 2:
                    queue.append((r, c))
                elif grid[r][c] == 1:
                    fresh_count += 1
        
        if fresh_count == 0:
            return 0
        
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        minutes = 0
        
        while queue:
            size = len(queue)
            for _ in range(size):
                row, col = queue.popleft()
                for dr, dc in directions:
                    nr, nc = row + dr, col + dc
                    if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                        grid[nr][nc] = 2
                        fresh_count -= 1
                        queue.append((nr, nc))
            
            if queue:
                minutes += 1
        
        return minutes if fresh_count == 0 else -1

class MonotonicQueue:
    def __init__(self):
        self.data = deque()
    
    def push(self, val: int):
        while self.data and self.data[-1] < val:
            self.data.pop()
        self.data.append(val)
    
    def max(self) -> int:
        return self.data[0] if self.data else None
    
    def pop(self, val: int):
        if self.data and self.data[0] == val:
            self.data.popleft()

class QueueWithMinMax:
    def __init__(self):
        self.data = deque()
        self.min_deque = deque()
        self.max_deque = deque()
    
    def enqueue(self, val: int):
        self.data.append(val)
        
        while self.min_deque and self.min_deque[-1] > val:
            self.min_deque.pop()
        self.min_deque.append(val)
        
        while self.max_deque and self.max_deque[-1] < val:
            self.max_deque.pop()
        self.max_deque.append(val)
    
    def dequeue(self) -> int:
        if not self.data:
            return None
        
        val = self.data.popleft()
        
        if self.min_deque and self.min_deque[0] == val:
            self.min_deque.popleft()
        
        if self.max_deque and self.max_deque[0] == val:
            self.max_deque.popleft()
        
        return val
    
    def get_min(self) -> int:
        return self.min_deque[0] if self.min_deque else None
    
    def get_max(self) -> int:
        return self.max_deque[0] if self.max_deque else None

class StackPermutationChecker:
    @staticmethod
    def is_stack_permutation(input_seq: List[int], output_seq: List[int]) -> bool:
        if len(input_seq) != len(output_seq):
            return False
        
        stack = []
        input_idx = 0
        
        for num in output_seq:
            while (input_idx < len(input_seq) and 
                   (not stack or stack[-1] != num)):
                stack.append(input_seq[input_idx])
                input_idx += 1
            
            if not stack or stack[-1] != num:
                return False
            
            stack.pop()
        
        return True

class CelebrityFinder:
    @staticmethod
    def find_celebrity(knows_matrix: List[List[int]]) -> int:
        n = len(knows_matrix)
        stack = list(range(n))
        
        while len(stack) > 1:
            a = stack.pop()
            b = stack.pop()
            
            if knows_matrix[a][b]:
                stack.append(b)
            else:
                stack.append(a)
        
        candidate = stack[0]
        
        for i in range(n):
            if i != candidate:
                if knows_matrix[candidate][i] or not knows_matrix[i][candidate]:
                    return -1
        
        return candidate

class StackReverser:
    @staticmethod
    def reverse_stack(stack: List[int]):
        if not stack:
            return
        
        temp = stack.pop()
        StackReverser.reverse_stack(stack)
        StackReverser.insert_at_bottom(stack, temp)
    
    @staticmethod
    def insert_at_bottom(stack: List[int], item: int):
        if not stack:
            stack.append(item)
            return
        
        temp = stack.pop()
        StackReverser.insert_at_bottom(stack, item)
        stack.append(temp)