
from collections import deque
import heapq
from typing import List, Optional, Any, Tuple

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

print("ArrayStack:")
print("Input: push(1), push(2), push(3), pop(), peek()")
stack = ArrayStack()
stack.push(1)
stack.push(2)
stack.push(3)
print(f"Output: pop()={stack.pop()}, peek()={stack.peek()}")

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

print("\nLinkedListStack:")
print("Input: push(5), push(10), pop()")
ll_stack = LinkedListStack()
ll_stack.push(5)
ll_stack.push(10)
print(f"Output: pop()={ll_stack.pop()}, size()={ll_stack.size()}")

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

print("\nMinStack:")
print("Input: push(3), push(5), push(2), push(1), get_min(), pop(), get_min()")
min_stack = MinStack()
min_stack.push(3)
min_stack.push(5)
min_stack.push(2)
min_stack.push(1)
print(f"Output: get_min()={min_stack.get_min()}, pop()={min_stack.pop()}, get_min()={min_stack.get_min()}")

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

print("\nMaxStack:")
print("Input: push(1), push(5), push(3), push(7), get_max(), pop(), get_max()")
max_stack = MaxStack()
max_stack.push(1)
max_stack.push(5)
max_stack.push(3)
max_stack.push(7)
print(f"Output: get_max()={max_stack.get_max()}, pop()={max_stack.pop()}, get_max()={max_stack.get_max()}")

class DLLNode:
    def __init__(self, val: int):
        self.val = val
        self.prev = None
        self.next = None

class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def add_to_head(self, node):
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
            self.middle = self.middle.next if self.middle else None
        self.count -= 1
        return val
    
    def find_middle(self) -> int:
        return self.middle.val if self.middle else None

print("\nStackWithMiddle:")
print("Input: push(1), push(2), push(3), push(4), find_middle(), pop(), find_middle()")
mid_stack = StackWithMiddle()
mid_stack.push(1)
mid_stack.push(2)
mid_stack.push(3)
mid_stack.push(4)
print(f"Output: find_middle()={mid_stack.find_middle()}, pop()={mid_stack.pop()}, find_middle()={mid_stack.find_middle()}")

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

print("\nTwoStacksInArray:")
print("Input: push1(1), push1(2), push2(3), push2(4), pop1(), pop2()")
two_stacks = TwoStacksInArray(10)
two_stacks.push1(1)
two_stacks.push1(2)
two_stacks.push2(3)
two_stacks.push2(4)
print(f"Output: pop1()={two_stacks.pop1()}, pop2()={two_stacks.pop2()}")

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

print("\nStackUsingQueue:")
print("Input: push(1), push(2), push(3), pop(), top()")
stack_q = StackUsingQueue()
stack_q.push(1)
stack_q.push(2)
stack_q.push(3)
print(f"Output: pop()={stack_q.pop()}, top()={stack_q.top()}")

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
    
    @staticmethod
    def infix_to_prefix(expression: str) -> str:
        reversed_exp = expression[::-1]
        modified_exp = ""
        
        for char in reversed_exp:
            if char == '(':
                modified_exp += ')'
            elif char == ')':
                modified_exp += '('
            else:
                modified_exp += char
        
        postfix = ExpressionEvaluator.infix_to_postfix(modified_exp)
        return postfix[::-1]

print("\nExpressionEvaluator:")
print("Input: 'a+b*c', '231*+9-'")
expr = "a+b*c"
postfix = ExpressionEvaluator.infix_to_postfix(expr)
print(f"Output: infix_to_postfix('{expr}')='{postfix}'")
result = ExpressionEvaluator.evaluate_postfix("231*+9-")
print(f"Output: evaluate_postfix('231*+9-')={result}")

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
    
    @staticmethod
    def longest_valid_parentheses(s: str) -> int:
        stack = [-1]
        max_len = 0
        
        for i, char in enumerate(s):
            if char == '(':
                stack.append(i)
            else:
                stack.pop()
                if not stack:
                    stack.append(i)
                else:
                    max_len = max(max_len, i - stack[-1])
        
        return max_len
    
    @staticmethod
    def remove_invalid_parentheses(s: str) -> List[str]:
        left_rem = right_rem = 0
        
        for char in s:
            if char == '(':
                left_rem += 1
            elif char == ')':
                if left_rem > 0:
                    left_rem -= 1
                else:
                    right_rem += 1
        
        result = set()
        
        def backtrack(index, left_count, right_count, left_rem, right_rem, expr):
            if index == len(s):
                if left_rem == 0 and right_rem == 0:
                    result.add(expr)
                return
            
            char = s[index]
            
            if char == '(' and left_rem > 0:
                backtrack(index + 1, left_count, right_count, left_rem - 1, right_rem, expr)
            
            if char == ')' and right_rem > 0:
                backtrack(index + 1, left_count, right_count, left_rem, right_rem - 1, expr)
            
            expr += char
            
            if char != '(' and char != ')':
                backtrack(index + 1, left_count, right_count, left_rem, right_rem, expr)
            elif char == '(':
                backtrack(index + 1, left_count + 1, right_count, left_rem, right_rem, expr)
            elif char == ')' and left_count > right_count:
                backtrack(index + 1, left_count, right_count + 1, left_rem, right_rem, expr)
        
        backtrack(0, 0, 0, left_rem, right_rem, "")
        return list(result)

print("\nParenthesesValidator:")
print("Input: '()[]{}', '(()', '()())'")
print(f"Output: is_valid('()[]')={ParenthesesValidator.is_valid('()[]{}')}")
print(f"Output: is_valid('(()')={ParenthesesValidator.is_valid('(()')}")
print(f"Output: longest_valid_parentheses('()())')={ParenthesesValidator.longest_valid_parentheses('()())')}")

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
    
    @staticmethod
    def next_greater_element_i(nums1: List[int], nums2: List[int]) -> List[int]:
        stack = []
        mapping = {}
        
        for num in nums2:
            while stack and stack[-1] < num:
                mapping[stack.pop()] = num
            stack.append(num)
        
        return [mapping.get(num, -1) for num in nums1]
    
    @staticmethod
    def daily_temperatures(temperatures: List[int]) -> List[int]:
        n = len(temperatures)
        result = [0] * n
        stack = []
        
        for i in range(n):
            while stack and temperatures[i] > temperatures[stack[-1]]:
                prev_index = stack.pop()
                result[prev_index] = i - prev_index
            stack.append(i)
        
        return result

print("\nNextGreaterElement:")
print("Input: [1, 2, 1], [1, 3, 4, 2]")
nge = NextGreaterElement.next_greater_elements([1, 2, 1])
print(f"Output: next_greater_elements([1, 2, 1])={nge}")
temps = NextGreaterElement.daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])
print(f"Output: daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73])={temps}")

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
    
    @staticmethod
    def maximal_rectangle(matrix: List[List[str]]) -> int:
        if not matrix or not matrix[0]:
            return 0
        
        heights = [0] * len(matrix[0])
        max_area = 0
        
        for row in matrix:
            for i in range(len(row)):
                heights[i] = heights[i] + 1 if row[i] == '1' else 0
            
            max_area = max(max_area, LargestRectangleHistogram.largest_rectangle_area(heights))
        
        return max_area

print("\nLargestRectangleHistogram:")
print("Input: [2, 1, 5, 6, 2, 3]")
area = LargestRectangleHistogram.largest_rectangle_area([2, 1, 5, 6, 2, 3])
print(f"Output: largest_rectangle_area([2, 1, 5, 6, 2, 3])={area}")

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
    
    @staticmethod
    def calculate_span(prices: List[int]) -> List[int]:
        stack = []
        spans = []
        
        for i, price in enumerate(prices):
            while stack and prices[stack[-1]] <= price:
                stack.pop()
            
            span = i + 1 if not stack else i - stack[-1]
            spans.append(span)
            stack.append(i)
        
        return spans

print("\nStockSpanProblem:")
print("Input: [100, 80, 60, 70, 60, 75, 85]")
spans = StockSpanProblem.calculate_span([100, 80, 60, 70, 60, 75, 85])
print(f"Output: calculate_span([100, 80, 60, 70, 60, 75, 85])={spans}")

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

print("\nStackSorter:")
print("Input: [3, 1, 4, 1, 5]")
sort_stack = [3, 1, 4, 1, 5]
StackSorter.sort_stack(sort_stack)
print(f"Output: sort_stack([3, 1, 4, 1, 5])={sort_stack}")

class TrappingRainWater:
    @staticmethod
    def trap(height: List[int]) -> int:
        if not height:
            return 0
        
        stack = []
        water = 0
        
        for i in range(len(height)):
            while stack and height[i] > height[stack[-1]]:
                top = stack.pop()
                if not stack:
                    break
                
                distance = i - stack[-1] - 1
                bounded_height = min(height[i], height[stack[-1]]) - height[top]
                water += distance * bounded_height
            
            stack.append(i)
        
        return water

print("\nTrappingRainWater:")
print("Input: [0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]")
trapped = TrappingRainWater.trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])
print(f"Output: trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1])={trapped}")

class AsteroidCollision:
    @staticmethod
    def asteroid_collision(asteroids: List[int]) -> List[int]:
        stack = []
        
        for asteroid in asteroids:
            while stack and asteroid < 0 < stack[-1]:
                if stack[-1] < -asteroid:
                    stack.pop()
                    continue
                elif stack[-1] == -asteroid:
                    stack.pop()
                break
            else:
                stack.append(asteroid)
        
        return stack

print("\nAsteroidCollision:")
print("Input: [5, 10, -5], [8, -8], [10, 2, -5]")
result1 = AsteroidCollision.asteroid_collision([5, 10, -5])
result2 = AsteroidCollision.asteroid_collision([8, -8])
result3 = AsteroidCollision.asteroid_collision([10, 2, -5])
print(f"Output: asteroid_collision([5, 10, -5])={result1}")
print(f"Output: asteroid_collision([8, -8])={result2}")
print(f"Output: asteroid_collision([10, 2, -5])={result3}")

class DecodeString:
    @staticmethod
    def decode_string(s: str) -> str:
        stack = []
        current_string = ""
        current_num = 0
        
        for char in s:
            if char.isdigit():
                current_num = current_num * 10 + int(char)
            elif char == '[':
                stack.append((current_string, current_num))
                current_string = ""
                current_num = 0
            elif char == ']':
                prev_string, num = stack.pop()
                current_string = prev_string + current_string * num
            else:
                current_string += char
        
        return current_string

print("\nDecodeString:")
print("Input: '3[a]2[bc]', '3[a2[c]]', '2[abc]3[cd]ef'")
decoded1 = DecodeString.decode_string("3[a]2[bc]")
decoded2 = DecodeString.decode_string("3[a2[c]]")
decoded3 = DecodeString.decode_string("2[abc]3[cd]ef")
print(f"Output: decode_string('3[a]2[bc]')='{decoded1}'")
print(f"Output: decode_string('3[a2[c]]')='{decoded2}'")
print(f"Output: decode_string('2[abc]3[cd]ef')='{decoded3}'")

class RemoveDuplicateLetters:
    @staticmethod
    def remove_duplicate_letters(s: str) -> str:
        last_occurrence = {char: i for i, char in enumerate(s)}
        stack = []
        in_stack = set()
        
        for i, char in enumerate(s):
            if char in in_stack:
                continue
            
            while (stack and stack[-1] > char and 
                   last_occurrence[stack[-1]] > i):
                removed = stack.pop()
                in_stack.remove(removed)
            
            stack.append(char)
            in_stack.add(char)
        
        return ''.join(stack)

print("\nRemoveDuplicateLetters:")
print("Input: 'bcabc', 'cbacdcbc'")
result1 = RemoveDuplicateLetters.remove_duplicate_letters("bcabc")
result2 = RemoveDuplicateLetters.remove_duplicate_letters("cbacdcbc")
print(f"Output: remove_duplicate_letters('bcabc')='{result1}'")
print(f"Output: remove_duplicate_letters('cbacdcbc')='{result2}'")

class BasicCalculator:
    @staticmethod
    def calculate(s: str) -> int:
        stack = []
        num = 0
        sign = '+'
        
        for i, char in enumerate(s):
            if char.isdigit():
                num = num * 10 + int(char)
            
            if char in '+-*/' or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    last = stack.pop()
                    stack.append(int(last / num))
                
                sign = char
                num = 0
        
        return sum(stack)

print("\nBasicCalculator:")
print("Input: '3+2*2', ' 3/2 ', ' 3+5 / 2 '")
calc1 = BasicCalculator.calculate("3+2*2")
calc2 = BasicCalculator.calculate(" 3/2 ")
calc3 = BasicCalculator.calculate(" 3+5 / 2 ")
print(f"Output: calculate('3+2*2')={calc1}")
print(f"Output: calculate(' 3/2 ')={calc2}")
print(f"Output: calculate(' 3+5 / 2 ')={calc3}")

class ScoreOfParentheses:
    @staticmethod
    def score_of_parentheses(s: str) -> int:
        stack = [0]
        
        for char in s:
            if char == '(':
                stack.append(0)
            else:
                v = stack.pop()
                stack[-1] += max(2 * v, 1)
        
        return stack[0]

print("\nScoreOfParentheses:")
print("Input: '()', '(())', '()()', '(()(()))'")
score1 = ScoreOfParentheses.score_of_parentheses("()")
score2 = ScoreOfParentheses.score_of_parentheses("(())")
score3 = ScoreOfParentheses.score_of_parentheses("()()")
score4 = ScoreOfParentheses.score_of_parentheses("(()(()))")
print(f"Output: score_of_parentheses('()')={score1}")
print(f"Output: score_of_parentheses('(())')={score2}")
print(f"Output: score_of_parentheses('()()')={score3}")
print(f"Output: score_of_parentheses('(()(()))')={score4}")

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

print("\nArrayQueue:")
print("Input: enqueue(1), enqueue(2), enqueue(3), dequeue(), peek()")
queue = ArrayQueue()
queue.enqueue(1)
queue.enqueue(2)
queue.enqueue(3)
print(f"Output: dequeue()={queue.dequeue()}, peek()={queue.peek()}")

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

print("\nLinkedListQueue:")
print("Input: enqueue(5), enqueue(10), dequeue(), get_size()")
ll_queue = LinkedListQueue()
ll_queue.enqueue(5)
ll_queue.enqueue(10)
print(f"Output: dequeue()={ll_queue.dequeue()}, get_size()={ll_queue.get_size()}")

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

print("\nCircularQueue:")
print("Input: enqueue(1), enqueue(2), enqueue(3), dequeue(), front(), rear()")
circular_queue = CircularQueue(3)
circular_queue.enqueue(1)
circular_queue.enqueue(2)
circular_queue.enqueue(3)
circular_queue.dequeue()
print(f"Output: front()={circular_queue.front()}, rear()={circular_queue.rear()}")

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

print("\nDeque:")
print("Input: add_front(1), add_rear(2), add_front(3), remove_rear(), remove_front()")
dq = Deque()
dq.add_front(1)
dq.add_rear(2)
dq.add_front(3)
print(f"Output: remove_rear()={dq.remove_rear()}, remove_front()={dq.remove_front()}")

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

print("\nPriorityQueue:")
print("Input: push('task1', 3), push('task2', 1), push('task3', 2), pop(), pop()")
pq = PriorityQueue()
pq.push('task1', 3)
pq.push('task2', 1)
pq.push('task3', 2)
print(f"Output: pop()={pq.pop()}, pop()={pq.pop()}")

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

print("\nQueueUsingStacks:")
print("Input: push(1), push(2), peek(), pop(), empty()")
queue_stack = QueueUsingStacks()
queue_stack.push(1)
queue_stack.push(2)
print(f"Output: peek()={queue_stack.peek()}, pop()={queue_stack.pop()}, empty()={queue_stack.empty()}")

class LRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = {}
        self.head = DLLNode(0)
        self.tail = DLLNode(0)
        self.head.next = self.tail
        self.tail.prev = self.head
    
    def add_node(self, node):
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node
    
    def remove_node(self, node):
        prev_node = node.prev
        new_node = node.next
        prev_node.next = new_node
        new_node.prev = prev_node
    
    def move_to_head(self, node):
        self.remove_node(node)
        self.add_node(node)
    
    def pop_tail(self):
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

print("\nLRUCache:")
print("Input: capacity=2, put(1,1), put(2,2), get(1), put(3,3), get(2), put(4,4), get(1), get(3), get(4)")
lru = LRUCache(2)
lru.put(1, 1)
lru.put(2, 2)
get1 = lru.get(1)
lru.put(3, 3)
get2 = lru.get(2)
lru.put(4, 4)
get3 = lru.get(1)
get4 = lru.get(3)
get5 = lru.get(4)
print(f"Output: get(1)={get1}, get(2)={get2}, get(1)={get3}, get(3)={get4}, get(4)={get5}")

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

print("\nSlidingWindowMaximum:")
print("Input: [1,3,-1,-3,5,3,6,7], k=3")
max_window = SlidingWindowMaximum.max_sliding_window([1, 3, -1, -3, 5, 3, 6, 7], 3)
print(f"Output: max_sliding_window([1,3,-1,-3,5,3,6,7], 3)={max_window}")

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

print("\nFirstNonRepeatingCharacter:")
print("Input: stream = 'aabc'")
fnr = FirstNonRepeatingCharacter()
results = []
for char in 'aabc':
    results.append(fnr.first_non_repeating(char))
print(f"Output: first_non_repeating('aabc')={results}")

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

print("\nBinaryNumberGenerator:")
print("Input: n=5")
binary_nums = BinaryNumberGenerator.generate_binary_numbers(5)
print(f"Output: generate_binary_numbers(5)={binary_nums}")

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

print("\nRottenOranges:")
print("Input: [[2,1,1],[1,1,0],[0,1,1]]")
grid = [[2, 1, 1], [1, 1, 0], [0, 1, 1]]
time_to_rot = RottenOranges.oranges_rotting(grid)
print(f"Output: oranges_rotting([[2,1,1],[1,1,0],[0,1,1]])={time_to_rot}")

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

print("\nMonotonicQueue:")
print("Input: push(3), push(1), push(4), max(), pop(3), max()")
mono_q = MonotonicQueue()
mono_q.push(3)
mono_q.push(1)
mono_q.push(4)
max1 = mono_q.max()
mono_q.pop(3)
max2 = mono_q.max()
print(f"Output: max()={max1}, after pop(3) max()={max2}")

class InterleaveQueue:
    @staticmethod
    def interleave_queue(queue: deque) -> deque:
        if len(queue) % 2 != 0:
            return queue
        
        stack = []
        half_size = len(queue) // 2
        
        for _ in range(half_size):
            stack.append(queue.popleft())
        
        while stack:
            queue.append(stack.pop())
        
        for _ in range(half_size):
            queue.append(queue.popleft())
        
        for _ in range(half_size):
            stack.append(queue.popleft())
        
        while stack:
            queue.appendleft(stack.pop())
            queue.appendleft(queue.pop())
        
        return queue

print("\nInterleaveQueue:")
print("Input: [1, 2, 3, 4, 5, 6, 7, 8]")
test_queue = deque([1, 2, 3, 4, 5, 6, 7, 8])
interleaved = InterleaveQueue.interleave_queue(test_queue)
print(f"Output: interleave_queue([1, 2, 3, 4, 5, 6, 7, 8])={list(interleaved)}")

class ReverseQueue:
    @staticmethod
    def reverse_queue(queue: deque) -> deque:
        if not queue:
            return queue
        
        front = queue.popleft()
        ReverseQueue.reverse_queue(queue)
        queue.append(front)
        return queue
    
    @staticmethod
    def reverse_first_k_elements(queue: deque, k: int) -> deque:
        if k <= 0 or k > len(queue):
            return queue
        
        stack = []
        
        for _ in range(k):
            stack.append(queue.popleft())
        
        while stack:
            queue.append(stack.pop())
        
        for _ in range(len(queue) - k):
            queue.append(queue.popleft())
        
        return queue

print("\nReverseQueue:")
print("Input: [1, 2, 3, 4, 5], k=3")
test_queue = deque([1, 2, 3, 4, 5])
reversed_k = ReverseQueue.reverse_first_k_elements(test_queue, 3)
print(f"Output: reverse_first_k_elements([1, 2, 3, 4, 5], 3)={list(reversed_k)}")

class CircularTour:
    @staticmethod
    def first_circular_tour(petrol: List[int], distance: List[int]) -> int:
        n = len(petrol)
        start = 0
        curr_petrol = 0
        total_petrol = 0
        total_distance = 0
        
        for i in range(n):
            total_petrol += petrol[i]
            total_distance += distance[i]
            curr_petrol += petrol[i] - distance[i]
            
            if curr_petrol < 0:
                start = i + 1
                curr_petrol = 0
        
        return start if total_petrol >= total_distance else -1

print("\nCircularTour:")
print("Input: petrol=[1, 2, 3, 4, 5], distance=[3, 4, 5, 1, 2]")
tour_start = CircularTour.first_circular_tour([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])
print(f"Output: first_circular_tour([1, 2, 3, 4, 5], [3, 4, 5, 1, 2])={tour_start}")

class QueueSortChecker:
    @staticmethod
    def can_sort_queue(queue: List[int]) -> bool:
        stack = []
        expected = 1
        i = 0
        
        while i < len(queue) or stack:
            if i < len(queue) and queue[i] == expected:
                expected += 1
                i += 1
            elif stack and stack[-1] == expected:
                stack.pop()
                expected += 1
            elif i < len(queue):
                stack.append(queue[i])
                i += 1
            else:
                return False
        
        return True

print("\nQueueSortChecker:")
print("Input: [5, 1, 2, 3, 4], [5, 1, 2, 6, 3, 4]")
can_sort1 = QueueSortChecker.can_sort_queue([5, 1, 2, 3, 4])
can_sort2 = QueueSortChecker.can_sort_queue([5, 1, 2, 6, 3, 4])
print(f"Output: can_sort_queue([5, 1, 2, 3, 4])={can_sort1}")
print(f"Output: can_sort_queue([5, 1, 2, 6, 3, 4])={can_sort2}")

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

print("\nStackReverser:")
print("Input: [1, 2, 3, 4, 5]")
reverse_stack = [1, 2, 3, 4, 5]
StackReverser.reverse_stack(reverse_stack)
print(f"Output: reverse_stack([1, 2, 3, 4, 5])={reverse_stack}")

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

print("\nCelebrityFinder:")
print("Input: knows_matrix=[[1,1,0],[0,1,0],[1,1,1]]")
celebrity = CelebrityFinder.find_celebrity([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
print(f"Output: find_celebrity([[1,1,0],[0,1,0],[1,1,1]])={celebrity}")

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

print("\nStackPermutationChecker:")
print("Input: input=[1, 2, 3], output=[2, 1, 3]")
is_valid_perm = StackPermutationChecker.is_stack_permutation([1, 2, 3], [2, 1, 3])
print(f"Output: is_stack_permutation([1, 2, 3], [2, 1, 3])={is_valid_perm}")

class QueueWithMiddle:
    def __init__(self):
        self.queue = deque()
    
    def enqueue_front(self, x: int):
        self.queue.appendleft(x)
    
    def enqueue_rear(self, x: int):
        self.queue.append(x)
    
    def dequeue_front(self) -> int:
        return self.queue.popleft() if self.queue else None
    
    def dequeue_rear(self) -> int:
        return self.queue.pop() if self.queue else None
    
    def get_middle(self) -> int:
        if not self.queue:
            return None
        n = len(self.queue)
        return self.queue[n // 2]
    
    def delete_middle(self) -> int:
        if not self.queue:
            return None
        n = len(self.queue)
        mid_idx = n // 2
        temp_queue = deque()
        
        for _ in range(mid_idx):
            temp_queue.append(self.queue.popleft())
        
        middle = self.queue.popleft()
        
        while temp_queue:
            self.queue.appendleft(temp_queue.pop())
        
        return middle

print("\nQueueWithMiddle:")
print("Input: enqueue_rear(1), enqueue_rear(2), enqueue_rear(3), get_middle(), delete_middle()")
queue_mid = QueueWithMiddle()
queue_mid.enqueue_rear(1)
queue_mid.enqueue_rear(2)
queue_mid.enqueue_rear(3)
middle = queue_mid.get_middle()
deleted_middle = queue_mid.delete_middle()
print(f"Output: get_middle()={middle}, delete_middle()={deleted_middle}")
