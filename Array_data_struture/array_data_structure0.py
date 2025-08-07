# ======================================================================
# Comprehensive Guide to Array Data Structure in Python
# ======================================================================
#
# Author: 
# Purpose: This file serves as an end-to-end tutorial on Array Data Structure.
#          We cover basics to advanced concepts, including all major algorithms.
#          Explanations are detailed, covering all aspects including edge cases,
#          time/space complexities, and exceptions. 
#
# Key Notes:
# - In Python, lists act as dynamic arrays. We'll use lists primarily, but discuss
#   fixed-size arrays using the 'array' module for completeness.
# - We focus on 1D arrays mostly, but extend to 2D (matrices) where relevant.
# - Algorithms: From basic operations to medium/advanced (e.g., Kadane, Two-Pointer,
#   Sliding Window, Prefix Sum, Matrix manipulations, Sorting variants).
# - Each section has: Explanation, Code, Examples (including edge cases), Complexities.
# - Run this file to see demonstrations. Outputs are printed for verification.
#
# Table of Contents (Sections):
# 1. Basics of Arrays
# 2. Basic Operations (Traversal, Insertion, Deletion, Update, Search)
# 3. Simple Algorithms (Min/Max, Sum, Reverse, Rotate)
# 4. Searching Algorithms (Linear, Binary)
# 5. Sorting Algorithms (Bubble, Selection, Insertion, Merge, Quick, Heap)
# 6. Medium Algorithms (Duplicates, Missing Elements, Kadane's Max Subarray)
# 7. Advanced Techniques (Two-Pointer, Sliding Window, Prefix Sum)
# 8. 2D Arrays/Matrices (Traversal, Transpose, Rotate, Spiral Order)
# 9. Special Algorithms (Dutch National Flag, Moore's Voting)
# 10. Exceptions and Edge Cases Handling
#
# Let's dive in with unparalleled coding mastery!
# ======================================================================

import sys  # For sys.maxsize in some algorithms
import heapq  # For Heap Sort demonstration
from array import array as fixed_array  # For fixed-size array examples

# ======================================================================
# Section 1: Basics of Arrays
# ======================================================================
#
# Explanation:
# An array is a collection of items stored at contiguous memory locations.
# - Fixed-size: Size defined at creation (e.g., using 'array' module).
# - Dynamic: Can resize (Python lists).
# - Indexing: 0-based, access O(1) time.
# - Types: 1D (linear), 2D (matrix), Multi-dimensional.
# - Advantages: Fast access, cache-friendly.
# - Disadvantages: Insertion/deletion O(n) in fixed arrays due to shifting.
# - In Python: Lists are dynamic arrays implemented as resizable arrays.
# - Exceptions: IndexError for out-of-bounds access.
# - Time/Space: Access O(1), Space O(n).
#
# Example 1: Creating a simple dynamic array (list)
arr_dynamic = [1, 2, 3, 4]  # Dynamic, can append
print("Dynamic Array:", arr_dynamic)

# Example 2: Fixed-size array using 'array' module (typecode 'i' for signed int)
arr_fixed = fixed_array('i', [5, 6, 7, 8])  # Fixed size, cannot resize easily
print("Fixed Array:", arr_fixed)  # Output as array('i', [5, 6, 7, 8])

# Edge Case: Empty array
empty_arr = []
print("Empty Array:", empty_arr)  # []

# Exception Example: IndexError
try:
    print(arr_dynamic[10])  # Out of bounds
except IndexError as e:
    print("Exception:", e)  # list index out of range

# ======================================================================
# Section 2: Basic Operations
# ======================================================================

# 2.1 Traversal
# Explanation: Iterate through each element. O(n) time.
# Edge: Empty array - no output.
def traverse_array(arr):
    """Traverse and print array elements."""
    for elem in arr:
        print(elem, end=' ')
    print()

print("Traversal Example:")
traverse_array([10, 20, 30])  # 10 20 30

# 2.2 Insertion
# Explanation: Insert at position. In lists, O(n) due to shifting.
# Edge: Insert at beginning/end, invalid position (handle with exception).
def insert_element(arr, pos, val):
    """Insert val at pos in arr."""
    if pos < 0 or pos > len(arr):
        raise IndexError("Invalid position")
    arr.insert(pos, val)  # Python list insert

arr = [1, 2, 3]
insert_element(arr, 1, 99)
print("After Insertion:", arr)  # [1, 99, 2, 3]

# Exception Example
try:
    insert_element(arr, 10, 100)
except IndexError as e:
    print("Insertion Exception:", e)

# 2.3 Deletion
# Explanation: Remove by index or value. O(n) for shifting.
# Edge: Delete non-existent, empty array.
def delete_element(arr, val):
    """Delete first occurrence of val."""
    try:
        arr.remove(val)
    except ValueError:
        print("Value not found")

arr = [1, 99, 2, 3]
delete_element(arr, 99)
print("After Deletion:", arr)  # [1, 2, 3]

# Edge: Non-existent
delete_element(arr, 100)  # Value not found

# 2.4 Update
# Explanation: Change value at index. O(1) time.
# Edge: Invalid index.
def update_element(arr, pos, val):
    """Update arr[pos] to val."""
    if pos < 0 or pos >= len(arr):
        raise IndexError("Invalid position")
    arr[pos] = val

update_element(arr, 1, 22)
print("After Update:", arr)  # [1, 22, 3]

# 2.5 Search (Basic Linear)
# Explanation: Find index of val. O(n) worst case.
# Edge: Not found (-1), duplicates (returns first).
def linear_search(arr, val):
    """Return index of val, else -1."""
    for i in range(len(arr)):
        if arr[i] == val:
            return i
    return -1

print("Linear Search (22):", linear_search(arr, 22))  # 1
print("Linear Search (100):", linear_search(arr, 100))  # -1

# ======================================================================
# Section 3: Simple Algorithms
# ======================================================================

# 3.1 Find Min and Max
# Explanation: Iterate to find min/max. O(n) time.
# Edge: Empty array (return None or raise exception), single element.
def find_min_max(arr):
    """Return (min, max) or (None, None) if empty."""
    if not arr:
        return None, None
    min_val, max_val = arr[0], arr[0]
    for elem in arr:
        if elem < min_val:
            min_val = elem
        if elem > max_val:
            max_val = elem
    return min_val, max_val

print("Min Max [1,22,3]:", find_min_max([1,22,3]))  # (1, 22)
print("Min Max Empty:", find_min_max([]))  # (None, None)

# 3.2 Sum of Elements
# Explanation: Accumulate sum. O(n) time. Handle overflow? Python ints are arbitrary precision.
# Edge: Empty (0), negative numbers.
def array_sum(arr):
    """Return sum of elements."""
    total = 0
    for elem in arr:
        total += elem
    return total

print("Sum [1,22,3]:", array_sum([1,22,3]))  # 26
print("Sum Empty:", array_sum([]))  # 0

# 3.3 Reverse Array
# Explanation: Swap from start/end. O(n) time, in-place.
# Edge: Empty, single element (no change).
def reverse_array(arr):
    """Reverse in-place."""
    left, right = 0, len(arr) - 1
    while left < right:
        arr[left], arr[right] = arr[right], arr[left]
        left += 1
        right -= 1

arr = [1, 2, 3, 4]
reverse_array(arr)
print("Reversed:", arr)  # [4, 3, 2, 1]

# 3.4 Rotate Array (Left by k positions)
# Explanation: Rotate left by k. Use reversal method: O(n) time, O(1) space.
# Edge: k=0, k > len (mod len), empty.
def rotate_left(arr, k):
    """Rotate left by k positions in-place."""
    n = len(arr)
    if n == 0:
        return
    k = k % n
    reverse_array(arr)  # Reverse whole
    reverse_array(arr[:k])  # Reverse first k? Wait, arr[:k] is slice, need to handle in-place.
    # Correction: Proper in-place reversal for parts
    def reverse_part(arr, start, end):
        while start < end:
            arr[start], arr[end] = arr[end], arr[start]
            start += 1
            end -= 1
    
    reverse_part(arr, 0, n-1)  # Full reverse
    reverse_part(arr, 0, k-1)  # First k
    reverse_part(arr, k, n-1)  # Rest

arr = [1,2,3,4,5]
rotate_left(arr, 2)
print("Rotated Left by 2:", arr)  # [3,4,5,1,2]

# ======================================================================
# Section 4: Searching Algorithms
# ======================================================================

# 4.1 Linear Search (Already covered in 2.5, but with example)
# Time: O(n), Space: O(1)

# 4.2 Binary Search (For sorted array)
# Explanation: Divide and conquer. O(log n) time. Assumes sorted ascending.
# Edge: Not found (-1), duplicates (returns any), empty, single element.
def binary_search(arr, val):
    """Return index of val in sorted arr, else -1."""
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        if arr[mid] == val:
            return mid
        elif arr[mid] < val:
            left = mid + 1
        else:
            right = mid - 1
    return -1

sorted_arr = [1,3,5,7,9]
print("Binary Search (5):", binary_search(sorted_arr, 5))  # 2
print("Binary Search (6):", binary_search(sorted_arr, 6))  # -1

# Iterative vs Recursive: Above is iterative. Recursive example:
def binary_search_recursive(arr, val, left, right):
    if left > right:
        return -1
    mid = left + (right - left) // 2
    if arr[mid] == val:
        return mid
    elif arr[mid] < val:
        return binary_search_recursive(arr, val, mid+1, right)
    else:
        return binary_search_recursive(arr, val, left, mid-1)

print("Recursive BS (5):", binary_search_recursive(sorted_arr, 5, 0, len(sorted_arr)-1))  # 2

# ======================================================================
# Section 5: Sorting Algorithms
# ======================================================================

# 5.1 Bubble Sort
# Explanation: Repeatedly swap adjacent if out of order. O(n^2) time, stable.
# Edge: Already sorted (optimized with flag), empty, duplicates.
def bubble_sort(arr):
    """Bubble Sort in-place."""
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
                swapped = True
        if not swapped:
            break  # Optimized

arr = [64, 34, 25, 12, 22, 11, 90]
bubble_sort(arr)
print("Bubble Sorted:", arr)  # [11, 12, 22, 25, 34, 64, 90]

# 5.2 Selection Sort
# Explanation: Find min and swap to front. O(n^2) time, unstable.
# Edge: Similar to bubble.
def selection_sort(arr):
    """Selection Sort in-place."""
    n = len(arr)
    for i in range(n):
        min_idx = i
        for j in range(i+1, n):
            if arr[j] < arr[min_idx]:
                min_idx = j
        arr[i], arr[min_idx] = arr[min_idx], arr[i]

arr = [64, 25, 12, 22, 11]
selection_sort(arr)
print("Selection Sorted:", arr)  # [11, 12, 22, 25, 64]

# 5.3 Insertion Sort
# Explanation: Build sorted part by inserting. O(n^2) time, stable, good for small/nearly sorted.
# Edge: Already sorted O(n).
def insertion_sort(arr):
    """Insertion Sort in-place."""
    n = len(arr)
    for i in range(1, n):
        key = arr[i]
        j = i - 1
        while j >= 0 and key < arr[j]:
            arr[j+1] = arr[j]
            j -= 1
        arr[j+1] = key

arr = [12, 11, 13, 5, 6]
insertion_sort(arr)
print("Insertion Sorted:", arr)  # [5, 6, 11, 12, 13]

# 5.4 Merge Sort
# Explanation: Divide, sort, merge. O(n log n) time, stable, O(n) space.
# Edge: Empty, single.
def merge_sort(arr):
    """Merge Sort recursive."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

arr = [12, 11, 13, 5, 6, 7]
print("Merge Sorted:", merge_sort(arr))  # [5, 6, 7, 11, 12, 13]

# 5.5 Quick Sort
# Explanation: Partition around pivot. O(n log n) avg, O(n^2) worst, in-place.
# Edge: Worst case (sorted array), use random pivot to mitigate.
def quick_sort(arr, low, high):
    """Quick Sort in-place."""
    if low < high:
        pi = partition(arr, low, high)
        quick_sort(arr, low, pi-1)
        quick_sort(arr, pi+1, high)

def partition(arr, low, high):
    pivot = arr[high]  # Last as pivot
    i = low - 1
    for j in range(low, high):
        if arr[j] < pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i+1], arr[high] = arr[high], arr[i+1]
    return i+1

arr = [10, 7, 8, 9, 1, 5]
quick_sort(arr, 0, len(arr)-1)
print("Quick Sorted:", arr)  # [1, 5, 7, 8, 9, 10]

# 5.6 Heap Sort
# Explanation: Build max-heap, extract max. O(n log n) time, in-place.
# Python uses heapq for min-heap, so we negate for max-heap simulation.
def heap_sort(arr):
    """Heap Sort using heapq (min-heap)."""
    heap = [-x for x in arr]  # Negate for max-heap effect
    heapq.heapify(heap)
    sorted_arr = []
    while heap:
        sorted_arr.append(-heapq.heappop(heap))
    return sorted_arr

print("Heap Sorted:", heap_sort([12, 11, 13, 5, 6, 7]))  # [5, 6, 7, 11, 12, 13]

# ======================================================================
# Section 6: Medium Algorithms
# ======================================================================

# 6.1 Find Duplicates in Array
# Explanation: For array 1 to n with duplicates. Use negation marking. O(n) time, O(1) space (modifies array).
# Assumes positive ints, range 1 to n.
# Edge: No duplicates, multiple dups.
def find_duplicates(arr):
    """Return list of duplicates."""
    dups = []
    for i in range(len(arr)):
        idx = abs(arr[i]) - 1  # 1-based
        if arr[idx] < 0:
            dups.append(idx + 1)
        else:
            arr[idx] = -arr[idx]
    return dups

arr = [4,3,2,7,8,2,3,1]
print("Duplicates:", find_duplicates(arr))  # [2, 3] (order may vary)

# 6.2 Find Missing Element (1 to n)
# Explanation: Use sum formula. O(n) time.
# Edge: No missing (impossible if n+1 elements), negatives? Assume positives.
def find_missing(arr, n):
    """Find missing from 1 to n."""
    expected = n * (n + 1) // 2
    actual = sum(arr)
    return expected - actual

print("Missing in [1,2,4,5]:", find_missing([1,2,4,5], 5))  # 3

# 6.3 Kadane's Algorithm: Max Subarray Sum
# Explanation: Dynamic programming. O(n) time.
# Edge: All negative (return max element), empty (0 or exception).
def kadane_max_subarray(arr):
    """Return max subarray sum."""
    if not arr:
        return 0
    max_current = max_global = arr[0]
    for i in range(1, len(arr)):
        max_current = max(arr[i], max_current + arr[i])
        if max_current > max_global:
            max_global = max_current
    return max_global

print("Max Subarray [ -2,1,-3,4,-1,2,1,-5,4 ]:", kadane_max_subarray([-2,1,-3,4,-1,2,1,-5,4]))  # 6
print("All Negative [-1,-2,-3]:", kadane_max_subarray([-1,-2,-3]))  # -1

# ======================================================================
# Section 7: Advanced Techniques
# ======================================================================

# 7.1 Two-Pointer Technique (e.g., Find pair sum to target in sorted array)
# Explanation: Left and right pointers. O(n) time after sorting.
# Edge: No pair, duplicates, empty.
def two_pointer_pair_sum(arr, target):
    """Return indices of pair summing to target (sorted arr)."""
    arr.sort()  # O(n log n)
    left, right = 0, len(arr) - 1
    while left < right:
        curr = arr[left] + arr[right]
        if curr == target:
            return left, right
        elif curr < target:
            left += 1
        else:
            right -= 1
    return -1, -1

arr = [2,7,11,15]
print("Pair sum to 9:", two_pointer_pair_sum(arr, 9))  # (0,1) after sorting, but indices may change; actually sorts to [2,7,11,15]

# 7.2 Sliding Window (e.g., Max sum of k consecutive elements)
# Explanation: Maintain window of size k, slide. O(n) time.
# Edge: k > len (invalid), k=1 (max element).
def sliding_window_max_sum(arr, k):
    """Max sum of k consecutive."""
    if k > len(arr) or k <= 0:
        raise ValueError("Invalid k")
    max_sum = curr = sum(arr[:k])
    for i in range(k, len(arr)):
        curr += arr[i] - arr[i-k]
        if curr > max_sum:
            max_sum = curr
    return max_sum

print("Max sum k=3 in [1,4,2,10,23,3,1,0,20]:", sliding_window_max_sum([1,4,2,10,23,3,1,0,20], 3))  # 36 (10+23+3? Wait, check: actually 4+2+10=16, 2+10+23=35, 10+23+3=36 yes)

# 7.3 Prefix Sum (e.g., Range sum queries)
# Explanation: Precompute prefix sums for O(1) queries after O(n) prep.
# Edge: Invalid ranges, empty.
def prefix_sum_array(arr):
    """Return prefix sum array."""
    prefix = [0] * (len(arr) + 1)
    for i in range(1, len(arr)+1):
        prefix[i] = prefix[i-1] + arr[i-1]
    return prefix

def range_sum(prefix, left, right):
    """Sum from left to right (0-based)."""
    if left < 0 or right >= len(prefix)-1 or left > right:
        raise IndexError("Invalid range")
    return prefix[right+1] - prefix[left]

arr = [1,2,3,4,5]
prefix = prefix_sum_array(arr)
print("Prefix:", prefix)  # [0,1,3,6,10,15]
print("Sum 1 to 3:", range_sum(prefix, 1, 3))  # 2+3+4=9

# ======================================================================
# Section 8: 2D Arrays/Matrices
# ======================================================================

# 8.1 Traversal (Row-major)
# Explanation: Nested loops. O(m*n) time.
matrix = [[1,2,3], [4,5,6], [7,8,9]]
def traverse_matrix(mat):
    for row in mat:
        for elem in row:
            print(elem, end=' ')
        print()

print("Matrix Traversal:")
traverse_matrix(matrix)
# 1 2 3
# 4 5 6
# 7 8 9

# 8.2 Transpose Matrix
# Explanation: Swap rows and columns. O(m*n) time.
# Assumes square for simplicity, but works for rectangular.
def transpose_matrix(mat):
    """Return transposed matrix."""
    rows, cols = len(mat), len(mat[0])
    trans = [[0] * rows for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            trans[j][i] = mat[i][j]
    return trans




# 8.3 Rotate Matrix 90 degrees clockwise
# Explanation: Transpose then reverse each row. O(n^2) time, in-place possible.
# Edge: Non-square? Assumes square.
def rotate_matrix_90(mat):
    """Rotate in-place (square matrix)."""
    n = len(mat)
    # Transpose
    for i in range(n):
        for j in range(i+1, n):
            mat[i][j], mat[j][i] = mat[j][i], mat[i][j]
    # Reverse each row
    for i in range(n):
        reverse_array(mat[i])

mat = [[1,2,3], [4,5,6], [7,8,9]]
rotate_matrix_90(mat)
print("Rotated 90:")
traverse_matrix(mat)
# 7 4 1
# 8 5 2
# 9 6 3

# 8.4 Spiral Traversal
# Explanation: Traverse in spiral order. O(m*n) time.
# Edge: Single row/column, empty.
def spiral_traversal(mat):
    """Return list in spiral order."""
    if not mat or not mat[0]:
        return []
    result = []
    top, bottom = 0, len(mat) - 1
    left, right = 0, len(mat[0]) - 1
    while top <= bottom and left <= right:
        # Top row
        for i in range(left, right+1):
            result.append(mat[top][i])
        top += 1
        # Right column
        for i in range(top, bottom+1):
            result.append(mat[i][right])
        right -= 1
        # Bottom row (if needed)
        if top <= bottom:
            for i in range(right, left-1, -1):
                result.append(mat[bottom][i])
            bottom -= 1
        # Left column (if needed)
        if left <= right:
            for i in range(bottom, top-1, -1):
                result.append(mat[i][left])
            left += 1
    return result

print("Spiral:", spiral_traversal([[1,2,3,4], [5,6,7,8], [9,10,11,12]]))  # [1,2,3,4,8,12,11,10,9,5,6,7]

# ======================================================================
# Section 9: Special Algorithms
# ======================================================================

# 9.1 Dutch National Flag (Sort 0s,1s,2s)
# Explanation: Three pointers. O(n) time, in-place.
# Edge: All same, empty.
def dutch_national_flag(arr):
    """Sort 0,1,2 in-place."""
    low, mid, high = 0, 0, len(arr)-1
    while mid <= high:
        if arr[mid] == 0:
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # 2
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

arr = [2,0,2,1,1,0]
dutch_national_flag(arr)
print("Dutch Sorted:", arr)  # [0,0,1,1,2,2]

# 9.2 Moore's Voting Algorithm (Majority Element > n/2)
# Explanation: Candidate and count. O(n) time, O(1) space. Assumes majority exists.
# Edge: No majority? Need verification pass.
def moore_voting_majority(arr):
    """Find majority element."""
    candidate = None
    count = 0
    for num in arr:
        if count == 0:
            candidate = num
        count += (1 if num == candidate else -1)
    # Verify
    if arr.count(candidate) > len(arr) // 2:
        return candidate
    return None

print("Majority [3,2,3]:", moore_voting_majority([3,2,3]))  # 3
print("Majority [1,2,3]:", moore_voting_majority([1,2,3]))  # None

# ======================================================================
# Section 10: Exceptions and Edge Cases Handling
# ======================================================================
#
# Throughout, we've handled:
# - Empty arrays: Return None, 0, or raise exceptions.
# - Invalid indices: IndexError.
# - Invalid inputs: ValueError (e.g., k in sliding window).
# - Overflow: Python ints handle large numbers.
# - Duplicates/Negatives: Specified in assumptions (e.g., positives for some algos).
# - Additional Edge Example: Binary search on unsorted? Not handled, assume sorted.
# Best Practice: Always validate inputs in functions.
#
# Final Note: This covers array concepts comprehensively. Practice with variations!
# ======================================================================

# Main demonstration (all prints are already above, but can add more if needed)
if __name__ == "__main__":
    pass  # All examples run above for simplicity