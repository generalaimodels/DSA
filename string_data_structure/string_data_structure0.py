# ======================================================================
# Comprehensive Guide to Strings in Data Structures and Algorithms
# ======================================================================
#
# Author: 
#
# This file serves as an end-to-end tutorial on Strings in Data Structures.
# We cover fundamental concepts, medium to advanced algorithms, with detailed
# explanations, edge cases, exceptions, time/space complexities, and multiple
# examples.
#
# Why Strings? Strings are sequences of characters, fundamental in DSA for
# problems like searching, pattern matching, and text processing. In Python,
# strings are immutable, hashable, and support O(1) access via indexing.
# Key properties:
# - Immutable: Cannot be changed in-place (e.g., s[0] = 'a' raises TypeError).
# - Sequence: Support slicing, concatenation (O(n) time due to copying).
# - Unicode support: Handles international characters.
#
# Exceptions to watch:
# - IndexError: Accessing out-of-bound indices.
# - TypeError: Modifying immutable strings or mismatched types in operations.
# - ValueError: In conversions or invalid operations (e.g., int('abc')).
#
# We structure this as follows:
# 1. Basic Operations
# 2. Medium Algorithms (e.g., Reversal, Anagram Check, Longest Common Prefix)
# 3. Advanced Algorithms (e.g., KMP, Rabin-Karp, LCS, Edit Distance, LPS, Z-Algorithm)
#
# Each section includes:
# - Detailed explanation in comments.
# - Clean, efficient code with docstrings.
# - Multiple examples (simple, edge cases, exceptions).
# - Time/Space complexity analysis.
#
# Run this file to see outputs from examples. Let's dive in!

# ======================================================================
# Section 1: Basic String Operations
# ======================================================================
#
# Basics are crucial for building advanced algos. We cover creation, access,
# concatenation, slicing, and common methods. These are O(1) to O(n) ops.

def basic_string_operations():
    """Demonstrates basic string operations with examples."""
    # Creation
    s1 = "Hello, World!"  # String literal
    s2 = str(123)         # Conversion from int
    print(f"Created strings: s1='{s1}', s2='{s2}'")
    
    # Access and Slicing (O(1) access, O(k) slicing for substring of length k)
    print(f"First char: {s1[0]}, Slice [7:12]: {s1[7:12]}")
    
    # Concatenation (O(n + m) time, creates new string)
    s3 = s1 + " " + s2
    print(f"Concatenated: '{s3}'")
    
    # Immutability demo
    try:
        s1[0] = 'h'  # This will raise TypeError
    except TypeError as e:
        print(f"Exception: {e} (Strings are immutable)")
    
    # Edge case: Empty string
    empty = ""
    print(f"Empty string length: {len(empty)}, Is empty? {not empty}")
    
    # Exception: IndexError
    try:
        print(empty[0])
    except IndexError as e:
        print(f"Exception: {e} (Out of bounds)")

# Example calls
print("=== Basic Operations Examples ===")
basic_string_operations()
print("\n")

# ======================================================================
# Section 2: Medium-Level String Algorithms
# ======================================================================

# Algorithm 2.1: String Reversal
# Explanation: Reverse a string. Naive way: Iterate from end to start.
# Time: O(n), Space: O(n). Python's slicing is efficient but creates new str.
# Edge cases: Empty string, single char, palindromes.
# Exceptions: Non-string input (handle via type check).

def reverse_string(s: str) -> str:
    """Reverses the input string.
    
    Args:
        s: Input string.
    
    Returns:
        Reversed string.
    
    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    return s[::-1]  # Efficient slicing, or use ''.join(reversed(s))

# Examples
print("=== String Reversal Examples ===")
print(reverse_string("abc"))  # 'cba'
print(reverse_string(""))     # '' (edge: empty)
print(reverse_string("a"))    # 'a' (edge: single char)
try:
    reverse_string(123)       # Exception demo
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 2.2: Check if Two Strings are Anagrams
# Explanation: Anagrams have same characters with same frequencies (ignore case/space?).
# Approach: Sort both and compare (O(n log n)), or use counters (O(n)).
# We use collections.Counter for efficiency.
# Time: O(n), Space: O(n) (for counters).
# Edge cases: Different lengths, empty, unicode chars.
# Exceptions: Non-string inputs.

from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    """Checks if two strings are anagrams.
    
    Args:
        s1, s2: Input strings.
    
    Returns:
        True if anagrams, False otherwise.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(s1, str) and isinstance(s2, str)):
        raise TypeError("Both inputs must be strings")
    return Counter(s1) == Counter(s2)

# Examples
print("=== Anagram Check Examples ===")
print(are_anagrams("listen", "silent"))  # True
print(are_anagrams("hello", "world"))    # False
print(are_anagrams("", ""))              # True (empty)
print(are_anagrams("a", "aa"))           # False (different lengths)
try:
    are_anagrams("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 2.3: Longest Common Prefix (LCP)
# Explanation: Find the longest prefix common to all strings in a list.
# Approach: Sort the list, compare first and last (they differ most).
# Time: O(n log n) for sort + O(m) comparison, where m is min length.
# Optimized: Vertical scanning (O(s) where s is sum of lengths).
# Edge cases: Empty list, single string, no common prefix.
# Exceptions: Non-list or non-string elements.

def longest_common_prefix(strs: list[str]) -> str:
    """Finds the longest common prefix in a list of strings.
    
    Args:
        strs: List of strings.
    
    Returns:
        Longest common prefix.
    
    Raises:
        ValueError: If list is empty or contains non-strings.
    """
    if not strs:
        raise ValueError("Input list cannot be empty")
    if not all(isinstance(s, str) for s in strs):
        raise ValueError("All elements must be strings")
    
    # Vertical scanning
    min_len = min(len(s) for s in strs)
    for i in range(min_len):
        char = strs[0][i]
        if not all(s[i] == char for s in strs):
            return strs[0][:i]
    return strs[0][:min_len]

# Examples
print("=== Longest Common Prefix Examples ===")
print(longest_common_prefix(["flower", "flow", "flight"]))  # 'fl'
print(longest_common_prefix(["dog", "racecar", "car"]))     # '' (no common)
print(longest_common_prefix(["a"]))                         # 'a' (single)
# print(longest_common_prefix([]))  # Raises ValueError
try:
    longest_common_prefix([])
except ValueError as e:
    print(f"Exception: {e}")
print("\n")

# ======================================================================
# Section 3: Advanced String Algorithms
# ======================================================================

# Algorithm 3.1: Naive String Matching
# Explanation: Find all occurrences of pattern in text.
# Approach: Slide pattern over text, check char by char.
# Time: O((n-m+1)*m) worst case (e.g., text='aaa...a', pat='aa...a').
# Space: O(1).
# Edge cases: Pattern longer than text, multiple overlaps, empty.
# Exceptions: Non-string inputs.

def naive_string_match(text: str, pattern: str) -> list[int]:
    """Naive algorithm to find pattern in text.
    
    Args:
        text: The main string.
        pattern: The substring to find.
    
    Returns:
        List of starting indices where pattern matches.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(text, str) and isinstance(pattern, str)):
        raise TypeError("Both inputs must be strings")
    matches = []
    n, m = len(text), len(pattern)
    if m == 0 or n < m:
        return matches
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            matches.append(i)
    return matches

# Examples
print("=== Naive String Matching Examples ===")
print(naive_string_match("abababc", "ab"))  # [0, 2, 4]
print(naive_string_match("aaaa", "aa"))     # [0, 1, 2] (overlaps)
print(naive_string_match("abc", "def"))     # [] (no match)
print(naive_string_match("a", ""))          # [] (empty pattern edge)
try:
    naive_string_match("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.2: Knuth-Morris-Pratt (KMP) Algorithm
# Explanation: Efficient string matching using prefix table (LPS array).
# Avoids rechecking matched prefixes on mismatch.
# Time: O(n + m) preprocessing + matching.
# Space: O(m) for LPS.
# Edge cases: All matches, no matches, repeated patterns.
# Exceptions: Similar to above.

def compute_lps(pattern: str) -> list[int]:
    """Computes Longest Prefix Suffix (LPS) array for KMP."""
    m = len(pattern)
    lps = [0] * m
    length = 0  # Length of previous longest prefix suffix
    i = 1
    while i < m:
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

def kmp_string_match(text: str, pattern: str) -> list[int]:
    """KMP algorithm to find pattern in text.
    
    Args:
        text: The main string.
        pattern: The substring to find.
    
    Returns:
        List of starting indices.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(text, str) and isinstance(pattern, str)):
        raise TypeError("Both inputs must be strings")
    matches = []
    n, m = len(text), len(pattern)
    if m == 0 or n < m:
        return matches
    lps = compute_lps(pattern)
    i = j = 0  # i for text, j for pattern
    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1
        if j == m:
            matches.append(i - j)
            j = lps[j - 1]
        elif i < n and pattern[j] != text[i]:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return matches

# Examples
print("=== KMP String Matching Examples ===")
print(kmp_string_match("abababc", "ab"))  # [0, 2, 4]
print(kmp_string_match("aaaa", "aa"))     # [0, 1, 2]
print(kmp_string_match("abc", "def"))     # []
print(kmp_string_match("aaaaa", "aaa"))   # [0, 1, 2] (repetitive)
try:
    kmp_string_match("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.3: Rabin-Karp Algorithm (Hash-based Matching)
# Explanation: Uses rolling hash to find pattern. Hash text windows and compare.
# Handles spurious hits by full comparison.
# Time: Average O(n + m), Worst O(nm) (many hash collisions).
# Space: O(1).
# We use a simple hash (sum of ord) for demo; real uses polynomial rolling hash.
# Edge cases: Hash collisions, large strings.
# Exceptions: Non-strings, overflow (Python ints are arbitrary precision).

def rabin_karp_match(text: str, pattern: str, prime: int = 101) -> list[int]:
    """Rabin-Karp algorithm for string matching.
    
    Args:
        text: Main string.
        pattern: Pattern to find.
        prime: A prime for hashing (reduces collisions).
    
    Returns:
        List of starting indices.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(text, str) and isinstance(pattern, str)):
        raise TypeError("Both inputs must be strings")
    matches = []
    n, m = len(text), len(pattern)
    if m == 0 or n < m:
        return matches
    
    # Constants for rolling hash
    d = 256  # Number of characters in input alphabet
    
    # Precompute pattern hash and first window hash
    pat_hash = 0
    txt_hash = 0
    h = 1  # h = d^(m-1) % prime
    for i in range(m - 1):
        h = (h * d) % prime
    for i in range(m):
        pat_hash = (d * pat_hash + ord(pattern[i])) % prime
        txt_hash = (d * txt_hash + ord(text[i])) % prime
    
    # Slide the window
    for i in range(n - m + 1):
        if pat_hash == txt_hash:
            # Check for spurious hit
            if text[i:i+m] == pattern:
                matches.append(i)
        if i < n - m:
            txt_hash = (d * (txt_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
            if txt_hash < 0:
                txt_hash += prime  # Handle negative modulo
    return matches

# Examples
print("=== Rabin-Karp Matching Examples ===")
print(rabin_karp_match("abababc", "ab"))  # [0, 2, 4]
print(rabin_karp_match("aaaa", "aa"))     # [0, 1, 2]
print(rabin_karp_match("abc", "def"))     # []
print(rabin_karp_match("GEEKS FOR GEEKS", "GEEK"))  # [0, 10] (with spaces)
try:
    rabin_karp_match("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.4: Longest Common Subsequence (LCS)
# Explanation: Find longest subsequence common to two strings (not necessarily contiguous).
# DP Approach: 2D table, dp[i][j] = length of LCS of s1[0..i-1] and s2[0..j-1].
# Time: O(n*m), Space: O(n*m) (can optimize to O(min(n,m))).
# Edge cases: Empty strings, identical, no common.
# Exceptions: Non-strings.

def longest_common_subsequence(s1: str, s2: str) -> int:
    """Computes length of LCS using DP.
    
    Args:
        s1, s2: Input strings.
    
    Returns:
        Length of LCS.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(s1, str) and isinstance(s2, str)):
        raise TypeError("Both inputs must be strings")
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[n][m]

# Examples
print("=== LCS Examples ===")
print(longest_common_subsequence("abcde", "ace"))  # 3 ('ace')
print(longest_common_subsequence("abc", "def"))    # 0
print(longest_common_subsequence("", "abc"))       # 0 (empty)
print(longest_common_subsequence("abcd", "abcd"))  # 4 (identical)
try:
    longest_common_subsequence("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.5: Edit Distance (Levenshtein Distance)
# Explanation: Minimum operations (insert, delete, replace) to convert s1 to s2.
# DP: Similar to LCS, but tracks operations.
# Time: O(n*m), Space: O(n*m).
# Edge cases: Empty, one empty, equal strings.
# Exceptions: Non-strings.

def edit_distance(s1: str, s2: str) -> int:
    """Computes minimum edit distance using DP.
    
    Args:
        s1, s2: Input strings.
    
    Returns:
        Minimum operations needed.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(s1, str) and isinstance(s2, str)):
        raise TypeError("Both inputs must be strings")
    n, m = len(s1), len(s2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i  # Delete all from s1 
    for j in range(m + 1):
        dp[0][j] = j  # Insert all to s2
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]  # No op
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],    # Delete
                    dp[i][j - 1],    # Insert
                    dp[i - 1][j - 1] # Replace
                )
    return dp[n][m]

# Examples
print("=== Edit Distance Examples ===")
print(edit_distance("kitten", "sitting"))  # 3 (k->s, e->i, add g)
print(edit_distance("abc", "abc"))         # 0 (equal)
print(edit_distance("", "abc"))            # 3 (insert all)
print(edit_distance("a", ""))              # 1 (delete)
try:
    edit_distance("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.6: Longest Palindromic Substring (LPS)
# Explanation: Find longest substring that is a palindrome.
# Approach: Expand around centers (odd/even lengths).
# Time: O(n^2), Space: O(1).
# DP alternative: O(n^2) time/space.
# Edge cases: All same chars, single char, empty.
# Exceptions: Non-string.

def longest_palindromic_substring(s: str) -> str:
    """Finds the longest palindromic substring by expanding around centers.
    
    Args:
        s: Input string.
    
    Returns:
        The longest palindromic substring.
    
    Raises:
        TypeError: If input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string")
    n = len(s)
    if n == 0:
        return ""
    start = end = 0
    for i in range(n):
        # Odd length palindrome
        left, right = i, i
        while left >= 0 and right < n and s[left] == s[right]:
            if right - left > end - start:
                start, end = left, right
            left -= 1
            right += 1
        # Even length palindrome
        left, right = i, i + 1
        while left >= 0 and right < n and s[left] == s[right]:
            if right - left > end - start:
                start, end = left, right
            left -= 1
            right += 1
    return s[start:end + 1]

# Examples
print("=== Longest Palindromic Substring Examples ===")
print(longest_palindromic_substring("babad"))  # 'bab' or 'aba'
print(longest_palindromic_substring("cbbd"))   # 'bb'
print(longest_palindromic_substring("a"))      # 'a'
print(longest_palindromic_substring(""))       # ''
print(longest_palindromic_substring("aaaa"))   # 'aaaa' (all same)
try:
    longest_palindromic_substring(123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# Algorithm 3.7: Z-Algorithm for Pattern Searching
# Explanation: Computes Z-array where z[i] = length of longest substring starting at i matching prefix.
# Used for efficient matching.
# Time: O(n + m), Space: O(n + m).
# Edge cases: Similar to KMP.
# Exceptions: Non-strings.

def compute_z_array(s: str) -> list[int]:
    """Computes Z-array for the given string."""
    n = len(s)
    z = [0] * n
    left = right = 0
    for i in range(1, n):
        if i < right:
            z[i] = min(right - i, z[i - left])
        while i + z[i] < n and s[z[i]] == s[i + z[i]]:
            z[i] += 1
        if i + z[i] > right:
            left = i
            right = i + z[i]
    return z

def z_algorithm_match(text: str, pattern: str) -> list[int]:
    """Z-algorithm to find pattern in text.
    
    Args:
        text: Main string.
        pattern: Pattern.
    
    Returns:
        List of starting indices.
    
    Raises:
        TypeError: If inputs are not strings.
    """
    if not (isinstance(text, str) and isinstance(pattern, str)):
        raise TypeError("Both inputs must be strings")
    concat = pattern + "$" + text  # Separator to avoid overlap
    z = compute_z_array(concat)
    matches = []
    m = len(pattern)
    for i in range(m + 1, len(concat)):
        if z[i] == m:
            matches.append(i - m - 1)
    return matches

# Examples
print("=== Z-Algorithm Matching Examples ===")
print(z_algorithm_match("abababc", "ab"))  # [0, 2, 4]
print(z_algorithm_match("aaaa", "aa"))     # [0, 1, 2]
print(z_algorithm_match("abc", "def"))     # []
print(z_algorithm_match("aaabaaa", "aaa")) # [0, 4] (overlaps)
try:
    z_algorithm_match("abc", 123)
except TypeError as e:
    print(f"Exception: {e}")
print("\n")

# ======================================================================
# End of Tutorial
# ======================================================================
# This covers a comprehensive set of string algorithms from medium to advanced.
# Practice with variations: e.g., case-insensitive anagrams, multi-pattern search.
# Remember: Strings are immutable—always consider copying costs in algos.
# For production, profile for large inputs (e.g., use str built-ins where possible).

