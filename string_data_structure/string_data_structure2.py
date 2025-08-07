# ==========================================================================================================
# String in Data Structure: Medium to Advanced Algorithms
# ==========================================================================================================
# Author:
# This file is a comprehensive, technical, and highly detailed guide to string algorithms in data structures.
# All code is written in Python, with best practices, clear style, and in-depth explanations.
# Each algorithm includes:
#   - Problem description
#   - Algorithmic approach
#   - Time and space complexity
#   - Edge cases and exceptions
#   - Multiple examples for clarity
# ==========================================================================================================

# ----------------------------------------------------------------------------------------------------------
# 1. String Reversal
# ----------------------------------------------------------------------------------------------------------
def reverse_string(s: str) -> str:
    """
    Reverses a string.
    Time: O(n), Space: O(n)
    Edge Cases: Empty string, single character, unicode.
    Exception: TypeError if input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")
    return s[::-1]

# Examples
print("Reverse String Examples:")
print(reverse_string("hello"))         # 'olleh'
print(reverse_string(""))              # ''
print(reverse_string("a"))             # 'a'
print(reverse_string("😀👍"))           # '👍😀'
try:
    reverse_string(123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 2. Check for Anagram
# ----------------------------------------------------------------------------------------------------------
from collections import Counter

def are_anagrams(s1: str, s2: str) -> bool:
    """
    Checks if two strings are anagrams.
    Time: O(n), Space: O(n)
    Edge Cases: Empty strings, different lengths, unicode.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Both inputs must be strings.")
    return Counter(s1) == Counter(s2)

# Examples
print("Anagram Check Examples:")
print(are_anagrams("listen", "silent"))    # True
print(are_anagrams("abc", "cab"))          # True
print(are_anagrams("abc", "abcc"))         # False
print(are_anagrams("", ""))                # True
print(are_anagrams("😀👍", "👍😀"))           # True
try:
    are_anagrams("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 3. Longest Common Prefix
# ----------------------------------------------------------------------------------------------------------
def longest_common_prefix(strs: list[str]) -> str:
    """
    Finds the longest common prefix among a list of strings.
    Time: O(S), S = sum of all characters in all strings.
    Edge Cases: Empty list, single string, no common prefix.
    Exception: ValueError if input is not a list of strings.
    """
    if not isinstance(strs, list) or not all(isinstance(s, str) for s in strs):
        raise ValueError("Input must be a list of strings.")
    if not strs:
        return ""
    prefix = strs[0]
    for s in strs[1:]:
        i = 0
        while i < len(prefix) and i < len(s) and prefix[i] == s[i]:
            i += 1
        prefix = prefix[:i]
        if not prefix:
            break
    return prefix

# Examples
print("Longest Common Prefix Examples:")
print(longest_common_prefix(["flower", "flow", "flight"]))  # 'fl'
print(longest_common_prefix(["dog", "racecar", "car"]))     # ''
print(longest_common_prefix(["interview"]))                 # 'interview'
print(longest_common_prefix([]))                            # ''
try:
    longest_common_prefix("notalist")
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 4. Naive Pattern Search
# ----------------------------------------------------------------------------------------------------------
def naive_pattern_search(text: str, pattern: str) -> list[int]:
    """
    Finds all occurrences of pattern in text using naive search.
    Time: O((n-m+1)*m), Space: O(1)
    Edge Cases: Empty pattern, pattern longer than text, overlapping matches.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Both inputs must be strings.")
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    result = []
    for i in range(n - m + 1):
        if text[i:i+m] == pattern:
            result.append(i)
    return result

# Examples
print("Naive Pattern Search Examples:")
print(naive_pattern_search("ababcabcab", "abc"))   # [2, 5]
print(naive_pattern_search("aaaaa", "aa"))         # [0, 1, 2, 3]
print(naive_pattern_search("abc", ""))             # []
print(naive_pattern_search("abc", "abcd"))         # []
try:
    naive_pattern_search("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 5. Knuth-Morris-Pratt (KMP) Pattern Search
# ----------------------------------------------------------------------------------------------------------
def compute_lps(pattern: str) -> list[int]:
    """
    Computes the Longest Prefix Suffix (LPS) array for KMP.
    """
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
                length = lps[length-1]
            else:
                lps[i] = 0
                i += 1
    return lps

def kmp_search(text: str, pattern: str) -> list[int]:
    """
    Finds all occurrences of pattern in text using KMP algorithm.
    Time: O(n + m), Space: O(m)
    Edge Cases: Empty pattern, pattern longer than text, repeated patterns.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Both inputs must be strings.")
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    lps = compute_lps(pattern)
    result = []
    i = j = 0
    while i < n:
        if text[i] == pattern[j]:
            i += 1
            j += 1
        if j == m:
            result.append(i - j)
            j = lps[j-1]
        elif i < n and text[i] != pattern[j]:
            if j != 0:
                j = lps[j-1]
            else:
                i += 1
    return result

# Examples
print("KMP Pattern Search Examples:")
print(kmp_search("ababcabcab", "abc"))   # [2, 5]
print(kmp_search("aaaaa", "aa"))         # [0, 1, 2, 3]
print(kmp_search("abc", ""))             # []
print(kmp_search("abc", "abcd"))         # []
try:
    kmp_search("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 6. Rabin-Karp Pattern Search
# ----------------------------------------------------------------------------------------------------------
def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> list[int]:
    """
    Rabin-Karp algorithm for pattern searching using rolling hash.
    Time: O(n + m), Space: O(1)
    Edge Cases: Hash collisions, empty pattern, pattern longer than text.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Both inputs must be strings.")
    n, m = len(text), len(pattern)
    if m == 0 or m > n:
        return []
    d = 256
    h = pow(d, m-1, prime)
    p = t = 0
    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime
    result = []
    for i in range(n - m + 1):
        if p == t:
            if text[i:i+m] == pattern:
                result.append(i)
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % prime
            if t < 0:
                t += prime
    return result

# Examples
print("Rabin-Karp Pattern Search Examples:")
print(rabin_karp_search("ababcabcab", "abc"))   # [2, 5]
print(rabin_karp_search("aaaaa", "aa"))         # [0, 1, 2, 3]
print(rabin_karp_search("abc", ""))             # []
print(rabin_karp_search("abc", "abcd"))         # []
try:
    rabin_karp_search("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 7. Longest Common Subsequence (LCS)
# ----------------------------------------------------------------------------------------------------------
def longest_common_subsequence(s1: str, s2: str) -> int:
    """
    Returns the length of the longest common subsequence between s1 and s2.
    Time: O(n*m), Space: O(n*m)
    Edge Cases: Empty strings, no common subsequence, identical strings.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Both inputs must be strings.")
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if s1[i] == s2[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[n][m]

# Examples
print("Longest Common Subsequence Examples:")
print(longest_common_subsequence("abcde", "ace"))    # 3
print(longest_common_subsequence("abc", "abc"))      # 3
print(longest_common_subsequence("abc", "def"))      # 0
print(longest_common_subsequence("", "abc"))         # 0
try:
    longest_common_subsequence("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 8. Edit Distance (Levenshtein Distance)
# ----------------------------------------------------------------------------------------------------------
def edit_distance(s1: str, s2: str) -> int:
    """
    Computes the minimum edit distance between two strings.
    Time: O(n*m), Space: O(n*m)
    Edge Cases: Empty strings, identical strings.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(s1, str) or not isinstance(s2, str):
        raise TypeError("Both inputs must be strings.")
    n, m = len(s1), len(s2)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1):
        dp[i][0] = i
    for j in range(m+1):
        dp[0][j] = j
    for i in range(1, n+1):
        for j in range(1, m+1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[n][m]

# Examples
print("Edit Distance Examples:")
print(edit_distance("kitten", "sitting"))   # 3
print(edit_distance("flaw", "lawn"))        # 2
print(edit_distance("abc", "abc"))          # 0
print(edit_distance("", "abc"))             # 3
try:
    edit_distance("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 9. Longest Palindromic Substring
# ----------------------------------------------------------------------------------------------------------
def longest_palindromic_substring(s: str) -> str:
    """
    Finds the longest palindromic substring in s.
    Time: O(n^2), Space: O(1)
    Edge Cases: Empty string, all same characters, single character.
    Exception: TypeError if input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")
    n = len(s)
    if n == 0:
        return ""
    start = end = 0
    for i in range(n):
        # Odd length
        l, r = i, i
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l > end - start:
                start, end = l, r
            l -= 1
            r += 1
        # Even length
        l, r = i, i+1
        while l >= 0 and r < n and s[l] == s[r]:
            if r - l > end - start:
                start, end = l, r
            l -= 1
            r += 1
    return s[start:end+1]

# Examples
print("Longest Palindromic Substring Examples:")
print(longest_palindromic_substring("babad"))    # 'bab' or 'aba'
print(longest_palindromic_substring("cbbd"))     # 'bb'
print(longest_palindromic_substring("a"))        # 'a'
print(longest_palindromic_substring(""))         # ''
print(longest_palindromic_substring("aaaa"))     # 'aaaa'
try:
    longest_palindromic_substring(123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 10. Z-Algorithm for Pattern Search
# ----------------------------------------------------------------------------------------------------------
def compute_z_array(s: str) -> list[int]:
    """
    Computes the Z-array for string s.
    Z[i] = length of the longest substring starting at i which is also a prefix of s.
    """
    n = len(s)
    Z = [0] * n
    l = r = 0
    for i in range(1, n):
        if i <= r:
            Z[i] = min(r - i + 1, Z[i - l])
        while i + Z[i] < n and s[Z[i]] == s[i + Z[i]]:
            Z[i] += 1
        if i + Z[i] - 1 > r:
            l, r = i, i + Z[i] - 1
    return Z

def z_algorithm_search(text: str, pattern: str) -> list[int]:
    """
    Finds all occurrences of pattern in text using Z-algorithm.
    Time: O(n + m), Space: O(n + m)
    Edge Cases: Empty pattern, pattern longer than text.
    Exception: TypeError if inputs are not strings.
    """
    if not isinstance(text, str) or not isinstance(pattern, str):
        raise TypeError("Both inputs must be strings.")
    concat = pattern + "$" + text
    Z = compute_z_array(concat)
    m = len(pattern)
    result = []
    for i in range(m+1, len(concat)):
        if Z[i] == m:
            result.append(i - m - 1)
    return result

# Examples
print("Z-Algorithm Pattern Search Examples:")
print(z_algorithm_search("ababcabcab", "abc"))   # [2, 5]
print(z_algorithm_search("aaaaa", "aa"))         # [0, 1, 2, 3]
print(z_algorithm_search("abc", ""))             # []
print(z_algorithm_search("abc", "abcd"))         # []
try:
    z_algorithm_search("abc", 123)
except Exception as e:
    print("Exception:", e)
print()


# ----------------------------------------------------------------------------------------------------------
# 11. Manacher's Algorithm (Longest Palindromic Substring in O(n))
# ----------------------------------------------------------------------------------------------------------
def manacher_longest_palindrome(s: str) -> str:
    """
    Manacher's algorithm to find the longest palindromic substring in O(n) time.
    Time: O(n), Space: O(n)
    Edge Cases: Empty string, all same characters.
    Exception: TypeError if input is not a string.
    """
    if not isinstance(s, str):
        raise TypeError("Input must be a string.")
    if not s:
        return ""
    t = '#' + '#'.join(s) + '#'
    n = len(t)
    p = [0] * n
    c = r = 0
    max_len = 0
    center = 0
    for i in range(n):
        mirror = 2 * c - i
        if i < r:
            p[i] = min(r - i, p[mirror])
        a = i + p[i] + 1
        b = i - p[i] - 1
        while a < n and b >= 0 and t[a] == t[b]:
            p[i] += 1
            a += 1
            b -= 1
        if i + p[i] > r:
            c, r = i, i + p[i]
        if p[i] > max_len:
            max_len = p[i]
            center = i
    start = (center - max_len) // 2
    return s[start:start + max_len]

# Examples
print("Manacher's Algorithm Examples:")
print(manacher_longest_palindrome("babad"))    # 'bab' or 'aba'
print(manacher_longest_palindrome("cbbd"))     # 'bb'
print(manacher_longest_palindrome("a"))        # 'a'
print(manacher_longest_palindrome(""))         # ''
print(manacher_longest_palindrome("aaaa"))     # 'aaaa'
try:
    manacher_longest_palindrome(123)
except Exception as e:
    print("Exception:", e)
print()


# ==========================================================================================================
# End of File: All major medium to advanced string algorithms are covered with best practices and clarity.
# ==========================================================================================================