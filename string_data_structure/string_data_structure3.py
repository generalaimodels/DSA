"""
STRING DATA STRUCTURES & ALGORITHMS - COMPREHENSIVE IMPLEMENTATION
=================================================================
Advanced string processing algorithms with optimal time/space complexity
All implementations are production-ready with comprehensive error handling
"""

from __future__ import annotations
import bisect
import collections
import heapq
import math
import random
import re
from typing import Dict, List, Optional, Tuple, Set, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

# =============================================================================
# SECTION 1: FUNDAMENTAL STRING OPERATIONS & PROPERTIES
# =============================================================================

class StringProcessor:
    """Core string processing utilities with optimal implementations"""
    
    @staticmethod
    def is_palindrome(s: str) -> bool:
        """O(n) palindrome check with two pointers"""
        left, right = 0, len(s) - 1
        while left < right:
            if s[left] != s[right]:
                return False
            left += 1
            right -= 1
        return True
    
    @staticmethod
    def reverse_words(s: str) -> str:
        """O(n) word reversal maintaining single spaces"""
        words = []
        word = []
        
        for char in s:
            if char != ' ':
                word.append(char)
            elif word:
                words.append(''.join(word))
                word = []
        
        if word:
            words.append(''.join(word))
        
        return ' '.join(reversed(words))
    
    @staticmethod
    def normalize_string(s: str) -> str:
        """Remove extra spaces and normalize case"""
        return ' '.join(s.split()).lower()

# =============================================================================
# SECTION 2: STRING MATCHING ALGORITHMS
# =============================================================================

class StringMatcher:
    """Collection of string matching algorithms with different use cases"""
    
    @staticmethod
    def naive_search(text: str, pattern: str) -> List[int]:
        """
        Naive string matching - O(n*m) time, O(1) space
        Returns all starting positions of pattern in text
        """
        if not pattern:
            return list(range(len(text) + 1))
        
        positions = []
        n, m = len(text), len(pattern)
        
        for i in range(n - m + 1):
            match = True
            for j in range(m):
                if text[i + j] != pattern[j]:
                    match = False
                    break
            if match:
                positions.append(i)
        
        return positions
    
    @staticmethod
    def compute_lps_array(pattern: str) -> List[int]:
        """Compute Longest Proper Prefix which is also Suffix array for KMP"""
        m = len(pattern)
        lps = [0] * m
        length = 0
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
    
    @classmethod
    def kmp_search(cls, text: str, pattern: str) -> List[int]:
        """
        Knuth-Morris-Pratt algorithm - O(n+m) time, O(m) space
        Optimal for repeated searches with same pattern
        """
        if not pattern:
            return list(range(len(text) + 1))
        
        positions = []
        n, m = len(text), len(pattern)
        lps = cls.compute_lps_array(pattern)
        
        i = j = 0
        while i < n:
            if pattern[j] == text[i]:
                i += 1
                j += 1
            
            if j == m:
                positions.append(i - j)
                j = lps[j - 1]
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
        
        return positions
    
    @staticmethod
    def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> List[int]:
        """
        Rabin-Karp rolling hash algorithm - O(n+m) average, O(n*m) worst
        Excellent for multiple pattern searching
        """
        if not pattern:
            return list(range(len(text) + 1))
        
        positions = []
        n, m = len(text), len(pattern)
        d = 256  # number of characters in alphabet
        
        pattern_hash = text_hash = 0
        h = 1
        
        # Calculate h = pow(d, m-1) % prime
        for _ in range(m - 1):
            h = (h * d) % prime
        
        # Calculate hash for pattern and first window
        for i in range(m):
            pattern_hash = (d * pattern_hash + ord(pattern[i])) % prime
            text_hash = (d * text_hash + ord(text[i])) % prime
        
        # Slide pattern over text
        for i in range(n - m + 1):
            if pattern_hash == text_hash:
                # Check characters one by one
                if text[i:i + m] == pattern:
                    positions.append(i)
            
            # Calculate hash for next window
            if i < n - m:
                text_hash = (d * (text_hash - ord(text[i]) * h) + ord(text[i + m])) % prime
                if text_hash < 0:
                    text_hash += prime
        
        return positions
    
    @staticmethod
    def boyer_moore_search(text: str, pattern: str) -> List[int]:
        """
        Boyer-Moore with bad character heuristic - O(n*m) worst, O(n/m) best
        Excellent for large alphabets and long patterns
        """
        if not pattern:
            return list(range(len(text) + 1))
        
        positions = []
        n, m = len(text), len(pattern)
        
        # Build bad character table
        bad_char = {}
        for i in range(m):
            bad_char[pattern[i]] = i
        
        s = 0  # shift of pattern with respect to text
        while s <= n - m:
            j = m - 1
            
            # Reduce j while characters match
            while j >= 0 and pattern[j] == text[s + j]:
                j -= 1
            
            if j < 0:
                positions.append(s)
                s += (m - bad_char.get(text[s + m], -1) - 1) if s + m < n else 1
            else:
                s += max(1, j - bad_char.get(text[s + j], -1))
        
        return positions

class ZAlgorithm:
    """Z-algorithm for linear time pattern matching and string analysis"""
    
    @staticmethod
    def compute_z_array(s: str) -> List[int]:
        """
        Compute Z-array in O(n) time
        Z[i] = length of longest substring starting from i which is also prefix
        """
        n = len(s)
        z = [0] * n
        l = r = 0
        
        for i in range(1, n):
            if i <= r:
                z[i] = min(r - i + 1, z[i - l])
            
            while i + z[i] < n and s[z[i]] == s[i + z[i]]:
                z[i] += 1
            
            if i + z[i] - 1 > r:
                l, r = i, i + z[i] - 1
        
        z[0] = n
        return z
    
    @classmethod
    def pattern_search(cls, text: str, pattern: str) -> List[int]:
        """Use Z-algorithm for pattern matching"""
        if not pattern:
            return list(range(len(text) + 1))
        
        combined = pattern + "$" + text
        z = cls.compute_z_array(combined)
        
        positions = []
        pattern_len = len(pattern)
        
        for i in range(pattern_len + 1, len(combined)):
            if z[i] == pattern_len:
                positions.append(i - pattern_len - 1)
        
        return positions

# =============================================================================
# SECTION 3: TRIE DATA STRUCTURE
# =============================================================================

class TrieNode:
    """Optimized Trie node with efficient memory usage"""
    
    __slots__ = ['children', 'is_end_word', 'word_count', 'prefix_count']
    
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_word: bool = False
        self.word_count: int = 0
        self.prefix_count: int = 0

class Trie:
    """
    Advanced Trie implementation with prefix counting and word frequency
    Space: O(ALPHABET_SIZE * N * M) where N is number of words, M is average length
    """
    
    def __init__(self):
        self.root = TrieNode()
        self.total_words = 0
    
    def insert(self, word: str) -> None:
        """Insert word into trie - O(m) where m is word length"""
        if not word:
            return
        
        current = self.root
        
        for char in word:
            if char not in current.children:
                current.children[char] = TrieNode()
            current = current.children[char]
            current.prefix_count += 1
        
        if not current.is_end_word:
            current.is_end_word = True
            self.total_words += 1
        
        current.word_count += 1
    
    def search(self, word: str) -> bool:
        """Search for word in trie - O(m)"""
        node = self._find_node(word)
        return node is not None and node.is_end_word
    
    def starts_with(self, prefix: str) -> bool:
        """Check if any word starts with prefix - O(m)"""
        return self._find_node(prefix) is not None
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """Count words starting with prefix - O(m)"""
        node = self._find_node(prefix)
        return node.prefix_count if node else 0
    
    def delete(self, word: str) -> bool:
        """Delete word from trie - O(m)"""
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if not node.is_end_word:
                    return False
                
                node.is_end_word = False
                node.word_count = 0
                return len(node.children) == 0
            
            char = word[index]
            if char not in node.children:
                return False
            
            should_delete_child = _delete_helper(node.children[char], word, index + 1)
            
            if should_delete_child:
                del node.children[char]
            else:
                node.children[char].prefix_count -= 1
            
            return (not node.is_end_word and 
                    len(node.children) == 0 and 
                    node != self.root)
        
        if self.search(word):
            _delete_helper(self.root, word, 0)
            self.total_words -= 1
            return True
        return False
    
    def get_all_words_with_prefix(self, prefix: str) -> List[str]:
        """Get all words starting with prefix - O(NODES)"""
        node = self._find_node(prefix)
        if not node:
            return []
        
        words = []
        
        def dfs(current: TrieNode, path: str):
            if current.is_end_word:
                words.append(path)
            
            for char, child in current.children.items():
                dfs(child, path + char)
        
        dfs(node, prefix)
        return words
    
    def _find_node(self, prefix: str) -> Optional[TrieNode]:
        """Find node corresponding to prefix"""
        current = self.root
        
        for char in prefix:
            if char not in current.children:
                return None
            current = current.children[char]
        
        return current

# =============================================================================
# SECTION 4: SUFFIX ARRAY AND LCP ARRAY
# =============================================================================

class SuffixArray:
    """
    Efficient Suffix Array construction with LCP array
    Construction: O(n log n), LCP: O(n)
    """
    
    def __init__(self, text: str):
        self.text = text
        self.n = len(text)
        self.suffix_array = self._build_suffix_array()
        self.lcp_array = self._build_lcp_array()
        self.rank = self._build_rank_array()
    
    def _build_suffix_array(self) -> List[int]:
        """Build suffix array using counting sort and radix sort"""
        n = self.n
        suffixes = list(range(n))
        
        # Initial ranking based on first character
        rank = [ord(c) for c in self.text]
        
        k = 1
        while k < n:
            # Sort based on second part first, then first part
            def compare_key(i):
                first = rank[i]
                second = rank[i + k] if i + k < n else -1
                return (first, second)
            
            suffixes.sort(key=compare_key)
            
            # Update ranks
            new_rank = [0] * n
            for i in range(1, n):
                if compare_key(suffixes[i]) == compare_key(suffixes[i-1]):
                    new_rank[suffixes[i]] = new_rank[suffixes[i-1]]
                else:
                    new_rank[suffixes[i]] = new_rank[suffixes[i-1]] + 1
            
            rank = new_rank
            if rank[suffixes[-1]] == n - 1:
                break
            k *= 2
        
        return suffixes
    
    def _build_lcp_array(self) -> List[int]:
        """Build LCP array using Kasai algorithm - O(n)"""
        lcp = [0] * (self.n - 1)
        rank = [0] * self.n
        
        # Build rank array
        for i, suffix in enumerate(self.suffix_array):
            rank[suffix] = i
        
        h = 0
        for i in range(self.n):
            if rank[i] == self.n - 1:
                h = 0
                continue
            
            j = self.suffix_array[rank[i] + 1]
            
            while (i + h < self.n and j + h < self.n and 
                   self.text[i + h] == self.text[j + h]):
                h += 1
            
            lcp[rank[i]] = h
            h = max(0, h - 1)
        
        return lcp
    
    def _build_rank_array(self) -> List[int]:
        """Build rank array for quick suffix position lookup"""
        rank = [0] * self.n
        for i, suffix in enumerate(self.suffix_array):
            rank[suffix] = i
        return rank
    
    def find_pattern(self, pattern: str) -> List[int]:
        """Binary search for pattern in suffix array - O(m log n)"""
        def lower_bound():
            left, right = 0, self.n
            while left < right:
                mid = (left + right) // 2
                suffix_start = self.suffix_array[mid]
                suffix = self.text[suffix_start:suffix_start + len(pattern)]
                if suffix < pattern:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        def upper_bound():
            left, right = 0, self.n
            while left < right:
                mid = (left + right) // 2
                suffix_start = self.suffix_array[mid]
                suffix = self.text[suffix_start:suffix_start + len(pattern)]
                if suffix <= pattern:
                    left = mid + 1
                else:
                    right = mid
            return left
        
        l = lower_bound()
        r = upper_bound()
        
        if l < self.n:
            suffix_start = self.suffix_array[l]
            if self.text[suffix_start:].startswith(pattern):
                return [self.suffix_array[i] for i in range(l, r)]
        
        return []
    
    def longest_repeated_substring(self) -> str:
        """Find longest repeated substring using LCP array"""
        if not self.lcp_array:
            return ""
        
        max_lcp = max(self.lcp_array)
        max_index = self.lcp_array.index(max_lcp)
        suffix_start = self.suffix_array[max_index]
        
        return self.text[suffix_start:suffix_start + max_lcp]

# =============================================================================
# SECTION 5: SUFFIX AUTOMATON (DAG)
# =============================================================================

@dataclass
class SAMState:
    """State in Suffix Automaton"""
    length: int = 0
    link: int = -1
    transitions: Dict[str, int] = None
    
    def __post_init__(self):
        if self.transitions is None:
            self.transitions = {}

class SuffixAutomaton:
    """
    Suffix Automaton - Linear space suffix tree alternative
    Construction: O(n), Space: O(n)
    Efficiently handles substring queries and counting
    """
    
    def __init__(self, text: str):
        self.text = text
        self.states = [SAMState()]
        self.last_state = 0
        
        for char in text:
            self._extend(char)
    
    def _extend(self, char: str) -> None:
        """Extend automaton with new character"""
        current = len(self.states)
        self.states.append(SAMState(length=self.states[self.last_state].length + 1))
        
        p = self.last_state
        while p != -1 and char not in self.states[p].transitions:
            self.states[p].transitions[char] = current
            p = self.states[p].link
        
        if p == -1:
            self.states[current].link = 0
        else:
            q = self.states[p].transitions[char]
            if self.states[p].length + 1 == self.states[q].length:
                self.states[current].link = q
            else:
                clone = len(self.states)
                self.states.append(SAMState(
                    length=self.states[p].length + 1,
                    link=self.states[q].link,
                    transitions=self.states[q].transitions.copy()
                ))
                
                while p != -1 and self.states[p].transitions.get(char) == q:
                    self.states[p].transitions[char] = clone
                    p = self.states[p].link
                
                self.states[q].link = clone
                self.states[current].link = clone
        
        self.last_state = current
    
    def contains_substring(self, substring: str) -> bool:
        """Check if substring exists - O(|substring|)"""
        state = 0
        for char in substring:
            if char not in self.states[state].transitions:
                return False
            state = self.states[state].transitions[char]
        return True
    
    def count_distinct_substrings(self) -> int:
        """Count total distinct substrings"""
        total = 0
        for i in range(1, len(self.states)):
            if self.states[i].link != -1:
                total += (self.states[i].length - 
                         self.states[self.states[i].link].length)
        return total

# =============================================================================
# SECTION 6: AHO-CORASICK ALGORITHM
# =============================================================================

class AhoCorasickNode:
    """Node for Aho-Corasick automaton"""
    
    def __init__(self):
        self.children: Dict[str, AhoCorasickNode] = {}
        self.failure_link: Optional[AhoCorasickNode] = None
        self.output: List[int] = []  # Pattern indices ending at this node

class AhoCorasick:
    """
    Aho-Corasick algorithm for multiple pattern matching
    Preprocessing: O(sum of pattern lengths)
    Searching: O(n + number of matches)
    """
    
    def __init__(self, patterns: List[str]):
        self.patterns = patterns
        self.root = AhoCorasickNode()
        self._build_trie()
        self._build_failure_links()
    
    def _build_trie(self) -> None:
        """Build trie from all patterns"""
        for pattern_idx, pattern in enumerate(self.patterns):
            current = self.root
            for char in pattern:
                if char not in current.children:
                    current.children[char] = AhoCorasickNode()
                current = current.children[char]
            current.output.append(pattern_idx)
    
    def _build_failure_links(self) -> None:
        """Build failure links for automaton"""
        queue = collections.deque()
        
        # All immediate children of root have failure link to root
        for child in self.root.children.values():
            child.failure_link = self.root
            queue.append(child)
        
        while queue:
            current = queue.popleft()
            
            for char, child in current.children.items():
                # Find failure link for child
                failure = current.failure_link
                
                while failure and char not in failure.children:
                    failure = failure.failure_link
                
                if failure and char in failure.children:
                    child.failure_link = failure.children[char]
                else:
                    child.failure_link = self.root
                
                # Add output patterns from failure link
                if child.failure_link:
                    child.output.extend(child.failure_link.output)
                
                queue.append(child)
    
    def search(self, text: str) -> Dict[int, List[int]]:
        """
        Search all patterns in text
        Returns: Dict[pattern_index] = List[start_positions]
        """
        results = collections.defaultdict(list)
        current = self.root
        
        for i, char in enumerate(text):
            # Follow failure links until we find a match or reach root
            while current and char not in current.children:
                current = current.failure_link or self.root
            
            if current and char in current.children:
                current = current.children[char]
                
                # Report all patterns ending at current position
                for pattern_idx in current.output:
                    pattern_len = len(self.patterns[pattern_idx])
                    start_pos = i - pattern_len + 1
                    results[pattern_idx].append(start_pos)
            else:
                current = self.root
        
        return dict(results)

# =============================================================================
# SECTION 7: PALINDROME ALGORITHMS
# =============================================================================

class PalindromeProcessor:
    """Advanced palindrome detection and manipulation algorithms"""
    
    @staticmethod
    def manacher_algorithm(s: str) -> Tuple[str, int, int]:
        """
        Manacher's algorithm for longest palindrome - O(n)
        Returns: (longest_palindrome, start_index, length)
        """
        if not s:
            return "", 0, 0
        
        # Transform string to handle even-length palindromes
        transformed = "#".join(f"^{s}$")
        n = len(transformed)
        radius = [0] * n
        center = right = 0
        
        max_len = 0
        max_center = 0
        
        for i in range(1, n - 1):
            mirror = 2 * center - i
            
            if i < right:
                radius[i] = min(right - i, radius[mirror])
            
            # Try to expand palindrome centered at i
            try:
                while transformed[i + 1 + radius[i]] == transformed[i - 1 - radius[i]]:
                    radius[i] += 1
            except IndexError:
                pass
            
            # Update center and right boundary
            if i + radius[i] > right:
                center, right = i, i + radius[i]
            
            # Track longest palindrome
            if radius[i] > max_len:
                max_len = radius[i]
                max_center = i
        
        # Extract actual palindrome
        start = (max_center - max_len) // 2
        palindrome = s[start:start + max_len]
        
        return palindrome, start, max_len
    
    @staticmethod
    def all_palindromic_substrings(s: str) -> List[str]:
        """Find all palindromic substrings using expand around centers"""
        palindromes = set()
        n = len(s)
        
        def expand_around_center(left: int, right: int):
            while (left >= 0 and right < n and s[left] == s[right]):
                palindromes.add(s[left:right + 1])
                left -= 1
                right += 1
        
        for i in range(n):
            # Odd length palindromes
            expand_around_center(i, i)
            # Even length palindromes
            expand_around_center(i, i + 1)
        
        return list(palindromes)
    
    @staticmethod
    def is_palindrome_after_one_removal(s: str) -> bool:
        """Check if string can become palindrome by removing at most one character"""
        def is_palindrome_range(left: int, right: int) -> bool:
            while left < right:
                if s[left] != s[right]:
                    return False
                left += 1
                right -= 1
            return True
        
        left, right = 0, len(s) - 1
        
        while left < right:
            if s[left] != s[right]:
                # Try removing either left or right character
                return (is_palindrome_range(left + 1, right) or 
                        is_palindrome_range(left, right - 1))
            left += 1
            right -= 1
        
        return True

# =============================================================================
# SECTION 8: STRING COMPRESSION ALGORITHMS
# =============================================================================

class StringCompressor:
    """Various string compression algorithms"""
    
    @staticmethod
    def run_length_encoding(s: str) -> str:
        """Run-length encoding compression"""
        if not s:
            return ""
        
        compressed = []
        current_char = s[0]
        count = 1
        
        for i in range(1, len(s)):
            if s[i] == current_char:
                count += 1
            else:
                compressed.append(f"{count}{current_char}")
                current_char = s[i]
                count = 1
        
        compressed.append(f"{count}{current_char}")
        result = "".join(compressed)
        
        # Return original if compression doesn't help
        return result if len(result) < len(s) else s
    
    @staticmethod
    def run_length_decoding(compressed: str) -> str:
        """Decode run-length encoded string"""
        if not compressed:
            return ""
        
        result = []
        i = 0
        
        while i < len(compressed):
            # Read count
            count = 0
            while i < len(compressed) and compressed[i].isdigit():
                count = count * 10 + int(compressed[i])
                i += 1
            
            # Read character
            if i < len(compressed):
                char = compressed[i]
                result.append(char * count)
                i += 1
        
        return "".join(result)
    
    @staticmethod
    def lz77_compress(s: str, window_size: int = 20) -> List[Tuple[int, int, str]]:
        """
        Simple LZ77 compression algorithm
        Returns list of (offset, length, next_char) tuples
        """
        compressed = []
        i = 0
        n = len(s)
        
        while i < n:
            max_length = 0
            max_offset = 0
            
            # Search in sliding window
            start = max(0, i - window_size)
            
            for j in range(start, i):
                length = 0
                while (i + length < n and 
                       j + length < i and 
                       s[j + length] == s[i + length]):
                    length += 1
                
                if length > max_length:
                    max_length = length
                    max_offset = i - j
            
            # Add compressed token
            if max_length > 0:
                next_char = s[i + max_length] if i + max_length < n else ""
                compressed.append((max_offset, max_length, next_char))
                i += max_length + 1
            else:
                compressed.append((0, 0, s[i]))
                i += 1
        
        return compressed

# =============================================================================
# SECTION 9: EDIT DISTANCE AND STRING TRANSFORMATION
# =============================================================================

class EditDistance:
    """Various edit distance algorithms and string transformations"""
    
    @staticmethod
    def levenshtein_distance(s1: str, s2: str) -> int:
        """
        Levenshtein distance (insert, delete, substitute) - O(n*m)
        Space optimized to O(min(n,m))
        """
        if len(s1) > len(s2):
            s1, s2 = s2, s1
        
        distances = list(range(len(s1) + 1))
        
        for i2, c2 in enumerate(s2):
            new_distances = [i2 + 1]
            for i1, c1 in enumerate(s1):
                if c1 == c2:
                    new_distances.append(distances[i1])
                else:
                    new_distances.append(min(
                        distances[i1] + 1,      # substitute
                        distances[i1 + 1] + 1,  # delete
                        new_distances[-1] + 1   # insert
                    ))
            distances = new_distances
        
        return distances[-1]
    
    @staticmethod
    def edit_distance_with_operations(s1: str, s2: str) -> Tuple[int, List[str]]:
        """
        Return edit distance and sequence of operations
        Operations: 'insert X', 'delete X', 'substitute X->Y'
        """
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        # Initialize base cases
        for i in range(n + 1):
            dp[i][0] = i
        for j in range(m + 1):
            dp[0][j] = j
        
        # Fill DP table
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(
                        dp[i-1][j],      # delete
                        dp[i][j-1],      # insert
                        dp[i-1][j-1]     # substitute
                    )
        
        # Reconstruct operations
        operations = []
        i, j = n, m
        
        while i > 0 or j > 0:
            if i > 0 and j > 0 and s1[i-1] == s2[j-1]:
                i -= 1
                j -= 1
            elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
                operations.append(f"substitute {s1[i-1]}->{s2[j-1]}")
                i -= 1
                j -= 1
            elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
                operations.append(f"delete {s1[i-1]}")
                i -= 1
            elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
                operations.append(f"insert {s2[j-1]}")
                j -= 1
        
        operations.reverse()
        return dp[n][m], operations
    
    @staticmethod
    def longest_common_subsequence(s1: str, s2: str) -> str:
        """Find longest common subsequence - O(n*m)"""
        n, m = len(s1), len(s2)
        dp = [[""] * (m + 1) for _ in range(n + 1)]
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + s1[i-1]
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1], key=len)
        
        return dp[n][m]
    
    @staticmethod
    def longest_common_substring(s1: str, s2: str) -> str:
        """Find longest common substring - O(n*m)"""
        n, m = len(s1), len(s2)
        dp = [[0] * (m + 1) for _ in range(n + 1)]
        
        max_length = 0
        ending_pos = 0
        
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                    if dp[i][j] > max_length:
                        max_length = dp[i][j]
                        ending_pos = i
        
        return s1[ending_pos - max_length:ending_pos]

# =============================================================================
# SECTION 10: ADVANCED STRING HASHING
# =============================================================================

class StringHasher:
    """Advanced string hashing with collision avoidance"""
    
    def __init__(self, base: int = 257, mod: int = 10**9 + 7):
        self.base = base
        self.mod = mod
        self.powers = [1]
    
    def _ensure_power(self, length: int) -> None:
        """Ensure we have enough precomputed powers"""
        while len(self.powers) <= length:
            self.powers.append((self.powers[-1] * self.base) % self.mod)
    
    def hash_string(self, s: str) -> int:
        """Compute hash of entire string"""
        hash_value = 0
        for char in s:
            hash_value = (hash_value * self.base + ord(char)) % self.mod
        return hash_value
    
    def build_prefix_hashes(self, s: str) -> List[int]:
        """Build prefix hashes for substring queries"""
        n = len(s)
        self._ensure_power(n)
        
        hashes = [0] * (n + 1)
        for i in range(n):
            hashes[i + 1] = (hashes[i] * self.base + ord(s[i])) % self.mod
        
        return hashes
    
    def substring_hash(self, prefix_hashes: List[int], start: int, end: int) -> int:
        """Get hash of substring s[start:end] using prefix hashes"""
        length = end - start
        self._ensure_power(length)
        
        hash_value = (prefix_hashes[end] - 
                     prefix_hashes[start] * self.powers[length]) % self.mod
        return hash_value if hash_value >= 0 else hash_value + self.mod
    
    def rolling_hash_search(self, text: str, pattern: str) -> List[int]:
        """Rolling hash pattern matching"""
        if not pattern or len(pattern) > len(text):
            return []
        
        n, m = len(text), len(pattern)
        self._ensure_power(m)
        
        pattern_hash = self.hash_string(pattern)
        text_hash = 0
        
        positions = []
        
        # Initialize first window
        for i in range(m):
            text_hash = (text_hash * self.base + ord(text[i])) % self.mod
        
        if text_hash == pattern_hash and text[:m] == pattern:
            positions.append(0)
        
        # Rolling hash for remaining windows
        for i in range(m, n):
            # Remove leftmost character and add rightmost
            text_hash = (text_hash - ord(text[i - m]) * self.powers[m - 1]) % self.mod
            text_hash = (text_hash * self.base + ord(text[i])) % self.mod
            
            if text_hash < 0:
                text_hash += self.mod
            
            if (text_hash == pattern_hash and 
                text[i - m + 1:i + 1] == pattern):
                positions.append(i - m + 1)
        
        return positions

# =============================================================================
# SECTION 11: COMPREHENSIVE TEST SUITE
# =============================================================================

def run_comprehensive_tests():
    """Execute comprehensive test suite for all string algorithms"""
    
    print("Testing String Data Structures & Algorithms...")
    
    # Test 1: Basic String Operations
    sp = StringProcessor()
    assert sp.is_palindrome("racecar") == True
    assert sp.is_palindrome("hello") == False
    assert sp.reverse_words("  the sky   is blue  ") == "blue is sky the"
    print("✓ Basic string operations")
    
    # Test 2: String Matching
    text = "ababcabcabababcab"
    pattern = "abcab"
    
    matcher = StringMatcher()
    naive_result = matcher.naive_search(text, pattern)
    kmp_result = matcher.kmp_search(text, pattern)
    rk_result = matcher.rabin_karp_search(text, pattern)
    bm_result = matcher.boyer_moore_search(text, pattern)
    
    assert naive_result == kmp_result == rk_result == bm_result
    print("✓ String matching algorithms")
    
    # Test 3: Z-Algorithm
    z_result = ZAlgorithm.pattern_search(text, pattern)
    assert z_result == naive_result
    print("✓ Z-algorithm")
    
    # Test 4: Trie
    trie = Trie()
    words = ["apple", "app", "application", "apply", "aptitude"]
    for word in words:
        trie.insert(word)
    
    assert trie.search("app") == True
    assert trie.search("appl") == False
    assert trie.count_words_with_prefix("app") == 4
    print("✓ Trie data structure")
    
    # Test 5: Suffix Array
    sa = SuffixArray("banana")
    pattern_positions = sa.find_pattern("ana")
    assert len(pattern_positions) == 2
    print("✓ Suffix array")
    
    # Test 6: Aho-Corasick
    patterns = ["he", "she", "his", "hers"]
    ac = AhoCorasick(patterns)
    results = ac.search("ushers")
    assert len(results) > 0
    print("✓ Aho-Corasick algorithm")
    
    # Test 7: Palindromes
    pp = PalindromeProcessor()
    longest, start, length = pp.manacher_algorithm("babad")
    assert longest in ["bab", "aba"]
    print("✓ Palindrome algorithms")
    
    # Test 8: Compression
    sc = StringCompressor()
    compressed = sc.run_length_encoding("aaabbbccccdddd")
    decompressed = sc.run_length_decoding(compressed)
    assert decompressed == "aaabbbccccdddd"
    print("✓ String compression")
    
    # Test 9: Edit Distance
    ed = EditDistance()
    distance = ed.levenshtein_distance("kitten", "sitting")
    assert distance == 3
    
    lcs = ed.longest_common_subsequence("ABCDGH", "AEDFHR")
    assert lcs == "ADH"
    print("✓ Edit distance algorithms")
    
    # Test 10: String Hashing
    hasher = StringHasher()
    prefix_hashes = hasher.build_prefix_hashes("abcdefgh")
    sub_hash = hasher.substring_hash(prefix_hashes, 2, 5)
    direct_hash = hasher.hash_string("cde")
    assert sub_hash == direct_hash
    print("✓ String hashing")
    
    print("\n🎉 All tests passed! String algorithms implementation is complete.")

# Example usage demonstrating various algorithms
def demonstrate_algorithms():
    """Demonstrate usage of various string algorithms"""
    
    print("\n" + "="*60)
    print("STRING ALGORITHMS DEMONSTRATION")
    print("="*60)
    
    # Text and patterns for demonstration
    text = "The quick brown fox jumps over the lazy dog. The dog was lazy."
    patterns = ["the", "dog", "lazy", "quick"]
    
    print(f"Text: {text}")
    print(f"Patterns: {patterns}")
    print()
    
    # 1. Multiple pattern matching with Aho-Corasick
    print("1. Aho-Corasick Multiple Pattern Matching:")
    ac = AhoCorasick([p.lower() for p in patterns])
    results = ac.search(text.lower())
    for i, pattern in enumerate(patterns):
        positions = results.get(i, [])
        print(f"   '{pattern}' found at positions: {positions}")
    print()
    
    # 2. Longest palindrome
    print("2. Longest Palindrome Detection:")
    test_string = "babad"
    pp = PalindromeProcessor()
    palindrome, start, length = pp.manacher_algorithm(test_string)
    print(f"   In '{test_string}': longest palindrome is '{palindrome}' at position {start}")
    print()
    
    # 3. String compression
    print("3. String Compression:")
    original = "aaabbbccccddddeeee"
    sc = StringCompressor()
    compressed = sc.run_length_encoding(original)
    print(f"   Original: {original} ({len(original)} chars)")
    print(f"   Compressed: {compressed} ({len(compressed)} chars)")
    print()
    
    # 4. Edit distance with operations
    print("4. Edit Distance Analysis:")
    s1, s2 = "kitten", "sitting"
    ed = EditDistance()
    distance, operations = ed.edit_distance_with_operations(s1, s2)
    print(f"   Transform '{s1}' to '{s2}':")
    print(f"   Distance: {distance}")
    print(f"   Operations: {operations}")
    print()
    
    # 5. Suffix array analysis
    print("5. Suffix Array Analysis:")
    sa_text = "banana"
    sa = SuffixArray(sa_text)
    print(f"   Text: {sa_text}")
    print(f"   Suffix Array: {sa.suffix_array}")
    print(f"   LCP Array: {sa.lcp_array}")
    repeated = sa.longest_repeated_substring()
    print(f"   Longest repeated substring: '{repeated}'")

if __name__ == "__main__":
    run_comprehensive_tests()
    demonstrate_algorithms()