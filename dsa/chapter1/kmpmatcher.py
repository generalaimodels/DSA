from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import time

class KMPMatcher:
    def __init__(self):
        self.console = Console()
    
    def build_lps_array(self, pattern, verbose=False):
        m = len(pattern)
        lps = [0] * m
        length = 0
        i = 1
        
        if verbose:
            self.display_lps_initialization(pattern)
        
        while i < m:
            if verbose:
                self.display_lps_step(pattern, lps, i, length)
            
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                if verbose:
                    self.display_lps_match(pattern, lps, i, length)
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                    if verbose:
                        self.display_lps_fallback(length)
                else:
                    lps[i] = 0
                    if verbose:
                        self.display_lps_no_match(pattern, lps, i)
                    i += 1
        
        if verbose:
            self.display_lps_complete(pattern, lps)
        
        return lps
    
    def search(self, text, pattern, verbose=False):
        if not pattern or not text:
            return []
        
        n = len(text)
        m = len(pattern)
        lps = self.build_lps_array(pattern, verbose)
        matches = []
        
        if verbose:
            self.display_search_initialization(text, pattern, lps)
        
        i = 0
        j = 0
        
        while i < n:
            if verbose:
                self.display_search_step(text, pattern, i, j, matches)
            
            if pattern[j] == text[i]:
                i += 1
                j += 1
                if verbose:
                    self.display_character_match(text[i-1], i-1, j-1)
            
            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
                if verbose:
                    self.display_pattern_found(i - m, matches)
            elif i < n and pattern[j] != text[i]:
                if j != 0:
                    old_j = j
                    j = lps[j - 1]
                    if verbose:
                        self.display_mismatch_with_shift(text[i], pattern[old_j], j)
                else:
                    if verbose:
                        self.display_mismatch_advance(text[i], i)
                    i += 1
        
        if verbose:
            self.display_search_results(matches)
        
        return matches
    
    def display_lps_initialization(self, pattern):
        self.console.print(Panel.fit(
            f"[bold blue]Building LPS Array for Pattern: [yellow]{pattern}[/yellow][/bold blue]\n"
            f"LPS[i] = Length of longest proper prefix which is also suffix for pattern[0..i]",
            title="[bold]LPS Array Construction[/bold]",
            box=box.ROUNDED
        ))
        time.sleep(0.5)
    
    def display_lps_step(self, pattern, lps, i, length):
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Index", justify="center", width=8)
        table.add_column("Character", justify="center", width=10)
        table.add_column("LPS Value", justify="center", width=10)
        table.add_column("Status", justify="center", width=15)
        
        for idx in range(len(pattern)):
            char = pattern[idx]
            lps_val = str(lps[idx]) if idx <= i else "?"
            
            if idx == i:
                status = "Current (i)"
                char_style = "bold red on white"
            elif idx == length:
                status = "Prefix (len)"
                char_style = "bold blue on white"
            elif idx < i:
                status = "Processed"
                char_style = "green"
            else:
                status = "Pending"
                char_style = "dim"
            
            table.add_row(
                str(idx),
                f"[{char_style}]{char}[/{char_style}]",
                f"[cyan]{lps_val}[/cyan]",
                f"[yellow]{status}[/yellow]"
            )
        
        self.console.print(f"\n[bold]Step: Comparing pattern[{i}]='{pattern[i]}' with pattern[{length}]='{pattern[length]}'[/bold]")
        self.console.print(table)
        time.sleep(0.4)
    
    def display_lps_match(self, pattern, lps, i, length):
        self.console.print(f"[bold green]✓ Match! LPS[{i}] = {lps[i]} (prefix length: {length})[/bold green]")
        time.sleep(0.3)
    
    def display_lps_fallback(self, new_length):
        self.console.print(f"[bold yellow]↩ Mismatch, falling back using LPS: new length = {new_length}[/bold yellow]")
        time.sleep(0.3)
    
    def display_lps_no_match(self, pattern, lps, i):
        self.console.print(f"[bold red]✗ No prefix match, LPS[{i}] = 0[/bold red]")
        time.sleep(0.3)
    
    def display_lps_complete(self, pattern, lps):
        table = Table(title="[bold green]Final LPS Array[/bold green]", box=box.DOUBLE)
        table.add_column("Index", justify="center", style="cyan")
        table.add_column("Character", justify="center", style="yellow")
        table.add_column("LPS Value", justify="center", style="green")
        table.add_column("Meaning", justify="center", style="magenta")
        
        for i in range(len(pattern)):
            meaning = f"Length {lps[i]}" if lps[i] > 0 else "No prefix"
            table.add_row(str(i), pattern[i], str(lps[i]), meaning)
        
        self.console.print(table)
        time.sleep(0.8)
    
    def display_search_initialization(self, text, pattern, lps):
        self.console.print(Panel.fit(
            f"[bold blue]KMP Pattern Matching Algorithm[/bold blue]\n"
            f"Text: [green]{text}[/green] (length: {len(text)})\n"
            f"Pattern: [yellow]{pattern}[/yellow] (length: {len(pattern)})\n"
            f"LPS Array: [cyan]{lps}[/cyan]",
            title="[bold]Search Phase Initialization[/bold]",
            box=box.ROUNDED
        ))
        time.sleep(0.5)
    
    def display_search_step(self, text, pattern, i, j, matches):
        highlighted_text = Text()
        pattern_alignment = " " * max(0, i - j)
        
        for idx, char in enumerate(text):
            if idx == i and i < len(text):
                highlighted_text.append(char, style="bold red on white")
            elif matches:
                in_previous_match = any(match <= idx < match + len(pattern) for match in matches)
                if in_previous_match:
                    highlighted_text.append(char, style="bold green")
                else:
                    highlighted_text.append(char, style="white")
            else:
                highlighted_text.append(char, style="white")
        
        pattern_display = Text()
        for idx, char in enumerate(pattern):
            if idx == j:
                pattern_display.append(char, style="bold yellow on blue")
            elif idx < j:
                pattern_display.append(char, style="bold green")
            else:
                pattern_display.append(char, style="dim")
        
        comparison_status = "Comparing" if i < len(text) and j < len(pattern) else "Complete"
        
        self.console.print(f"\n[bold]Text[{i}] vs Pattern[{j}] - {comparison_status}[/bold]")
        self.console.print(f"Text:    {highlighted_text}")
        self.console.print(f"Pattern: {pattern_alignment}{pattern_display}")
        
        if i < len(text) and j < len(pattern):
            match_symbol = "=" if text[i] == pattern[j] else "≠"
            self.console.print(f"         {' ' * (len(pattern_alignment) + i)}[bold]'{text[i]}' {match_symbol} '{pattern[j]}'[/bold]")
        
        time.sleep(0.3)
    
    def display_character_match(self, char, text_pos, pattern_pos):
        self.console.print(f"[bold green]✓ Character match: '{char}' (text[{text_pos}] = pattern[{pattern_pos}])[/bold green]")
        time.sleep(0.2)
    
    def display_pattern_found(self, position, matches):
        self.console.print(f"[bold bright_green]🎯 Complete pattern match found at position {position}![/bold bright_green]")
        self.console.print(f"[bold bright_green]   Total matches so far: {len(matches)}[/bold bright_green]")
        time.sleep(0.5)
    
    def display_mismatch_with_shift(self, text_char, pattern_char, new_j):
        self.console.print(f"[bold red]✗ Mismatch: '{text_char}' ≠ '{pattern_char}'[/bold red]")
        self.console.print(f"[bold yellow]→ Using LPS to shift pattern: new j = {new_j}[/bold yellow]")
        time.sleep(0.4)
    
    def display_mismatch_advance(self, text_char, text_pos):
        self.console.print(f"[bold orange3]✗ Mismatch at start, advancing text: '{text_char}' at position {text_pos}[/bold orange3]")
        time.sleep(0.3)
    
    def display_search_results(self, matches):
        if matches:
            self.console.print(Panel.fit(
                f"[bold green]Pattern matches found at positions: {matches}[/bold green]\n"
                f"Total occurrences: [bold bright_green]{len(matches)}[/bold bright_green]\n"
                f"Algorithm complexity: [cyan]O(n + m) time, O(m) space[/cyan]",
                title="[bold]KMP Search Results[/bold]",
                box=box.DOUBLE
            ))
        else:
            self.console.print(Panel.fit(
                "[bold red]No pattern matches found[/bold red]\n"
                f"Algorithm complexity: [cyan]O(n + m) time, O(m) space[/cyan]",
                title="[bold]KMP Search Results[/bold]",
                box=box.DOUBLE
            ))

def demonstrate_kmp_algorithm():
    matcher = KMPMatcher()
    
    test_cases = [
        ("ABABCABABA", "ABA"),
        ("AABAACAADAABAABA", "AABA"),
        ("ABCABCABCABC", "ABCAB"),
        ("ABABABABAB", "ABABAB"),
        ("GEEKSFORGEEKS", "GEEK")
    ]
    
    for text, pattern in test_cases:
        console = Console()
        console.print(f"\n[bold magenta]{'='*80}[/bold magenta]")
        console.print(f"[bold white]KMP Algorithm Demo: Searching '{pattern}' in '{text}'[/bold white]")
        console.print(f"[bold magenta]{'='*80}[/bold magenta]")
        
        start_time = time.time()
        matches = matcher.search(text, pattern, verbose=True)
        end_time = time.time()
        
        console.print(f"\n[bold yellow]Performance Analysis:[/bold yellow]")
        console.print(f"Text length (n): [cyan]{len(text)}[/cyan]")
        console.print(f"Pattern length (m): [cyan]{len(pattern)}[/cyan]")
        console.print(f"Execution time: [green]{(end_time - start_time):.4f} seconds[/green]")
        console.print(f"Theoretical complexity: [green]O({len(text)} + {len(pattern)}) = O({len(text) + len(pattern)})[/green]")
        console.print(f"Space complexity: [green]O({len(pattern)})[/green]")
        
        if matches:
            console.print(f"\n[bold bright_green]Success! Found {len(matches)} occurrence(s) at: {matches}[/bold bright_green]")
        else:
            console.print(f"\n[bold red]Pattern not found in the given text[/bold red]")
        
        time.sleep(1.5)

if __name__ == "__main__":
    demonstrate_kmp_algorithm()