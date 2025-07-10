from typing import List
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
import time

class KMPMatcher:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = Console()

    def search(self, text: str, pattern: str) -> List[int]:
        if not text or not pattern:
            return []

        lps = self._compute_lps(pattern)
        matches = []
        i = j = step = 0

        if self.verbose:
            self.console.print(Panel.fit("[bold blue]KMP Pattern Matching Started[/bold blue]", border_style="blue"))
            self._display_lps_array(pattern, lps)

        while i < len(text):
            step += 1
            current_text_char = text[i]
            current_pattern_char = pattern[j] if j < len(pattern) else '-'

            if self.verbose:
                self._display_step_info(step, i, j, current_text_char, current_pattern_char)

            if current_text_char == current_pattern_char:
                i += 1
                j += 1
                if j == len(pattern):
                    matches.append(i - j)
                    if self.verbose:
                        self.console.print(f"[green]✓ Match found at index {i - j}[/green]")
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
            if self.verbose:
                time.sleep(0.5)

        if self.verbose:
            self._display_final_result(matches)

        return matches

    def _compute_lps(self, pattern: str) -> List[int]:
        lps = [0] * len(pattern)
        length = 0
        i = 1

        if self.verbose:
            self.console.print(Panel.fit("[bold yellow]Building LPS (Longest Prefix Suffix) Array[/bold yellow]", border_style="yellow"))

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

    def _display_lps_array(self, pattern: str, lps: List[int]):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Index", style="cyan")
        table.add_column("Pattern Char", style="yellow")
        table.add_column("LPS Value", style="green")

        for idx, (ch, val) in enumerate(zip(pattern, lps)):
            table.add_row(str(idx), ch, str(val))

        self.console.print(table)

    def _display_step_info(self, step: int, i: int, j: int, t_char: str, p_char: str):
        self.console.rule(f"[bold cyan]Step {step}[/bold cyan]", style="cyan")
        info_table = Table(show_header=True, header_style="bold white")
        info_table.add_column("Variable", style="bold white")
        info_table.add_column("Value", style="bold yellow")
        info_table.add_row("Text Index (i)", str(i))
        info_table.add_row("Text Char", f"[green]{t_char}[/green]")
        info_table.add_row("Pattern Index (j)", str(j))
        info_table.add_row("Pattern Char", f"[yellow]{p_char}[/yellow]")
        info_table.add_row("Match Status", "[green]Match[/green]" if t_char == p_char else "[red]Mismatch[/red]")
        self.console.print(info_table)

    def _display_final_result(self, matches: List[int]):
        if matches:
            result = ", ".join(str(m) for m in matches)
            self.console.print(Panel.fit(f"[bold green]Matches Found at Indices: {result}[/bold green]", border_style="green"))
        else:
            self.console.print(Panel.fit("[bold red]No Matches Found[/bold red]", border_style="red"))

if __name__ == "__main__":
    test_cases = [
        ("ababcabcabababd", "ababd"),
        ("aaaaa", "aaa"),
        ("abcxabcdabxabcdabcdabcy", "abcdabcy"),
        ("abc", "abc"),
        ("abc", "abcd"),
        ("", "abc"),
        ("abc", ""),
        ("aabaabaafa", "aabaaf")
    ]

    matcher = KMPMatcher(verbose=True)
    for index, (text, pattern) in enumerate(test_cases, 1):
        matcher.console.rule(f"[bold magenta]Test Case {index}[/bold magenta]", style="magenta")
        matcher.console.print(f"[bold white]Text:    [/bold white]{text}")
        matcher.console.print(f"[bold white]Pattern: [/bold white]{pattern}")
        result = matcher.search(text, pattern)
        matcher.console.print(f"[bold green]Result:  [/bold green]{result}")
        matcher.console.print("═" * 60)
        time.sleep(1.0)