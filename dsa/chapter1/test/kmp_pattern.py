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
        if not pattern or not text:
            return []

        lps = self._compute_lps(pattern)
        result = []
        i = j = 0
        step = 1

        if self.verbose:
            self.console.print(Panel.fit("[bold blue]KMP Pattern Searching Initiated[/bold blue]", border_style="blue"))
            self._render_lps_table(pattern, lps)

        while i < len(text):
            if self.verbose:
                self._render_step_header(step, i, j, text[i], pattern[j] if j < len(pattern) else '-')

            if text[i] == pattern[j]:
                i += 1
                j += 1
                if j == len(pattern):
                    result.append(i - j)
                    if self.verbose:
                        self.console.print(f"[green]✓ Match found at index {i - j}[/green]")
                    j = lps[j - 1]
            else:
                if j != 0:
                    j = lps[j - 1]
                else:
                    i += 1
            step += 1
            if self.verbose:
                time.sleep(0.5)

        if self.verbose:
            self._render_final_results(result)
        
        return result

    def _compute_lps(self, pattern: str) -> List[int]:
        lps = [0] * len(pattern)
        length = 0
        i = 1

        if self.verbose:
            self.console.print(Panel.fit("[bold yellow]Computing Longest Prefix Suffix (LPS) Array[/bold yellow]", border_style="yellow"))

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

    def _render_lps_table(self, pattern: str, lps: List[int]):
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Index", style="bold white")
        table.add_column("Character", style="bold magenta")
        table.add_column("LPS Value", style="bold green")

        for idx, (char, value) in enumerate(zip(pattern, lps)):
            table.add_row(str(idx), char, str(value))

        self.console.print(table)

    def _render_step_header(self, step: int, i: int, j: int, text_char: str, pattern_char: str):
        self.console.print(f"\n[bold cyan]Step {step}[/bold cyan]")
        self.console.print(f"[white]Text Index (i):[/white] {i} - [bold green]{text_char}[/bold green]")
        self.console.print(f"[white]Pattern Index (j):[/white] {j} - [bold yellow]{pattern_char}[/bold yellow]")

    def _render_final_results(self, matches: List[int]):
        if matches:
            match_str = ", ".join(str(idx) for idx in matches)
            self.console.print(Panel.fit(f"[bold green]All Matches Found at Indices: {match_str}[/bold green]", border_style="green"))
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
    for idx, (text, pattern) in enumerate(test_cases):
        matcher.console.print(f"\n[bold magenta]══════ Test Case {idx + 1} ══════[/bold magenta]")
        matcher.console.print(f"[bold white]Text:    [/bold white]{text}")
        matcher.console.print(f"[bold white]Pattern: [/bold white]{pattern}")
        result = matcher.search(text, pattern)
        matcher.console.print(f"[bold cyan]Result:[/bold cyan] {result}")
        matcher.console.print("═" * 80)
        time.sleep(1)