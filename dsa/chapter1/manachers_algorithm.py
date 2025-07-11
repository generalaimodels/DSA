from typing import List, Tuple
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box
import time

class Manacher:
    def __init__(self, verbose: bool = False, delay: float = 0.1):
        self.verbose = verbose
        self.delay = delay
        self.console = Console()

    @staticmethod
    def _transform(sequence: str) -> str:
        return "^#" + "#".join(sequence) + "#$"

    def _render(self, transformed: str, radii: List[int], idx: int, center: int, right: int) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True, pad_edge=False)
        table.add_column("i", justify="right", style="bold")
        table.add_column("T[i]", justify="center")
        table.add_column("P[i]", justify="right")
        for i, ch in enumerate(transformed):
            style = "bold red" if i == idx else ("bold green" if center - radii[center] <= i <= center + radii[center] else "white")
            table.add_row(str(i), Text(ch, style=style), str(radii[i]) if radii[i] else "")
        info = Text(f"C={center}  R={right}  i={idx}", style="bold cyan")
        return Panel(Group(table, info), title="Manacher’s Algorithm", border_style="blue")

    def _compute_radii(self, sequence: str) -> Tuple[str, List[int]]:
        T = self._transform(sequence)
        n = len(T)
        P = [0] * n
        center = right = 0
        if not self.verbose:
            for i in range(1, n - 1):
                mirror = 2 * center - i
                if i < right:
                    P[i] = min(right - i, P[mirror])
                while T[i + P[i] + 1] == T[i - P[i] - 1]:
                    P[i] += 1
                if i + P[i] > right:
                    center, right = i, i + P[i]
            return T, P
        with Live(self._render(T, P, 0, center, right), console=self.console, refresh_per_second=24, screen=True) as live:
            for i in range(1, n - 1):
                mirror = 2 * center - i
                if i < right:
                    P[i] = min(right - i, P[mirror])
                while T[i + P[i] + 1] == T[i - P[i] - 1]:
                    P[i] += 1
                    live.update(self._render(T, P, i, center, right))
                    time.sleep(self.delay)
                if i + P[i] > right:
                    center, right = i, i + P[i]
                live.update(self._render(T, P, i, center, right))
                time.sleep(self.delay)
        return T, P

    def longest_palindrome(self, sequence: str) -> Tuple[str, int, int]:
        if not sequence:
            return "", 0, -1
        _, P = self._compute_radii(sequence)
        max_len, center_idx = max((v, i) for i, v in enumerate(P))
        start = (center_idx - max_len) // 2
        return sequence[start:start + max_len], start, start + max_len - 1

if __name__ == "__main__":
    test_string = "forgeeksskeegfor"
    manacher = Manacher(verbose=True, delay=0.15)
    palindrome, left, right = manacher.longest_palindrome(test_string)
    manacher.console.print(Text(f"Longest palindrome '{palindrome}' found at [{left}, {right}]", style="bold magenta"))
