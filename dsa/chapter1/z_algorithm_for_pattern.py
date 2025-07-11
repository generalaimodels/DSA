from typing import List
from rich.console import Console, Group
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich import box
import time

class ZAlgorithm:
    def __init__(self, verbose: bool = False, delay: float = 0.1):
        self.verbose = verbose
        self.delay = delay
        self.console = Console()

    def _render(self, seq: str, z: List[int], idx: int, l: int, r: int) -> Panel:
        table = Table(box=box.SIMPLE_HEAVY, expand=True)
        table.add_column("i", justify="right", style="bold")
        table.add_column("s[i]", justify="center")
        table.add_column("Z[i]", justify="right")
        for i, ch in enumerate(seq):
            style = "bold red" if i == idx else ("bold green" if l <= i <= r else "white")
            table.add_row(str(i), Text(ch, style=style), str(z[i]) if z[i] else "")
        info = Text(f"L={l}  R={r}  i={idx}", style="bold cyan")
        return Panel(Group(table, info), title="Z-Algorithm", border_style="blue")

    def compute_z_array(self, seq: str) -> List[int]:
        n = len(seq)
        z = [0] * n
        l = r = 0
        if not self.verbose:
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < n and seq[z[i]] == seq[i + z[i]]:
                    z[i] += 1
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
            return z
        with Live(self._render(seq, z, 0, l, r), console=self.console, refresh_per_second=24, screen=True) as live:
            for i in range(1, n):
                if i <= r:
                    z[i] = min(r - i + 1, z[i - l])
                while i + z[i] < n and seq[z[i]] == seq[i + z[i]]:
                    z[i] += 1
                    live.update(self._render(seq, z, i, l, r))
                    time.sleep(self.delay)
                if i + z[i] - 1 > r:
                    l, r = i, i + z[i] - 1
                live.update(self._render(seq, z, i, l, r))
                time.sleep(self.delay)
        return z

    def pattern_matching(self, text: str, pattern: str) -> List[int]:
        concat = pattern + "$" + text
        z = self.compute_z_array(concat)
        m = len(pattern)
        return [i - m - 1 for i in range(m + 1, len(concat)) if z[i] == m]

if __name__ == "__main__":
    text_input = "abacabadabacaba"
    pattern_input = "aba"
    algo = ZAlgorithm(verbose=True, delay=0.15)
    result = algo.pattern_matching(text_input, pattern_input)
    algo.console.print(Text(f"Matches at indices {result}", style="bold magenta"))
