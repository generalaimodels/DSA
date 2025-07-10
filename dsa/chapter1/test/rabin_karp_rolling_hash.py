from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

console = Console()

class RabinKarpMatcher:
    def __init__(self, base=256, modulus=10**9 + 7, verbose=False):
        self.base = base
        self.modulus = modulus
        self.verbose = verbose

    def _hash(self, s, length):
        h = 0
        for i in range(length):
            h = (h * self.base + ord(s[i])) % self.modulus
        return h

    def _rehash(self, prev_hash, left_char, right_char, high_base):
        return ((prev_hash - ord(left_char) * high_base) * self.base + ord(right_char)) % self.modulus

    def search(self, text, pattern):
        n, m = len(text), len(pattern)
        if m == 0 or m > n:
            return []

        result = []
        pattern_hash = self._hash(pattern, m)
        text_hash = self._hash(text, m)
        high_base = pow(self.base, m - 1, self.modulus)

        if self.verbose:
            self._show_intro(text, pattern)
            self._render_step(0, text[:m], text_hash, pattern_hash, text[:m] == pattern)

        for i in range(1, n - m + 1):
            text_hash = self._rehash(text_hash, text[i - 1], text[i + m - 1], high_base)
            substring = text[i:i + m]
            match_found = text_hash == pattern_hash and substring == pattern
            if match_found:
                result.append(i)
            if self.verbose:
                self._render_step(i, substring, text_hash, pattern_hash, match_found)

        if text[:m] == pattern:
            result.insert(0, 0)
        elif text_hash == pattern_hash and text[:m] == pattern:
            result.append(0)

        return result

    def _show_intro(self, text, pattern):
        console.print(Panel.fit(f"[bold cyan]Rabin-Karp Rolling Hash Search[/bold cyan]\n[bold green]Text:[/bold green] {text}\n[bold yellow]Pattern:[/bold yellow] {pattern}", box=box.ROUNDED))

    def _render_step(self, index, window, current_hash, pattern_hash, is_match):
        table = Table(box=box.SIMPLE, expand=False)
        table.add_column("Window Index", justify="center", style="bold")
        table.add_column("Substring", justify="center", style="green" if is_match else "")
        table.add_column("Current Hash", justify="center")
        table.add_column("Pattern Hash", justify="center")
        table.add_column("Match", justify="center", style="bold red" if is_match else "dim")

        table.add_row(str(index), window, str(current_hash), str(pattern_hash), "✅ Match" if is_match else "❌")

        console.print(table)

if __name__ == "__main__":
    text_input = "ababcabcabababd"
    pattern_input = "ababd"

    matcher = RabinKarpMatcher(verbose=True)
    matches = matcher.search(text_input, pattern_input)

    console.print(f"[bold underline green]Pattern found at indices: {matches}[/bold underline green]")