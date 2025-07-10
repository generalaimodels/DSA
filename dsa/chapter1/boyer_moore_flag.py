from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

class BoyerMooreMajorityVote:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.console = Console() if verbose else None

    def find_majority(self, nums):
        candidate = None
        count = 0
        if self.verbose:
            self._print_intro(nums)
        for idx, num in enumerate(nums):
            prev_candidate, prev_count = candidate, count
            if count == 0:
                candidate = num
                count = 1
                if self.verbose:
                    self._print_step(idx, num, prev_candidate, prev_count, candidate, count, "Reset candidate")
            elif num == candidate:
                count += 1
                if self.verbose:
                    self._print_step(idx, num, prev_candidate, prev_count, candidate, count, "Increment count")
            else:
                count -= 1
                if self.verbose:
                    self._print_step(idx, num, prev_candidate, prev_count, candidate, count, "Decrement count")
        if self.verbose:
            self._print_candidate(candidate)
        if nums.count(candidate) > len(nums) // 2:
            if self.verbose:
                self._print_verification(candidate, True)
            return candidate
        if self.verbose:
            self._print_verification(candidate, False)
        return None

    def _print_intro(self, nums):
        self.console.print(Panel(Text("Boyer-Moore Majority Vote Algorithm", style="bold magenta"), expand=False))
        self.console.print(f"[bold yellow]Input Array:[/bold yellow] {nums}\n")
        self.console.print("[bold cyan]Stepwise Execution:[/bold cyan]\n")

    def _print_step(self, idx, num, prev_candidate, prev_count, candidate, count, action):
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Index", style="bold")
        table.add_column("Element", style="bold")
        table.add_column("Prev Candidate", style="dim")
        table.add_column("Prev Count", style="dim")
        table.add_column("New Candidate", style="green")
        table.add_column("New Count", style="green")
        table.add_column("Action", style="bold magenta")
        table.add_row(
            str(idx),
            str(num),
            str(prev_candidate),
            str(prev_count),
            str(candidate),
            str(count),
            action
        )
        self.console.print(table)

    def _print_candidate(self, candidate):
        self.console.print(f"\n[bold green]Candidate after first pass:[/bold green] {candidate}\n")

    def _print_verification(self, candidate, is_majority):
        if is_majority:
            self.console.print(f"[bold green]Verified:[/bold green] {candidate} is the majority element.\n")
        else:
            self.console.print(f"[bold red]No majority element found.[/bold red]\n")

if __name__ == "__main__":
    nums = [2, 2, 1, 1, 1, 2, 2]
    verbose = True
    bm = BoyerMooreMajorityVote(verbose=verbose)
    result = bm.find_majority(nums)
    print("Majority Element:", result)