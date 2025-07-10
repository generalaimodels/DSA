from typing import List, Tuple
from rich.console import Console
from rich.table import Table
from rich.live import Live

console = Console()


def execute_kadane(sequence: List[int], verbose: bool = False) -> Tuple[int, int, int]:
    current_sum = best_sum = sequence[0]
    start = end = temp_start = 0
    if verbose:
        table = Table(title="Kadane Trace", show_lines=True, expand=True)
        for header in (
            "Idx",
            "Val",
            "Cur Sum",
            "Best Sum",
            "Tmp Start",
            "Start",
            "End",
            "Flag",
        ):
            table.add_column(header, justify="center")
        with Live(table, console=console, refresh_per_second=12):
            for idx, val in enumerate(sequence):
                if idx:
                    if current_sum + val < val:
                        current_sum = val
                        temp_start = idx
                    else:
                        current_sum += val
                    if current_sum > best_sum:
                        best_sum = current_sum
                        start = temp_start
                        end = idx
                flag = "[bold green]★[/bold green]" if best_sum == current_sum and end == idx else ""
                table.add_row(
                    str(idx),
                    str(val),
                    str(current_sum),
                    str(best_sum),
                    str(temp_start),
                    str(start),
                    str(end),
                    flag,
                )
    else:
        for idx in range(1, len(sequence)):
            val = sequence[idx]
            if current_sum + val < val:
                current_sum = val
                temp_start = idx
            else:
                current_sum += val
            if current_sum > best_sum:
                best_sum = current_sum
                start = temp_start
                end = idx
    return best_sum, start, end


if __name__ == "__main__":
    data = [-2, 1, -3, 4, -1, 2, 1, -5, 4]
    verbose_mode = True
    result_sum, result_start, result_end = execute_kadane(data, verbose=verbose_mode)
    if not verbose_mode:
        console.print(
            f"[bold cyan]Maximum:[/bold cyan] {result_sum} "
            f"[bold cyan]Start:[/bold cyan] {result_start} "
            f"[bold cyan]End:[/bold cyan] {result_end}"
        )