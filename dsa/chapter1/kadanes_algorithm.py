from typing import List, Tuple, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
from rich.text import Text
from rich.layout import Layout
from rich.live import Live
import time

class KadaneMaximumSubarray:
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        self.current_sum = 0
        self.maximum_sum = float('-inf')
        self.optimal_start = 0
        self.optimal_end = 0
        self.temporary_start = 0
        
    def execute_algorithm(self, elements: List[int]) -> Tuple[int, int, int]:
        if not elements:
            return 0, 0, 0
            
        self._initialize_state(elements)
        
        for index in range(len(elements)):
            self._process_current_element(elements, index)
            
        return self.maximum_sum, self.optimal_start, self.optimal_end
    
    def _initialize_state(self, elements: List[int]) -> None:
        self.current_sum = 0
        self.maximum_sum = float('-inf')
        self.optimal_start = 0
        self.optimal_end = 0
        self.temporary_start = 0
        
        if self.verbose:
            self._render_initialization_display(elements)
    
    def _process_current_element(self, elements: List[int], index: int) -> None:
        previous_sum = self.current_sum
        
        if self.current_sum < 0:
            self.current_sum = elements[index]
            self.temporary_start = index
        else:
            self.current_sum += elements[index]
        
        if self.current_sum > self.maximum_sum:
            self.maximum_sum = self.current_sum
            self.optimal_start = self.temporary_start
            self.optimal_end = index
            
        if self.verbose:
            self._render_processing_step(elements, index, previous_sum)
    
    def _render_initialization_display(self, elements: List[int]) -> None:
        header_panel = Panel.fit(
            "[bold cyan]Kadane's Maximum Subarray Algorithm[/bold cyan]\n[dim]Optimal O(n) Time | O(1) Space[/dim]",
            border_style="blue"
        )
        self.console.print(header_panel)
        
        input_table = Table(title="Input Array", show_header=True, header_style="bold magenta")
        input_table.add_column("Index", justify="center", style="cyan")
        input_table.add_column("Value", justify="center", style="yellow")
        
        for idx, value in enumerate(elements):
            input_table.add_row(str(idx), str(value))
        
        self.console.print(input_table)
        self.console.print(f"[bold green]Initial State:[/bold green] current_sum=0, max_sum=-∞\n")
    
    def _render_processing_step(self, elements: List[int], current_index: int, previous_sum: int) -> None:
        step_number = current_index + 1
        current_element = elements[current_index]
        
        self.console.print(f"[bold white]Step {step_number}:[/bold white] Processing element [bold red]{current_element}[/bold red] at index [bold cyan]{current_index}[/bold cyan]")
        
        array_visualization = Text()
        for idx, value in enumerate(elements):
            if idx == current_index:
                array_visualization.append(f"[{value}]", style="bold black on yellow")
            elif self.optimal_start <= idx <= self.optimal_end:
                array_visualization.append(f" {value} ", style="bold green on black")
            else:
                array_visualization.append(f" {value} ", style="dim white")
            
            if idx < len(elements) - 1:
                array_visualization.append(" ", style="white")
        
        self.console.print(array_visualization)
        
        decision_logic = self._determine_decision_logic(previous_sum, current_element)
        algorithm_state = self._generate_algorithm_state_table(decision_logic)
        
        self.console.print(algorithm_state)
        self.console.print()
    
    def _determine_decision_logic(self, previous_sum: int, current_element: int) -> str:
        if previous_sum < 0:
            return f"Reset: previous_sum({previous_sum}) < 0, start new subarray"
        else:
            return f"Extend: previous_sum({previous_sum}) ≥ 0, continue current subarray"
    
    def _generate_algorithm_state_table(self, decision_logic: str) -> Table:
        state_table = Table(show_header=True, header_style="bold cyan", border_style="blue")
        state_table.add_column("Variable", style="bold white")
        state_table.add_column("Current Value", justify="center", style="yellow")
        state_table.add_column("Decision Logic", style="italic green")
        
        state_table.add_row("current_sum", str(self.current_sum), decision_logic)
        state_table.add_row("maximum_sum", str(self.maximum_sum), 
                           "Updated" if self.current_sum == self.maximum_sum else "Unchanged")
        state_table.add_row("optimal_range", f"[{self.optimal_start}:{self.optimal_end}]", 
                           "Best subarray indices")
        
        return state_table
    
    def generate_final_results(self) -> None:
        if self.verbose:
            final_panel = Panel(
                f"[bold green]Maximum Sum:[/bold green] {self.maximum_sum}\n"
                f"[bold blue]Subarray Range:[/bold blue] [{self.optimal_start}:{self.optimal_end}]\n"
                f"[bold yellow]Time Complexity:[/bold yellow] O(n)\n"
                f"[bold yellow]Space Complexity:[/bold yellow] O(1)",
                title="🎯 Algorithm Results",
                border_style="green",
                expand=False
            )
            self.console.print(final_panel)

class KadaneAlgorithmRunner:
    def __init__(self):
        self.console = Console()
    
    def execute_with_dataset(self, test_cases: List[Tuple[str, List[int], bool]]) -> None:
        for case_name, data, verbose_mode in test_cases:
            self._execute_single_case(case_name, data, verbose_mode)
    
    def _execute_single_case(self, case_name: str, data: List[int], verbose_mode: bool) -> None:
        self.console.print(f"\n[bold magenta]═══ {case_name} ═══[/bold magenta]")
        
        algorithm = KadaneMaximumSubarray(verbose=verbose_mode)
        max_sum, start_idx, end_idx = algorithm.execute_algorithm(data)
        
        algorithm.generate_final_results()
        
        if not verbose_mode:
            subarray = data[start_idx:end_idx + 1] if data else []
            result_table = Table(show_header=False, border_style="green")
            result_table.add_column("Metric", style="bold")
            result_table.add_column("Value", style="cyan")
            
            result_table.add_row("Maximum Sum", str(max_sum))
            result_table.add_row("Optimal Subarray", str(subarray))
            result_table.add_row("Index Range", f"[{start_idx}:{end_idx}]")
            
            self.console.print(result_table)

if __name__ == "__main__":
    test_datasets = [
        ("Classic Mixed Array", [-2, 1, -3, 4, -1, 2, 1, -5, 4], True),
        ("All Negative Numbers", [-5, -2, -8, -1], False),
        ("All Positive Numbers", [1, 2, 3, 4, 5], False),
        ("Single Element", [42], False),
        ("Alternating Pattern", [5, -3, 8, -2, 4, -1], True),
        ("Large Negative Start", [-10, 2, 3, -1, 4], False),
        ("Edge Case Empty", [], False)
    ]
    
    runner = KadaneAlgorithmRunner()
    runner.execute_with_dataset(test_datasets)