from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import time

class DutchNationalFlagSorter:
    def __init__(self, verbose=False):
        self.console = Console()
        self.verbose = verbose
        
    def sort_array(self, array):
        if not array:
            return array
            
        low_boundary = 0
        mid_pointer = 0
        high_boundary = len(array) - 1
        iteration_count = 0
        
        if self.verbose:
            self._render_initial_configuration(array, low_boundary, mid_pointer, high_boundary)
        
        while mid_pointer <= high_boundary:
            iteration_count += 1
            current_element = array[mid_pointer]
            
            if self.verbose:
                self._render_iteration_header(iteration_count, mid_pointer, current_element)
            
            if current_element == 0:
                self._handle_zero_placement(array, low_boundary, mid_pointer)
                low_boundary += 1
                mid_pointer += 1
            elif current_element == 1:
                self._handle_one_placement(mid_pointer)
                mid_pointer += 1
            else:
                self._handle_two_placement(array, mid_pointer, high_boundary)
                high_boundary -= 1
            
            if self.verbose:
                self._render_current_configuration(array, low_boundary, mid_pointer, high_boundary, iteration_count)
                time.sleep(0.8)
        
        if self.verbose:
            self._render_completion_state(array)
        
        return array
    
    def _render_initial_configuration(self, array, low, mid, high):
        self.console.print(Panel.fit(
            "[bold blue]Dutch National Flag Algorithm Initialization[/bold blue]",
            border_style="blue"
        ))
        
        configuration_table = Table(show_header=True, header_style="bold magenta")
        configuration_table.add_column("Position", style="cyan", no_wrap=True)
        configuration_table.add_column("Element", style="yellow")
        configuration_table.add_column("Pointer Status", style="green")
        configuration_table.add_column("Region Classification", style="white")
        
        for position, element in enumerate(array):
            pointer_markers = self._generate_pointer_markers(position, low, mid, high)
            region_label = self._classify_region(position, low, mid, high)
            element_style = self._get_element_style(element)
            
            configuration_table.add_row(
                str(position),
                f"[bold {element_style}]{element}[/bold {element_style}]",
                pointer_markers,
                region_label
            )
        
        self.console.print(configuration_table)
        self.console.print()
    
    def _render_iteration_header(self, iteration, position, value):
        self.console.print(f"[bold yellow]Iteration {iteration}:[/bold yellow] Evaluating element [bold red]{value}[/bold red] at position [bold cyan]{position}[/bold cyan]")
    
    def _handle_zero_placement(self, array, low_pos, mid_pos):
        if self.verbose:
            self.console.print(f"  → Zero detected: Executing swap between positions {low_pos} ↔ {mid_pos}")
            self.console.print(f"  → Advancing both LOW and MID boundaries")
        array[low_pos], array[mid_pos] = array[mid_pos], array[low_pos]
    
    def _handle_one_placement(self, mid_pos):
        if self.verbose:
            self.console.print(f"  → One detected: Element correctly positioned, advancing MID pointer")
    
    def _handle_two_placement(self, array, mid_pos, high_pos):
        if self.verbose:
            self.console.print(f"  → Two detected: Executing swap between positions {mid_pos} ↔ {high_pos}")
            self.console.print(f"  → Contracting HIGH boundary, MID remains for re-evaluation")
        array[mid_pos], array[high_pos] = array[high_pos], array[mid_pos]
    
    def _render_current_configuration(self, array, low, mid, high, iteration):
        state_table = Table(show_header=True, header_style="bold magenta", title=f"State After Iteration {iteration}")
        state_table.add_column("Position", style="cyan", no_wrap=True)
        state_table.add_column("Element", style="yellow")
        state_table.add_column("Pointer Status", style="green")
        state_table.add_column("Region Classification", style="white")
        
        for position, element in enumerate(array):
            pointer_markers = self._generate_pointer_markers(position, low, mid, high)
            region_label = self._classify_region(position, low, mid, high)
            element_style = self._get_element_style(element)
            
            state_table.add_row(
                str(position),
                f"[bold {element_style}]{element}[/bold {element_style}]",
                pointer_markers,
                region_label
            )
        
        self.console.print(state_table)
        self.console.print()
    
    def _render_completion_state(self, array):
        self.console.print(Panel.fit(
            "[bold green]Algorithm Execution Complete[/bold green]",
            border_style="green"
        ))
        
        final_representation = " ".join([
            f"[{self._get_element_style(element)}]{element}[/{self._get_element_style(element)}]"
            for element in array
        ])
        
        self.console.print(f"[bold]Final Sorted Sequence:[/bold] {final_representation}")
        
        complexity_table = Table(show_header=False, box=None)
        complexity_table.add_column("Metric", style="bold white")
        complexity_table.add_column("Complexity", style="bold green")
        complexity_table.add_row("Time Complexity:", "O(n)")
        complexity_table.add_row("Space Complexity:", "O(1)")
        complexity_table.add_row("Algorithm Class:", "Three-way partitioning")
        
        self.console.print(complexity_table)
    
    def _generate_pointer_markers(self, position, low, mid, high):
        markers = []
        if position == low:
            markers.append("[red]LOW[/red]")
        if position == mid:
            markers.append("[blue]MID[/blue]")
        if position == high:
            markers.append("[green]HIGH[/green]")
        return " ".join(markers) if markers else ""
    
    def _classify_region(self, position, low, mid, high):
        if position < low:
            return "[red]Zeros Region[/red]"
        elif position < mid:
            return "[yellow]Ones Region[/yellow]"
        elif position <= high:
            return "[white]Unprocessed[/white]"
        else:
            return "[blue]Twos Region[/blue]"
    
    def _get_element_style(self, element):
        style_mapping = {0: "red", 1: "yellow", 2: "blue"}
        return style_mapping.get(element, "white")

if __name__ == "__main__":
    test_datasets = [
        [2, 0, 2, 1, 1, 0],
        [2, 0, 1],
        [0, 0, 0],
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 0, 0, 1, 1, 0, 2],
        [0, 1, 2, 0, 1, 2, 1, 0],
        [1, 0, 2, 1, 0, 2],
        []
    ]
    
    execution_console = Console()
    
    for dataset_index, input_array in enumerate(test_datasets):
        execution_console.print(f"\n[bold cyan]Dataset {dataset_index + 1}:[/bold cyan] {input_array}")
        
        algorithm_instance = DutchNationalFlagSorter(verbose=True)
        original_sequence = input_array.copy()
        sorted_result = algorithm_instance.sort_array(input_array)
        
        execution_console.print(f"[bold]Input Sequence:[/bold] {original_sequence}")
        execution_console.print(f"[bold]Output Sequence:[/bold] {sorted_result}")
        execution_console.print("═" * 80)
        
        time.sleep(1.5)