
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
                self._render_iteration_header(iteration_count, mid_pointer, current_element, low_boundary, high_boundary)
            
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
        
        self._display_pointer_positions(low, mid, high, len(array))
        
        configuration_table = Table(show_header=True, header_style="bold magenta")
        configuration_table.add_column("Index", style="cyan", no_wrap=True, width=8)
        configuration_table.add_column("Value", style="yellow", width=8)
        configuration_table.add_column("Pointers", style="green", width=15)
        configuration_table.add_column("Region", style="white", width=12)
        
        for position, element in enumerate(array):
            pointer_markers = self._generate_pointer_markers(position, low, mid, high, len(array))
            region_label = self._classify_region(position, low, mid, high, len(array))
            element_style = self._get_element_style(element)
            
            configuration_table.add_row(
                f"[bold]{position}[/bold]",
                f"[bold {element_style}]{element}[/bold {element_style}]",
                pointer_markers,
                region_label
            )
        
        self.console.print(configuration_table)
        self.console.print()
    
    def _display_pointer_positions(self, low, mid, high, array_length):
        pointer_info = Table(show_header=True, header_style="bold cyan", title="Current Pointer Positions")
        pointer_info.add_column("Pointer", style="bold white")
        pointer_info.add_column("Position", style="bold yellow")
        pointer_info.add_column("Status", style="bold green")
        
        pointer_info.add_row("LOW", f"{low}", "Active" if low < array_length else "Out of bounds")
        pointer_info.add_row("MID", f"{mid}", "Active" if mid < array_length else "Out of bounds")
        pointer_info.add_row("HIGH", f"{high}", "Active" if high >= 0 else "Out of bounds")
        
        self.console.print(pointer_info)
    
    def _render_iteration_header(self, iteration, position, value, low, high):
        self.console.print(f"\n[bold yellow]═══ Iteration {iteration} ═══[/bold yellow]")
        self.console.print(f"[bold white]Current MID position:[/bold white] [bold cyan]{position}[/bold cyan]")
        self.console.print(f"[bold white]Element at MID:[/bold white] [bold red]{value}[/bold red]")
        self.console.print(f"[bold white]Boundaries:[/bold white] LOW={low}, HIGH={high}")
    
    def _handle_zero_placement(self, array, low_pos, mid_pos):
        if self.verbose:
            self.console.print(f"  [bold green]→[/bold green] [red]ZERO[/red] detected: Swap positions {low_pos} ↔ {mid_pos}")
            self.console.print(f"  [bold green]→[/bold green] Increment LOW: {low_pos} → {low_pos + 1}")
            self.console.print(f"  [bold green]→[/bold green] Increment MID: {mid_pos} → {mid_pos + 1}")
        array[low_pos], array[mid_pos] = array[mid_pos], array[low_pos]
    
    def _handle_one_placement(self, mid_pos):
        if self.verbose:
            self.console.print(f"  [bold green]→[/bold green] [yellow]ONE[/yellow] detected: Already in correct position")
            self.console.print(f"  [bold green]→[/bold green] Increment MID: {mid_pos} → {mid_pos + 1}")
    
    def _handle_two_placement(self, array, mid_pos, high_pos):
        if self.verbose:
            self.console.print(f"  [bold green]→[/bold green] [blue]TWO[/blue] detected: Swap positions {mid_pos} ↔ {high_pos}")
            self.console.print(f"  [bold green]→[/bold green] Decrement HIGH: {high_pos} → {high_pos - 1}")
            self.console.print(f"  [bold green]→[/bold green] Keep MID at {mid_pos} for re-evaluation")
        array[mid_pos], array[high_pos] = array[high_pos], array[mid_pos]
    
    def _render_current_configuration(self, array, low, mid, high, iteration):
        self.console.print(f"\n[bold magenta]Array State After Iteration {iteration}:[/bold magenta]")
        
        self._display_pointer_positions(low, mid, high, len(array))
        
        state_table = Table(show_header=True, header_style="bold magenta")
        state_table.add_column("Index", style="cyan", no_wrap=True, width=8)
        state_table.add_column("Value", style="yellow", width=8)
        state_table.add_column("Pointers", style="green", width=15)
        state_table.add_column("Region", style="white", width=12)
        
        for position, element in enumerate(array):
            pointer_markers = self._generate_pointer_markers(position, low, mid, high, len(array))
            region_label = self._classify_region(position, low, mid, high, len(array))
            element_style = self._get_element_style(element)
            
            state_table.add_row(
                f"[bold]{position}[/bold]",
                f"[bold {element_style}]{element}[/bold {element_style}]",
                pointer_markers,
                region_label
            )
        
        self.console.print(state_table)
        self.console.print(f"[bold white]Continue condition:[/bold white] MID({mid}) <= HIGH({high}) = {mid <= high}")
        self.console.print()
    
    def _render_completion_state(self, array):
        self.console.print(Panel.fit(
            "[bold green]Dutch National Flag Algorithm Complete[/bold green]",
            border_style="green"
        ))
        
        final_representation = " ".join([
            f"[{self._get_element_style(element)}]{element}[/{self._get_element_style(element)}]"
            for element in array
        ])
        
        self.console.print(f"[bold]Final Sorted Array:[/bold] {final_representation}")
        
        complexity_table = Table(show_header=False, box=None)
        complexity_table.add_column("Metric", style="bold white")
        complexity_table.add_column("Value", style="bold green")
        complexity_table.add_row("Time Complexity:", "O(n)")
        complexity_table.add_row("Space Complexity:", "O(1)")
        complexity_table.add_row("Algorithm Type:", "Three-way partitioning")
        
        self.console.print(complexity_table)
    
    def _generate_pointer_markers(self, position, low, mid, high, array_length):
        markers = []
        
        if position == low and low < array_length:
            markers.append("[bold red]L[/bold red]")
        if position == mid and mid < array_length:
            markers.append("[bold blue]M[/bold blue]")
        if position == high and high >= 0:
            markers.append("[bold green]H[/bold green]")
            
        if markers:
            return " ".join(markers)
        else:
            return "·"
    
    def _classify_region(self, position, low, mid, high, array_length):
        if mid > high:
            if position < low:
                return "[red]0s[/red]"
            elif position <= high:
                return "[yellow]1s[/yellow]"
            else:
                return "[blue]2s[/blue]"
        else:
            if position < low:
                return "[red]0s[/red]"
            elif position < mid:
                return "[yellow]1s[/yellow]"
            elif position <= high:
                return "[white]?[/white]"
            else:
                return "[blue]2s[/blue]"
    
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
        execution_console.print(f"\n[bold cyan]═══ Test Case {dataset_index + 1} ═══[/bold cyan]")
        execution_console.print(f"[bold white]Input:[/bold white] {input_array}")
        
        if input_array:
            algorithm_instance = DutchNationalFlagSorter(verbose=True)
            original_sequence = input_array.copy()
            sorted_result = algorithm_instance.sort_array(input_array)
            
            execution_console.print(f"\n[bold green]✓ RESULT:[/bold green]")
            execution_console.print(f"[bold white]Original:[/bold white] {original_sequence}")
            execution_console.print(f"[bold white]Sorted:  [/bold white] {sorted_result}")
        else:
            execution_console.print("[bold yellow]Empty array - no processing needed[/bold yellow]")
        
        execution_console.print("═" * 80)
        time.sleep(1.0)