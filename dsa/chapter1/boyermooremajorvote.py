from typing import List, Optional, Tuple
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
import time

class BoyerMooreMajorityVote:
    def __init__(self, verbose: bool = False):
        self.console = Console() if verbose else None
        self.verbose = verbose
        
    def find_majority_element(self, nums: List[int]) -> Optional[int]:
        if not nums:
            return None
            
        candidate = self._find_candidate(nums)
        
        if self._verify_candidate(nums, candidate):
            return candidate
        return None
    
    def _find_candidate(self, nums: List[int]) -> int:
        candidate = nums[0]
        count = 1
        
        if self.verbose:
            self._display_phase_header("Phase 1: Finding Candidate")
            self._create_state_table(0, nums[0], candidate, count, "Initialize")
        
        for i in range(1, len(nums)):
            if self.verbose:
                time.sleep(0.3)
                
            if nums[i] == candidate:
                count += 1
                action = f"[green]Vote FOR[/green] candidate {candidate}"
            else:
                count -= 1
                action = f"[red]Vote AGAINST[/red] candidate {candidate}"
                
            if count == 0:
                candidate = nums[i]
                count = 1
                action += f" → [yellow]New candidate: {candidate}[/yellow]"
                
            if self.verbose:
                self._create_state_table(i, nums[i], candidate, count, action)
        
        if self.verbose:
            self._display_candidate_result(candidate)
            
        return candidate
    
    def _verify_candidate(self, nums: List[int], candidate: int) -> bool:
        if self.verbose:
            self._display_phase_header("Phase 2: Verifying Candidate")
            
        count = 0
        majority_threshold = len(nums) // 2
        
        for i, num in enumerate(nums):
            if self.verbose:
                time.sleep(0.2)
                
            if num == candidate:
                count += 1
                
            if self.verbose:
                is_match = "✓" if num == candidate else "✗"
                progress = f"{count}/{len(nums)}"
                self._create_verification_table(i, num, candidate, is_match, count, progress)
        
        is_majority = count > majority_threshold
        
        if self.verbose:
            self._display_verification_result(candidate, count, majority_threshold, is_majority)
            
        return is_majority
    
    def _display_phase_header(self, phase_title: str):
        panel = Panel(
            f"[bold cyan]{phase_title}[/bold cyan]",
            border_style="cyan",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def _create_state_table(self, index: int, current_element: int, candidate: int, count: int, action: str):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Index", style="dim", width=8)
        table.add_column("Element", style="bold", width=10)
        table.add_column("Candidate", style="green", width=12)
        table.add_column("Count", style="yellow", width=8)
        table.add_column("Action", width=40)
        
        table.add_row(
            str(index),
            str(current_element),
            str(candidate),
            str(count),
            action
        )
        
        self.console.print(table)
    
    def _create_verification_table(self, index: int, element: int, candidate: int, match: str, count: int, progress: str):
        table = Table(show_header=True, header_style="bold blue")
        table.add_column("Index", style="dim", width=8)
        table.add_column("Element", style="bold", width=10)
        table.add_column("Candidate", style="green", width=12)
        table.add_column("Match", style="cyan", width=8)
        table.add_column("Count", style="yellow", width=8)
        table.add_column("Progress", width=12)
        
        table.add_row(
            str(index),
            str(element),
            str(candidate),
            match,
            str(count),
            progress
        )
        
        self.console.print(table)
    
    def _display_candidate_result(self, candidate: int):
        result_text = Text(f"Candidate Found: {candidate}", style="bold green")
        panel = Panel(result_text, border_style="green", padding=(1, 2))
        self.console.print(panel)
    
    def _display_verification_result(self, candidate: int, count: int, threshold: int, is_majority: bool):
        status = "MAJORITY ELEMENT" if is_majority else "NOT MAJORITY"
        style = "bold green" if is_majority else "bold red"
        
        result_text = f"Candidate: {candidate}\nCount: {count}\nThreshold: >{threshold}\nResult: {status}"
        panel = Panel(result_text, border_style="green" if is_majority else "red", title="Verification Result", title_align="center")
        self.console.print(panel, style=style)

def run_boyer_moore_tests():
    test_cases = [
        ([3, 2, 3], "Simple majority case"),
        ([2, 2, 1, 1, 1, 2, 2], "Mixed elements with majority"),
        ([1], "Single element"),
        ([1, 2, 3, 4], "No majority element"),
        ([1, 1, 1, 2, 2], "Clear majority")
    ]
    
    console = Console()
    
    for i, (nums, description) in enumerate(test_cases, 1):
        console.print(f"\n[bold blue]Test Case {i}: {description}[/bold blue]")
        console.print(f"Input: {nums}")
        
        verbose_mode = i <= 2
        solver = BoyerMooreMajorityVote(verbose=verbose_mode)
        result = solver.find_majority_element(nums)
        
        if not verbose_mode:
            console.print(f"Result: {result}")
        
        console.print("-" * 60)

if __name__ == "__main__":
    run_boyer_moore_tests()