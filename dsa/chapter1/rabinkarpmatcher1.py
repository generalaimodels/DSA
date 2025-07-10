from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box
import time

class RabinKarpMatcher:
    def __init__(self, base=256, modulus=1000000007):
        self.base = base
        self.modulus = modulus
        self.console = Console()
        
    def compute_hash(self, string, length):
        hash_value = 0
        for i in range(length):
            hash_value = (hash_value * self.base + ord(string[i])) % self.modulus
        return hash_value
    
    def compute_power(self, length):
        power = 1
        for _ in range(length - 1):
            power = (power * self.base) % self.modulus
        return power
    
    def roll_hash(self, old_hash, old_char, new_char, power):
        old_hash = (old_hash - (ord(old_char) * power) % self.modulus + self.modulus) % self.modulus
        new_hash = (old_hash * self.base + ord(new_char)) % self.modulus
        return new_hash
    
    def verify_match(self, text, pattern, start_index):
        for i in range(len(pattern)):
            if text[start_index + i] != pattern[i]:
                return False
        return True
    
    def search(self, text, pattern, verbose=False):
        if len(pattern) > len(text):
            return []
            
        matches = []
        pattern_length = len(pattern)
        text_length = len(text)
        
        pattern_hash = self.compute_hash(pattern, pattern_length)
        window_hash = self.compute_hash(text, pattern_length)
        power = self.compute_power(pattern_length)
        
        if verbose:
            self.display_initialization(text, pattern, pattern_hash, window_hash)
        
        for i in range(text_length - pattern_length + 1):
            if verbose:
                self.display_step(text, pattern, i, window_hash, pattern_hash)
            
            if window_hash == pattern_hash:
                if verbose:
                    self.display_hash_match(i)
                
                if self.verify_match(text, pattern, i):
                    matches.append(i)
                    if verbose:
                        self.display_verified_match(i)
                else:
                    if verbose:
                        self.display_collision(i)
            
            if i < text_length - pattern_length:
                window_hash = self.roll_hash(
                    window_hash, 
                    text[i], 
                    text[i + pattern_length], 
                    power
                )
                
                if verbose:
                    self.display_rolling_hash(i, text[i], text[i + pattern_length], window_hash)
        
        if verbose:
            self.display_results(matches)
        
        return matches
    
    def display_initialization(self, text, pattern, pattern_hash, window_hash):
        self.console.print(Panel.fit(
            f"[bold blue]Rabin-Karp Algorithm Initialization[/bold blue]\n"
            f"Text: [green]{text}[/green]\n"
            f"Pattern: [yellow]{pattern}[/yellow]\n"
            f"Base: {self.base}, Modulus: {self.modulus}\n"
            f"Pattern Hash: [cyan]{pattern_hash}[/cyan]\n"
            f"Initial Window Hash: [cyan]{window_hash}[/cyan]",
            box=box.ROUNDED
        ))
        time.sleep(0.5)
    
    def display_step(self, text, pattern, position, window_hash, pattern_hash):
        highlighted_text = Text()
        for i, char in enumerate(text):
            if position <= i < position + len(pattern):
                highlighted_text.append(char, style="bold red on white")
            else:
                highlighted_text.append(char, style="dim")
        
        table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE)
        table.add_column("Position", justify="center")
        table.add_column("Window", justify="center")
        table.add_column("Window Hash", justify="center")
        table.add_column("Pattern Hash", justify="center")
        table.add_column("Match?", justify="center")
        
        window = text[position:position + len(pattern)]
        match_status = "✓" if window_hash == pattern_hash else "✗"
        match_color = "green" if window_hash == pattern_hash else "red"
        
        table.add_row(
            str(position),
            f"[yellow]{window}[/yellow]",
            f"[cyan]{window_hash}[/cyan]",
            f"[cyan]{pattern_hash}[/cyan]",
            f"[{match_color}]{match_status}[/{match_color}]"
        )
        
        self.console.print(f"\n[bold]Step {position + 1}:[/bold]")
        self.console.print(highlighted_text)
        self.console.print(table)
        time.sleep(0.3)
    
    def display_hash_match(self, position):
        self.console.print(f"[bold green]Hash match found at position {position}! Verifying...[/bold green]")
        time.sleep(0.2)
    
    def display_verified_match(self, position):
        self.console.print(f"[bold bright_green]✓ Verified match at position {position}[/bold bright_green]")
        time.sleep(0.2)
    
    def display_collision(self, position):
        self.console.print(f"[bold red]✗ Hash collision at position {position} - not a real match[/bold red]")
        time.sleep(0.2)
    
    def display_rolling_hash(self, position, old_char, new_char, new_hash):
        self.console.print(
            f"[dim]Rolling hash: removing '[red]{old_char}[/red]', "
            f"adding '[green]{new_char}[/green]' → new hash: [cyan]{new_hash}[/cyan][/dim]"
        )
        time.sleep(0.1)
    
    def display_results(self, matches):
        if matches:
            self.console.print(Panel.fit(
                f"[bold green]Pattern found at positions: {matches}[/bold green]\n"
                f"Total matches: {len(matches)}",
                title="[bold]Search Results[/bold]",
                box=box.DOUBLE
            ))
        else:
            self.console.print(Panel.fit(
                "[bold red]No matches found[/bold red]",
                title="[bold]Search Results[/bold]",
                box=box.DOUBLE
            ))

def demonstrate_rabin_karp():
    matcher = RabinKarpMatcher()
    
    test_cases = [
        ("ABABCABABA", "ABA"),
        ("GEEKSFORGEEKS", "GEEK"),
        ("AABAACAADAABAABA", "AABA"),
        ("ABCDEFGHIJKLMNOP", "XYZ")
    ]
    
    for text, pattern in test_cases:
        console = Console()
        console.print(f"\n[bold magenta]{'='*60}[/bold magenta]")
        console.print(f"[bold]Searching for pattern '{pattern}' in text '{text}'[/bold]")
        console.print(f"[bold magenta]{'='*60}[/bold magenta]")
        
        matches = matcher.search(text, pattern, verbose=True)
        
        console.print(f"\n[bold yellow]Summary:[/bold yellow]")
        console.print(f"Pattern: [yellow]{pattern}[/yellow]")
        console.print(f"Text: [green]{text}[/green]")
        console.print(f"Matches found: [cyan]{len(matches)}[/cyan]")
        if matches:
            console.print(f"Positions: [bright_green]{matches}[/bright_green]")
        
        time.sleep(1)

if __name__ == "__main__":
    demonstrate_rabin_karp()