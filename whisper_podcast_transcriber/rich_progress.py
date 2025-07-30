import time
import threading
from typing import Optional, List, Dict, Any
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from rich.spinner import Spinner
from rich.layout import Layout
from rich.columns import Columns

console = Console()

class PersistentProgress:
    """Persistent spinner at bottom of screen with current step and time counter."""
    
    def __init__(self):
        self.spinner = Spinner("dots", text="")
        self.total_start_time: float = 0.0
        self.running = False
        self.current_text = ""
        self.live = None
        self.spinner_thread = None
        self.spinner_started = False
        
    def __enter__(self):
        self.total_start_time = time.time()
        self.running = True
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.running = False
        if self.live:
            self.live.stop()
        if self.spinner_thread and self.spinner_thread.is_alive():
            self.spinner_thread.join(timeout=1)
        
    def start_task(self, description: str):
        """Start a new task with persistent spinner."""
        self.current_text = description
        
        # Only start the spinner once, then just update the text
        if not self.spinner_started:
            def update_spinner():
                with Live(self.spinner, console=console, refresh_per_second=1, transient=True) as live:
                    self.live = live
                    while self.running:
                        elapsed = int(time.time() - self.total_start_time)
                        minutes = elapsed // 60
                        seconds = elapsed % 60
                        time_str = f"{minutes:02d}:{seconds:02d}"
                        
                        # Update spinner text with description and time
                        self.spinner.text = f"{self.current_text} {time_str}"
                        time.sleep(1)
            
            # Run the spinner in a daemon thread
            self.spinner_thread = threading.Thread(target=update_spinner, daemon=True)
            self.spinner_thread.start()
            self.spinner_started = True
        else:
            # Update the current text for existing spinner
            self.current_text = description
                
    def complete_task(self, final_message: Optional[str] = None):
        """Complete the current task."""
        if final_message:
            console.print(f"✅ {final_message}")

def print_success_panel(message: str):
    """Print a success message in a nice panel."""
    panel = Panel(
        Text(message, style="bold green"),
        title="🎉 Success",
        border_style="green"
    )
    console.print(panel)

def print_info_panel(message: str):
    """Print an info message in a nice panel."""
    panel = Panel(
        Text(message, style="bold blue"),
        title="ℹ️  Info",
        border_style="blue"
    )
    console.print(panel) 