import sys
import time
import threading
from typing import Optional

class Spinner:
    """A simple spinner animation like Homebrew's loading indicator."""
    
    def __init__(self, message: str = "Working"):
        self.message = message
        self.spinner_chars = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        self.running = False
        self.thread = None
        
    def start(self):
        """Start the spinner animation in a separate thread."""
        self.running = True
        self.thread = threading.Thread(target=self._spin)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self, final_message: Optional[str] = None):
        """Stop the spinner and optionally show a final message."""
        self.running = False
        if self.thread:
            self.thread.join()
        # Clear the spinner line
        sys.stdout.write('\r' + ' ' * 80 + '\r')
        if final_message:
            print(f"✅ {final_message}")
        
    def _spin(self):
        """The spinning animation loop."""
        i = 0
        while self.running:
            sys.stdout.write(f'\r{self.spinner_chars[i]} {self.message}...')
            sys.stdout.flush()
            time.sleep(0.1)
            i = (i + 1) % len(self.spinner_chars)

def show_progress(current: int, total: int, message: str = "Processing"):
    """Show a simple text progress bar."""
    bar_length = 40
    filled = int(bar_length * current // total)
    bar = '█' * filled + '░' * (bar_length - filled)
    percent = current * 100 // total
    sys.stdout.write(f'\r{message}: |{bar}| {percent}%')
    sys.stdout.flush()

def clear_progress():
    """Clear the current line (useful after progress bars)."""
    sys.stdout.write('\r' + ' ' * 80 + '\r')
    sys.stdout.flush() 