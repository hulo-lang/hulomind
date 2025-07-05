"""Rich-based terminal logger with goreleaser-style output."""

import logging
from typing import Optional, Any, Dict
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.text import Text
from rich import print as rprint
from rich.logging import RichHandler
from contextlib import contextmanager

console = Console()


class HuloLogger:
    """Hulo Knowledge Base Logger with goreleaser-style output."""
    
    def __init__(self):
        self.console = console
        self._progress: Optional[Progress] = None
    
    def info(self, message: str, **kwargs):
        """Print info message."""
        self.console.print(f"[green]•[/green] {message}", **kwargs)
    
    def success(self, message: str, **kwargs):
        """Print success message."""
        # 确保只有一个 ✓ 符号
        if message.startswith("✓"):
            self.console.print(f"[bold green]{message}[/bold green]", **kwargs)
        else:
            self.console.print(f"[bold green]✓[/bold green] {message}", **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Print warning message."""
        self.console.print(f"[yellow]![/yellow] {message}", **kwargs)
    
    def error(self, message: str, **kwargs):
        """Print error message."""
        self.console.print(f"[red]✗[/red] {message}", **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Print debug message."""
        self.console.print(f"[blue]→[/blue] {message}", **kwargs)
    
    def step(self, message: str, **kwargs):
        """Print step message."""
        self.console.print(f"[cyan]→[/cyan] {message}", **kwargs)
    
    def section(self, title: str, content: str = "", **kwargs):
        """Print a section header."""
        self.console.print(f"\n[bold blue]{title}[/bold blue]")
        if content:
            self.console.print(content)
    
    def table(self, title: str, data: Dict[str, Any], **kwargs):
        """Print data in a table format."""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        for key, value in data.items():
            table.add_column(key, style="cyan")
            table.add_row(str(value))
        
        self.console.print(table, **kwargs)
    
    def stats(self, stats: Dict[str, Any]):
        """Print statistics in a nice format."""
        self.console.print("\n[bold blue]Statistics:[/bold blue]")
        
        for key, value in stats.items():
            if isinstance(value, int):
                self.console.print(f"  {key}: [bold green]{value:,}[/bold green]")
            elif isinstance(value, float):
                self.console.print(f"  {key}: [bold green]{value:.2f}[/bold green]")
            else:
                self.console.print(f"  {key}: [bold green]{value}[/bold green]")
    
    @contextmanager
    def progress(self, description: str, total: Optional[int] = None):
        """Context manager for progress bar."""
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        ) as progress:
            task = progress.add_task(description, total=total)
            yield progress
    
    @contextmanager
    def status(self, message: str):
        """Show status with spinner."""
        with self.console.status(f"[bold green]{message}"):
            yield


# Global logger instance
logger = HuloLogger()


def setup_rich_logging():
    """Setup rich logging for the application."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)]
    )


# Convenience functions
def info(message: str, **kwargs):
    """Print info message."""
    logger.info(message, **kwargs)


def success(message: str, **kwargs):
    """Print success message."""
    logger.success(message, **kwargs)


def warning(message: str, **kwargs):
    """Print warning message."""
    logger.warning(message, **kwargs)


def error(message: str, **kwargs):
    """Print error message."""
    logger.error(message, **kwargs)


def debug(message: str, **kwargs):
    """Print debug message."""
    logger.debug(message, **kwargs)


def step(message: str, **kwargs):
    """Print step message."""
    logger.step(message, **kwargs)


def section(title: str, content: str = "", **kwargs):
    """Print section header."""
    logger.section(title, content, **kwargs)


def stats(stats: Dict[str, Any]):
    """Print statistics."""
    logger.stats(stats)


@contextmanager
def progress(description: str, total: Optional[int] = None):
    """Progress bar context manager."""
    with logger.progress(description, total) as task:
        yield task


@contextmanager
def status(message: str):
    """Status spinner context manager."""
    with logger.status(message):
        yield 