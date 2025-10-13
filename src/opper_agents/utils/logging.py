"""
Agent logging system with modular, extensible loggers.

This module provides a base AgentLogger class and implementations
for different logging styles (simple, rich, custom).
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, List
from contextlib import contextmanager


class AgentLogger(ABC):
    """
    Base class for agent loggers.

    Users can extend this class to create custom loggers
    with their own formatting and output destinations.
    """

    @abstractmethod
    def log_iteration(self, iteration: int, max_iterations: int) -> None:
        """Log the start of an iteration."""
        pass

    @abstractmethod
    @contextmanager
    def log_thinking(self) -> Any:
        """
        Context manager for thinking phase.
        Can show spinners or progress indicators.
        """
        yield

    @abstractmethod
    def log_thought(self, reasoning: str, tool_count: int) -> None:
        """Log the agent's reasoning and planned actions."""
        pass

    @abstractmethod
    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """Log a tool call."""
        pass

    @abstractmethod
    def log_tool_result(
        self, tool_name: str, success: bool, result: Any, error: Optional[str] = None
    ) -> None:
        """Log the result of a tool execution."""
        pass

    @abstractmethod
    def log_memory_read(self, keys: List[str]) -> None:
        """Log memory read operation."""
        pass

    @abstractmethod
    def log_memory_loaded(self, data: Dict[str, Any]) -> None:
        """Log loaded memory data."""
        pass

    @abstractmethod
    def log_memory_write(self, keys: List[str]) -> None:
        """Log memory write operation."""
        pass

    @abstractmethod
    def log_final_result(self) -> None:
        """Log that the final result is being generated."""
        pass

    @abstractmethod
    def log_warning(self, message: str) -> None:
        """Log a warning message."""
        pass

    @abstractmethod
    def log_error(self, message: str) -> None:
        """Log an error message."""
        pass


class SimpleLogger(AgentLogger):
    """
    Simple print-based logger.
    Minimal formatting, good for basic debugging.
    """

    def log_iteration(self, iteration: int, max_iterations: int) -> None:
        print(f"\n--- Iteration {iteration}/{max_iterations} ---")

    @contextmanager
    def log_thinking(self) -> Any:
        yield

    def log_thought(self, reasoning: str, tool_count: int) -> None:
        print(f"Reasoning: {reasoning}")
        print(f"Tool calls: {tool_count}")

    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        print(f"[TOOL CALL] - {tool_name} with {parameters}")

    def log_tool_result(
        self, tool_name: str, success: bool, result: Any, error: Optional[str] = None
    ) -> None:
        status = "SUCCESS" if success else "FAILED"
        if error:
            print(f"---> [RESULT] Status: {status} Error: {error}\n")
        else:
            print(f"---> [RESULT] Status: {status} Result: {result}\n")

    def log_memory_read(self, keys: List[str]) -> None:
        print(f"Loading memory keys: {keys}")

    def log_memory_loaded(self, data: Dict[str, Any]) -> None:
        print(f"Loaded memory: {data}")

    def log_memory_write(self, keys: List[str]) -> None:
        print(f"Writing to memory: {keys}")

    def log_final_result(self) -> None:
        print("\n[GENERATING FINAL RESULT]\n")

    def log_warning(self, message: str) -> None:
        print(f"Warning: {message}")

    def log_error(self, message: str) -> None:
        print(f"Error: {message}")


class RichLogger(AgentLogger):
    """
    Rich-based logger with nice formatting and spinners.
    Uses Rich library for beautiful console output with Opper brand colors.

    Opper Color Palette:
    - Water Leaf (#8CF0DC) - Turquoise/cyan for success and headers
    - Savoy Purple (#3C3CAF) - Purple for tools and actions
    - Cotton Candy Coral (#FFB186) - Coral for errors
    - Translucent Silk (#FFD7D7) - Light pink for warnings
    """

    def __init__(self) -> None:
        try:
            from rich.console import Console

            self.console: Any = Console()
            self._status = None
            self._rich_available = True

            # Opper brand colors
            self.color_water_leaf = "#8CF0DC"  # Turquoise
            self.color_savoy_purple = "#3C3CAF"  # Purple
            self.color_coral = "#FFB186"  # Coral (from Cotton Candy)
            self.color_silk = "#FFD7D7"  # Light pink
            self.color_cyan = "#8CECF2"  # Cyan (from Cotton Candy)

        except ImportError:  # pragma: no cover
            # Fallback if Rich is not available
            self._rich_available = False
            self.console = None

    def log_iteration(self, iteration: int, max_iterations: int) -> None:
        if not self._rich_available:
            print(f"\n--- Iteration {iteration}/{max_iterations} ---")
            return

        self.console.print()
        self.console.rule(
            f"[bold {self.color_cyan}]Iteration {iteration}/{max_iterations}[/bold {self.color_cyan}]",
            style=self.color_cyan,
        )

    @contextmanager
    def log_thinking(self) -> Any:
        if not self._rich_available:
            yield
            return

        with self.console.status(
            f"[bold {self.color_savoy_purple}]Thinking...", spinner="dots"
        ) as status:
            self._status = status
            yield
            self._status = None

    def log_thought(self, reasoning: str, tool_count: int) -> None:
        if not self._rich_available:
            print(f"Reasoning: {reasoning}")
            print(f"Tool calls: {tool_count}")
            return

        from rich.panel import Panel

        # Show reasoning in a panel with Opper purple
        self.console.print(
            Panel(
                f"[white]{reasoning}[/white]",
                title=f"[bold {self.color_savoy_purple}]Reasoning[/bold {self.color_savoy_purple}]",
                border_style=self.color_savoy_purple,
            )
        )

        # Show tool count
        if tool_count > 0:
            self.console.print(
                f"[{self.color_cyan}]Tool calls planned:[/{self.color_cyan}] {tool_count}"
            )
        else:
            self.console.print(
                f"[{self.color_water_leaf}]No tool calls - ready for final result[/{self.color_water_leaf}]"
            )

    def log_tool_call(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        if not self._rich_available:
            print(f"[TOOL CALL] - {tool_name} with {parameters}")
            return

        # Format parameters nicely with Opper purple for tools
        params_str = ", ".join(f"{k}={v}" for k, v in parameters.items())
        self.console.print(
            f"  [bold {self.color_savoy_purple}]>>[/bold {self.color_savoy_purple}] Calling [{self.color_savoy_purple}]{tool_name}[/{self.color_savoy_purple}]([dim]{params_str}[/dim])"
        )

    def log_tool_result(
        self, tool_name: str, success: bool, result: Any, error: Optional[str] = None
    ) -> None:
        if not self._rich_available:
            status = "SUCCESS" if success else "FAILED"
            if error:
                print(f"---> [RESULT] Status: {status} Error: {error}\n")
            else:
                print(f"---> [RESULT] Status: {status} Result: {result}\n")
            return

        if success:
            # Truncate long results and show with Opper turquoise for success
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."

            self.console.print(
                f"  [bold {self.color_water_leaf}]OK[/bold {self.color_water_leaf}] [dim]{result_str}[/dim]"
            )
        else:
            # Use Opper coral for errors
            self.console.print(
                f"  [bold {self.color_coral}]FAILED[/bold {self.color_coral}] [{self.color_coral}]{error}[/{self.color_coral}]"
            )

    def log_memory_read(self, keys: List[str]) -> None:
        if not self._rich_available:
            print(f"Loading memory keys: {keys}")
            return

        keys_str = ", ".join(keys)
        self.console.print(
            f"[{self.color_cyan}]Reading memory:[/{self.color_cyan}] [dim]{keys_str}[/dim]"
        )

    def log_memory_loaded(self, data: Dict[str, Any]) -> None:
        if not self._rich_available:
            print(f"Loaded memory: {data}")
            return

        from rich.table import Table

        if not data:
            self.console.print("[dim]No memory data loaded[/dim]")
            return

        # Use Opper colors for memory table
        table = Table(
            show_header=True,
            header_style=f"bold {self.color_savoy_purple}",
            box=None,
        )
        table.add_column("Key", style=self.color_cyan)
        table.add_column("Value", style="white")

        for key, value in data.items():
            value_str = str(value)
            if len(value_str) > 100:
                value_str = value_str[:100] + "..."
            table.add_row(key, value_str)

        self.console.print(table)

    def log_memory_write(self, keys: List[str]) -> None:
        if not self._rich_available:
            print(f"Writing to memory: {keys}")
            return

        keys_str = ", ".join(keys)
        self.console.print(
            f"[{self.color_cyan}]Writing to memory:[/{self.color_cyan}] [dim]{keys_str}[/dim]"
        )

    def log_final_result(self) -> None:
        if not self._rich_available:
            print("\n[GENERATING FINAL RESULT]\n")
            return

        self.console.print()
        self.console.rule(
            f"[bold {self.color_water_leaf}]Generating Final Result",
            style=self.color_water_leaf,
        )

    def log_warning(self, message: str) -> None:
        if not self._rich_available:
            print(f"Warning: {message}")
            return

        # Use Opper light pink for warnings
        self.console.print(f"[{self.color_silk}]Warning:[/{self.color_silk}] {message}")

    def log_error(self, message: str) -> None:
        if not self._rich_available:
            print(f"Error: {message}")
            return

        # Use Opper coral for errors
        self.console.print(
            f"[bold {self.color_coral}]Error:[/bold {self.color_coral}] {message}"
        )
