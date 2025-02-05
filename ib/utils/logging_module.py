"""Custom logging module."""

import logging as default_logging
import sys
from pathlib import Path
from typing import Union

from rich.panel import Panel
from rich.console import Console


class LoggingModule:
    """Custom logging module: save to gfile and print to console."""

    def __init__(self) -> None:
        self.log_file_path = None
        self.logger = default_logging.getLogger("rich_logger")
        self.logger.setLevel(default_logging.DEBUG)
        self.file_handler = None
        self.console = Console(record=True)

    def set_log_file(self, file_dir: Path) -> None:
        """Set the log file dynamically at runtime."""

        self.log_file_path = file_dir / f"logs.txt"

        # Remove previous file handler, if any
        if self.file_handler:
            self.logger.removeHandler(self.file_handler)

        # Create a new file handler
        self.file_handler = default_logging.FileHandler(self.log_file_path)
        self.file_handler.setLevel(default_logging.DEBUG)
        formatter = default_logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        self.file_handler.setFormatter(formatter)
        self.logger.addHandler(self.file_handler)

    def log_to_file(self, message: str, level: str = "INFO") -> None:
        """Log a message to the file."""
        if self.logger and self.file_handler:
            log_method = getattr(self.logger, level.lower(), self.logger.info)
            log_method(message)

    def _write(
        self,
        inputs: Union[str, Panel],
        color: str = "",
        newline: bool = False,
        level: str = "INFO",
    ) -> None:
        """Wrapper around rich.print() and file logging."""
        outputs = inputs
        if color:
            outputs = f"[{color}]{inputs}[/{color}]"
        if newline:
            outputs = "\n" + outputs

        # Log to console if there is one.
        if sys.stdout.isatty():
            self.console.print(outputs)

        # Log to file (strip Rich formatting for file output)
        if isinstance(inputs, Panel):
            # Render Rich objects as plain text
            temp_console = Console()
            with temp_console.capture() as capture:
                temp_console.print(inputs)
            stripped_message = capture.get()

            if inputs.title:
                stripped_message = f"Panel: {inputs.title}\n{stripped_message}"
        else:
            stripped_message = str(inputs)

        self.log_to_file(stripped_message, level)

    def info(self, inputs: str) -> None:
        """Log info messages."""
        self._write(inputs, level="INFO")

    def warning(self, inputs: str) -> None:
        """Log warning messages."""
        self._write(inputs, color="yellow", newline=True, level="WARNING")

    def stage(self, inputs: str) -> None:
        """Log notices."""
        self._write(inputs, color="green", newline=True, level="INFO")

    def panel(self, title: str, inputs: str) -> None:
        """Log panels using _write for consistent handling."""
        pl = Panel(inputs, title=title, expand=False, highlight=True)
        self._write(pl, level="INFO")


logging = LoggingModule()
