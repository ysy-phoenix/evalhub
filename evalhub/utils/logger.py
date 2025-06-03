"""Modern logging utility for EvalHub.

This module provides a rich, colorful logging interface with support for
console and file outputs, configurable log levels, and structured logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler
from rich.theme import Theme

# Define custom theme for various log levels
RICH_THEME = Theme(
    {
        "info": "bold cyan",
        "warning": "bold yellow",
        "error": "bold red",
        "critical": "bold white on red",
        "success": "bold green",
        "debug": "bold magenta",
    }
)

# Console for standard output
console = Console(theme=RICH_THEME)

# Console for error output
error_console = Console(stderr=True, theme=RICH_THEME)


class EvalHubLogger:
    """Modern logger for EvalHub with rich formatting and multi-output support."""

    def __init__(
        self,
        name: str = "evalhub",
        level: int | str = "INFO",
        log_file: str | None = None,
        log_dir: str | None = None,
    ):
        r"""Initialize the logger.

        Args:
            name: Name of the logger.
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
            log_file: Specific log file name. If None but log_dir is specified,
            a timestamped file name will be generated.
            log_dir: Directory to store log files. If None, no file logging is used.

        """
        # Set up the logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        # Remove any existing handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        # Set up console handler with rich formatting
        console_handler = RichHandler(
            console=console,
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
            markup=True,
            omit_repeated_times=False,
        )
        console_format = "%(message)s"
        console_formatter = logging.Formatter(console_format)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        # Set up file handler if requested
        if log_dir:
            log_dir_path = Path(log_dir)
            log_dir_path.mkdir(exist_ok=True, parents=True)

            if not log_file:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_file = f"{name}_{timestamp}.log"

            log_file_path = log_dir_path / log_file

            file_handler = logging.FileHandler(log_file_path)
            file_format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
            file_formatter = logging.Formatter(file_format)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)

            self.info(f"Log file created at: {log_file_path.absolute()}")

    def debug(self, message: str, **kwargs):
        r"""Log a debug message, optionally with additional data."""
        self._log("debug", message, **kwargs)

    def info(self, message: str, **kwargs):
        r"""Log an info message, optionally with additional data."""
        self._log("info", message, **kwargs)

    def warning(self, message: str, **kwargs):
        r"""Log a warning message, optionally with additional data."""
        self._log("warning", message, **kwargs)

    def error(self, message: str, exc_info=False, **kwargs):
        r"""Log an error message, optionally with exception info and additional data."""
        self._log("error", message, exc_info=exc_info, **kwargs)

    def critical(self, message: str, exc_info=True, **kwargs):
        r"""Log a critical message, with exception info and additional data."""
        self._log("critical", message, exc_info=exc_info, **kwargs)

    def success(self, message: str, **kwargs):
        r"""Log a success message (INFO level with success formatting)."""
        self.info(f"[success]{message}[/success]", **kwargs)

    def _log(self, level: str, message: str, exc_info=False, **kwargs):
        r"""Handle logging with optional structured data.

        Args:
            level: Log level name
            message: Log message
            exc_info: Whether to include exception information
            **kwargs: Additional structured data to include in the log

        """
        # If we have additional data, format it as key=value pairs
        if kwargs:
            extra_data = " ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {extra_data}"

        # Log with appropriate level
        log_method = getattr(self.logger, level)
        log_method(message, exc_info=exc_info)


# Create default logger instance
logger = EvalHubLogger(
    name="evalhub",
    level=os.environ.get("LOG_LEVEL", "INFO"),
    log_dir=os.environ.get("LOG_DIR", None),
)


def get_logger(name: str, **kwargs) -> EvalHubLogger:
    r"""Get a custom named logger.

    Args:
        name: Logger name
        **kwargs: Additional configuration options for the logger

    Returns:
        Configured logger instance

    """
    return EvalHubLogger(name=name, **kwargs)
