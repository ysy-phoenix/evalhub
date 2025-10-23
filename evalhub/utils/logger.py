"""Modern logging utility for EvalHub using loguru.

This module provides a clean, elegant logging interface with beautiful console output
and automatic file rotation using the loguru library.
"""

import os
import sys
from pathlib import Path

from loguru import logger

# Remove default handler
logger.remove()

# Get log level from environment or default to INFO
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

# Add console handler with beautiful formatting
logger.level("INFO", color="<green>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red><bold>")
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - "
        "<level>{level: <8}</level> - "
        "<cyan>{file.name}:{line}</cyan> - {message}"
    ),
    level=LOG_LEVEL,
    colorize=True,
)

# Add file handler if LOG_DIR is specified
LOG_DIR = os.environ.get("LOG_DIR", None)
if LOG_DIR:
    log_dir_path = Path(LOG_DIR)
    log_dir_path.mkdir(exist_ok=True, parents=True)
    log_file_path = log_dir_path / "evalhub.log"
    logger.add(
        log_file_path,
        format="{time:YYYY-MM-DD HH:mm:ss} - {level: <7} - {module}:{line} - {message}",
        level=LOG_LEVEL,
        rotation="100 MB",  # Rotate when file reaches 100MB
        retention="10 days",  # Keep logs for 10 days
        compression="zip",  # Compress rotated logs
        enqueue=True,  # Thread-safe logging
    )
    logger.info(f"Log file created at: {log_file_path.absolute()}")
