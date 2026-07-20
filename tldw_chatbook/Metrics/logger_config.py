# logger_config.py
#
# Imports
import json
import sys
import os
from datetime import datetime
from typing import Optional

#
# 3rd-Party Imports
from loguru import logger
# Local Imports
#
#
############################################################################################################
#
# Functions:

# Sensible default locations for logs
DEFAULT_APP_LOG_PATH = "~/.local/tldw_cli/Logs/tldw_app.log"
DEFAULT_METRICS_LOG_PATH = "~/.local/tldw_cli/Logs/tldw_metrics.json"


def _ensure_log_dir_exists(file_path: str):
    """Ensure the directory for the log file exists."""
    # Expand the user's home directory if '~' is used
    expanded_path = os.path.expanduser(file_path)
    log_dir = os.path.dirname(expanded_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    return expanded_path


def retention_function(files):
    """
    A retention function to mimic backupCount=5.

    Given a list of log file paths, this function sorts them by modification time
    and returns the list of files to be removed so that at most 5 are kept.
    """
    if len(files) > 5:
        # Sort files by modification time (oldest first)
        files.sort(key=lambda filename: os.path.getmtime(filename))
        # Remove all but the 5 most recent files.
        return files[:-5]
    return []


def json_formatter(record):
    """
    Custom JSON formatter for file logging.
    """
    try:
        # Format the log time as a string.
        dt = record["time"].strftime("%Y-%m-%d %H:%M:%S.%f")
        # Grab any extra data passed with the log.
        extra = record["extra"]

        # Handle potential non-serializable fields in 'extra'
        def serialize(value):
            if isinstance(value, datetime):
                return value.isoformat()
            return value

        log_record = {
            "time": dt,
            "levelname": record["level"].name,
            "name": record["name"],
            "message": record["message"],
            "event": extra.get("event"),
            "type": extra.get("type"),
            "value": extra.get("value"),
            "labels": extra.get("labels"),
            "timestamp": serialize(extra.get("timestamp")),
        }
        return json.dumps(log_record)
    except Exception as e:
        # Fallback to a safe JSON structure if serialization fails
        return json.dumps(
            {
                "error": f"Log formatting failed: {str(e)}",
                "original_message": record.get("message", ""),
            }
        )


def setup_logger(
    log_level: str = "DEBUG",
    console_format: str = "{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    app_log_path: Optional[str] = DEFAULT_APP_LOG_PATH,
    metrics_log_path: Optional[str] = DEFAULT_METRICS_LOG_PATH,
):
    """
    Sets up Loguru sinks for console, a standard application log, and a JSON metrics log.

    Args:
        log_level (str): The minimum log level to output (e.g., "DEBUG", "INFO").
        console_format (str): The format string for console output.
        app_log_path (Optional[str]): Path for the standard text log file. If None, this sink is disabled.
        metrics_log_path (Optional[str]): Path for the structured JSON metrics log. If None, this sink is disabled.

    Returns:
        The configured logger instance.
    """
    # Start with a clean slate
    logger.remove()

    # 1. Console Sink (always enabled)
    logger.add(sys.stdout, level=log_level.upper(), format=console_format)

    # 2. Standard Application File Sink
    if app_log_path:
        path = _ensure_log_dir_exists(app_log_path)
        logger.add(
            path,
            level=log_level.upper(),
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",  # Rotate file when it reaches 10 MB
            retention="7 days",  # Keep logs for 7 days
            enqueue=True,  # Make logging non-blocking
            backtrace=True,  # Show full stack trace on exceptions
            diagnose=True,  # Add exception variable values
        )
        logger.info(f"Application logs will be written to: {path}")

    # 3. JSON Metrics File Sink
    if metrics_log_path:
        path = _ensure_log_dir_exists(metrics_log_path)
        logger.add(
            path,
            level="DEBUG",  # Typically, you want all levels for metrics
            serialize=True,  # This is the key for JSON output
            rotation="10 MB",
            retention=5,  # Keeps the 5 most recent log files
            enqueue=True,
        )
        logger.info(f"JSON metrics logs will be written to: {path}")

    return logger


# def setup_logger(log_file_path="tldw_app_logs.json"):
#     """
#     Sets up the logger with both StreamHandler and FileHandler, formatted in JSON.
#
#     Parameters:
#         log_file_path (str): Path to the JSON log file.
#
#     Returns:
#         logging.Logger: Configured logger instance.
#     """
#     logger = logging.getLogger("tldw_app_logs")
#     logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs
#
#     # Prevent adding multiple handlers if the logger is already configured
#     if not logger.handlers:
#         # StreamHandler for console output
#         stream_handler = logging.StreamHandler()
#         stream_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         stream_handler.setFormatter(stream_formatter)
#         logger.addHandler(stream_handler)
#
#         # Ensure the directory for log_file_path exists
#         log_dir = os.path.dirname(log_file_path)
#         if log_dir and not os.path.exists(log_dir):
#             os.makedirs(log_dir, exist_ok=True)
#
#         # RotatingFileHandler for writing logs to a JSON file with rotation
#         file_handler = RotatingFileHandler(
#             log_file_path, maxBytes=10*1024*1024, backupCount=5  # 10 MB per file, keep 5 backups
#         )
#         file_formatter = jsonlogger.JsonFormatter(
#             '%(asctime)s %(levelname)s %(name)s event %(event)s type %(type)s value %(value)s labels %(labels)s timestamp %(timestamp)s'
#         )
#         file_handler.setFormatter(file_formatter)
#         logger.addHandler(file_handler)
#
#     return logger
#
# # Initialize the logger
# logger = setup_logger()


#
# End of Functions
############################################################################################################
