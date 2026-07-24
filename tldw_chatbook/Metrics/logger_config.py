"""Legacy console-only Loguru setup.

Persistent application diagnostics are owned by ``Logging_Config``. This
module previously installed two independent Loguru file sinks with exception
diagnostics enabled, bypassing the private-path and metadata-only boundaries.
No production caller uses those sinks, so their path parameters are retained
only for API compatibility and fail closed.
"""

from __future__ import annotations

import sys
import warnings
from typing import Optional

from loguru import logger

DEFAULT_APP_LOG_PATH = "~/.local/tldw_cli/Logs/tldw_app.log"
DEFAULT_METRICS_LOG_PATH = "~/.local/tldw_cli/Logs/tldw_metrics.json"


def setup_logger(
    log_level: str = "DEBUG",
    console_format: str = "{time:YYYY-MM-DD HH:mm:ss} - {level} - {message}",
    app_log_path: Optional[str] = DEFAULT_APP_LOG_PATH,
    metrics_log_path: Optional[str] = DEFAULT_METRICS_LOG_PATH,
):
    """Configure the legacy Loguru console sink.

    Direct file sinks are intentionally disabled. Persistent metadata must use
    the application logging boundary, which owns file permissions, rotation,
    source admission, and payload exclusion.
    """

    logger.remove()
    logger.add(sys.stdout, level=log_level.upper(), format=console_format)
    if app_log_path is not None or metrics_log_path is not None:
        warnings.warn(
            "Legacy Loguru file sinks are disabled; use application logging.",
            RuntimeWarning,
            stacklevel=2,
        )
    return logger
