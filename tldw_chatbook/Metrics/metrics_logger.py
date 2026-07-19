# metrics_logger.py
#
# Imports
import functools
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Optional, Dict, Union, Callable
import psutil
#
# Third-party Imports
#
# Local Imports
from loguru import logger
#
############################################################################################################
#
# Functions:

# 1. Refined Type Hinting for clarity and correctness
LabelValue = Union[str, int, float, bool]
LabelDict = Dict[str, LabelValue]
_METRICS_LOGGING_TRUTHY = {"1", "true", "yes", "on"}
_metrics_enabled_cache: bool | None = None

# 2. (Gold Standard) Define a custom "METRIC" level for powerful filtering
# This allows separating metrics from regular application logs at the sink level.
logger.level("METRIC", no=25, color="<blue>", icon="📊")


def is_metrics_logging_enabled() -> bool:
    """Return whether high-volume Loguru metric lines should be emitted."""

    global _metrics_enabled_cache
    if "PYTEST_CURRENT_TEST" in os.environ:
        return os.environ.get("TLDW_METRICS_LOGGING", "").lower() in _METRICS_LOGGING_TRUTHY
    if _metrics_enabled_cache is None:
        _metrics_enabled_cache = os.environ.get("TLDW_METRICS_LOGGING", "").lower() in _METRICS_LOGGING_TRUTHY
    return _metrics_enabled_cache


def _log_metric(
        metric_name: str,
        metric_type: str,
        value: Any,
        labels: Optional[LabelDict] = None,
):
    """
    Private helper to log a structured metric using idiomatic loguru binding.
    """
    if not is_metrics_logging_enabled():
        return

    # 3. Bind each piece of data to the top level for a flatter, queryable JSON
    bound_logger = logger.bind(
        event=metric_name,
        type=metric_type,
        value=value,
        labels=labels or {},
        timestamp=datetime.now(timezone.utc).isoformat(),
    )
    # Use the custom METRIC level
    bound_logger.log("METRIC", f"{metric_type.capitalize()} '{metric_name}': {value}")


def timeit(
        metric_name: Optional[str] = None,
        labels: Optional[LabelDict] = None,
        log_summary: bool = True,
        log_call_count: bool = False,
):
    """
    A robust decorator that times a function, logging a histogram and status.

    Args:
        metric_name (str, optional): Custom name for the metric. Defaults to function name.
        labels (dict, optional): Extra labels to add to the metric.
        log_summary (bool): If True, logs a human-readable summary at INFO level.
        log_call_count (bool): If True, also logs a counter metric for each call.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 4. Robust timing and status tracking
            m_name = metric_name or f"{func.__name__}_duration_seconds"
            all_labels = {"function": func.__name__}
            if labels:
                all_labels.update(labels)

            start_time = time.perf_counter()
            status = "success"
            try:
                result = func(*args, **kwargs)
                return result
            except Exception:
                status = "failure"
                raise  # Re-raise the exception after marking status
            finally:
                elapsed_time = time.perf_counter() - start_time
                final_labels = {**all_labels, "status": status}

                # Log the primary histogram metric
                _log_metric(m_name, "histogram", elapsed_time, final_labels)

                # Optionally log a separate counter metric
                if log_call_count:
                    counter_name = f"{func.__name__}_calls_total"
                    _log_metric(counter_name, "counter", 1, final_labels)

                if log_summary:
                    logger.info(
                        f"Function '{func.__name__}' finished in {elapsed_time:.4f}s "
                        f"with status '{status}'."
                    )

        return wrapper

    return decorator


class MetricsLogger:
    """
    5. A class-based API for providing context (base labels) to a set of metrics.

    This is useful for grouping all metrics from a specific module or request.
    """

    def __init__(self, base_labels: Optional[LabelDict] = None):
        self._base_labels = base_labels or {}
        # Cached across calls (not re-created per log_resource_usage() call)
        # so psutil's non-blocking cpu_percent(interval=None) has a prior
        # sample to diff against after the first call -- a fresh
        # psutil.Process() has no history and would always report 0.0.
        self._resource_process: Optional[psutil.Process] = None

    def _get_labels(self, labels: Optional[LabelDict]) -> LabelDict:
        """Merge instance labels with call-specific labels."""
        final_labels = self._base_labels.copy()
        if labels:
            final_labels.update(labels)
        return final_labels

    def log_counter(self, name: str, value: int = 1, labels: Optional[LabelDict] = None):
        _log_metric(name, "counter", value, self._get_labels(labels))

    def log_gauge(self, name: str, value: float, labels: Optional[LabelDict] = None):
        _log_metric(name, "gauge", value, self._get_labels(labels))

    def log_histogram(self, name: str, value: float, labels: Optional[LabelDict] = None):
        _log_metric(name, "histogram", value, self._get_labels(labels))

    def log_resource_usage(self, labels: Optional[LabelDict] = None):
        if self._resource_process is None:
            self._resource_process = psutil.Process()
        process = self._resource_process
        combined_labels = self._get_labels(labels)
        self.log_gauge("process_memory_mb", process.memory_info().rss / (1024 ** 2), combined_labels)
        # Non-blocking: interval=None compares against the last recorded
        # sample instead of sleeping the calling thread to measure a fresh
        # window (see task-248 / performance audit finding A3). The first
        # call after construction returns 0.0 (no prior sample) -- the
        # cached process handle above means later calls are accurate.
        self.log_gauge("process_cpu_percent", process.cpu_percent(interval=None), combined_labels)


# For convenience, a default instance for simple, one-off logging
default_metrics = MetricsLogger()
log_counter = default_metrics.log_counter
log_gauge = default_metrics.log_gauge
log_histogram = default_metrics.log_histogram
log_resource_usage = default_metrics.log_resource_usage

# # Example usage block to demonstrate the new features
# if __name__ == "__main__":
#     from logger_config import setup_logger
#
#     # Configure sinks. One for console, one just for metrics.
#     logger.remove()
#     logger.add(sys.stdout, level="INFO", format="{level.icon} {level.name}: {message}")
#     logger.add(
#         "test_metrics_only.json",
#         level="METRIC",  # This sink will ONLY capture our metrics!
#         serialize=True
#     )
#
#     logger.info("--- Testing Advanced Metrics Logger ---")
#
#
#     # 1. Test the robust @timeit decorator
#     @timeit(log_call_count=True)
#     def successful_task():
#         time.sleep(0.1)
#
#
#     @timeit
#     def failing_task():
#         time.sleep(0.1)
#         raise ValueError("Something went wrong")
#
#
#     successful_task()
#     try:
#         failing_task()
#     except ValueError as e:
#         logger.warning(f"Caught expected exception: {e}")
#
#     # 2. Test the class-based logger with context
#     api_logger = MetricsLogger(base_labels={"component": "api", "version": "v2"})
#     api_logger.log_counter("requests_total", labels={"endpoint": "/users"})
#     api_logger.log_counter("requests_total", labels={"endpoint": "/data"})
#
#     # 3. Test the default instance for one-off metrics
#     log_resource_usage()
#
#     logger.info("--- Test complete. Check 'test_metrics_only.json' ---")

#
# End of metrics_logger.py
############################################################################################################
