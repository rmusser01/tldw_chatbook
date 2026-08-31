# metrics.py
"""
A thread-safe, generic metrics library built on top of the official
prometheus_client.

Key Features:
- Dynamically creates metrics, avoiding hardcoded names.
- Thread-safe metric creation for use in web servers.
- Ergonomic API that doesn't require repeating documentation.
- Decorators for common patterns like timing functions.

IMPORTANT: A NOTE ON LABEL CARDINALITY
Metric labels should only be used for values with a small, finite set of
possibilities (low cardinality). Using labels with high cardinality values
(e.g., user_id, request_id, file_path) will cause an explosion in the number
of time series, overwhelming your Prometheus server.

- DO use labels for: status codes, environments, machine types, API endpoints.
- DO NOT use labels for: user IDs, session IDs, trace IDs, URLs, or any
  unbounded unique identifier.
"""

#
# Imports
import functools
import os
import threading
import time
import logging
from typing import Any, Dict, Optional

import psutil  #

# Third-party Imports
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Create dummy classes to prevent errors when prometheus_client is not installed
    class Counter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    class Histogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    class Gauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def labels(self, **kwargs):
            return self

    def start_http_server(*args, **kwargs):
        logging.warning("Prometheus client not installed. Metrics server not started.")
#
# Local Imports
#
######################################################################################################################
#
# Functions:

# A thread-safe registry for dynamically created metrics.
_metrics_registry = {}
_registry_lock = threading.Lock()


def _get_or_create_metric(metric_type, name, documentation, label_keys=None):
    """
    Internal function to get a metric from the registry or create it if it
    doesn't exist. Uses a double-checked lock for thread safety and performance.
    """
    label_keys = tuple(sorted(label_keys or []))
    registry_key = (metric_type, name, label_keys)

    # Fast path: check if metric exists without locking.
    if registry_key in _metrics_registry:
        return _metrics_registry[registry_key]

    # Slow path: acquire lock to safely create the metric.
    with _registry_lock:
        # Double-check if another thread created it while we were waiting.
        if registry_key in _metrics_registry:
            return _metrics_registry[registry_key]

        if metric_type == "counter":
            metric = Counter(name, documentation, label_keys)
        elif metric_type == "histogram":
            metric = Histogram(name, documentation, label_keys)
        elif metric_type == "gauge":
            metric = Gauge(name, documentation, label_keys)
        else:
            raise ValueError(f"Unsupported metric type: {metric_type}")

        _metrics_registry[registry_key] = metric
        return metric


def log_counter(metric_name, value=1, labels=None, documentation=""):
    """
    Increments a counter metric. The metric is created on first use.
    Documentation is only used during the initial creation of the metric.
    """
    if not PROMETHEUS_AVAILABLE:
        logging.debug(
            f"Prometheus not available. Would have logged counter: {metric_name}"
        )
        return
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        counter = _get_or_create_metric(
            "counter", metric_name, documentation, label_keys
        )
        if label_keys:
            counter.labels(**eff_labels).inc(value)
        else:
            counter.inc(value)
    except Exception as e:
        logging.error(f"Failed to log counter {metric_name}: {e}")


def log_histogram(metric_name, value, labels=None, documentation=""):
    """
    Observes a value for a histogram metric. The metric is created on first use.
    Documentation is only used during the initial creation of the metric.
    """
    if not PROMETHEUS_AVAILABLE:
        logging.debug(
            f"Prometheus not available. Would have logged histogram: {metric_name} = {value}"
        )
        return
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        histogram = _get_or_create_metric(
            "histogram", metric_name, documentation, label_keys
        )
        if label_keys:
            histogram.labels(**eff_labels).observe(value)
        else:
            histogram.observe(value)
    except Exception as e:
        logging.error(f"Failed to log histogram {metric_name}: {e}")


def log_gauge(metric_name, value, labels=None, documentation=""):
    """

    Sets the value of a gauge metric. The metric is created on first use.
    Documentation is only used during the initial creation of the metric.
    """
    if not PROMETHEUS_AVAILABLE:
        logging.debug(
            f"Prometheus not available. Would have logged gauge: {metric_name} = {value}"
        )
        return
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        gauge = _get_or_create_metric("gauge", metric_name, documentation, label_keys)
        if label_keys:
            gauge.labels(**eff_labels).set(value)
        else:
            gauge.set(value)
    except Exception as e:
        logging.error(f"Failed to log gauge {metric_name}: {e}")


def timeit(metric_name=None, documentation="Execution time of a function."):
    """
    Decorator that times a function, logging a histogram for duration and a
    counter for total calls. It also adds a 'status' label for success/error.
    """

    def decorator(func):
        base_name = metric_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            status = "error"  # Default to error
            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            finally:
                elapsed = time.time() - start
                common_labels = {"function": func.__name__, "status": status}

                log_histogram(
                    metric_name=f"{base_name}_duration_seconds",
                    value=elapsed,
                    labels=common_labels,
                    documentation=documentation,
                )

                log_counter(
                    metric_name=f"{base_name}_calls_total",
                    labels=common_labels,
                    documentation=f"Total calls to {func.__name__}",
                )

        return wrapper

    return decorator


def log_resource_usage(labels=None):
    """Logs current CPU and Memory usage of the process as gauges."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024**2)
    cpu_percent = process.cpu_percent(interval=None)  # Non-blocking

    log_gauge(
        "process_memory_mb",
        memory_mb,
        labels=labels,
        documentation="Current memory usage of the process in Megabytes.",
    )
    log_gauge(
        "process_cpu_percent",
        cpu_percent,
        labels=labels,
        documentation="Current CPU usage of the process as a percentage.",
    )


#: Shipped defaults for the metrics listener. Disabled, and loopback-only when
#: enabled -- ``prometheus_client.start_http_server`` defaults to ``0.0.0.0``,
#: which we deliberately do not inherit.
_METRICS_DEFAULT_ENABLED = False
_METRICS_DEFAULT_PORT = 8000
_METRICS_DEFAULT_BIND_ADDRESS = "127.0.0.1"


def _get_cli_setting(section: str, key: str, default: Any) -> Any:
    """Read one config value.

    Indirection on purpose: it keeps the ``config`` import lazy (this module is
    imported early) and lets tests exercise resolution without a config file on
    disk.
    """
    from tldw_chatbook.config import get_cli_setting

    return get_cli_setting(section, key, default)


def _metrics_server_config() -> Dict[str, Any]:
    """Resolve whether to listen, and where.

    ``METRICS_PORT`` continues to override the port because it predates this
    function, but it does NOT enable the listener -- enabling is a config
    decision (TASK-25914 AC#1). Invalid values fall back to the default rather
    than raising during startup.
    """
    port = _get_cli_setting("metrics", "port", _METRICS_DEFAULT_PORT)
    env_port = os.environ.get("METRICS_PORT")
    if env_port:
        port = env_port
    try:
        resolved_port = int(port)
    except (TypeError, ValueError):
        resolved_port = _METRICS_DEFAULT_PORT

    return {
        "enabled": bool(
            _get_cli_setting("metrics", "enabled", _METRICS_DEFAULT_ENABLED)
        ),
        "port": resolved_port,
        "bind_address": str(
            _get_cli_setting(
                "metrics", "bind_address", _METRICS_DEFAULT_BIND_ADDRESS
            )
        ),
    }


def init_metrics_server(port: Optional[int] = None, addr: Optional[str] = None) -> bool:
    """Start the Prometheus listener if the user has asked for one.

    Binding a network socket is opt-in. Having ``prometheus_client`` installed
    -- which the ``dev`` and ``debugging`` extras both do -- is not consent, so
    the config gate is checked before anything is bound (TASK-25914).

    Args:
        port: Overrides the configured port when given.
        addr: Overrides the configured bind address when given.

    Returns:
        True when a listener was started, False otherwise.
    """
    settings = _metrics_server_config()

    if not settings["enabled"]:
        logging.debug(
            "Prometheus metrics listener disabled; set [metrics] enabled = true "
            "to expose one. Metric collection is unaffected."
        )
        return False

    if not PROMETHEUS_AVAILABLE:
        logging.info(
            "Prometheus metrics listener is enabled in config but the optional "
            "dependency is missing. Install tldw_chatbook[debugging] to use it."
        )
        return False

    bind_port = settings["port"] if port is None else port
    bind_address = settings["bind_address"] if addr is None else addr

    start_http_server(bind_port, addr=bind_address)
    logging.info(
        "Prometheus metrics listener started on %s:%s (unauthenticated -- "
        "bind address is configurable via [metrics] bind_address)",
        bind_address,
        bind_port,
    )
    return True


# --- Sample Usage ---
# pip install opentelemetry-sdk opentelemetry-exporter-prometheus opentelemetry-instrumentation-system-metrics
# OTEL_SERVICE_NAME=video-processor OTEL_SERVICE_VERSION=1.2.3 python main_app.py
#
# @timeit() # Uses the function name `process_data` to build metric names
# def process_data(user_id):
#     """A sample function to process some data."""
#     print(f"Processing data for user {user_id}...")
#     time.sleep(0.5)
#     if user_id % 5 == 0:
#         # You can still log custom counters inside your functions
#         log_counter(
#             "special_user_processed_total",
#             "Counter for a special type of user.",
#             labels={"user_type": "vip"}
#         )
#     print("Done.")
#
# def main():
#     # Start the metrics server once at the beginning of your app
#     init_metrics_server(port=8000)
#
#     # Example usage
#     user_id = 0
#     while True:
#         process_data(user_id)
#         log_resource_usage() # Log resource usage in your main loop
#         user_id += 1
#         time.sleep(1)
#
# if __name__ == "__main__":
#     main()

#
# End of metrics.py
############################################################################################################
