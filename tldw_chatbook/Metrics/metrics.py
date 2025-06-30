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
import threading
import time
import logging
import psutil#
# Third-party Imports
from prometheus_client import Counter, Histogram, Gauge, start_http_server
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

        if metric_type == 'counter':
            metric = Counter(name, documentation, label_keys)
        elif metric_type == 'histogram':
            metric = Histogram(name, documentation, label_keys)
        elif metric_type == 'gauge':
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
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        counter = _get_or_create_metric('counter', metric_name, documentation, label_keys)
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
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        histogram = _get_or_create_metric('histogram', metric_name, documentation, label_keys)
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
    try:
        label_keys = list(labels.keys()) if labels else []
        eff_labels = labels or {}
        gauge = _get_or_create_metric('gauge', metric_name, documentation, label_keys)
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
                    documentation=documentation
                )

                log_counter(
                    metric_name=f"{base_name}_calls_total",
                    labels=common_labels,
                    documentation=f"Total calls to {func.__name__}"
                )

        return wrapper

    return decorator


def log_resource_usage(labels=None):
    """Logs current CPU and Memory usage of the process as gauges."""
    process = psutil.Process()
    memory_mb = process.memory_info().rss / (1024 ** 2)
    cpu_percent = process.cpu_percent(interval=None)  # Non-blocking

    log_gauge(
        "process_memory_mb",
        memory_mb,
        labels=labels,
        documentation="Current memory usage of the process in Megabytes."
    )
    log_gauge(
        "process_cpu_percent",
        cpu_percent,
        labels=labels,
        documentation="Current CPU usage of the process as a percentage."
    )


def init_metrics_server(port=8000):
    """Starts the Prometheus HTTP server in a separate thread."""
    start_http_server(port)
    logging.info(f"Prometheus metrics server started on port {port}")


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
