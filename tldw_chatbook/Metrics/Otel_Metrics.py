# Otel_Metrics.py
"""
A thread-safe, generic metrics library built on top of the OpenTelemetry API.

This library decouples your application's instrumentation from the observability
backend. It is configured here to export to Prometheus, but could be easily
swapped to another exporter (like OTLP) with minimal code changes.

Key Features:
- Uses the standard OpenTelemetry API for future-proofing.
- Thread-safe instrument creation.
- Configuration via standard environment variables (e.g., OTEL_SERVICE_NAME).
- Decorator that emits metrics and adds events to active traces.
- Automatic collection of system and runtime metrics.

IMPORTANT: A NOTE ON ATTRIBUTE CARDINALITY
In OpenTelemetry, labels are called 'attributes'. The same warning applies:
attributes should only be used for values with low cardinality. Do not use
user IDs, request IDs, etc., as attributes.
"""
#
# Imports
import functools
import os
import threading
import time
import logging
#
# Third-Party Libraries
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.system_metrics import SystemMetricsInstrumentor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource, SERVICE_NAME, SERVICE_VERSION
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    logging.warning("OpenTelemetry not installed. Advanced metrics features will be disabled.")
#
# Local Imports
#
#######################################################################################################################
#
# Statics:
# Global Meter object. The "Meter" is how you create instruments (counters, etc.)
_meter = None
#############################################################
#
# Functions:

# A thread-safe registry for dynamically created OTel instruments.
_instrument_registry = {}
_instrument_lock = threading.Lock()
_meter = None


def init_metrics():
    """
    Initializes the OpenTelemetry SDK. Should be called once at startup.

    Configures a Prometheus exporter and sets global resource attributes
    which are attached to all emitted metrics. Configuration is read from
    standard OTel environment variables.
    """
    if not OTEL_AVAILABLE:
        logging.warning("OpenTelemetry not available. Metrics initialization skipped.")
        return
        
    global _meter

    # Use standard OTel env vars for configuration.
    service_name = os.getenv("OTEL_SERVICE_NAME", "unknown_service")
    service_version = os.getenv("OTEL_SERVICE_VERSION", "0.1.0")

    resource = Resource(attributes={
        SERVICE_NAME: service_name,
        SERVICE_VERSION: service_version,
    })

    # The reader is the "exporter" for metrics.
    # This one starts a Prometheus-compatible server.
    reader = PrometheusMetricReader()
    provider = MeterProvider(resource=resource, metric_readers=[reader])
    metrics.set_meter_provider(provider)

    _meter = metrics.get_meter("app.metrics.library")

    # Automatically instrument system metrics (CPU, memory, etc.)
    SystemMetricsInstrumentor().instrument()

    logging.info(
        f"OTel metrics initialized for service '{service_name}'. "
        f"Prometheus exporter available on port 9464 at /metrics"
    )


def _get_meter():
    """Returns the global meter, initializing if necessary."""
    if not OTEL_AVAILABLE:
        return None
    if not _meter:
        logging.warning("Metrics not explicitly initialized. Calling init_metrics() with defaults.")
        init_metrics()
    return _meter


def _get_or_create_instrument(instrument_type, name, unit="", description=""):
    """
    Internal function to get an instrument or create it if it doesn't exist.
    Uses a double-checked lock for thread safety and performance.
    """
    if not OTEL_AVAILABLE:
        return None
        
    if name in _instrument_registry:
        return _instrument_registry[name]

    with _instrument_lock:
        if name in _instrument_registry:
            return _instrument_registry[name]

        meter = _get_meter()
        if not meter:
            return None
            
        instrument = None
        if instrument_type == 'counter':
            instrument = meter.create_counter(name, unit=unit, description=description)
        elif instrument_type == 'histogram':
            instrument = meter.create_histogram(name, unit=unit, description=description)
        else:
            raise ValueError(f"Unsupported instrument type: {instrument_type}")

        _instrument_registry[name] = instrument
        return instrument


def log_counter(metric_name, value=1, labels=None, documentation=""):
    """
    Increments a counter. Documentation is used only on first creation.
    In OTel, 'labels' are called 'attributes'.
    """
    if not OTEL_AVAILABLE:
        logging.debug(f"OpenTelemetry not available. Would have logged counter: {metric_name}")
        return
    try:
        counter = _get_or_create_instrument(
            'counter', metric_name, unit="1", description=documentation
        )
        if counter:
            counter.add(value, attributes=(labels or {}))
    except Exception as e:
        logging.error(f"Failed to log OTel counter {metric_name}: {e}")


def log_histogram(metric_name, value, labels=None, documentation=""):
    """
    Records a value in a histogram. Documentation is used only on first creation.
    """
    if not OTEL_AVAILABLE:
        logging.debug(f"OpenTelemetry not available. Would have logged histogram: {metric_name} = {value}")
        return
    try:
        histogram = _get_or_create_instrument(
            'histogram', metric_name, unit="s", description=documentation
        )
        if histogram:
            histogram.record(value, attributes=(labels or {}))
    except Exception as e:
        logging.error(f"Failed to log OTel histogram {metric_name}: {e}")


def timeit(metric_name=None, documentation="Execution time and call count of a function."):
    """
    Decorator that times a function.

    - Emits a histogram for duration.
    - Emits a counter for calls.
    - Adds a 'status' attribute for success/error.
    - Adds an event to the current trace span, if one exists.
    """

    def decorator(func):
        base_name = metric_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get current span from the context. It's a no-op if no tracer is configured.
            span = None
            if OTEL_AVAILABLE:
                span = trace.get_current_span()
            start_time = time.time()
            status = "error"

            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            finally:
                elapsed_time = time.time() - start_time
                common_attributes = {"function": func.__name__, "status": status}

                # 1. Log metrics for aggregation
                log_histogram(
                    metric_name=f"{base_name}_duration_seconds",
                    value=elapsed_time,
                    labels=common_attributes,
                    documentation="Duration of function execution in seconds."
                )
                log_counter(
                    metric_name=f"{base_name}_calls_total",
                    labels=common_attributes,
                    documentation=f"Total calls to the function."
                )

                # 2. Add a precise event to the active trace for debugging
                if span and OTEL_AVAILABLE:
                    span.add_event(
                        name=f"finished {func.__name__}",
                        attributes={
                            "duration_sec": round(elapsed_time, 4),
                            "status": status,
                        }
                    )

        return wrapper

    return decorator



# --- Example Usage ----
# --- Application Code ---
# @timeit(metric_name="data_processing")
# import time
# import logging
# from metrics_otel import init_metrics, timeit, log_counter
# def process_data(user_id):
#     """A sample function to process some data."""
#     logging.info(f"Processing data for user {user_id}...")
#     time.sleep(0.2)
#
#     if user_id % 5 == 0:
#         # This is a good use of a custom counter
#         log_counter(
#             "special_user_processed_total",
#             labels={"user_type": "vip"},
#             documentation="Counter for a special type of user processing."
#         )
#
#     if user_id % 10 == 0:
#         raise ValueError("Simulating a failure")
#
#     logging.info("Done.")
#
#
# def main():
#     # Initialize OpenTelemetry metrics ONCE at application start.
#     # It reads configuration from environment variables.
#     init_metrics()
#
#     # Main application loop
#     user_id = 0
#     while True:
#         try:
#             process_data(user_id)
#         except ValueError as e:
#             logging.error(f"Failed to process data for user {user_id}: {e}")
#
#         user_id += 1
#         time.sleep(1)
#
#
# if __name__ == "__main__":
#     main()

#
# End of Otel_Metrics.py
#######################################################################################################################
