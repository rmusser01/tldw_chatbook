# ollama_model_mgmt.py
#
# Imports
#
# Third-Party Libraries
#
# Local Imports
#
#######################################################################################################################
# Ollama Management API Calls
#######################################################################################################################
import logging
import json as json_parser
import time
from typing import Optional, Callable, Tuple, Any, Dict
import requests
from ..Metrics.metrics_logger import log_counter, log_histogram


# Note: TypingDict is used as an alias for Dict in type hints for Ollama functions
# to avoid any potential (though unlikely) conflict if 'Dict' itself were a parameter name.
# `logging` refers to the project's logger, already configured.
# `requests` is imported at the top.
# `json_parser` is an alias for the standard `json` module.
_KNOWN_ENDPOINTS = {
    "copy",
    "create",
    "delete",
    "embeddings",
    "ps",
    "pull",
    "push",
    "show",
    "tags",
}


def _request_metadata(method: str, endpoint: str) -> tuple[str, str]:
    """Return bounded allowlisted request metadata for diagnostics."""

    safe_method = (
        method.upper() if method.upper() in {"GET", "POST", "DELETE"} else "OTHER"
    )
    endpoint_name = endpoint.rsplit("/", 1)[-1]
    safe_endpoint = endpoint_name if endpoint_name in _KNOWN_ENDPOINTS else "unknown"
    return safe_method, safe_endpoint


def _request_failure(
    method: str,
    endpoint: str,
    category: str,
    *,
    status_code: int | None = None,
) -> Tuple[None, str]:
    """Emit and return a generic request failure without caller-controlled text."""

    safe_method, safe_endpoint = _request_metadata(method, endpoint)
    labels = {
        "method": safe_method,
        "endpoint": safe_endpoint,
        "error": category,
    }
    if status_code is not None:
        labels["status_code"] = str(status_code)
    log_counter("ollama_api_request_error", labels=labels)
    logging.error(
        "Ollama request failed (method=%s, endpoint=%s, category=%s, status=%s).",
        safe_method,
        safe_endpoint,
        category,
        status_code if status_code is not None else "none",
    )
    status = f", status={status_code}" if status_code is not None else ""
    return None, f"Ollama request failed (category={category}{status})."


def _ollama_request(
    method: str,
    base_url: str,
    endpoint: str,
    stream_log_callback: Optional[Callable[[str], None]] = None,
    **kwargs: Any,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Make one Ollama API request with metadata-only failure diagnostics."""

    start_time = time.time()
    safe_method, safe_endpoint = _request_metadata(method, endpoint)
    log_counter(
        "ollama_api_request_attempt",
        labels={"method": safe_method, "endpoint": safe_endpoint},
    )

    if not base_url.startswith(("http://", "https://")):
        log_counter("ollama_api_request_error", labels={"error": "invalid_url_scheme"})
        logging.error("Ollama request rejected (category=invalid_url_scheme).")
        return None, "Invalid Ollama server URL."
    full_url = f"{base_url.rstrip('/')}{endpoint}"
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if stream_log_callback:
        if "json" in kwargs and isinstance(kwargs["json"], dict):
            kwargs["json"]["stream"] = True
    logging.debug(
        "Ollama request started (method=%s, endpoint=%s, streaming=%s).",
        safe_method,
        safe_endpoint,
        bool(stream_log_callback),
    )
    try:
        with requests.Session() as session:
            session.headers.update(headers)
            response = session.request(
                method,
                full_url,
                stream=bool(stream_log_callback),
                timeout=300,
                **kwargs,
            )
            if stream_log_callback:
                last_line_json: Optional[Dict[str, Any]] = None
                stream_error = False
                if response.status_code >= 400:
                    stream_log_callback(
                        f"Ollama operation failed (status={response.status_code}).\n"
                    )
                    return _request_failure(
                        method,
                        endpoint,
                        "http_error",
                        status_code=response.status_code,
                    )
                for line_bytes in response.iter_lines():
                    if line_bytes:
                        try:
                            json_line = json_parser.loads(line_bytes.decode("utf-8"))
                            if isinstance(json_line, dict):
                                last_line_json = json_line
                                stream_error = stream_error or "error" in json_line
                        except (UnicodeDecodeError, json_parser.JSONDecodeError):
                            continue
                        stream_log_callback("Ollama operation progress received.\n")
                response.raise_for_status()
                if stream_error:
                    _, error = _request_failure(
                        method,
                        endpoint,
                        "stream_error",
                    )
                    return last_line_json, error

                duration = time.time() - start_time
                log_histogram(
                    "ollama_api_request_duration",
                    duration,
                    labels={
                        "method": safe_method,
                        "endpoint": safe_endpoint,
                        "streaming": "true",
                        "status": "success",
                    },
                )
                log_counter(
                    "ollama_api_request_success",
                    labels={
                        "method": safe_method,
                        "endpoint": safe_endpoint,
                        "streaming": "true",
                    },
                )

                return last_line_json, None
            else:
                response.raise_for_status()

                duration = time.time() - start_time
                log_histogram(
                    "ollama_api_request_duration",
                    duration,
                    labels={
                        "method": safe_method,
                        "endpoint": safe_endpoint,
                        "streaming": "false",
                        "status": "success",
                    },
                )
                log_counter(
                    "ollama_api_request_success",
                    labels={
                        "method": safe_method,
                        "endpoint": safe_endpoint,
                        "streaming": "false",
                    },
                )

                if response.content and "application/json" in response.headers.get(
                    "Content-Type", ""
                ):
                    return response.json(), None
                if response.status_code == 200 and not response.content:
                    return {
                        "status": "success",
                        "message": "Operation completed successfully.",
                    }, None
                if response.status_code == 200:
                    logging.warning(
                        "Ollama request returned a successful non-JSON response "
                        "(method=%s, endpoint=%s).",
                        safe_method,
                        safe_endpoint,
                    )
                    return {
                        "status": "success",
                        "message": "Operation completed successfully.",
                    }, None
                return _request_failure(
                    method,
                    endpoint,
                    "unexpected_response",
                    status_code=response.status_code,
                )
    except requests.exceptions.HTTPError as exc:
        response = getattr(exc, "response", None)
        return _request_failure(
            method,
            endpoint,
            "http_error",
            status_code=getattr(response, "status_code", None),
        )
    except requests.exceptions.ConnectionError:
        return _request_failure(
            method,
            endpoint,
            "connection_error",
        )
    except requests.exceptions.Timeout:
        return _request_failure(
            method,
            endpoint,
            "timeout",
        )
    except requests.exceptions.RequestException:
        return _request_failure(
            method,
            endpoint,
            "request_exception",
        )
    except json_parser.JSONDecodeError:
        return _request_failure(
            method,
            endpoint,
            "json_decode_error",
        )
    except Exception:
        return _request_failure(
            method,
            endpoint,
            "unexpected_error",
        )


def ollama_list_local_models(
    base_url: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    result, error = _ollama_request("GET", base_url, "/api/tags")

    if result and not error:
        # Log model count if successful
        models = result.get("models", [])
        log_histogram("ollama_local_models_count", len(models))

    return result, error


def ollama_model_info(
    base_url: str, model_name: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return _ollama_request("POST", base_url, "/api/show", json={"name": model_name})


def ollama_copy_model(
    base_url: str, source: str, destination: str
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    return _ollama_request(
        "POST",
        base_url,
        "/api/copy",
        json={"source": source, "destination": destination},
    )


def ollama_delete_model(
    base_url: str,
    model_name: str,
    stream_log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    log_counter("ollama_delete_model_attempt")

    result, error = _ollama_request(
        "DELETE",
        base_url,
        "/api/delete",
        stream_log_callback=stream_log_callback,
        json={"name": model_name},
    )

    if not error:
        log_counter("ollama_delete_model_success")
    else:
        log_counter("ollama_delete_model_error")

    return result, error


def ollama_pull_model(
    base_url: str,
    model_name: str,
    insecure: bool = False,
    stream_log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    log_counter(
        "ollama_pull_model_attempt",
        labels={"insecure": str(insecure)},
    )

    payload = {"name": model_name, "insecure": insecure}
    result, error = _ollama_request(
        "POST",
        base_url,
        "/api/pull",
        stream_log_callback=stream_log_callback,
        json=payload,
    )

    if not error:
        log_counter("ollama_pull_model_success")
    else:
        log_counter("ollama_pull_model_error")

    return result, error


def ollama_create_model(
    base_url: str,
    model_name: str,
    path: str,
    stream_log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload = {"name": model_name, "path": path}
    return _ollama_request(
        "POST",
        base_url,
        "/api/create",
        stream_log_callback=stream_log_callback,
        json=payload,
    )


def ollama_push_model(
    base_url: str,
    model_name: str,
    insecure: bool = False,
    stream_log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload = {"name": model_name, "insecure": insecure}
    return _ollama_request(
        "POST",
        base_url,
        "/api/push",
        stream_log_callback=stream_log_callback,
        json=payload,
    )


def ollama_generate_embeddings(
    base_url: str,
    model_name: str,
    prompt: str,
    options: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    payload: Dict[str, Any] = {"model": model_name, "prompt": prompt}
    if options:
        payload["options"] = options
    return _ollama_request("POST", base_url, "/api/embeddings", json=payload)


def ollama_list_running_models(
    base_url: str,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    result, error = _ollama_request("GET", base_url, "/api/ps")

    if result and not error:
        # Log running model count if successful
        models = result.get("models", [])
        log_histogram("ollama_running_models_count", len(models))

    return result, error
