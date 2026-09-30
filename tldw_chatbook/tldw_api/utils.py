# tldw_chatbook/tldw_api/utils.py
#
#
# Imports
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Dict, Any, Optional, List, IO, Tuple, NoReturn
import mimetypes

#
# 3rd-party Libraries
import httpx
from pydantic import BaseModel

#
#######################################################################################################################
#
# Functions:


def model_to_form_data(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic model instance into a dictionary suitable for
    FastAPI Form data submission (handles None, bool, list to string).
    """
    form_data = {}
    for field_name, field_value in model_instance.model_dump(exclude_none=True).items():
        if field_name == "keywords" and isinstance(field_value, list):
            form_data[field_name] = ",".join(
                field_value
            )  # Server expects comma-separated string for "keywords"
        elif isinstance(field_value, bool):
            form_data[field_name] = str(field_value).lower()  # FastAPI Form booleans
        elif isinstance(field_value, list):
            # For lists other than keywords (e.g., urls), httpx handles them correctly
            # if the server endpoint expects multiple values for the same form field name.
            form_data[field_name] = field_value
        elif field_value is not None:
            form_data[field_name] = str(field_value)  # Most other fields as strings
    return form_data


def prepare_files_for_httpx(
    file_paths: Optional[List[str]], upload_field_name: str = "files"
) -> Optional[List[Tuple[str, Tuple[str, IO[bytes], Optional[str]]]]]:
    """
    Prepares a list of file paths for httpx multipart upload.

    Args:
        file_paths: A list of string paths to local files.
        upload_field_name: The name of the field for file uploads (FastAPI often uses 'files').

    Returns:
        A list of tuples formatted for httpx's `files` argument, or None.
        Example: [('files', ('filename.mp4', <file_obj>, 'video/mp4')), ...]

    Note:
        Callers are responsible for calling cleanup_file_objects() after using
        the returned file objects to prevent resource leaks.
    """
    if not file_paths:
        return None

    logging.debug(f"prepare_files_for_httpx called with file_paths: {file_paths}")
    httpx_files_list = []
    for file_path_str in file_paths:
        file_obj = None
        try:
            file_path_obj = Path(file_path_str)
            if not file_path_obj.is_file():
                logging.warning(
                    f"Warning: File not found or not a file: {file_path_str}"
                )
                continue

            file_obj = open(file_path_obj, "rb")

            mime_type, _ = mimetypes.guess_type(file_path_obj.name)

            if mime_type is None:
                mime_type = "application/octet-stream"
                logging.warning(
                    f"Could not guess MIME type for {file_path_obj.name}. Defaulting to {mime_type}."
                )

            httpx_files_list.append(
                (upload_field_name, (file_path_obj.name, file_obj, mime_type))
            )
        except Exception as e:
            logging.error(f"Error preparing file {file_path_str} for upload: {e}")
            # Close the file if it was opened but failed to be added to the list
            if file_obj:
                try:
                    file_obj.close()
                except Exception:
                    pass  # Ignore close errors
            # Continue to next file instead of breaking the loop
            continue
    return httpx_files_list if httpx_files_list else None


def cleanup_file_objects(
    httpx_files: Optional[List[Tuple[str, Tuple[str, IO[bytes], Optional[str]]]]],
) -> None:
    """
    Closes all file objects in an httpx files list to prevent resource leaks.

    Args:
        httpx_files: The list returned by prepare_files_for_httpx()
    """
    if not httpx_files:
        return

    for field_name, (filename, file_obj, mime_type) in httpx_files:
        try:
            if hasattr(file_obj, "close"):
                file_obj.close()
        except Exception as e:
            logging.warning(f"Failed to close file object for {filename}: {e}")


#
# End of utils.py
#######################################################################################################################


# task-19557 Qodo round: actual redirect statuses only. The whole 3xx band
# also contains 304 Not Modified, which is a cache-validation response (no
# `Location`, not a redirect) that conditional-GET callers rely on reaching
# normal processing -- e.g. `get_user_profile_catalog(if_none_match=...)`.
# Treating 304 as a refused redirect would break that path.
_REDIRECT_STATUS_CODES = frozenset({301, 302, 303, 307, 308})


def _validate_timeout(value: Any, field: str) -> float:
    """Reject a nonsensical timeout at the boundary instead of at request time.

    httpx validates none of this -- ``httpx.Timeout(300.0, connect=-5)``,
    ``connect=nan`` and even ``connect="abc"`` are all accepted -- so an
    invalid value would otherwise cross this public boundary and only
    misbehave later, at the request, far from the call that caused it.
    NaN is the sharp case: ``min(nan, cap)`` is ``nan``, which would
    silently defeat the connect ceiling that keeps an unreachable host
    from freezing the app.

    Args:
        value: The candidate timeout, in seconds.
        field: Parameter name, used in the error message.

    Returns:
        The validated timeout as a float.

    Raises:
        TypeError: If ``value`` is not a real number.
        ValueError: If ``value`` is not finite and positive.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(
            f"{field} must be a number of seconds, got {type(value).__name__}"
        )
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"{field} must be a finite, positive number of seconds, got {value!r}"
        )
    return value


async def _raise_if_redirected(response: httpx.Response, endpoint: str) -> None:
    """Refuse a redirect response rather than following it with credentials.

    The shared client carries the ``X-API-KEY`` (and possibly bearer
    ``Authorization``) header and is constructed with
    ``follow_redirects=False`` (see ``_get_client``) specifically so a
    redirect response lands here instead of httpx silently completing
    the hop. There is no legitimate reason for this client to follow a
    redirect -- ``base_url`` is the server the caller explicitly
    configured -- so an actual redirect is treated as hostile/
    misconfigured and refused outright.

    Only ``_REDIRECT_STATUS_CODES`` (301/302/303/307/308) trigger the
    refusal -- NOT the whole 3xx band. 304 Not Modified is a
    cache-validation response, not a redirect (no ``Location``), and
    conditional-GET callers (e.g. ``get_user_profile_catalog``'s
    ``if_none_match``) rely on it reaching normal processing rather
    than being refused here.

    The redirect ``Location`` is deliberately never echoed in the
    raised message -- it is server- (and on a hostile/compromised
    endpoint, attacker-) controlled data, same reasoning as the
    Anthropic/Google redirect-refusal sites in ``LLM_API_Calls.py``.

    Explicitly closes ``response`` before raising. httpx's own
    ``send()``/``stream()`` already release the connection on the
    paths that reach here (an eagerly-read non-streaming response, or
    the ``stream()`` context manager's own ``finally: aclose()``), but
    ``aclose()`` is idempotent and this makes the guarantee explicit
    here rather than resting on a reader's trust of that internal
    contract.

    Args:
        response: The response to inspect.
        endpoint: The request path, used only for the error message.

    Raises:
        APIConnectionError: If ``response`` is an actual redirect.
    """
    from .exceptions import APIConnectionError

    if response.status_code not in _REDIRECT_STATUS_CODES:
        return
    await response.aclose()
    raise APIConnectionError(
        f"Server returned a redirect ({response.status_code}) for "
        f"{endpoint}; refusing to follow with the X-API-KEY/Authorization "
        "credential."
    )


def _raise_api_error_from(error: httpx.HTTPStatusError) -> NoReturn:
    """Translate a non-2xx response into this package's exception family.

    One implementation for all five request primitives. There were five
    hand-rolled copies and only ``_request``'s carried the structured
    ``{"detail": {...}}`` branch, so a tldw_server refusal reached the user
    as the raw httpx text on the other four -- the "schedules task 6
    round 2, D9" regression, still live on both streaming paths. ``detail``
    is handled in all three shapes the server sends: a pydantic validation
    list, a string, or the structured refusal object.

    The streaming primitives read the body before ``raise_for_status()``,
    so ``response.json()`` works here; the ``except`` arm covers a non-JSON
    body and an unread stream (``.text`` then raises ``ResponseNotRead``,
    a ``RuntimeError``, not a ``ValueError``).

    Args:
        error: The ``httpx.HTTPStatusError`` from ``raise_for_status()``.

    Raises:
        AuthenticationError: On 401.
        APIRequestError: On 422.
        APIResponseError: On any other non-2xx status.
    """
    from .exceptions import APIRequestError, APIResponseError, AuthenticationError

    response = error.response
    error_detail = str(error)
    response_data: Any
    try:
        response_data = response.json()
    except Exception:
        try:
            raw_text = response.text
        except Exception:  # unread streaming response
            raw_text = ""
        response_data = {"raw_text": raw_text}
    else:
        detail = (
            response_data.get("detail") if isinstance(response_data, dict) else None
        )
        if isinstance(detail, list) and detail:
            first = detail[0] if isinstance(detail[0], dict) else {}
            loc = ".".join(map(str, first.get("loc", [])))
            error_detail = f"Validation Error: {first.get('msg', '')} for field '{loc}'"
        elif isinstance(detail, str):
            error_detail = detail
        elif isinstance(detail, dict):
            # Structured refusal: tldw_server returns `{"detail": {"code",
            # "message", "details", "retryable"}}` for its deterministic
            # 4xx refusals. Without this branch `error_detail` stayed the
            # raw httpx text ("Client error '409 Conflict' for url ... For
            # more information check: https://developer.mozilla.org/..."),
            # so the server's own explanation was dropped on the floor and
            # callers could only report a generic failure -- exactly what
            # made a 409 `scheduled_task_definition_archived` surface to
            # the user as "this action requires a server connection"
            # (schedules task 6 round 2, D9). `message` is the human
            # sentence, `code` the machine token; prefer the former, fall
            # back to the latter, and only then to the raw text.
            error_detail = str(
                detail.get("message") or detail.get("code") or error_detail
            )

    status_code = response.status_code
    if status_code == 401:
        raise AuthenticationError(
            f"Authentication failed: {error_detail}", response_data=response_data
        )
    if status_code == 422:  # Unprocessable Entity (pydantic validation error)
        raise APIRequestError(
            f"Validation Error: {error_detail}", response_data=response_data
        )
    raise APIResponseError(status_code, error_detail, response_data=response_data)
