# tldw_chatbook/tldw_api/exceptions.py
#
#
#######################################################################################################################
#
# Functions:

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NoReturn

if TYPE_CHECKING:
    import httpx

    from .sync_schemas import SyncPersonalContextBootstrapAttention


class TLDWAPIError(Exception):
    """Base exception for tldw_api errors."""

    pass


class APIConnectionError(TLDWAPIError):
    """Raised for network or connection issues."""

    pass


class APIRequestError(TLDWAPIError):
    """Raised for errors in constructing or sending the request (e.g., bad data)."""

    def __init__(self, message: str, response_data: dict | None = None):
        super().__init__(message)
        self.response_data = response_data or {}


class APIResponseError(TLDWAPIError):
    """Raised for non-2xx responses or issues parsing the response."""

    def __init__(self, status_code: int, message: str, response_data: dict = None):
        super().__init__(f"API Error {status_code}: {message}")
        self.status_code = status_code
        self.response_data = response_data or {}


class PersonalContextBootstrapAttentionError(TLDWAPIError):
    """Raise one strictly validated, content-free bootstrap attention result."""

    def __init__(self, attention: "SyncPersonalContextBootstrapAttention") -> None:
        super().__init__("personal_context_bootstrap_attention_required")
        self.attention = attention


class AuthenticationError(TLDWAPIError):
    """Raised for authentication failures."""

    def __init__(self, message: str, response_data: dict | None = None):
        super().__init__(message)
        self.response_data = response_data or {}


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


#
# End of tldw_chatbook/tldw_api/exceptions.py
########################################################################################################################
