"""Provider-failure classification shared by chat and agent runtimes.

TASK-335: the agent service captures provider exceptions on its own thread
(`run_turn`'s catch-all) and previously stamped raw ``str(exc)`` — httpx's
status line plus MDN boilerplate — into the STEP_ERROR summary that becomes
user-facing failure copy, while the response body's actionable message was
discarded. Both the Console controller and the agent service now classify
through this module.
"""

import asyncio


_MDN_BOILERPLATE_MARKER = "For more information check:"


def _provider_error_body_detail(response: object) -> str:
    """Best-effort human detail from a provider error response body.

    Providers put the actionable message in the body (e.g. llama.cpp's
    "image input is not supported - hint: you may need to provide the
    mmproj"); httpx's ``str(exc)`` carries only the status line plus MDN
    boilerplate. JSON bodies are probed for the common message keys; plain
    text is used as-is. The result is whitespace-collapsed and truncated.
    """
    try:
        text = (getattr(response, "text", "") or "").strip()
    except Exception:
        return ""
    if not text:
        return ""
    detail = text
    try:
        import json as _json

        payload = _json.loads(text)
        if isinstance(payload, dict):
            candidate = payload.get("error")
            if isinstance(candidate, dict):
                candidate = candidate.get("message") or candidate.get("detail")
            candidate = (
                candidate
                or payload.get("message")
                or payload.get("detail")
            )
            if isinstance(candidate, str) and candidate.strip():
                detail = candidate
    except (ValueError, TypeError):
        pass
    detail = " ".join(str(detail).split())
    if len(detail) > 240:
        detail = detail[:237] + "..."
    return detail


def describe_stream_failure(exc: BaseException) -> str:
    """Return user-facing copy classifying a provider stream failure.

    ``str(exc)`` alone can be empty (observed live as ``"Provider stream
    failed: "`` rendering ``"[failed]"``), so the failure class is always
    included in user terms: timeout vs connection vs HTTP status.

    Args:
        exc: The exception raised by the provider stream.

    Returns:
        A short, user-readable failure description that is never empty.
    """
    detail = str(exc).strip()
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None) or getattr(
        exc, "status_code", None
    )
    exc_name = type(exc).__name__
    lowered_name = exc_name.lower()

    if (
        isinstance(exc, (asyncio.TimeoutError, TimeoutError))
        or "timeout" in lowered_name
    ):
        summary = "request timed out waiting for the provider"
    elif isinstance(
        exc, ConnectionRefusedError
    ) or "connectrefused" in lowered_name.replace("_", ""):
        summary = "connection refused - is the provider server running?"
    elif isinstance(exc, ConnectionError) or "connect" in lowered_name:
        summary = "could not connect to the provider"
    elif status_code is not None:
        summary = f"provider returned HTTP {status_code}"
        # TASK-335: the response BODY carries the provider's actionable
        # message; str(exc) is just the status line + MDN boilerplate.
        body_detail = _provider_error_body_detail(response)
        if body_detail:
            detail = body_detail
    else:
        summary = f"{exc_name} error"

    if detail:
        # Never emit httpx's MDN "For more information check: https://…"
        # boilerplate into a chat transcript (TASK-335).
        marker = detail.find(_MDN_BOILERPLATE_MARKER)
        if marker != -1:
            detail = detail[:marker].rstrip()
        detail = " ".join(detail.split())
    if detail and detail.lower() != summary.lower():
        return f"{summary} ({detail})"
    return summary
