"""Provider-failure classification shared by chat and agent runtimes.

TASK-335: the agent service captures provider exceptions on its own thread
(`run_turn`'s catch-all) and previously stamped raw ``str(exc)`` — httpx's
status line plus MDN boilerplate — into the STEP_ERROR summary that becomes
user-facing failure copy, while the response body's actionable message was
discarded. Both the Console controller and the agent service now classify
through this module.
"""

import asyncio
from collections.abc import Callable

from tldw_chatbook.Chat.Chat_Deps import ChatAPIError, ChatConfigurationError

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
            candidate = candidate or payload.get("message") or payload.get("detail")
            if isinstance(candidate, str) and candidate.strip():
                detail = candidate
    except (ValueError, TypeError):
        pass
    detail = " ".join(str(detail).split())
    # TASK-32808.3: shared truncator, single-char ellipsis (Console renders it).
    from tldw_chatbook.Utils.Utils import truncate

    return truncate(detail, 240)


#: TASK-34100.5 AC#4: the longest failure summary an agent step keeps. The
#: provider's reason (<= 200 chars) plus the per-category fix overran the old
#: 500-character cut, which ended a 404 at '...another model from the.'.
FAILURE_SUMMARY_MAX_CHARS = 1200


def describe_stream_failure(exc: BaseException) -> str:
    """Return user-facing copy classifying a provider stream failure.

    ``str(exc)`` alone can be empty (observed live as ``"Provider stream
    failed: "`` rendering ``"[failed]"``), so the failure class is always
    included in user terms: timeout vs connection vs HTTP status.

    task-2154.16 (FB-06): the exception CLASS NAME is an internal
    classification signal only -- it must never reach user copy. A generic
    ``Exception`` subclass (``RuntimeError``, ``ValueError``, a provider
    SDK's own error type) falls through to the plain "unexpected provider
    error" category (mirroring ``safe_provider_error_copy``), with any
    useful detail (e.g. "Connection refused: ... <url>") kept in the
    parenthesized detail.

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

    stall_seconds = getattr(exc, "timeout_seconds", None)
    if exc_name == "StreamStallError" and isinstance(stall_seconds, (int, float)):
        # TASK-34100.5 AC#4/#5: a stall is its own category with its own
        # fixes, never "unexpected provider error". A self-hosted first-token
        # stall is a model still loading, so it names the cold-load setting;
        # a hosted model loads nothing here (review round 2, R2-F1).
        if getattr(exc, "first_token", False):
            from tldw_chatbook.Chat.provider_readiness import (
                is_self_hosted_provider,
            )

            if not is_self_hosted_provider(getattr(exc, "provider", None)):
                return (
                    f"no first token after {stall_seconds:g} s — the provider "
                    "hasn't started answering. Retry, or wait longer by "
                    "raising chat_defaults.first_token_timeout_seconds in "
                    "config.toml."
                )
            return (
                f"no first token after {stall_seconds:g} s — the model may still "
                "be loading. Wait longer by raising "
                "chat_defaults.first_token_timeout_seconds in config.toml, or "
                "try a smaller model."
            )
        return (
            f"no reply for {stall_seconds:g} s — the provider stopped sending. "
            "Retry, or wait longer by raising "
            "chat_defaults.stream_stall_timeout_seconds in config.toml."
        )
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
    elif isinstance(exc, ChatConfigurationError) and status_code is None:
        # task-32342: a status-less configuration error never reached the
        # provider -- either the request could not be built or its reply
        # could not be read. Calling either a "provider error" blames the
        # wrong party for a client-side bug.
        summary = "the app could not build the request or read the reply"
    elif status_code is not None:
        summary = f"provider returned HTTP {status_code}"
        # TASK-335: the response BODY carries the provider's actionable
        # message; str(exc) is just the status line + MDN boilerplate.
        body_detail = _provider_error_body_detail(response)
        if body_detail:
            detail = body_detail
    else:
        # FB-06 (task-2154.16): never surface the exception class name --
        # "RuntimeError error" means nothing to a user. Plain category
        # only; the actionable detail (connection refused, URL) is kept
        # below in the parenthesized detail.
        summary = "unexpected provider error"

    if detail:
        # Never emit httpx's MDN "For more information check: https://…"
        # boilerplate into a chat transcript (TASK-335).
        marker = detail.find(_MDN_BOILERPLATE_MARKER)
        if marker != -1:
            detail = detail[:marker].rstrip()
        detail = " ".join(detail.split())
    if detail and getattr(exc, "gateway_sentence", False) is True:
        # Review round 2 (V2-F6): the gateway's own finished copy already
        # names the status and the fix; wrapping it again read
        # "HTTP 404 (... Status: 404. ... (Alt+M: Switch model).)." Decided by
        # who raised it, never by the words "Status: N." (TASK-33621.11 AC#5).
        return detail
    if detail and detail.lower() != summary.lower():
        # The closing parenthesis ends the clause; a period inside it read
        # "(... broke.)." once the caller closed the sentence.
        return f"{summary} ({detail.rstrip('.')})"
    return summary


def console_failure_copy(exc: BaseException, lead_in: str) -> str:
    """Return one Console failure sentence for ``exc`` (TASK-33621.11 AC#4).

    The gateway's finished sentence already names the provider, the category
    and the status, so it is shown alone; prefixing it read "Agent run
    failed: Provider error from Anthropic: ... Status: 401.". Only a failure
    the gateway did not describe gets the path's own lead-in.

    Args:
        exc: The failure that ended the turn.
        lead_in: The path's lead-in, e.g. ``"Agent run failed"``.

    Returns:
        One user-facing sentence.
    """
    if isinstance(exc, ChatAPIError) and exc.console_copy:
        return exc.console_copy
    return f"{lead_in}: {describe_stream_failure(exc)}"


#: TASK-335: a provider that cannot read an image fails every later turn of
#: the conversation (history re-sends it), so the refusal names the way out.
IMAGE_REFUSAL_HINT = (
    " This conversation includes an image attachment; if the model can't accept "
    "images, switch to a vision-capable model, or type /rewind and restore to the "
    "prompt that added the image (nothing is deleted; later turns stay a branch)."
)


def image_refusal_hint(failure: object, carries_images: Callable[[], bool]) -> str:
    """Return the image hint only for a provider's image refusal (TASK-33621.11).

    Chosen from the failure's typed verdict -- a 400/413/415/422 whose
    provider body is about image input (``is_image_input_rejection``) --
    never from the failure copy's wording, and never for a 401.

    Args:
        failure: The raised error, or an agent outcome carrying the verdict
            as ``console_image_rejection``.
        carries_images: Whether the conversation holds an image; the hint
            says it does, so without one it would be false.

    Returns:
        The hint, with its leading space, or ``""``.
    """
    if isinstance(failure, BaseException):
        from tldw_chatbook.Chat.provider_error_reason import is_image_input_rejection

        refused = is_image_input_rejection(failure)
    else:
        refused = getattr(failure, "console_image_rejection", False) is True
    return IMAGE_REFUSAL_HINT if refused and carries_images() else ""
