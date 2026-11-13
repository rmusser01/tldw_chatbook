"""The urllib3 retry a requests-based chat handler mounts on its session.

TASK-34100.2 AC#18 (review round 1, F1; review round 2, R3-1). The OpenAI,
Anthropic, Cohere, Hugging Face and Google handlers, the local
OpenAI-compatible family (``_chat_with_openai_compatible_local_server``:
Custom / Custom 2, llama.cpp, vLLM, Ollama, TabbyAPI, Aphrodite, MLX-LM,
Oobabooga) and KoboldCpp each built ``Retry(status_forcelist=[429, 500, 502,
503, 504])`` with ``raise_on_status`` left on. urllib3 then turns
every 429, and every 5xx that outlasts the retries, into
``requests.exceptions.RetryError`` -- a ``RequestException`` -- whatever
``total`` is, so the response never reached ``raise_for_status``: the
dispatcher and the handlers reported a rate limit (OpenAI's
``insufficient_quota`` on a new account is the most common first-chat
failure) as a 504 'Network error', and the provider's own reason was lost.

The policy here:

* never retries a 429 in the transport -- a per-minute or per-day quota
  cannot clear in a 1-4 s backoff, and a retry only spends more of it. That
  includes urllib3's own Retry-After rule, which retries a 429 or 503 that
  carries the header even when the status is not in ``status_forcelist`` and
  sleeps for as long as the header says;
* still retries the transient 5xx statuses, with the caller's backoff;
* hands the last response back (``raise_on_status=False``), so the handler's
  ``raise_for_status`` classifies it with the provider's body.

Every requests-based chat handler mounts this retry now, except OpenAI's
non-streaming call, which mounts none (dev's TASK-38600; TASK-34100.2
pre-merge review I-1): its 429 and 5xx reach ``raise_for_status`` on the
first answer, and a read timeout is never re-sent. ``exhausted_retry_
status`` reads the status out of a ``RetryError`` that still reaches the
dispatcher (a handler added later with its own ``Retry``), so it is
classified by its status instead of being called a network error.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from requests.exceptions import RetryError
from urllib3.exceptions import MaxRetryError
from urllib3.util.retry import Retry

#: Server statuses worth one more try after a short backoff.
TRANSIENT_SERVER_STATUSES: tuple[int, ...] = (500, 502, 503, 504)

# urllib3's ``ResponseError.SPECIFIC_ERROR`` is "too many {status_code} error
# responses"; requests wraps it in ``RetryError``'s message.
_EXHAUSTED_STATUS = re.compile(r"too many (\d{3}) error responses")


def chat_transport_retry(
    total: int,
    backoff_factor: float,
    *,
    statuses: Iterable[int] = TRANSIENT_SERVER_STATUSES,
    read: int | None = None,
) -> Retry:
    """Build the retry a chat handler mounts on its ``requests`` session.

    Args:
        total: How many retries the handler's config allows (already passed
            through ``llm_retry_count`` for sensitive requests).
        backoff_factor: urllib3's exponential backoff factor, in seconds.
        statuses: The 5xx statuses to retry; 429 is never one of them.
        read: urllib3's read-error retry budget; ``None`` leaves it to
            ``total``. The local family passes 0 (TASK-34100.5 review B-F1:
            a read timeout means the server is still working on the prompt,
            and a re-send makes it start over).

    Returns:
        A ``Retry`` that returns the final response instead of raising.

    Raises:
        ValueError: ``statuses`` names 429.
    """
    forcelist = sorted({int(status) for status in statuses})
    if 429 in forcelist:
        raise ValueError("a 429 is never retried in the transport")
    return Retry(
        total=total,
        read=read,
        backoff_factor=backoff_factor,
        status_forcelist=forcelist,
        allowed_methods=["POST"],
        raise_on_status=False,
        respect_retry_after_header=False,
    )


def exhausted_retry_status(exc: BaseException) -> int | None:
    """The HTTP status a urllib3 retry gave up on, or None.

    Args:
        exc: Any exception; only a ``requests`` ``RetryError`` or urllib3
            ``MaxRetryError`` that names a status matches.

    Returns:
        The status code urllib3 kept getting, or None when ``exc`` is not an
        exhausted status retry (a refused connection, a timeout, ...).
    """
    if not isinstance(exc, (RetryError, MaxRetryError)):
        return None
    match = _EXHAUSTED_STATUS.search(str(exc))
    return int(match.group(1)) if match else None
