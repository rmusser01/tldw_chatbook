"""The one HTTP seam for word bench capture.

Word bench calls are provider calls and follow the LLM_Calls precedent --
direct to the user's configured endpoint, no egress policy. That is only
safe because the endpoint comes from configuration and never from bench
content: a bench that could name its own endpoint would be an SSRF vector.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

import httpx
from loguru import logger

from .models import CellCapture, CellError, PreflightResult, PromptMode, Target
from .normalizer import CONTENT_TOKEN_WINDOW, NormalizerError, normalize_logprobs

#: Pinned neutral sampling. Servers -- llama.cpp especially -- apply samplers
#: BEFORE reporting logprobs, so a server configured with top_k=40 would make
#: every number an artifact of that setting. temperature is 1.0, NOT 0:
#: temperature zero collapses the distribution being observed.
NEUTRAL_SAMPLER: dict[str, Any] = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

#: Chat mode must look past leading control tokens, so it asks for a window
#: exactly as wide as the normalizer will search -- one literal 8, not two.
CHAT_TOKEN_WINDOW = CONTENT_TOKEN_WINDOW

#: Distribution sanity canary. Confirming a target RETURNS logprobs is not the
#: same as confirming they mean anything: a heavily chat-tuned model was
#: observed continuing this prompt with "thought" rather than " Paris".
CANARY_PROMPT = "The capital of France is"
CANARY_EXPECT = (" Paris", "Paris")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_status_code(detail: str) -> Optional[int]:
    try:
        return int(detail)
    except (TypeError, ValueError):
        return None


def _preflight_state_for_error(result: CellError) -> str:
    """Classify a failed canary capture into a preflight state.

    A 4xx status means the server was reachable and understood enough to
    respond, but rejected the request outright -- that reads as Blocked, not
    Unavailable. A 5xx, or any transport-level failure (connection refused,
    timeout, DNS), means the server could not be reached in any way that
    would let it reject anything, so it stays Unavailable.

    Within the 4xx family, a 404 on the request URL is a distinct, reliable
    signal rather than a guess: ``_build_request`` posts to a FIXED path
    (``/v1/completions`` or ``/v1/chat/completions``) that this module picks
    from ``mode``, so a 404 means THAT ROUTE does not exist on this server --
    exactly the design spec's "raw mode unsupported by endpoint" case, and
    unlike inferring an unobserved JSON response *shape* (which
    ``normalizer.py``'s module docstring rules out), interpreting a standard
    HTTP status code by its own defined meaning is not a guess. Every OTHER
    4xx (401, 400, 422, 429, ...) reports a rejection for some other reason
    a bare status code cannot distinguish, and stays the more generic
    ``no_logprobs`` state (both render the same "Blocked" label; see
    ``models._STATUS_LABELS``).
    """
    if result.reason == "http_error":
        status_code = _parse_status_code(result.detail)
        if status_code == 404:
            return "mode_unsupported"
        if status_code is not None and 400 <= status_code < 500:
            return "no_logprobs"
        return "unreachable"
    if result.reason in ("unreachable", "no_logprobs", "no_content_token"):
        return result.reason
    return "no_logprobs"


class WordBenchCaptureClient:
    """Captures one next-token distribution per call.

    Holds ONE ``httpx.AsyncClient`` for the lifetime of this instance
    (created lazily, on the first request) rather than opening a fresh
    connection per call -- a 100+ cell grid against the same target would
    otherwise pay a new TCP/TLS handshake for every single cell instead of
    reusing keep-alive connections. ``httpx.AsyncClient`` is documented as
    safe for concurrent use by multiple coroutines, which is what
    ``WordBenchRunner`` does under ``BenchConfig.concurrency > 1`` -- several
    ``capture()`` calls for the SAME target never overlap in practice
    (the runner never dispatches two in-flight requests to one target at
    once), but calls for DIFFERENT targets sharing a client instance would
    be fine either way.

    Call ``aclose()`` (or use this object as an ``async with`` context
    manager) when done with it to release the underlying connection pool;
    ``WordBenchRunner.run`` does this automatically for every client it
    creates, after the whole run (including any concurrent in-flight
    requests) has finished -- never while a request could still be using it.
    """

    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: float = 120.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._transport = transport
        self._client: Optional[httpx.AsyncClient] = None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _ensure_client(self) -> httpx.AsyncClient:
        """Build the pooled client on first use. No ``await`` happens
        between the ``None`` check and the assignment, so this is safe to
        call from multiple concurrently-running coroutines without a lock:
        asyncio is cooperative, and nothing here yields control in between."""
        if self._client is None:
            kwargs: dict[str, Any] = {"timeout": self._timeout}
            if self._transport is not None:
                kwargs["transport"] = self._transport
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def aclose(self) -> None:
        """Release the pooled connection. Safe to call even if no request
        was ever made (the client is created lazily), and safe to call more
        than once."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def __aenter__(self) -> "WordBenchCaptureClient":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    def _build_request(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> tuple[str, dict[str, Any]]:
        payload: dict[str, Any] = {"model": target.model_id, **NEUTRAL_SAMPLER}
        if mode == "raw":
            prompt = f"{target.prefix}{snippet}" if target.prefix else snippet
            payload.update({"prompt": prompt, "max_tokens": 1, "logprobs": top_k})
            return f"{self._base_url}/v1/completions", payload

        messages: list[dict[str, str]] = []
        if target.system_prompt:
            messages.append({"role": "system", "content": target.system_prompt})
        messages.append({"role": "user", "content": snippet})
        payload.update(
            {
                "messages": messages,
                "max_tokens": CHAT_TOKEN_WINDOW,
                "logprobs": True,
                "top_logprobs": top_k,
            }
        )
        return f"{self._base_url}/v1/chat/completions", payload

    async def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        client = self._ensure_client()
        response = await client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def capture(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> CellCapture | CellError:
        """Measure one cell. Never raises -- failures become CellError."""
        url, payload = self._build_request(snippet, target, mode, top_k)
        try:
            data = await self._post(url, payload)
        except httpx.HTTPStatusError as exc:
            return CellError(reason="http_error", detail=f"{exc.response.status_code}")
        except httpx.HTTPError as exc:
            return CellError(reason="unreachable", detail=str(exc))
        except ValueError as exc:
            # response.json() raises a ValueError (json.JSONDecodeError) when
            # the body isn't JSON at all -- e.g. a proxy returning an HTML
            # error page with a 200 status. One malformed response must not
            # abort an entire multi-hundred-cell run.
            return CellError(reason="bad_response", detail=f"invalid JSON body: {exc}")

        try:
            tokens, offset = normalize_logprobs(data, want_content_token=(mode == "chat"))
        except NormalizerError as exc:
            return CellError(reason=exc.code, detail=str(exc))
        except (KeyError, TypeError) as exc:
            # A malformed top_logprobs entry (missing "token"/"logprob", or a
            # value of the wrong type) must not raise past this call either.
            return CellError(reason="bad_response", detail=f"malformed entry: {exc!r}")

        return CellCapture(
            prompt_mode=mode,
            k_requested=top_k,
            k_returned=len(tokens),
            content_offset=offset,
            top_k=tuple(tokens),
            canary="unchecked",
            captured_at=_utcnow(),
        )

    async def preflight(
        self, target: Target, mode: PromptMode, top_k: int
    ) -> PreflightResult:
        """Resolve a target's readiness, including distribution sanity.

        A degenerate canary does NOT block the run -- a target whose raw
        continuation is out-of-distribution may be exactly what a user wants
        to study. It downgrades to a warned state that every cell carries.

        The canary prompt is sent through the SAME steering (``prefix`` in
        raw mode, ``system_prompt`` in chat mode) as every measured cell for
        this target, never a bare, unsteered request -- readiness is a
        property of what the run will actually send, not of the endpoint in
        the abstract. This means a legitimate, working steering prefix can
        itself push the canary continuation out of distribution and warn
        ``degenerate`` on that column even though nothing is actually wrong;
        that is a real, honest signal about THIS bench's specific
        configuration, not a defect in the target, and any UI surfacing this
        state must not present it as an unqualified target failure.
        """
        result = await self.capture(CANARY_PROMPT, target, mode, top_k)
        checked_at = _utcnow()

        if isinstance(result, CellError):
            state = _preflight_state_for_error(result)
            return PreflightResult(
                state=state, k_returned=None, canary="unchecked",
                detail=result.detail, checked_at=checked_at,
            )

        observed = {tok.token for tok in result.top_k}
        canary = "pass" if observed & set(CANARY_EXPECT) else "degenerate"
        if canary == "degenerate":
            logger.warning(
                "Word bench canary degenerate for target {}: {!r} continued with {!r}",
                target.name,
                CANARY_PROMPT,
                [t.token for t in result.top_k[:3]],
            )
        return PreflightResult(
            state="ok", k_returned=result.k_returned, canary=canary,
            checked_at=checked_at,
        )
