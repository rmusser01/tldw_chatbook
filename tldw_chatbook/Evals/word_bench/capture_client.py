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

#: task-1691: how many tokens the SEPARATE raw-mode continuation request
#: (see `_capture_raw_continuation`) asks for. Long enough that chat-template
#: scaffolding a heavily chat-tuned model emits in raw mode (the motivating
#: UAT: `'<|channel><|channel>thought\n<channel|>The sky is **blue'`, several
#: control tokens before any real content) reliably clears into legible text
#: rather than stopping mid-scaffolding; short enough that this one extra
#: per-TARGET preflight request (never per-cell -- see this module's and
#: `preflight`'s own docstrings) stays cheap.
CONTINUATION_MAX_TOKENS = 24

#: task-1691: upper bound on what a captured continuation is STORED as, in
#: characters. This is a storage cap on the run snapshot, not a display cap --
#: the UI is free to preview an even shorter window of this.
CONTINUATION_CHAR_CAP = 200


def _cap_continuation(text: str) -> str:
    """Bound what preflight stores for a continuation preview.

    Args:
        text: The raw generated continuation text.

    Returns:
        ``text`` truncated to ``CONTINUATION_CHAR_CAP`` characters.
    """
    return text[:CONTINUATION_CHAR_CAP]


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _parse_status_code(detail: str) -> Optional[int]:
    """Recover the numeric status code CellError(reason="http_error") stashed
    as a string in its ``detail`` field.

    Args:
        detail: A ``CellError.detail`` value. Only meaningful when the
            error's ``reason`` is ``"http_error"``, where ``capture()``
            writes it as ``f"{status_code}"``; any other detail string
            (a transport error message, a JSON-decode error, ...) is not a
            status code and falls through to the except clause below.

    Returns:
        The status code, or ``None`` if ``detail`` is not a bare integer
        string.
    """
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

    Args:
        result: The ``CellError`` returned by a canary ``capture()`` call.

    Returns:
        One of the ``PreflightResult.state`` values: ``"mode_unsupported"``,
        ``"no_logprobs"``, ``"unreachable"``, or ``"no_content_token"``.
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
        """Args:
            base_url: The provider endpoint's base URL (trailing ``/``
                stripped); ``/v1/completions`` or ``/v1/chat/completions``
                is appended per call by ``_build_request``.
            api_key: Sent as ``Authorization: Bearer <api_key>`` when set;
                omitted entirely otherwise.
            timeout: Per-request timeout, in seconds, for the pooled
                ``httpx.AsyncClient`` created on first use.
            transport: Overrides the client's transport -- tests pass an
                ``httpx.MockTransport`` here so no real network call is
                made.
        """
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
        asyncio is cooperative, and nothing here yields control in between.

        Returns:
            This instance's single pooled ``httpx.AsyncClient``, creating it
            first if this is the first call.
        """
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
        """Returns:
            This instance, so it can be used as
            ``async with WordBenchCaptureClient(...) as client:``.
        """
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        """Releases the pooled connection via ``aclose()`` on context exit.

        Args:
            *exc_info: The exception type, value, and traceback if the
                ``with`` body raised; unused -- ``aclose()`` always runs and
                never suppresses the original exception.
        """
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
        """POST one request through the pooled client and return its parsed
        JSON body.

        Args:
            url: Full request URL, already mode-selected by
                ``_build_request``.
            payload: The JSON request body.

        Returns:
            The decoded JSON response body.

        Raises:
            httpx.HTTPStatusError: If the response status is 4xx/5xx.
            httpx.HTTPError: For transport-level failures (connection
                refused, timeout, DNS, ...).
            ValueError: If the response body is not valid JSON.
        """
        client = self._ensure_client()
        response = await client.post(url, json=payload, headers=self._headers())
        response.raise_for_status()
        return response.json()

    async def capture(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> CellCapture | CellError:
        """Measure one cell. Never raises -- failures become CellError.

        Args:
            snippet: The text whose continuation is measured. Sent as the
                raw completion prompt in ``"raw"`` mode (after ``target``'s
                ``prefix``, if any), or as the user message in ``"chat"``
                mode.
            target: The (provider, model, steering) column being measured.
            mode: ``"raw"`` or ``"chat"`` -- selects both the request shape
                (``_build_request``) and how the response is normalized
                (``normalize_logprobs``'s ``want_content_token``).
            top_k: Requested top-K distribution width.

        Returns:
            A ``CellCapture`` with ``canary="unchecked"`` -- turning that
            into the target's real preflight verdict is
            ``WordBenchRunner._stamp_canary``'s job, not this method's -- or
            a ``CellError`` describing why the cell could not be measured
            (never raised).
        """
        result, _payload = await self._capture_with_payload(snippet, target, mode, top_k)
        return result

    async def _capture_with_payload(
        self, snippet: str, target: Target, mode: PromptMode, top_k: int
    ) -> tuple[CellCapture | CellError, Optional[dict[str, Any]]]:
        """Same behaviour as ``capture()``, but also hands back the raw
        decoded response body (or ``None`` on any failure).

        Exists so ``preflight`` can read the actual generated text off the
        SAME response it uses for the canary verdict, in chat mode, without
        a second request -- see ``_resolve_continuation``. ``capture()``
        itself is unchanged behaviourally; this is a pure factoring split,
        not a new code path for the cell-measurement contract.

        Returns:
            ``(result, payload)`` -- ``payload`` is the decoded JSON body
            when one was successfully received and parsed (even if
            ``normalize_logprobs`` then failed on its shape), or ``None``
            when the request itself never produced a body (transport
            failure, non-2xx status, or an undecodable body).
        """
        url, payload = self._build_request(snippet, target, mode, top_k)
        try:
            data = await self._post(url, payload)
        except httpx.HTTPStatusError as exc:
            return CellError(reason="http_error", detail=f"{exc.response.status_code}"), None
        except httpx.HTTPError as exc:
            return CellError(reason="unreachable", detail=str(exc)), None
        except ValueError as exc:
            # response.json() raises a ValueError (json.JSONDecodeError) when
            # the body isn't JSON at all -- e.g. a proxy returning an HTML
            # error page with a 200 status. One malformed response must not
            # abort an entire multi-hundred-cell run.
            return CellError(reason="bad_response", detail=f"invalid JSON body: {exc}"), None

        try:
            tokens, offset = normalize_logprobs(data, want_content_token=(mode == "chat"))
        except NormalizerError as exc:
            return CellError(reason=exc.code, detail=str(exc)), data
        except (KeyError, TypeError) as exc:
            # A malformed top_logprobs entry (missing "token"/"logprob", or a
            # value of the wrong type) must not raise past this call either.
            return CellError(reason="bad_response", detail=f"malformed entry: {exc!r}"), data

        return CellCapture(
            prompt_mode=mode,
            k_requested=top_k,
            k_returned=len(tokens),
            content_offset=offset,
            top_k=tuple(tokens),
            canary="unchecked",
            captured_at=_utcnow(),
        ), data

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

        Args:
            target: The column being checked, steered exactly as it would be
                during the real run.
            mode: ``"raw"`` or ``"chat"`` -- must match the mode the actual
                run will use, since readiness is mode-specific.
            top_k: The top-K width the actual run will request.

        Returns:
            A ``PreflightResult`` carrying the resolved ``state`` (see
            ``models._STATUS_LABELS`` for how it maps to a UI label),
            ``k_returned``, ``canary`` verdict, and (task-1691) a short,
            best-effort ``continuation`` -- a generated continuation of
            ``CANARY_PROMPT`` through this same steering/mode, meant to make
            a degenerate canary's behaviour (e.g. chat-template scaffolding
            leaking into raw mode) legible as text rather than only as a
            distribution. Never raises -- a failed canary capture becomes a
            non-``"ok"`` ``state``, not an exception, and a failed or empty
            continuation capture becomes ``""``, exactly like the canary's
            own "never raises" contract.
        """
        result, canary_payload = await self._capture_with_payload(
            CANARY_PROMPT, target, mode, top_k
        )
        checked_at = _utcnow()

        if isinstance(result, CellError):
            state = _preflight_state_for_error(result)
            return PreflightResult(
                state=state, k_returned=None, canary="unchecked",
                detail=result.detail, checked_at=checked_at, continuation="",
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
        continuation = await self._resolve_continuation(target, mode, canary_payload)
        return PreflightResult(
            state="ok", k_returned=result.k_returned, canary=canary,
            checked_at=checked_at, continuation=continuation,
        )

    async def _resolve_continuation(
        self, target: Target, mode: PromptMode, canary_payload: Optional[dict[str, Any]]
    ) -> str:
        """A short, best-effort continuation for the readiness surface.

        Chosen approach (task-1691's option (a), separate-request): chat
        mode's canary call above already asked for ``CHAT_TOKEN_WINDOW``
        tokens and this method's caller (``preflight``) already has that
        response's raw payload in hand, discarded until now -- salvaging the
        generated text from it costs nothing extra. Raw mode's canary is
        deliberately ``max_tokens: 1`` (see this class's ``NEUTRAL_SAMPLER``/
        module docstring and ``preflight``'s own docstring), so there is no
        free text to salvage there; a real continuation needs a genuinely
        separate request (``_capture_raw_continuation``), issued and parsed
        entirely independently of the canary response already used to
        compute ``canary`` above -- it can never perturb that verdict, no
        matter what it returns or how it fails, because nothing it produces
        is ever passed through ``normalize_logprobs``.

        Never raises: any failure (transport, HTTP status, malformed body,
        missing/non-string text field) degrades to ``""``, matching the
        canary capture's own "never raises" contract.

        Args:
            target: The column being checked, steered exactly as ``preflight``
                steered its own canary request.
            mode: ``"raw"`` or ``"chat"`` -- selects salvage vs. a fresh
                request, same as everywhere else in this class.
            canary_payload: The decoded JSON body from ``preflight``'s own
                canary request, or ``None`` if that request never produced
                one. Only consulted in chat mode.

        Returns:
            The captured continuation, capped by ``_cap_continuation``, or
            ``""``.
        """
        if mode == "chat":
            return _extract_chat_continuation(canary_payload)
        return await self._capture_raw_continuation(target)

    async def _capture_raw_continuation(self, target: Target) -> str:
        """Issue the SEPARATE, continuation-only raw completion request.

        Deliberately built from scratch rather than through
        ``_build_request``/``_capture_with_payload``: this request never
        requests ``logprobs`` at all (only generated text is wanted) and its
        response is never handed to ``normalize_logprobs``, so it is
        structurally incapable of influencing the canary verdict computed
        from the separate ``max_tokens: 1`` canary request.

        Steered identically to the canary request (``target.prefix``
        prepended, same as ``_build_request``'s raw-mode branch) -- see
        ``preflight``'s docstring: readiness, and now its continuation, are
        properties of what the run will actually send.

        Never raises: any failure (transport, HTTP status, malformed body,
        missing/non-string ``text``) degrades to ``""``.

        Args:
            target: The column being checked; only ``model_id`` and
                ``prefix`` are used.

        Returns:
            The captured continuation, capped by ``_cap_continuation``, or
            ``""``.
        """
        prompt = f"{target.prefix}{CANARY_PROMPT}" if target.prefix else CANARY_PROMPT
        payload: dict[str, Any] = {
            "model": target.model_id,
            **NEUTRAL_SAMPLER,
            "prompt": prompt,
            "max_tokens": CONTINUATION_MAX_TOKENS,
        }
        url = f"{self._base_url}/v1/completions"
        try:
            data = await self._post(url, payload)
        except httpx.HTTPError:
            return ""
        except ValueError:
            # response.json() raises ValueError (json.JSONDecodeError) for a
            # non-JSON 200 body (e.g. an HTML proxy error page) -- same
            # failure class _capture_with_payload guards against.
            return ""

        try:
            text = data["choices"][0]["text"]
        except (KeyError, IndexError, TypeError):
            return ""
        if not isinstance(text, str):
            return ""
        return _cap_continuation(text)


def _extract_chat_continuation(payload: Optional[dict[str, Any]]) -> str:
    """Salvage the generated continuation text out of a chat-mode canary
    response, never raising.

    Args:
        payload: The decoded JSON body from a chat-mode canary request, or
            ``None``.

    Returns:
        The message content, capped by ``_cap_continuation``, or ``""`` if
        ``payload`` is ``None``, unrecognized, or carries no string
        ``message.content``.
    """
    if not payload:
        return ""
    try:
        text = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        return ""
    if not isinstance(text, str):
        return ""
    return _cap_continuation(text)
