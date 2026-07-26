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
from .normalizer import NormalizerError, normalize_logprobs

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

#: Chat mode must look past leading control tokens, so it asks for a window.
CHAT_TOKEN_WINDOW = 8

#: Distribution sanity canary. Confirming a target RETURNS logprobs is not the
#: same as confirming they mean anything: a heavily chat-tuned model was
#: observed continuing this prompt with "thought" rather than " Paris".
CANARY_PROMPT = "The capital of France is"
CANARY_EXPECT = (" Paris", "Paris")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


class WordBenchCaptureClient:
    """Captures one next-token distribution per call."""

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

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

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
        kwargs: dict[str, Any] = {"timeout": self._timeout}
        if self._transport is not None:
            kwargs["transport"] = self._transport
        async with httpx.AsyncClient(**kwargs) as client:
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

        try:
            tokens, offset = normalize_logprobs(data, want_content_token=(mode == "chat"))
        except NormalizerError as exc:
            reason = "no_content_token" if "no_content_token" in str(exc) else "no_logprobs"
            return CellError(reason=reason, detail=str(exc))

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
        """
        result = await self.capture(CANARY_PROMPT, target, mode, top_k)
        checked_at = _utcnow()

        if isinstance(result, CellError):
            state = result.reason if result.reason != "http_error" else "unreachable"
            if state not in ("unreachable", "no_logprobs", "no_content_token"):
                state = "no_logprobs"
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
