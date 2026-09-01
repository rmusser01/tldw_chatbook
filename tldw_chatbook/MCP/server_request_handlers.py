"""TASK-26029: server-initiated MCP request handlers (sampling + elicitation).

Previously the client answered ``ping`` and returned ``-32601`` for every
server-initiated request, so a server asking the client to run a completion
(``sampling/createMessage``) or to ask the user a question
(``elicitation/create``) could not work. This module supplies the two handlers,
wiring them to the existing chat provider and approval surface through injected
callables so the policy/gating/bounding logic is pure and testable, while the
impure provider/UI wiring lives at the call site.

Fail-closed by construction: sampling is denied unless the user has explicitly
allowed a server (AC#2) and is bounded by rate + token budget (AC#3);
elicitation that asks for credentials or free-form secrets is refused rather
than presented (AC#5); a declined or absent surface returns a well-formed
protocol error rather than hanging (AC#6).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union


# JSON-RPC error codes.
_METHOD_NOT_FOUND = -32601
_INVALID_PARAMS = -32602
# Application-defined: a request the client refused by policy.
_REQUEST_REFUSED = -32001
_REQUEST_DECLINED = -32002


@dataclass(frozen=True)
class JsonRpcError:
    """A well-formed JSON-RPC error result (never raised across the wire)."""

    code: int
    message: str

    def to_payload(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class SamplingPolicy:
    """Per-server sampling permission and bounds. Default: not allowed (AC#2)."""

    allowed: bool = False
    max_requests_per_minute: int = 0
    max_total_tokens: int = 0


@dataclass
class SamplingBudget:
    """Mutable per-server usage accounting for rate + token bounds (AC#3)."""

    request_times: List[float] = field(default_factory=list)
    tokens_used: int = 0


@dataclass(frozen=True)
class SamplingDecision:
    allow: bool
    reason: str


def evaluate_sampling_request(
    policy: SamplingPolicy,
    budget: SamplingBudget,
    requested_tokens: int,
    now: float,
) -> SamplingDecision:
    """Decide whether one sampling request may proceed (pure).

    Denies unless the server is explicitly allowed (AC#2), the per-minute rate
    is under the cap, and the token budget would not be exceeded (AC#3).
    """
    if not policy.allowed:
        return SamplingDecision(False, "sampling is not allowed for this server (no consent)")

    if policy.max_requests_per_minute > 0:
        recent = [t for t in budget.request_times if now - t < 60.0]
        if len(recent) >= policy.max_requests_per_minute:
            return SamplingDecision(
                False,
                f"sampling rate limit reached ({policy.max_requests_per_minute}/min)",
            )

    if policy.max_total_tokens > 0:
        projected = budget.tokens_used + max(0, requested_tokens)
        if projected > policy.max_total_tokens:
            return SamplingDecision(
                False,
                f"sampling token budget exhausted ({budget.tokens_used}/{policy.max_total_tokens})",
            )

    return SamplingDecision(True, "allowed")


# Field/prompt shapes that indicate a request for a credential or secret.
_SECRET_NAME_RE = re.compile(
    r"(?:password|passwd|secret|api[_-]?key|apikey|token|credential|private[_-]?key"
    r"|passphrase|access[_-]?key|client[_-]?secret|auth)",
    re.IGNORECASE,
)


def screen_elicitation_for_secrets(request: Dict[str, Any]) -> Optional[str]:
    """Return a refusal reason if an elicitation asks for secrets, else None (AC#5)."""
    message = str(request.get("message") or "")
    if _SECRET_NAME_RE.search(message):
        return "elicitation asks for a credential/secret in its prompt"

    schema = request.get("requestedSchema") or {}
    properties = schema.get("properties") if isinstance(schema, dict) else None
    if isinstance(properties, dict):
        for name, spec in properties.items():
            if _SECRET_NAME_RE.search(str(name)):
                return f"elicitation asks for a secret-shaped field '{name}'"
            if isinstance(spec, dict):
                fmt = str(spec.get("format") or "")
                if fmt.lower() == "password" or _SECRET_NAME_RE.search(fmt):
                    return f"elicitation field '{name}' has a secret format"
                title = str(spec.get("title") or "") + " " + str(spec.get("description") or "")
                if _SECRET_NAME_RE.search(title):
                    return f"elicitation field '{name}' describes a secret"
    return None


# Injected impure callables.
CompleteFn = Callable[[List[Dict[str, Any]], int, Optional[str]], Awaitable[str]]
ElicitFn = Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]

HandleResult = Union[Dict[str, Any], JsonRpcError]


class ServerRequestDispatcher:
    """Dispatch server-initiated sampling/elicitation to injected handlers.

    ``complete_fn`` runs a completion through the existing chat provider and
    returns the assistant text. ``elicit_fn`` presents the question through the
    approval-style surface and returns the user's response dict, or ``None``
    when declined / no surface is available. Either may be ``None``, in which
    case that method reports method-not-found (AC#7: servers not using these
    methods, or a client without the wiring, are unaffected).
    """

    def __init__(
        self,
        *,
        sampling_policy: Optional[SamplingPolicy] = None,
        sampling_budget: Optional[SamplingBudget] = None,
        complete_fn: Optional[CompleteFn] = None,
        elicit_fn: Optional[ElicitFn] = None,
        now_fn: Optional[Callable[[], float]] = None,
    ):
        self.sampling_policy = sampling_policy or SamplingPolicy()
        self.sampling_budget = sampling_budget or SamplingBudget()
        self._complete_fn = complete_fn
        self._elicit_fn = elicit_fn
        self._now_fn = now_fn

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        from time import monotonic
        return monotonic()

    async def handle(self, method: str, params: Dict[str, Any]) -> HandleResult:
        if method == "sampling/createMessage":
            return await self._handle_sampling(params or {})
        if method == "elicitation/create":
            return await self._handle_elicitation(params or {})
        return JsonRpcError(_METHOD_NOT_FOUND, f"Method not found: {method}")

    async def _handle_sampling(self, params: Dict[str, Any]) -> HandleResult:
        if self._complete_fn is None:
            return JsonRpcError(_METHOD_NOT_FOUND, "Method not found: sampling/createMessage")

        messages = params.get("messages")
        if not isinstance(messages, list):
            return JsonRpcError(_INVALID_PARAMS, "sampling requires a messages array")
        requested_tokens = params.get("maxTokens")
        try:
            requested_tokens = int(requested_tokens)
        except (TypeError, ValueError):
            requested_tokens = 0

        now = self._now()
        decision = evaluate_sampling_request(
            self.sampling_policy, self.sampling_budget, requested_tokens, now
        )
        if not decision.allow:
            return JsonRpcError(_REQUEST_REFUSED, decision.reason)

        model_hint = None
        prefs = params.get("modelPreferences")
        if isinstance(prefs, dict):
            hints = prefs.get("hints")
            if isinstance(hints, list) and hints and isinstance(hints[0], dict):
                model_hint = hints[0].get("name")

        try:
            text = await self._complete_fn(messages, requested_tokens, model_hint)
        except Exception as exc:  # provider failure -> well-formed error, no hang
            return JsonRpcError(_REQUEST_DECLINED, f"sampling completion failed: {exc}")

        # record usage AFTER a successful call
        self.sampling_budget.request_times.append(now)
        self.sampling_budget.tokens_used += max(0, requested_tokens)

        return {
            "role": "assistant",
            "content": {"type": "text", "text": text},
            "model": model_hint or "tldw_chatbook",
            "stopReason": "endTurn",
        }

    async def _handle_elicitation(self, params: Dict[str, Any]) -> HandleResult:
        if self._elicit_fn is None:
            return JsonRpcError(_METHOD_NOT_FOUND, "Method not found: elicitation/create")

        refusal = screen_elicitation_for_secrets(params)
        if refusal is not None:
            return JsonRpcError(_REQUEST_REFUSED, refusal)

        message = str(params.get("message") or "")
        schema = params.get("requestedSchema") or {}
        try:
            response = await self._elicit_fn(message, schema)
        except Exception as exc:
            return JsonRpcError(_REQUEST_DECLINED, f"elicitation failed: {exc}")

        if response is None:
            return JsonRpcError(_REQUEST_DECLINED, "elicitation declined")
        return response
