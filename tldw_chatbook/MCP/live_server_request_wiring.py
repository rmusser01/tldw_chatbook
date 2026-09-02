"""TASK-27019: wire server-initiated MCP requests to the LIVE surfaces.

TASK-26029 shipped the fail-closed protocol core (``ServerRequestDispatcher``)
with injected callables. This module supplies the production callables:

- **Sampling** runs through the real chat provider (``chat_api_call``, off the
  event loop), with the provider/model from flat ``[mcp]`` keys
  (``sampling_provider`` / ``sampling_model``) falling back to the
  ``[chat_defaults]`` provider/model. Per-server policy comes from config and
  is DENY by default (AC#3): a server samples only if listed in
  ``[mcp] sampling_allowed_servers``.
- **Elicitation** rides the existing MCP hub approval-request store as a
  CONFIRMATION: the request (message + schema) is saved pending, the hub's
  approvals surface shows it, and approve/deny resolves it. Deliberately the
  honest slice: a ``requestedSchema`` asking for actual field values is
  refused up front (a form-filling surface is separate work), so the user is
  never shown a prompt whose answer we cannot represent.
- **Per-server dispatchers** come from a factory the client calls per
  connection (AC#4), with sampling budgets kept per server across reconnects.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Mapping, Optional

from loguru import logger

from tldw_chatbook.Chat.Chat_Functions import chat_api_call
from tldw_chatbook.config import get_cli_setting
from tldw_chatbook.MCP.local_store import LocalApprovalRequest, LocalMCPStore
from tldw_chatbook.MCP.server_request_handlers import (
    SamplingBudget,
    SamplingPolicy,
    ServerRequestDispatcher,
)

#: action_name for elicitation confirmations in the approval store.
ELICITATION_ACTION_NAME = "mcp.elicitation"

_DEFAULT_RPM = 6
_DEFAULT_TOKEN_CAP = 50_000
_DEFAULT_POLL_SECONDS = 1.0


def sampling_policy_for_server(server_id: str) -> SamplingPolicy:
    """Per-server sampling policy from flat ``[mcp]`` config keys (AC#3).

    Default DENY: a server may sample only if listed in
    ``sampling_allowed_servers``.
    """
    raw_allowed = get_cli_setting("mcp", "sampling_allowed_servers", []) or []
    allowed_ids = {str(item).strip() for item in raw_allowed if str(item).strip()}
    if str(server_id).strip() not in allowed_ids:
        return SamplingPolicy(allowed=False)
    try:
        rpm = int(get_cli_setting("mcp", "sampling_max_requests_per_minute", _DEFAULT_RPM))
    except (TypeError, ValueError):
        rpm = _DEFAULT_RPM
    try:
        cap = int(get_cli_setting("mcp", "sampling_max_total_tokens", _DEFAULT_TOKEN_CAP))
    except (TypeError, ValueError):
        cap = _DEFAULT_TOKEN_CAP
    return SamplingPolicy(
        allowed=True,
        max_requests_per_minute=max(1, rpm),
        max_total_tokens=max(1, cap),
    )


def _plain_chat_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """MCP sampling messages -> plain chat messages (text parts only)."""
    out: List[Dict[str, str]] = []
    for item in messages:
        role = str(item.get("role") or "user")
        content = item.get("content")
        if isinstance(content, dict):
            text = str(content.get("text") or "")
        elif isinstance(content, list):
            text = "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        else:
            text = str(content or "")
        if text:
            out.append({"role": role, "content": text})
    return out


def _extract_text(response: Any) -> str:
    """Provider-shape-tolerant text extraction (mirrors the eval runner)."""
    if isinstance(response, tuple) and response:
        return str(response[0] or "")
    if isinstance(response, str):
        return response
    if isinstance(response, Mapping):
        for key in ("response", "text"):
            if isinstance(response.get(key), str):
                return response[key]
        content = response.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):  # anthropic shape
            return "\n".join(
                str(part.get("text") or "")
                for part in content
                if isinstance(part, dict) and part.get("type") == "text"
            )
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            message = choices[0].get("message") if isinstance(choices[0], dict) else None
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
    return ""


def build_live_complete_fn() -> Callable[..., Awaitable[str]]:
    """The production ``complete_fn``: one bounded, non-streaming provider call."""

    async def complete(
        messages: List[Dict[str, Any]], max_tokens: int, model_hint: Optional[str]
    ) -> str:
        provider = str(
            get_cli_setting("mcp", "sampling_provider", None)
            or get_cli_setting("chat_defaults", "provider", "OpenAI")
        )
        model = str(
            model_hint
            or get_cli_setting("mcp", "sampling_model", None)
            or get_cli_setting("chat_defaults", "model", "")
        )
        payload = _plain_chat_messages(messages)
        if not payload:
            raise ValueError("sampling request contained no text content")
        response = await asyncio.to_thread(
            chat_api_call,
            api_endpoint=provider,
            messages_payload=payload,
            model=model or None,
            streaming=False,
            max_tokens=max(1, int(max_tokens)),
        )
        text = _extract_text(response)
        if not text:
            raise ValueError("provider returned no text for the sampling request")
        return text

    return complete


def _schema_is_confirmation_only(schema: Mapping[str, Any]) -> bool:
    """True when the elicitation needs only a yes/no, not field values."""
    if not schema:
        return True
    properties = schema.get("properties")
    if not properties:
        return True
    if not isinstance(properties, Mapping):
        return False
    # boolean-only properties are representable as approve/deny
    return all(
        isinstance(spec, Mapping) and spec.get("type") == "boolean"
        for spec in properties.values()
    )


def build_live_elicit_fn(
    store: LocalMCPStore,
    *,
    poll_seconds: float = _DEFAULT_POLL_SECONDS,
    timeout_seconds: Optional[float] = None,
) -> Callable[[str, Dict[str, Any]], Awaitable[Optional[Dict[str, Any]]]]:
    """The production ``elicit_fn``: a confirmation through the approval store.

    approve -> ``{"action": "accept", "content": {}}``; deny -> ``None``
    (declined); no answer within the approval timeout -> ``TimeoutError`` and
    the pending request is expired so it cannot be approved later into a
    request nobody is waiting on.
    """

    def _timeout() -> float:
        if timeout_seconds is not None:
            return timeout_seconds
        try:
            return float(get_cli_setting("mcp", "approval_timeout_seconds", 120.0))
        except (TypeError, ValueError):
            return 120.0

    async def elicit(message: str, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not _schema_is_confirmation_only(schema):
            # Honest refusal: we can only represent approve/deny today. A
            # form-filling surface is separate work; never show a prompt whose
            # answer we cannot return.
            raise ValueError(
                "elicitation requests field values; only confirmation-style "
                "elicitations are supported"
            )
        request_id = f"elicit-{uuid.uuid4().hex[:12]}"
        payload = {"message": str(message)[:2000], "schema": dict(schema)}
        fingerprint = hashlib.sha256(
            json.dumps(
                {"resolved_action_id": ELICITATION_ACTION_NAME, "payload": payload},
                sort_keys=True,
                default=str,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        store.save_approval_request(
            LocalApprovalRequest(
                request_id=request_id,
                action_name=ELICITATION_ACTION_NAME,
                resolved_action_id=ELICITATION_ACTION_NAME,
                payload=payload,
                payload_fingerprint=fingerprint,
            )
        )
        deadline = asyncio.get_event_loop().time() + _timeout()
        try:
            while True:
                current = next(
                    (
                        item
                        for item in store.list_approval_requests()
                        if item.request_id == request_id
                    ),
                    None,
                )
                if current is None:
                    return None  # vanished (pruned) -> declined
                if current.status == "approved":
                    return {"action": "accept", "content": {}}
                if current.status not in ("pending",):
                    return None  # denied or any other terminal state
                if asyncio.get_event_loop().time() >= deadline:
                    raise TimeoutError("elicitation approval timed out")
                await asyncio.sleep(poll_seconds)
        finally:
            # never leave an unanswered request pending forever
            current = next(
                (
                    item
                    for item in store.list_approval_requests()
                    if item.request_id == request_id
                ),
                None,
            )
            if current is not None and current.status == "pending":
                try:
                    store.resolve_approval_request(request_id, "expired")
                except Exception:  # noqa: BLE001 - cleanup is best-effort
                    logger.debug("could not expire elicitation request {}", request_id)

    return elicit


def build_server_request_dispatcher_factory(
    store: LocalMCPStore,
) -> Callable[[str], ServerRequestDispatcher]:
    """Factory the MCP client calls per connection (AC#4).

    Policies re-read config on every (re)connect; sampling BUDGETS persist per
    server for this service's lifetime so a reconnect cannot reset spend.
    """
    budgets: Dict[str, SamplingBudget] = {}
    complete = build_live_complete_fn()
    elicit = build_live_elicit_fn(store)

    def factory(server_id: str) -> ServerRequestDispatcher:
        budget = budgets.setdefault(str(server_id), SamplingBudget())
        return ServerRequestDispatcher(
            sampling_policy=sampling_policy_for_server(server_id),
            sampling_budget=budget,
            complete_fn=complete,
            elicit_fn=elicit,
        )

    return factory
