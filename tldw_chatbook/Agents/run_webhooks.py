"""TASK-26031: outbound signed webhooks for run lifecycle events.

Nothing outside the TUI could learn that an agent run finished. This module
emits HMAC-signed, fire-and-forget notifications for run lifecycle events
(completed / failed / needs-approval) to a user-configured endpoint, so a
dashboard, phone, or script can be notified.

Fail-open for the run, fail-closed for data: delivery is disabled unless the
user configures an endpoint (AC#7), payloads carry identifiers and an outcome
category only -- never message content, tool arguments, or credentials (AC#3),
the destination is subject to the existing SSRF egress policy (AC#6), and a
slow or dead endpoint can never delay or fail the run -- delivery is bounded
and its errors are logged, never raised into the run path (AC#4/#5).

Signature scheme (AC#2): ``X-Tldw-Signature: sha256=<hex>`` where ``<hex>`` is
``HMAC-SHA256(secret, raw_request_body)``. The receiver recomputes the HMAC
over the exact bytes received and compares.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import math
import threading
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional, Sequence, Tuple

from loguru import logger

from ..Utils.egress import EgressBlockedError, check_url_or_raise_async

try:  # metrics are best-effort; never let their absence break delivery
    from ..Metrics.metrics_logger import log_counter
except Exception:  # noqa: BLE001
    def log_counter(*_args: Any, **_kwargs: Any) -> None:  # type: ignore
        return None


WEBHOOK_SIGNATURE_HEADER = "X-Tldw-Signature"
#: Lifecycle events a webhook may subscribe to.
WEBHOOK_EVENTS = ("completed", "failed", "needs-approval")
_DEFAULT_TIMEOUT_SECONDS = 5.0
_MAX_TIMEOUT_SECONDS = 120.0

PostFn = Callable[[str, bytes, Mapping[str, str], float], Awaitable[None]]


@dataclass(frozen=True)
class WebhookConfig:
    enabled: bool = False
    url: str = ""
    secret: str = ""
    events: Tuple[str, ...] = ()
    timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS


def webhook_config_from_settings(settings: Mapping[str, Any]) -> WebhookConfig:
    """Read the ``[webhooks]`` config. Disabled by default (AC#7)."""
    section = settings.get("webhooks") if isinstance(settings, Mapping) else None
    if not isinstance(section, Mapping):
        return WebhookConfig()
    raw_events = section.get("events")
    if isinstance(raw_events, (list, tuple)):
        events = tuple(str(e).strip() for e in raw_events if str(e).strip())
    else:
        events = WEBHOOK_EVENTS  # subscribe to all when unspecified
    # Qodo #11 (PR #2301): reject NaN/inf and clamp to a finite band so a
    # junk value can never leave delivery effectively unbounded.
    try:
        timeout = float(section.get("timeout_seconds", _DEFAULT_TIMEOUT_SECONDS))
    except (TypeError, ValueError):
        timeout = _DEFAULT_TIMEOUT_SECONDS
    if not math.isfinite(timeout):
        timeout = _DEFAULT_TIMEOUT_SECONDS
    timeout = min(max(0.1, timeout), _MAX_TIMEOUT_SECONDS)
    # Qodo #9 (PR #2301): strict boolean coercion -- "false"/"0"/junk must
    # not enable an outbound network feature (shared helper, default False).
    from ..config import coerce_bool_setting

    enabled = bool(coerce_bool_setting(section.get("enabled", False), False))
    return WebhookConfig(
        enabled=enabled,
        url=str(section.get("url") or "").strip(),
        secret=str(section.get("secret") or ""),
        events=events,
        timeout_seconds=timeout,
    )


def build_webhook_payload(
    event: str,
    run_id: str,
    *,
    agent_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    extra_ids: Optional[Mapping[str, str]] = None,
) -> dict:
    """Build a content-free lifecycle payload (AC#3).

    Only identifiers and the outcome category are included -- deliberately no
    message content, tool arguments, or credentials.

    CONTRACT (lane-6 review M3): ``extra_ids`` is for additional IDENTIFIERS
    only (e.g. a parent run id, a workspace id) -- never message content, tool
    arguments, or secrets. Values are stringified verbatim into the payload
    that leaves the machine.
    """
    payload: dict[str, Any] = {"event": str(event), "run_id": str(run_id)}
    if agent_id:
        payload["agent_id"] = str(agent_id)
    if timestamp:
        payload["timestamp"] = str(timestamp)
    if extra_ids:
        # only stringy identifier values; never nested content
        for key, value in extra_ids.items():
            payload[str(key)] = str(value)
    return payload


def sign_payload(secret: str, body: bytes) -> str:
    """HMAC-SHA256 signature of the raw body, as ``sha256=<hex>`` (AC#2)."""
    digest = hmac.new((secret or "").encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def _log_safe_origin(url: str) -> str:
    """Scheme+host(+port) only -- webhook URLs may embed tokens in userinfo,
    path, or query (Qodo #12, PR #2301)."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or "?"
        port = f":{parts.port}" if parts.port else ""
        return f"{parts.scheme}://{host}{port}"
    except Exception:  # noqa: BLE001 - sanitizer must never raise
        return "<unparseable-url>"


async def _default_post(url: str, body: bytes, headers: Mapping[str, str], timeout: float) -> None:
    import httpx

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, content=body, headers=dict(headers))
        # Qodo #8 (PR #2301): a 4xx/5xx receiver response is a FAILED
        # delivery, not a success -- raise so the caller records it as one.
        response.raise_for_status()


async def deliver_webhook(
    config: WebhookConfig,
    event: str,
    run_id: str,
    *,
    agent_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    extra_ids: Optional[Mapping[str, str]] = None,
    post_fn: Optional[PostFn] = None,
) -> bool:
    """Deliver one lifecycle webhook. Returns True iff a POST was made and
    succeeded. Never raises into the caller (AC#4/#5).

    Gated on config (AC#7) and event subscription; the destination is checked
    against the SSRF egress policy before any request (AC#6); delivery uses a
    bounded timeout; any failure is logged + counted, not raised.
    """
    if not config.enabled or not config.url:
        return False
    # Qodo #10 (PR #2301): a signature over an empty key is forgeable by
    # anyone; enabled-without-secret fails closed rather than emitting
    # unauthenticatable events.
    if not config.secret:
        logger.warning(
            "Run webhook is enabled but [webhooks] secret is empty; "
            "refusing to deliver unauthenticated events."
        )
        log_counter("run_webhook_blocked", labels={"reason": "no_secret"})
        return False
    if event not in config.events:
        return False

    try:
        await check_url_or_raise_async(config.url)
    except EgressBlockedError as exc:
        logger.warning("Run webhook blocked by egress policy: {}", exc)
        log_counter("run_webhook_blocked", labels={"reason": "egress"})
        return False
    except Exception as exc:  # noqa: BLE001 - policy check must not raise into run
        logger.warning("Run webhook egress check failed: {!r}", exc)
        log_counter("run_webhook_blocked", labels={"reason": "egress_error"})
        return False

    payload = build_webhook_payload(
        event, run_id, agent_id=agent_id, timestamp=timestamp, extra_ids=extra_ids
    )
    body = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        WEBHOOK_SIGNATURE_HEADER: sign_payload(config.secret, body),
    }
    post = post_fn or _default_post
    try:
        await post(config.url, body, headers, config.timeout_seconds)
        log_counter("run_webhook_delivered", labels={"event": event})
        return True
    except Exception as exc:  # noqa: BLE001 - a dead endpoint never fails the run
        logger.warning(
            "Run webhook delivery to {} failed ({}): {!r}",
            _log_safe_origin(config.url),
            event,
            exc,
        )
        log_counter("run_webhook_failed", labels={"event": event})
        return False


def schedule_run_webhook(
    config: WebhookConfig,
    event: str,
    run_id: str,
    *,
    agent_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    extra_ids: Optional[Mapping[str, str]] = None,
) -> bool:
    """Fire-and-forget a lifecycle webhook from ANY context, incl. the sync
    worker thread the agent runtime runs on (AC#4).

    Returns True iff a delivery thread was started (config enabled + endpoint
    set + event subscribed). Delivery runs on a daemon thread with its own
    event loop so it can never delay or fail the run; all errors are handled
    inside ``deliver_webhook``.
    """
    if not config.enabled or not config.url or event not in config.events:
        return False

    def _runner() -> None:
        try:
            asyncio.run(
                deliver_webhook(
                    config,
                    event,
                    run_id,
                    agent_id=agent_id,
                    timestamp=timestamp,
                    extra_ids=extra_ids,
                )
            )
        except Exception as exc:  # noqa: BLE001 - fire-and-forget, never propagate
            logger.warning("Run webhook delivery thread failed: {!r}", exc)

    threading.Thread(
        target=_runner, name=f"run-webhook-{event}", daemon=True
    ).start()
    return True
