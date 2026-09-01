"""TASK-26031: outbound signed webhooks for run lifecycle events."""

from __future__ import annotations

import asyncio
import hashlib
import hmac

import pytest

from tldw_chatbook.Agents.run_webhooks import (
    WebhookConfig,
    build_webhook_payload,
    sign_payload,
    WEBHOOK_SIGNATURE_HEADER,
    deliver_webhook,
    webhook_config_from_settings,
)


# --- payload carries identifiers + outcome only (AC#3) ---

def test_payload_has_ids_and_outcome_no_content():
    payload = build_webhook_payload(
        event="completed", run_id="run-123", agent_id="agent-9", timestamp="2026-09-01T00:00:00Z"
    )
    assert payload["event"] == "completed"
    assert payload["run_id"] == "run-123"
    assert payload["agent_id"] == "agent-9"
    # never leak content-shaped keys
    blob = str(payload).lower()
    for forbidden in ("message", "content", "tool_arg", "api_key", "secret", "token", "prompt"):
        assert forbidden not in blob, f"payload leaks {forbidden}: {payload}"


# --- HMAC signing is verifiable with a documented scheme (AC#2) ---

def test_signature_is_verifiable_hmac_sha256():
    secret = "s3cr3t"
    body = b'{"event":"completed"}'
    sig = sign_payload(secret, body)
    assert sig.startswith("sha256=")
    expected = "sha256=" + hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert sig == expected


# --- config gating: default off (AC#7) ---

def test_config_default_disabled():
    cfg = webhook_config_from_settings({})
    assert cfg.enabled is False
    assert cfg.url == ""

def test_config_enabled_when_configured():
    cfg = webhook_config_from_settings(
        {"webhooks": {"enabled": True, "url": "https://hook.example/x", "secret": "s", "events": ["completed"]}}
    )
    assert cfg.enabled is True
    assert cfg.url == "https://hook.example/x"
    assert "completed" in cfg.events


def _run(coro):
    return asyncio.run(coro)


# --- delivery gating + egress + fire-and-forget (AC#1/#4/#5/#6/#7) ---

def test_disabled_config_makes_no_request(monkeypatch):
    """AC#7: no endpoint configured => no request ever."""
    posted = {"n": 0}
    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1
    result = _run(deliver_webhook(
        WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
        "completed", "run-1", post_fn=fake_post,
    ))
    assert posted["n"] == 0
    assert result is False


def test_event_not_subscribed_makes_no_request():
    posted = {"n": 0}
    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1
    result = _run(deliver_webhook(
        WebhookConfig(enabled=True, url="https://h/x", secret="s", events=("failed",)),
        "completed", "run-1", post_fn=fake_post,
    ))
    assert posted["n"] == 0  # "completed" not in subscribed events
    assert result is False


def test_egress_blocked_url_is_not_posted(monkeypatch):
    """AC#6: destination subject to the SSRF egress policy."""
    from tldw_chatbook.Agents import run_webhooks
    async def blocked(url, **k):
        from tldw_chatbook.Utils.egress import EgressBlockedError
        raise EgressBlockedError(url, "private ip")
    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", blocked)
    posted = {"n": 0}
    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1
    result = _run(deliver_webhook(
        WebhookConfig(enabled=True, url="http://169.254.169.254/x", secret="s", events=("completed",)),
        "completed", "run-1", post_fn=fake_post,
    ))
    assert posted["n"] == 0, "an egress-blocked URL must not be POSTed"
    assert result is False


def test_successful_delivery_signs_and_posts(monkeypatch):
    """AC#1/#2: a configured, allowed endpoint gets a signed POST."""
    from tldw_chatbook.Agents import run_webhooks
    async def allowed(url, **k):
        return None
    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    captured = {}
    async def fake_post(url, body, headers, timeout):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout"] = timeout
    result = _run(deliver_webhook(
        WebhookConfig(enabled=True, url="https://h/x", secret="sec", events=("completed",)),
        "completed", "run-1", post_fn=fake_post,
    ))
    assert result is True
    assert captured["url"] == "https://h/x"
    assert WEBHOOK_SIGNATURE_HEADER in captured["headers"]
    expected = sign_payload("sec", captured["body"])
    assert captured["headers"][WEBHOOK_SIGNATURE_HEADER] == expected
    assert captured["timeout"] > 0  # AC#4 bounded


def test_delivery_failure_is_visible_not_raised(monkeypatch):
    """AC#4/#5: a dead endpoint never raises into the run; failure is logged."""
    from tldw_chatbook.Agents import run_webhooks
    async def allowed(url, **k):
        return None
    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    async def boom(url, body, headers, timeout):
        raise TimeoutError("dead endpoint")
    result = _run(deliver_webhook(
        WebhookConfig(enabled=True, url="https://h/x", secret="s", events=("completed",)),
        "completed", "run-1", post_fn=boom,
    ))
    assert result is False  # swallowed, run unaffected


def test_scheduler_gates_before_spawning_a_thread():
    """AC#7/#4: disabled or unsubscribed => no thread, no delivery."""
    from tldw_chatbook.Agents.run_webhooks import schedule_run_webhook
    assert schedule_run_webhook(
        WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
        "completed", "run-1",
    ) is False
    assert schedule_run_webhook(
        WebhookConfig(enabled=True, url="https://h/x", secret="s", events=("failed",)),
        "completed", "run-1",
    ) is False


def test_scheduler_delivers_when_enabled():
    """The scheduler starts a delivery that reaches the endpoint (AC#1)."""
    import time
    from tldw_chatbook.Agents import run_webhooks
    delivered = {"ok": False}

    async def allowed(url, **k):
        return None

    real_check = run_webhooks.check_url_or_raise_async
    run_webhooks.check_url_or_raise_async = allowed
    orig_post = run_webhooks._default_post

    async def fake_post(url, body, headers, timeout):
        delivered["ok"] = True

    run_webhooks._default_post = fake_post
    try:
        started = run_webhooks.schedule_run_webhook(
            WebhookConfig(enabled=True, url="https://h/x", secret="s", events=("completed",)),
            "completed", "run-1",
        )
        assert started is True
        for _ in range(50):
            if delivered["ok"]:
                break
            time.sleep(0.02)
    finally:
        run_webhooks.check_url_or_raise_async = real_check
        run_webhooks._default_post = orig_post
    assert delivered["ok"] is True


# --- terminal-seam wiring in AgentService (AC#1 completed/failed) ---

def test_agent_service_terminal_seam_fires_completed_and_failed(monkeypatch):
    """AC#1: a fresh terminal transition maps to a lifecycle webhook event;
    non-notify terminal states (cancelled/superseded) do not fire."""
    from tldw_chatbook.Agents import agent_service as svc_mod
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.agent_models import (
        RUN_DONE, RUN_ERROR, RUN_CANCELLED,
    )

    captured = []
    monkeypatch.setattr(svc_mod, "safe_utc_timestamp", lambda *_a, **_k: "2026-09-01T00:00:00Z")

    def fake_schedule(config, event, run_id, **kwargs):
        captured.append((event, run_id))
        return True

    import tldw_chatbook.Agents.run_webhooks as rw
    monkeypatch.setattr(rw, "schedule_run_webhook", fake_schedule)
    monkeypatch.setattr(
        rw, "webhook_config_from_settings",
        lambda *_a, **_k: rw.WebhookConfig(
            enabled=True, url="https://h/x", secret="s", events=rw.WEBHOOK_EVENTS
        ),
    )

    svc = AgentService.__new__(AgentService)
    svc.wall_clock = None

    svc._maybe_emit_run_webhook("run-done", RUN_DONE)
    svc._maybe_emit_run_webhook("run-err", RUN_ERROR)
    svc._maybe_emit_run_webhook("run-cancel", RUN_CANCELLED)

    assert ("completed", "run-done") in captured
    assert ("failed", "run-err") in captured
    assert not any(rid == "run-cancel" for _, rid in captured), "cancelled must not notify"
