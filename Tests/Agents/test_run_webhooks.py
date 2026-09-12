"""TASK-26031: outbound signed webhooks for run lifecycle events."""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import threading
import time

from tldw_chatbook.Agents.run_webhooks import (
    WEBHOOK_SIGNATURE_HEADER,
    WebhookConfig,
    _WebhookDelivery,
    _WebhookDeliveryWorker,
    build_webhook_payload,
    deliver_webhook,
    sign_payload,
    webhook_config_from_settings,
)


def _delivery(run_id: str, *, extra_ids=None) -> _WebhookDelivery:
    return _WebhookDelivery(
        config=WebhookConfig(
            enabled=True,
            url="https://hook.example/x",
            secret="secret",
            events=("completed",),
        ),
        event="completed",
        run_id=run_id,
        agent_id=None,
        timestamp=None,
        extra_ids=extra_ids,
    )


def _join_worker(worker: _WebhookDeliveryWorker, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with worker._state_lock:
            thread = worker._thread
        if thread is None:
            return
        thread.join(min(0.05, max(0.0, deadline - time.monotonic())))
    raise AssertionError("webhook worker did not retire")


# --- bounded reusable delivery worker (TASK-31511) ---


def test_worker_reuses_one_thread_and_event_loop_without_blocking_submit(monkeypatch):
    """Per-event thread/loop scheduling or waiting in submit breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def held_delivery(config, event, run_id, **kwargs):
        observed.append((threading.get_ident(), id(asyncio.get_running_loop()), run_id))
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=2, idle_seconds=0.02)
    try:
        assert worker.submit(_delivery("one")) is True
        assert entered.wait(1.0)
        started = time.monotonic()
        assert worker.submit(_delivery("two")) is True
        assert time.monotonic() - started < 0.1
    finally:
        release.set()
        _join_worker(worker)

    assert [item[2] for item in observed] == ["one", "two"]
    assert len({item[0] for item in observed}) == 1
    assert len({item[1] for item in observed}) == 1


def test_capacity_one_refuses_third_delivery_without_exposing_canaries(monkeypatch):
    """An unbounded queue or sensitive saturation diagnostic breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    warnings = []
    metrics = []

    async def held_delivery(config, event, run_id, **kwargs):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(
        run_webhooks,
        "log_counter",
        lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    canaries = ("https://secret.example/token", "signing-secret", "run-sensitive")
    try:
        assert worker.submit(_delivery(canaries[2])) is True
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        assert (
            worker.submit(
                _WebhookDelivery(
                    config=WebhookConfig(
                        enabled=True,
                        url=canaries[0],
                        secret=canaries[1],
                        events=("completed",),
                    ),
                    event="completed",
                    run_id="overflow",
                    agent_id=None,
                    timestamp=None,
                    extra_ids=None,
                )
            )
            is False
        )
    finally:
        release.set()
        _join_worker(worker)

    diagnostic_blob = repr((warnings, metrics))
    assert all(canary not in diagnostic_blob for canary in canaries)
    assert metrics == [
        (
            "run_webhook_dropped",
            {"labels": {"reason": "queue_full", "event": "completed"}},
        )
    ]


def test_worker_retires_then_restarts_with_a_new_generation(monkeypatch):
    """Keeping a retired generation or failing to restart breaks this test."""
    from tldw_chatbook.Agents import run_webhooks

    observed_threads = []

    async def record_delivery(config, event, run_id, **kwargs):
        observed_threads.append(threading.current_thread())
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    assert worker.submit(_delivery("one")) is True
    _join_worker(worker)
    assert worker.submit(_delivery("two")) is True
    _join_worker(worker)
    assert len(observed_threads) == 2
    assert observed_threads[0] is not observed_threads[1]


def test_submission_during_runner_close_is_refused_then_restart_succeeds(monkeypatch):
    """Publishing retirement before Runner close can strand an admitted event."""
    from tldw_chatbook.Agents import run_webhooks

    real_runner = asyncio.Runner
    closing = threading.Event()
    release_close = threading.Event()
    delivered = []

    class GatedRunner:
        def __init__(self):
            self._runner = real_runner()

        def __enter__(self):
            self._runner.__enter__()
            return self

        def run(self, coro):
            return self._runner.run(coro)

        def __exit__(self, *args):
            closing.set()
            assert release_close.wait(1.0)
            return self._runner.__exit__(*args)

    async def record_delivery(config, event, run_id, **kwargs):
        delivered.append(run_id)
        return True

    monkeypatch.setattr(run_webhooks.asyncio, "Runner", GatedRunner)
    monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.01)
    try:
        assert worker.submit(_delivery("first")) is True
        assert closing.wait(1.0)
        assert worker.submit(_delivery("during-close")) is False
        release_close.set()
        _join_worker(worker)
        assert worker.submit(_delivery("after-close")) is True
    finally:
        release_close.set()
        _join_worker(worker)
    assert delivered == ["first", "after-close"]


def test_failed_thread_start_is_sanitized_and_later_submit_recovers(monkeypatch):
    """A failed start must not strand work or poison later generations."""
    from tldw_chatbook.Agents import run_webhooks

    real_thread = threading.Thread
    warnings = []
    delivered = []

    class FailedThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("run-sensitive secret-url signing-secret")

    async def record_delivery(config, event, run_id, **kwargs):
        delivered.append(run_id)
        return True

    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda message, *args: warnings.append(message.format(*args)),
    )
    monkeypatch.setattr(run_webhooks.threading, "Thread", FailedThread)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.01)
    assert worker.submit(_delivery("not-admitted")) is False
    monkeypatch.setattr(run_webhooks.threading, "Thread", real_thread)
    monkeypatch.setattr(run_webhooks, "deliver_webhook", record_delivery)
    assert worker.submit(_delivery("recovered")) is True
    _join_worker(worker)
    assert delivered == ["recovered"]
    assert "run-sensitive" not in repr(warnings)
    assert "secret-url" not in repr(warnings)
    assert "signing-secret" not in repr(warnings)


def test_callback_failure_does_not_stop_fifo_and_admission_copies_inputs(monkeypatch):
    """A callback escape or caller mutation must not alter later admitted work."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    observed = []

    async def delivery(config, event, run_id, **kwargs):
        if run_id == "held":
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            raise KeyboardInterrupt("identifier-canary")
        observed.append((run_id, config.events, dict(kwargs["extra_ids"])))
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", delivery)
    mutable_events = ["completed"]
    mutable_ids = {"workspace_id": "original"}
    config = WebhookConfig(
        enabled=True,
        url="https://hook.example/x",
        secret="secret",
        events=mutable_events,  # type: ignore[arg-type]
    )
    worker = _WebhookDeliveryWorker(queue_capacity=2, idle_seconds=0.02)
    try:
        assert worker.submit(_delivery("held")) is True
        assert entered.wait(1.0)
        assert (
            worker.submit(
                _WebhookDelivery(
                    config=config,
                    event="completed",
                    run_id="copied",
                    agent_id=None,
                    timestamp=None,
                    extra_ids=mutable_ids,
                )
            )
            is True
        )
        mutable_events.append("failed")
        mutable_ids["workspace_id"] = "mutated"
    finally:
        release.set()
        _join_worker(worker)
    assert observed == [("copied", ("completed",), {"workspace_id": "original"})]


def test_diagnostic_failures_do_not_escape_admission_or_stop_fifo(monkeypatch):
    """A broken diagnostic sink must not strand admitted webhook work."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    delivered = []

    async def delivery(config, event, run_id, **kwargs):
        if run_id == "held":
            entered.set()
            while not release.is_set():
                await asyncio.sleep(0.005)
            raise RuntimeError("callback")
        delivered.append(run_id)
        return True

    def diagnostic_failure(*args, **kwargs):
        raise RuntimeError("diagnostic")

    monkeypatch.setattr(run_webhooks, "deliver_webhook", delivery)
    monkeypatch.setattr(run_webhooks.logger, "warning", diagnostic_failure)
    monkeypatch.setattr(run_webhooks, "log_counter", diagnostic_failure)
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    try:
        assert worker.submit(_delivery("held")) is True
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        assert worker.submit(_delivery("overflow")) is False
    finally:
        release.set()
        _join_worker(worker)
    assert delivered == ["waiting"]


def test_worker_metrics_bound_unknown_event_labels(monkeypatch):
    """Caller-supplied event names must not create unbounded metric labels."""
    from tldw_chatbook.Agents import run_webhooks

    entered = threading.Event()
    release = threading.Event()
    metrics = []

    async def held_delivery(config, event, run_id, **kwargs):
        entered.set()
        while not release.is_set():
            await asyncio.sleep(0.005)
        return True

    monkeypatch.setattr(run_webhooks, "deliver_webhook", held_delivery)
    monkeypatch.setattr(
        run_webhooks,
        "log_counter",
        lambda name, **kwargs: metrics.append((name, kwargs)),
    )
    worker = _WebhookDeliveryWorker(queue_capacity=1, idle_seconds=0.02)
    try:
        assert worker.submit(_delivery("held")) is True
        assert entered.wait(1.0)
        assert worker.submit(_delivery("waiting")) is True
        unknown = _WebhookDelivery(
            config=_delivery("unused").config,
            event="user-supplied-sensitive-event",
            run_id="overflow",
            agent_id=None,
            timestamp=None,
            extra_ids=None,
        )
        assert worker.submit(unknown) is False
    finally:
        release.set()
        _join_worker(worker)
    assert metrics == [
        (
            "run_webhook_dropped",
            {"labels": {"reason": "queue_full", "event": "other"}},
        )
    ]


# --- payload carries identifiers + outcome only (AC#3) ---


def test_payload_has_ids_and_outcome_no_content():
    payload = build_webhook_payload(
        event="completed",
        run_id="run-123",
        agent_id="agent-9",
        timestamp="2026-09-01T00:00:00Z",
    )
    assert payload["event"] == "completed"
    assert payload["run_id"] == "run-123"
    assert payload["agent_id"] == "agent-9"
    # never leak content-shaped keys
    blob = str(payload).lower()
    for forbidden in (
        "message",
        "content",
        "tool_arg",
        "api_key",
        "secret",
        "token",
        "prompt",
    ):
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
        {
            "webhooks": {
                "enabled": True,
                "url": "https://hook.example/x",
                "secret": "s",
                "events": ["completed"],
            }
        }
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

    result = _run(
        deliver_webhook(
            WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert posted["n"] == 0
    assert result is False


def test_event_not_subscribed_makes_no_request():
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("failed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
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

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True,
                url="http://169.254.169.254/x",
                secret="s",
                events=("completed",),
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
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

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="sec", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
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

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=boom,
        )
    )
    assert result is False  # swallowed, run unaffected


def test_scheduler_gates_before_spawning_a_thread():
    """AC#7/#4: disabled or unsubscribed => no thread, no delivery."""
    from tldw_chatbook.Agents.run_webhooks import schedule_run_webhook

    assert (
        schedule_run_webhook(
            WebhookConfig(enabled=False, url="", secret="", events=("completed",)),
            "completed",
            "run-1",
        )
        is False
    )
    assert (
        schedule_run_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("failed",)
            ),
            "completed",
            "run-1",
        )
        is False
    )


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
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",
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
    from tldw_chatbook.Agents.agent_models import (
        RUN_CANCELLED,
        RUN_DONE,
        RUN_ERROR,
    )
    from tldw_chatbook.Agents.agent_service import AgentService

    captured = []
    monkeypatch.setattr(
        svc_mod, "safe_utc_timestamp", lambda *_a, **_k: "2026-09-01T00:00:00Z"
    )

    def fake_schedule(config, event, run_id, **kwargs):
        captured.append((event, run_id))
        return True

    import tldw_chatbook.Agents.run_webhooks as rw

    monkeypatch.setattr(rw, "schedule_run_webhook", fake_schedule)
    monkeypatch.setattr(
        rw,
        "webhook_config_from_settings",
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
    assert not any(rid == "run-cancel" for _, rid in captured), (
        "cancelled must not notify"
    )


# --- Qodo review round (PR #2301) ---


def test_qodo9_string_false_does_not_enable():
    """Qodo #9: 'false'/'0' strings must not enable the outbound gate."""
    for junk in ("false", "0", "no", "off", "FALSE"):
        cfg = webhook_config_from_settings(
            {"webhooks": {"enabled": junk, "url": "https://h/x", "secret": "s"}}
        )
        assert cfg.enabled is False, junk
    cfg = webhook_config_from_settings(
        {"webhooks": {"enabled": "true", "url": "https://h/x", "secret": "s"}}
    )
    assert cfg.enabled is True


def test_qodo10_enabled_without_secret_never_delivers(monkeypatch):
    """Qodo #10: no signing secret => fail closed, no POST."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)
    posted = {"n": 0}

    async def fake_post(url, body, headers, timeout):
        posted["n"] += 1

    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="", events=("completed",)
            ),
            "completed",
            "run-1",
            post_fn=fake_post,
        )
    )
    assert result is False and posted["n"] == 0


def test_qodo11_timeout_rejects_nan_and_infinity():
    """Qodo #11: NaN/inf timeouts fall back to a finite bound."""
    for junk in ("inf", "nan", float("inf"), float("nan"), -5, 1e12):
        cfg = webhook_config_from_settings(
            {
                "webhooks": {
                    "enabled": True,
                    "url": "https://h/x",
                    "secret": "s",
                    "timeout_seconds": junk,
                }
            }
        )
        import math

        assert math.isfinite(cfg.timeout_seconds), junk
        assert 0.1 <= cfg.timeout_seconds <= 120.0, (junk, cfg.timeout_seconds)


def test_qodo8_http_error_status_is_a_failed_delivery(monkeypatch):
    """Qodo #8: a 4xx/5xx response is a FAILED delivery, not a success."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)

    class _Resp:
        status_code = 500

        def raise_for_status(self):
            raise RuntimeError("500 Server Error")

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, content=None, headers=None):
            return _Resp()

    import httpx

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    result = _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True, url="https://h/x", secret="s", events=("completed",)
            ),
            "completed",
            "run-1",  # no post_fn: exercise the REAL default transport
        )
    )
    assert result is False


def test_qodo12_failure_log_never_contains_full_url(monkeypatch, caplog):
    """Qodo #12: a failing delivery logs a sanitized origin, never credentials
    embedded in the URL path/query/userinfo."""
    from tldw_chatbook.Agents import run_webhooks

    async def allowed(url, **k):
        return None

    monkeypatch.setattr(run_webhooks, "check_url_or_raise_async", allowed)

    async def boom(url, body, headers, timeout):
        raise TimeoutError("dead")

    captured = []
    monkeypatch.setattr(
        run_webhooks.logger,
        "warning",
        lambda msg, *a, **k: captured.append(msg.format(*a) if a else str(msg)),
    )
    _run(
        deliver_webhook(
            WebhookConfig(
                enabled=True,
                url="https://user:tok3n@h.example/hook?apikey=sekret",
                secret="s",
                events=("completed",),
            ),
            "completed",
            "run-1",
            post_fn=boom,
        )
    )
    joined = " ".join(captured)
    assert "tok3n" not in joined and "sekret" not in joined, joined
    assert "h.example" in joined  # origin still identifiable
