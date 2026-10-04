"""TASK-34100.5 AC#5 (gap-07): a cold local model gets a first-token window.

A large local model can spend minutes loading and processing the prompt
before its first token. The fixed 90 s content-stall window ended those runs
with no allowance. The first token now gets its own longer, configurable
window (300 s for self-hosted providers); gaps between tokens keep 90 s. The
wait shows elapsed time and a cold-load hint, and a first-token timeout names
the setting that waits longer and suggests a smaller model.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Chat.provider_failures import describe_stream_failure

#: The first-token wait line's opening words.
_WAITING = "Waiting for a reply"
from tldw_chatbook.Chat.stream_stall_watchdog import (
    DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT_SECONDS,
    StreamStallError,
    first_token_timeout_seconds,
    watch_content_stalls,
)


async def _collect(source, timeout, **kwargs):
    return [item async for item in watch_content_stalls(source, timeout, **kwargs)]


def test_a_slow_first_token_inside_its_own_window_does_not_trip() -> None:
    async def source():
        await asyncio.sleep(0.25)  # longer than the gap window
        yield "first"
        await asyncio.sleep(0.02)
        yield "second"

    assert asyncio.run(
        _collect(source(), 0.1, provider="llama_cpp", first_item_timeout_seconds=1.0)
    ) == ["first", "second"]


def test_gaps_between_tokens_keep_the_shorter_window() -> None:
    async def source():
        yield "first"
        await asyncio.sleep(0.5)
        yield "late"

    with pytest.raises(StreamStallError) as caught:
        asyncio.run(_collect(source(), 0.1, first_item_timeout_seconds=1.0))
    assert caught.value.first_token is False
    assert caught.value.timeout_seconds == 0.1


def test_no_first_token_inside_the_window_is_a_first_token_stall() -> None:
    async def source():
        await asyncio.sleep(5)
        yield "never"

    with pytest.raises(StreamStallError) as caught:
        asyncio.run(_collect(source(), 0.05, first_item_timeout_seconds=0.15))
    assert caught.value.first_token is True
    assert caught.value.timeout_seconds == 0.15


@pytest.mark.parametrize("provider", ["llama_cpp", "custom", "ollama", "vllm"])
def test_self_hosted_providers_get_about_five_minutes_by_default(
    monkeypatch, provider
) -> None:
    monkeypatch.delenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda _section, _key, default=None: default,
    )
    assert DEFAULT_LOCAL_FIRST_TOKEN_TIMEOUT_SECONDS == 300.0
    assert first_token_timeout_seconds(provider, stall_timeout=90.0) == 300.0


def test_cloud_providers_keep_the_stall_window_for_the_first_token(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda _section, _key, default=None: default,
    )
    assert first_token_timeout_seconds("openai", stall_timeout=90.0) == 90.0


def test_the_first_token_window_is_configurable(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)
    seen: list[tuple[str, str]] = []

    def setting(section, key, default=None):
        seen.append((section, key))
        return 600 if key == "first_token_timeout_seconds" else default

    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", setting)
    assert first_token_timeout_seconds("llama_cpp", stall_timeout=90.0) == 600.0
    assert ("chat_defaults", "first_token_timeout_seconds") in seen
    monkeypatch.setenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", "450")
    assert first_token_timeout_seconds("llama_cpp", stall_timeout=90.0) == 450.0
    # Never shorter than the gap window; a disabled watchdog stays disabled.
    monkeypatch.setenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", "60")
    assert first_token_timeout_seconds("llama_cpp", stall_timeout=120.0) == 120.0
    assert first_token_timeout_seconds("llama_cpp", stall_timeout=0) is None


def test_a_first_token_timeout_names_the_setting_and_a_smaller_model() -> None:
    copy = describe_stream_failure(
        StreamStallError(300, provider="llama_cpp", first_token=True)
    )

    assert "unexpected provider error" not in copy
    assert "300 s" in copy
    assert "chat_defaults.first_token_timeout_seconds" in copy
    assert "smaller model" in copy
    assert "Wait longer" in copy


def test_a_long_wait_for_the_first_token_shows_elapsed_and_a_cold_load_hint() -> None:
    from tldw_chatbook.UI.Console_Modules.agent import console_turn_activity_text

    usage = SimpleNamespace(started_at=0.0, output_tokens=0, source="local")
    snapshot = SimpleNamespace(status="running", steps=(), turn_usage=usage)

    early = console_turn_activity_text(snapshot, now=5.0, self_hosted=True)
    late = console_turn_activity_text(snapshot, now=42.0, self_hosted=True)
    # Live: a cold llama.cpp streams a stray delta or two while it is still
    # processing the prompt -- that is not the answer starting.
    usage.output_tokens = 2
    stray = console_turn_activity_text(snapshot, now=91.0, self_hosted=True)
    assert stray.startswith(_WAITING)
    assert "1m 31s" in stray
    usage.output_tokens = 40
    answering = console_turn_activity_text(snapshot, now=91.0, self_hosted=True)
    assert answering.startswith("Generating…")
    usage.output_tokens = 0

    assert early.startswith("Generating…") and "5s" in early
    assert late.startswith(_WAITING)
    assert "42s" in late
    assert "may still be loading" in late


def test_a_first_send_with_no_published_usage_still_times_the_wait() -> None:
    """Live (2026-10-03): the first send of a new conversation showed a bare
    'Generating…' for 230 s -- the bridge had published no usage for it, so
    nothing could time the wait. The view's own run-start time is the base."""
    from tldw_chatbook.UI.Console_Modules.agent import console_turn_activity_text

    unpublished = SimpleNamespace(status="idle", steps=(), turn_usage=None)
    running = SimpleNamespace(status="running", steps=(), turn_usage=None)

    assert console_turn_activity_text(unpublished, now=50.0) == ""
    for snapshot in (unpublished, running):
        line = console_turn_activity_text(snapshot, now=50.0, turn_started_at=10.0)
        assert line.startswith(_WAITING), line
        assert "40s" in line
    finished = SimpleNamespace(status="done", steps=(), turn_usage=None)
    assert console_turn_activity_text(finished, now=50.0, turn_started_at=10.0) == ""


# --- Review round 1 -----------------------------------------------------------


def test_a_whitespace_delta_does_not_end_the_first_token_window() -> None:
    """A-F5: a cold server that sends a blank delta while it is still reading
    the prompt has not started answering; the long first-token window holds
    until real content arrives."""

    async def source():
        yield "\n"
        await asyncio.sleep(0.3)  # longer than the gap window, inside the first
        yield "answer"

    assert asyncio.run(
        _collect(source(), 0.1, first_item_timeout_seconds=1.0)
    ) == ["\n", "answer"]


def test_real_content_still_hands_over_to_the_gap_window() -> None:
    """The control: once the answer has begun, gaps keep the short window."""

    async def source():
        yield " Hi"
        await asyncio.sleep(0.3)
        yield "late"

    with pytest.raises(StreamStallError) as caught:
        asyncio.run(_collect(source(), 0.1, first_item_timeout_seconds=1.0))
    assert caught.value.first_token is False


def _no_first_token_config(monkeypatch) -> None:
    monkeypatch.delenv("TLDW_FIRST_TOKEN_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr(
        "tldw_chatbook.config.get_cli_setting",
        lambda _section, _key, default=None: default,
    )


class _RecordingSession:
    def __init__(self) -> None:
        self.timeouts: list[object] = []
        self.adapters: list[object] = []

    def mount(self, _prefix, adapter) -> None:
        self.adapters.append(adapter)

    def post(self, _url, json=None, headers=None, timeout=None, stream=False):
        self.timeouts.append(timeout)

        class _Response:
            status_code = 200
            text = '{"choices":[{"message":{"content":"ok"}}]}'

            @staticmethod
            def json():
                return {"choices": [{"message": {"content": "ok"}}]}

            @staticmethod
            def raise_for_status():
                return None

        return _Response()

    def close(self) -> None:
        return None


def test_a_self_hosted_read_timeout_never_undercuts_the_first_token_window(
    monkeypatch,
) -> None:
    """B-F1 (live, 2026-10-03): a Custom OpenAI-compatible endpoint's 120 s
    HTTP read timeout fired before the 300 s first-token window, urllib3
    re-sent the prompt (llama-server processed it twice), and the 503 that
    followed was retried as transient into a refused trace reservation. The
    read timeout now outlasts the window, so the watchdog -- with its own
    copy -- decides, and a read timeout is never re-sent."""
    from tldw_chatbook.LLM_Calls import LLM_API_Calls_Local as local

    _no_first_token_config(monkeypatch)
    session = _RecordingSession()
    monkeypatch.setattr(local, "create_default_session", lambda: session)

    local._chat_with_openai_compatible_local_server(
        api_base_url="http://127.0.0.1:9412/v1",
        model_name="qwen",
        input_data=[{"role": "user", "content": "hi"}],
        streaming=False,
        provider_name="Custom OpenAI",
        timeout=120,
        api_retries=2,
    )

    assert session.timeouts and session.timeouts[0] > 300, session.timeouts
    retry = session.adapters[0].max_retries
    assert retry.read == 0, "a read timeout must not re-send the prompt"


def test_the_custom_hosted_engine_read_timeout_outlasts_the_window(
    monkeypatch,
) -> None:
    """The same floor on the engine path a registry custom endpoint uses."""
    from tldw_chatbook.LLM_Calls import hosted_provider_engine as engine
    from tldw_chatbook.provider_registry import CUSTOM_HOSTED

    _no_first_token_config(monkeypatch)
    monkeypatch.setattr(
        engine,
        "resolve_hosted_request",
        lambda record, **_kwargs: engine.HostedProviderResolution(
            provider="custom-hosted",
            model="qwen",
            api_key="",
            base_url="http://127.0.0.1:9412/v1",
            timeout=120.0,
            retries=1,
            retry_delay=1.0,
            streaming=False,
        ),
    )
    captured: dict[str, object] = {}

    def fake_post(**kwargs):
        captured.update(kwargs)
        return {
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "ok"},
                 "finish_reason": "stop"}
            ]
        }

    monkeypatch.setattr(engine, "owned_json_post", fake_post)
    engine.build_hosted_chat_handler(CUSTOM_HOSTED)(
        input_data=[{"role": "user", "content": "hi"}], streaming=False
    )

    assert captured["config"].timeout > 300


def _bridge_run_with_gateway(tmp_path, monkeypatch, gateway, **over):
    from Tests.Chat.test_console_agent_bridge import _bridge_with_gateway, _run
    from tldw_chatbook.Chat import console_agent_bridge as bridge_mod
    from tldw_chatbook.Chat import stream_stall_watchdog as wd

    wd._SESSION_TRACKERS.clear()
    monkeypatch.setattr(bridge_mod, "_stall_timeout_seconds", lambda: 0.1)
    stalls: list[str] = []
    real = wd.record_session_stall

    def spy(session_id, provider, **kw):
        stalls.append(provider)
        return real(session_id, provider, **kw)

    monkeypatch.setattr(bridge_mod, "record_session_stall", spy)
    bridge, _db, store, session, assistant_id = _bridge_with_gateway(
        tmp_path, gateway
    )
    outcome = _run(bridge, store, session, assistant_id, **over)
    wd._SESSION_TRACKERS.clear()
    return outcome, stalls


class _SlowFirstTokenGateway:
    async def stream_chat(self, resolution, messages, tools=None, **kwargs):
        await asyncio.sleep(0.5)  # past the 0.1 s gap window
        yield "hello from a cold model"


def test_the_bridge_gives_a_self_hosted_first_token_its_own_window(
    tmp_path, monkeypatch
) -> None:
    """C-F1: the one line that hands the bridge's real stream path the longer
    window. Deleting it left every other suite green."""
    _no_first_token_config(monkeypatch)

    outcome, stalls = _bridge_run_with_gateway(
        tmp_path / "local", monkeypatch, _SlowFirstTokenGateway()
    )

    assert stalls == []
    assert "hello from a cold model" in (outcome.final_text or "")


def test_the_bridge_keeps_the_stall_window_for_a_cloud_first_token(
    tmp_path, monkeypatch
) -> None:
    from Tests.Chat.test_console_agent_bridge import _test_resolution

    _no_first_token_config(monkeypatch)

    outcome, stalls = _bridge_run_with_gateway(
        tmp_path / "cloud",
        monkeypatch,
        _SlowFirstTokenGateway(),
        resolution=_test_resolution(provider="anthropic", execution_key="anthropic"),
    )

    assert stalls == ["anthropic"]
    summaries = " ".join(getattr(step, "summary", "") for step in outcome.steps)
    assert "no first token after 0.1 s" in summaries, summaries


def test_the_read_floor_follows_a_configured_longer_gap_window(monkeypatch) -> None:
    """A stall window configured above 300 s lengthens the first-token
    window too; the read floor follows it."""
    from tldw_chatbook.Chat.stream_stall_watchdog import self_hosted_read_timeout

    _no_first_token_config(monkeypatch)
    monkeypatch.setenv("TLDW_STREAM_STALL_TIMEOUT_SECONDS", "400")

    assert self_hosted_read_timeout(120) == 430
    assert self_hosted_read_timeout(900) == 900
    monkeypatch.setenv("TLDW_STREAM_STALL_TIMEOUT_SECONDS", "0")  # watchdog off
    assert self_hosted_read_timeout(120) == 120


def test_the_cold_load_hint_is_for_self_hosted_models_only() -> None:
    """A-F6: a cloud reasoning model can think for a while before its first
    visible token; it is not loading anything."""
    from tldw_chatbook.UI.Console_Modules.agent import console_turn_activity_text

    usage = SimpleNamespace(started_at=0.0, output_tokens=0, source="provider")
    snapshot = SimpleNamespace(status="running", steps=(), turn_usage=usage)

    cloud = console_turn_activity_text(snapshot, now=42.0)

    assert cloud.startswith(_WAITING) and "42s" in cloud
    assert "loading" not in cloud


@pytest.mark.parametrize("self_hosted", [True, False])
def test_the_whole_wait_line_fits_the_reply_header_at_120_columns(self_hosted) -> None:
    """B-F3 (live, 120x40): the line rides the assistant row's one-line
    header, which has 64 cells with the rail open; it was cut at 'a large',
    hiding the hint. Every elapsed value up to an hour fits."""
    from rich.cells import cell_len

    from tldw_chatbook.UI.Console_Modules.agent import console_turn_activity_text

    usage = SimpleNamespace(started_at=0.0, output_tokens=3, source="local")
    snapshot = SimpleNamespace(status="running", steps=(), turn_usage=usage)

    for now in (16.0, 99.0, 599.0, 3599.0):
        line = console_turn_activity_text(snapshot, now=now, self_hosted=self_hosted)
        assert line.startswith(_WAITING)
        assert cell_len(line) <= 62, (cell_len(line), line)


class _WaitingController:
    """The controller seams ``console_turn_activity`` reads, for a running
    first send whose bridge has published nothing yet."""

    def __init__(self, provider: str) -> None:
        from tldw_chatbook.UI.Screens.chat_screen import ConsoleRunStatus

        self.run_state = SimpleNamespace(status=ConsoleRunStatus.STREAMING)
        self.store = SimpleNamespace(
            active_session_id="sess-1",
            session_settings=lambda _sid: SimpleNamespace(provider=provider),
        )


def _waiting_view(provider: str):
    from tldw_chatbook.UI.Console_Modules.agent import ConsoleAgentController

    class _View(ConsoleAgentController):
        def __init__(self) -> None:
            self._controller = _WaitingController(provider)

        @property
        def _console_chat_controller(self):
            return self._controller

        @property
        def _console_agent_bridge(self):
            unpublished = SimpleNamespace(status="idle", steps=(), turn_usage=None)
            return SimpleNamespace(live_snapshot=lambda _cid: unpublished)

        @property
        def _current_console_rail_conversation_id(self):
            return lambda: "conv-1"

    return _View()


@pytest.mark.parametrize(
    ("provider", "hinted"), [("custom", True), ("llama_cpp", True), ("openai", False)]
)
def test_the_view_times_an_unpublished_first_send_from_when_it_saw_it_run(
    monkeypatch, provider, hinted
) -> None:
    """C-F4: the wiring, not just the pure function. The view records when it
    first saw the run and hands that in; the session's provider decides the
    hint. Live, the first send showed a bare 'Generating…' for 230 s."""
    from tldw_chatbook.UI.Console_Modules import agent as agent_module

    clock = [1000.0]
    monkeypatch.setattr(
        agent_module, "time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    view = _waiting_view(provider)

    assert view.console_turn_activity().startswith("Generating…")
    clock[0] += 20.0
    line = view.console_turn_activity()

    assert line.startswith(_WAITING), line
    assert "20s" in line
    assert ("may still be loading" in line) is hinted
