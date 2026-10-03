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

    early = console_turn_activity_text(snapshot, now=5.0)
    late = console_turn_activity_text(snapshot, now=42.0)

    assert early.startswith("Generating…") and "5s" in early
    assert late.startswith("Waiting for the model to start answering")
    assert "42s" in late
    assert "can take a few minutes" in late
    from tldw_chatbook.UI.Console_Modules.composer_run_controls import (
        STOP_RUN_KEY_LABEL,
    )

    assert f"Stop: {STOP_RUN_KEY_LABEL}" in late
