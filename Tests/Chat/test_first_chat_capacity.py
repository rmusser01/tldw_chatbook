"""TASK-34100.2 AC#1 (cross-cutting-01): no model is pre-blocked to zero input.

The first-run review found every OpenAI send refused before dispatch: the
shipped default had no catalog window, the window fell back to 4,096 tokens,
and the 4,096-token reply reservation filled all of it ("Response reservation
and safety margin leave no model input capacity"). PR #2975 dropped the stale
4,096 row; these tests make the state impossible for any *estimated* window
(the reply reservation is capped at a quarter of it) and fail the build when a
shipped default or curated model resolves a window no larger than its
provider's ``max_tokens``.
"""

from __future__ import annotations

import pytest

from Tests.private_profile import private_profile_test
from tldw_chatbook.Chat.console_history_budget import DEFAULT_RESPONSE_RESERVATION
from tldw_chatbook.Chat.console_prepared_request import resolve_request_capacity
from tldw_chatbook.Chat.provider_readiness import provider_config_key
from tldw_chatbook.config import DEFAULT_CONFIG_FROM_TOML
from tldw_chatbook.Utils.token_counter import resolve_context_window


def test_an_estimated_window_keeps_half_of_itself_for_input() -> None:
    capacity = resolve_request_capacity(
        context_window_tokens=4096,
        context_window_verified=False,
        requested_response_tokens=4096,
    )

    assert capacity.limit_source == "estimated"
    assert capacity.requested_response_tokens == 4096
    # 4,096 - 512 margin - 2,048 kept for input.
    assert capacity.effective_response_tokens == 1536
    assert capacity.effective_input_ceiling_tokens == 2048


def test_an_estimated_window_keeps_a_reservation_that_already_fits() -> None:
    """CONTROL: passes on dev 4b90eb860b, as it must. The AC#1 cap leaves
    alone a reply that already leaves input room (review v5r4, TE-R4-3)."""
    capacity = resolve_request_capacity(
        context_window_tokens=32000,
        context_window_verified=False,
        requested_response_tokens=4096,
    )

    assert capacity.effective_response_tokens == 4096


def test_an_estimated_window_never_cuts_a_reply_that_leaves_room() -> None:
    """Review round 1 (F5): a flat window // 4 cap cut every unlisted Gemini
    id's shipped 8,192-token reply to 7,680 (30,720-token provider estimate)
    although 22,000 tokens of input were left. CONTROL: passes on dev
    4b90eb860b, which has no cap; its RED was against this branch's first
    cut (review v5r4, TE-R4-3)."""
    capacity = resolve_request_capacity(
        context_window_tokens=30720,
        context_window_verified=False,
        requested_response_tokens=8192,
    )

    assert capacity.effective_response_tokens == 8192


def test_a_reservation_as_large_as_an_estimated_window_still_leaves_input() -> None:
    capacity = resolve_request_capacity(
        context_window_tokens=32000,
        context_window_verified=False,
        requested_response_tokens=32000,
    )

    assert capacity.effective_response_tokens == 32000 - 640 - 16000
    assert capacity.effective_input_ceiling_tokens == 16000


def test_a_detected_window_keeps_the_configured_reservation() -> None:
    """CONTROL: passes on dev 4b90eb860b, as it must (review v5r4,
    TE-R4-3). A verified window is a fact, not a guess: the readiness
    verdict names the exhausted state ('replies fill the context') instead
    of silently shrinking a reply the user asked for (AC#2)."""
    capacity = resolve_request_capacity(
        context_window_tokens=4096,
        context_window_verified=True,
        requested_response_tokens=4096,
    )

    assert capacity.effective_response_tokens == 4096
    assert capacity.effective_input_ceiling_tokens == 0


def test_a_user_window_override_keeps_the_configured_reservation() -> None:
    """CONTROL: passes on dev 4b90eb860b, as it must. A window the user set
    is not an estimate, so the AC#1 cap leaves its reply alone (review v5r4,
    TE-R4-3)."""
    capacity = resolve_request_capacity(
        context_window_tokens=4096,
        context_window_verified=False,
        context_window_override_tokens=8192,
        requested_response_tokens=4096,
    )

    assert capacity.limit_source == "user_override"
    assert capacity.effective_response_tokens == 4096


def _shipped_provider_rows() -> list[tuple[str, str, int]]:
    """Every shipped (provider, model, reply reservation) a fresh profile sends.

    The provider's ``[api_settings]`` default model plus every curated id in
    its ``[providers]`` list, read from the shipped template -- the values a
    fresh profile (and the env-key path that never sees the wizard) uses. A
    provider whose table sets no integer ``max_tokens`` reserves the Console
    default (``DEFAULT_RESPONSE_RESERVATION``), so it is swept with that,
    never skipped (review round 1, C7: 29 provider lists, 69 curated ids --
    MiniMax, Cloudflare, OpenCodeZen and others -- were silently skipped).
    """
    template = DEFAULT_CONFIG_FROM_TOML
    api_settings = template["api_settings"]
    rows: list[tuple[str, str, int]] = []
    sections = {
        provider_config_key(name): table
        for name, table in api_settings.items()
        if isinstance(table, dict)
    }
    for list_key, models in template["providers"].items():
        key = provider_config_key(list_key)
        table = sections.get(key) or {}
        max_tokens = table.get("max_tokens")
        reservation = (
            max_tokens if type(max_tokens) is int else DEFAULT_RESPONSE_RESERVATION
        )
        ids = [str(table["model"])] if table.get("model") else []
        ids += [str(model) for model in models if str(model).strip()]
        rows += [(key, model, reservation) for model in dict.fromkeys(ids)]
    return rows


def test_the_shipped_template_has_provider_rows_to_sweep() -> None:
    rows = _shipped_provider_rows()
    keys = {key for key, _model, _tokens in rows}
    curated = {
        provider_config_key(list_key)
        for list_key, models in DEFAULT_CONFIG_FROM_TOML["providers"].items()
        if any(str(model).strip() for model in models)
    }

    # Guard the sweep itself: an empty or OpenAI-less sweep proves nothing,
    # and every provider with a curated id is swept.
    assert {"openai", "anthropic", "google", "minimax", "cloudflare"} <= keys
    assert curated <= keys
    assert len(rows) > 120


@private_profile_test
def test_no_shipped_default_or_curated_model_starts_with_zero_input_capacity(
    request: pytest.FixtureRequest,
) -> None:
    """The send preflight's own check, for every swept row (AC#1)."""
    from tldw_chatbook.Chat.console_session_settings import (
        console_reply_capacity_exhausted,
    )

    exhausted = [
        f"{key}/{model} at {reservation}"
        for key, model, reservation in _shipped_provider_rows()
        if console_reply_capacity_exhausted(key, model, reservation)
    ]

    assert exhausted == []


@private_profile_test
def test_every_shipped_default_and_curated_model_resolves_more_than_max_tokens(
    request: pytest.FixtureRequest,
) -> None:
    too_small = [
        f"{key}/{model}: window {window.tokens} <= max_tokens {max_tokens}"
        for key, model, max_tokens in _shipped_provider_rows()
        if (window := resolve_context_window(key, model)).tokens <= max_tokens
    ]

    assert too_small == []


# --- the cap reaches every send path (review round 1, F1 live) ---------------


@pytest.mark.asyncio
async def test_a_send_resolves_the_capped_reply_for_every_path(monkeypatch) -> None:
    """Live: a fresh OpenAI profile with max_tokens = 32,000 read 'Ready · not
    tested' for gpt-5.6-terra (~32k estimated window) while every send failed
    'Agent run failed: first request exceeds model context budget' -- the
    Agents fit check reserved the whole 32,000 against the 32,000 estimate.
    The cap is applied where a send resolves, so the Agents reservation,
    history bounding and the wire ``max_tokens`` all reserve the same reply."""
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import build_first_request_schema_plan
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )

    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_kwargs: None)

    async def unclassified(_selection):
        return ConsoleProviderResolution(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-5.6-terra",
            ready=True,
            execution_key="openai",
            readiness_key="openai",
            api_key="sk-test-first-chat-capacity",
            max_tokens=32000,
        )

    monkeypatch.setattr(gateway, "_resolve_for_send_unclassified", unclassified)
    monkeypatch.setattr(gateway, "_schedule_context_window_refresh", lambda _t: None)

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-5.6-terra")
    )

    assert resolution.context_window.verified is False
    assert resolution.max_tokens == 32000 - 640 - 16000
    plan = build_first_request_schema_plan(
        ToolCatalogRegistry(),
        (),
        AgentConfig(
            model="gpt-5.6-terra",
            system_prompt="You are a helpful assistant.",
            allowed_tools=(),
            response_reserve_tokens=resolution.max_tokens,
        ),
        "openai",
        [{"role": "user", "content": "Reply with OK"}],
        skill_file_enabled=False,
        install_skill_enabled=False,
        run_skill_script_enabled=False,
        run_log_active=False,
    )
    assert plan.request_fits is True


@pytest.mark.asyncio
async def test_a_verified_window_keeps_the_configured_reply_at_send(monkeypatch) -> None:
    """CONTROL: passes on dev 4b90eb860b, as it must. A send to a verified
    window keeps the configured reply maximum; the cap is for estimates only
    (review v5r4, TE-R4-3)."""
    from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
    from tldw_chatbook.Chat.console_provider_gateway import (
        ConsoleProviderGateway,
        ConsoleProviderResolution,
    )

    gateway = ConsoleProviderGateway(chat_api_call_fn=lambda **_kwargs: None)

    async def unclassified(_selection):
        return ConsoleProviderResolution(
            provider="openai",
            base_url="https://api.openai.com/v1",
            model="gpt-4o",
            ready=True,
            execution_key="openai",
            readiness_key="openai",
            api_key="sk-test-first-chat-capacity",
            max_tokens=16000,
        )

    monkeypatch.setattr(gateway, "_resolve_for_send_unclassified", unclassified)
    monkeypatch.setattr(gateway, "_schedule_context_window_refresh", lambda _t: None)

    resolution = await gateway.resolve_for_send(
        ConsoleProviderSelection(provider="openai", explicit_model="gpt-4o")
    )

    assert resolution.context_window.verified is True  # gpt-4o's catalog window
    assert resolution.max_tokens == 16000


def test_a_readiness_build_never_imports_the_request_builder(tmp_path) -> None:
    """Review round 2 (R2-4): the capacity check imported
    ``console_prepared_request`` (the Agents runtime, LLM_API_Calls,
    Chat_Functions -- about 120 modules), so the first readiness build of a
    process, Home's first paint among them, paid that import on the UI
    thread. The capacity rule lives in a dependency-light module."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    code = """
import sys
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    build_console_settings_readiness,
)
readiness = build_console_settings_readiness(
    ConsoleSessionSettings(provider="openai", model="gpt-4o", max_tokens=128000),
    app_config={"api_settings": {"openai": {"api_key": "sk-import-probe"}}},
    environ={},
)
assert readiness.blocker == "capacity_exhausted", readiness.blocker
heavy = sorted(
    name
    for name in (
        "tldw_chatbook.Chat.console_prepared_request",
        "tldw_chatbook.Agents.agent_runtime",
        "tldw_chatbook.Chat.Chat_Functions",
    )
    if name in sys.modules
)
assert not heavy, heavy
"""
    repo_root = Path(__file__).resolve().parents[2]
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in {"TLDW_CONFIG_PATH", "XDG_CONFIG_HOME", "XDG_DATA_HOME"}
    }
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=str(repo_root),
        PYTHON_KEYRING_BACKEND="keyring.backends.null.Keyring",
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        cwd=str(tmp_path),
        env=env,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr[-2000:]


def test_the_request_builder_reexports_the_one_capacity_rule() -> None:
    """Structural pin (re-export identity, not behavioural evidence): the
    capacity behaviour itself is covered by the cap tests above."""
    from tldw_chatbook.Chat import console_prepared_request, console_request_capacity

    for name in (
        "ConsoleRequestCapacity",
        "ESTIMATED_WINDOW_RESERVATION_DIVISOR",
        "MINIMUM_SAFETY_MARGIN_TOKENS",
        "estimated_window_reply_cap",
        "resolve_request_capacity",
    ):
        assert getattr(console_prepared_request, name) is getattr(
            console_request_capacity, name
        )


# Pass 3 review round 1 (R1-6): AC#1's "no model is pre-blocked" holds above a
# floor. At or below 682 estimated tokens the 512-token safety margin plus the
# quarter-window reservation fill the window; the rule states that floor, and
# every window the resolver can ESTIMATE is far above it.
from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402


@settings(max_examples=300, deadline=None)
@given(
    window=st.integers(min_value=1, max_value=50_000_000),
    requested=st.integers(min_value=1, max_value=50_000_000),
)
def test_an_estimated_window_at_or_above_the_floor_always_leaves_input(
    window: int, requested: int
) -> None:
    from tldw_chatbook.Chat.console_request_capacity import (
        ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS,
    )

    capacity = resolve_request_capacity(
        context_window_tokens=window,
        context_window_verified=False,
        requested_response_tokens=requested,
    )

    assert (capacity.effective_input_ceiling_tokens > 0) is (
        window >= ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS
    )


def test_every_window_the_resolver_estimates_is_above_the_floor() -> None:
    """The estimated windows come only from the provider fallbacks and the
    application fallback (a catalog or server window is verified)."""
    from tldw_chatbook.Chat.console_request_capacity import (
        ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS,
    )
    from tldw_chatbook.Utils.token_counter import (
        PROVIDER_CONTEXT_WINDOWS,
        SYSTEM_CONTEXT_WINDOW,
    )

    estimated = [*PROVIDER_CONTEXT_WINDOWS.values(), SYSTEM_CONTEXT_WINDOW]
    assert min(estimated) >= ESTIMATED_WINDOW_INPUT_FLOOR_TOKENS
    assert resolve_context_window("openai", "an-unlisted-model").verified is False
    assert resolve_context_window("openai", "an-unlisted-model").tokens in estimated
