"""TASK-34100.2 review round 1: the switcher's per-model capacity rows (AC#2).

C3: no mounted test showed a 'Not ready · window too small' row, so removing
the popover's per-row judgement went unnoticed. F7: rows judged capacity with
the raw provider table's ``max_tokens`` while the header used the chat's own
value, so lowering it in Conversation settings fixed the header but not this
chat's row. F6: a capacity row had no fix and Enter applied the blocked model
with only 'This chat updated'.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
from textual.widgets import Input, Static

from Tests.UI.test_console_model_switcher import (
    Recorder,
    SwitcherHarness,
    _settle,
    line_with,
    list_lines,
    open_switcher,
)
from Tests.UI.test_console_switcher_local_probe import _group
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    CONSOLE_SEND_OUTCOMES,
    ConsoleSessionSettings,
    build_console_settings_readiness,
    build_target_default_console_session_settings,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsDraftState,
    ConsoleSettingsFieldDraft,
    ConsoleSettingsFieldProvenance,
    ConsoleSettingsOrigin,
)
from tldw_chatbook.Widgets.Console.console_model_popover import ConsoleModelPopover

# Review round 1 (TE-2): mounted switcher cases read app config inside the
# test, so they keep the collection-time profile; without it the file passed
# only beside modules that import the app at collection (RecoveryRequired).
pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

#: gpt-4o's catalog window is 128,000 tokens; the provider's replies reserve
#: all of it. gpt-9-unlisted has only an estimated window (capped, AC#1).
CONFIG = {"api_settings": {"openai": {"api_key": "sk-cloud", "max_tokens": 128_000}}}
MODELS = {"openai": ["gpt-4o", "gpt-9-unlisted"]}


@pytest.fixture(autouse=True)
def _fresh_ledger():
    CONSOLE_SEND_OUTCOMES.clear()
    yield
    CONSOLE_SEND_OUTCOMES.clear()


def _draft(model: str, max_tokens: int | None) -> ConsoleSettingsDraftState:
    settings = ConsoleSessionSettings(
        provider="openai", model=model, temperature=0.7, max_tokens=max_tokens
    )
    return ConsoleSettingsDraftState(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        field_drafts=tuple(
            ConsoleSettingsFieldDraft(
                name=name,
                effective_value=getattr(settings, name),
                profile_override=getattr(settings, name),
                provenance=ConsoleSettingsFieldProvenance.INHERITED,
                dirty=False,
            )
            for name in ("temperature", "max_tokens", "streaming")
        ),
        model_drafts=(),
        endpoint_draft=None,
    )


def _rebase(state: ConsoleSettingsDraftState, **kwargs) -> ConsoleSettingsDraftState:
    """Like the controller's: each pair starts from its own inherited values."""
    inherited = build_target_default_console_session_settings(
        CONFIG, kwargs["provider"], kwargs["model"]
    )
    settings = replace(
        state.settings,
        provider=kwargs["provider"],
        model=kwargs["model"],
        max_tokens=inherited.max_tokens,
    )
    return replace(
        state,
        settings=settings,
        field_drafts=tuple(
            replace(
                field,
                effective_value=getattr(settings, field.name),
                profile_override=getattr(settings, field.name),
            )
            for field in state.field_drafts
        ),
    )


def _build(recorder: Recorder, *, current: str, session_max_tokens: int | None):
    def readiness(provider, model):  # ChatScreen._console_default_readiness
        return build_console_settings_readiness(
            build_target_default_console_session_settings(CONFIG, provider, model),
            app_config=CONFIG,
        )

    return ConsoleModelPopover(
        origin=ConsoleSettingsOrigin("session-a", None, 0),
        app_config=CONFIG,
        initial_draft=_draft(current, session_max_tokens),
        providers_models=MODELS,
        scope_copy="Applies to: this chat only",
        durability_copy="Temporary until this chat is promoted",
        draft_rebaser=_rebase,
        live_committer=recorder.commit,
        default_readiness_resolver=readiness,
        setup_opener=lambda provider, model: recorder.setup.append((provider, model)),
    )


async def test_rows_judge_capacity_per_model_under_ready_providers() -> None:
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, _build(recorder, current="gpt-9-unlisted", session_max_tokens=None)
        )
        await _settle(app, pilot)
        lines = list_lines(app, switcher)
        ready = _group(switcher, "READY PROVIDERS")

    assert "Not ready · window too small" in line_with(lines, "gpt-4o")
    # The provider is set up and keyed: it stays a ready provider, and a model
    # whose window is only estimated stays sendable (its reply is capped).
    assert any(row.provider == "openai" for row in ready), ready
    assert "Ready · not tested" in line_with(lines, "gpt-9-unlisted")


@pytest.mark.parametrize(
    ("session_max_tokens", "word"),
    [
        # Conversation settings lowered this chat's maximum: the row agrees
        # with the header, whatever the provider table says.
        (4096, "Ready · not tested"),
        (128_000, "Not ready · window too small"),
    ],
)
async def test_this_chats_row_uses_the_chats_own_reply_maximum(
    session_max_tokens: int, word: str
) -> None:
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, _build(recorder, current="gpt-4o", session_max_tokens=session_max_tokens)
        )
        await _settle(app, pilot)
        lines = list_lines(app, switcher)

    assert word in line_with(lines, "gpt-4o")


async def test_a_full_context_row_names_its_fix_and_enter_refuses_until_fixed() -> None:
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, _build(recorder, current="gpt-9-unlisted", session_max_tokens=None)
        )
        await pilot.press(*"gpt-4o")
        await pilot.pause()
        assert switcher.highlighted_row().model == "gpt-4o"
        assert "lower Max tokens" in line_with(list_lines(app, switcher), "gpt-4o")

        await pilot.press("enter")
        await pilot.pause()

        assert recorder.submissions == []
        error = switcher.query_one("#console-popover-error", Static)
        assert "context window" in str(error.render())
        assert app.focused is switcher.query_one("#console-popover-max-tokens", Input)

        app.focused.value = "4096"
        await pilot.pause()
        switcher.query_one("#console-popover-find", Input).focus()
        await pilot.press("enter")
        await _settle(app, pilot)

    assert len(recorder.submissions) == 1
    assert recorder.submissions[0].draft.settings.model == "gpt-4o"
    assert recorder.submissions[0].draft.settings.max_tokens == 4096


async def test_the_highlighted_row_is_judged_with_the_pending_max_tokens() -> None:
    """Review round 2 (F6): with 200000 typed into the values strip, the
    highlighted row still read 'Ready · not tested' while Enter refused with
    'Replies fill … context window'. The row is now judged with the value
    Enter would apply, so the two agree as the user types."""
    recorder = Recorder()
    app = SwitcherHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        switcher = await open_switcher(
            app, pilot, _build(recorder, current="gpt-4o", session_max_tokens=4096)
        )
        await _settle(app, pilot)
        before = line_with(list_lines(app, switcher), "gpt-4o")
        max_tokens = switcher.query_one("#console-popover-max-tokens", Input)
        max_tokens.focus()
        max_tokens.value = "200000"
        await pilot.pause()
        edited = line_with(list_lines(app, switcher), "gpt-4o")
        max_tokens.value = "4096"
        await pilot.pause()
        restored = line_with(list_lines(app, switcher), "gpt-4o")

    assert "Ready · not tested" in before
    assert "Not ready · window too small" in edited
    assert "Ready · not tested" in restored


# --- 'Fix model limits' lands on Max tokens (review round 2, F5) --------------


async def test_fix_model_limits_opens_settings_on_max_tokens() -> None:
    """Review round 2 (F5): 'Fix model limits' opened Conversation settings
    focused on Provider, away from Max tokens -- the field to change. Since
    TASK-33006.1 Max tokens is a CORE row of Chat settings; a capacity
    verdict focuses it and, not being a connection problem, leaves the
    Connection disclosure closed (pass 2 rebase)."""
    from textual.widgets import Collapsible

    from Tests.UI.test_console_session_settings import ModalHarness
    from tldw_chatbook.Chat.console_session_settings import (
        ConsoleSettingsContextEstimate,
    )
    from tldw_chatbook.Widgets.Console.console_settings_field_row import (
        CONNECTION_DISCLOSURE_ID,
    )
    from tldw_chatbook.Widgets.Console.console_settings_modal import (
        ConsoleSettingsModal,
    )

    app = ModalHarness()
    settings = ConsoleSessionSettings(
        provider="openai", model="gpt-4o", max_tokens=200_000
    )
    assert (
        build_console_settings_readiness(settings, app_config=app.app_config).blocker
        == "capacity_exhausted"
    )
    async with app.run_test(size=(120, 40)) as pilot:
        await app.push_screen(
            ConsoleSettingsModal(
                settings=settings,
                app_config=app.app_config,
                providers_models={"openai": ["gpt-4o"]},
                context_estimate=ConsoleSettingsContextEstimate(10, 128_000, "10 / 128k"),
                can_save=True,
            )
        )
        for _ in range(6):
            await pilot.pause()
        connection = app.screen.query_one(f"#{CONNECTION_DISCLOSURE_ID}", Collapsible)
        focused_id = getattr(app.focused, "id", None)

    assert focused_id == "console-settings-max-tokens"
    assert connection.collapsed is True
