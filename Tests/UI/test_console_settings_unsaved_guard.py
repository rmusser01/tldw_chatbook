"""Chat settings asks before a close gesture discards unapplied edits.

TASK-33003.5 (ADR-031 task-16211): Esc, a backdrop click and Cancel never
discard an edited Chat settings draft by themselves. They open an
unsaved-edits prompt (Apply to this chat / Discard / Keep editing) that names
the edited fields; an unedited draft still closes at once.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import replace
from pathlib import Path

import pytest
from textual.containers import Vertical
from textual.widgets import Button, Input, Select, Static

import tldw_chatbook.UI.Screens.settings_screen as settings_screen_module
import tldw_chatbook.Widgets.Console.console_settings_modal as settings_modal_module
from Tests.Chat.test_console_session_settings import (
    _settings_close_modal,
    _SettingsCloseHarness,
)
from Tests.private_profile import private_profile_test
from Tests.UI.app_factory import _build_test_app as _build_production_app
from Tests.UI.consolidated_css import ConsolidatedCSSApp
from Tests.UI.test_console_provider_apply_defaults_flow import (
    _ConsoleFlowHarness,
    _drain_settings_tasks,
    _persisted_console_app,
    _reset_default_intent_state,  # noqa: F401 - shared autouse fixture
)
from Tests.UI.test_destination_shells import _wait_for_selector
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Chat.console_context_policy import ConsoleContextPolicyOverrides
from tldw_chatbook.Chat.console_session_settings import (
    ConsoleSessionSettings,
    ConsoleSettingsContextEstimate,
)
from tldw_chatbook.Chat.console_settings_apply import (
    ConsoleSettingsAction,
    ConsoleSettingsCommittedSubmission,
)
from tldw_chatbook.config import ConfigMutationResult
from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
from tldw_chatbook.UI.Screens.settings_screen import SettingsScreen
from tldw_chatbook.Widgets.Console.console_settings_modal import (
    ConsoleSettingsDraftSnapshot,
    ConsoleSettingsModal,
)
from tldw_chatbook.Widgets.Console.console_settings_unsaved import (
    chat_settings_values,
    esc_hint_copy,
    unsaved_baseline,
    unsaved_labels,
    unsaved_prompt_copy,
)
from tldw_chatbook.Widgets.model_search_picker import ModelSearchPicker

_PROMPT_KEYS = "Enter apply · d discard · Esc keep editing"


class _GuardHarness(ConsolidatedCSSApp):
    """Mount Chat settings under the production stylesheet."""

    CSS_PATH = TldwCli.CSS_PATH

    def __init__(self) -> None:
        super().__init__()
        self.results: list[object] = []

    def capture(self, result: object) -> None:
        self.results.append(result)


def _modal(**kwargs: object) -> ConsoleSettingsModal:
    return ConsoleSettingsModal(
        settings=ConsoleSessionSettings(
            provider="llama_cpp", model="model-a", temperature=0.7
        ),
        app_config={
            "api_settings": {"llama_cpp": {"api_url": "http://127.0.0.1:9099"}}
        },
        providers_models={"llama_cpp": ["model-a", "model-b"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
        **kwargs,
    )


async def _open(app, pilot, modal: ConsoleSettingsModal) -> None:
    await app.push_screen(modal, callback=app.capture)
    await _settle(pilot, modal)


async def _settle(pilot, modal: ConsoleSettingsModal) -> None:
    """Wait until the modal has recorded what an unedited draft looks like."""
    for _ in range(40):
        await pilot.pause()
        if getattr(modal, "_unsaved_baseline", None) is not None:
            break


def _text(modal: ConsoleSettingsModal, selector: str) -> str:
    return str(modal.query_one(selector, Static).renderable)


def _visible_prompt_buttons(modal: ConsoleSettingsModal) -> list[str]:
    return [
        str(button.label)
        for button in modal.query("#console-settings-close-guard Button")
        if button.display
    ]


async def _edit(pilot, modal: ConsoleSettingsModal, selector: str, value: str):
    control = modal.query_one(selector, Input)
    control.focus()
    control.value = value
    await pilot.pause()
    await pilot.pause()
    return control


async def _gesture(pilot, source: str) -> None:
    if source == "cancel":
        await pilot.click("#console-settings-cancel")
    elif source == "escape":
        await pilot.press("escape")
    else:
        await pilot.click(offset=(0, 0))
    await pilot.pause()
    await pilot.pause()


def test_unsaved_labels_follow_effective_values_and_carried_edits() -> None:
    """A field is unsaved when its effective value left the committed one."""
    committed = chat_settings_values(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1", temperature=0.7),
        ConsoleContextPolicyOverrides(),
        None,
    )
    # The quick surface carried Temperature 0.23 in; the controls normalized
    # the blank committed Endpoint to a configured default at mount.
    opened = dict(committed, Temperature=0.23)
    mounted = dict(opened, Endpoint="https://api.openai.com/v1")
    baseline = unsaved_baseline(mounted, opened, committed)

    assert unsaved_labels(mounted, baseline) == ("Temperature",)
    assert unsaved_labels(dict(mounted, Temperature=0.7), baseline) == ()
    assert unsaved_labels(dict(mounted, Model="gpt-5"), baseline) == (
        "Model",
        "Temperature",
    )
    invalid_context = chat_settings_values(
        ConsoleSessionSettings(provider="openai", model="gpt-4.1", temperature=0.7),
        None,
        None,
    )
    assert unsaved_labels(invalid_context, committed) == ("Context and memory",)
    assert esc_hint_copy(0) == "Esc close"
    assert esc_hint_copy(2) == "Esc close (asks: 2 unsaved)"
    # A side-effect guard Esc opens first outranks the unsaved count.
    assert esc_hint_copy(2, pending="memory reset") == "Esc close (asks: memory reset)"
    prompt = unsaved_prompt_copy(("Model", "Temperature"))
    assert "Model, Temperature" in prompt
    assert _PROMPT_KEYS in prompt
    blocked = unsaved_prompt_copy(("Model",), can_apply=False)
    assert "Enter apply" not in blocked
    assert "d discard · Esc keep editing" in blocked


def test_snapshot_carries_the_unsaved_baseline_and_refuses_a_malformed_one() -> None:
    """The suspended draft keeps the first modal's baseline, fail-closed."""
    settings = ConsoleSessionSettings(provider="llama_cpp", model="model-a")
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=settings,
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={},
        provider_model_drafts={},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id=None,
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    mapping = snapshot.to_mapping()
    # A draft captured before the modal's initial sync carries no baseline.
    legacy = {key: value for key, value in mapping.items() if key != "unsaved_baseline"}
    assert ConsoleSettingsDraftSnapshot.from_mapping(legacy).unsaved_baseline is None
    for baseline in (
        chat_settings_values(settings, ConsoleContextPolicyOverrides(), "Ada"),
        chat_settings_values(settings, None, None),
    ):
        restored = ConsoleSettingsDraftSnapshot.from_mapping(
            dict(mapping, unsaved_baseline=baseline)
        )
        assert restored is not None and restored.unsaved_baseline == baseline
    for malformed in ({"Not a field": 1}, {"Model": object()}, ["Model"]):
        assert (
            ConsoleSettingsDraftSnapshot.from_mapping(
                dict(mapping, unsaved_baseline=malformed)
            )
            is None
        )


@pytest.mark.asyncio
async def test_prompt_labels_match_the_labels_the_modal_renders() -> None:
    """Final review M8/5.3: the prompt's labels are copies of the modal's
    literals, so a renamed field label must not leave the prompt naming a
    field the user cannot find."""
    from tldw_chatbook.Widgets.Console.console_settings_unsaved import (
        _BASELINE_LABELS,
        _CONTEXT_INVALID_LABEL,
    )

    app = _GuardHarness()
    async with app.run_test(size=(211, 44)) as pilot:
        modal = _modal()
        await _open(app, pilot, modal)
        rendered = {
            str(label.renderable)
            for label in modal.query(".console-settings-modal-label")
        }
    assert _BASELINE_LABELS - {_CONTEXT_INVALID_LABEL} <= rendered, sorted(
        _BASELINE_LABELS - {_CONTEXT_INVALID_LABEL} - rendered
    )


@pytest.mark.parametrize("source", ["escape", "backdrop", "cancel"])
@pytest.mark.asyncio
async def test_edited_draft_asks_before_each_close_gesture(source: str) -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        assert _text(modal, "#console-settings-esc-hint") == "Esc close"

        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
        assert (
            _text(modal, "#console-settings-esc-hint")
            == "Esc close (asks: 1 unsaved)"
        )
        await _gesture(pilot, source)

        assert app.screen is modal
        assert app.results == []
        assert modal.query_one("#console-settings-close-guard", Vertical).display
        message = _text(modal, "#console-settings-close-message")
        assert "Temperature" in message
        assert _PROMPT_KEYS in message
        assert _visible_prompt_buttons(modal) == [
            "Apply to this chat",
            "Discard",
            "Keep editing",
        ]
        assert modal.focused is modal.query_one(
            "#console-settings-close-apply", Button
        )


@pytest.mark.asyncio
async def test_prompt_without_an_available_apply_offers_discard_and_keep() -> None:
    app = _GuardHarness()
    modal = _modal(active_run=True)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        assert modal.query_one("#console-settings-save", Button).disabled
        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
        await _gesture(pilot, "escape")

        apply = modal.query_one("#console-settings-close-apply", Button)
        assert apply.display and apply.disabled
        assert modal.focused is modal.query_one(
            "#console-settings-close-return", Button
        )
        message = _text(modal, "#console-settings-close-message")
        assert "Enter apply" not in message
        assert "d discard · Esc keep editing" in message
        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert app.screen is modal
        assert app.results == []
        assert not modal.query_one("#console-settings-close-guard", Vertical).display


@pytest.mark.parametrize("source", ["escape", "backdrop", "cancel"])
@pytest.mark.asyncio
async def test_unedited_draft_closes_immediately(source: str) -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        temperature = await _edit(
            pilot, modal, "#console-settings-temperature", "0.9"
        )
        # Restoring the committed value is not an edit, even spelled
        # differently: the effective value is what counts.
        temperature.value = "0.70"
        await pilot.pause()
        await pilot.pause()
        assert _text(modal, "#console-settings-esc-hint") == "Esc close"

        await _gesture(pilot, source)

        assert app.results == [None]
        assert app.screen is not modal


async def _round_trip_edit(pilot, modal: ConsoleSettingsModal, edit: str) -> None:
    """Make one edit a credential round-trip must carry as unsaved."""
    if edit == "temperature":
        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
    elif edit == "endpoint":
        await _edit(pilot, modal, "#console-settings-base-url", "http://127.0.0.1:9100")
    elif edit == "provider":
        modal.query_one("#console-settings-provider", Select).value = "openai"
    elif edit == "model":
        # The route a user takes: committing a catalog row.
        modal.query_one(ModelSearchPicker)._commit_catalog_model("model-b")
    elif edit == "streaming":
        modal.query_one("#console-settings-streaming", Button).press()
    for _ in range(4):
        await pilot.pause()


@pytest.mark.parametrize(
    ("edit", "label"),
    [
        (None, None),
        ("temperature", "Temperature"),
        ("provider", "Provider"),
        ("model", "Model"),
        ("endpoint", "Endpoint"),
        ("streaming", "Streaming"),
    ],
)
@pytest.mark.asyncio
async def test_suspended_draft_round_trip_keeps_its_edits_unsaved(
    edit: str | None, label: str | None
) -> None:
    """A credential round-trip reopens the draft; its edits still ask.

    The snapshot's ``settings`` are the ones the first modal opened with;
    Provider, Model, Endpoint and Streaming edits are already in the reopened
    modal's controls when it composes, so only the first modal knows what
    they were unedited. The committed ``base_url`` is blank and shown as the
    configured default, which is not an edit either.
    """
    app = _GuardHarness()
    first = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, first)
        if edit is not None:
            await _round_trip_edit(pilot, first, edit)
        labels = first._unsaved_field_labels()
        assert (label in labels) if label else labels == ()
        snapshot = first.capture_suspended_draft()
        app.pop_screen()
        await pilot.pause()

        modal = _modal(suspended_draft=snapshot)
        await _open(app, pilot, modal)
        assert modal._unsaved_field_labels() == labels
        assert _text(modal, "#console-settings-esc-hint") == esc_hint_copy(len(labels))
        await _gesture(pilot, "cancel")

        if not labels:
            assert app.results == [None]
            assert app.screen is not modal
            return
        assert app.results == []
        assert app.screen is modal
        assert _text(modal, "#console-settings-close-message").startswith(
            unsaved_prompt_copy(labels).split("\n")[0]
        )


@pytest.mark.parametrize("how", ["escape", "button"])
@pytest.mark.asyncio
async def test_keep_editing_returns_focus_to_the_edited_control(how: str) -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        temperature = await _edit(
            pilot, modal, "#console-settings-temperature", "0.9"
        )
        await _gesture(pilot, "escape")
        assert modal.query_one("#console-settings-close-guard", Vertical).display

        if how == "escape":
            await pilot.press("escape")
        else:
            await pilot.click("#console-settings-close-return")
        await pilot.pause()
        await pilot.pause()

        assert not modal.query_one("#console-settings-close-guard", Vertical).display
        assert modal.focused is temperature
        assert temperature.value == "0.9"
        assert app.screen is modal
        assert app.results == []
        assert (
            str(modal.query_one("#console-settings-close-return", Button).label)
            == "Keep editing"
        )


@pytest.mark.parametrize("how", ["key", "button"])
@pytest.mark.asyncio
async def test_discard_closes_without_applying(how: str) -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
        await _gesture(pilot, "cancel")
        assert app.results == []
        assert modal.query_one("#console-settings-close-guard", Vertical).display

        if how == "key":
            await pilot.press("d")
        else:
            await pilot.click("#console-settings-close-discard")
        await pilot.pause()

        assert app.results == [None]
        assert app.screen is not modal


@pytest.mark.asyncio
async def test_d_types_into_a_field_while_no_prompt_is_open() -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        name = modal.query_one("#console-settings-user-display-name", Input)
        name.focus()
        await pilot.pause()
        await pilot.press("d")
        await pilot.pause()

        assert name.value == "d"
        assert app.screen is modal
        assert app.results == []
        assert not modal.query_one("#console-settings-close-guard", Vertical).display


@pytest.mark.asyncio
async def test_apply_from_prompt_uses_the_apply_path_and_writes_no_config(
    monkeypatch,
) -> None:
    def no_config_write(*_args, **_kwargs):
        raise AssertionError("Apply to this chat must not write configuration")

    monkeypatch.setattr(
        settings_modal_module, "save_settings_to_cli_config", no_config_write
    )
    submissions = []

    def live_committer(submission):
        submissions.append(submission)
        return ConsoleSettingsModal._transitional_live_commit(submission)

    app = _GuardHarness()
    modal = _modal(live_committer=live_committer)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
        await _gesture(pilot, "escape")

        await pilot.press("enter")
        await pilot.pause()

        assert app.screen is not modal
    assert len(submissions) == 1
    assert submissions[0].action is ConsoleSettingsAction.APPLY_TO_CHAT
    assert submissions[0].default_field_mask == frozenset()
    assert submissions[0].draft.settings.temperature == pytest.approx(0.9)
    assert len(app.results) == 1
    assert isinstance(app.results[0], ConsoleSettingsCommittedSubmission)


@pytest.mark.asyncio
async def test_apply_from_prompt_with_invalid_draft_stays_open() -> None:
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        temperature = await _edit(
            pilot, modal, "#console-settings-temperature", "hot"
        )
        await _gesture(pilot, "escape")
        assert "Temperature" in _text(modal, "#console-settings-close-message")

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()

        assert app.screen is modal
        assert app.results == []
        assert not modal.query_one("#console-settings-close-guard", Vertical).display
        error = modal.query_one("#console-settings-error", Static)
        assert error.display
        assert str(error.renderable).strip()
        assert modal.focused is temperature


@pytest.mark.asyncio
async def test_context_view_edit_is_named_in_the_prompt() -> None:
    app = _GuardHarness()
    modal = _modal(focus_context=True)
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        mode = modal.query_one("#console-context-compaction-mode", Select)
        mode.value = "off" if mode.value != "off" else "ask"
        await pilot.pause()
        await pilot.pause()

        await _gesture(pilot, "escape")

        assert app.screen is modal
        assert "When limit nears" in _text(modal, "#console-settings-close-message")


@pytest.mark.parametrize(
    "choice", ["#console-settings-close-undo", "#console-settings-close-keep"]
)
@pytest.mark.asyncio
async def test_memory_reset_guard_comes_first_then_asks_about_edits(
    choice: str,
) -> None:
    app = _SettingsCloseHarness()
    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: True,
    )
    async with app.run_test(size=(120, 42)) as pilot:
        await app.push_screen(modal, callback=app.capture)
        await _settle(pilot, modal)
        mode = modal.query_one("#console-context-compaction-mode", Select)
        mode.value = "off" if mode.value != "off" else "ask"
        await pilot.pause()
        modal.query_one("#console-context-reset-current", Button).press()
        await pilot.pause()

        await pilot.press("escape")
        await pilot.pause()
        # Reset guard first, with its own copy and buttons.
        assert modal._settings_close_guard_mode == "reset"
        assert _visible_prompt_buttons(modal) == [
            "Undo and close",
            "Keep reset and close",
            "Return",
        ]

        await pilot.click(choice)
        await pilot.pause()
        await pilot.pause()
        assert app.screen is modal
        assert app.results == []
        assert "When limit nears" in _text(modal, "#console-settings-close-message")
        assert _visible_prompt_buttons(modal) == [
            "Apply to this chat",
            "Discard",
            "Keep editing",
        ]

        await pilot.press("d")
        await pilot.pause()
        assert app.results == [None]


@pytest.mark.asyncio
async def test_esc_hint_names_the_reset_and_compaction_guards_esc_opens() -> None:
    """Final review I4 (ADR-031 task-16211): with no edits, a pending memory
    reset or a running compaction still makes Esc ask first, so the hint must
    say so, and must drop back to "Esc close" once undo or compaction ends."""
    app = _SettingsCloseHarness()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(
        reset_current_memory=lambda: ("memory-1", 2),
        undo_current_memory_reset=lambda _memory_id, _revision: True,
        compact_now=compact_now,
    )
    hint = "#console-settings-esc-hint"
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            await _settle(pilot, modal)
            assert _text(modal, hint) == "Esc close"

            modal.query_one("#console-context-reset-current", Button).press()
            await pilot.pause()
            await pilot.pause()
            assert _text(modal, hint) == "Esc close (asks: memory reset)"
            modal.query_one("#console-context-undo-reset", Button).press()
            await pilot.pause()
            await pilot.pause()
            assert _text(modal, hint) == "Esc close"

            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await pilot.pause()
            assert _text(modal, hint) == "Esc close (asks: compaction running)"
            release.set()
            for _ in range(20):
                await pilot.pause()
                if not modal._compaction_is_active():
                    break
            await pilot.pause()
            await pilot.pause()
            assert _text(modal, hint) == "Esc close"
            assert app.results == []
    finally:
        release.set()


@pytest.mark.asyncio
async def test_compaction_close_anyway_then_asks_about_edits() -> None:
    app = _SettingsCloseHarness()
    entered = asyncio.Event()
    release = asyncio.Event()

    async def compact_now() -> tuple[bool, str]:
        entered.set()
        await release.wait()
        return True, "Compaction complete."

    modal = _settings_close_modal(compact_now=compact_now)
    try:
        async with app.run_test(size=(120, 42)) as pilot:
            await app.push_screen(modal, callback=app.capture)
            await _settle(pilot, modal)
            mode = modal.query_one("#console-context-compaction-mode", Select)
            mode.value = "off" if mode.value != "off" else "ask"
            await pilot.pause()
            modal.query_one("#console-context-compact-now", Button).press()
            await pilot.pause()
            await asyncio.wait_for(entered.wait(), timeout=1)

            await pilot.press("escape")
            await pilot.pause()
            assert modal._settings_close_guard_mode == "compaction"
            assert _visible_prompt_buttons(modal) == ["Close anyway", "Return"]

            await pilot.click("#console-settings-close-anyway")
            await pilot.pause()
            await pilot.pause()
            assert app.results == []
            assert "When limit nears" in _text(
                modal, "#console-settings-close-message"
            )

            await pilot.press("d")
            await pilot.pause()
            assert app.results == [None]
            assert any("may still be billed" in notice for notice in app.notices)
    finally:
        release.set()


@pytest.mark.asyncio
@private_profile_test
async def test_quick_surface_edit_carried_in_asks_then_applies_to_this_chat(
    request,
):
    """Real Console: a transferred edit counts; Apply commits it, not config."""
    app = _persisted_console_app()
    harness = _ConsoleFlowHarness(app)
    async with harness.run_test(size=(211, 44)) as pilot:
        console = harness.screen
        await _wait_for_selector(console, pilot, "#console-settings-summary")
        store = console._ensure_console_chat_store()
        session_id = store.active_session_id
        committed_temperature = store.session_settings(session_id).temperature
        config_path = Path(os.environ["TLDW_CONFIG_PATH"])
        config_before = config_path.read_bytes()

        await console._open_console_settings(focus_model=True)
        await pilot.pause()
        full = harness.screen
        assert isinstance(full, ConsoleSettingsModal)
        await _settle(pilot, full)
        # (A focused picker takes the first Escape itself; start from a field.)
        full.query_one("#console-settings-temperature", Input).focus()
        await _gesture(pilot, "escape")
        # Nothing edited: Esc closes at once.
        assert harness.screen is console

        await console.action_open_console_model_popover()
        await pilot.pause()
        quick = harness.screen
        quick.query_one("#console-popover-temperature", Input).value = "0.23"
        await pilot.pause()
        quick.query_one("#console-popover-full-settings", Button).press()
        await pilot.pause()
        full = harness.screen
        assert isinstance(full, ConsoleSettingsModal)
        await _settle(pilot, full)
        assert (
            _text(full, "#console-settings-esc-hint")
            == "Esc close (asks: 1 unsaved)"
        )

        full.query_one("#console-settings-temperature", Input).focus()
        await _gesture(pilot, "escape")
        assert harness.screen is full
        assert "Temperature" in _text(full, "#console-settings-close-message")
        assert store.session_settings(session_id).temperature == pytest.approx(
            committed_temperature
        )

        await pilot.press("enter")
        await pilot.pause()
        assert harness.screen is console
        await _drain_settings_tasks(harness.app_instance)
        assert store.session_settings(session_id).temperature == pytest.approx(0.23)
        assert config_path.read_bytes() == config_before


@pytest.mark.parametrize("edit", ["temperature", "model"])
@pytest.mark.asyncio
async def test_credential_round_trip_keeps_the_restored_edit_unsaved(
    monkeypatch, edit: str
):
    """Real router: Configure credential -> Settings -> Return keeps the ask.

    The returned modal is rebuilt from the suspended draft; the edits made
    before the handoff must still count, and nothing else may. The reopened
    modal composes Model from the draft, so only the first modal knows its
    unedited value.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        settings_screen_module,
        "persist_provider_settings_atomic",
        lambda *_args, **_kwargs: ConfigMutationResult(True, True, None),
    )
    app = _build_production_app(configured_default="chat")
    app.chat_api_provider_value = "openai"
    app.chat_api_model_value = "gpt-5"
    app.app_config["chat_defaults"] = {"provider": "openai", "model": "gpt-5"}
    app.app_config["api_settings"] = {"openai": {}}
    app.providers_models = {"openai": ["gpt-5", "model-b"]}
    committed = ConsoleSessionSettings(
        provider="openai", model="gpt-5", temperature=0.7
    )

    async with app.run_test(size=(211, 44)) as pilot:
        console = None
        for _ in range(200):
            console = app._navigation_outgoing_screen()
            if isinstance(console, ChatScreen):
                break
            await pilot.pause(0.05)
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(
            console, pilot, "#console-native-composer", timeout=10.0
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(session.id, committed)
        assert await console._open_console_settings() is True
        for _ in range(80):
            if isinstance(app.screen, ConsoleSettingsModal):
                break
            await pilot.pause(0.05)
        first = app.screen
        assert isinstance(first, ConsoleSettingsModal)
        await _settle(pilot, first)
        await _round_trip_edit(pilot, first, edit)
        labels = first._unsaved_field_labels()
        # Switching model re-bases Streaming's explicit On to Inherit, which
        # counts: Inherit is a draft value of its own (Qodo #2937).
        assert labels == (
            ("Temperature",) if edit == "temperature" else ("Model", "Streaming")
        )
        await pilot.click("#console-settings-configure-credential")

        settings = None
        for _ in range(200):
            if isinstance(app.screen, SettingsScreen):
                settings = app.screen
                break
            await pilot.pause(0.05)
        assert settings is not None
        await _wait_for_selector(
            settings, pilot, "#settings-provider-api-key", timeout=10.0
        )
        key = settings.query_one("#settings-provider-api-key", Input)
        key.value = "DUMMY-ROUND-TRIP-KEY"
        await pilot.pause()
        settings.action_settings_save_category(allow_text_entry_focus=True)
        for _ in range(80):
            if settings.query_one("#settings-provider-return", Button).has_focus:
                break
            await pilot.pause(0.05)
        settings.query_one("#settings-provider-return", Button).press()

        returned = None
        for _ in range(240):
            top = app.screen_stack[-1]
            if isinstance(top, ConsoleSettingsModal) and top is not first:
                returned = top
                break
            await pilot.pause(0.05)
        assert returned is not None
        await _settle(pilot, returned)
        assert returned._unsaved_field_labels() == labels
        assert _text(returned, "#console-settings-esc-hint") == esc_hint_copy(
            len(labels)
        )
        await _gesture(pilot, "cancel")

        assert app.screen is returned
        assert _text(returned, "#console-settings-close-message").startswith(
            unsaved_prompt_copy(labels).split("\n")[0]
        )
        assert store.session_settings(session.id) == committed


async def _cycle_streaming_to_inherit(pilot, modal: ConsoleSettingsModal) -> None:
    """Press Streaming until the draft says Inherit (no per-chat override)."""
    for _ in range(3):
        if modal._streaming_draft is None:
            break
        modal.query_one("#console-settings-streaming", Button).press()
        await pilot.pause()
    assert modal._streaming_draft is None


def _streaming_label(modal: ConsoleSettingsModal) -> str:
    return str(modal.query_one("#console-settings-streaming", Button).label)


@pytest.mark.asyncio
async def test_inherit_streaming_survives_the_suspended_draft_round_trip() -> None:
    """TASK-33003.10: Inherit is a draft value the snapshot must carry.

    The real capture path wrote ``None`` for Inherit and the snapshot refused
    it with ``ValueError``; the reopened modal also coerced it to Off.
    """
    app = _GuardHarness()
    first = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, first)
        await _cycle_streaming_to_inherit(pilot, first)
        snapshot = first.capture_suspended_draft()
        assert snapshot.raw_values["console-settings-streaming"] is None
        restored = ConsoleSettingsDraftSnapshot.from_mapping(snapshot.to_mapping())
        assert restored is not None
        assert restored.raw_values["console-settings-streaming"] is None
        app.pop_screen()
        await pilot.pause()

        modal = _modal(suspended_draft=restored)
        await _open(app, pilot, modal)
        assert modal._streaming_draft is None
        assert _streaming_label(modal) == "Inherit"



@pytest.mark.parametrize(
    "control_id",
    [
        "console-context-budget-mode",
        "console-context-compaction-mode",
        "console-context-failure-behavior",
        "console-context-carry-forward",
    ],
)
@pytest.mark.asyncio
async def test_blank_context_choice_survives_the_suspended_draft_round_trip(
    control_id: str,
) -> None:
    """TASK-33003.10 sibling: a blank Select is captured as "" and the
    reopened modal raised InvalidSelectValueError assigning it back."""
    app = _GuardHarness()
    first = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, first)
        first.query_one(f"#{control_id}", Select).clear()
        await pilot.pause()
        snapshot = first.capture_suspended_draft()
        assert snapshot.raw_values[control_id] == ""
        app.pop_screen()
        await pilot.pause()

        modal = _modal(suspended_draft=snapshot)
        await _open(app, pilot, modal)
        assert app.is_running
        assert modal.query_one(f"#{control_id}", Select).is_blank()

@pytest.mark.parametrize("value", [None, True, False])
def test_snapshot_carries_each_streaming_state_and_refuses_a_malformed_one(
    value: bool | None,
) -> None:
    snapshot = ConsoleSettingsDraftSnapshot(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        context_policy_overrides=ConsoleContextPolicyOverrides(),
        raw_values={"console-settings-streaming": value},
        provider_model_drafts={},
        provider_base_url_drafts={},
        active_view="model",
        scroll_anchor=0,
        focus_control_id=None,
        disclosure_state={"advanced_generation": False, "connection_details": False},
    )
    mapping = snapshot.to_mapping()
    restored = ConsoleSettingsDraftSnapshot.from_mapping(mapping)
    assert restored is not None
    assert restored.raw_values["console-settings-streaming"] is value
    for malformed in ("", "inherit", "true", 0, 1):
        mapping["raw_values"] = {"console-settings-streaming": malformed}
        assert ConsoleSettingsDraftSnapshot.from_mapping(mapping) is None


@pytest.mark.asyncio
@private_profile_test
async def test_configure_credential_with_inherit_streaming_keeps_the_app_up(
    request, monkeypatch
):
    """TASK-33003.10 repro, real router: a llama.cpp chat switched to OpenAI
    (no key) leaves Streaming at Inherit; Configure credential -> Settings ->
    Return must not exit the app, and the reopened modal shows Inherit."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(
        settings_screen_module,
        "persist_provider_settings_atomic",
        lambda *_args, **_kwargs: ConfigMutationResult(True, True, None),
    )
    app = _build_production_app(configured_default="chat")
    app.chat_api_provider_value = "llama_cpp"
    app.chat_api_model_value = "model-a"
    app.app_config["chat_defaults"] = {"provider": "llama_cpp", "model": "model-a"}
    app.app_config["api_settings"] = {
        "llama_cpp": {"api_url": "http://127.0.0.1:9099"},
        "openai": {},
    }
    app.providers_models = {"llama_cpp": ["model-a"], "openai": ["gpt-5"]}

    async with app.run_test(size=(211, 44)) as pilot:
        console = None
        for _ in range(200):
            console = app._navigation_outgoing_screen()
            if isinstance(console, ChatScreen):
                break
            await pilot.pause(0.05)
        assert isinstance(console, ChatScreen)
        await _wait_for_selector(
            console, pilot, "#console-native-composer", timeout=10.0
        )
        store = console._ensure_console_chat_store()
        session = store.ensure_session()
        store.replace_session_settings(
            session.id,
            ConsoleSessionSettings(provider="llama_cpp", model="model-a"),
        )
        assert await console._open_console_settings() is True
        for _ in range(80):
            if isinstance(app.screen, ConsoleSettingsModal):
                break
            await pilot.pause(0.05)
        first = app.screen
        assert isinstance(first, ConsoleSettingsModal)
        await _settle(pilot, first)
        first.query_one("#console-settings-provider", Select).value = "openai"
        for _ in range(20):
            await pilot.pause()
            if first._active_provider == "openai":
                break
        await pilot.pause()
        # The provider switch alone lands on Inherit (the task's repro).
        assert first._streaming_draft is None
        assert _streaming_label(first) == "Inherit"
        await pilot.click("#console-settings-configure-credential")

        settings = None
        for _ in range(200):
            if isinstance(app.screen, SettingsScreen):
                settings = app.screen
                break
            await pilot.pause(0.05)
        assert settings is not None, "Configure credential did not reach Settings"
        await _wait_for_selector(
            settings, pilot, "#settings-provider-api-key", timeout=10.0
        )
        settings.query_one("#settings-provider-api-key", Input).value = (
            "DUMMY-ROUND-TRIP-KEY"
        )
        await pilot.pause()
        settings.action_settings_save_category(allow_text_entry_focus=True)
        for _ in range(80):
            if settings.query_one("#settings-provider-return", Button).has_focus:
                break
            await pilot.pause(0.05)
        settings.query_one("#settings-provider-return", Button).press()

        returned = None
        for _ in range(240):
            top = app.screen_stack[-1]
            if isinstance(top, ConsoleSettingsModal) and top is not first:
                returned = top
                break
            await pilot.pause(0.05)
        assert returned is not None
        await _settle(pilot, returned)
        for _ in range(4):
            await pilot.pause()
        assert app.is_running
        assert returned._active_provider == "openai"
        assert returned._streaming_draft is None
        assert _streaming_label(returned) == "Inherit"


@pytest.mark.asyncio
async def test_configure_credential_keeps_the_draft_when_the_snapshot_refuses_it(
    monkeypatch,
):
    """Guard (TASK-33003.10): any snapshot ValueError degrades to a notice;
    the modal and its edits stay, and the app keeps running."""

    def refuse(_modal: ConsoleSettingsModal) -> ConsoleSettingsDraftSnapshot:
        raise ValueError("raw modal values are invalid")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(ConsoleSettingsModal, "capture_suspended_draft", refuse)
    app = _GuardHarness()
    app.app_config = {"api_settings": {"openai": {}}}
    modal = ConsoleSettingsModal(
        settings=ConsoleSessionSettings(provider="openai", model="gpt-5"),
        app_config={"api_settings": {"openai": {}}},
        providers_models={"openai": ["gpt-5"]},
        context_estimate=ConsoleSettingsContextEstimate(10, 4096, "10 / 4k"),
        can_save=True,
    )
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        await _edit(pilot, modal, "#console-settings-temperature", "0.9")
        await pilot.click("#console-settings-configure-credential")
        await pilot.pause()
        await pilot.pause()

        assert app.is_running
        assert app.results == []
        assert app.screen is modal
        assert modal.query_one("#console-settings-temperature", Input).value == "0.9"
        assert any(
            "Settings could not open" in notice.message
            for notice in app._notifications
        )


def test_every_snapshot_focus_target_survives_the_credential_handoff() -> None:
    """TASK-33003.10: the return intent refused 13 focus targets the snapshot
    accepts (the provider picker among them), so Configure credential exited
    the app whenever one of them held the last focus."""
    from tldw_chatbook.UI.Navigation.conversation_settings_navigation import (
        ConversationSettingsReturnIntent,
    )

    for control_id in sorted(settings_modal_module._SNAPSHOT_FOCUS_CONTROL_IDS):
        intent = ConversationSettingsReturnIntent(
            session_id="session-1",
            settings_revision=1,
            active_view="model",
            focus_control_id=control_id,
        )
        assert intent.focus_control_id == control_id


@pytest.mark.parametrize(
    ("selector", "label"),
    [
        ("#console-settings-temperature", "Temperature"),
        ("#console-settings-top-p", "Top P"),
    ],
)
@pytest.mark.asyncio
async def test_clearing_a_required_sampling_field_asks_before_closing(
    selector: str, label: str
) -> None:
    """Qodo #2937: a blank Temperature or Top P builds the draft from the
    opened value, so a cleared field read as unedited and Esc closed the
    modal. Apply refuses the blank; the close guard must ask."""
    app = _GuardHarness()
    modal = _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        await _edit(pilot, modal, selector, "")
        assert _text(modal, "#console-settings-esc-hint") == esc_hint_copy(1)

        await _gesture(pilot, "escape")

        assert app.screen is modal
        assert app.results == []
        assert f": {label}." in _text(modal, "#console-settings-close-message")


def _inherit_streaming_modal() -> ConsoleSettingsModal:
    """A modal opened with Streaming at Inherit over an inherited On."""
    settings = ConsoleSessionSettings(
        provider="llama_cpp", model="model-a", temperature=0.7
    )
    draft = ConsoleSettingsModal._initial_full_draft(settings)
    fields = tuple(
        replace(field, profile_override=None) if field.name == "streaming" else field
        for field in draft.field_drafts
    )
    return _modal(initial_draft=replace(draft, field_drafts=fields))


@pytest.mark.parametrize(
    ("opened_at_inherit", "presses", "unsaved"),
    [
        (True, 1, True),  # Inherit -> On, the inherited default On
        (False, 2, True),  # On -> Off -> Inherit, whose default is On
        (True, 3, False),  # all the way round, back to Inherit
    ],
    ids=["inherit-to-on", "on-to-inherit", "round-trip"],
)
@pytest.mark.asyncio
async def test_streaming_inherit_changes_count_as_edits(
    opened_at_inherit: bool, presses: int, unsaved: bool
) -> None:
    """Qodo #2937: the guard compared the effective bool, so a change between
    Inherit and the value Inherit resolves to closed without asking and lost
    the override. Inherit is a draft value of its own (TASK-33003.10)."""
    app = _GuardHarness()
    modal = _inherit_streaming_modal() if opened_at_inherit else _modal()
    async with app.run_test(size=(211, 44)) as pilot:
        await _open(app, pilot, modal)
        assert modal._streaming_draft is (None if opened_at_inherit else True)
        toggle = modal.query_one("#console-settings-streaming", Button)
        toggle.focus()
        for _ in range(presses):
            toggle.press()
            await pilot.pause()
        await pilot.pause()
        # The effective value never moved; only the Inherit/On/Off draft did.
        assert modal._effective_streaming_value() is True
        assert _text(modal, "#console-settings-esc-hint") == esc_hint_copy(
            int(unsaved)
        )

        await _gesture(pilot, "escape")

        if not unsaved:
            assert app.results == [None]
            return
        assert app.screen is modal
        assert app.results == []
        assert ": Streaming." in _text(modal, "#console-settings-close-message")
