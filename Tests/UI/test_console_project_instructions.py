"""Console project-instruction display, chooser, notice, and preview contracts."""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Footer, Static

from tldw_chatbook.Agents.agent_service import AgentService
from tldw_chatbook.Agents.project_instruction_resolver import (
    InstructionOutcome,
    InstructionSource,
    StartupInstructionCandidate,
)
from tldw_chatbook.Chat.console_chat_controller import (
    ConsoleChatController,
    ProjectInstructionDispatchNotice,
    build_project_instruction_preview,
)
import tldw_chatbook.Chat.console_chat_controller as controller_module
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.UI.Console_Modules.session import ConsoleSessionController
from tldw_chatbook.Chat.console_project_instructions import (
    ProjectInstructionControlState,
)


def _ui_module():
    return importlib.import_module(
        "tldw_chatbook.Widgets.Console.console_project_instructions"
    )


def _candidate(tmp_path: Path) -> StartupInstructionCandidate:
    body = "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW"
    source = InstructionSource(
        canonical_path=tmp_path / "AGENTS.md",
        relative_path="AGENTS.md",
        scope=".",
        kind="standard",
        body=body,
        byte_count=len(body.encode()),
        digest="d" * 64,
    )
    return StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=source,
        outcomes=(),
    )


def test_display_states_are_metadata_only():
    display = importlib.import_module("tldw_chatbook.Chat.console_display_state")
    build = display.build_console_project_instruction_state
    row = display.ConsoleProjectInstructionSourceRow(
        relative_source="AGENTS.md",
        scope=".",
        byte_count=12,
        outcome="active",
    )
    cases = [
        (
            ProjectInstructionControlState.legacy_disabled(),
            {},
            "Off",
        ),
        (
            ProjectInstructionControlState.new_session(),
            {},
            "Choose folder",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {"binding_label": "Repo", "locator_matches": True},
            "None",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {
                "binding_label": "Repo",
                "locator_matches": True,
                "sources": (row,),
            },
            "1 loaded",
        ),
        (
            ProjectInstructionControlState(True, "b", "f" * 64, None),
            {
                "binding_label": "Repo",
                "locator_matches": False,
                "warning_codes": ("binding_retargeted",),
            },
            "Warning",
        ),
    ]
    for control, kwargs, expected in cases:
        state = build(control, **kwargs)
        assert state.status == expected
        assert "body" not in state.__dataclass_fields__
        assert all("body" not in item.__dataclass_fields__ for item in state.sources)


def test_preview_uses_copies_and_is_repeatable_without_live_side_effects(tmp_path):
    candidate = _candidate(tmp_path)
    base = [{"role": "user", "content": "question"}]
    original = copy.deepcopy(base)
    calls: list[list[dict]] = []

    def request_builder(messages: list[dict]) -> dict:
        calls.append(copy.deepcopy(messages))
        return {"model": "gpt-test", "messages": messages}

    first = build_project_instruction_preview(base, candidate, request_builder)
    second = build_project_instruction_preview(base, candidate, request_builder)

    assert base == original
    assert first == second
    assert len(calls) == 2
    assert calls[0] is not base
    assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" in str(first.next_send_payload)
    assert first.relative_source == "AGENTS.md"
    assert not hasattr(first, "ledger")


@pytest.mark.asyncio
async def test_controller_preview_does_not_advance_live_session_state(
    tmp_path, monkeypatch
):
    candidate = StartupInstructionCandidate(
        binding_id="binding-1",
        binding_root=tmp_path,
        locator_fingerprint="f" * 64,
        dispatch_started_wall_ns=1,
        source=None,
        outcomes=(InstructionOutcome("AGENTS.md", ".", "omitted_token_budget"),),
    )
    control = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="binding-1",
        working_folder_locator_fingerprint="f" * 64,
        project_instruction_notice_key="n" * 64,
    )
    store = ConsoleChatStore()
    session = store.create_session(project_instruction_state=control)
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    consent = Mock(side_effect=AssertionError("preview acknowledged notice"))
    controller = ConsoleChatController(
        store=store,
        provider_gateway=Mock(),
        agent_runtime_enabled=True,
        confirm_project_instruction_dispatch=consent,
    )
    controller.app = SimpleNamespace(workspace_registry_service=object())
    controller._project_instruction_candidates[session.id] = _candidate(tmp_path)
    controller._run_state_histories[session.id] = []
    selection = SimpleNamespace(
        binding=SimpleNamespace(binding_id="binding-1"),
        root=tmp_path,
        locator_fingerprint="f" * 64,
    )
    monkeypatch.setattr(
        controller_module,
        "resolve_project_instruction_binding",
        lambda _session, _registry: selection,
    )
    monkeypatch.setattr(
        controller_module.ProjectInstructionResolver,
        "resolve_startup",
        lambda _resolver, **_kwargs: candidate,
    )
    state_setter = Mock(side_effect=AssertionError("preview changed controls"))
    monkeypatch.setattr(store, "set_session_project_instruction_state", state_setter)
    token_admission = Mock(side_effect=AssertionError("preview spent token budget"))
    ledger_snapshot = Mock(side_effect=AssertionError("preview activated a source"))
    monkeypatch.setattr(
        AgentService, "safe_project_instruction_tokens", token_admission
    )
    monkeypatch.setattr(AgentService, "_freeze_startup_snapshot", ledger_snapshot)

    before_messages = copy.deepcopy(store.messages_for_session(session.id))
    before_control = copy.deepcopy(session.project_instruction_state)
    before_candidates = dict(controller._project_instruction_candidates)
    before_run_states = dict(controller._run_states)
    before_run_history = copy.deepcopy(controller._run_state_histories)
    payload = {
        "model": "gpt-test",
        "messages": [{"role": "user", "content": "question"}],
        "admission": {"warning": "omitted_token_budget"},
    }
    preview = await controller._build_project_instruction_preview_for_session(
        session.id, payload
    )

    assert preview is not None
    assert preview.outcomes == ("omitted_token_budget",)
    assert preview.warning_codes == ("omitted_token_budget",)
    assert preview.next_send_payload == payload
    assert store.messages_for_session(session.id) == before_messages
    assert session.project_instruction_state == before_control
    assert controller._project_instruction_candidates == before_candidates
    assert controller._run_states == before_run_states
    assert controller._run_state_histories == before_run_history
    consent.assert_not_called()
    state_setter.assert_not_called()
    token_admission.assert_not_called()
    ledger_snapshot.assert_not_called()


@pytest.mark.asyncio
async def test_context_preview_without_bound_textual_app_degrades_only_preview():
    store = ConsoleChatStore()
    session = store.create_session(
        project_instruction_state=ProjectInstructionControlState.new_session()
    )
    store.append_message(
        session.id,
        role=ConsoleMessageRole.USER,
        content="question",
    )
    controller = ConsoleChatController(store=store, provider_gateway=Mock())

    snapshot = await controller.build_context_snapshot(draft="next")

    assert "error" not in snapshot.next_send_payload
    assert snapshot.project_instruction_preview is None


@pytest.mark.asyncio
async def test_setup_callback_is_scoped_to_the_owning_session():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: []
    )
    push_screen_wait = AsyncMock(
        side_effect=AssertionError("mounted for closed session")
    )
    controller._screen = SimpleNamespace(
        app=SimpleNamespace(push_screen_wait=push_screen_wait)
    )

    result = await controller._select_project_instruction_binding(
        "closed-session", (), "binding_unavailable"
    )

    assert result == ("cancel", None)
    push_screen_wait.assert_not_called()


def test_notice_callback_fails_closed_when_main_loop_is_unavailable():
    controller = ConsoleSessionController.__new__(ConsoleSessionController)
    controller._current_chat_store_accessor = lambda: SimpleNamespace(
        sessions=lambda: [SimpleNamespace(id="session-a")]
    )
    controller.app_instance = SimpleNamespace(
        call_from_thread=Mock(side_effect=RuntimeError("loop closed"))
    )

    decision = controller._confirm_project_instruction_dispatch(
        SimpleNamespace(session_id="session-a")
    )

    assert decision == "cancel"


class _ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")


@pytest.mark.asyncio
async def test_setup_modal_selects_only_eligible_binding_and_has_true_footer():
    ui = _ui_module()
    options = (
        ui.ProjectInstructionBindingOption("ready", "Ready repo", True),
        ui.ProjectInstructionBindingOption(
            "stale", "Stale [repo]", False, "Folder is unavailable"
        ),
    )
    app = _ModalHarness()
    result = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(ui.ProjectInstructionSetupModal(options), result.append)
        await pilot.pause()
        modal = app.screen
        assert modal.query_one(Footer)
        stale = modal.query_one("#console-project-binding-1", Button)
        assert stale.disabled
        rendered = " ".join(str(item.renderable) for item in modal.query(Static))
        assert "Stale [repo]" in str(stale.label)
        assert "Folder is unavailable" in rendered
        await pilot.click("#console-project-binding-1")
        await pilot.pause()
        assert app.screen is modal
        await pilot.click("#console-project-binding-0")
        await pilot.pause()
    assert result[0].action == "select"
    assert result[0].binding_id == "ready"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "action"),
    (("d", "disable"), ("c", "cancel")),
)
async def test_setup_modal_disable_and_cancel_bindings_are_exact(key, action):
    ui = _ui_module()
    app = _ModalHarness()
    results = []
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ui.ProjectInstructionSetupModal(
                (ui.ProjectInstructionBindingOption("one", "Repo", True),)
            ),
            results.append,
        )
        await pilot.pause()
        await pilot.press(key)
        await pilot.pause()
    assert results == [ui.ProjectInstructionSetupResult(action)]


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
async def test_notice_modal_is_usable_at_supported_sizes(size):
    ui = _ui_module()
    notice = ProjectInstructionDispatchNotice(
        session_id="session-a",
        destination_label="OpenAI (https://api.example)",
        relative_source="AGENTS[1].md",
        scope=".",
        byte_count=42,
        outcomes=("omitted_token_budget",),
        warning_codes=("omitted_token_budget",),
    )
    app = _ModalHarness()
    decisions = []
    async with app.run_test(size=size) as pilot:
        app.push_screen(ui.ProjectInstructionNoticeModal(notice), decisions.append)
        await pilot.pause()
        modal = app.screen
        assert modal.query_one(Footer)
        assert modal.query_one("#console-project-notice-proceed", Button).region.width
        visible_copy = " ".join(
            str(widget.renderable) for widget in modal.query(Static)
        )
        assert "AGENTS[1].md" in visible_copy
        assert "[1]" in visible_copy
        assert "deeper AGENTS.md" in visible_copy
        assert "omitted_token_budget" in visible_copy
        await pilot.press("p")
        await pilot.pause()
    assert decisions == ["proceed"]


@pytest.mark.asyncio
async def test_notice_modal_cancel_and_disable_bindings_match_actions():
    ui = _ui_module()
    notice = ProjectInstructionDispatchNotice(
        session_id="session-a",
        destination_label="Provider",
        relative_source=None,
        scope=".",
        byte_count=0,
        outcomes=(),
        warning_codes=(),
    )
    for key, expected in (("c", "cancel"), ("d", "disable")):
        app = _ModalHarness()
        decisions = []
        async with app.run_test(size=(80, 24)) as pilot:
            app.push_screen(ui.ProjectInstructionNoticeModal(notice), decisions.append)
            await pilot.pause()
            await pilot.press(key)
            await pilot.pause()
        assert decisions == [expected]
