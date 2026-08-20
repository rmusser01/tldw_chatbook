"""Next Send tab of the Conversation Inspector (task-10 ported this pane's
behavior wholesale from the retired standalone context modal; this file was
that modal's own dedicated test file and is re-targeted here rather than
deleted -- see the module's own docstring for what's ported verbatim).

Construction/queries now go through ``ConsoleConversationInspector`` (with
``initial_tab=TAB_NEXT_SEND`` so the pane under test is the visible,
clickable tab) instead of the standalone modal; only DOM ids that used to
identify the old modal's own FRAME changed to the pane's ids. Every
behavioral pin from the original file is kept: the "No conversation
context" empty-state copy prefix, the save-blocked tooltip, the raw-JSON
toggle, the 1 MiB size threshold (covered elsewhere, in
``test_console_conversation_inspector.py``), and the in-progress warning.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, Collapsible, Label, Static, TextArea

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
    ProjectInstructionPreview,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleProjectInstructionSourceRow,
    build_console_project_instruction_state,
)
from tldw_chatbook.Chat.console_project_instructions import (
    EPHEMERAL_ORIGIN_KEY,
    ProjectInstructionControlState,
)
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRowTotals
from tldw_chatbook.Widgets.Console import console_conversation_inspector
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    CLOSE_BUTTON_ID,
    TAB_NEXT_SEND,
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.Console.console_project_instructions import (
    ConsoleProjectInstructionContextPanel,
)


SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
        ConsoleChatMessage(role=ConsoleMessageRole.ASSISTANT, content="Hi"),
    ],
    next_send_payload={
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Hello"}],
    },
)

EMPTY_SNAPSHOT = ConsoleContextSnapshot(current_messages=[], next_send_payload={})

PROJECT_SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[],
    next_send_payload={"messages": []},
    project_instruction_preview=ProjectInstructionPreview(
        relative_source="nested/AGENTS.md",
        scope="nested",
        byte_count=33,
        outcomes=("active",),
        warning_codes=(),
        next_send_payload={"messages": []},
    ),
)


async def _project_snapshot_factory() -> ConsoleContextSnapshot:
    return PROJECT_SNAPSHOT


async def _snapshot_factory() -> ConsoleContextSnapshot:
    return SNAPSHOT


async def _empty_factory() -> ConsoleContextSnapshot:
    return EMPTY_SNAPSHOT


async def _empty_exchanges_loader(
    _native_message_id: str,
) -> list[tuple[object, bool]]:
    return []


def _inspector(snapshot_factory=_snapshot_factory, **overrides: object) -> ConsoleConversationInspector:
    """Build a ``ConsoleConversationInspector`` scoped to the Next Send tab.

    Mirrors the retired modal's own constructor surface (``snapshot_
    factory`` plus ``token_estimate``/``estimate_factory``/``in_progress``/
    ``ephemeral``) -- the Costs-tab-only kwargs (``rows``/``totals``/
    ``turns``/``exchanges_loader``) are filled with empty stand-ins since
    this file only exercises the Next Send pane.
    """
    kwargs: dict[str, object] = dict(
        rows=[],
        totals=ConsoleCostRowTotals(0, 0.0, False, 0),
        turns=[],
        exchanges_loader=_empty_exchanges_loader,
        snapshot_factory=snapshot_factory,
        initial_tab=TAB_NEXT_SEND,
    )
    kwargs.update(overrides)
    return ConsoleConversationInspector(**kwargs)


class ModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(_inspector(token_estimate=42))


@pytest.mark.asyncio
async def test_context_modal_renders_tabs():
    app = ModalHarness()

    async with app.run_test(size=(120, 44)) as _pilot:
        modal = app.screen
        header = modal.query_one("#console-inspector-next-send-header", Static)
        header_text = str(header.renderable)
        assert "Chat Context" in header_text
        assert "42 tokens" in header_text

        current_container = modal.query_one(
            "#console-inspector-next-send-current-body", Vertical
        )
        text_areas = current_container.query(TextArea)
        assert any("Hello" in ta.text for ta in text_areas)

        next_container = modal.query_one(
            "#console-inspector-next-send-payload-body", Vertical
        )
        labels = list(next_container.query(Label))
        assert any("gpt-4" in str(label.renderable) for label in labels)


@pytest.mark.asyncio
async def test_context_modal_shows_metadata_only_project_instruction_section():
    state = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo [untrusted]",
        locator_matches=True,
        sources=(
            ConsoleProjectInstructionSourceRow(
                relative_source="AGENTS.md",
                scope=".",
                byte_count=12,
                outcome="active",
            ),
            ConsoleProjectInstructionSourceRow(
                relative_source="nested/AGENTS.md",
                scope="nested",
                byte_count=33,
                outcome="active",
            ),
        ),
    )
    app = ActionHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                _project_snapshot_factory,
                project_instruction_state=state,
            )
        )
        await pilot.pause()
        panel = app.screen.query_one("#console-context-project-instructions")
        text = " ".join(str(item.renderable) for item in panel.query(Static))
        assert "Repo [untrusted]" in text
        assert "AGENTS.md" in text
        assert "12 bytes" in text
        assert "nested/AGENTS.md" in text
        assert "33 bytes" in text
        assert "AUTOMATIC_BODY_ONLY_IN_EXPLICIT_PREVIEW" not in text


@pytest.mark.asyncio
async def test_context_modal_empty_state():
    app = ModalHarness()
    app._push_empty = lambda: app.push_screen(_inspector(_empty_factory))

    async with app.run_test(size=(120, 44)) as pilot:
        app._push_empty()
        await pilot.pause()
        modal = app.screen
        current_container = modal.query_one(
            "#console-inspector-next-send-current-body", Vertical
        )
        labels = list(current_container.query(Label))
        assert any(
            "No conversation context" in str(label.renderable) for label in labels
        )


@pytest.mark.asyncio
async def test_context_modal_in_progress_warning():
    app = ModalHarness()
    app._push_in_progress = lambda: app.push_screen(
        _inspector(_snapshot_factory, in_progress=True)
    )

    async with app.run_test(size=(120, 44)) as pilot:
        app._push_in_progress()
        await pilot.pause()
        modal = app.screen
        warning = modal.query_one("#console-inspector-next-send-warning", Static)
        assert "in progress" in str(warning.renderable)
        refresh_button = modal.query_one(
            "#console-inspector-next-send-refresh", Button
        )
        assert refresh_button.disabled
        assert app.focused is modal.query_one("#console-context-close", Button)


class ActionHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")


class RecoveryHarness(ActionHarness):
    def __init__(self) -> None:
        super().__init__()
        self.recoveries = []

    def on_console_project_instruction_context_panel_recovery_requested(
        self,
        event: ConsoleProjectInstructionContextPanel.RecoveryRequested,
    ) -> None:
        self.recoveries.append((event.session_id, event.action))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
async def test_context_modal_stays_within_supported_viewports(size):
    app = ActionHarness()
    async with app.run_test(size=size) as pilot:
        app.push_screen(ConsoleContextModal(_snapshot_factory))
        await pilot.pause()
        modal = app.screen.query_one("#console-context-modal", Vertical)
        actions = app.screen.query_one("#console-context-actions", Horizontal)
        assert modal.region.x >= 0
        assert modal.region.y >= 0
        assert modal.region.right <= size[0]
        assert modal.region.bottom <= size[1]
        assert actions.region.width > 0
        assert actions.region.bottom <= modal.region.bottom
        assert all(control.region.right <= modal.region.right for control in actions.children)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40)])
async def test_project_warning_recovery_and_metadata_fit_supported_viewports(size):
    state = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo [literal]",
        locator_matches=False,
        sources=(
            ConsoleProjectInstructionSourceRow(
                relative_source="pkg/AGENTS.override.md",
                scope="pkg",
                byte_count=33,
                outcome="omitted_token_budget",
                warning_code="omitted_token_budget",
            ),
        ),
        warning_codes=("binding_retargeted",),
    )
    app = RecoveryHarness()
    async with app.run_test(size=size) as pilot:
        app.push_screen(
            ConsoleContextModal(
                _empty_factory,
                project_instruction_state=state,
                project_instruction_session_id="captured-session",
            )
        )
        await pilot.pause()
        modal = app.screen.query_one("#console-context-modal", Vertical)
        panel = app.screen.query_one(
            "#console-context-project-instructions",
            ConsoleProjectInstructionContextPanel,
        )
        rendered = " ".join(str(item.renderable) for item in panel.query(Static))
        assert "Repo [literal]" in rendered
        assert "Precedence: override" in rendered
        assert "scope pkg" in rendered
        assert "omitted_token_budget" in rendered
        assert modal.region.right <= size[0]
        assert modal.region.bottom <= size[1]
        choose = panel.query_one("#console-project-instruction-choose", Button)
        assert choose.region.bottom <= modal.region.bottom
        assert app.focused is choose
        await pilot.click("#console-project-instruction-choose")
        await pilot.pause()
        assert app.recoveries == [("captured-session", "choose")]


@pytest.mark.asyncio
async def test_project_off_state_focuses_enable_and_context_escape_closes():
    state = build_console_project_instruction_state(
        ProjectInstructionControlState.legacy_disabled()
    )
    app = RecoveryHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                _empty_factory,
                project_instruction_state=state,
                project_instruction_session_id="captured-session",
            )
        )
        await pilot.pause()
        enable = app.screen.query_one("#console-project-instruction-enable", Button)
        assert app.focused is enable
        await pilot.press("escape")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleContextModal)


@pytest.mark.asyncio
async def test_project_recovery_uses_captured_session_and_replaces_panel_state():
    calls = []
    enabled = build_console_project_instruction_state(
        ProjectInstructionControlState.new_session()
    )

    async def recover(session_id, action):
        calls.append((session_id, action))
        return build_console_project_instruction_state(
            ProjectInstructionControlState.legacy_disabled()
        )

    app = RecoveryHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                _empty_factory,
                project_instruction_state=enabled,
                project_instruction_session_id="captured-session",
                project_instruction_recovery=recover,
            )
        )
        await pilot.pause()
        await pilot.click("#console-project-instruction-disable")
        await pilot.pause()

        panel = app.screen.query_one(
            "#console-context-project-instructions",
            ConsoleProjectInstructionContextPanel,
        )
        assert calls == [("captured-session", "disable")]
        assert "State: Off" in " ".join(
            str(item.renderable) for item in panel.query(Static)
        )


@pytest.mark.asyncio
async def test_context_modal_toggle_raw_json():
    app = ActionHarness()
    expected_raw = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector())
        await pilot.pause()

        await pilot.click("#console-inspector-next-send-raw")
        await pilot.pause()

        modal = app.screen
        next_container = modal.query_one(
            "#console-inspector-next-send-payload-body", Vertical
        )
        text_areas = list(next_container.query(TextArea))
        assert any(ta.text == expected_raw for ta in text_areas)


@pytest.mark.asyncio
async def test_context_modal_refresh_invokes_factory():
    calls = 0

    async def counting_factory() -> ConsoleContextSnapshot:
        nonlocal calls
        calls += 1
        return SNAPSHOT

    app = ActionHarness()

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector(counting_factory))
        await pilot.pause()
        assert calls == 1

        await pilot.click("#console-inspector-next-send-refresh")
        await pilot.pause()
        assert calls == 2


@pytest.mark.asyncio
async def test_context_modal_refreshes_project_metadata_in_place():
    display = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo",
        locator_matches=True,
    )
    snapshots = [
        ConsoleContextSnapshot(current_messages=[], next_send_payload={}),
        PROJECT_SNAPSHOT,
    ]

    async def changing_factory() -> ConsoleContextSnapshot:
        return snapshots.pop(0)

    app = ActionHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                changing_factory,
                project_instruction_state=display,
            )
        )
        await pilot.pause()
        modal = app.screen
        await pilot.click("#console-context-refresh")
        await pilot.pause()
        assert app.screen is modal
        assert modal.snapshot.project_instruction_preview is not None
        panel = modal.query_one("#console-context-project-instructions")
        text = " ".join(str(item.renderable) for item in panel.query(Static))
        assert "nested/AGENTS.md" in text
        assert "33 bytes" in text


@pytest.mark.asyncio
async def test_context_modal_none_preview_replaces_stale_loaded_state():
    loaded = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo",
        locator_matches=True,
        sources=(
            ConsoleProjectInstructionSourceRow(
                relative_source="AGENTS.md",
                scope=".",
                byte_count=33,
                outcome="active",
            ),
        ),
    )
    disabled = build_console_project_instruction_state(
        ProjectInstructionControlState.legacy_disabled()
    )
    snapshots = [PROJECT_SNAPSHOT, EMPTY_SNAPSHOT]
    states = [loaded, disabled]

    async def changing_factory() -> ConsoleContextSnapshot:
        return snapshots.pop(0)

    async def changing_state_factory():
        return states.pop(0)

    app = ActionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                changing_factory,
                project_instruction_state=loaded,
                project_instruction_state_factory=changing_state_factory,
            )
        )
        await pilot.pause()
        modal = app.screen
        panel = modal.query_one("#console-context-project-instructions")
        assert "AGENTS.md" in " ".join(
            str(item.renderable) for item in panel.query(Static)
        )

        await pilot.click("#console-context-refresh")
        await pilot.pause()

        assert app.screen is modal
        assert modal.snapshot.project_instruction_preview is None
        text = " ".join(str(item.renderable) for item in panel.query(Static))
        assert "State: Off" in text
        assert "AGENTS.md" not in text


@pytest.mark.asyncio
async def test_context_modal_authority_warning_suppresses_stale_preview_rows():
    loaded = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo",
        locator_matches=True,
    )
    warning = build_console_project_instruction_state(
        ProjectInstructionControlState(True, "binding", "f" * 64, None),
        binding_label="Repo moved",
        locator_matches=False,
        warning_codes=("binding_retargeted",),
    )

    async def state_factory():
        return warning

    app = ActionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(
            ConsoleContextModal(
                _project_snapshot_factory,
                project_instruction_state=loaded,
                project_instruction_state_factory=state_factory,
            )
        )
        await pilot.pause()

        panel = app.screen.query_one("#console-context-project-instructions")
        text = " ".join(str(item.renderable) for item in panel.query(Static))
        assert "State: Warning" in text
        assert "binding_retargeted" in text
        assert "nested/AGENTS.md" not in text


@pytest.mark.asyncio
async def test_context_modal_close_dismisses():
    app = ActionHarness()

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector())
        await pilot.pause()
        assert isinstance(app.screen, ConsoleConversationInspector)

        await pilot.click(f"#{CLOSE_BUTTON_ID}")
        await pilot.pause()
        assert not isinstance(app.screen, ConsoleConversationInspector)


@pytest.mark.parametrize("source", ["close", "backdrop"])
@pytest.mark.asyncio
async def test_context_modal_close_and_backdrop_return_none(source: str):
    app = ActionHarness()
    results: list[object] = []

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector(), results.append)
        await pilot.pause()

        if source == "close":
            await pilot.click(f"#{CLOSE_BUTTON_ID}")
        else:
            await pilot.click(offset=(0, 0))
        await pilot.pause()

    assert results == [None]


@pytest.mark.asyncio
async def test_context_modal_copy_json(monkeypatch):
    app = ActionHarness()
    expected_text = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)
    fake_copy = types.SimpleNamespace(copy=Mock())
    monkeypatch.setitem(sys.modules, "pyperclip", fake_copy)

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector())
        await pilot.pause()

        await pilot.click("#console-inspector-next-send-copy")
        await pilot.pause()

        fake_copy.copy.assert_called_once_with(expected_text)


@pytest.mark.asyncio
async def test_context_modal_copy_omits_automatic_project_instruction_body(
    monkeypatch,
):
    sentinel = "AUTOMATIC_BODY_MUST_NOT_EXPORT"
    snapshot = ConsoleContextSnapshot(
        current_messages=[],
        next_send_payload={
            "messages": [
                {"role": "user", "content": "ordinary"},
                {
                    "role": "user",
                    "content": sentinel,
                    EPHEMERAL_ORIGIN_KEY: "project_instructions",
                },
            ]
        },
    )

    async def factory() -> ConsoleContextSnapshot:
        return snapshot

    fake_copy = types.SimpleNamespace(copy=Mock())
    monkeypatch.setitem(sys.modules, "pyperclip", fake_copy)
    app = ActionHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        app.push_screen(ConsoleContextModal(factory))
        await pilot.pause()
        assert sentinel in app.screen._format_next_send_text()
        await pilot.click("#console-context-copy")
        await pilot.pause()

    exported = fake_copy.copy.call_args.args[0]
    assert sentinel not in exported
    assert "ordinary" in exported


@pytest.mark.asyncio
async def test_context_modal_save_to_file(tmp_path, monkeypatch):
    app = ActionHarness()
    expected_text = json.dumps(SNAPSHOT.next_send_payload, indent=2, default=str)

    class FakePath:
        """Redirect filesystem operations under ``tmp_path`` for hermetic tests."""

        def __init__(self, *parts: str | Path) -> None:
            self._path = tmp_path.joinpath(*parts)

        @classmethod
        def home(cls):
            return cls(tmp_path)

        def __truediv__(self, other: str) -> "FakePath":
            return FakePath(self._path, other)

        def __getattr__(self, name: str):
            return getattr(self._path, name)

    monkeypatch.setattr(
        console_conversation_inspector,
        "Path",
        FakePath,
    )

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector())
        await pilot.pause()

        await pilot.click("#console-inspector-next-send-save")
        await pilot.pause()

        saved_files = list((tmp_path / "Downloads").glob("chatbook_context_*.json"))
        assert len(saved_files) == 1
        assert saved_files[0].read_text(encoding="utf-8") == expected_text


@pytest.mark.asyncio
async def test_context_modal_save_omits_automatic_project_instruction_body(
    tmp_path, monkeypatch
):
    sentinel = "AUTOMATIC_BODY_MUST_NOT_SAVE"
    snapshot = ConsoleContextSnapshot(
        current_messages=[],
        next_send_payload={
            "messages": [
                {"role": "user", "content": "ordinary"},
                {
                    "role": "user",
                    "content": sentinel,
                    EPHEMERAL_ORIGIN_KEY: "project_instructions",
                },
            ]
        },
    )

    async def factory() -> ConsoleContextSnapshot:
        return snapshot

    class FakePath:
        def __init__(self, *parts: str | Path) -> None:
            self._path = tmp_path.joinpath(*parts)

        @classmethod
        def home(cls):
            return cls(tmp_path)

        def __truediv__(self, other: str) -> "FakePath":
            return FakePath(self._path, other)

        def __getattr__(self, name: str):
            return getattr(self._path, name)

    monkeypatch.setattr(console_context_modal, "Path", FakePath)
    app = ActionHarness()
    async with app.run_test(size=(80, 24)) as pilot:
        app.push_screen(ConsoleContextModal(factory))
        await pilot.pause()
        assert sentinel in app.screen._format_next_send_text()
        await pilot.click("#console-context-save")
        await pilot.pause()

    saved = next((tmp_path / "Downloads").glob("*.json")).read_text(encoding="utf-8")
    assert sentinel not in saved
    assert "ordinary" in saved


@pytest.mark.asyncio
async def test_context_modal_save_to_file_failure(monkeypatch):
    app = ActionHarness()

    class FailingPath:
        """Path stand-in whose ``write_text`` always raises ``OSError``."""

        @classmethod
        def home(cls):
            return cls()

        def __truediv__(self, other: str) -> "FailingPath":
            return self

        def mkdir(self, **kwargs: object) -> None:
            return None

        def write_text(self, *args: object, **kwargs: object) -> None:
            raise OSError("disk full")

    monkeypatch.setattr(
        console_conversation_inspector,
        "Path",
        FailingPath,
    )

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector())
        await pilot.pause()

        # Should not crash; notification severity is checked best-effort.
        await pilot.click("#console-inspector-next-send-save")
        await pilot.pause()


PREFILL_SNAPSHOT = ConsoleContextSnapshot(
    current_messages=[
        ConsoleChatMessage(role=ConsoleMessageRole.USER, content="Hello"),
    ],
    next_send_payload={
        "model": "gpt-4",
        "messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Sure thing:"},
        ],
        "response_prefill": {
            "source": "one-shot",
            "text": "Sure thing:",
            "agent_loop_bypassed": True,
        },
    },
)


async def _prefill_factory() -> ConsoleContextSnapshot:
    return PREFILL_SNAPSHOT


class PrefillModalHarness(App):
    def compose(self) -> ComposeResult:
        yield Static("background")

    def on_mount(self) -> None:
        self.push_screen(_inspector(_prefill_factory, token_estimate=7))


@pytest.mark.asyncio
async def test_context_modal_renders_response_prefill_section():
    """task-401: an armed prefill renders as its own Next Send section with
    the agent-loop-bypass note; absent entirely when the key is missing."""
    app = PrefillModalHarness()

    async with app.run_test(size=(120, 44)) as _pilot:
        modal = app.screen
        next_container = modal.query_one(
            "#console-inspector-next-send-payload-body", Vertical
        )
        collapsibles = list(next_container.query(Collapsible))
        titles = [c.title for c in collapsibles]
        assert "Response Prefill" in titles
        labels = [str(label.renderable) for label in next_container.query(Label)]
        assert any("agent" in text and "skipped" in text for text in labels)
        text_areas = [ta.text for ta in next_container.query(TextArea)]
        assert any("one-shot" in text for text in text_areas)


@pytest.mark.asyncio
async def test_context_modal_no_prefill_section_without_key():
    app = ModalHarness()

    async with app.run_test(size=(120, 44)) as _pilot:
        modal = app.screen
        next_container = modal.query_one(
            "#console-inspector-next-send-payload-body", Vertical
        )
        titles = [c.title for c in next_container.query(Collapsible)]
        assert "Response Prefill" not in titles


@pytest.mark.asyncio
async def test_context_modal_save_button_is_disabled_with_a_reason_when_ephemeral():
    """Save Context writes a JSON file -- blocked when temporary, and still
    enabled otherwise (the control)."""
    from tldw_chatbook.Chat.console_ephemeral import blocked_reason

    app = ActionHarness()

    async with app.run_test(size=(120, 44)) as pilot:
        app.push_screen(_inspector(ephemeral=True))
        await pilot.pause()

        save_button = app.screen.query_one(
            "#console-inspector-next-send-save", Button
        )
        assert save_button.disabled is True
        assert save_button.tooltip == blocked_reason("save-context", ephemeral=True)

        await app.pop_screen()
        await pilot.pause()

        app.push_screen(_inspector(ephemeral=False))
        await pilot.pause()
        normal_button = app.screen.query_one(
            "#console-inspector-next-send-save", Button
        )
        assert normal_button.disabled is False


@pytest.mark.asyncio
async def test_context_modal_empty_state_compacts_pane_and_guides():
    """LY-13 (TASK-2154.23): the empty viewer sizes its PANE to content and
    guides -- ported from the old modal's own top-level-frame compaction
    (see ``ConsoleConversationInspector.DEFAULT_CSS``'s Next Send comment
    for why this now scopes to the pane rather than the shared modal
    frame: the frame is shared with the Costs/Exchange tabs and stays a
    fixed size across all three)."""
    app = ModalHarness()
    app._push_empty = lambda: app.push_screen(_inspector(_empty_factory))

    async with app.run_test(size=(120, 44)) as pilot:
        app._push_empty()
        await pilot.pause()
        modal_screen = app.screen
        current_container = modal_screen.query_one(
            "#console-inspector-next-send-current-body", Vertical
        )
        labels = [str(label.renderable) for label in current_container.query(Label)]
        guidance = next(
            (text for text in labels if "No conversation context" in text), ""
        )
        assert "Next Send" in guidance
        pane = modal_screen.query_one("#console-inspector-next-send-pane", Vertical)
        assert pane.has_class("context-empty")


@pytest.mark.asyncio
async def test_context_modal_populated_state_keeps_full_height():
    """The compact class is empty-state-only; populated keeps the room."""
    app = ModalHarness()  # pushes the populated snapshot on mount

    async with app.run_test(size=(120, 44)) as pilot:
        await pilot.pause()
        modal_screen = app.screen
        pane = modal_screen.query_one("#console-inspector-next-send-pane", Vertical)
        assert not pane.has_class("context-empty")
