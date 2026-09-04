"""Console context rail colour grammar (TASK-31429).

Four meanings, no decoration: primary hue = what you are in, accent hue = a
value, status hues = state, muted = labels/help. Both new tokens reference
Textual's generated polarity-aware variables so every theme recolors the
rail without editor changes (mechanism note atop themes.py: theme-dict
``ds-*`` entries are inert, tcss references to generated names are not).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from textual.widgets import Static

from Tests.UI.test_console_agent_tool_row_css import (
    _STYLESHEETS,
    _css_block,
    _stylesheet_text,
)
from tldw_chatbook.Chat.console_chat_models import (
    CONSOLE_RUN_MARKER_GLYPHS,
    ConsoleRunMarker,
)

_CORE_VARIABLES = Path("tldw_chatbook/css/core/_variables.tcss")
_SOURCE = Path("tldw_chatbook/css/components/_agentic_terminal.tcss")

_ROW_MARKER_CLASSES = {
    marker: f"console-workspace-conversation-row-{marker.value}"
    for marker in ConsoleRunMarker
    if marker is not ConsoleRunMarker.NONE
}

# selector -> (property, value) the rail grammar pins in source AND in the
# generated sheet the running app loads.
_EXPECTED_DECLARATIONS = {
    ".console-workspace-status-value": ("color", "$ds-value-fg"),
    ".console-model-section-value": ("color", "$ds-value-fg"),
    "#console-active-workspace-value": ("color", "$ds-active-fg"),
    # id+class: the muted placeholder rule on the same Static is an id rule.
    "#console-workspace-selected-conversation.console-workspace-selected-conversation-active": (
        "color",
        "$ds-active-fg",
    ),
    ".console-workspace-conversation-row-selected": ("color", "$ds-active-fg"),
    "#console-model-section-recovery": ("color", "$ds-status-error-readable"),
    ".console-agent-section-status-running": ("color", "$ds-status-running"),
    ".console-agent-section-status-done": ("color", "$ds-status-ready"),
    ".console-agent-section-status-stuck": ("color", "$ds-status-warning"),
    ".console-agent-section-status-error": ("color", "$ds-status-error-readable"),
    ".console-agent-section-status-cancelled": ("color", "$ds-status-error-readable"),
    ".console-workspace-conversation-row-running": ("color", "$ds-status-running"),
    ".console-workspace-conversation-row-needs-approval": (
        "color",
        "$ds-status-approval-required",
    ),
    ".console-workspace-conversation-row-finished-ok": ("color", "$ds-status-ready"),
    ".console-workspace-conversation-row-finished-failed": (
        "color",
        "$ds-status-error-readable",
    ),
    ".console-workspace-conversation-row-subagent-unseen": ("color", "$ds-status-info"),
}


def _declaration(block: str, prop: str) -> str:
    match = re.search(rf"(?m)^\s*{re.escape(prop)}\s*:\s*([^;]+);", block)
    assert match, f"no `{prop}` declaration in block: {block!r}"
    return match.group(1).strip()


# --- tokens -----------------------------------------------------------------


def test_rail_tokens_reference_generated_theme_variables() -> None:
    text = _CORE_VARIABLES.read_text(encoding="utf-8")
    assert re.search(r"^\$ds-active-fg:\s*\$text-primary;", text, re.MULTILINE)
    assert re.search(r"^\$ds-value-fg:\s*\$text-accent;", text, re.MULTILINE)


# --- agent status line --------------------------------------------------------


@pytest.mark.parametrize(
    ("status_line", "expected"),
    [
        ("Agent: running · step 3", "running"),
        ("Agent: done", "done"),
        ("Agent: stuck", "stuck"),
        ("Agent: error", "error"),
        ("Agent: cancelled", "cancelled"),
        ("Sub-agent · running", "running"),
        ("Sub-agent · done", "done"),
        ("Agent: idle", ""),
        ("Agent: unavailable", ""),
        ("", ""),
    ],
)
def test_agent_status_state_maps_known_statuses(
    status_line: str, expected: str
) -> None:
    from tldw_chatbook.UI.Console_Modules.agent import console_agent_status_state

    assert console_agent_status_state(status_line) == expected


def test_apply_agent_status_state_swaps_one_state_class() -> None:
    from tldw_chatbook.UI.Console_Modules.agent import apply_console_agent_status_state

    status = Static("", classes="console-agent-section-line")
    apply_console_agent_status_state(status, "Agent: running · step 1")
    assert status.has_class("console-agent-section-status-running")

    apply_console_agent_status_state(status, "Agent: done")
    assert status.has_class("console-agent-section-status-done")
    assert not status.has_class("console-agent-section-status-running")

    apply_console_agent_status_state(status, "Agent: idle")
    assert not any(
        cls.startswith("console-agent-section-status-") for cls in status.classes
    ), "idle must leave the line uncolored"
    assert status.has_class("console-agent-section-line"), "base class must survive"


@pytest.mark.asyncio
async def test_mounted_agent_status_line_carries_state_class(monkeypatch) -> None:
    """The screen's section sync applies the state class to the real Static."""
    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-agent-section-status")
        real_payload = console._agent._console_agent_section_payload()
        patched = ("Agent: running · step 1",) + tuple(real_payload[1:])
        monkeypatch.setattr(
            console._agent, "_console_agent_section_payload", lambda: patched
        )
        console._sync_console_agent_section()
        await pilot.pause()
        status = console.query_one("#console-agent-section-status", Static)
        assert status.has_class("console-agent-section-status-running")


# --- conversation rows ---------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [m for m in ConsoleRunMarker if m is not ConsoleRunMarker.NONE],
    ids=lambda m: m.value,
)
def test_conversation_row_carries_run_marker_state_class(
    marker: ConsoleRunMarker,
) -> None:
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )

    button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary",
        id="row-marked",
        conversation_id="c1",
        run_marker=CONSOLE_RUN_MARKER_GLYPHS[marker],
    )
    assert button.has_class(_ROW_MARKER_CLASSES[marker])
    others = set(_ROW_MARKER_CLASSES.values()) - {_ROW_MARKER_CLASSES[marker]}
    assert not any(button.has_class(cls) for cls in others)


def test_unmarked_conversation_row_has_no_state_class() -> None:
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )

    button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary", id="row-plain", conversation_id="c1", run_marker=""
    )
    assert not any(button.has_class(cls) for cls in _ROW_MARKER_CLASSES.values())


@pytest.mark.asyncio
async def test_selected_conversation_line_is_marked_active_only_with_a_summary() -> (
    None
):
    from dataclasses import replace

    from Tests.UI.test_console_workspace_context_rail import (
        ConsoleHarness,
        _base_grouped_workspace_state,
        _browser_row,
        _build_test_app,
        _grouped_browser_state,
        _wait_for_selector,
    )
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )

        active_class = "console-workspace-selected-conversation-active"

        async def _line_after_recompose(expected_text: str) -> Static:
            # sync_state schedules refresh(recompose=True); wait for the
            # rebuilt Static to carry the new summary before asserting.
            for _ in range(200):
                line = console.query_one(
                    "#console-workspace-selected-conversation", Static
                )
                if str(line.renderable).strip() == expected_text:
                    return line
                await pilot.pause(0.01)
            raise AssertionError(f"line never showed {expected_text!r}")

        def _chats_bucket_row(selected: bool):
            # The Conversations tray summarises only the Chats bucket
            # (global rows); workspace rows are summarised by their tray.
            return _browser_row(
                "conv-a",
                "Loose chat",
                scope_type="global",
                workspace_id=None,
                workspace_label="Chats",
                selected=selected,
            )

        # A selected row yields a non-empty summary -> the line names the
        # active chat and takes the active hue.
        state = _base_grouped_workspace_state()
        selected = _grouped_browser_state(rows=(_chats_bucket_row(True),))
        assert selected.selected_summary, "fixture must select a row"
        tray.sync_state(replace(state, conversation_browser=selected))
        line = await _line_after_recompose(selected.selected_summary)
        assert line.has_class(active_class)

        # No selected row -> placeholder copy stays muted.
        unselected = _grouped_browser_state(rows=(_chats_bucket_row(False),))
        assert not unselected.selected_summary
        tray.sync_state(replace(state, conversation_browser=unselected))
        line = await _line_after_recompose("No active conversation.")
        assert not line.has_class(active_class)


# --- stylesheet contract -------------------------------------------------------


@pytest.mark.parametrize("entry", _STYLESHEETS, ids=lambda e: _stylesheet_text(e)[0])
@pytest.mark.parametrize(
    ("selector", "declaration"),
    sorted(_EXPECTED_DECLARATIONS.items()),
    ids=lambda v: v if isinstance(v, str) else v[1],
)
def test_rail_grammar_declarations_in_source_and_generated_sheet(
    entry, selector: str, declaration: tuple[str, str]
) -> None:
    label, text = _stylesheet_text(entry)
    prop, value = declaration
    assert _declaration(_css_block(text, selector), prop) == value, (
        f"{label}: {selector} {prop} must be {value}"
    )


@pytest.mark.parametrize("entry", _STYLESHEETS, ids=lambda e: _stylesheet_text(e)[0])
def test_status_labels_are_muted_and_not_bold(entry) -> None:
    label, text = _stylesheet_text(entry)
    block = _css_block(text, ".console-workspace-status-label")
    assert _declaration(block, "color") == "$ds-text-muted", label
    assert "bold" not in block, f"{label}: labels must not compete with values"


def test_selected_row_rule_follows_every_marker_rule_in_source() -> None:
    """Equal specificity: source order decides, and selection must win."""
    text = re.sub(
        r"/\*.*?\*/", "", _SOURCE.read_text(encoding="utf-8"), flags=re.DOTALL
    )
    selected_at = text.index(".console-workspace-conversation-row-selected {")
    for cls in _ROW_MARKER_CLASSES.values():
        assert text.index(f".{cls} {{") < selected_at, cls


# --- resolved paint --------------------------------------------------------------


@pytest.mark.asyncio
async def test_rail_grammar_resolves_to_the_active_theme_colors() -> None:
    """Rule-match probe on the mounted Console (lessons-testing-evidence,
    task-31264): the tokens must RESOLVE to the running theme's generated
    values, not merely be declared -- a shadowed token would pass the text
    contracts above and still paint the wrong colour."""
    from textual.color import Color

    from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
    from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
        ConsoleHarness,
    )
    from tldw_chatbook.UI.Console_Modules.agent import apply_console_agent_status_state

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(180, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-agent-section-status")
        theme = host.get_css_variables()

        status = console.query_one("#console-agent-section-status", Static)
        apply_console_agent_status_state(status, "Agent: running · step 1")
        await pilot.pause()
        # $ds-status-running -> $primary
        assert status.styles.color.hex == Color.parse(theme["primary"]).hex

        # Model and Workspaces ship collapsed; a closed section has no body.
        rail_state = console._current_console_rail_state()
        if not rail_state.model_open:
            console._toggle_console_rail_section("model")
        if not rail_state.workspace_open:
            console._toggle_console_rail_section("workspace")
        await _wait_for_selector(console, pilot, "#console-model-section-temperature")
        await _wait_for_selector(console, pilot, "#console-active-workspace-value")

        value = console.query_one(
            "#console-model-section-temperature .console-model-section-value", Static
        )
        label = console.query_one(
            "#console-model-section-temperature .console-model-section-label", Static
        )
        # $ds-value-fg -> $text-accent; the label stays a different, muted colour.
        assert value.styles.color.hex == Color.parse(theme["text-accent"]).hex
        assert label.styles.color.hex != value.styles.color.hex

        workspace = console.query_one("#console-active-workspace-value", Static)
        # $ds-active-fg -> $text-primary
        assert workspace.styles.color.hex == Color.parse(theme["text-primary"]).hex
        assert workspace.styles.color.hex != value.styles.color.hex
