"""Tests for the Library prompts list canvas widget and its screen wiring.

Widget-only tests mount ``LibraryPromptsListCanvas`` directly in a bare
``App`` subclass (mirrors ``test_library_export_cancel.py``'s
``test_cancel_button_visible_only_while_running``) -- this harness has no
app CSS loaded, so assertions stick to structure/content, never geometry
(Horizontal's own `layout: horizontal` is baked into the Textual widget
class itself -- not the app's custom stylesheet -- but pixel/region
assertions are still avoided here per the "no geometry assertions" rule;
"one row" is instead proven structurally via shared Horizontal parentage).

Screen-wiring tests call ``LibraryScreen`` bound methods directly against a
``SimpleNamespace`` stand-in for ``self`` (mirrors
``test_library_export_cancel.py``'s direct-method style), plus one real
``App.run_test()`` integration test reusing the existing
``Tests.UI.test_library_shell`` / ``Tests.UI.test_destination_shells``
harness fixtures to prove the rail row -> snapshot fetch -> canvas mount
path end to end.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from loguru import logger
from textual.app import App
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Button, Checkbox, Collapsible, Input, Static, TextArea

from tldw_chatbook.DB.Prompts_DB import ConflictError, DatabaseError, PromptsDatabase
from tldw_chatbook.Library.library_prompts_state import (
    PromptBrowseResult,
    PromptBrowseScope,
    PromptEditorState,
    PromptHistoryState,
    PromptHistoryRestoreOutcome,
    PromptListRow,
    PromptsListState,
    apply_prompt_history_count,
    apply_prompt_history_page,
    apply_prompt_history_preview,
    begin_prompt_browse,
    build_prompt_browse_error,
    build_prompt_browse_result,
    build_prompt_editor_state,
    build_prompt_history_page,
    build_prompt_history_state,
    begin_prompt_history_count,
    begin_prompt_history_page,
    begin_prompt_history_preview,
    prepare_prompt_artifact_save,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_PROMPTS,
    LIBRARY_ROW_CREATE_PROMPT,
)
from tldw_chatbook.Prompt_Management.Prompts_Interop import (
    parse_markdown_prompts_from_content,
)
from tldw_chatbook.Prompt_Management.prompt_markdown_export import (
    render_prompt_markdown,
)
from tldw_chatbook.Prompt_Management.prompt_restore_errors import (
    PromptRestoreError,
    PromptRestoreErrorCode,
)
from tldw_chatbook.Prompt_Management.prompt_variables import (
    PromptVariableApplication,
    fingerprint_system_text,
)
from tldw_chatbook.Prompt_Management import (
    prompt_scope_service as prompt_scope_service_module,
)
from tldw_chatbook.Prompt_Management.prompt_scope_service import (
    LocalPromptService as ScopeLocalPromptService,
    PromptScopeService,
)
from tldw_chatbook.Prompt_Management.prompt_source_capabilities import (
    local_prompt_capabilities,
)
from tldw_chatbook.runtime_policy.enforcement import ServicePolicyEnforcer
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Library_Modules import (
    LibraryPromptHistoryRegion,
    prompt_history as prompt_history_module,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Widgets.confirmation_dialog import ConfirmationDialog
from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel
from tldw_chatbook.UI.Navigation.screen_state_store import (
    ConsolePromptTargetProjection,
)
from tldw_chatbook.Widgets.Library.library_prompts_canvas import (
    LibraryPromptsListCanvas,
)
from tldw_chatbook.Widgets.Library.prompt_delete_confirmation_modal import (
    PromptDeleteConfirmationModal,
    PromptDeleteDecision,
)
from tldw_chatbook.Widgets.Prompts.prompt_block_editor import PromptBlockEditor
from tldw_chatbook.Widgets.Console.prompt_variables_dialog import (
    PromptVariablesDialog,
)

from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
)
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _active_library_screen,
    _fake_import_dialog_result,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.app_factory import _build_test_app


REPO_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_TERMINAL = REPO_ROOT / "tldw_chatbook/css/components/_agentic_terminal.tcss"
BUNDLED_STYLESHEET = REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


def _css_block(text: str, selector: str) -> str:
    """Return a CSS rule body starting at ``selector`` (mirrors
    ``test_product_maturity_phase3_library_contract_layout.py``'s helper of
    the same name)."""
    start = text.index(selector)
    block_start = text.index("{", start)
    block_end = text.index("}", block_start)
    return text[block_start:block_end]


def _painted_relative_luminance(color) -> float:
    """Return WCAG luminance for a compositor-painted Rich colour."""
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


def _painted_contrast(first, second) -> float:
    lighter, darker = sorted(
        (_painted_relative_luminance(first), _painted_relative_luminance(second)),
        reverse=True,
    )
    return (lighter + 0.05) / (darker + 0.05)


def _painted_style_of_text(app: App, region, needle: str):
    """Return the final compositor style painting ``needle`` in ``region``."""
    strips = list(app.screen._compositor.render_strips())
    for y in range(region.y, region.y + region.height):
        if y >= len(strips):
            break
        segments = list(strips[y]._segments)
        row_text = "".join(segment.text for segment in segments)
        index = row_text.find(needle)
        if index == -1:
            continue
        x = 0
        for segment in segments:
            if x + len(segment.text) > index:
                return segment.style
            x += len(segment.text)
    return None


def _three_row_state(*, sort: str = "newest") -> PromptsListState:
    return PromptsListState(
        rows=(
            PromptListRow(prompt_id=1, name="Summarize", secondary="Alice · 3m"),
            PromptListRow(
                prompt_id=2, name="[draft] Q3 plan [wip]", secondary="Bob · 1h"
            ),
            PromptListRow(prompt_id=3, name="Translate", secondary="2d"),
        ),
        count=3,
        sort=sort,
    )


def _browse_result(
    *,
    items: list[dict[str, Any]] | None = None,
    query: str = "",
    collection_id: int | None = None,
    page: int = 1,
    page_size: int = 50,
    total_items: int | None = None,
    request_token: int = 1,
) -> PromptBrowseResult:
    """Build one exact settled result for prompt-canvas tests."""
    rows = items or []
    total = len(rows) if total_items is None else total_items
    total_pages = (total + page_size - 1) // page_size if total else 0
    return build_prompt_browse_result(
        PromptBrowseScope(
            query=query,
            collection_id=collection_id,
            page=page,
            page_size=page_size,
        ),
        {
            "items": rows,
            "total_items": total,
            "total_pages": total_pages,
            "current_page": page if total else 1,
            "per_page": page_size,
        },
        request_token=request_token,
    )


class _CanvasHost(App):
    def __init__(self, state: PromptsListState | None, **kwargs: Any) -> None:  # type: ignore[valid-type]
        super().__init__()
        self._state = state
        self._kwargs = kwargs

    def compose(self):
        yield LibraryPromptsListCanvas(
            self._state, id="library-prompts-canvas", **self._kwargs
        )


class _StyledCanvasHost(_CanvasHost):
    """Canvas harness with the application's real layout rules loaded."""

    CSS_PATH = str(BUNDLED_STYLESHEET)


def _capture_region_messages(region: LibraryPromptHistoryRegion) -> list[Any]:
    """Capture semantic messages emitted by one mounted history region."""
    posted: list[Any] = []
    original_post_message = region.post_message

    def capture(message):
        posted.append(message)
        return original_post_message(message)

    region.post_message = capture  # type: ignore[method-assign]
    return posted


def _structured_editor_state(*, artifact_type: str = "prompt") -> PromptEditorState:
    kind = "block_recipe" if artifact_type == "recipe" else "block_prompt"
    return build_prompt_editor_state(
        {
            "id": 41,
            "name": "Blocks",
            "artifact_type": artifact_type,
            "prompt_format": "structured",
            "prompt_schema_version": 2,
            "prompt_definition": {
                "schema_version": 2,
                "kind": kind,
                "lanes": [
                    {
                        "id": "system",
                        "blocks": [
                            {
                                "id": "role",
                                "title": "Role",
                                "syntax": "markdown",
                                "content": "Be exact.",
                            }
                        ],
                    },
                    {
                        "id": "user",
                        "blocks": [
                            {
                                "id": "goal",
                                "title": "Goal",
                                "syntax": "freeform",
                                "content": "Ship it.",
                            }
                        ],
                    },
                ],
            },
            "system_prompt": "# Role\n\nBe exact.",
            "user_prompt": "Ship it.",
            "version": 3,
            "backend": "local",
        }
    )


def _history_state(
    *,
    current_version: int = 3,
    selected_version: int | None = 2,
    restore_eligible: bool = True,
    compatibility_reason: str = "",
    keywords_captured: bool = False,
) -> PromptHistoryState:
    """Build one loaded real reducer state for widget-level history pilots."""
    state = build_prompt_history_state(
        prompt_uuid="prompt-history-uuid",
        current_version=current_version,
        scope_token=1,
    )
    state, count_request = begin_prompt_history_count(state, request_token=1)
    state = apply_prompt_history_count(state, count_request, total_count=2)
    state, page_request = begin_prompt_history_page(state, request_token=2)
    state = apply_prompt_history_page(
        state,
        page_request,
        build_prompt_history_page(
            {
                "items": [
                    {
                        "prompt_uuid": "prompt-history-uuid",
                        "change_id": 22,
                        "version": 2,
                        "timestamp": "2026-08-08T12:00:00+00:00",
                        "artifact_type": "prompt",
                        "artifact_type_raw": "",
                        "name": "Current-ish [draft]",
                        "author": "A",
                        "details": "Second retained row",
                        "system_prompt": "[bold]literal system[/bold]",
                        "user_prompt": "[/] literal user [wip]",
                        "keywords": ["alpha"],
                        "keywords_captured": keywords_captured,
                        "compatibility_state": (
                            "compatible" if restore_eligible else "future_artifact_type"
                        ),
                        "compatibility_reason": compatibility_reason,
                        "restore_eligible": restore_eligible,
                        "changed_fields": ["system_prompt"],
                        "change_summary": "System prompt",
                    },
                    {
                        "prompt_uuid": "prompt-history-uuid",
                        "change_id": 11,
                        "version": 1,
                        "timestamp": "2026-08-08T11:00:00+00:00",
                        "artifact_type": "prompt",
                        "artifact_type_raw": "future-artifact",
                        "name": "Original",
                        "author": "A",
                        "details": "First retained row",
                        "system_prompt": "old system",
                        "user_prompt": "old user",
                        "keywords": [],
                        "keywords_captured": False,
                        "compatibility_state": "future_artifact_type",
                        "compatibility_reason": "Future artifact types are preview-only.",
                        "restore_eligible": False,
                        "changed_fields": [],
                        "change_summary": "Created",
                    },
                ],
                "total_count": 2,
                "has_more": False,
                "next_before_change_id": None,
            }
        ),
    )
    if selected_version is None:
        return state
    selected = next(row for row in state.rows if row.version == selected_version)
    state, preview_request = begin_prompt_history_preview(
        state,
        change_id=selected.change_id,
        source_version=selected.version,
        request_token=3,
    )
    return apply_prompt_history_preview(state, preview_request)


# ---------------------------------------------------------------------------
# Widget-only tests (Step 2 of the brief)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompt_history_region_owns_disclosure_row_page_and_restore_events():
    """The screen-specific region emits semantic events instead of leaking children."""
    history = replace(_history_state(), has_more=True, next_before_change_id=11)
    app = _CanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=history,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        region = pilot.app.query_one(LibraryPromptHistoryRegion)
        posted = _capture_region_messages(region)
        disclosure = region.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )

        disclosure.collapsed = True
        await pilot.pause()
        disclosure.collapsed = False
        await pilot.pause()
        region.query_one("#library-prompt-history-row-11", Button).press()
        region.query_one("#library-prompt-history-load-older", Button).press()
        region.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()

        assert any(
            isinstance(message, LibraryPromptHistoryRegion.DisclosureClosed)
            for message in posted
        )
        assert any(
            isinstance(message, LibraryPromptHistoryRegion.DisclosureOpened)
            for message in posted
        )
        selected = next(
            message
            for message in posted
            if isinstance(message, LibraryPromptHistoryRegion.RowSelected)
        )
        assert (selected.change_id, selected.source_version) == (11, 1)
        assert any(
            isinstance(message, LibraryPromptHistoryRegion.PageRequested)
            for message in posted
        )
        assert any(
            isinstance(message, LibraryPromptHistoryRegion.RestoreRequested)
            for message in posted
        )
        scoped = [
            message
            for message in posted
            if isinstance(
                message,
                (
                    LibraryPromptHistoryRegion.DisclosureClosed,
                    LibraryPromptHistoryRegion.DisclosureOpened,
                    LibraryPromptHistoryRegion.PageRequested,
                    LibraryPromptHistoryRegion.RowSelected,
                    LibraryPromptHistoryRegion.RestoreRequested,
                ),
            )
        ]
        assert scoped
        assert {(message.prompt_uuid, message.scope_token) for message in scoped} == {
            ("prompt-history-uuid", 1)
        }
        with pytest.raises(AttributeError):
            scoped[0].prompt_uuid = "mutated"


@pytest.mark.asyncio
async def test_prompt_history_region_owns_retry_and_snapshot_reload_events():
    """Every recovery control is translated into one typed region event."""
    app = _CanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=replace(_history_state(), count_status="error"),
    )

    async with app.run_test(size=(100, 30)) as pilot:
        region = pilot.app.query_one(LibraryPromptHistoryRegion)
        posted = _capture_region_messages(region)

        region.query_one("#library-prompt-history-retry-count", Button).press()
        await pilot.pause()
        region.sync_state(
            replace(
                _history_state(),
                rows=(),
                page_status="error",
                error="Couldn't load retained history.",
            ),
            dirty=False,
            current_compatible=True,
        )
        await pilot.pause()
        region.query_one("#library-prompt-history-retry-page", Button).press()
        await pilot.pause()
        region.sync_state(
            replace(
                _history_state(),
                restore_outcome=PromptHistoryRestoreOutcome(
                    kind="snapshot_unavailable",
                    message="The retained snapshot is unavailable. Reload retained history.",
                    reload_required=True,
                ),
            ),
            dirty=False,
            current_compatible=True,
        )
        await pilot.pause()
        assert (
            str(
                region.query_one(
                    "#library-prompt-history-restore-reason", Static
                ).renderable
            )
            == "The retained snapshot is unavailable. Reload retained history."
        )
        assert len(region.query("#library-prompt-history-outcome")) == 0
        region.query_one("#library-prompt-history-reload", Button).press()
        await pilot.pause()

        assert any(
            isinstance(message, LibraryPromptHistoryRegion.CountRetryRequested)
            for message in posted
        )
        assert (
            sum(
                isinstance(message, LibraryPromptHistoryRegion.PageRequested)
                for message in posted
            )
            == 1
        )
        assert any(
            isinstance(message, LibraryPromptHistoryRegion.ReloadRequested)
            for message in posted
        )
        scoped = [
            message
            for message in posted
            if isinstance(
                message,
                (
                    LibraryPromptHistoryRegion.CountRetryRequested,
                    LibraryPromptHistoryRegion.PageRequested,
                    LibraryPromptHistoryRegion.ReloadRequested,
                ),
            )
        ]
        assert {(message.prompt_uuid, message.scope_token) for message in scoped} == {
            ("prompt-history-uuid", 1)
        }


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["close", "page", "reload", "row", "restore"])
async def test_prompt_history_region_old_dom_action_keeps_rendered_scope(
    action: str,
):
    """An outgoing control emits its rendered scope during recompose overlap."""
    old_state = replace(
        _history_state(),
        has_more=True,
        next_before_change_id=11,
        restore_outcome=PromptHistoryRestoreOutcome(
            kind="snapshot_unavailable",
            message="This retained version is no longer available. Reload retained history.",
            reload_required=True,
        ),
    )
    app = _CanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=old_state,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        region = pilot.app.query_one(LibraryPromptHistoryRegion)
        posted = _capture_region_messages(region)
        old_disclosure = region.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        old_page = region.query_one("#library-prompt-history-load-older", Button)
        old_reload = region.query_one("#library-prompt-history-reload", Button)
        old_row = region.query_one("#library-prompt-history-row-11", Button)
        old_restore = region.query_one("#library-prompt-history-restore", Button)
        new_state = build_prompt_history_state(
            prompt_uuid="new-prompt-uuid",
            current_version=9,
            scope_token=2,
        )

        region.sync_state(new_state, dirty=False, current_compatible=True)
        assert region.view_model[0] is new_state
        before = len(posted)
        if action == "close":
            old_disclosure.collapsed = True
            region._on_disclosure_toggled(Collapsible.Toggled(old_disclosure))
        elif action == "page":
            region._on_page_requested(Button.Pressed(old_page))
        elif action == "reload":
            region._on_reload_requested(Button.Pressed(old_reload))
        elif action == "row":
            region._on_row_selected(Button.Pressed(old_row))
        else:
            region._on_restore_requested(Button.Pressed(old_restore))

        emitted = posted[before:]
        assert len(emitted) == 1
        assert (emitted[0].prompt_uuid, emitted[0].scope_token) == (
            "prompt-history-uuid",
            1,
        )


@pytest.mark.asyncio
async def test_prompts_canvas_renders_a_button_per_row():
    """A 3-row state renders exactly 3 prompt row buttons, ids
    ``library-prompt-row-<id>`` keyed by the row's ``prompt_id`` (not
    index, unlike the notes canvas)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        for prompt_id in (1, 2, 3):
            button = pilot.app.query_one(f"#library-prompt-row-{prompt_id}", Button)
            assert button.prompt_id == prompt_id
        rows = pilot.app.query(".library-prompt-row")
        assert len(rows) == 3


@pytest.mark.asyncio
async def test_prompts_canvas_rows_show_first_class_type_source_and_lane_summary():
    state = PromptsListState(
        rows=(
            PromptListRow(
                prompt_id=8,
                name="Outcome first",
                secondary="Reusable structure",
                artifact_type="recipe",
                type_label="Recipe",
                source_label="Server",
                lane_summary="System + User",
            ),
        ),
        count=1,
        sort="newest",
    )
    app = _CanvasHost(state)

    async with app.run_test() as pilot:
        label = str(pilot.app.query_one("#library-prompt-row-8", Button).label)
        assert "Recipe · Server · System + User" in label
        assert "Reusable structure" in label


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 40), (80, 24)])
async def test_prompts_canvas_supported_artifact_uses_shared_editor_and_read_only_preview(
    size: tuple[int, int],
):
    editor_state = _structured_editor_state(artifact_type="recipe")
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)

    async with app.run_test(size=size) as pilot:
        editor = pilot.app.query_one(PromptBlockEditor)
        assert editor.state.artifact_type == "recipe"
        assert editor.state.definition.kind == "block_recipe"
        system_preview = pilot.app.query_one("#library-prompt-system", TextArea)
        user_preview = pilot.app.query_one("#library-prompt-user", TextArea)
        assert system_preview.read_only is True
        assert user_preview.read_only is True
        assert system_preview.text == "# Role\n\nBe exact."
        assert user_preview.text == "Ship it."
        assert (
            pilot.app.query_one("#library-prompt-recipe-starter", Checkbox).value
            is False
        )
        system_apply = pilot.app.query_one("#prompt-editor-apply-system", Checkbox)
        assert system_apply.disabled is True
        assert system_apply.value is False
        assert "System apply is unavailable in Library" in str(
            pilot.app.query_one("#prompt-editor-apply-reason", Static).renderable
        )
        if size[0] == 80:
            assert editor.has_class("-narrow")
            assert editor.query_one("#prompt-editor-footer").has_class("two-row")


@pytest.mark.asyncio
async def test_prompts_canvas_shared_editor_patches_preview_without_recomposing_textareas():
    editor_state = _structured_editor_state()
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)

    async with app.run_test(size=(120, 40)) as pilot:
        editor = pilot.app.query_one(PromptBlockEditor)
        role = pilot.app.query_one("#prompt-block-content-role", TextArea)
        goal = pilot.app.query_one("#prompt-block-content-goal", TextArea)
        preview = pilot.app.query_one("#library-prompt-system", TextArea)
        role_identity = id(role)
        goal_identity = id(goal)
        preview_identity = id(preview)

        await editor._change_field("role", "content", "Be concise.")
        await pilot.pause()

        assert (
            id(editor.query_one("#prompt-block-content-role", TextArea))
            == role_identity
        )
        assert (
            id(editor.query_one("#prompt-block-content-goal", TextArea))
            == goal_identity
        )
        assert (
            id(pilot.app.query_one("#library-prompt-system", TextArea))
            == preview_identity
        )
        assert preview.text == "# Role\n\nBe concise."


@pytest.mark.asyncio
async def test_prompts_canvas_guarded_foreign_artifact_is_read_only_with_conversion_recovery():
    detail = {
        "id": 9,
        "name": "Foreign recipe",
        "artifact_type": "recipe",
        "prompt_format": "structured",
        "prompt_schema_version": 1,
        "prompt_definition": {"schema_version": 1, "kind": "future"},
        "system_prompt": "compat system",
        "user_prompt": "compat user",
    }
    editor_state = build_prompt_editor_state(detail)
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)

    async with app.run_test() as pilot:
        assert len(pilot.app.query(PromptBlockEditor)) == 0
        assert pilot.app.query_one("#library-prompt-system", TextArea).read_only is True
        assert pilot.app.query_one("#library-prompt-user", TextArea).read_only is True
        reason = pilot.app.query_one("#library-prompt-compatibility", Static)
        assert "read-only" in str(reason.renderable).lower()
        assert pilot.app.query_one("#library-prompt-convert", Button).disabled is False


@pytest.mark.asyncio
async def test_prompts_canvas_escapes_bracket_titles_verbatim():
    """A prompt named "[draft] Q3 plan [wip]" renders its bracket segments
    verbatim in the row label instead of having them consumed as Rich
    markup (the search-history Button-label lesson)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-prompt-row-2", Button)
        first_line = str(button.label).splitlines()[0]
        assert first_line == "[draft] Q3 plan [wip]"


@pytest.mark.asyncio
async def test_prompts_canvas_escapes_secondary_line_markup_and_unmatched_close_tag():
    """The row's secondary line (``details`` -- the prompt description) must
    be escaped exactly like the name is. Unescaped, a description containing
    ``[wip]`` is silently swallowed by Rich markup parsing, and an unmatched
    closing tag like ``[/]`` raises a ``MarkupError`` that crashes the ENTIRE
    list render (every row gone) -- not just this one row's label. This must
    render cleanly and keep the bracketed text literal."""
    state = PromptsListState(
        rows=(
            # "[/]" appears before any opening tag, so Rich's markup parser
            # treats it as an unmatched closing tag (MarkupError) rather
            # than a tag pair that quietly swallows its contents -- this is
            # the crash class the fix must prevent, not just the "silently
            # swallowed" class covered by "[wip]" alone.
            PromptListRow(
                prompt_id=1, name="Draft plan", secondary="[/] notes [wip] · 2h"
            ),
        ),
        count=1,
        sort="newest",
    )
    app = _CanvasHost(state)
    async with app.run_test() as pilot:
        button = pilot.app.query_one("#library-prompt-row-1", Button)
        label_text = str(button.label)
        assert "[wip]" in label_text
        assert "[/]" in label_text


@pytest.mark.asyncio
async def test_prompts_canvas_toolbar_is_one_horizontal_row():
    """sort/Import share a single ``ds-toolbar`` Horizontal parent -- proven
    structurally (shared parentage), not via region/geometry (the bare
    harness has no app CSS loaded)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-prompts-sort", Button)
        import_button = pilot.app.query_one("#library-prompts-import", Button)
        toolbar = sort_button.parent
        assert toolbar is not None and toolbar.has_class("ds-toolbar")
        assert import_button.parent is toolbar


@pytest.mark.asyncio
async def test_prompts_canvas_list_toolbar_has_no_dead_export_button():
    """D5 (Task 8c): the list-toolbar "Export..." button had no handler
    anywhere -- pressing it silently no-op'd. Bulk export is deferred to
    task-197; per-prompt export lives in the editor's own
    ``#library-prompt-export`` and still works. The dead affordance is
    removed rather than wired to a fake bulk export."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-prompts-export")) == 0


@pytest.mark.asyncio
async def test_prompt_history_canvas_renders_expanded_literal_read_only_preview():
    """Removing history composition loses the disclosure and exact stored lanes."""
    history = _history_state()
    app = _CanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=history,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        disclosure = pilot.app.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        assert disclosure.collapsed is False
        assert str(disclosure.title) == "Retained history (2)"
        assert [
            button.source_version
            for button in pilot.app.query(".library-prompt-history-row")
        ] == [2, 1]
        selected_label = str(
            pilot.app.query_one("#library-prompt-history-row-22", Button).label
        )
        assert "v2 · change 22" in selected_label
        assert "System prompt" in selected_label
        future_label = str(
            pilot.app.query_one("#library-prompt-history-row-11", Button).label
        )
        assert "future-artifact" in future_label

        system = pilot.app.query_one("#library-prompt-history-system", TextArea)
        user = pilot.app.query_one("#library-prompt-history-user", TextArea)
        assert system.read_only is True
        assert user.read_only is True
        assert system.text == "[bold]literal system[/bold]"
        assert user.text == "[/] literal user [wip]"
        assert "Current keywords were not captured" in str(
            pilot.app.query_one(
                "#library-prompt-history-keywords-disclosure", Static
            ).renderable
        )
        assert len(pilot.app.query(PromptBlockEditor)) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("history", "dirty", "current_compatible", "reason"),
    [
        (
            _history_state(),
            True,
            True,
            "Save or discard unsaved changes before restoring retained history.",
        ),
        (
            _history_state(),
            False,
            False,
            "This compatibility-only editor cannot restore retained history.",
        ),
        (
            _history_state(
                restore_eligible=False,
                compatibility_reason="Future artifact types are preview-only.",
            ),
            False,
            True,
            "Future artifact types are preview-only.",
        ),
        (
            _history_state(selected_version=None),
            False,
            True,
            "Select a retained version to restore.",
        ),
    ],
    ids=["dirty", "compatibility-editor", "preview-only-row", "no-selection"],
)
async def test_prompt_history_restore_disabled_reason_is_visible_while_preview_remains_available(
    history: PromptHistoryState,
    dirty: bool,
    current_compatible: bool,
    reason: str,
):
    """Each restore refusal must disable only restore, never retained viewing."""
    app = _CanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=history,
        dirty=dirty,
        history_current_compatible=current_compatible,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        assert (
            pilot.app.query_one(
                "#library-prompt-history-collapsible", Collapsible
            ).collapsed
            is False
        )
        restore = pilot.app.query_one("#library-prompt-history-restore", Button)
        assert restore.disabled is True
        assert (
            str(
                pilot.app.query_one(
                    "#library-prompt-history-restore-reason", Static
                ).renderable
            )
            == reason
        )
        assert len(pilot.app.query(".library-prompt-history-row")) == 2


@pytest.mark.asyncio
async def test_prompts_canvas_filter_input_prefilled():
    app = _CanvasHost(_three_row_state(), filter_value="plan")
    async with app.run_test() as pilot:
        filter_input = pilot.app.query_one("#library-prompts-filter", Input)
        assert filter_input.value == "plan"


@pytest.mark.asyncio
async def test_prompts_canvas_empty_state_renders_empty_copy_not_list():
    empty_state = PromptsListState(rows=(), count=0, sort="newest")
    app = _CanvasHost(empty_state)
    async with app.run_test() as pilot:
        empty = pilot.app.query_one("#library-prompts-empty")
        assert "No prompts yet" in str(empty.renderable)
        assert len(pilot.app.query(".library-prompt-row")) == 0


@pytest.mark.asyncio
async def test_prompts_canvas_sort_label_reflects_sort_mode():
    app = _CanvasHost(_three_row_state(sort="name"), sort_mode="name")
    async with app.run_test() as pilot:
        sort_button = pilot.app.query_one("#library-prompts-sort", Button)
        assert "Name" in str(sort_button.label)


@pytest.mark.asyncio
async def test_prompts_canvas_loading_keeps_search_and_toolbar_stable():
    scope = PromptBrowseScope(query="plan")
    loading = begin_prompt_browse(scope, request_token=4)
    app = _CanvasHost(
        PromptsListState(rows=(), count=0, sort="newest"),
        browse_result=loading,
        filter_value=scope.query,
    )

    async with app.run_test() as pilot:
        header = pilot.app.query_one("#library-prompts-header", Static)
        assert str(header.renderable) == "Prompts (…)"
        assert pilot.app.query_one("#library-prompts-filter", Input).value == "plan"
        assert pilot.app.query_one("#library-prompts-sort", Button)
        assert pilot.app.query_one("#library-prompts-import", Button)
        loading_line = pilot.app.query_one("#library-prompts-loading", Static)
        assert str(loading_line.renderable) == "Loading prompts…"
        assert not pilot.app.query(".library-prompt-row")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_browse_result(), "No prompts yet. Create or import a prompt to begin."),
        (
            _browse_result(collection_id=9),
            "This collection has no prompts. Choose another collection or add prompts.",
        ),
        (
            _browse_result(query="[draft]"),
            'No prompts match "[draft]". Clear the search or try different words.',
        ),
    ],
    ids=["empty-library", "empty-collection", "no-matches"],
)
async def test_prompts_canvas_renders_truthful_literal_empty_states(
    result: PromptBrowseResult, expected: str
):
    app = _CanvasHost(
        PromptsListState(rows=(), count=0, sort="newest"),
        browse_result=result,
        filter_value=result.scope.query,
    )

    async with app.run_test() as pilot:
        header = pilot.app.query_one("#library-prompts-header", Static)
        assert str(header.renderable) == "Prompts (0)"
        empty = pilot.app.query_one("#library-prompts-empty", Static)
        assert str(empty.renderable) == expected
        assert str(empty.render()) == expected
        assert not pilot.app.query("#library-prompts-retry")


@pytest.mark.asyncio
async def test_prompts_canvas_error_names_recovery_and_renders_retry():
    error = build_prompt_browse_error(
        PromptBrowseScope(query="plan"),
        request_token=3,
        error="Couldn't load prompts. Check the local Library and retry.",
    )
    app = _CanvasHost(
        PromptsListState(rows=(), count=0, sort="newest"),
        browse_result=error,
        filter_value="plan",
    )

    async with app.run_test() as pilot:
        header = pilot.app.query_one("#library-prompts-header", Static)
        assert str(header.renderable) == "Prompts (…)"
        line = pilot.app.query_one("#library-prompts-error", Static)
        assert str(line.renderable) == error.error
        assert str(line.render()) == error.error
        assert pilot.app.query_one("#library-prompts-retry", Button)
        assert not pilot.app.query("#library-prompts-empty")


@pytest.mark.asyncio
async def test_prompts_canvas_minimal_paging_is_literal_and_keyboard_ordered():
    items = [
        {
            "id": f"local:prompt:uuid-{index}",
            "local_id": index,
            "name": f"Prompt {index}",
            "backend": "local",
        }
        for index in range(51, 76)
    ]
    result = _browse_result(
        items=items,
        page=2,
        page_size=50,
        total_items=75,
    )
    state = PromptsListState(
        rows=tuple(
            PromptListRow(prompt_id=index, name=f"Prompt {index}", secondary="")
            for index in range(51, 76)
        ),
        count=25,
        sort="newest",
    )
    app = _CanvasHost(state, browse_result=result)

    async with app.run_test() as pilot:
        header = pilot.app.query_one("#library-prompts-header", Static)
        assert str(header.renderable) == "Prompts (75)"
        page_label = pilot.app.query_one("#library-prompts-page-label", Static)
        assert str(page_label.renderable) == "Page 2 of 2 · showing 51–75 of 75"
        assert str(page_label.render()) == "Page 2 of 2 · showing 51–75 of 75"
        previous = pilot.app.query_one("#library-prompts-page-previous", Button)
        next_page = pilot.app.query_one("#library-prompts-page-next", Button)
        assert previous.disabled is False
        assert next_page.disabled is True
        focusable_ids = [
            widget.id
            for widget in pilot.app.screen.focus_chain
            if widget.id is not None
        ]
        assert focusable_ids.index("library-prompts-filter") < focusable_ids.index(
            "library-prompts-sort"
        )
        assert focusable_ids.index("library-prompts-sort") < focusable_ids.index(
            "library-prompts-page-previous"
        )
        assert "library-prompts-page-next" not in focusable_ids
        assert focusable_ids.index(
            "library-prompts-page-previous"
        ) < focusable_ids.index("library-prompt-row-51")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(64, 24), (120, 40)], ids=["narrow", "wide"])
@pytest.mark.parametrize("page", [1, 2], ids=["first-page", "second-page"])
async def test_prompts_canvas_paging_actions_are_render_safe(
    size: tuple[int, int], page: int
) -> None:
    """Paging copy wraps separately while enabled actions stay fully visible."""
    first_item = 1 if page == 1 else 51
    last_item = 50 if page == 1 else 75
    items = [
        {
            "id": f"local:prompt:uuid-{index}",
            "local_id": index,
            "name": f"Prompt {index}",
            "backend": "local",
        }
        for index in range(first_item, last_item + 1)
    ]
    result = _browse_result(
        items=items,
        page=page,
        page_size=50,
        total_items=75,
    )
    state = PromptsListState(
        rows=tuple(
            PromptListRow(index, f"Prompt {index}", "")
            for index in range(first_item, last_item + 1)
        ),
        count=len(items),
        sort="newest",
    )
    app = _StyledCanvasHost(state, browse_result=result)

    async with app.run_test(size=size) as pilot:
        canvas = pilot.app.query_one(
            "#library-prompts-canvas", LibraryPromptsListCanvas
        )
        label = pilot.app.query_one("#library-prompts-page-label", Static)
        previous = pilot.app.query_one("#library-prompts-page-previous", Button)
        next_page = pilot.app.query_one("#library-prompts-page-next", Button)
        enabled = next_page if page == 1 else previous
        toolbar = previous.parent

        assert isinstance(toolbar, Horizontal)
        assert toolbar is next_page.parent
        assert toolbar.has_class("ds-toolbar")
        assert tuple(toolbar.children) == (previous, next_page)
        assert label.parent is canvas
        assert str(label.renderable) == (
            f"Page {page} of 2 · showing {first_item}–{last_item} of 75"
        )
        assert label.region.width > 0
        assert label.region.height > 0
        assert label.region.right <= size[0]
        assert label.region.bottom <= size[1]
        assert enabled.disabled is False
        assert enabled.region.width > 0
        assert enabled.region.height > 0
        assert enabled.region.right <= size[0]
        assert enabled.region.bottom <= size[1]
        assert previous.region.y == next_page.region.y
        assert previous.region.height == next_page.region.height


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(64, 24), (120, 40)], ids=["narrow", "wide"])
@pytest.mark.parametrize(
    ("result", "state", "selector"),
    [
        (
            _browse_result(
                items=[
                    {
                        "id": "local:prompt:one",
                        "local_id": 1,
                        "name": "Visible prompt",
                        "backend": "local",
                    }
                ]
            ),
            PromptsListState(
                rows=(PromptListRow(1, "Visible prompt", ""),),
                count=1,
                sort="newest",
            ),
            "#library-prompt-row-1",
        ),
        (
            begin_prompt_browse(PromptBrowseScope(), request_token=8),
            PromptsListState(rows=(), count=0, sort="newest"),
            "#library-prompts-loading",
        ),
        (
            _browse_result(),
            PromptsListState(rows=(), count=0, sort="newest"),
            "#library-prompts-empty",
        ),
        (
            _browse_result(query="[missing]"),
            PromptsListState(rows=(), count=0, sort="newest"),
            "#library-prompts-empty",
        ),
        (
            build_prompt_browse_error(
                PromptBrowseScope(),
                request_token=9,
                error="Couldn't load prompts. Check the local Library and retry.",
            ),
            PromptsListState(rows=(), count=0, sort="newest"),
            "#library-prompts-error",
        ),
    ],
    ids=["normal", "loading", "empty", "no-match", "error"],
)
async def test_prompts_canvas_browse_states_render_at_narrow_and_wide_sizes(
    result: PromptBrowseResult,
    state: PromptsListState,
    selector: str,
    size: tuple[int, int],
):
    """All exact-browse states paint with stable toolbar dimensions."""
    app = _StyledCanvasHost(
        state,
        browse_result=result,
        filter_value=result.scope.query,
    )

    async with app.run_test(size=size) as pilot:
        rendered = pilot.app.query_one(selector)
        search = pilot.app.query_one("#library-prompts-filter", Input)
        sort = pilot.app.query_one("#library-prompts-sort", Button)
        assert rendered.region.width > 0
        assert rendered.region.height > 0
        assert search.region.width > 0
        assert sort.region.width > 0
        assert search.region.y < rendered.region.bottom


# ---------------------------------------------------------------------------
# Task 5: toolbar Import… row widget tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_hidden_by_default():
    """The Import row (path Input, Run/Cancel actions, outcome line) is not
    mounted at all while ``import_open`` is ``False`` (the default)."""
    app = _CanvasHost(_three_row_state())
    async with app.run_test() as pilot:
        assert len(pilot.app.query("#library-prompts-import-path")) == 0
        assert len(pilot.app.query("#library-prompts-import-run")) == 0
        assert len(pilot.app.query("#library-prompts-import-cancel")) == 0


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_renders_when_open():
    """``import_open=True`` renders the path Input (prefilled from
    ``import_path``), Run/Cancel actions, and the outcome Static (showing
    ``import_status`` verbatim)."""
    app = _CanvasHost(
        _three_row_state(),
        import_open=True,
        import_path="/tmp/my-prompts.md",
        import_status="2 imported · 1 skipped (duplicate name)",
    )
    async with app.run_test() as pilot:
        path_input = pilot.app.query_one("#library-prompts-import-path", Input)
        assert path_input.value == "/tmp/my-prompts.md"
        assert pilot.app.query_one("#library-prompts-import-run", Button)
        assert pilot.app.query_one("#library-prompts-import-cancel", Button)
        status = pilot.app.query_one("#library-prompts-import-status", Static)
        assert str(status.renderable) == "2 imported · 1 skipped (duplicate name)"


@pytest.mark.asyncio
async def test_prompts_canvas_import_row_browse_button_shares_toolbar_with_run_cancel():
    """Task 8b D4: Browse… reuses ``.library-canvas-action`` and shares the
    same fixed-width-only ``ds-toolbar`` as Import/Cancel (no per-id CSS
    block needed -- see the canvas family's render-safe-shape docstring)."""
    app = _CanvasHost(_three_row_state(), import_open=True)
    async with app.run_test() as pilot:
        browse_button = pilot.app.query_one("#library-prompts-import-browse", Button)
        run_button = pilot.app.query_one("#library-prompts-import-run", Button)
        assert browse_button.parent is run_button.parent
        assert browse_button.parent.has_class("ds-toolbar")
        assert browse_button.has_class("library-canvas-action")


@pytest.mark.asyncio
async def test_prompts_canvas_import_path_input_is_not_packed_into_a_toolbar_row():
    """The path Input is its own full-width sibling, NOT packed into a
    ``Horizontal`` alongside the Run/Cancel Buttons -- this canvas family's
    documented non-rendering failure mode is a ``Horizontal`` mixing a 1fr
    Input with fixed-width compact Buttons (see
    ``LibraryIngestCanvas``'s docstring)."""
    app = _CanvasHost(_three_row_state(), import_open=True)
    async with app.run_test() as pilot:
        path_input = pilot.app.query_one("#library-prompts-import-path", Input)
        assert path_input.parent is not None
        assert not path_input.parent.has_class("ds-toolbar")
        run_button = pilot.app.query_one("#library-prompts-import-run", Button)
        assert run_button.parent is not None
        assert run_button.parent.has_class("ds-toolbar")


# ---------------------------------------------------------------------------
# Screen-wiring unit tests (direct-method style, mirrors
# test_library_export_cancel.py)
# ---------------------------------------------------------------------------


def test_build_library_prompts_state_reads_browse_result_not_sampled_source():
    result = _browse_result(
        items=[
            {
                "id": "local:prompt:result-2",
                "local_id": 2,
                "name": "Browse result",
                "backend": "local",
            }
        ]
    )
    fake = SimpleNamespace(
        _local_source_records={
            "prompts": (
                99,
                (
                    {
                        "id": 999,
                        "name": "Sampled row that must never render",
                    },
                ),
            )
        },
        _library_prompt_browse_controller=SimpleNamespace(result=result),
    )

    state = LibraryScreen._build_library_prompts_state(fake)

    assert state.count == 1
    assert [(row.prompt_id, row.name) for row in state.rows] == [(2, "Browse result")]


def test_build_library_prompts_state_uses_loading_result_without_source_lookup():
    loading = begin_prompt_browse(PromptBrowseScope(), request_token=2)
    fake = SimpleNamespace(
        _local_source_records={"prompts": (500, ({"id": 999},))},
        _library_prompt_browse_controller=SimpleNamespace(result=loading),
    )

    state = LibraryScreen._build_library_prompts_state(fake)

    assert state.rows == ()
    assert state.count == 0


def test_handle_library_prompts_sort_opens_the_choice_strip():
    """task-14902: the sort press no longer cycles the scope -- it toggles
    the direct-pick strip (the pick handler owns the exact-scope request;
    see Tests/UI/test_library_choice_strips.py)."""
    calls: list[PromptBrowseScope] = []
    fake = SimpleNamespace(
        _library_prompt_browse_controller=SimpleNamespace(scope=PromptBrowseScope()),
        _request_library_prompts_browse=lambda scope, **_kwargs: calls.append(scope),
        _library_prompts_sort_choices_visible=False,
        refresh=lambda recompose=False: None,
        call_after_refresh=lambda *args, **kwargs: None,
        _focus_library_control=lambda selector: None,
        _focus_library_choice_strip_active=lambda selector, active: None,
    )
    event = SimpleNamespace(stop=lambda: None)
    LibraryScreen.handle_library_prompts_sort(fake, event)
    assert fake._library_prompts_sort_choices_visible is True
    assert calls == []
    LibraryScreen.handle_library_prompts_sort(fake, event)
    assert fake._library_prompts_sort_choices_visible is False
    assert calls == []


def test_handle_library_prompts_filter_submitted_flushes_debounce_once():
    calls: list[str] = []
    fake = SimpleNamespace(
        _flush_library_prompts_search=lambda query: calls.append(query),
    )
    event = SimpleNamespace(value="plan", stop=lambda: None)
    LibraryScreen.handle_library_prompts_filter(fake, event)
    assert calls == ["plan"]


## ``test_handle_library_prompt_row_records_selected_id`` (the old
## recording-only, no-editor behavior) was superseded by Task 4, which
## upgrades this handler to open the in-canvas editor -- see the
## ``handle_library_prompt_row`` end-to-end coverage in the "Task 4" section
## at the bottom of this file, which exercises the real handler (row press
## -> editor opens -> fields populated) through a mounted screen instead of
## a bare ``SimpleNamespace`` stand-in.


# ---------------------------------------------------------------------------
# Real end-to-end integration test
# ---------------------------------------------------------------------------


class _FakePromptScopeServiceWithList:
    """Prompt-scope fake proving browse is isolated from the sampled list seam."""

    def __init__(self, prompts, *, browse_failures: int = 0):
        self._prompts = prompts
        self._browse_failures = browse_failures
        self.browse_calls: list[dict[str, Any]] = []
        self.browse_threads: list[int] = []
        self.list_calls = 0

    async def count_prompts(self, *, mode="local", **kwargs):
        return len(self._prompts)

    async def list_prompts(self, *, mode="local", page=1, per_page=10, **kwargs):
        self.list_calls += 1
        raise AssertionError("Library Prompt rows must not use list_prompts")

    async def browse_prompts(
        self,
        *,
        mode="local",
        query="",
        collection_id=None,
        sort_by="last_modified",
        sort_order="desc",
        page=1,
        page_size=50,
    ):
        self.browse_threads.append(threading.get_ident())
        self.browse_calls.append(
            {
                "mode": mode,
                "query": query,
                "collection_id": collection_id,
                "sort_by": sort_by,
                "sort_order": sort_order,
                "page": page,
                "page_size": page_size,
            }
        )
        if self._browse_failures:
            self._browse_failures -= 1
            raise RuntimeError("temporary browse failure")
        matching = [
            prompt
            for prompt in self._prompts
            if not query
            or query.casefold()
            in f"{prompt.get('name', '')} {prompt.get('details', '')}".casefold()
        ]
        matching.sort(
            key=(
                (lambda prompt: str(prompt.get("name") or "").casefold())
                if sort_by == "name"
                else (lambda prompt: str(prompt.get("last_modified") or ""))
            ),
            reverse=sort_order == "desc",
        )
        total_items = len(matching)
        total_pages = (total_items + page_size - 1) // page_size if total_items else 0
        current_page = min(page, max(1, total_pages))
        start = (current_page - 1) * page_size
        items = [
            {
                "id": f"local:prompt:{prompt['id']}",
                "backend": "local",
                "local_id": prompt["id"],
                "name": prompt["name"],
                "author": prompt.get("author"),
                "details": prompt.get("details"),
                "keywords": prompt.get("keywords") or [],
                "last_modified": prompt.get("last_modified"),
                "version": prompt.get("version"),
            }
            for prompt in matching[start : start + page_size]
        ]
        return {
            "items": items,
            "total_pages": total_pages,
            "current_page": current_page,
            "total_items": total_items,
            "page": current_page,
            "per_page": page_size,
        }


@pytest.mark.asyncio
async def test_library_shell_prompts_row_press_renders_list_canvas():
    """Prompt rows come only from exact browse, whose local work is off-loop."""
    ui_thread = threading.get_ident()
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    prompt_service = _FakePromptScopeServiceWithList(
        [
            {
                "id": 5,
                "name": "Summarize",
                "author": "Alice",
                "last_modified": "2026-07-01T00:00:00+00:00",
            },
            {
                "id": 6,
                "name": "Translate",
                "author": "Bob",
                "last_modified": "2026-07-02T00:00:00+00:00",
            },
        ]
    )
    app.prompt_scope_service = prompt_service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_PROMPTS
        canvas = screen.query_one("#library-prompts-canvas", LibraryPromptsListCanvas)
        assert canvas is not None
        assert screen.query_one("#library-prompt-row-5", Button)
        assert screen.query_one("#library-prompt-row-6", Button)
        assert prompt_service.list_calls == 0
        assert prompt_service.browse_calls == [
            {
                "mode": "local",
                "query": "",
                "collection_id": None,
                "sort_by": "last_modified",
                "sort_order": "desc",
                "page": 1,
                "page_size": 50,
            }
        ]
        assert prompt_service.browse_threads
        assert set(prompt_service.browse_threads) == {prompt_service.browse_threads[0]}
        assert prompt_service.browse_threads[0] != ui_thread


@pytest.mark.asyncio
async def test_library_prompts_restored_create_row_list_dispatches_browse_once():
    """A restored create-row/list state settles one exact browse request."""
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    service = _FakePromptScopeServiceWithList([{"id": 5, "name": "Restored prompt"}])
    app.prompt_scope_service = service

    original = LibraryScreen(app)
    original._library_selected_row_id = LIBRARY_ROW_CREATE_PROMPT
    original._library_prompts_view = "list"
    saved_state = original.save_state()
    restored = LibraryScreen(app)
    restored.restore_state(saved_state)
    host = LibraryHarness(app, screen=restored)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")
        await pilot.pause(0.1)

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_PROMPT
        assert screen._library_prompts_view == "list"
        assert screen._library_prompt_browse_controller.result.status == "ready"
        assert service.browse_calls == [
            {
                "mode": "local",
                "query": "",
                "collection_id": None,
                "sort_by": "last_modified",
                "sort_order": "desc",
                "page": 1,
                "page_size": 50,
            }
        ]


@pytest.mark.asyncio
async def test_library_prompts_enter_flushes_debounce_without_duplicate_call():
    """Enter dispatches the pending token once and cancels its timer."""
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    service = _FakePromptScopeServiceWithList(
        [{"id": 5, "name": "Summarize", "details": "Summarize a plan"}]
    )
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")

        screen._queue_library_prompts_search("plan")
        assert screen._library_prompts_debounce_timer is not None
        screen._flush_library_prompts_search("plan")
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")
        await pilot.pause(0.35)

        assert [call["query"] for call in service.browse_calls] == ["", "plan"]
        assert screen._library_prompts_debounce_timer is None


@pytest.mark.asyncio
async def test_library_prompts_retry_recovers_service_error():
    """The explicit Retry control reruns the same scope with a fresh token."""
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    service = _FakePromptScopeServiceWithList(
        [{"id": 5, "name": "Recovered"}],
        browse_failures=1,
    )
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompts-retry")
        first_token = screen._library_prompt_browse_controller.result.request_token

        screen.query_one("#library-prompts-retry", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")

        assert len(service.browse_calls) == 2
        assert (
            screen._library_prompt_browse_controller.result.request_token > first_token
        )
        assert screen._library_prompt_browse_controller.result.status == "ready"


@pytest.mark.asyncio
async def test_library_prompts_browse_failure_keeps_exception_details_out_of_logs():
    """A failure logs only its fixed operation and exception category."""
    secret = "TASK198_SECRET_PROMPT_BROWSE_PAYLOAD"

    class SecretFailingPromptService(_FakePromptScopeServiceWithList):
        async def browse_prompts(self, **_kwargs: Any) -> dict[str, Any]:
            raise RuntimeError(secret)

    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = SecretFailingPromptService([{"id": 5, "name": "X"}])
    host = LibraryHarness(app)
    logged: list[str] = []
    sink_id = logger.add(
        lambda message: logged.append(str(message)),
        filter=lambda record: (
            record["name"]
            == "tldw_chatbook.UI.Library_Modules.library_prompt_browse_controller"
        ),
    )

    try:
        async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            screen.query_one("#library-row-browse-prompts").press()
            error = await _wait_for_selector(screen, pilot, "#library-prompts-error")

            assert str(error.renderable) == (
                "Couldn't load prompts. Check the local Library and retry."
            )
            assert screen.query_one("#library-prompts-retry", Button)
    finally:
        logger.remove(sink_id)

    joined_logs = "".join(logged)
    assert secret not in joined_logs
    assert "Traceback" not in joined_logs
    assert "Library Prompt browse failed; operation=browse_prompts" in joined_logs
    assert "exception_type=RuntimeError" in joined_logs


@pytest.mark.asyncio
@pytest.mark.parametrize("next_page", [1, 2], ids=["surviving-row", "fallback"])
async def test_library_prompts_result_restores_row_or_toolbar_focus(next_page: int):
    """Settle to a surviving row, or the stable sort control if it vanished."""
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    service = _FakePromptScopeServiceWithList(
        [
            {
                "id": index,
                "name": f"Prompt {index:02d}",
                "last_modified": "2026-07-01T00:00:00+00:00",
            }
            for index in range(1, 52)
        ]
    )
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-1")
        row = screen.query_one("#library-prompt-row-1", Button)
        row.focus()

        screen._request_library_prompts_browse(
            replace(screen._library_prompt_browse_controller.scope, page=next_page),
            focus_identity="library-prompt-row-1",
        )
        selector = (
            "#library-prompt-row-1" if next_page == 1 else "#library-prompt-row-51"
        )
        await _wait_for_selector(screen, pilot, selector)
        await pilot.pause()

        expected_focus = (
            "library-prompt-row-1" if next_page == 1 else "library-prompts-sort"
        )
        assert screen.focused is not None
        assert screen.focused.id == expected_focus


@pytest.mark.asyncio
async def test_library_shell_prompts_row_secondary_line_shows_details_not_author():
    """Task 8b D2/U1: the list row's secondary line surfaces the prompt's
    PURPOSE (``details``) instead of ``author · age`` -- exercises the full
    exact-browse pipeline: normalized service records carry ``details``
    through, and the pure state projection uses it instead of ``author``."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    app.prompt_scope_service = _FakePromptScopeServiceWithList(
        [
            {
                "id": 5,
                "name": "Summarize",
                "author": "Alice",
                "details": "Summarizes text",
                "last_modified": "2026-07-01T00:00:00+00:00",
            },
        ]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-row-5")

        button = screen.query_one("#library-prompt-row-5", Button)
        label_text = str(button.label)
        assert "Summarizes text" in label_text
        assert "Alice" not in label_text


@pytest.mark.asyncio
async def test_library_real_recipe_list_pipeline_preserves_type_source_and_lanes(
    tmp_path,
):
    """Normalized list metadata must survive the Screen's raw-row adapter."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Outcome recipe",
        author="Author",
        details="Reusable outcome structure",
        system_prompt="# Role\n\nBe exact.",
        user_prompt="# Goal\n\nShip it.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition={
            "kind": "block_recipe",
            "schema_version": 2,
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Role",
                            "syntax": "markdown",
                            "content": "Be exact.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "markdown",
                            "content": "Ship it.",
                        }
                    ],
                },
            ],
        },
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-prompts").press()
        await _wait_for_selector(screen, pilot, f"#library-prompt-row-{prompt_id}")

        button = screen.query_one(f"#library-prompt-row-{prompt_id}", Button)
        assert "Recipe · Local · System + User" in str(button.label)
        [record] = screen._library_prompt_browse_controller.result.items
        assert record["id"] == f"local:prompt:{prompt_uuid}"
        assert record["local_id"] == prompt_id
        assert record["source_id"] == prompt_uuid
        assert record["uuid"] == prompt_uuid
        assert record["version"] == 1
        assert record["backend"] == "local"
        assert record["artifact_type"] == "recipe"
        assert record["has_system_prompt"] is True
        assert record["has_user_prompt"] is True


# ---------------------------------------------------------------------------
# Stylesheet parity pin (review finding: the canvas's ids/classes had no
# stylesheet rules at all, so prompt rows silently rendered as auto-width
# default Buttons instead of matching the sibling notes list's look).
# ---------------------------------------------------------------------------


def test_library_prompt_row_class_matches_notes_row_visual_parity():
    """``.library-prompt-row`` (the row Buttons in
    ``library_prompts_canvas.py``) must have a stylesheet block, with the
    same width/height/border/background as ``.library-notes-row`` -- visual
    parity with the sibling notes list, not default auto-width Buttons."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert ".library-prompt-row {" in text
        prompt_row_block = _css_block(text, ".library-prompt-row {")
        notes_row_block = _css_block(text, ".library-notes-row {")
        for pinned in (
            "width: 100%;",
            "height: 2;",
            "border: none;",
            "background: $ds-surface-panel;",
        ):
            assert pinned in prompt_row_block
            assert pinned in notes_row_block


def test_library_prompts_header_filter_empty_have_css_blocks():
    """``#library-prompts-header``/``#library-prompts-filter``
    (+ ``:focus``)/``#library-prompts-empty`` (``library_prompts_canvas.py``)
    must have stylesheet rules matching their ``#library-notes-*`` siblings,
    instead of silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompts-header {" in text
        assert "#library-prompts-filter {" in text
        assert "#library-prompts-filter:focus {" in text
        assert "#library-prompts-empty {" in text

        header_block = _css_block(text, "#library-prompts-header {")
        notes_header_block = _css_block(text, "#library-notes-header {")
        assert "height: auto;" in header_block
        assert "height: auto;" in notes_header_block

        filter_block = _css_block(text, "#library-prompts-filter {")
        notes_filter_block = _css_block(text, "#library-notes-filter {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in filter_block
            assert pinned in notes_filter_block

        focus_block = _css_block(text, "#library-prompts-filter:focus {")
        notes_focus_block = _css_block(text, "#library-notes-filter:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block
            assert pinned in notes_focus_block

        empty_block = _css_block(text, "#library-prompts-empty {")
        notes_empty_block = _css_block(text, "#library-notes-empty {")
        assert "color: $ds-text-muted;" in empty_block
        assert "color: $ds-text-muted;" in notes_empty_block


def test_library_prompt_editor_field_css_blocks_match_notes_editor_parity():
    """Editor field ids introduced by Task 4 (name/author/details/keywords
    Inputs, system/user TextAreas, meta line, conflict/status Statics) must
    have stylesheet rules matching their ``#library-note-*`` siblings."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompt-name," in text
        assert "#library-prompt-keywords {" in text
        input_block = _css_block(text, "#library-prompt-keywords {")
        note_input_block = _css_block(text, "#library-note-keywords {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in input_block
            assert pinned in note_input_block

        assert "#library-prompt-name:focus," in text
        assert "#library-prompt-keywords:focus {" in text
        focus_block = _css_block(text, "#library-prompt-keywords:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block

        assert ".library-prompt-field-label {" in text
        label_block = _css_block(text, ".library-prompt-field-label {")
        assert "color: $ds-text-muted;" in label_block

        assert "#library-prompt-system," in text
        assert "#library-prompt-user {" in text
        textarea_block = _css_block(text, "#library-prompt-user {")
        assert "min-height: 6;" in textarea_block
        assert "max-height: 14;" in textarea_block

        assert "#library-prompt-meta {" in text
        meta_block = _css_block(text, "#library-prompt-meta {")
        assert "color: $ds-text-muted;" in meta_block

        assert "#library-prompt-conflict-copy," in text
        assert "#library-prompt-save-status {" in text
        status_block = _css_block(text, "#library-prompt-save-status {")
        assert "color: $ds-text-muted;" in status_block


def test_library_prompt_field_hint_css_block_matches_field_label_parity():
    """U7 (Task 8c): ``.library-prompt-field-hint`` (the one-line dim hint
    under the System/User prompt labels) must have a stylesheet rule, same
    muted tier as its ``.library-prompt-field-label`` sibling -- instead of
    silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert ".library-prompt-field-hint {" in text
        hint_block = _css_block(text, ".library-prompt-field-hint {")
        label_block = _css_block(text, ".library-prompt-field-label {")
        assert "color: $ds-text-muted;" in hint_block
        assert "color: $ds-text-muted;" in label_block


def test_library_prompts_import_row_css_blocks_match_filter_status_parity():
    """Toolbar Import… row ids introduced by Task 5 (the path Input, its
    outcome Static) must have stylesheet rules matching their
    ``#library-prompts-filter``/``#library-prompt-save-status`` siblings,
    instead of silently falling back to unstyled defaults."""
    agentic_terminal = AGENTIC_TERMINAL.read_text(encoding="utf-8")
    bundled_stylesheet = BUNDLED_STYLESHEET.read_text(encoding="utf-8")

    for text in (agentic_terminal, bundled_stylesheet):
        assert "#library-prompts-import-path {" in text
        assert "#library-prompts-import-path:focus {" in text
        assert "#library-prompts-import-status {" in text

        input_block = _css_block(text, "#library-prompts-import-path {")
        filter_block = _css_block(text, "#library-prompts-filter {")
        for pinned in (
            "height: 3;",
            "border: tall $ds-grid-line;",
            "background: $ds-surface-raised;",
        ):
            assert pinned in input_block
            assert pinned in filter_block

        focus_block = _css_block(text, "#library-prompts-import-path:focus {")
        filter_focus_block = _css_block(text, "#library-prompts-filter:focus {")
        for pinned in ("border: tall $ds-input-focus-accent;", "outline: none;"):
            assert pinned in focus_block
            assert pinned in filter_focus_block

        status_block = _css_block(text, "#library-prompts-import-status {")
        assert "color: $ds-text-muted;" in status_block


# ---------------------------------------------------------------------------
# Task 4: editor canvas, explicit Save, conflict outcomes, delete
#
# Uses a REAL ``PromptsDatabase`` + ``PromptScopeService`` (mirroring
# ``Tests/Library/test_library_prompts_seam.py``'s Task 1 precedent)
# rather than a hand-rolled fake -- the conflict/name-collision scenarios
# below depend on the real DB's actual exception/return-value shapes
# (``Prompts.name`` is globally UNIQUE regardless of soft-delete state;
# ``update_prompt_by_id`` has no caller-supplied expected-version
# parameter, so the screen's own pre-check, exercised here, is what
# actually detects staleness -- see ``_save_library_prompt``'s docstring).
# ---------------------------------------------------------------------------


def _real_prompt_scope_service(tmp_path):
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    service = PromptScopeService(
        local_service=ScopeLocalPromptService(db), server_service=None
    )
    return db, service


class _RecordingHistoryPromptScopeService:
    """Delegate to the real scope while exposing its UI-facing history calls."""

    def __init__(
        self,
        service: PromptScopeService,
        *,
        count_failures: int = 0,
        page_failures: int = 0,
        restore_error: Exception | None = None,
        restore_result: Mapping[str, Any] | None = None,
        count_gate: threading.Event | None = None,
        page_gate: threading.Event | None = None,
        restore_gate: threading.Event | None = None,
        post_restore_detail_gate: threading.Event | None = None,
        post_restore_detail_started: threading.Event | None = None,
    ) -> None:
        self.service = service
        self.count_calls: list[str | int] = []
        self.page_calls: list[tuple[str | int, int, int | None]] = []
        self.restore_calls: list[tuple[str | int, int, int, int]] = []
        self.count_failures = count_failures
        self.page_failures = page_failures
        self.restore_error = restore_error
        self.restore_result = restore_result
        self.count_gate = count_gate
        self.page_gate = page_gate
        self.restore_gate = restore_gate
        self.post_restore_detail_gate = post_restore_detail_gate
        self.post_restore_detail_started = post_restore_detail_started

    def __getattr__(self, name: str) -> Any:
        return getattr(self.service, name)

    async def count_prompt_versions(self, **kwargs: Any) -> int:
        self.count_calls.append(kwargs["prompt_identifier"])
        if self.count_gate is not None:
            self.count_gate.wait(timeout=5)
        if self.count_failures:
            self.count_failures -= 1
            raise RuntimeError("injected count failure")
        return await self.service.count_prompt_versions(**kwargs)

    async def list_prompt_versions(self, **kwargs: Any) -> dict[str, Any]:
        self.page_calls.append(
            (
                kwargs["prompt_identifier"],
                kwargs["page_size"],
                kwargs.get("before_change_id"),
            )
        )
        if self.page_gate is not None:
            self.page_gate.wait(timeout=5)
        if self.page_failures:
            self.page_failures -= 1
            raise RuntimeError("injected page failure")
        return await self.service.list_prompt_versions(**kwargs)

    async def restore_prompt_version(self, **kwargs: Any) -> dict[str, Any]:
        self.restore_calls.append(
            (
                kwargs["prompt_identifier"],
                kwargs["change_id"],
                kwargs["version"],
                kwargs["expected_version"],
            )
        )
        if self.restore_gate is not None:
            self.restore_gate.wait(timeout=5)
        if self.restore_error is not None:
            raise self.restore_error
        if self.restore_result is not None:
            return dict(self.restore_result)
        return await self.service.restore_prompt_version(**kwargs)

    async def get_prompt(self, **kwargs: Any) -> dict[str, Any]:
        if self.restore_calls and self.post_restore_detail_gate is not None:
            if self.post_restore_detail_started is not None:
                self.post_restore_detail_started.set()
            self.post_restore_detail_gate.wait(timeout=5)
        return await self.service.get_prompt(**kwargs)


def _real_prompt_scope_service_with_production_policy_enforcer(tmp_path):
    """Like ``_real_prompt_scope_service``, but wires the real production
    runtime-policy seam instead of leaving ``policy_enforcer`` unset.

    ``_real_prompt_scope_service`` (used by every other test below) passes
    no ``policy_enforcer`` at all, so ``PromptScopeService._enforce_policy``
    short-circuits and never calls ``require_allowed`` -- that is exactly
    why the Phase-1 gate defect (clicking a prompt row raised
    ``PolicyDeniedError: Unknown runtime-policy action_id:
    prompts.detail.local``) went uncaught by every existing UI test here.

    This mirrors how ``app.py`` (~2345-2350, 2513-2517) actually builds the
    production seam: a ``ServicePolicyEnforcer`` around the real
    ``CAPABILITY_REGISTRY`` (via its default ``PolicyEngine``), fed a
    ``RuntimeSourceState`` in local mode.
    """
    db = PromptsDatabase(tmp_path / "prompts.db", client_id="test-client")
    policy_enforcer = ServicePolicyEnforcer(
        state_provider=lambda: RuntimeSourceState(active_source="local"),
    )
    service = PromptScopeService(
        local_service=ScopeLocalPromptService(db),
        server_service=None,
        policy_enforcer=policy_enforcer,
    )
    return db, service


def _wire_empty_non_prompt_services(app) -> None:
    app.notes_scope_service = StaticLibraryNotesListScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])


async def _open_prompt_editor(screen, pilot, prompt_id: int) -> None:
    """Open the rail's Prompts row, then a specific prompt's row."""
    screen.query_one("#library-row-browse-prompts").press()
    row = await _wait_for_selector(
        screen,
        pilot,
        f"#library-prompt-row-{prompt_id}",
    )
    assert isinstance(row, Button)
    row.press()
    await pilot.pause()
    for _ in range(150):
        if screen._library_prompt_detail is not None:
            break
        await pilot.pause(0.02)
    await pilot.pause()


async def _wait_for_prompt_status(screen, pilot, *, attempts=150) -> str:
    status_text = ""
    for _ in range(attempts):
        status_text = str(screen.query_one("#library-prompt-save-status").renderable)
        if status_text:
            return status_text
        await pilot.pause(0.02)
    return status_text


@pytest.mark.asyncio
async def test_library_prompt_history_count_is_index_only_and_first_page_is_lazy(
    tmp_path,
):
    """Opening a saved local Prompt counts first and reads no page until expand."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Retained",
        author="A",
        details="v1",
        system_prompt="[bold]literal system[/bold]",
        user_prompt="[/] literal user [wip]",
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    service = _RecordingHistoryPromptScopeService(real_service)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        disclosure = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-collapsible"
        )
        for _ in range(150):
            if str(disclosure.title) == "Retained history (2)":
                break
            await pilot.pause(0.02)

        assert disclosure.collapsed is True
        assert str(disclosure.title) == "Retained history (2)"
        assert service.count_calls == [prompt_uuid]
        assert service.page_calls == []
        assert len(screen.query(".library-prompt-history-row")) == 0

        disclosure.collapsed = False
        await pilot.pause()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)

        assert len(service.page_calls) == 1
        assert service.page_calls[0] == (prompt_uuid, 10, None)
        assert [
            row.source_version for row in screen.query(".library-prompt-history-row")
        ] == [2, 1]

        oldest = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        oldest.press()
        await pilot.pause()
        assert screen.query_one("#library-prompt-history-system", TextArea).text == (
            "[bold]literal system[/bold]"
        )
        assert screen.query_one("#library-prompt-history-user", TextArea).text == (
            "[/] literal user [wip]"
        )
        assert screen.query_one("#library-prompt-history-system", TextArea).read_only
        assert screen.query_one("#library-prompt-history-user", TextArea).read_only
        assert len(screen.query(PromptBlockEditor)) == 1

        screen.refresh(recompose=True)
        await pilot.pause()
        remounted = screen.query_one("#library-prompt-history-collapsible", Collapsible)
        assert remounted.collapsed is False
        assert screen.query_one("#library-prompt-history-system", TextArea).text == (
            "[bold]literal system[/bold]"
        )
        assert len(service.page_calls) == 1

        screen.query_one("#library-prompt-author", Input).value = "Dirty"
        await pilot.pause()
        assert screen._library_prompt_dirty is True
        assert len(screen.query(".library-prompt-history-row")) == 2
        assert screen.query_one("#library-prompt-history-restore", Button).disabled
        assert (
            str(
                screen.query_one(
                    "#library-prompt-history-restore-reason", Static
                ).renderable
            )
            == "Save or discard unsaved changes before restoring retained history."
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("snapshot_kind", "reason", "system_preview", "user_preview"),
    [
        (
            "legacy-recipe",
            "Legacy Recipe snapshots are preview-only.",
            "[bold]literal legacy system[/bold]",
            "literal legacy user",
        ),
        (
            "over-limit-structured",
            "This retained version is not supported by current local Prompt "
            "capabilities.",
            "",
            "literal structured user",
        ),
    ],
)
async def test_library_prompt_history_previews_nonrestorable_local_snapshots_without_mutation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    snapshot_kind: str,
    reason: str,
    system_preview: str,
    user_preview: str,
) -> None:
    capabilities = replace(local_prompt_capabilities(), compiled_lane_limit=4)
    monkeypatch.setattr(
        prompt_scope_service_module,
        "local_prompt_capabilities",
        lambda: capabilities,
    )
    db, service = _real_prompt_scope_service(tmp_path)
    if snapshot_kind == "legacy-recipe":
        prompt_id, prompt_uuid, _message = db.add_prompt(
            name="History compatibility",
            author="A",
            details="legacy recipe",
            system_prompt=system_preview,
            user_prompt=user_preview,
            prompt_format="legacy",
            artifact_type="recipe",
        )
    else:
        definition = {
            "schema_version": 2,
            "kind": "block_prompt",
            "lanes": [
                {"id": "system", "blocks": []},
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "user-1",
                            "title": "User",
                            "syntax": "freeform",
                            "content": user_preview,
                        }
                    ],
                },
            ],
        }
        prompt_id, prompt_uuid, _message = db.add_prompt(
            name="History compatibility",
            author="A",
            details="structured over limit",
            system_prompt=system_preview,
            user_prompt=user_preview,
            prompt_format="structured",
            prompt_schema_version=2,
            prompt_definition=definition,
            artifact_type="prompt",
        )
    db.update_prompt_by_id(
        prompt_id,
        {
            "details": "current Prompt",
            "system_prompt": "current system",
            "user_prompt": "current user",
            "prompt_format": "legacy",
            "prompt_schema_version": None,
            "prompt_definition": None,
            "artifact_type": "prompt",
        },
        expected_version=1,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        disclosure = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-collapsible"
        )
        disclosure.collapsed = False
        for _ in range(150):
            rows = list(screen.query(".library-prompt-history-row"))
            if len(rows) == 2:
                break
            await pilot.pause(0.02)
        source = next(row for row in rows if row.source_version == 1)
        source.press()
        for _ in range(150):
            if screen.query("#library-prompt-history-compatibility"):
                break
            await pilot.pause(0.02)

        assert screen.query_one("#library-prompt-history-system", TextArea).text == (
            system_preview
        )
        assert screen.query_one("#library-prompt-history-user", TextArea).text == (
            user_preview
        )
        assert (
            str(
                screen.query_one(
                    "#library-prompt-history-compatibility", Static
                ).renderable
            )
            == reason
        )
        assert screen.query_one("#library-prompt-history-restore", Button).disabled
        assert (
            str(
                screen.query_one(
                    "#library-prompt-history-restore-reason", Static
                ).renderable
            )
            == reason
        )

        before_detail = db.fetch_prompt_details(prompt_uuid)
        before_count = db.get_prompt_history_count(prompt_uuid)
        history_source = next(
            row
            for row in screen._library_prompt_history_controller.state.rows
            if row.version == 1
        )
        with pytest.raises(PromptRestoreError) as exc_info:
            await service.restore_prompt_version(
                mode="local",
                prompt_identifier=prompt_uuid,
                change_id=history_source.change_id,
                version=history_source.version,
                expected_version=before_detail["version"],
            )

        assert exc_info.value.code is PromptRestoreErrorCode.VALIDATION
        assert db.fetch_prompt_details(prompt_uuid) == before_detail
        assert db.get_prompt_history_count(prompt_uuid) == before_count


@pytest.mark.asyncio
async def test_library_prompt_history_title_starts_with_ellipsis_before_count_settles(
    tmp_path,
):
    """The collapsed disclosure is truthful while the scalar worker is pending."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Pending count", author="", details="", user_prompt="v1"
    )
    count_gate = threading.Event()
    service = _RecordingHistoryPromptScopeService(real_service, count_gate=count_gate)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(80, 24)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            disclosure = screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            )

            assert str(disclosure.title) == "Retained history (…)"
            assert disclosure.collapsed is True
            assert service.count_calls == [prompt_uuid]
            assert service.page_calls == []

            count_gate.set()
            for _ in range(150):
                disclosure = screen.query_one(
                    "#library-prompt-history-collapsible", Collapsible
                )
                if str(disclosure.title) == "Retained history (1)":
                    break
                await pilot.pause(0.02)
            assert str(disclosure.title) == "Retained history (1)"
    finally:
        count_gate.set()


@pytest.mark.asyncio
async def test_library_prompt_history_collapse_resets_page_without_refetching_count(
    tmp_path,
):
    """Close uses the pure reset while preserving the already-settled count."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Collapse retained", author="", details="", user_prompt="v1"
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    service = _RecordingHistoryPromptScopeService(real_service)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        disclosure = screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        disclosure.collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        screen.query_one("#library-prompt-history-row-1", Button).press()
        await pilot.pause()
        assert screen._library_prompt_history_state.selected is not None

        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = True
        for _ in range(100):
            state = screen._library_prompt_history_state
            if state is not None and not state.is_open and not state.rows:
                break
            await pilot.pause(0.02)
        state = screen._library_prompt_history_state
        assert state is not None
        assert state.prompt_uuid == prompt_uuid
        assert state.retained_count == 2
        assert state.count_status == "loaded"
        assert state.rows == ()
        assert state.selected is None
        assert state.page_request is None
        assert service.count_calls == [prompt_uuid]

        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        assert service.count_calls == [prompt_uuid]
        assert len(service.page_calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("late_kind", ["count", "page", "restore"])
async def test_library_prompt_history_late_results_after_prompt_switch_are_noops(
    tmp_path,
    late_kind: str,
):
    """Prompt UUID + scope + request guards reject every late worker class."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    first_id, first_uuid, _msg = db.add_prompt(
        name="First retained", author="", details="", user_prompt="first v1"
    )
    db.update_prompt_by_id(first_id, {"details": "first v2"}, expected_version=1)
    second_id, second_uuid, _msg = db.add_prompt(
        name="Second retained", author="", details="", user_prompt="second"
    )
    gate = threading.Event()
    service = _RecordingHistoryPromptScopeService(
        real_service,
        count_gate=gate if late_kind == "count" else None,
        page_gate=gate if late_kind == "page" else None,
        restore_gate=gate if late_kind == "restore" else None,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, first_id)
            if late_kind in {"page", "restore"}:
                screen.query_one(
                    "#library-prompt-history-collapsible", Collapsible
                ).collapsed = False
                for _ in range(150):
                    if late_kind == "page" and service.page_calls:
                        break
                    if (
                        late_kind == "restore"
                        and len(screen.query(".library-prompt-history-row")) == 2
                    ):
                        break
                    await pilot.pause(0.02)
            if late_kind == "restore":
                row = next(
                    row
                    for row in screen.query(".library-prompt-history-row")
                    if row.source_version == 1
                )
                row.press()
                await pilot.pause()
                screen.query_one("#library-prompt-history-restore", Button).press()
                await pilot.pause()
                assert isinstance(host.screen, ConfirmationDialog)
                host.screen.query_one("#confirm-button", Button).press()
                for _ in range(100):
                    if service.restore_calls:
                        break
                    await pilot.pause(0.02)
                assert service.restore_calls
            elif late_kind == "count":
                for _ in range(100):
                    if service.count_calls:
                        break
                    await pilot.pause(0.02)
                assert service.count_calls == [first_uuid]
            else:
                assert service.page_calls

            for _ in range(100):
                if host.screen is screen:
                    break
                await pilot.pause(0.02)
            screen.query_one("#library-prompt-back", Button).press()
            await _wait_for_selector(screen, pilot, f"#library-prompt-row-{second_id}")
            screen.query_one(f"#library-prompt-row-{second_id}", Button).press()
            gate.set()
            for _ in range(200):
                state = screen._library_prompt_history_state
                if (
                    screen._selected_prompt_id == second_id
                    and isinstance(screen._library_prompt_detail, Mapping)
                    and screen._library_prompt_detail.get("uuid") == second_uuid
                    and state is not None
                    and state.prompt_uuid == second_uuid
                    and state.count_status == "loaded"
                ):
                    break
                await pilot.pause(0.02)

            state = screen._library_prompt_history_state
            assert screen._selected_prompt_id == second_id
            assert state is not None
            assert state.prompt_uuid == second_uuid
            assert state.rows == ()
            assert state.selected is None
            assert screen._library_prompt_version == 1
            for _ in range(100):
                disclosures = list(screen.query("#library-prompt-history-collapsible"))
                if not disclosures:
                    await pilot.pause(0.02)
                    continue
                disclosure = disclosures[0]
                if str(disclosure.title) == "Retained history (1)":
                    break
                await pilot.pause(0.02)
            assert disclosures
            assert str(disclosure.title) == "Retained history (1)"
    finally:
        gate.set()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "action",
    ["close", "reload", "page", "restore", "select"],
)
async def test_queued_outgoing_history_action_is_noop_after_prompt_adoption(
    tmp_path,
    action: str,
):
    """A queued semantic action cannot cross the immutable Prompt scope."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    first_id, _first_uuid, _message = db.add_prompt(
        name="Outgoing history", author="", details="v1", user_prompt="first"
    )
    db.update_prompt_by_id(first_id, {"details": "v2"}, expected_version=1)
    second_id, second_uuid, _message = db.add_prompt(
        name="Current history", author="", details="v1", user_prompt="second"
    )
    db.update_prompt_by_id(second_id, {"details": "v2"}, expected_version=1)
    second_detail = await real_service.get_prompt(
        mode="local", prompt_identifier=second_id, include_deleted=True
    )
    second_page_payload = await real_service.list_prompt_versions(
        mode="local",
        prompt_identifier=second_uuid,
        page_size=10,
        before_change_id=None,
    )
    service = _RecordingHistoryPromptScopeService(real_service)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, first_id)
        disclosure = screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        disclosure.collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        first_row = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        first_row.press()
        await pilot.pause()
        outgoing_state = screen._library_prompt_history_state
        assert outgoing_state is not None
        scoped = {
            "prompt_uuid": outgoing_state.prompt_uuid,
            "scope_token": outgoing_state.scope_token,
        }
        if action == "close":
            stale_message = LibraryPromptHistoryRegion.DisclosureClosed(**scoped)
        elif action == "reload":
            stale_message = LibraryPromptHistoryRegion.ReloadRequested(**scoped)
        elif action == "page":
            stale_message = LibraryPromptHistoryRegion.PageRequested(**scoped)
        elif action == "restore":
            stale_message = LibraryPromptHistoryRegion.RestoreRequested(**scoped)
        else:
            stale_message = LibraryPromptHistoryRegion.RowSelected(
                **scoped,
                change_id=first_row.change_id,
                source_version=first_row.source_version,
            )

        outgoing_region = screen.query_one(LibraryPromptHistoryRegion)
        screen._adopt_library_prompt_persisted_detail(
            second_detail,
            open_history=True,
        )
        adopted = screen._library_prompt_history_state
        assert adopted is not None
        current = build_prompt_history_state(
            prompt_uuid=second_uuid,
            current_version=2,
            scope_token=adopted.scope_token,
        )
        current, count_request = begin_prompt_history_count(
            current, request_token=10_000
        )
        current = apply_prompt_history_count(current, count_request, total_count=2)
        current, page_request = begin_prompt_history_page(current, request_token=10_001)
        current = apply_prompt_history_page(
            current,
            page_request,
            build_prompt_history_page(second_page_payload),
        )
        second_row = next(row for row in current.rows if row.version == 1)
        current, preview_request = begin_prompt_history_preview(
            current,
            change_id=second_row.change_id,
            source_version=second_row.version,
            request_token=10_002,
        )
        current = apply_prompt_history_preview(current, preview_request)
        current = replace(
            current,
            has_more=True,
            next_before_change_id=current.rows[-1].change_id,
        )
        screen._library_prompt_history_controller._publish(current)
        for _ in range(150):
            if any(call[0] == second_uuid for call in service.page_calls):
                break
            await pilot.pause(0.02)
        before = screen._library_prompt_history_state
        page_calls = list(service.page_calls)
        restore_calls = list(service.restore_calls)

        outgoing_region.post_message(stale_message)
        await pilot.pause()
        await pilot.pause()

        assert host.screen is screen
        assert screen._selected_prompt_id == second_id
        assert screen._library_prompt_history_state == before
        assert screen._library_prompt_history_state.prompt_uuid == second_uuid
        assert service.page_calls == page_calls
        assert service.restore_calls == restore_calls


@pytest.mark.asyncio
@pytest.mark.parametrize("late_kind", ["count", "page"])
async def test_library_prompt_history_late_load_is_noop_after_duplicate_detaches(
    tmp_path,
    late_kind: str,
):
    """A saved-to-unsaved transition invalidates the source worker scope."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Detach while paging",
        author="Author",
        details="v1",
        user_prompt="User",
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    load_gate = threading.Event()
    service = _RecordingHistoryPromptScopeService(
        real_service,
        count_gate=load_gate if late_kind == "count" else None,
        page_gate=load_gate if late_kind == "page" else None,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            assert screen._library_prompt_history_state is not None
            assert screen._library_prompt_history_state.prompt_uuid == prompt_uuid
            if late_kind == "page":
                screen.query_one(
                    "#library-prompt-history-collapsible", Collapsible
                ).collapsed = False
            for _ in range(100):
                if (
                    late_kind == "count"
                    and service.count_calls
                    or late_kind == "page"
                    and service.page_calls
                ):
                    break
                await pilot.pause(0.02)
            assert service.count_calls if late_kind == "count" else service.page_calls

            screen.query_one("#library-prompt-duplicate", Button).press()
            await pilot.pause()
            assert screen._selected_prompt_id is None
            assert screen._library_prompt_history_state is None
            assert len(screen.query("#library-prompt-history-collapsible")) == 0
            assert "uuid" not in screen._library_prompt_detail

            load_gate.set()
            for _ in range(100):
                await pilot.pause(0.02)
            assert screen._library_prompt_history_state is None
            assert len(screen.query("#library-prompt-history-collapsible")) == 0
    finally:
        load_gate.set()


@pytest.mark.asyncio
async def test_library_prompt_history_retries_count_and_page_errors(tmp_path):
    """Removing either retry leaves a real worker failure permanently stuck."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Retry retained", author="", details="", user_prompt="v1"
    )
    service = _RecordingHistoryPromptScopeService(
        real_service,
        count_failures=1,
        page_failures=1,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        disclosure = screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        for _ in range(150):
            if screen._library_prompt_history_state.count_status == "error":
                break
            await pilot.pause(0.02)

        disclosure.collapsed = False
        await pilot.pause()
        await _wait_for_selector(screen, pilot, "#library-prompt-history-retry-count")
        screen.query_one("#library-prompt-history-retry-count", Button).press()
        for _ in range(150):
            if str(disclosure.title) == "Retained history (1)":
                break
            disclosure = screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            )
            await pilot.pause(0.02)
        assert service.count_calls == [prompt_uuid, prompt_uuid]
        assert str(disclosure.title) == "Retained history (1)"

        retry_page = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-retry-page"
        )
        retry_page.press()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 1:
                break
            await pilot.pause(0.02)
        assert len(service.page_calls) == 2
        assert len(screen.query(".library-prompt-history-row")) == 1


@pytest.mark.asyncio
async def test_library_prompt_history_loads_older_pages_once_per_cursor(tmp_path):
    """Paging appends strictly older rows and rejects duplicate in-flight presses."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Paged", author="", details="", user_prompt="v1"
    )
    for version in range(1, 12):
        db.update_prompt_by_id(
            prompt_id,
            {"details": f"v{version + 1}"},
            expected_version=version,
        )
    service = _RecordingHistoryPromptScopeService(real_service)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        disclosure = screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        disclosure.collapsed = False
        await pilot.pause()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 10:
                break
            await pilot.pause(0.02)
        first_versions = [
            row.source_version for row in screen.query(".library-prompt-history-row")
        ]
        assert first_versions == list(range(12, 2, -1))

        load_older = screen.query_one("#library-prompt-history-load-older", Button)
        load_older.press()
        load_older.press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 12:
                break
            await pilot.pause(0.02)

        assert len(service.page_calls) == 2
        assert service.page_calls[0] == (prompt_uuid, 10, None)
        assert service.page_calls[1][0:2] == (prompt_uuid, 10)
        assert service.page_calls[1][2] is not None
        assert [
            row.source_version for row in screen.query(".library-prompt-history-row")
        ] == list(range(12, 0, -1))
        assert len(screen.query("#library-prompt-history-load-older")) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "type_change", [False, True], ids=["same-type", "prompt-to-recipe"]
)
async def test_library_prompt_history_confirms_and_restores_as_new_current_version(
    tmp_path,
    type_change: bool,
):
    """Restore confirmation names both versions and any artifact-type change."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_definition = {
        "kind": "block_prompt",
        "schema_version": 2,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "freeform",
                        "content": "Original",
                    }
                ],
            },
        ],
    }
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Restore source",
        author="",
        details="",
        user_prompt="Original",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=prompt_definition,
        artifact_type="prompt",
    )
    update: dict[str, Any] = {"details": "Current"}
    if type_change:
        update.update(
            {
                "artifact_type": "recipe",
                "user_prompt": "Current recipe",
                "prompt_definition": {
                    **prompt_definition,
                    "kind": "block_recipe",
                    "lanes": [
                        {"id": "system", "blocks": []},
                        {
                            "id": "user",
                            "blocks": [
                                {
                                    "id": "goal",
                                    "title": "Goal",
                                    "syntax": "freeform",
                                    "content": "Current recipe",
                                }
                            ],
                        },
                    ],
                },
            }
        )
    db.update_prompt_by_id(prompt_id, update, expected_version=1)
    service = _RecordingHistoryPromptScopeService(real_service)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.app_instance.notify = host.notify
        host._notifications.clear()
        disclosure = screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        )
        disclosure.collapsed = False
        await pilot.pause()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()

        modal = host.screen
        assert isinstance(modal, ConfirmationDialog)
        assert "retained v1" in modal.message
        assert "current v2" in modal.message
        assert "creates a new current version" in modal.message
        if type_change:
            assert "Recipe to Prompt" in modal.message
        else:
            assert "changes the artifact type" not in modal.message

        modal.query_one("#confirm-button", Button).press()
        for _ in range(200):
            restored = db.fetch_prompt_details(prompt_id)
            if (
                host.screen is screen
                and restored
                and restored["version"] == 3
                and screen._library_prompt_version == 3
                and host._notifications
            ):
                break
            await pilot.pause(0.02)
        restored = db.fetch_prompt_details(prompt_id)
        assert restored is not None
        assert restored["version"] == 3
        assert restored["artifact_type"] == "prompt"
        assert len(service.restore_calls) == 1
        assert service.restore_calls[0][0] == prompt_uuid
        assert type(service.restore_calls[0][1]) is int
        assert service.restore_calls[0][2:] == (1, 2)
        assert [notice.message for notice in host._notifications] == [
            "Restored v1 as current v3."
        ]
        assert screen._library_prompt_version == 3
        disclosure = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-collapsible"
        )
        assert disclosure.collapsed is False


@pytest.mark.asyncio
async def test_library_prompt_history_restore_stays_gated_until_detail_adoption(
    tmp_path,
):
    """A successful write cannot be restored again while detail refresh waits."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Adoption gate", author="", details="v1", user_prompt="source"
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    detail_gate = threading.Event()
    service = _RecordingHistoryPromptScopeService(
        real_service,
        post_restore_detail_gate=detail_gate,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            ).collapsed = False
            for _ in range(150):
                if len(screen.query(".library-prompt-history-row")) == 2:
                    break
                await pilot.pause(0.02)
            source = next(
                row
                for row in screen.query(".library-prompt-history-row")
                if row.source_version == 1
            )
            source.press()
            await pilot.pause()
            restore_gate = screen._library_prompt_history_controller.restore_gate(
                dirty=False
            )
            assert restore_gate is not None and restore_gate.target is not None
            target = restore_gate.target
            screen.query_one("#library-prompt-history-restore", Button).press()
            await pilot.pause()
            assert isinstance(host.screen, ConfirmationDialog)
            host.screen.query_one("#confirm-button", Button).press()
            for _ in range(150):
                state = screen._library_prompt_history_state
                if (
                    state is not None
                    and state.restore_outcome is not None
                    and state.restore_outcome.kind == "restored"
                ):
                    break
                await pilot.pause(0.02)

            state = screen._library_prompt_history_state
            assert state is not None
            assert state.restore_refresh_pending is True
            restore = await _wait_for_selector(
                screen, pilot, "#library-prompt-history-restore"
            )
            assert restore.disabled is True
            assert (
                str(
                    screen.query_one(
                        "#library-prompt-history-restore-reason", Static
                    ).renderable
                )
                == "Refreshing the restored Prompt…"
            )
            restore.press()
            screen._confirm_library_prompt_history_restore(
                True,
                prompt_uuid=target.prompt_uuid,
                change_id=target.change_id,
                source_version=target.source_version,
                expected_current_version=target.expected_current_version,
            )
            await pilot.pause(0.1)

            assert len(service.restore_calls) == 1
            assert screen._library_prompt_conflict_snapshot is None

            detail_gate.set()
            for _ in range(200):
                state = screen._library_prompt_history_state
                if (
                    screen._library_prompt_version == 3
                    and state is not None
                    and state.current_version == 3
                    and not state.restore_refresh_pending
                ):
                    break
                await pilot.pause(0.02)

            state = screen._library_prompt_history_state
            assert state is not None
            assert state.current_version == 3
            assert state.restore_refresh_pending is False
            assert len(service.restore_calls) == 1
            assert screen._library_prompt_conflict_snapshot is None
    finally:
        detail_gate.set()


@pytest.mark.asyncio
async def test_library_prompt_history_restore_refresh_rejects_same_id_aba_scope(
    tmp_path,
):
    """An old restore refresh cannot adopt into a reopened same-id editor."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="ABA refresh", author="", details="v1", user_prompt="source"
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    detail_gate = threading.Event()
    detail_started = threading.Event()
    service = _RecordingHistoryPromptScopeService(
        real_service,
        post_restore_detail_gate=detail_gate,
        post_restore_detail_started=detail_started,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            ).collapsed = False
            for _ in range(150):
                if len(screen.query(".library-prompt-history-row")) == 2:
                    break
                await pilot.pause(0.02)
            source = next(
                row
                for row in screen.query(".library-prompt-history-row")
                if row.source_version == 1
            )
            source.press()
            await pilot.pause()
            screen.query_one("#library-prompt-history-restore", Button).press()
            await pilot.pause()
            assert isinstance(host.screen, ConfirmationDialog)
            host.screen.query_one("#confirm-button", Button).press()
            for _ in range(200):
                if detail_started.is_set():
                    break
                await pilot.pause(0.02)
            assert detail_started.is_set()
            old_state = screen._library_prompt_history_state
            assert old_state is not None
            assert old_state.prompt_uuid == prompt_uuid
            old_scope_token = old_state.scope_token

            latest = await real_service.get_prompt(
                mode="local",
                prompt_identifier=prompt_id,
                include_deleted=True,
            )
            screen._reset_library_prompt_editor_state()
            screen._selected_prompt_id = prompt_id
            screen._library_selected_row_id = LIBRARY_ROW_BROWSE_PROMPTS
            screen._library_prompts_view = "editor"
            screen._adopt_library_prompt_persisted_detail(latest, open_history=False)
            screen.refresh(recompose=True)
            screen.call_after_refresh(screen._arm_library_prompt_editor)
            await pilot.pause()
            reopened = screen._library_prompt_history_state
            assert reopened is not None
            assert reopened.prompt_uuid == prompt_uuid
            assert reopened.scope_token != old_scope_token
            reopened_scope_token = reopened.scope_token
            name = screen.query_one("#library-prompt-name", Input)
            name.value = "ABA dirty working copy"
            await pilot.pause()
            assert screen._library_prompt_dirty is True

            detail_gate.set()
            for _ in range(200):
                if not any(
                    worker.name == "library_prompt_history_restore"
                    for worker in screen.workers
                ):
                    break
                await pilot.pause(0.02)
            await pilot.pause()

            current = screen._library_prompt_history_state
            assert current is not None
            assert current.prompt_uuid == prompt_uuid
            assert current.scope_token == reopened_scope_token
            assert screen._selected_prompt_id == prompt_id
            assert screen.query_one("#library-prompt-name", Input).value == (
                "ABA dirty working copy"
            )
            assert screen._library_prompt_dirty is True
    finally:
        detail_gate.set()


@pytest.mark.asyncio
async def test_library_prompt_history_collapse_during_restore_detail_fetch_stays_closed(
    tmp_path,
):
    """Collapse during post-restore fetch stays closed after detail adoption."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Collapse restore", author="", details="v1", user_prompt="source"
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    detail_gate = threading.Event()
    detail_started = threading.Event()
    service = _RecordingHistoryPromptScopeService(
        real_service,
        post_restore_detail_gate=detail_gate,
        post_restore_detail_started=detail_started,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    try:
        async with host.run_test(size=(100, 30)) as pilot:
            screen = _active_library_screen(host)
            await _wait_for_library_shell(screen, pilot)
            await _open_prompt_editor(screen, pilot, prompt_id)
            screen.app_instance.notify = host.notify
            host._notifications.clear()
            disclosure = screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            )
            disclosure.collapsed = False
            for _ in range(150):
                if len(screen.query(".library-prompt-history-row")) == 2:
                    break
                await pilot.pause(0.02)
            source = next(
                row
                for row in screen.query(".library-prompt-history-row")
                if row.source_version == 1
            )
            source.press()
            await pilot.pause()
            screen.query_one("#library-prompt-history-restore", Button).press()
            await pilot.pause()
            assert isinstance(host.screen, ConfirmationDialog)
            host.screen.query_one("#confirm-button", Button).press()
            for _ in range(200):
                if detail_started.is_set():
                    break
                await pilot.pause(0.02)
            assert len(service.restore_calls) == 1
            assert detail_started.is_set()
            persisted = db.fetch_prompt_details(prompt_id)
            assert persisted is not None and persisted["version"] == 3
            assert screen._library_prompt_version == 2

            disclosure = screen.query_one(
                "#library-prompt-history-collapsible", Collapsible
            )
            disclosure.collapsed = True
            for _ in range(100):
                state = screen._library_prompt_history_state
                if state is not None and not state.is_open:
                    break
                await pilot.pause(0.02)
            collapsed = screen._library_prompt_history_state
            assert collapsed is not None
            assert collapsed.is_open is False
            assert collapsed.rows == ()
            assert collapsed.selected is None
            assert collapsed.restore_request is None
            assert collapsed.restore_refresh_pending is True

            detail_gate.set()
            for _ in range(250):
                persisted = db.fetch_prompt_details(prompt_id)
                current = screen._library_prompt_history_state
                if (
                    persisted is not None
                    and persisted["version"] == 3
                    and screen._library_prompt_version == 3
                    and current is not None
                    and current.current_version == 3
                    and host._notifications
                ):
                    break
                await pilot.pause(0.02)

            persisted = db.fetch_prompt_details(prompt_id)
            assert persisted is not None and persisted["version"] == 3
            assert screen._library_prompt_version == 3
            assert [notice.message for notice in host._notifications] == [
                "Restored v1 as current v3."
            ]
            current = screen._library_prompt_history_state
            assert current is not None
            assert current.current_version == 3
            disclosure = await _wait_for_selector(
                screen, pilot, "#library-prompt-history-collapsible"
            )
            assert disclosure.collapsed is True
    finally:
        detail_gate.set()


@pytest.mark.asyncio
async def test_library_prompt_history_no_change_keeps_selection_and_retry_available(
    tmp_path,
):
    """Restoring a byte-identical current snapshot reports no new version."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _message = db.add_prompt(
        name="Already current",
        author="Author",
        details="Same",
        user_prompt="Same user",
        keywords=["same"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 1:
                break
            await pilot.pause(0.02)
        screen.query_one(".library-prompt-history-row", Button).press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(150):
            state = screen._library_prompt_history_state
            if state is not None and state.restore_outcome is not None:
                break
            await pilot.pause(0.02)

        state = screen._library_prompt_history_state
        assert state is not None
        assert state.restore_outcome is not None
        assert state.restore_outcome.kind == "no_change"
        assert state.restore_outcome.message == (
            "Retained v1 already matches current v1; no new version was created."
        )
        assert state.selected is not None
        assert state.selected.source_version == 1
        assert (
            screen.query_one("#library-prompt-history-restore", Button).disabled
            is False
        )
        assert db.fetch_prompt_details(prompt_id)["version"] == 1


@pytest.mark.asyncio
async def test_library_prompt_history_snapshot_unavailable_has_first_page_reload(
    tmp_path,
):
    """Snapshot recovery reloads page zero while retaining the settled count."""
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Missing snapshot",
        author="",
        details="v1",
        user_prompt="User",
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    service = _RecordingHistoryPromptScopeService(
        real_service,
        restore_result={
            "outcome": "snapshot_unavailable",
            "snapshot_unavailable": True,
            "no_change": False,
            "source_version": 1,
            "current_version": 2,
            "new_version": None,
            "retained_current_keywords": False,
        },
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        reload_button = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-reload"
        )
        await pilot.pause()
        reload_button = screen.query_one("#library-prompt-history-reload", Button)

        assert str(reload_button.label) == "Reload retained history"
        assert service.count_calls == [prompt_uuid]
        assert len(service.page_calls) == 1
        reload_button.press()
        for _ in range(150):
            state = screen._library_prompt_history_state
            if (
                state is not None
                and state.restore_outcome is None
                and state.page_status == "loaded"
                and len(service.page_calls) == 2
            ):
                break
            await pilot.pause(0.02)

        state = screen._library_prompt_history_state
        assert state is not None
        assert state.selected is None
        assert state.restore_outcome is None
        assert [row.version for row in state.rows] == [2, 1]
        assert service.count_calls == [prompt_uuid]
        assert len(service.page_calls) == 2


@pytest.mark.asyncio
async def test_library_prompt_history_generic_db_failure_is_private_and_retryable(
    tmp_path,
    monkeypatch,
):
    """A generic DB boundary failure crosses real services with private copy."""
    logger_spy = Mock()
    monkeypatch.setattr(prompt_history_module, "logger", logger_spy)
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _msg = db.add_prompt(
        name="Generic DB refusal",
        author="",
        details="v1",
        user_prompt="private body",
        keywords=["original"],
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    before_detail = db.fetch_prompt_details(prompt_id)
    before_history = db.get_prompt_history_entries(prompt_uuid, page_size=10)

    def fail_restore(*_args: Any, **_kwargs: Any):
        raise DatabaseError("SECRET retained prompt body from SQLite")

    monkeypatch.setattr(db, "restore_prompt_history_entry", fail_restore)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = real_service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        await pilot.pause()
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, ConfirmationDialog)
        modal.query_one("#confirm-button", Button).press()
        for _ in range(150):
            outcome = screen._library_prompt_history_state.restore_outcome
            if outcome is not None:
                break
            await pilot.pause(0.02)

        state = screen._library_prompt_history_state
        assert state is not None
        assert state.selected is not None
        assert state.selected.source_version == 1
        assert state.restore_outcome is not None
        assert state.restore_outcome.kind == "error"
        outcome_widget = await _wait_for_selector(
            screen, pilot, "#library-prompt-history-outcome"
        )
        assert str(outcome_widget.renderable) == "Couldn't restore retained history."
        assert (
            screen.query_one("#library-prompt-history-restore", Button).disabled
            is False
        )
        assert db.fetch_prompt_details(prompt_id) == before_detail
        assert (
            db.get_prompt_history_entries(prompt_uuid, page_size=10) == before_history
        )
        rendered = " ".join(str(static.renderable) for static in screen.query("Static"))
        assert "SECRET retained prompt body" not in rendered
        assert "private body" not in rendered
        assert "SECRET retained prompt body" not in str(logger_spy.method_calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("restore_error", "expected_copy", "enters_conflict"),
    [
        (
            RuntimeError("SECRET retained body from adapter"),
            "Couldn't restore retained history.",
            False,
        ),
        (
            ValueError("SECRET validation payload from adapter"),
            "Couldn't restore retained history.",
            False,
        ),
        (
            PromptRestoreError(PromptRestoreErrorCode.EXPECTED_VERSION),
            "This Prompt changed elsewhere. Reload before restoring.",
            True,
        ),
    ],
    ids=[
        "privacy-safe-generic",
        "privacy-safe-unexpected-value-error",
        "expected-version-conflict",
    ],
)
async def test_library_prompt_history_restore_failures_are_truthful_and_guarded(
    tmp_path,
    restore_error: Exception,
    expected_copy: str,
    enters_conflict: bool,
    monkeypatch,
):
    """Unexpected details stay private; stale expected versions enter conflict."""
    logger_spy = Mock()
    monkeypatch.setattr(prompt_history_module, "logger", logger_spy)
    db, real_service = _real_prompt_scope_service(tmp_path)
    prompt_id, _prompt_uuid, _msg = db.add_prompt(
        name="Guarded restore", author="", details="", user_prompt="v1"
    )
    db.update_prompt_by_id(prompt_id, {"details": "v2"}, expected_version=1)
    service = _RecordingHistoryPromptScopeService(
        real_service,
        restore_error=restore_error,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(150):
            outcome = screen._library_prompt_history_state.restore_outcome
            if outcome is not None:
                break
            await pilot.pause(0.02)

        outcome = screen._library_prompt_history_state.restore_outcome
        assert outcome is not None
        assert outcome.message == expected_copy
        assert "SECRET retained body" not in outcome.message
        assert "SECRET validation payload" not in outcome.message
        assert (screen._library_prompt_conflict_snapshot is not None) is enters_conflict
        if enters_conflict:
            await _wait_for_selector(screen, pilot, "#library-prompt-conflict-copy")
            assert len(screen.query("#library-prompt-conflict-copy")) == 1
        else:
            assert screen._library_prompt_history_state.selected is not None
            restore = await _wait_for_selector(
                screen, pilot, "#library-prompt-history-restore"
            )
            assert restore.disabled is False
        logged = str(logger_spy.method_calls)
        assert "SECRET retained body" not in logged
        assert "SECRET validation payload" not in logged


@pytest.mark.asyncio
async def test_library_prompt_history_duplicate_name_is_retryable_not_stale_conflict(
    tmp_path,
):
    """The real DB/service/scope/UI stack must distinguish name collisions."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Retained name",
        author="Original author",
        details="Original details",
        user_prompt="Original user",
        keywords=["original-keyword"],
    )
    db.update_prompt_by_id(
        prompt_id,
        {
            "name": "Current name",
            "author": "Current author",
            "details": "Current details",
            "user_prompt": "Current user",
            "keywords": ["current-keyword"],
        },
        expected_version=1,
    )
    db.add_prompt(
        name="Retained name",
        author="Other owner",
        details="Occupies the retained name",
        user_prompt="Other user",
    )
    before_detail = db.fetch_prompt_details(prompt_id)
    before_history = db.get_prompt_history_entries(prompt_uuid, page_size=10)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(150):
            state = screen._library_prompt_history_state
            if state is not None and state.restore_outcome is not None:
                break
            await pilot.pause(0.02)

        state = screen._library_prompt_history_state
        assert state is not None
        assert state.restore_outcome is not None
        assert state.restore_outcome.kind == "name_conflict"
        assert state.restore_outcome.message == (
            "Another active Prompt already uses this name. "
            "Rename it or choose another retained version, then retry."
        )
        assert screen._library_prompt_conflict_snapshot is None
        assert state.selected is not None
        assert state.selected.source_version == 1
        assert (
            screen.query_one("#library-prompt-history-restore", Button).disabled
            is False
        )

    after_detail = db.fetch_prompt_details(prompt_id)
    after_history = db.get_prompt_history_entries(prompt_uuid, page_size=10)
    assert after_detail == before_detail
    assert after_history == before_history


@pytest.mark.asyncio
async def test_library_prompt_history_stale_conflict_reload_refreshes_and_can_retry(
    tmp_path,
):
    """Reload adopts persisted identity/version and rebuilds the open disclosure."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Reload after conflict",
        author="Author",
        details="Retained v1",
        user_prompt="User",
    )
    db.update_prompt_by_id(prompt_id, {"details": "Opened v2"}, expected_version=1)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.app_instance.notify = host.notify
        host._notifications.clear()
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            if len(screen.query(".library-prompt-history-row")) == 2:
                break
            await pilot.pause(0.02)
        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()

        db.update_prompt_by_id(
            prompt_id, {"details": "External v3"}, expected_version=2
        )
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-conflict-copy")
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.restore_outcome is not None
        assert screen._library_prompt_history_state.restore_outcome.kind == "conflict"

        screen.query_one("#library-prompt-conflict-reload", Button).press()
        for _ in range(200):
            state = screen._library_prompt_history_state
            if (
                state is not None
                and state.current_version == 3
                and state.page_status == "loaded"
                and len(state.rows) == 3
                and len(screen.query("#library-prompt-conflict-copy")) == 0
                and any(
                    row.source_version == 1
                    for row in screen.query(".library-prompt-history-row")
                )
            ):
                break
            await pilot.pause(0.02)

        state = screen._library_prompt_history_state
        assert state is not None
        assert state.prompt_uuid == prompt_uuid
        assert state.current_version == 3
        assert state.is_open is True
        assert state.restore_outcome is None
        assert state.selected is None
        assert [row.version for row in state.rows] == [3, 2, 1]
        assert screen._library_prompt_detail["uuid"] == prompt_uuid
        assert screen._library_prompt_version == 3

        source = next(
            row
            for row in screen.query(".library-prompt-history-row")
            if row.source_version == 1
        )
        source.press()
        await pilot.pause()
        screen.query_one("#library-prompt-history-restore", Button).press()
        await pilot.pause()
        assert isinstance(host.screen, ConfirmationDialog)
        host.screen.query_one("#confirm-button", Button).press()
        for _ in range(200):
            detail = db.fetch_prompt_details(prompt_id)
            if detail is not None and detail["version"] == 4:
                break
            await pilot.pause(0.02)

        detail = db.fetch_prompt_details(prompt_id)
        assert detail is not None
        assert detail["version"] == 4
        assert [notice.message for notice in host._notifications] == [
            "Restored v1 as current v4."
        ]


@pytest.mark.asyncio
async def test_library_prompt_conflict_overwrite_adopts_persisted_detail_and_history(
    tmp_path,
):
    """The overwrite seam adopts its returned persisted record like Reload."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Overwrite conflict",
        author="Author",
        details="v1",
        user_prompt="User",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=(140, 40)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one(
            "#library-prompt-history-collapsible", Collapsible
        ).collapsed = False
        for _ in range(150):
            state = screen._library_prompt_history_state
            if state is not None and state.page_status == "loaded":
                break
            await pilot.pause(0.02)
        db.update_prompt_by_id(
            prompt_id, {"details": "external v2"}, expected_version=1
        )
        screen._enter_library_prompt_conflict(
            name="Overwrite conflict",
            author="Author",
            details="kept overwrite v3",
            system_prompt="",
            user_prompt="User",
            keywords_text="",
        )
        await pilot.pause()

        await screen._resolve_library_prompt_conflict(overwrite=True)
        for _ in range(200):
            state = screen._library_prompt_history_state
            if (
                state is not None
                and state.current_version == 3
                and state.page_status == "loaded"
            ):
                break
            await pilot.pause(0.02)

        persisted = db.fetch_prompt_details(prompt_id)
        state = screen._library_prompt_history_state
        assert persisted is not None
        assert persisted["version"] == 3
        assert persisted["details"] == "kept overwrite v3"
        assert screen._library_prompt_detail["uuid"] == persisted["uuid"] == prompt_uuid
        assert screen._library_prompt_detail["version"] == 3
        assert screen._library_prompt_version == 3
        assert state is not None
        assert state.prompt_uuid == prompt_uuid
        assert state.current_version == 3
        assert state.is_open is True
        assert state.restore_outcome is None


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_with_six_fields_populated(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        assert screen._library_prompts_view == "editor"
        assert screen.query_one("#library-prompt-name", Input).value == "Summarize"
        assert screen.query_one("#library-prompt-author", Input).value == "Alice"
        assert (
            screen.query_one("#library-prompt-details", Input).value == "A summarizer"
        )
        assert (
            screen.query_one("#library-prompt-system", TextArea).text
            == "You are concise."
        )
        assert (
            screen.query_one("#library-prompt-user", TextArea).text
            == "Summarize: {text}"
        )
        assert (
            screen.query_one("#library-prompt-keywords", Input).value
            == "summary, writing"
        )


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_with_modified_meta_not_new_prompt(
    tmp_path,
):
    """Critical regression: ``handle_library_prompt_row`` ->
    ``_refresh_library_prompt_detail`` fetches through the REAL production
    seam (``PromptScopeService.get_prompt`` -> ``normalize_prompt_record``),
    whose ``detail["id"]`` is a composite string (``"local:prompt:<uuid>"``)
    with the raw int id under ``detail["local_id"]`` instead.
    ``build_prompt_editor_state`` used to read only ``detail["id"]``, so
    ``_to_int`` silently returned ``None`` and every EXISTING saved prompt's
    meta line rendered "New prompt" (the D1 blank-create sentinel) instead
    of "Modified ... · vN". This is the assertion whose absence let that
    slip past ``test_library_prompt_row_opens_editor_with_six_fields_populated``
    above (which never inspects the meta line)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        meta = screen.query_one("#library-prompt-meta", Static)
        meta_text = str(meta.renderable)
        assert "New prompt" not in meta_text
        assert "Modified" in meta_text
        assert "v1" in meta_text


@pytest.mark.asyncio
async def test_library_prompt_row_opens_editor_under_real_runtime_policy_enforcer(
    tmp_path,
):
    """Regression test for the Phase-1 gate defect (live-blocking): clicking
    a Library prompt row raised
    ``PolicyDeniedError: Unknown runtime-policy action_id: prompts.detail.local``
    from ``PromptScopeService.get_prompt`` -> ``_enforce_policy``, which
    ``_refresh_library_prompt_detail`` (``UI/Screens/library_screen.py``)
    swallows via a bare ``except Exception`` and then treats as "prompt no
    longer available" -- the editor never opened.

    Unlike ``test_library_prompt_row_opens_editor_with_six_fields_populated``
    above (and every other test in this module), this wires the *real*
    production runtime-policy seam -- ``ServicePolicyEnforcer`` bound to the
    real ``CAPABILITY_REGISTRY`` -- via
    ``_real_prompt_scope_service_with_production_policy_enforcer`` instead of
    leaving ``policy_enforcer`` unset. That gap (no test exercised the real
    enforcer+registry combination against the Library Prompts screen) is why
    the missing ``prompts.detail.local`` registry row went uncaught.
    """
    db, service = _real_prompt_scope_service_with_production_policy_enforcer(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Summarize",
        author="Alice",
        details="A summarizer",
        system_prompt="You are concise.",
        user_prompt="Summarize: {text}",
        keywords=["writing", "summary"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        assert screen._library_prompts_view == "editor"
        assert screen._library_prompt_detail is not None
        assert screen.query_one("#library-prompt-name", Input).value == "Summarize"


@pytest.mark.asyncio
async def test_library_prompt_save_name_already_in_use_shows_status_copy(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Alpha", author="A", details="d", user_prompt="x")
    beta_id, _uuid, _msg = db.add_prompt(
        name="Beta", author="B", details="d", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert (
            status_text
            == "Name already in use — pick another or open the existing prompt."
        )


@pytest.mark.asyncio
async def test_library_prompt_save_onto_soft_deleted_name_shows_status_copy(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Gamma", author="A", details="d", user_prompt="x")
    db.soft_delete_prompt("Gamma")
    delta_id, _uuid, _msg = db.add_prompt(
        name="Delta", author="B", details="d", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, delta_id)

        screen.query_one("#library-prompt-name", Input).value = "Gamma"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert (
            status_text
            == "A deleted prompt holds this name — restore it or choose another."
        )


@pytest.mark.asyncio
async def test_library_prompt_save_stale_version_shows_conflict_bar(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Epsilon", author="A", details="d1", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_version == 1

        # A second, real service call bumps the version behind the open
        # editor's back -- simulating another writer, exactly like the
        # brief's "bump version through a second service call" scenario.
        await service.save_prompt(
            mode="local", prompt_identifier=prompt_id, details="changed elsewhere"
        )

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-save-new")) > 0:
                break
            await pilot.pause(0.02)

        assert screen.query_one("#library-prompt-conflict-save-new", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)


@pytest.mark.asyncio
async def test_library_prompt_conflict_save_as_new_replaces_source_history_identity(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, source_uuid, _message = db.add_prompt(
        name="Conflict source", author="Author", details="v1", user_prompt="User"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == source_uuid

        await service.save_prompt(
            mode="local", prompt_identifier=prompt_id, details="external v2"
        )
        screen.query_one("#library-prompt-name", Input).value = "Conflict copy"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await _wait_for_selector(screen, pilot, "#library-prompt-conflict-save-new")

        screen.query_one("#library-prompt-conflict-save-new", Button).press()
        for _ in range(200):
            state = screen._library_prompt_history_state
            if (
                screen._selected_prompt_id is not None
                and screen._selected_prompt_id != prompt_id
                and state is not None
                and state.prompt_uuid != source_uuid
                and state.count_status == "loaded"
            ):
                break
            await pilot.pause(0.02)

        new_id = screen._selected_prompt_id
        persisted = db.fetch_prompt_details(new_id)
        state = screen._library_prompt_history_state
        assert new_id is not None and new_id != prompt_id
        assert persisted is not None
        assert persisted["name"] == "Conflict copy"
        assert state is not None
        assert state.prompt_uuid == persisted["uuid"]
        assert state.prompt_uuid != source_uuid
        assert state.current_version == 1
        assert screen._library_prompt_detail["uuid"] == persisted["uuid"]


@pytest.mark.asyncio
async def test_library_prompt_save_write_time_conflict_shows_conflict_bar(tmp_path):
    """A ``ConflictError`` raised by the actual write itself -- a race the
    pre-checks cannot see (a second app instance / external writer landing
    between this save's pre-read and its real write) -- must route into
    the SAME conflict banner as the pre-check staleness path, not the
    generic "Couldn't save this prompt." status line.

    The pre-checks (name lookup, version pre-read) are left alone here --
    only the real write call (``service.save_prompt``) is monkeypatched to
    raise a real ``tldw_chatbook.DB.Prompts_DB.ConflictError`` on its next
    invocation, so this exercises the exception path inside
    ``_save_library_prompt``'s own write attempt, not the earlier
    version-mismatch pre-check.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Kappa", author="Original", details="d1", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt was modified by another writer.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_version == 1

        screen.query_one("#library-prompt-author", Input).value = "Race Author"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-save-new")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen.query_one("#library-prompt-conflict-save-new", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)
        status_widgets = screen.query("#library-prompt-save-status")
        if len(status_widgets) > 0:
            assert str(status_widgets.first().renderable) != (
                "Couldn't save this prompt. Try again."
            )

        # The stashed snapshot the banner's Save-as-new/Reload actions read
        # from must carry this entry path's live-edit fields too, exactly
        # like the pre-check path's snapshot.
        snapshot = screen._library_prompt_conflict_snapshot
        assert snapshot is not None
        assert snapshot.prompt_id == prompt_id
        assert snapshot.author == "Race Author"

        # The conflict banner leaves the shared block editor live. Edits made
        # after the conflict must belong to the detached copy rather than be
        # overwritten by the snapshot captured when the conflict first arose.
        live_block = screen.query(".prompt-block-content").first()
        assert isinstance(live_block, TextArea)
        live_block.text = "Edited during conflict"
        await pilot.pause()

        # Saving as new is detached from the conflicted source. Give the copy
        # a distinct name, then the second call delegates to create.
        screen.query_one("#library-prompt-name", Input).value = "Kappa copy"
        screen.query_one("#library-prompt-conflict-save-new", Button).press()
        await pilot.pause()
        # The handler removes the conflict banner synchronously, then starts
        # the detached save after the recompose. Wait for that save's actual
        # success state instead of treating banner removal as completion.
        for _ in range(150):
            if (
                calls["count"] == 2
                and len(screen.query("#library-prompt-conflict-save-new")) == 0
                and screen._library_prompt_dirty is False
                and screen._selected_prompt_id is not None
            ):
                break
            await pilot.pause(0.02)

        assert calls["count"] == 2
        assert len(screen.query("#library-prompt-conflict-save-new")) == 0
        assert screen._selected_prompt_id != prompt_id
        persisted = db.fetch_prompt_details(screen._selected_prompt_id)
        assert persisted["author"] == "Race Author"
        assert persisted["user_prompt"] == "Edited during conflict"
        assert db.fetch_prompt_details(prompt_id)["author"] == "Original"


@pytest.mark.asyncio
async def test_library_shell_create_prompt_write_time_conflict_recovers_on_reload(
    tmp_path,
):
    """Task 8b Fix wave 1: the CREATE flow (``_selected_prompt_id`` is
    ``None``) must recover from a genuine write-time ``ConflictError`` the
    same way the update flow does above -- NOT silently no-op both
    Save as new and Reload just because ``prompt_id`` happens to be the
    create-flow's ``None`` sentinel.

    Regression for the finding: ``_resolve_library_prompt_conflict``'s
    ``if not prompt_id or ...: return`` guard treated a create's ``None``
    prompt_id as "nothing to resolve" and returned immediately for BOTH
    buttons, so ``_library_prompt_dirty`` was never cleared either --
    ``flush_pending_work`` (and therefore Back/rail-row/prompt-row/app-tab
    navigation) then vetoed forever, trapping the user in the editor with
    no in-app recovery.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt 'Brand New' already exists.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#prompt-lane-add-user", Button).press()
        await pilot.pause()
        content = screen.query_one("#prompt-block-content-block", TextArea)
        content.text = "Hello {name}"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-save-new")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen._selected_prompt_id is None
        assert screen.query_one("#library-prompt-conflict-save-new", Button)
        assert screen.query_one("#library-prompt-conflict-reload", Button)
        # The trap the finding describes: dirty stuck true, so every other
        # exit is vetoed too -- assert it up front so a regression here is
        # unambiguous, not just inferred from the buttons doing nothing.
        assert screen._library_prompt_dirty is True

        screen.query_one("#library-prompt-conflict-reload", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-save-new")) == 0:
                break
            await pilot.pause(0.02)

        # Reload must land on a usable, blank create state -- not a
        # permanently stuck banner -- and must clear the dirty flag so
        # navigation is no longer vetoed.
        assert len(screen.query("#library-prompt-conflict-save-new")) == 0
        assert screen._library_prompt_conflict_snapshot is None
        assert screen._library_prompt_dirty is False
        assert screen.query_one("#library-prompt-name", Input).value == ""
        assert screen.query_one("#library-prompt-user", TextArea).text == ""

        allowed = await screen.flush_pending_work()
        assert allowed is True

        # The colliding name was never actually persisted under this
        # editor session -- only one prompt (the pre-existing "Brand New"
        # implied by the monkeypatched race) may exist, and this session's
        # own record was correctly abandoned rather than double-written.
        assert calls["count"] == 1


@pytest.mark.asyncio
async def test_library_shell_create_prompt_write_time_conflict_save_as_new_retries_create(
    tmp_path,
):
    """Save as new on a CREATE-flow conflict retries a detached create."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    original_save_prompt = service.save_prompt
    calls = {"count": 0}

    async def _raise_once_then_delegate(**kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise ConflictError("Prompt 'Brand New' already exists.")
        return await original_save_prompt(**kwargs)

    service.save_prompt = _raise_once_then_delegate

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#prompt-lane-add-user", Button).press()
        await pilot.pause()
        content = screen.query_one("#prompt-block-content-block", TextArea)
        content.text = "Hello {name}"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        for _ in range(150):
            if len(screen.query("#library-prompt-conflict-save-new")) > 0:
                break
            await pilot.pause(0.02)

        assert calls["count"] == 1
        assert screen._selected_prompt_id is None

        screen.query_one("#library-prompt-conflict-save-new", Button).press()
        await pilot.pause()
        # The banner disappears before the post-refresh save worker starts,
        # so wait for the retry to reach its persisted success state.
        for _ in range(150):
            if (
                calls["count"] == 2
                and len(screen.query("#library-prompt-conflict-save-new")) == 0
                and screen._library_prompt_dirty is False
                and screen._selected_prompt_id is not None
            ):
                break
            await pilot.pause(0.02)

        assert calls["count"] == 2
        assert len(screen.query("#library-prompt-conflict-save-new")) == 0
        assert screen._library_prompt_dirty is False
        assert screen._selected_prompt_id is not None
        persisted = db.fetch_prompt_details(screen._selected_prompt_id)
        assert persisted is not None
        assert persisted["name"] == "Brand New"
        assert persisted["user_prompt"] == "Hello {name}"
        assert screen._library_prompt_detail["uuid"] == persisted["uuid"]
        assert screen._library_prompt_detail["version"] == persisted["version"] == 1
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == persisted["uuid"]
        assert screen._library_prompt_history_state.current_version == 1

        allowed = await screen.flush_pending_work()
        assert allowed is True


@pytest.mark.asyncio
async def test_library_prompt_flush_pending_work_vetoes_dirty_editor(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Zeta", author="A", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-author", Input).value = "Changed mid switch"
        await pilot.pause()
        assert screen._library_prompt_dirty is True

        allowed = await screen.flush_pending_work()

        assert allowed is False
        assert screen._library_prompt_dirty is True


@pytest.mark.asyncio
async def test_library_prompt_delete_returns_to_list_and_decrements_count(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    eta_id, _uuid, _msg = db.add_prompt(
        name="Eta", author="A", details="d", user_prompt="x"
    )
    db.add_prompt(name="Theta", author="B", details="d", user_prompt="y")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, eta_id)

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, PromptDeleteConfirmationModal)
        modal.query_one("#prompt-delete-confirm", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._library_prompts_view == "list":
                break
            await pilot.pause(0.02)
        await pilot.pause()

        assert screen._library_prompts_view == "list"
        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-prompts").label)
            if "(1)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(1)" in rail_label
        assert len(screen.query(f"#library-prompt-row-{eta_id}")) == 0
        deleted = db.fetch_prompt_details(eta_id, include_deleted=True)
        assert deleted is not None
        assert deleted["deleted"] == 1


@pytest.mark.asyncio
async def test_library_prompt_delete_modal_cancel_preserves_dirty_editor_and_request(
    tmp_path,
):
    """Delete opens a typed dirty request; Cancel performs no soft delete."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Keep me", author="A", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one("#library-prompt-author", Input).value = "Unsaved author"
        await pilot.pause()

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, PromptDeleteConfirmationModal)
        request = modal.request
        assert request.dirty is True
        assert request.items[0].name == "Keep me"
        assert request.items[0].artifact_type == "prompt"
        assert request.fingerprint == screen._library_prompt_delete_fingerprint()

        modal.query_one("#prompt-delete-cancel", Button).press()
        await pilot.pause()

        assert host.screen is screen
        assert screen._library_prompts_view == "editor"
        assert screen._selected_prompt_id == prompt_id
        assert db.fetch_prompt_details(prompt_id) is not None


@pytest.mark.asyncio
async def test_library_prompt_delete_rejects_a_stale_modal_result(tmp_path):
    """A confirmation for an earlier editor identity must not delete either row."""
    db, service = _real_prompt_scope_service(tmp_path)
    first_id, _uuid, _msg = db.add_prompt(
        name="First", author="A", details="d", user_prompt="x"
    )
    second_id, _uuid, _msg = db.add_prompt(
        name="Second", author="B", details="d", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, first_id)
        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, PromptDeleteConfirmationModal)

        screen._selected_prompt_id = second_id
        modal.dismiss(PromptDeleteDecision(True, modal.request.fingerprint))
        await pilot.pause()

        assert host.screen is screen
        assert db.fetch_prompt_details(first_id) is not None
        assert db.fetch_prompt_details(second_id) is not None
        assert (
            screen._library_prompt_status == "Delete confirmation is no longer current."
        )


def test_library_prompt_delete_ignores_duplicate_modal_settlement() -> None:
    """Only the first matching confirmation can schedule the delete worker."""
    screen = SimpleNamespace(
        _library_prompt_delete_pending_fingerprint="library-prompt:9:2:prompt",
        _library_prompts_view="editor",
        _selected_prompt_id=9,
        _library_prompt_version=2,
        _library_prompt_block_state=SimpleNamespace(artifact_type="prompt"),
        _library_prompt_delete_fingerprint=lambda: "library-prompt:9:2:prompt",
        run_worker=Mock(),
        _delete_library_prompt=Mock(),
        _update_library_prompt_status_static=Mock(),
    )
    decision = PromptDeleteDecision(True, "library-prompt:9:2:prompt")

    LibraryScreen._settle_library_prompt_delete(screen, decision)
    LibraryScreen._settle_library_prompt_delete(screen, decision)

    screen.run_worker.assert_called_once()


@pytest.mark.asyncio
async def test_library_prompt_save_success_updates_status_and_persists(tmp_path):
    """Happy-path Save: not one of the brief's six numbered scenarios, but
    foundational coverage the others all rest on (a broken success path
    would make every other Save-outcome test meaningless)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Iota", author="Original", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-author", Input).value = "Updated Author"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._library_prompt_dirty is False
        assert screen._library_prompt_version == 2

        persisted = db.fetch_prompt_details(prompt_id)
        assert persisted["author"] == "Updated Author"
        assert persisted["version"] == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("include_starter", [False, True])
async def test_library_save_recipe_respects_explicit_starter_content_choice(
    tmp_path, include_starter
):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Recipe source",
        author="A",
        details="d",
        system_prompt="Stay direct.",
        user_prompt="Draft the plan.",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        starter = screen.query_one("#library-prompt-recipe-starter", Checkbox)
        assert starter.value is False
        starter.value = include_starter
        screen.query_one(
            "#library-prompt-name", Input
        ).value = f"Saved recipe {include_starter}"
        screen.query_one("#prompt-editor-save-recipe", Button).press()
        status_text = await _wait_for_prompt_status(screen, pilot)

        assert status_text == "Recipe saved as a new artifact."
        assert screen._selected_prompt_id == prompt_id
        persisted = db.fetch_prompt_details(f"Saved recipe {include_starter}")
        assert persisted["artifact_type"] == "recipe"
        assert db.fetch_prompt_details(prompt_id)["artifact_type"] == "prompt"
        raw_definition = persisted["prompt_definition"]
        definition = (
            json.loads(raw_definition)
            if isinstance(raw_definition, str)
            else raw_definition
        )
        contents = [
            block["content"] for lane in definition["lanes"] for block in lane["blocks"]
        ]
        if include_starter:
            assert contents == ["Stay direct.", "Draft the plan."]
            assert persisted["system_prompt"] == "Stay direct."
            assert persisted["user_prompt"] == "Draft the plan."
        else:
            assert contents == ["", ""]
            assert persisted["system_prompt"] == ""
            assert persisted["user_prompt"] == ""


@pytest.mark.asyncio
async def test_library_embedded_save_prompt_adopts_complete_new_identity(tmp_path):
    """Save Prompt as-new must replace every persisted identity field."""
    db, service = _real_prompt_scope_service(tmp_path)
    source_id, source_uuid, _message = db.add_prompt(
        name="Embedded source",
        author="A",
        details="source",
        system_prompt="System",
        user_prompt="User",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, source_id)
        assert screen._library_prompt_detail["local_id"] == source_id
        assert screen._library_prompt_detail["uuid"] == source_uuid

        screen.query_one("#library-prompt-name", Input).value = "Embedded saved copy"
        screen.query_one("#prompt-editor-save-prompt", Button).press()
        for _ in range(200):
            persisted = db.fetch_prompt_details("Embedded saved copy")
            state = screen._library_prompt_history_state
            if (
                persisted is not None
                and state is not None
                and state.prompt_uuid == persisted["uuid"]
            ):
                break
            await pilot.pause(0.02)

        persisted = db.fetch_prompt_details("Embedded saved copy")
        assert persisted is not None
        new_id = persisted["id"]
        new_uuid = persisted["uuid"]
        assert new_id != source_id
        assert new_uuid != source_uuid
        assert screen._selected_prompt_id == new_id
        assert screen._library_prompt_detail["id"] == new_id
        assert screen._library_prompt_detail["local_id"] == new_id
        assert screen._library_prompt_detail["uuid"] == new_uuid
        assert screen._current_library_prompt_editor_state().prompt_id == new_id
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == new_uuid


@pytest.mark.asyncio
async def test_library_use_recipe_creates_unsaved_prompt_copy_without_staging(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Planning recipe",
        author="A",
        details="d",
        system_prompt="",
        user_prompt="Draft the plan.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition={
            "kind": "block_recipe",
            "schema_version": 2,
            "lanes": [
                {"id": "system", "blocks": []},
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "markdown",
                            "content": "Draft the plan.",
                            "mapping_hint": "State the outcome.",
                        }
                    ],
                },
            ],
        },
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.stage_console_prompt_insert = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        app.stage_console_prompt_insert.assert_not_called()
        assert screen._selected_prompt_id is None
        assert screen._library_prompt_dirty is True
        assert screen._library_prompt_block_state is not None
        assert screen._library_prompt_block_state.artifact_type == "prompt"
        assert screen._library_prompt_block_state.definition.kind == "block_prompt"
        assert "unsaved Prompt copy" in screen._library_prompt_status
        assert db.fetch_prompt_details(prompt_id)["artifact_type"] == "recipe"


@pytest.mark.asyncio
async def test_library_use_mismatched_structured_prompt_requires_conversion(tmp_path):
    """A mismatched discriminator cannot execute compatibility User text."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _message = db.add_prompt(
        name="Mismatched artifact",
        author="Author",
        details="Outer Prompt, inner Recipe",
        user_prompt="# Goal\n\nDo not stage this.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition={
            "kind": "block_recipe",
            "schema_version": 2,
            "lanes": [
                {"id": "system", "blocks": []},
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "markdown",
                            "content": "Do not stage this.",
                        }
                    ],
                },
            ],
        },
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.stage_console_prompt_insert = Mock()
    app.notify = Mock()
    record_usage = AsyncMock()
    service.record_prompt_usage = record_usage
    host = LibraryHarness(app)
    before = db.fetch_prompt_details(prompt_id)
    session_before = app.get_acp_runtime_session_state()

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        assert screen._library_prompt_block_state is None
        assert screen._current_library_prompt_editor_state().definition_state == (
            "mismatched"
        )
        convert = screen.query_one("#library-prompt-convert", Button)
        assert str(convert.label) == "Convert and save as new Prompt"
        assert convert.disabled is False

        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        app.stage_console_prompt_insert.assert_not_called()
        record_usage.assert_not_awaited()
        assert not app.pending_handoffs.has_pending(
            HandoffChannel.CONSOLE_PROMPT_INSERT
        )
        assert host.seen_routes == []
        assert app.get_acp_runtime_session_state() == session_before
        app.notify.assert_called_once()
        assert "Convert and save as new Prompt" in app.notify.call_args.args[0]
        assert screen._selected_prompt_id == prompt_id
        assert screen._library_prompt_dirty is False
        after = db.fetch_prompt_details(prompt_id)
        assert after["version"] == before["version"]
        assert after["artifact_type"] == "prompt"


@pytest.mark.asyncio
async def test_library_prompt_editing_shows_unsaved_marker_and_save_clears_it(tmp_path):
    """U6 (Task 8c): editing a field surfaces a visible unsaved-changes
    marker on the meta line -- previously the dirty flag was invisible
    until the ``flush_pending_work`` veto fired on nav-away. Saving clears
    it. The meta ``Static`` instance itself must never change identity
    across the edit (a full recompose would remount the Input/TextArea
    fields, re-arm-race the editor, and silently re-trigger the mount-time
    ``Changed`` event the arm-delay guards against)."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Mu", author="Original", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        meta_before = screen.query_one("#library-prompt-meta", Static)
        assert "Unsaved" not in str(meta_before.renderable)

        screen.query_one("#library-prompt-author", Input).value = "Changed"
        await pilot.pause()

        assert screen._library_prompt_dirty is True
        meta_after_edit = screen.query_one("#library-prompt-meta", Static)
        assert meta_after_edit is meta_before  # no recompose -- same widget instance
        assert "• Unsaved changes" in str(meta_after_edit.renderable)

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."

        assert screen._library_prompt_dirty is False
        meta_after_save = screen.query_one("#library-prompt-meta", Static)
        assert meta_after_save is meta_before
        assert "Unsaved" not in str(meta_after_save.renderable)


# ---------------------------------------------------------------------------
# Task 5: toolbar Import… + editor Export .md, end-to-end (real DB + service)
# ---------------------------------------------------------------------------


async def _open_prompts_list(screen, pilot) -> None:
    """Open the rail's Prompts row (list view, not the editor)."""
    screen.query_one("#library-row-browse-prompts").press()
    await pilot.pause()
    await pilot.pause()


@pytest.mark.asyncio
async def test_library_prompts_import_button_opens_row(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        assert len(screen.query("#library-prompts-import-path")) == 0
        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()

        assert screen.query_one("#library-prompts-import-path", Input)


@pytest.mark.asyncio
async def test_library_prompts_import_cancel_closes_row(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()
        assert screen.query_one("#library-prompts-import-path", Input)

        screen.query_one("#library-prompts-import-cancel", Button).press()
        await pilot.pause()

        assert len(screen.query("#library-prompts-import-path")) == 0


@pytest.mark.asyncio
async def test_library_prompts_import_preserves_recipe_structure_in_real_db(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    definition = {
        "kind": "block_recipe",
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "markdown",
                        "content": "Starter role",
                        "mapping_hint": "Define the job.",
                    },
                    {
                        "id": "voice",
                        "title": "Voice",
                        "syntax": "freeform",
                        "content": "Direct",
                        "mapping_hint": "Set the tone.",
                    },
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "xml",
                        "xml_tag": "goal",
                        "content": "Ship the outcome",
                        "mapping_hint": "State success.",
                    }
                ],
            },
        ],
    }
    export_path = tmp_path / "recipe.md"
    export_path.write_text(
        render_prompt_markdown(
            {
                "name": "Imported Recipe",
                "author": "Author",
                "details": "Details",
                "artifact_type": "recipe",
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": definition,
                "system_prompt": "# Role\n\nStarter role\n\nDirect",
                "user_prompt": "<goal>Ship the outcome</goal>",
                "keywords": ["structured"],
            }
        ),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        await screen._run_library_prompts_import(str(export_path))
        await pilot.pause()

        persisted = db.fetch_prompt_details("Imported Recipe")
        assert persisted is not None
        assert persisted["artifact_type"] == "recipe"
        assert persisted["prompt_format"] == "structured"
        assert persisted["prompt_schema_version"] == 2
        assert json.loads(persisted["prompt_definition"]) == definition
        assert persisted["system_prompt"] == "# Role\n\nStarter role\n\nDirect"
        assert persisted["user_prompt"] == "<goal>Ship the outcome</goal>"


@pytest.mark.asyncio
async def test_library_prompt_export_pushes_file_save_dialog(tmp_path):
    """Export… pushes a ``FileSave`` dialog pre-filled with a sanitized
    default filename derived from the prompt's current name -- mirrors
    ``test_library_shell_note_export_markdown_pushes_file_save_dialog``."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me",
        author="Author",
        details="d",
        system_prompt="s",
        user_prompt="u",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-export", Button).press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export… never pushed a FileSave dialog.")

        dialog = host.screen_stack[-1]
        assert dialog._default_file == "Export Me.md"

        await host.pop_screen()
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_prompt_export_blocks_invalid_recipe_without_downgrade(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    definition = {
        "kind": "block_recipe",
        "schema_version": 2,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "xml",
                        "xml_tag": "goal",
                        "content": "Starter",
                    }
                ],
            },
        ],
    }
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Invalid export",
        author="Author",
        details="Details",
        user_prompt="<goal>Starter</goal>",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#prompt-block-xml-tag-goal", Input).value = "bad tag"
        await pilot.pause()
        assert screen._library_prompt_block_state is not None
        assert screen._library_prompt_block_state.issues

        stack_size = len(host.screen_stack)
        await screen._export_library_prompt()
        await pilot.pause()

        assert len(host.screen_stack) == stack_size
        app.notify.assert_called_once()
        args, kwargs = app.notify.call_args
        assert "Fix block validation errors before exporting" in args[0]
        assert kwargs.get("severity") == "warning"
        assert not (tmp_path / "Invalid export.md").exists()


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_writes_roundtrippable_markdown(
    tmp_path,
):
    """The export write-path (bypassing the dialog UI, exercised separately
    above) writes content that round-trips through the real parser --
    mirrors ``test_library_shell_note_write_export_file_writes_expected_content``."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me",
        author="Author",
        details="d",
        system_prompt="s",
        user_prompt="u",
        keywords=["k1", "k2"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        destination = tmp_path / "export.md"
        screen._write_library_prompt_export_file(
            destination,
            "Export Me",
            "Author",
            "d",
            "s",
            "u",
            "k1, k2",
            prompt_id,
        )

        written = destination.read_text(encoding="utf-8")
        parsed = parse_markdown_prompts_from_content(written)
        assert len(parsed) == 1
        p = parsed[0]
        assert (
            p["name"],
            p["author"],
            p["details"],
            p["system_prompt"],
            p["user_prompt"],
        ) == (
            "Export Me",
            "Author",
            "d",
            "s",
            "u",
        )
        assert p["keywords"] == ["k1", "k2"]


@pytest.mark.asyncio
async def test_library_prompt_write_export_preserves_live_recipe_structure(tmp_path):
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Recipe export", author="Author", details="Details", user_prompt=""
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)
    definition = {
        "kind": "block_recipe",
        "schema_version": 2,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "xml",
                        "xml_tag": "goal",
                        "content": "Starter",
                        "mapping_hint": "State the outcome.",
                    }
                ],
            },
        ],
    }

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        destination = tmp_path / "recipe.md"
        screen._write_library_prompt_export_file(
            destination,
            "Recipe export",
            "Author",
            "Details",
            "",
            "<goal>Starter</goal>",
            "recipe",
            prompt_id,
            {
                "artifact_type": "recipe",
                "prompt_format": "structured",
                "prompt_schema_version": 2,
                "prompt_definition": definition,
                "system_prompt": "",
                "user_prompt": "<goal>Starter</goal>",
            },
        )

        [parsed] = parse_markdown_prompts_from_content(
            destination.read_text(encoding="utf-8")
        )
        assert parsed["artifact_type"] == "recipe"
        assert parsed["prompt_definition"] == definition
        app.notify.assert_called_once()
        assert "exported successfully" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_rejects_invalid_path(
    tmp_path, monkeypatch
):
    """A ``FileSave``-returned path that fails ``validate_path_simple`` must
    be rejected with a quiet warning notice -- no write, no crash."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me", author="A", details="d", user_prompt="u"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    monkeypatch.setattr(library_screen_module, "validate_path_simple", _reject_path)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        destination = tmp_path / "export.md"
        screen._write_library_prompt_export_file(
            destination,
            "Export Me",
            "A",
            "d",
            "",
            "u",
            "",
            prompt_id,
        )

        assert not destination.exists()
        app.notify.assert_called_once()
        args, kwargs = app.notify.call_args
        assert "Rejected export path" in args[0]
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_prompt_write_export_file_cancelled_dialog_notifies_quietly(
    tmp_path,
):
    """A cancelled ``FileSave`` dialog (``selected_path=None``) is a silent
    no-op plus a quiet notice -- no write, no crash."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Export Me", author="A", details="d", user_prompt="u"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen._write_library_prompt_export_file(
            None,
            "Export Me",
            "A",
            "d",
            "",
            "u",
            "",
            prompt_id,
        )

        app.notify.assert_called_once()
        assert "cancelled" in app.notify.call_args.args[0]


# ---------------------------------------------------------------------------
# Task 8b, Group 1: D1 (New prompt create entry), U2 (Author demoted), U3
# (Duplicate), U4 ("Description" label)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_editor_description_label_replaces_details():
    """U4: the rendered field label reads "Description", not "Details" --
    the DB/record field name (``#library-prompt-details``) is untouched."""
    editor_state = PromptEditorState(
        prompt_id=1,
        name="X",
        author="A",
        details="d",
        system_prompt="s",
        user_prompt="u",
        keywords_csv="",
        version=1,
        created="",
        modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        labels = [
            str(getattr(s.renderable, "plain", s.renderable))
            for s in pilot.app.query(".library-prompt-field-label")
        ]
        assert "Description" in labels
        assert "Details" not in labels
        assert pilot.app.query_one("#library-prompt-details", Input).value == "d"


@pytest.mark.asyncio
async def test_prompts_canvas_editor_field_order_author_last_beside_keywords():
    """U2: compose order is Name, Description, System prompt, User prompt,
    Keywords, Author -- Author moves from 2nd/3rd position to last."""
    editor_state = PromptEditorState(
        prompt_id=1,
        name="X",
        author="A",
        details="d",
        system_prompt="s",
        user_prompt="u",
        keywords_csv="kw1, kw2",
        version=1,
        created="",
        modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        shell = pilot.app.query_one("#library-prompt-editor-shell")
        content = shell.query_one("#library-prompt-editor-content")
        assert content.parent is shell
        ids = [child.id for child in content.walk_children() if child.id]
        assert (
            ids.index("library-prompt-name")
            < ids.index("library-prompt-details")
            < ids.index("library-prompt-system")
            < ids.index("library-prompt-user")
            < ids.index("library-prompt-keywords")
            < ids.index("library-prompt-author")
        )


# ---------------------------------------------------------------------------
# Task 8c: U7 (System/User field help) + U8 (Copy vs Duplicate relabel)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_prompts_canvas_editor_renders_system_and_user_field_hints():
    """U7: a one-line dim hint renders under each of the System prompt/User
    prompt labels, explaining the two-part prompt model to a new user."""
    editor_state = PromptEditorState(
        prompt_id=1,
        name="X",
        author="A",
        details="d",
        system_prompt="s",
        user_prompt="u",
        keywords_csv="",
        version=1,
        created="",
        modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        hints = [
            str(getattr(s.renderable, "plain", s.renderable))
            for s in pilot.app.query(".library-prompt-field-hint")
        ]
        assert "Instructions the model always follows." in hints
        assert "The message inserted into the composer." in hints


@pytest.mark.asyncio
async def test_prompts_canvas_editor_copy_and_duplicate_relabeled():
    """Catches the Task-202 copy-label mutation while stable ids remain unchanged."""
    editor_state = PromptEditorState(
        prompt_id=1,
        name="X",
        author="A",
        details="d",
        system_prompt="s",
        user_prompt="u",
        keywords_csv="",
        version=1,
        created="",
        modified="2026-07-07T11:00:00+00:00",
    )
    app = _CanvasHost(None, mode="editor", editor_state=editor_state)
    async with app.run_test() as pilot:
        copy_button = pilot.app.query_one("#library-prompt-copy", Button)
        duplicate_button = pilot.app.query_one("#library-prompt-duplicate", Button)
        assert str(copy_button.label) == "Copy Markdown"
        assert str(duplicate_button.label) == "Duplicate prompt"


# ---------------------------------------------------------------------------
# Task 202: intentionally-red editor action geometry, grouping, and copy
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40), (200, 50)])
@pytest.mark.parametrize("conflict", [False, True])
async def test_library_prompt_editor_geometry_keeps_actions_visible_without_covering_author(
    size: tuple[int, int],
    conflict: bool,
    tmp_path,
):
    """Catches the planned ``_compose_editor`` shell/content/action split.

    The production mutation must give the bounded editor a single scrollable
    content owner plus a visible, non-scrolling action area. A flat trailing
    toolbar leaves the actions below the viewport at these real terminal sizes.
    """
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Geometry prompt",
        author="A",
        details="d",
        system_prompt="# Role\n\nBe exact.",
        user_prompt="Ship it.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition={
            "kind": "block_prompt",
            "schema_version": 2,
            "lanes": [
                {
                    "id": "system",
                    "blocks": [
                        {
                            "id": "role",
                            "title": "Role",
                            "syntax": "markdown",
                            "content": "Be exact.",
                        }
                    ],
                },
                {
                    "id": "user",
                    "blocks": [
                        {
                            "id": "goal",
                            "title": "Goal",
                            "syntax": "markdown",
                            "content": "Ship it.",
                        }
                    ],
                },
            ],
        },
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        if conflict:
            screen._library_prompt_conflict_snapshot = (
                screen._current_library_prompt_editor_state()
            )
            screen._library_prompt_dirty = True
            screen.refresh(recompose=True)
            await pilot.pause()

        canvas = screen.query_one("#library-prompts-canvas")
        shell = screen.query_one("#library-prompt-editor-shell")
        content = screen.query_one("#library-prompt-editor-content")
        actions = screen.query_one("#library-prompt-editor-actions")
        author = screen.query_one("#library-prompt-author", Input)

        assert canvas.region.contains_region(shell.region)
        assert shell.region.contains_region(actions.region)
        assert actions.region.width > 0
        assert actions.region.height > 0
        assert content.max_scroll_y > 0
        assert actions.max_scroll_y == 0
        assert list(content.query(VerticalScroll)) == []

        content.scroll_end(animate=False)
        await pilot.pause()
        assert not actions.region.overlaps(author.region)
        action_ids = (
            (
                "library-prompt-conflict-save-new",
                "library-prompt-conflict-reload",
            )
            if conflict
            else (
                "library-prompt-save",
                "library-prompt-insert-console",
                "library-prompt-export",
                "library-prompt-copy",
                "library-prompt-duplicate",
                "library-prompt-delete",
            )
        )
        for action_id in action_ids:
            action = screen.query_one(f"#{action_id}", Button)
            assert action.region.width > 0
            assert action.region.height > 0
            assert screen.region.contains_region(action.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40), (200, 50)])
@pytest.mark.parametrize(
    "history_mode", ["normal", "dirty", "error", "snapshot-unavailable"]
)
async def test_library_prompt_history_geometry_uses_only_the_outer_editor_scroll(
    size: tuple[int, int],
    history_mode: str,
):
    """History, dirty gates, and recovery copy preserve the fixed actions."""
    history = _history_state()
    if history_mode == "error":
        history = replace(
            history,
            page_status="error",
            rows=(),
            selected=None,
            has_more=False,
            next_before_change_id=None,
            error="Couldn't load retained history. Try again.",
        )
    elif history_mode == "snapshot-unavailable":
        history = replace(
            history,
            restore_outcome=PromptHistoryRestoreOutcome(
                kind="snapshot_unavailable",
                message=(
                    "This retained version is no longer available. "
                    "Reload retained history."
                ),
                reload_required=True,
            ),
        )
    app = _StyledCanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=history,
        dirty=history_mode == "dirty",
    )

    async with app.run_test(size=size) as pilot:
        canvas = pilot.app.query_one("#library-prompts-canvas")
        shell = pilot.app.query_one("#library-prompt-editor-shell")
        content = pilot.app.query_one("#library-prompt-editor-content")
        actions = pilot.app.query_one("#library-prompt-editor-actions")
        history_region = pilot.app.query_one("#library-prompt-history-region")

        assert canvas.region.contains_region(shell.region)
        assert shell.region.contains_region(actions.region)
        assert actions.region.width > 0
        assert actions.region.height > 0
        assert actions.max_scroll_y == 0
        assert list(content.query(VerticalScroll)) == []
        assert history_region.region.width > 0
        assert not history_region.region.overlaps(actions.region)
        for action in actions.query(Button):
            assert action.region.width > 0
            assert action.region.height > 0
            assert pilot.app.screen.region.contains_region(action.region)

        if history_mode == "dirty":
            restore = pilot.app.query_one("#library-prompt-history-restore", Button)
            assert restore.disabled is True
            assert restore.styles.opacity == 1.0
            content.scroll_to_widget(restore, animate=False, force=True)
            await pilot.pause()
            painted = _painted_style_of_text(
                pilot.app, restore.region, "Restore selected version"
            )
            assert painted is not None
            assert painted.color is not None and painted.bgcolor is not None
            ratio = _painted_contrast(painted.color, painted.bgcolor)
            assert ratio >= 3.0, f"disabled restore paints at only {ratio:.2f}:1"

        if history_mode != "error":
            unselected = pilot.app.query_one("#library-prompt-history-row-11", Button)
            height_before = unselected.region.height
            unselected.add_class("history-selected")
            await pilot.pause()
            assert unselected.region.height == height_before
            content.scroll_to_widget(unselected, animate=False, force=True)
            await pilot.pause()
            assert not unselected.region.overlaps(actions.region)
            if history_mode == "snapshot-unavailable":
                reload_history = pilot.app.query_one(
                    "#library-prompt-history-reload", Button
                )
                content.scroll_to_widget(
                    reload_history,
                    animate=False,
                    force=True,
                )
                await pilot.pause()
                assert not reload_history.region.overlaps(actions.region)
        else:
            retry = pilot.app.query_one("#library-prompt-history-retry-page", Button)
            content.scroll_to_widget(retry, animate=False, force=True)
            await pilot.pause()
            assert retry.region.width > 0
            assert retry.region.height > 0
            assert not retry.region.overlaps(actions.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(80, 24), (100, 30), (140, 40), (200, 50)])
async def test_library_prompt_history_confirmation_geometry_is_contained(
    size: tuple[int, int],
):
    """The validated confirmation remains readable at supported geometries."""
    app = _StyledCanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        history_state=_history_state(),
    )

    async with app.run_test(size=size) as pilot:
        modal = ConfirmationDialog(
            title="Restore retained version?",
            message=(
                "Restore retained v1 over current v2? This changes the artifact "
                "type from Recipe to Prompt. Confirming creates a new current "
                "version."
            ),
            confirm_label="Restore",
            cancel_label="Cancel",
        )
        pilot.app.push_screen(modal)
        await pilot.pause()
        dialog = modal.query_one(Container)
        assert modal.region.contains_region(dialog.region)
        assert dialog.region.width > 0
        assert dialog.region.height > 0
        for button_id in ("confirm-button", "cancel-button"):
            button = modal.query_one(f"#{button_id}", Button)
            assert dialog.region.contains_region(button.region)
            assert button.region.width > 0
            assert button.region.height > 0


@pytest.mark.asyncio
async def test_library_prompt_action_groups_preserve_normal_dom_and_focus_order():
    """Catches the action-group wrapper mutation replacing the flat toolbar."""
    app = _StyledCanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
    )

    async with app.run_test(size=(140, 40)) as pilot:
        actions = pilot.app.query_one("#library-prompt-editor-actions")
        primary = pilot.app.query_one("#library-prompt-actions-primary")
        content = pilot.app.query_one("#library-prompt-actions-content")
        lifecycle = pilot.app.query_one("#library-prompt-actions-lifecycle")

        assert [child.id for child in actions.children] == [
            "library-prompt-actions-primary",
            "library-prompt-actions-content",
            "library-prompt-actions-lifecycle",
        ]
        assert [button.id for button in primary.query(Button)] == [
            "library-prompt-save"
        ]
        assert [button.id for button in content.query(Button)] == [
            "library-prompt-insert-console",
            "library-prompt-export",
            "library-prompt-copy",
        ]
        assert [button.id for button in lifecycle.query(Button)] == [
            "library-prompt-duplicate",
            "library-prompt-delete",
        ]
        assert [button.id for button in actions.query(Button)] == [
            "library-prompt-save",
            "library-prompt-insert-console",
            "library-prompt-export",
            "library-prompt-copy",
            "library-prompt-duplicate",
            "library-prompt-delete",
        ]
        assert str(pilot.app.query_one("#library-prompt-copy", Button).label) == (
            "Copy Markdown"
        )


@pytest.mark.asyncio
async def test_library_prompt_action_groups_preserve_conflict_action_order():
    """Catches the conflict action-area mutation replacing the flat toolbar."""
    app = _StyledCanvasHost(
        None,
        mode="editor",
        editor_state=_structured_editor_state(),
        conflict=True,
    )

    async with app.run_test(size=(100, 30)) as pilot:
        actions = pilot.app.query_one("#library-prompt-editor-actions")
        assert [button.id for button in actions.query(Button)] == [
            "library-prompt-conflict-save-new",
            "library-prompt-conflict-reload",
        ]
        assert [str(button.label) for button in actions.query(Button)] == [
            "Save as new",
            "Reload",
        ]


@pytest.mark.asyncio
async def test_library_prompt_copy_uses_live_unsaved_legacy_lane_markdown(tmp_path):
    """Catches the missing legacy-lane ``handle_library_prompt_copy`` path."""
    _db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = _db.add_prompt(
        name="Copy source",
        author="Original author",
        details="Original details",
        system_prompt="Original system",
        user_prompt="Original user",
        keywords=["alpha", "beta"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one("#library-prompt-system", TextArea).text = "Edited system"
        screen.query_one("#library-prompt-user", TextArea).text = "Edited user"
        await pilot.pause()

        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [
            render_prompt_markdown(
                {
                    "name": "Copy source",
                    "author": "Original author",
                    "details": "Original details",
                    "system_prompt": "Edited system",
                    "user_prompt": "Edited user",
                    "keywords": ["alpha", "beta"],
                }
            )
        ]
        assert [notification.message for notification in host._notifications] == [
            "Prompt copied to clipboard as markdown!"
        ]


@pytest.mark.asyncio
async def test_library_prompt_copy_uses_current_structured_block_working_copy(tmp_path):
    """Catches a copy handler that serializes preview text but drops structure."""
    definition = {
        "kind": "block_prompt",
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "markdown",
                        "content": "Original role.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "markdown",
                        "content": "Original goal.",
                    }
                ],
            },
        ],
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Structured copy source",
        author="Original author",
        details="Original details",
        system_prompt="# Role\n\nOriginal role.",
        user_prompt="# Goal\n\nOriginal goal.",
        keywords=["alpha", "beta"],
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one("#prompt-block-content-role", TextArea).text = "Edited role."
        await pilot.pause()

        block_state = screen._library_prompt_block_state
        assert block_state is not None
        assert block_state.definition.lanes[0].blocks[0].content == "Edited role."
        _draft, artifact_fields, _prepared = prepare_prompt_artifact_save(
            block_state,
            artifact_type=block_state.artifact_type,
            include_recipe_starter_content=True,
            request_fields={},
        )
        expected_markdown = render_prompt_markdown(
            {
                "name": screen.query_one("#library-prompt-name", Input).value,
                "author": screen.query_one("#library-prompt-author", Input).value,
                "details": screen.query_one("#library-prompt-details", Input).value,
                "keywords": screen.query_one("#library-prompt-keywords", Input).value,
                **artifact_fields,
            }
        )
        assert "### ARTIFACT_TYPE ###\nprompt\n" in expected_markdown
        assert "### STRUCTURE ###\n```json\n" in expected_markdown

        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [expected_markdown]
        assert [notification.message for notification in host._notifications] == [
            "Prompt copied to clipboard as markdown!"
        ]


@pytest.mark.asyncio
async def test_library_prompt_copy_warns_when_clipboard_is_unavailable(tmp_path):
    """Catches the missing unavailable-clipboard branch in the copy handler."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Copy source", author="A", details="d", user_prompt="u"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.app_instance = host
        host.copy_to_clipboard = None
        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        notifications = list(host._notifications)
        assert [notification.message for notification in notifications] == [
            "Clipboard copy is unavailable in this runtime."
        ]
        assert [notification.severity for notification in notifications] == ["warning"]


@pytest.mark.asyncio
async def test_library_prompt_copy_reports_clipboard_error_without_success_notice(
    tmp_path,
):
    """Catches the missing clipboard-exception branch in the copy handler."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Copy source", author="A", details="d", user_prompt="u"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    def unavailable_clipboard(_markdown: str) -> None:
        raise RuntimeError("clipboard unavailable")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.app_instance = host
        host.copy_to_clipboard = unavailable_clipboard
        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        notifications = list(host._notifications)
        assert [notification.message for notification in notifications] == [
            "Error copying prompt: RuntimeError"
        ]
        assert [notification.severity for notification in notifications] == ["error"]
        assert all(
            "copied to clipboard" not in notification.message.lower()
            for notification in notifications
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("artifact_type", "schema_version"),
    [("prompt", 1), ("recipe", 3)],
)
async def test_library_prompt_copy_preserves_compatibility_structured_metadata(
    tmp_path, artifact_type, schema_version
):
    """Copy must preserve raw structured metadata when no editable block exists."""
    kind = f"foreign_{artifact_type}"
    definition = {
        "schema_version": schema_version,
        "kind": kind,
        "opaque": {"keep": "this definition"},
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name=f"Compatibility {artifact_type}",
        author="A",
        details="d",
        system_prompt="compat system",
        user_prompt="compat user",
        prompt_format="structured",
        prompt_schema_version=schema_version,
        prompt_definition=definition,
        artifact_type=artifact_type,
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_block_state is None
        assert isinstance(screen._library_prompt_detail, dict)
        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert len(copied) == 1
        assert "### SYSTEM ###\ncompat system\n" in copied[0]
        assert "### USER ###\ncompat user\n" in copied[0]
        assert f"### ARTIFACT_TYPE ###\n{artifact_type}\n" in copied[0]
        assert "### STRUCTURE ###\n```json\n" in copied[0]
        assert f'"kind":"{kind}"' in copied[0]
        assert f'"schema_version":{schema_version}' in copied[0]
        assert '"keep":"this definition"' in copied[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("prompt_format", "outer_schema", "raw_definition"),
    [
        ("structured", 2, "{malformed-json"),
        ("structured", 2, "[]"),
        (
            "future_structured",
            2,
            json.dumps(
                {
                    "kind": "block_prompt",
                    "schema_version": 2,
                    "lanes": [],
                    "private_definition": "DO_NOT_DISCLOSE",
                }
            ),
        ),
        (
            "legacy",
            2,
            json.dumps(
                {
                    "kind": "block_prompt",
                    "schema_version": 2,
                    "lanes": [],
                    "private_definition": "DO_NOT_DISCLOSE",
                }
            ),
        ),
        (
            "structured",
            3,
            json.dumps(
                {
                    "kind": "future_prompt",
                    "schema_version": 2,
                    "private_definition": "DO_NOT_DISCLOSE",
                }
            ),
        ),
        (
            "structured",
            2,
            json.dumps(
                {
                    "kind": "block_recipe",
                    "schema_version": 2,
                    "lanes": [],
                    "private_definition": "DO_NOT_DISCLOSE",
                }
            ),
        ),
    ],
    ids=[
        "malformed-json",
        "non-object-json",
        "unknown-format",
        "non-structured-format",
        "schema-mismatch",
        "artifact-kind-mismatch",
    ],
)
async def test_library_prompt_copy_and_export_reject_unrepresentable_metadata(
    tmp_path, prompt_format, outer_schema, raw_definition
):
    """Copy/Export cannot flatten modern metadata the Markdown grammar loses."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Private metadata",
        author="A",
        details="d",
        system_prompt="SECRET_SYSTEM_BODY",
        user_prompt="SECRET_USER_BODY",
        prompt_format="structured",
        prompt_schema_version=outer_schema,
        prompt_definition=raw_definition,
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        if prompt_format != "structured":
            assert isinstance(screen._library_prompt_detail, dict)
            screen._library_prompt_detail["prompt_format"] = prompt_format
            screen._library_prompt_block_state = None
        screen.app_instance = host
        host.copy_to_clipboard = copied.append

        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == []
        assert [notice.message for notice in host._notifications] == [
            "This structured artifact cannot be represented as Markdown "
            "without losing metadata. Use Convert and save as a new Prompt first."
        ]
        assert [notice.severity for notice in host._notifications] == ["warning"]
        copy_notices = list(host._notifications)
        assert all(
            private not in copy_notices[0].message
            for private in (
                "SECRET_SYSTEM_BODY",
                "SECRET_USER_BODY",
                "DO_NOT_DISCLOSE",
                "malformed-json",
                "ValueError",
            )
        )

        host._notifications.clear()
        stack_size = len(host.screen_stack)
        screen.query_one("#library-prompt-export", Button).press()
        await pilot.pause()

        assert len(host.screen_stack) == stack_size
        assert not any(isinstance(item, FileSave) for item in host.screen_stack)
        assert [notice.message for notice in host._notifications] == [
            "This structured artifact cannot be represented as Markdown "
            "without losing metadata. Use Convert and save as a new Prompt first."
        ]
        export_notices = list(host._notifications)
        assert all(
            private not in export_notices[0].message
            for private in (
                "SECRET_SYSTEM_BODY",
                "SECRET_USER_BODY",
                "DO_NOT_DISCLOSE",
                "malformed-json",
                "ValueError",
            )
        )


@pytest.mark.asyncio
async def test_library_prompt_export_preserves_representable_foreign_recipe(tmp_path):
    """A foreign Recipe with agreeing metadata remains exportable without conversion."""
    definition = {
        "kind": "foreign_recipe",
        "schema_version": 3,
        "opaque": {"keep": "FOREIGN_RECIPE_DEFINITION"},
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Foreign Recipe",
        author="A",
        details="d",
        system_prompt="foreign system",
        user_prompt="foreign user",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    destination = tmp_path / "foreign-recipe.md"

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one("#library-prompt-export", Button).press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        dialog = host.screen_stack[-1]
        assert isinstance(dialog, FileSave)

        dialog.dismiss(destination)
        for _ in range(150):
            if destination.exists():
                break
            await pilot.pause(0.02)

        assert destination.exists()
        exported = destination.read_text(encoding="utf-8")
        assert "### SYSTEM ###\nforeign system\n" in exported
        assert "### USER ###\nforeign user\n" in exported
        assert "### ARTIFACT_TYPE ###\nrecipe\n" in exported
        assert '"kind":"foreign_recipe"' in exported
        assert '"schema_version":3' in exported
        assert '"keep":"FOREIGN_RECIPE_DEFINITION"' in exported


@pytest.mark.asyncio
async def test_library_prompt_copy_after_compatibility_recipe_conversion_uses_prompt(
    tmp_path,
):
    """Convert detaches canonical Prompt metadata from the source Recipe."""
    source_definition = {
        "schema_version": 3,
        "kind": "future_recipe",
        "opaque": {"source": "must not survive conversion"},
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, source_uuid, _msg = db.add_prompt(
        name="Compatibility recipe",
        author="A",
        details="d",
        system_prompt="compat system",
        user_prompt="compat user",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition=source_definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_block_state is None
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == source_uuid

        screen.query_one("#library-prompt-convert", Button).press()
        await pilot.pause()
        assert screen._selected_prompt_id is None
        assert screen._library_prompt_history_state is None
        assert len(screen.query("#library-prompt-history-collapsible")) == 0
        assert "uuid" not in screen._library_prompt_detail
        assert "version" not in screen._library_prompt_detail
        detached_state = screen._current_library_prompt_editor_state()
        assert detached_state.prompt_id is None
        save = screen.query_one("#library-prompt-save", Button)
        assert str(save.label) == "Save Prompt"
        assert save.disabled is False
        assert str(screen.query_one("#library-prompt-meta", Static).renderable) == (
            "New prompt · • Unsaved changes"
        )
        converted_content = screen.query_one(
            "#prompt-block-content-legacy-system-1", TextArea
        )
        converted_content.text = "Converted system"
        await pilot.pause()

        block_state = screen._library_prompt_block_state
        assert block_state is not None
        assert block_state.artifact_type == "prompt"
        _draft, artifact_fields, _prepared = prepare_prompt_artifact_save(
            block_state,
            artifact_type="prompt",
            include_recipe_starter_content=True,
            request_fields={},
        )
        expected = render_prompt_markdown(
            {
                "name": screen.query_one("#library-prompt-name", Input).value,
                "author": screen.query_one("#library-prompt-author", Input).value,
                "details": screen.query_one("#library-prompt-details", Input).value,
                "keywords": [],
                **artifact_fields,
            }
        )
        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [expected]
        assert "### ARTIFACT_TYPE ###\nprompt\n" in copied[0]
        assert '"kind":"block_prompt"' in copied[0]
        assert '"schema_version":3' not in copied[0]
        assert "future_recipe" not in copied[0]


@pytest.mark.asyncio
async def test_library_prompt_copy_keeps_both_edited_legacy_lanes_plain(tmp_path):
    """Editing both real legacy blocks cannot implicitly change Copy format."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Legacy lanes",
        author="A",
        details="d",
        system_prompt="Original system",
        user_prompt="Original user",
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._current_library_prompt_editor_state().definition_state == (
            "legacy"
        )

        screen.query_one(
            "#prompt-block-content-legacy-system-1", TextArea
        ).text = "Edited system"
        await pilot.pause()
        screen.query_one(
            "#prompt-block-content-legacy-user-1", TextArea
        ).text = "Edited user"
        await pilot.pause()
        block_state = screen._library_prompt_block_state
        assert block_state is not None
        assert block_state.system_origin is None
        assert block_state.user_origin is None

        expected = render_prompt_markdown(
            {
                "name": "Legacy lanes",
                "author": "A",
                "details": "d",
                "system_prompt": "Edited system",
                "user_prompt": "Edited user",
                "keywords": [],
            }
        )
        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [expected]
        assert "### ARTIFACT_TYPE ###" not in copied[0]
        assert "### STRUCTURE ###" not in copied[0]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("schema_version", "definition_state"),
    [(1, "foreign_v1"), (2, "malformed")],
)
async def test_library_prompt_delete_uses_compatibility_recipe_type(
    tmp_path, schema_version, definition_state
):
    """Read-only/foreign Recipes must still be named correctly in Delete."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Compatibility recipe",
        author="A",
        details="d",
        user_prompt="compat user",
        prompt_format="structured",
        prompt_schema_version=schema_version,
        prompt_definition={"schema_version": schema_version, "kind": "future"},
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_block_state is None
        assert screen._current_library_prompt_editor_state().definition_state == (
            definition_state
        )

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, PromptDeleteConfirmationModal)
        assert modal.request.items[0].artifact_type == "recipe"
        assert modal.request.fingerprint is not None
        assert modal.request.fingerprint.endswith(":recipe")


@pytest.mark.asyncio
async def test_library_prompt_delete_allows_only_one_in_flight_service_call(tmp_path):
    """A second confirmation during a slow delete cannot start a second worker."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Slow delete", author="A", details="d", user_prompt="x"
    )
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    async def delayed_delete(*, mode, prompt_identifier):
        calls.append(prompt_identifier)
        started.set()
        await asyncio.to_thread(release.wait)
        return True

    service.delete_prompt = delayed_delete
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        host.screen.query_one("#prompt-delete-confirm", Button).press()
        for _ in range(100):
            if started.is_set():
                break
            await pilot.pause(0.02)
        assert started.is_set()

        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        assert host.screen is screen
        assert calls == [prompt_id]
        release.set()
        for _ in range(100):
            if screen._library_prompts_view == "list":
                break
            await pilot.pause(0.02)
        assert screen._library_prompts_view == "list"


@pytest.mark.asyncio
async def test_library_prompt_delete_reset_rejects_a_late_modal_dismissal(tmp_path):
    """Leaving an editor clears its pending confirmation before late settlement."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Late result", author="A", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()
        modal = host.screen
        assert isinstance(modal, PromptDeleteConfirmationModal)

        screen._reset_library_prompt_editor_state()
        assert screen._library_prompt_delete_pending_fingerprint is None
        modal.dismiss(PromptDeleteDecision(True, modal.request.fingerprint))
        await pilot.pause()

        assert host.screen is screen
        assert db.fetch_prompt_details(prompt_id) is not None


@pytest.mark.asyncio
async def test_library_prompt_copy_and_delete_fail_closed_for_unknown_future_type(
    tmp_path,
):
    """An explicit future artifact type cannot copy flattened data or delete."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Future artifact", author="A", details="d", user_prompt="x"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert isinstance(screen._library_prompt_detail, dict)
        screen._library_prompt_detail["artifact_type"] = "future_prompt"
        screen.app_instance = host
        host.copy_to_clipboard = copied.append

        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == []
        assert [notice.message for notice in host._notifications] == [
            "This artifact type is unsupported."
        ]
        assert all(
            "copied to clipboard" not in notice.message.lower()
            for notice in host._notifications
        )

        host._notifications.clear()
        screen.query_one("#library-prompt-delete", Button).press()
        await pilot.pause()

        assert host.screen is screen
        assert [notice.message for notice in host._notifications] == [
            "This artifact type is unsupported."
        ]
        assert db.fetch_prompt_details(prompt_id) is not None


@pytest.mark.asyncio
async def test_library_prompt_export_and_duplicate_fail_closed_for_unknown_future_type(
    tmp_path,
):
    """Export/Duplicate share the explicit Prompt/Recipe admission boundary."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Future artifact",
        author="A",
        details="PRIVATE_DETAILS",
        system_prompt="PRIVATE_SYSTEM",
        user_prompt="PRIVATE_USER",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert isinstance(screen._library_prompt_detail, dict)
        screen._library_prompt_detail["artifact_type"] = "future_prompt"
        original_detail = dict(screen._library_prompt_detail)
        original_block_state = screen._library_prompt_block_state
        screen.app_instance = host

        host._notifications.clear()
        stack_size = len(host.screen_stack)
        screen.query_one("#library-prompt-export", Button).press()
        await pilot.pause()

        assert len(host.screen_stack) == stack_size
        assert not any(isinstance(item, FileSave) for item in host.screen_stack)
        assert [notice.message for notice in host._notifications] == [
            "This artifact type is unsupported."
        ]

        host._notifications.clear()
        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id == prompt_id
        assert screen._library_prompt_detail == original_detail
        assert screen._library_prompt_block_state is original_block_state
        assert db.fetch_prompt_details(prompt_id) is not None
        assert db.fetch_prompt_details("Future artifact (copy)") is None
        assert [notice.message for notice in host._notifications] == [
            "This artifact type is unsupported."
        ]
        duplicate_notices = list(host._notifications)
        assert all(
            private not in duplicate_notices[0].message
            for private in ("PRIVATE_DETAILS", "PRIVATE_SYSTEM", "PRIVATE_USER")
        )


@pytest.mark.asyncio
async def test_library_prompt_duplicate_requires_conversion_for_compatibility_artifact(
    tmp_path,
):
    """A compatibility duplicate cannot silently become a legacy Prompt draft."""
    definition = {
        "kind": "foreign_recipe",
        "schema_version": 1,
        "opaque": {"private": "DO_NOT_DISCLOSE"},
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Compatibility recipe",
        author="A",
        details="PRIVATE_DETAILS",
        system_prompt="PRIVATE_SYSTEM",
        user_prompt="PRIVATE_USER",
        prompt_format="structured",
        prompt_schema_version=1,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_block_state is None
        assert screen._current_library_prompt_editor_state().definition_state == (
            "foreign_v1"
        )
        original_detail = dict(screen._library_prompt_detail)
        screen.app_instance = host

        host._notifications.clear()
        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id == prompt_id
        assert screen._library_prompt_detail == original_detail
        assert screen._library_prompt_block_state is None
        assert db.fetch_prompt_details("Compatibility recipe (copy)") is None
        assert [notice.message for notice in host._notifications] == [
            "Convert this compatibility artifact and save it as a new Prompt "
            "before duplicating."
        ]
        duplicate_notices = list(host._notifications)
        assert all(
            private not in duplicate_notices[0].message
            for private in (
                "PRIVATE_DETAILS",
                "PRIVATE_SYSTEM",
                "PRIVATE_USER",
                "DO_NOT_DISCLOSE",
            )
        )


@pytest.mark.asyncio
async def test_library_prompt_copy_rejects_legacy_recipe_without_clipboard_write(
    tmp_path,
):
    """A legacy Recipe cannot be copied as Prompt-looking Markdown."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Legacy Recipe",
        author="A",
        details="PRIVATE_DETAILS",
        system_prompt="PRIVATE_SYSTEM_BODY",
        user_prompt="PRIVATE_USER_BODY",
        prompt_format="legacy",
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        state = screen._current_library_prompt_editor_state()
        assert state.definition_state == "legacy"
        assert state.artifact_type == "recipe"
        assert screen._library_prompt_block_state is None
        screen.app_instance = host
        host.copy_to_clipboard = copied.append

        host._notifications.clear()
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == []
        assert [notice.message for notice in host._notifications] == [
            "This Recipe cannot use this action without losing its type. "
            "Convert and save as a new Prompt first."
        ]
        assert [notice.severity for notice in host._notifications] == ["warning"]
        notice = list(host._notifications)[0].message
        assert "copied" not in notice.lower()
        assert "PRIVATE_DETAILS" not in notice
        assert "PRIVATE_SYSTEM_BODY" not in notice
        assert "PRIVATE_USER_BODY" not in notice


@pytest.mark.asyncio
async def test_library_prompt_export_rejects_legacy_recipe_before_file_save(
    tmp_path,
):
    """A legacy Recipe cannot open FileSave for a type-losing export."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Legacy Recipe",
        author="A",
        details="PRIVATE_DETAILS",
        system_prompt="PRIVATE_SYSTEM_BODY",
        user_prompt="PRIVATE_USER_BODY",
        prompt_format="legacy",
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        state = screen._current_library_prompt_editor_state()
        assert state.definition_state == "legacy"
        assert state.artifact_type == "recipe"
        assert screen._library_prompt_block_state is None
        screen.app_instance = host

        host._notifications.clear()
        stack_size = len(host.screen_stack)
        screen.query_one("#library-prompt-export", Button).press()
        await pilot.pause()

        assert len(host.screen_stack) == stack_size
        assert not any(isinstance(item, FileSave) for item in host.screen_stack)
        assert list(tmp_path.glob("*.md")) == []
        assert [notice.message for notice in host._notifications] == [
            "This Recipe cannot use this action without losing its type. "
            "Convert and save as a new Prompt first."
        ]
        notice = list(host._notifications)[0].message
        assert "exported" not in notice.lower()
        assert "PRIVATE_DETAILS" not in notice
        assert "PRIVATE_SYSTEM_BODY" not in notice
        assert "PRIVATE_USER_BODY" not in notice


@pytest.mark.asyncio
async def test_library_prompt_duplicate_rejects_legacy_recipe_without_state_mutation(
    tmp_path,
):
    """A legacy Recipe must use Convert instead of becoming a legacy Prompt copy."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Legacy Recipe",
        author="A",
        details="PRIVATE_DETAILS",
        system_prompt="PRIVATE_SYSTEM_BODY",
        user_prompt="PRIVATE_USER_BODY",
        prompt_format="legacy",
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        state = screen._current_library_prompt_editor_state()
        assert state.definition_state == "legacy"
        assert state.artifact_type == "recipe"
        assert screen._library_prompt_block_state is None
        original_detail = dict(screen._library_prompt_detail)
        original_dirty = screen._library_prompt_dirty
        screen.app_instance = host

        host._notifications.clear()
        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id == prompt_id
        assert screen._library_prompt_detail == original_detail
        assert screen._library_prompt_block_state is None
        assert screen._library_prompt_dirty is original_dirty
        assert db.fetch_prompt_details("Legacy Recipe (copy)") is None
        assert [notice.message for notice in host._notifications] == [
            "This Recipe cannot use this action without losing its type. "
            "Convert and save as a new Prompt first."
        ]
        notice = list(host._notifications)[0].message
        assert "PRIVATE_DETAILS" not in notice
        assert "PRIVATE_SYSTEM_BODY" not in notice
        assert "PRIVATE_USER_BODY" not in notice


@pytest.mark.asyncio
async def test_library_prompt_copy_uses_unsaved_legacy_create_working_copy(tmp_path):
    """A not-yet-saved create copies its live lanes without requiring an ID."""
    _db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")
        assert screen._selected_prompt_id is None

        screen.query_one("#library-prompt-name", Input).value = "Unsaved create"
        screen.query_one("#library-prompt-author", Input).value = "Draft author"
        screen.query_one("#library-prompt-details", Input).value = "Draft details"
        screen.query_one("#library-prompt-system", TextArea).text = "Draft system"
        screen.query_one("#library-prompt-user", TextArea).text = "Draft user"
        screen.query_one("#library-prompt-keywords", Input).value = "draft, live"
        await pilot.pause()

        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [
            render_prompt_markdown(
                {
                    "name": "Unsaved create",
                    "author": "Draft author",
                    "details": "Draft details",
                    "system_prompt": "Draft system",
                    "user_prompt": "Draft user",
                    "keywords": ["draft", "live"],
                }
            )
        ]


@pytest.mark.asyncio
async def test_library_prompt_copy_uses_unsaved_structured_duplicate_working_copy(
    tmp_path,
):
    """A structured duplicate copies the mounted edited blocks without an ID."""
    definition = {
        "kind": "block_prompt",
        "schema_version": 2,
        "lanes": [
            {
                "id": "system",
                "blocks": [
                    {
                        "id": "role",
                        "title": "Role",
                        "syntax": "markdown",
                        "content": "Original role.",
                    }
                ],
            },
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "markdown",
                        "content": "Original goal.",
                    }
                ],
            },
        ],
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, source_uuid, _msg = db.add_prompt(
        name="Structured source",
        author="A",
        details="d",
        system_prompt="# Role\n\nOriginal role.",
        user_prompt="# Goal\n\nOriginal goal.",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="prompt",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)
    copied: list[str] = []

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == source_uuid
        screen.query_one("#prompt-block-content-role", TextArea).text = "Edited role."
        await pilot.pause()
        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id is None
        assert screen._library_prompt_history_state is None
        assert len(screen.query("#library-prompt-history-collapsible")) == 0
        assert "uuid" not in screen._library_prompt_detail
        assert "version" not in screen._library_prompt_detail
        block_state = screen._library_prompt_block_state
        assert block_state is not None
        assert block_state.definition.lanes[0].blocks[0].content == "Edited role."
        _draft, artifact_fields, _prepared = prepare_prompt_artifact_save(
            block_state,
            artifact_type=block_state.artifact_type,
            include_recipe_starter_content=True,
            request_fields={},
        )
        expected = render_prompt_markdown(
            {
                "name": "Structured source (copy)",
                "author": "A",
                "details": "d",
                "keywords": [],
                **artifact_fields,
            }
        )
        screen.app_instance = host
        host.copy_to_clipboard = copied.append
        screen.query_one("#library-prompt-copy", Button).press()
        await pilot.pause()

        assert copied == [expected]


@pytest.mark.asyncio
async def test_library_shell_create_prompt_row_opens_blank_editor(tmp_path):
    """D1: the Create rail's "New prompt" row opens the in-canvas editor on
    a blank, not-yet-saved record -- empty fields, meta line reads "New
    prompt" (not "Modified … · vN"), prompt_id None."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        assert screen._library_prompts_view == "editor"
        assert screen._selected_prompt_id is None
        assert screen.query_one("#library-prompt-name", Input).value == ""
        assert screen.query_one("#library-prompt-author", Input).value == ""
        assert screen.query_one("#library-prompt-details", Input).value == ""
        assert screen.query_one("#library-prompt-system", TextArea).text == ""
        assert screen.query_one("#library-prompt-user", TextArea).text == ""
        assert screen.query_one("#library-prompt-keywords", Input).value == ""
        meta = screen.query_one("#library-prompt-meta", Static)
        assert str(meta.renderable) == "New prompt"
        assert len(screen.query("#library-prompt-open-existing")) == 0


@pytest.mark.asyncio
async def test_library_shell_create_prompt_save_creates_and_increments_count(tmp_path):
    """D1: Save with a fresh name CREATES via the scope service's create
    path (not update) -- the Prompts rail count increments and the editor
    adopts the new id + switches to the normal "Modified … · vN" meta."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Existing", author="A", details="d", user_prompt="x")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Brand New"
        await pilot.pause()
        screen.query_one("#prompt-lane-add-user", Button).press()
        await pilot.pause()
        content = screen.query_one("#prompt-block-content-block", TextArea)
        content.text = "Hello {name}"
        await pilot.pause()
        content_identity = id(content)
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._selected_prompt_id is not None
        created_id = screen._selected_prompt_id
        persisted = db.fetch_prompt_details(created_id)
        assert persisted is not None
        assert persisted["name"] == "Brand New"
        assert persisted["user_prompt"] == "Hello {name}"

        meta = screen.query_one("#library-prompt-meta", Static)
        for _ in range(150):
            if "Modified" in str(meta.renderable):
                break
            await pilot.pause(0.02)
        assert "Modified" in str(meta.renderable)
        assert "v1" in str(meta.renderable)

        rail_label = ""
        for _ in range(150):
            rail_label = str(screen.query_one("#library-row-browse-prompts").label)
            if "(2)" in rail_label:
                break
            await pilot.pause(0.02)
        assert "(2)" in rail_label
        assert id(screen.query_one("#prompt-block-content-block", TextArea)) == (
            content_identity
        )
        outer_update = screen.query_one("#library-prompt-save", Button)
        shared_update = screen.query_one("#prompt-editor-update-original", Button)
        assert str(outer_update.label) == "Update original"
        assert outer_update.disabled is False
        assert shared_update.disabled is False
        assert (
            str(screen.query_one("#prompt-editor-update-reason", Static).renderable)
            == ""
        )


@pytest.mark.asyncio
async def test_library_shell_create_prompt_save_existing_name_shows_name_in_use(
    tmp_path,
):
    """D1: the three save outcomes apply to create too -- an existing name
    shows the same name-in-use status the update path uses."""
    db, service = _real_prompt_scope_service(tmp_path)
    db.add_prompt(name="Taken", author="A", details="d", user_prompt="x")
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one(f"#library-row-{LIBRARY_ROW_CREATE_PROMPT}").press()
        await _wait_for_selector(screen, pilot, "#library-prompt-name")

        screen.query_one("#library-prompt-name", Input).value = "Taken"
        await pilot.pause()
        screen.query_one("#library-prompt-user", TextArea).text = "hi"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert (
            status_text
            == "Name already in use — pick another or open the existing prompt."
        )
        assert screen._selected_prompt_id is None
        _prompts, _tp, _cp, total = db.list_prompts()
        assert total == 1


@pytest.mark.asyncio
async def test_library_prompt_duplicate_button_between_copy_and_delete(tmp_path):
    """U3: the Duplicate action sits between Copy and Delete in the editor's
    action row."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="X", author="A", details="d", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        actions = screen.query_one("#library-prompt-editor-actions")
        ids = [button.id for button in actions.query(Button)]
        assert (
            ids.index("library-prompt-copy")
            < ids.index("library-prompt-duplicate")
            < ids.index("library-prompt-delete")
        )


@pytest.mark.asyncio
async def test_library_prompt_duplicate_prefills_blank_editor_and_saves_distinct_prompt(
    tmp_path,
):
    """U3: Duplicate opens the editor on a NEW blank-id record pre-filled
    from the current prompt's fields, name "<name> (copy)", dirty/unsaved.
    Reuses the D1 create path on Save -- a distinct prompt is created."""
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, _uuid, _msg = db.add_prompt(
        name="Original",
        author="Alice",
        details="d",
        system_prompt="sys",
        user_prompt="usr",
        keywords=["a", "b"],
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)

        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id is None
        assert screen._library_prompt_dirty is True
        assert (
            screen.query_one("#library-prompt-name", Input).value == "Original (copy)"
        )
        assert screen.query_one("#library-prompt-system", TextArea).text == "sys"
        assert screen.query_one("#library-prompt-user", TextArea).text == "usr"
        assert screen.query_one("#library-prompt-author", Input).value == "Alice"

        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        status_text = await _wait_for_prompt_status(screen, pilot)
        assert status_text == "Saved."
        assert screen._selected_prompt_id is not None
        assert screen._selected_prompt_id != prompt_id

        _prompts, _tp, _cp, total = db.list_prompts()
        assert total == 2
        original = db.fetch_prompt_details(prompt_id)
        assert original["name"] == "Original"


@pytest.mark.asyncio
async def test_library_recipe_duplicate_outer_save_is_honest_and_preserves_starter(
    tmp_path,
):
    db, service = _real_prompt_scope_service(tmp_path)
    definition = {
        "kind": "block_recipe",
        "schema_version": 2,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "markdown",
                        "content": "Starter",
                        "mapping_hint": "State the outcome.",
                    }
                ],
            },
        ],
    }
    recipe_id, _uuid, _msg = db.add_prompt(
        name="Original Recipe",
        author="Author",
        details="Details",
        user_prompt="# Goal\n\nStarter",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, recipe_id)

        screen.query_one("#library-prompt-duplicate", Button).press()
        await pilot.pause()
        save = screen.query_one("#library-prompt-save", Button)
        assert str(save.label) == "Save Recipe"
        screen.query_one("#library-prompt-recipe-starter", Checkbox).value = True
        save.press()
        assert await _wait_for_prompt_status(screen, pilot) == "Saved."

        duplicate = db.fetch_prompt_details("Original Recipe (copy)")
        assert duplicate is not None
        assert duplicate["artifact_type"] == "recipe"
        persisted_definition = json.loads(duplicate["prompt_definition"])
        assert persisted_definition["lanes"][1]["blocks"][0]["content"] == "Starter"


# ---------------------------------------------------------------------------
# Task 8b, Group 3: D3 (Open existing) + D4 (import Browse…)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_prompt_open_existing_button_shows_only_in_name_in_use_state_and_opens_it(
    tmp_path,
):
    """D3: the "Open existing" button appears ONLY in the name-in-use
    state, and pressing it loads the colliding prompt into the editor."""
    db, service = _real_prompt_scope_service(tmp_path)
    alpha_id, _uuid, _msg = db.add_prompt(
        name="Alpha", author="A", details="d-alpha", user_prompt="x"
    )
    beta_id, _uuid, _msg = db.add_prompt(
        name="Beta", author="B", details="d-beta", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        assert len(screen.query("#library-prompt-open-existing")) == 0

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()

        status_text = await _wait_for_prompt_status(screen, pilot)
        assert (
            status_text
            == "Name already in use — pick another or open the existing prompt."
        )

        for _ in range(150):
            if len(screen.query("#library-prompt-open-existing")) > 0:
                break
            await pilot.pause(0.02)
        open_existing = screen.query_one("#library-prompt-open-existing", Button)

        open_existing.press()
        await pilot.pause()
        for _ in range(150):
            if screen._selected_prompt_id == alpha_id:
                break
            await pilot.pause(0.02)
        assert screen._selected_prompt_id == alpha_id

        for _ in range(150):
            if screen.query_one("#library-prompt-name", Input).value == "Alpha":
                break
            await pilot.pause(0.02)
        assert screen.query_one("#library-prompt-name", Input).value == "Alpha"
        assert screen.query_one("#library-prompt-details", Input).value == "d-alpha"
        assert len(screen.query("#library-prompt-open-existing")) == 0


@pytest.mark.asyncio
async def test_library_prompt_open_existing_resolves_offending_name_not_drifted_field(
    tmp_path,
):
    """Task 8b Fix wave 1 (Minor): once the name-in-use status is showing,
    "Open existing" stays mounted even if the user keeps typing in the
    Name field without re-saving -- it must still resolve against the
    name that actually collided ("Alpha"), not whatever text is currently
    sitting in the (drifted, never re-saved) Name field."""
    db, service = _real_prompt_scope_service(tmp_path)
    alpha_id, _uuid, _msg = db.add_prompt(
        name="Alpha", author="A", details="d-alpha", user_prompt="x"
    )
    beta_id, _uuid, _msg = db.add_prompt(
        name="Beta", author="B", details="d-beta", user_prompt="y"
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, beta_id)

        screen.query_one("#library-prompt-name", Input).value = "Alpha"
        await pilot.pause()
        screen.query_one("#library-prompt-save", Button).press()
        await pilot.pause()
        await _wait_for_prompt_status(screen, pilot)
        for _ in range(150):
            if len(screen.query("#library-prompt-open-existing")) > 0:
                break
            await pilot.pause(0.02)
        assert screen._library_prompt_name_in_use == "Alpha"

        # Drift: the user keeps editing the Name field to something that
        # collides with NEITHER prompt, without pressing Save again -- the
        # status/button never clear (nothing re-checks on plain typing).
        screen.query_one("#library-prompt-name", Input).value = "Not A Real Prompt"
        await pilot.pause()
        assert len(screen.query("#library-prompt-open-existing")) > 0

        screen.query_one("#library-prompt-open-existing", Button).press()
        await pilot.pause()
        for _ in range(150):
            if screen._selected_prompt_id == alpha_id:
                break
            await pilot.pause(0.02)

        # Resolves to the prompt that ACTUALLY collided ("Alpha"), not a
        # failed/empty lookup for the drifted "Not A Real Prompt" text.
        assert screen._selected_prompt_id == alpha_id
        for _ in range(150):
            if screen.query_one("#library-prompt-name", Input).value == "Alpha":
                break
            await pilot.pause(0.02)
        assert screen.query_one("#library-prompt-name", Input).value == "Alpha"


@pytest.mark.asyncio
async def test_library_prompts_import_browse_button_fills_path_input(tmp_path):
    """D4: Browse… (beside the import path Input) opens the same FileOpen
    dialog the media-ingest form's Browse action uses; on pick, it fills
    the path Input."""
    db, service = _real_prompt_scope_service(tmp_path)
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    picked_file = tmp_path / "prompts.json"
    picked_file.write_text("[]", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompts_list(screen, pilot)

        screen.query_one("#library-prompts-import", Button).press()
        await pilot.pause()
        assert screen.query_one("#library-prompts-import-browse", Button)

        push_calls = _fake_import_dialog_result(screen, picked_file)
        screen.query_one("#library-prompts-import-browse", Button).press()
        await pilot.pause()

        assert push_calls and isinstance(push_calls[0], FileOpen)
        assert screen.query_one("#library-prompts-import-path", Input).value == str(
            picked_file
        )


# ---------------------------------------------------------------------------
# Task 12: editor "Use in Console" guard functions. The successful producer
# and consumer route is exercised by the full production application in
# ``Tests/ProductionApp/test_personas_library_root_state.py``.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_library_recipe_use_in_console_detaches_source_history(tmp_path):
    definition = {
        "kind": "block_recipe",
        "schema_version": 2,
        "lanes": [
            {"id": "system", "blocks": []},
            {
                "id": "user",
                "blocks": [
                    {
                        "id": "goal",
                        "title": "Goal",
                        "syntax": "freeform",
                        "content": "Recipe content",
                    }
                ],
            },
        ],
    }
    db, service = _real_prompt_scope_service(tmp_path)
    prompt_id, prompt_uuid, _message = db.add_prompt(
        name="Recipe source",
        author="Author",
        details="Details",
        user_prompt="Recipe content",
        prompt_format="structured",
        prompt_schema_version=2,
        prompt_definition=definition,
        artifact_type="recipe",
    )
    app = _build_test_app()
    _wire_empty_non_prompt_services(app)
    app.prompt_scope_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_prompt_editor(screen, pilot, prompt_id)
        assert screen._library_prompt_history_state is not None
        assert screen._library_prompt_history_state.prompt_uuid == prompt_uuid

        screen.query_one("#library-prompt-insert-console", Button).press()
        await pilot.pause()

        assert screen._selected_prompt_id is None
        assert screen._library_prompt_version is None
        assert screen._library_prompt_history_state is None
        assert len(screen.query("#library-prompt-history-collapsible")) == 0
        assert "uuid" not in screen._library_prompt_detail
        assert "version" not in screen._library_prompt_detail
        assert screen._library_prompt_dirty is True


def test_library_prompt_insert_console_refuses_while_dirty():
    """An unsaved in-progress edit refuses the action outright (rather than
    staging text that a vetoed navigation would later fire unexpectedly on
    some unrelated future Console visit) -- the prompt is never lost either
    way, since the edit simply stays in the still-open editor."""
    notify = Mock()
    stage = Mock()
    read_fields = Mock()
    screen = SimpleNamespace(
        _library_prompts_view="editor",
        _library_prompt_dirty=True,
        app_instance=SimpleNamespace(
            notify=notify,
            stage_console_prompt_insert=stage,
        ),
        _current_library_prompt_editor_state=lambda: _structured_editor_state(),
        _read_library_prompt_editor_fields=read_fields,
    )
    event = SimpleNamespace(stop=Mock())

    LibraryScreen.handle_library_prompt_insert_console(screen, event)

    event.stop.assert_called_once_with()
    notify.assert_called_once_with(
        "Save your changes before using this prompt in Console.",
        severity="warning",
    )
    read_fields.assert_not_called()
    stage.assert_not_called()


def _library_prompt_target() -> ConsolePromptTargetProjection:
    return ConsolePromptTargetProjection(
        target_session_id="console-session",
        system_fingerprint=fingerprint_system_text("current system"),
    )


class _LibraryPromptHandlerHarness(SimpleNamespace):
    _stage_library_prompt_for_console = LibraryScreen._stage_library_prompt_for_console


def test_library_block_editor_apply_preserves_both_selected_lanes():
    state = _structured_editor_state()
    apply_working_copy = Mock()
    screen = SimpleNamespace(
        _library_prompt_block_state=None,
        _apply_library_prompt_working_copy=apply_working_copy,
    )
    event = SimpleNamespace(
        stop=Mock(),
        state=state.block_editor_state,
        apply_system=True,
        system_prompt="System {customer}",
        apply_user=True,
        user_prompt="User {customer}",
    )

    LibraryScreen.on_prompt_block_editor_apply_requested(screen, event)

    event.stop.assert_called_once_with()
    apply_working_copy.assert_called_once_with(
        state=state.block_editor_state,
        system_prompt="System {customer}",
        user_prompt="User {customer}",
    )


@pytest.mark.parametrize("projection_error", [False, True], ids=["absent", "error"])
def test_library_prompt_insert_console_refuses_without_published_target(
    projection_error,
):
    notify = Mock()
    stage = Mock()
    push_screen = Mock()

    def target_getter():
        if projection_error:
            raise RuntimeError("private projection failure")
        return None

    screen = _LibraryPromptHandlerHarness(
        _library_prompts_view="editor",
        _library_prompt_dirty=False,
        app_instance=SimpleNamespace(
            notify=notify,
            stage_console_prompt_insert=stage,
            console_prompt_target_projection=target_getter,
            push_screen=push_screen,
        ),
        _current_library_prompt_editor_state=lambda: _structured_editor_state(),
        _read_library_prompt_editor_fields=lambda: (
            "Prompt",
            "",
            "",
            "System",
            "User",
            "",
        ),
        _sanitize_note_content=lambda value, *, max_length: value,
    )
    event = SimpleNamespace(stop=Mock())

    LibraryScreen.handle_library_prompt_insert_console(screen, event)

    event.stop.assert_called_once_with()
    notify.assert_called_once_with(
        "Open Console once, then retry Use in Console.",
        severity="warning",
    )
    stage.assert_not_called()
    push_screen.assert_not_called()


def test_library_prompt_insert_console_system_only_uses_shared_dialog():
    notify = Mock()
    stage = Mock()
    pushed: list[tuple[object, object]] = []

    def push_screen(screen, *, callback):
        pushed.append((screen, callback))

    target = _library_prompt_target()
    screen = _LibraryPromptHandlerHarness(
        _library_prompts_view="editor",
        _library_prompt_dirty=False,
        app_instance=SimpleNamespace(
            notify=notify,
            stage_console_prompt_insert=stage,
            console_prompt_target_projection=lambda: target,
            push_screen=push_screen,
        ),
        _current_library_prompt_editor_state=lambda: _structured_editor_state(),
        _read_library_prompt_editor_fields=lambda: (
            "System Only",
            "",
            "",
            "You are {role}.",
            "",
            "",
        ),
        _sanitize_note_content=lambda value, *, max_length: value,
    )
    event = SimpleNamespace(stop=Mock())

    LibraryScreen.handle_library_prompt_insert_console(screen, event)

    assert len(pushed) == 1
    dialog, callback = pushed[0]
    assert isinstance(dialog, PromptVariablesDialog)
    assert dialog.request.destination == "append_active"
    assert dialog.request.target_session_id == "console-session"
    assert dialog.request.system_text == "You are {role}."
    assert dialog.request.user_text is None
    stage.assert_not_called()

    application = PromptVariableApplication(
        system_text="You are concise.",
        user_text=None,
        apply_system=True,
        apply_user=False,
        destination="append_active",
        target_session_id=target.target_session_id,
        composer_fingerprint=None,
        system_fingerprint=target.system_fingerprint,
    )
    callback(application)
    stage.assert_called_once_with(application)
    notify.assert_not_called()


@pytest.mark.parametrize("structured", [False, True], ids=["legacy", "supported-v2"])
def test_library_prompt_insert_console_stages_safe_prompt_states(structured):
    notify = Mock()
    stage = Mock()
    editor_state = (
        _structured_editor_state()
        if structured
        else build_prompt_editor_state(
            {
                "id": 41,
                "name": "Legacy Prompt",
                "system_prompt": "Stay direct.",
                "user_prompt": "Ship it.",
            }
        )
    )
    target = _library_prompt_target()
    screen = _LibraryPromptHandlerHarness(
        _library_prompts_view="editor",
        _library_prompt_dirty=False,
        app_instance=SimpleNamespace(
            notify=notify,
            stage_console_prompt_insert=stage,
            console_prompt_target_projection=lambda: target,
            push_screen=Mock(),
        ),
        _current_library_prompt_editor_state=lambda: editor_state,
        _read_library_prompt_editor_fields=lambda: (
            "Prompt",
            "",
            "",
            "",
            "Ship it.",
            "",
        ),
        _sanitize_note_content=lambda value, *, max_length: value,
    )
    event = SimpleNamespace(stop=Mock())

    LibraryScreen.handle_library_prompt_insert_console(screen, event)

    event.stop.assert_called_once_with()
    notify.assert_not_called()
    application = stage.call_args.args[0]
    assert isinstance(application, PromptVariableApplication)
    assert application.destination == "append_active"
    assert application.target_session_id == "console-session"
    assert application.apply_system is False
    assert application.user_text == "Ship it."


def test_library_prompt_insert_console_variables_share_one_dialog_and_use_original():
    stage = Mock()
    pushed: list[tuple[PromptVariablesDialog, object]] = []

    def push_screen(dialog, *, callback):
        pushed.append((dialog, callback))

    target = _library_prompt_target()
    screen = _LibraryPromptHandlerHarness(
        _library_prompts_view="editor",
        _library_prompt_dirty=False,
        app_instance=SimpleNamespace(
            notify=Mock(),
            stage_console_prompt_insert=stage,
            console_prompt_target_projection=lambda: target,
            push_screen=push_screen,
        ),
        _current_library_prompt_editor_state=lambda: _structured_editor_state(),
        _read_library_prompt_editor_fields=lambda: (
            "Prompt",
            "",
            "",
            "Address {customer}.",
            "Draft for {customer}.",
            "",
        ),
        _sanitize_note_content=lambda value, *, max_length: value,
    )

    LibraryScreen.handle_library_prompt_insert_console(
        screen,
        SimpleNamespace(stop=Mock()),
    )

    assert len(pushed) == 1
    dialog, callback = pushed[0]
    assert dialog.request.system_text == "Address {customer}."
    assert dialog.request.user_text == "Draft for {customer}."
    assert dialog.request.destination == "append_active"
    stage.assert_not_called()

    original = PromptVariableApplication(
        system_text=None,
        user_text="Draft for {customer}.",
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=target.target_session_id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )
    callback(original)
    stage.assert_called_once_with(original)


@pytest.mark.parametrize("projection_error", [False, True], ids=["absent", "error"])
def test_library_prompt_dialog_rechecks_projection_before_staging(projection_error):
    notify = Mock()
    stage = Mock()
    pushed: list[tuple[PromptVariablesDialog, object]] = []
    target = _library_prompt_target()
    projection_calls = 0

    def target_getter():
        nonlocal projection_calls
        projection_calls += 1
        if projection_calls == 1:
            return target
        if projection_error:
            raise RuntimeError("private projection failure")
        return None

    def push_screen(dialog, *, callback):
        pushed.append((dialog, callback))

    screen = _LibraryPromptHandlerHarness(
        _library_prompts_view="editor",
        _library_prompt_dirty=False,
        app_instance=SimpleNamespace(
            notify=notify,
            stage_console_prompt_insert=stage,
            console_prompt_target_projection=target_getter,
            push_screen=push_screen,
        ),
        _current_library_prompt_editor_state=lambda: _structured_editor_state(),
        _read_library_prompt_editor_fields=lambda: (
            "Prompt",
            "",
            "",
            "System {customer}",
            "User {customer}",
            "",
        ),
        _sanitize_note_content=lambda value, *, max_length: value,
    )
    LibraryScreen.handle_library_prompt_insert_console(
        screen,
        SimpleNamespace(stop=Mock()),
    )
    _dialog, callback = pushed[0]
    application = PromptVariableApplication(
        system_text=None,
        user_text="User Acme",
        apply_system=False,
        apply_user=True,
        destination="append_active",
        target_session_id=target.target_session_id,
        composer_fingerprint=None,
        system_fingerprint=None,
    )

    callback(application)

    stage.assert_not_called()
    notify.assert_called_once_with(
        "Open Console once, then retry Use in Console.",
        severity="warning",
    )
