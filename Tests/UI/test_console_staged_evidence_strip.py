"""Console staged-evidence strip: visibility, un-stage, and consume-on-send.

Covers RAG-40 (staging was invisible unless the Inspector rail happened to be
open), the sticky-staging defect (`_consume_pending_console_launch` never
cleared its field, so one staged bundle rode every later send), and the
hardcoded `staged_source_count=1` chip lie.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from Tests.UI.test_console_dictionary_send_integration import _CapturingGateway
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_destination_shells import _build_test_app, _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.citation_evidence_models import (
    EvidenceBundle,
    EvidenceReference,
)
from tldw_chatbook.Chat.console_display_state import (
    ConsoleStagedEvidenceStripState,
    build_console_staged_evidence_strip_state,
    console_prompted_evidence_text,
    console_prompted_source_count,
    console_staged_source_count,
)
from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch
from tldw_chatbook.Event_Handlers.Chat_Events.chat_rag_events import (
    LocalRagContextResult,
)
from tldw_chatbook.UI.Console_Modules import workspace as workspace_module
from tldw_chatbook.UI.Console_Modules import retrieval as retrieval_module
from tldw_chatbook.Widgets.Console.console_staged_context import (
    ConsoleStagedContextTray,
)
from tldw_chatbook.Widgets.Console.console_staged_evidence_strip import (
    ConsoleStagedEvidenceStrip,
)

STRIP_ID = "#console-staged-evidence-strip"
UNSTAGE_ID = "#console-unstage-evidence"


def _reference(
    index: int,
    *,
    title: str | None = None,
    status: str = "available",
    source_owner: str = "local",
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=f"S{index}",
        source_id=f"media-{index}",
        source_type="media",
        title=title if title is not None else f"Source {index}",
        snippet=f"Body {index}",
        authority_label="local",
        status=status,
        source_owner=source_owner,
    )


def _mixed_launch() -> ConsoleLiveWorkLaunch:
    """Four staged references, of which only two can ever reach the prompt."""
    bundle = EvidenceBundle(
        bundle_id="bundle-mixed",
        query="question",
        source="Library Search/RAG",
        references=(
            _reference(1),
            _reference(2, status="blocked"),
            _reference(3),
            _reference(4, source_owner="server"),
        ),
    )
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={"query": "question", "evidence_bundle": bundle.to_payload()},
        status="staged",
    )


def _bundle(count: int, *, first_title: str | None = None) -> EvidenceBundle:
    return EvidenceBundle(
        bundle_id="bundle-1",
        query="question",
        source="Library Search/RAG",
        references=tuple(
            _reference(index, title=first_title if index == 1 else None)
            for index in range(1, count + 1)
        ),
    )


def _launch(count: int, *, first_title: str | None = None) -> ConsoleLiveWorkLaunch:
    return ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={
            "query": "question",
            "evidence_bundle": _bundle(count, first_title=first_title).to_payload(),
        },
        status="staged",
        recovery="Review citations before sending.",
    )


# --------------------------------------------------------------------------
# (a)+(b)+(f) pure display state
# --------------------------------------------------------------------------


def test_strip_state_hidden_when_nothing_is_staged() -> None:
    state = build_console_staged_evidence_strip_state(None)
    assert state.visible is False
    assert state.rows == ()
    assert state.notice == ""


def test_strip_state_lists_bundle_references_with_overflow() -> None:
    state = build_console_staged_evidence_strip_state(_launch(5))
    assert state.visible is True
    assert len(state.rows) == 3
    assert [row.title for row in state.rows] == ["Source 1", "Source 2", "Source 3"]
    assert all(row.source == "media" for row in state.rows)
    assert state.overflow == "+2 more"
    assert "5 sources" in state.heading


def test_strip_state_escapes_untrusted_library_titles() -> None:
    state = build_console_staged_evidence_strip_state(
        _launch(1, first_title="[bold]pwn[/bold] <script>")
    )
    row = state.rows[0]
    # Markup brackets survive verbatim (they are rendered with markup off) and
    # angle brackets are html-escaped exactly like the Inspector tray does.
    assert row.title == "[bold]pwn[/bold] &lt;script&gt;"


def test_strip_state_falls_back_to_the_launch_when_no_bundle_is_attached() -> None:
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Transformer notes",
        payload={"source_id": "note-1"},
        status="ready",
    )
    state = build_console_staged_evidence_strip_state(launch)
    assert state.visible is True
    assert len(state.rows) == 1
    assert state.rows[0].title == "Transformer notes"
    assert state.rows[0].source == "Library Search/RAG"
    assert state.overflow == ""
    assert "1 source" in state.heading


def test_strip_state_renders_the_one_send_sent_notice() -> None:
    state = build_console_staged_evidence_strip_state(None, sent_source_count=4)
    assert state.visible is True
    assert state.rows == ()
    assert state.notice == "Evidence sent with this message · 4 sources"


def test_strip_state_prefers_new_staging_over_a_stale_sent_notice() -> None:
    state = build_console_staged_evidence_strip_state(_launch(2), sent_source_count=4)
    assert state.notice == ""
    assert len(state.rows) == 2


def test_staged_source_count_is_the_bundle_reference_count() -> None:
    assert console_staged_source_count(None) == 0
    assert console_staged_source_count(_launch(4)) == 4
    bundleless = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG", title="Notes", payload={"source_id": "n1"}
    )
    assert console_staged_source_count(bundleless) == 1


def test_prompted_source_count_applies_the_captures_own_filter() -> None:
    """"How much is staged" and "how much reaches the model" are different."""
    launch = _mixed_launch()
    # Four staged...
    assert console_staged_source_count(launch) == 4
    # ...but one is blocked and one is server-owned, so two are prompted.
    assert console_prompted_source_count(launch) == 2
    assert console_prompted_source_count(None) == 0


def test_prompted_evidence_text_applies_the_same_filter_as_the_count() -> None:
    """task-6: the text the context/cost estimates count must match exactly
    what `console_prompted_source_count` counts -- same filter, same
    references. `capture_console_staged_evidence_for_chat` re-validates
    identity/authority but never re-fetches content, so each reference's
    (already truncated) `snippet` is exactly what reaches the prompt."""
    launch = _mixed_launch()
    text = console_prompted_evidence_text(launch)
    assert "Body 1" in text
    assert "Body 3" in text
    assert "Body 2" not in text  # blocked
    assert "Body 4" not in text  # server-owned
    assert console_prompted_evidence_text(None) == ""


def test_prompted_evidence_text_is_empty_for_a_bundleless_launch() -> None:
    bundleless = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG", title="Notes", payload={"source_id": "n1"}
    )
    assert console_prompted_evidence_text(bundleless) == ""


# --------------------------------------------------------------------------
# widget rendering
# --------------------------------------------------------------------------


class _StripApp(App):
    def __init__(self, state: ConsoleStagedEvidenceStripState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStagedEvidenceStrip(self._state, id=STRIP_ID.lstrip("#"))


@pytest.mark.asyncio
async def test_strip_widget_is_hidden_and_actionless_without_staging() -> None:
    app = _StripApp(build_console_staged_evidence_strip_state(None))
    async with app.run_test(size=(120, 12)):
        strip = app.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is False
        assert not list(strip.query(UNSTAGE_ID))


@pytest.mark.asyncio
async def test_strip_widget_renders_escaped_rows_overflow_and_one_unstage() -> None:
    app = _StripApp(
        build_console_staged_evidence_strip_state(
            _launch(5, first_title="[bold]pwn[/bold]")
        )
    )
    async with app.run_test(size=(120, 12)):
        strip = app.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is True
        row = strip.query_one("#console-staged-evidence-row-0", Static)
        assert "[bold]pwn[/bold]" in str(row.renderable)
        # Escaping is TERMINAL: markup parsing stays off at the render step.
        assert row._render_markup is False
        assert len(strip.query(".console-staged-evidence-row")) == 3
        overflow = strip.query_one("#console-staged-evidence-overflow", Static)
        assert str(overflow.renderable) == "+2 more"
        assert len(strip.query(UNSTAGE_ID)) == 1


@pytest.mark.asyncio
async def test_strip_widget_sync_state_swaps_rows_for_the_sent_notice() -> None:
    app = _StripApp(build_console_staged_evidence_strip_state(_launch(2)))
    async with app.run_test(size=(120, 12)) as pilot:
        strip = app.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        strip.sync_state(
            build_console_staged_evidence_strip_state(None, sent_source_count=2)
        )
        await pilot.pause()
        notice = strip.query_one("#console-staged-evidence-notice", Static)
        assert str(notice.renderable) == "Evidence sent with this message · 2 sources"
        assert not list(strip.query(UNSTAGE_ID))

        strip.sync_state(build_console_staged_evidence_strip_state(None))
        await pilot.pause()
        assert strip.display is False


# --------------------------------------------------------------------------
# screen wiring
# --------------------------------------------------------------------------


def _strip_text(screen) -> str:
    strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
    return "\n".join(str(child.renderable) for child in strip.query(Static))


class _StagedEvidenceStripPanelHost(App):
    """`ConsoleStagedEvidenceStrip` plus the TWO production CSS rules this
    pin cares about: the shared `.ds-panel { min-height: 3 }` every Console
    panel inherits (`classes="ds-panel"` is how the real screen mounts this
    widget -- see `chat_screen.py`'s `compose_content`), and this widget's
    own `#console-staged-evidence-strip` override plus its children's row
    heights.

    `_StripApp` above (and `ConsoleHarness`, used by the rest of this file)
    never loads the app's real stylesheet -- only a widget's OWN Python-level
    `self.styles.*` assignments or `DEFAULT_CSS` take effect there, so
    neither can reproduce a bug that lives in the external `.tcss` bundle.
    Mirrors `test_console_session_tab_strip.py`'s `_PaddedTabStripHost`
    (RAG-47): reproduce just the ONE production CSS this test needs, not the
    whole multi-thousand-line bundle.
    """

    CSS = """
    .ds-panel {
        height: auto;
        min-height: 3;
    }

    #console-staged-evidence-strip {
        width: 100%;
        min-width: 0;
        height: auto;
        min-height: 1;
        max-height: 6;
        border: none;
        padding: 0 1;
        margin: 0;
    }

    .console-staged-evidence-header {
        width: 100%;
        height: 1;
        min-height: 1;
        layout: horizontal;
    }

    .console-staged-evidence-heading {
        width: 1fr;
        min-width: 0;
        height: 1;
    }

    .console-staged-evidence-unstage {
        width: auto;
        min-width: 9;
        height: 1;
        min-height: 1;
    }

    .console-staged-evidence-row {
        width: 100%;
        min-width: 0;
        height: 1;
        min-height: 1;
    }
    """

    def __init__(self, state: ConsoleStagedEvidenceStripState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStagedEvidenceStrip(
            self._state, id=STRIP_ID.lstrip("#"), classes="ds-panel"
        )


@pytest.mark.asyncio
async def test_strip_has_no_blank_filler_rows_for_small_staged_counts() -> None:
    """M3 (final review): the strip is mounted with `classes="ds-panel"`
    (the shared Console panel rule, `.ds-panel { min-height: 3 }`), so with
    only 1-2 rows of real content (one header row + N evidence rows)
    Textual padded the box up to 3 rows with blank filler beneath the last
    row. Sibling to the `#console-status-chips` `min-height: 1` precedent,
    which already overrides `.ds-panel` for that other small fixed-content
    Console panel. Pin the strip's rendered region height against its own
    content height -- no filler.
    """
    app = _StagedEvidenceStripPanelHost(
        build_console_staged_evidence_strip_state(_launch(1))
    )
    async with app.run_test(size=(120, 24)):
        strip = app.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is True
        assert not list(strip.query("#console-staged-evidence-overflow"))
        # Content: 1 header row + 1 evidence row, no overflow line.
        content_rows = 1 + len(strip.query(".console-staged-evidence-row"))
        assert strip.region.height == content_rows, (
            "strip padded with blank filler rows beyond its own content -- "
            f"region.height={strip.region.height}, content_rows={content_rows}"
        )


@pytest.mark.asyncio
async def test_console_run_staging_fans_out_to_strip_and_truthful_chip() -> None:
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is False

        screen._retrieval._stage_console_library_rag_launch(_launch(4))
        await pilot.pause()

        strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is True
        text = _strip_text(screen)
        assert "Source 1" in text
        assert "+1 more" in text
        sources_chip = screen.query_one("#console-sources-label", Static)
        # task-15791: 7dbbc401b (TASK-2154 UX remediation, owner-driven)
        # shortened the chip to "Sources: N" -- the truthful COUNT is the
        # contract; the " staged" suffix went with the same pass that
        # renamed "RAG:" to "Library search:".
        assert "Sources: 4" in str(sources_chip.renderable)


@pytest.mark.asyncio
async def test_console_unstage_clears_context_strip_chip_and_tray() -> None:
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        screen._retrieval._stage_console_library_rag_launch(_launch(3))
        await pilot.pause()

        screen.query_one(UNSTAGE_ID, Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen._pending_console_launch_context is None
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is False
        sources_chip = screen.query_one("#console-sources-label", Static)
        assert "Sources: 0" in str(sources_chip.renderable)
        tray = screen.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        )
        assert tray.state.is_empty is True


@pytest.mark.asyncio
async def test_console_unstage_click_heals_a_stale_strip_when_context_already_none() -> None:
    """M4 (final review): the handler's early return (`if
    self._pending_console_launch_context is None: return`) fires with no
    self-heal when the field was already cleared out from under a strip
    that never got resynced -- clicking Un-stage was a silent no-op that
    left the stale rows on screen. Reproduce that staleness directly (clear
    the field WITHOUT going through the normal sync path, exactly what a
    send's consume-on-send clear or another surface's write could leave
    behind) and assert the click still heals the strip instead of dead-ending.
    """
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        screen._retrieval._stage_console_library_rag_launch(_launch(3))
        await pilot.pause()
        strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is True

        # Simulate staleness: the context field is cleared directly, never
        # going through `_sync_console_pending_launch_surfaces`, so the
        # mounted strip's own `.state` (and its Un-stage button) is still
        # the old staged snapshot.
        screen._pending_console_launch_context = None

        screen.query_one(UNSTAGE_ID, Button).press()
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is False


@pytest.mark.asyncio
async def test_console_library_u_key_handoff_populates_the_strip() -> None:
    """The Library `u` handoff writes the SAME field, so the strip must list it."""
    from tldw_chatbook.UI.Navigation.pending_handoff_store import HandoffChannel

    app = _build_test_app()
    app.pending_handoffs.stage(
        HandoffChannel.CONSOLE_LIVE_WORK,
        _launch(2).to_pending_payload(),
    )
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is True
        assert "Source 1" in _strip_text(screen)
        sources_chip = screen.query_one("#console-sources-label", Static)
        assert "Sources: 2" in str(sources_chip.renderable)


async def _submit(screen, draft: str):
    controller = screen._ensure_console_chat_controller()
    controller.provider_gateway = _CapturingGateway()
    controller._agent_runtime_enabled = False
    return await controller.submit_draft(draft)


@pytest.mark.asyncio
async def test_console_send_consumes_staging_and_shows_the_sent_transient(
    monkeypatch,
) -> None:
    app = _build_test_app()
    launch = _launch(2)
    capture = AsyncMock(
        return_value=LocalRagContextResult(
            context="[S1] MEDIA — Source 1\nBody 1",
            citation_builder=None,
        )
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        result = await _submit(screen, "question")
        assert result.accepted is True
        await pilot.pause()

        # The staged bundle is gone -- the modal's "Running again replaces it"
        # promise is now true.
        assert screen._pending_console_launch_context is None
        assert "Evidence sent with this message · 2 sources" in _strip_text(screen)
        sources_chip = screen.query_one("#console-sources-label", Static)
        assert "Sources: 0" in str(sources_chip.renderable)

        # A SECOND send captures nothing: the evidence no longer rides along.
        await _submit(screen, "follow up")
        await pilot.pause()
        assert capture.await_count == 2
        assert capture.await_args_list[0].args[1] is launch
        assert capture.await_args_list[1].args[1] is None
        # The transient is one-send only.
        assert "Evidence sent with this message" not in _strip_text(screen)


@pytest.mark.asyncio
async def test_console_sent_notice_counts_only_what_reached_the_model(
    monkeypatch,
) -> None:
    """4 staged, 2 promptable -- the notice must claim 2, not 4."""
    app = _build_test_app()
    launch = _mixed_launch()
    capture = AsyncMock(
        return_value=LocalRagContextResult(
            context="[S1] MEDIA — Source 1\nBody 1",
            citation_builder=None,
        )
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()
        sources_chip = screen.query_one("#console-sources-label", Static)
        assert "Sources: 4" in str(sources_chip.renderable)

        await _submit(screen, "question")
        await pilot.pause()

        assert "Evidence sent with this message · 2 sources" in _strip_text(screen)


@pytest.mark.asyncio
async def test_console_sent_notice_prefers_the_exact_prompted_entry_count(
    monkeypatch,
) -> None:
    """The repair contract's ordinals are one-per-prompt-entry: the exact count."""
    from tldw_chatbook.Chat.citation_repair import CitationRepairContract
    from tldw_chatbook.Chat.citation_trace_models import MarkerNamespace

    app = _build_test_app()
    launch = _mixed_launch()
    context = "[S1] MEDIA — Source 1\nBody 1"
    capture = AsyncMock(
        return_value=LocalRagContextResult(
            context=context,
            citation_builder=None,
            citation_repair_contract=CitationRepairContract(
                schema_version=1,
                marker_namespace=MarkerNamespace.CHATBOOK_S_V1,
                # Prompt authority dropped one of the two eligible refs.
                allowed_ordinals=(1,),
                evidence_context=context,
            ),
        )
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        await _submit(screen, "question")
        await pilot.pause()

        assert "Evidence sent with this message · 1 source" in _strip_text(screen)


@pytest.mark.asyncio
async def test_console_surface_refresh_failure_never_costs_the_send_its_evidence(
    monkeypatch,
) -> None:
    """The release sits on the provider path: it must not be able to throw.

    ``ConsoleChatController._capture_rag_context`` turns any provider
    exception into ``context=None``, so a fan-out failure escaping the
    release would send the message WITHOUT the evidence it just consumed.
    """
    app = _build_test_app()
    launch = _launch(2)
    context = "[S1] MEDIA — Source 1\nBody 1"
    capture = AsyncMock(
        return_value=LocalRagContextResult(context=context, citation_builder=None)
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        def _explode() -> bool:
            raise RuntimeError("rail body vanished mid-send")

        monkeypatch.setattr(
            screen, "_sync_console_pending_launch_surfaces", _explode
        )

        controller = screen._ensure_console_chat_controller()
        # The send still receives the captured context...
        captured = await controller._capture_rag_context("question")
        assert captured[0] == context
        # ...and the staging is still consumed.
        assert screen._pending_console_launch_context is None


@pytest.mark.asyncio
async def test_console_use_in_console_handoff_reaches_the_strip() -> None:
    """The CHAT_HANDOFF writer must fan out like Console's own staging.

    It finishes via ``_sync_native_console_chat_ui``, which refreshes the
    chip but neither the strip nor the tray -- so a bare field assignment
    left the chip claiming staged sources the user could not see or clear.
    """
    from tldw_chatbook.Chat.chat_handoff_models import ChatHandoffPayload

    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is False

        screen._stage_handoff_as_console_live_work(
            ChatHandoffPayload(
                source="Library Search/RAG",
                title="Transformer notes",
                body="Attention is all you need.",
                item_type="media",
                source_id="media-77",
                content_ref="media-77#c1",
                runtime_backend="local",
            )
        )
        await pilot.pause()

        strip = screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip)
        assert strip.display is True
        assert "Transformer notes" in _strip_text(screen)
        assert len(strip.query(UNSTAGE_ID)) == 1
        tray = screen.query_one(
            "#console-staged-context-tray", ConsoleStagedContextTray
        )
        assert tray.state.is_empty is False


@pytest.mark.asyncio
async def test_console_rail_badge_and_chip_report_the_same_staged_count() -> None:
    """The rail badge and settings estimate read the workspace context; the
    chip reads the bundle. Both must land on the same number."""
    app = _build_test_app()
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, STRIP_ID)
        screen._retrieval._stage_console_library_rag_launch(_launch(4))
        await pilot.pause()

        workspace_context = screen._workspace._current_console_workspace_context()
        assert len(workspace_context.staged_sources) == 4
        assert [source.label for source in workspace_context.staged_sources] == [
            "Source 1",
            "Source 2",
            "Source 3",
            "Source 4",
        ]
        sources_chip = screen.query_one("#console-sources-label", Static)
        assert "Sources: 4" in str(sources_chip.renderable)


def test_workspace_context_falls_back_to_one_row_for_a_bundleless_launch() -> None:
    """A generic handoff has no bundle -- all three surfaces still say 1."""
    from types import SimpleNamespace

    launch = ConsoleLiveWorkLaunch.from_values(
        source="Watchlists",
        title="Daily papers",
        payload={"target_id": "local:watchlist_run:daily"},
        status="ready",
    )
    screen = SimpleNamespace(
        _pending_console_launch_context=launch,
        app_instance=SimpleNamespace(),
    )
    context = workspace_module.ConsoleWorkspaceController._current_console_workspace_context(
        screen
    )
    assert len(context.staged_sources) == 1
    assert context.staged_sources[0].label == "Daily papers"
    assert console_staged_source_count(launch) == 1


@pytest.mark.asyncio
async def test_context_estimate_counts_staged_evidence_before_send() -> None:
    """task-6: reproduces the critique's exact bug -- staged evidence used to
    move `staged_source_count` (the label suffix) but never `used_tokens`,
    so the settings summary's context row silently read as if nothing had
    been staged. Staging a large source must move `used_tokens` off its
    pre-staging baseline."""
    app = _build_test_app()
    _configure_native_ready_console(app)
    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")

        baseline = screen._active_console_settings_context_estimate()
        assert baseline.used_tokens is not None

        large_source = EvidenceReference(
            evidence_id="S1",
            source_id="media-1",
            source_type="media",
            title="Large corpus",
            snippet="large corpus text " * 300,
            authority_label="local",
            status="available",
            source_owner="local",
        )
        bundle = EvidenceBundle(
            bundle_id="bundle-large",
            query="question",
            source="Library Search/RAG",
            references=(large_source,),
        )
        launch = ConsoleLiveWorkLaunch.from_values(
            source="Library Search/RAG",
            title="Library Search/RAG retrieval",
            payload={"query": "question", "evidence_bundle": bundle.to_payload()},
            status="staged",
        )
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        staged = screen._active_console_settings_context_estimate()
        assert staged.used_tokens is not None
        assert staged.used_tokens > baseline.used_tokens
        assert "1 source staged" in staged.label


@pytest.mark.asyncio
async def test_console_capture_without_prompt_context_keeps_staging(
    monkeypatch,
) -> None:
    """No prompt context reached the provider -- discarding would lose evidence."""
    app = _build_test_app()
    launch = _launch(2)
    capture = AsyncMock(return_value=LocalRagContextResult(None, None))
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        await _submit(screen, "question")
        await pilot.pause()

        assert screen._pending_console_launch_context is launch
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is True


@pytest.mark.asyncio
async def test_console_blocked_send_keeps_staged_evidence(monkeypatch) -> None:
    app = _build_test_app()
    launch = _launch(2)
    capture = AsyncMock(
        return_value=LocalRagContextResult(context="ctx", citation_builder=None)
    )
    monkeypatch.setattr(
        retrieval_module, "capture_console_staged_evidence_for_chat", capture
    )

    class _BlockedGateway:
        async def resolve_for_send(self, _selection):
            class _R:
                ready = False
                visible_copy = "Send blocked: choose a provider."

            return _R()

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        controller = screen._ensure_console_chat_controller()
        controller.provider_gateway = _BlockedGateway()
        controller._agent_runtime_enabled = False
        result = await controller.submit_draft("question")
        await pilot.pause()

        assert result.accepted is False
        capture.assert_not_awaited()
        assert screen._pending_console_launch_context is launch
        assert screen.query_one(STRIP_ID, ConsoleStagedEvidenceStrip).display is True


# --------------------------------------------------------------------------
# D1c blast radius: Library launches gaining real bundles changes
# `_console_send_blocked_reason`'s inputs (it gates on
# `evidence_state.available_count == 0` for any RAG-labeled staged launch).
# Pin both directions so a bundleless-vs-bundled Library launch keeps
# sending exactly the way it already does.
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_console_send_blocked_reason_sendable_for_library_staged_one_ref() -> (
    None
):
    """A Library-staged launch with one AVAILABLE local reference must be
    sendable -- the same shape `library_screen.py`'s own "Use in Console"
    produces for a single selected result."""
    app = _build_test_app()
    app.app_config = {
        "chat_defaults": {
            "provider": "OpenAI",
            "model": "gpt-4.1-2025-04-14",
        },
        "api_settings": {"openai": {"api_key": "configured-test-key"}},
    }
    app.chat_api_provider_value = "OpenAI"
    app.chat_api_model_value = "gpt-4.1-2025-04-14"
    launch = _launch(1)

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        assert screen._console_send_blocked_reason() == ""


@pytest.mark.asyncio
async def test_console_send_blocked_reason_blocks_for_library_staged_zero_available_refs() -> (
    None
):
    """A Library-staged launch whose sole reference has no available
    evidence must still block with the EXISTING copy -- unchanged by
    attaching real bundles to Library launches."""
    app = _build_test_app()
    bundle = EvidenceBundle(
        bundle_id="bundle-blocked",
        query="question",
        source="Library Search/RAG",
        references=(_reference(1, status="blocked"),),
    )
    launch = ConsoleLiveWorkLaunch.from_values(
        source="Library Search/RAG",
        title="Library Search/RAG retrieval",
        payload={"query": "question", "evidence_bundle": bundle.to_payload()},
        status="staged",
    )

    async with ConsoleHarness(app).run_test(size=(180, 48)) as pilot:
        screen = pilot.app.screen_stack[-1]
        await _wait_for_selector(screen, pilot, "#console-native-composer")
        screen._retrieval._stage_console_library_rag_launch(launch)
        await pilot.pause()

        reason = screen._console_send_blocked_reason()
        assert (
            # task-15791: same TASK-2154 rename as the chip ("RAG" -> "Library
            # search") reached this blocked-reason copy.
            "Console send blocked: Library search has no available evidence. "
            "Review source authority before sending." in reason
        )
