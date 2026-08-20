# test_reactive_default_aliasing.py
# Description: Cross-instance leak regressions for shared mutable reactive defaults (task-15771, task-16843)
"""
task-15771: ``reactive([])`` / ``reactive({})`` with a non-callable mutable
default installs the *same* list/dict object as the backing value on every
instance of the widget class that has not explicitly reassigned it
(``Reactive._initialize_reactive`` does ``default_or_callable() if
callable(...) else default_or_callable``). Any instance that then mutates the
value in place (``.append``/``[k] =``/``.insert``/...) leaks that state into
every other instance — and into instances created later, including across
screen remounts.

Worse, "reassigns before use" is not a defense when the reassigned value is
empty-equal: Textual's ``Reactive._set`` only stores the new object when
``current_value != value``, so ``self.attr = []`` over the pristine shared
``[]`` default is a complete no-op and the instance keeps aliasing the shared
object (verified against Textual 8.2.8).

These tests demonstrate the cross-instance leak on the most user-facing live
cases found by the task's sweep, driving the same mutations the production
handlers perform. (The sweep's third case, the media keyword manager's
``selected_keywords``, lost its test when task-19046 retired that widget as
dead code.) They were born red against the pre-fix tree
(instance B observed instance A's mutation) and are green once the defaults
are callables (``reactive(list)`` / ``reactive(dict)``).

The final test pins that the fix did not change ``recompose=True`` behavior:
a recompose reactive with a callable default still rebuilds its children on
reassignment.

task-16843 extends the same aliasing bug to the ``reactive(SomeClass())``
shared *instance* default shape (15771's review F2 gap — the AST guard only
flagged ``list()``/``dict()``/``set()`` call results, not arbitrary
constructor calls). ``test_console_conversation_inspector_snapshots_do_not_leak_across_instances``
covers the one site of the five found that actually carries mutable field
values (``ConsoleContextSnapshot``'s ``current_messages: list`` /
``next_send_payload: dict`` — the dataclass itself is ``frozen=True`` but
that only blocks *reassigning* those fields, not mutating the list/dict
objects they point to in place). The other four sites (``RegionLayout``,
``TreeScope`` x3) are frozen dataclasses whose *only* field types are
themselves immutable (``frozenset``, ``Literal`` str, ``int | None``,
``Region`` enum) — there is no in-place mutation to demonstrate, so they are
handled by documentation + the guard's allowlist instead of a leak test; see
``Tests/Architecture/test_reactive_mutable_default_inventory.py``.

task-10 retired the standalone modal this last test originally targeted
(its Next Send tab was ported wholesale into
``ConsoleConversationInspector``, callable-default reactive included) --
the test now constructs that shared inspector instead; the regression
itself (and its rationale above) is unchanged.
"""

from __future__ import annotations

import asyncio

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Static

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleContextSnapshot,
    ConsoleMessageRole,
)
from tldw_chatbook.Chat.console_cost_tracker import ConsoleCostRowTotals
from tldw_chatbook.TTS.audiobook_generator import Chapter
from tldw_chatbook.UI.Watchlists_Modules.overview_pane import OverviewPane
from tldw_chatbook.Widgets.collections_tag_window import CollectionsTagWindow
from tldw_chatbook.Widgets.Console.console_conversation_inspector import (
    ConsoleConversationInspector,
)
from tldw_chatbook.Widgets.TTS.chapter_editor_widget import ChapterEditorWidget
from tldw_chatbook.Widgets.TTS.character_voice_widget import CharacterVoiceWidget


class _CharacterVoiceApp(App[None]):
    """Minimal host mounting a single CharacterVoiceWidget."""

    def compose(self) -> ComposeResult:
        yield CharacterVoiceWidget(provider="openai", id="character-voice-widget")


@pytest.mark.asyncio
async def test_character_voice_widget_characters_do_not_leak_across_instances() -> None:
    """Instance A adds a character via the real button handler; a second,
    pristine instance must not see it.

    Born red pre-fix: ``_add_character_manually`` appends to the class-shared
    default list, so the fresh instance B started life already containing A's
    "Character 1".
    """
    app = _CharacterVoiceApp()
    async with app.run_test() as pilot:
        widget_a = app.query_one("#character-voice-widget", CharacterVoiceWidget)
        widget_a._add_character_manually()
        await pilot.pause()
        assert len(widget_a.characters) == 1

        widget_b = CharacterVoiceWidget(provider="openai")
        assert widget_b.characters is not widget_a.characters
        assert list(widget_b.characters) == []
        # A's instance keeps its own state.
        assert len(widget_a.characters) == 1


@pytest.mark.asyncio
async def test_character_voice_widget_voice_assignments_do_not_leak() -> None:
    """Same class, dict-shaped default: assigning a voice on instance A
    (the exact mutation ``_on_voice_selected``/``_apply_voice_to_all``
    perform: ``self.voice_assignments[name] = voice_id``) must not appear in
    a pristine instance B."""
    app = _CharacterVoiceApp()
    async with app.run_test() as pilot:
        widget_a = app.query_one("#character-voice-widget", CharacterVoiceWidget)
        widget_a.voice_assignments["Narrator"] = "voice-1"
        await pilot.pause()

        widget_b = CharacterVoiceWidget(provider="openai")
        assert widget_b.voice_assignments is not widget_a.voice_assignments
        assert dict(widget_b.voice_assignments) == {}


def test_chapter_editor_widgets_do_not_share_chapters() -> None:
    """Two ChapterEditorWidgets must not alias one chapters list.

    ``__init__`` does ``self.chapters = chapters if chapters else []`` — with
    no chapters argument that assignment is equality-skipped by the reactive
    (``[] != []`` is False), so pre-fix both instances still aliased the
    class-shared default, and the ``self.chapters.insert(...)`` mutation the
    add/split handlers perform leaked into every other editor instance.
    """
    editor_a = ChapterEditorWidget()
    editor_b = ChapterEditorWidget()

    editor_a.chapters.insert(
        0,
        Chapter(
            number=1,
            title="Leaked chapter",
            content="",
            start_position=0,
            end_position=0,
        ),
    )

    assert editor_b.chapters is not editor_a.chapters
    assert list(editor_b.chapters) == []
    assert len(editor_a.chapters) == 1


class _OverviewPaneApp(App[None]):
    """Minimal host mounting an OverviewPane (data = reactive({}, recompose=True))."""

    def compose(self) -> ComposeResult:
        yield OverviewPane(id="overview-pane")


@pytest.mark.asyncio
async def test_recompose_reactive_still_recomposes_with_callable_default() -> None:
    """The callable-default fix must not change recompose behavior.

    OverviewPane.data is ``reactive(..., recompose=True)``: while the value is
    the (now per-instance) default ``{}`` the pane composes its loading state,
    and reassigning a populated payload must rebuild the children into the
    dashboard grid.
    """
    app = _OverviewPaneApp()
    async with app.run_test() as pilot:
        pane = app.query_one("#overview-pane", OverviewPane)
        assert pane.query("#overview-loading").nodes, (
            "expected the loading state while data is the default {}"
        )
        assert not pane.query("#watchlists-overview-grid").nodes

        pane.data = {
            "total_sources": 3,
            "active_sources": 2,
            "sources_in_error": 0,
            "total_items": 5,
            "new_items": 1,
            "latest_run_status": "success",
            "active_alert_rules": 0,
            "failed_runs": [],
        }
        await pilot.pause()

        assert pane.query("#watchlists-overview-grid").nodes, (
            "expected recompose to rebuild the pane into the dashboard grid"
        )
        card = pane.query_one("#overview-total-sources", Static)
        assert "3" in str(card.renderable)
        assert not pane.query("#overview-loading").nodes


class _ConsoleContextHarness(App[None]):
    """Minimal host so a pushed ConsoleConversationInspector has a screen to
    sit on."""

    def compose(self) -> ComposeResult:
        yield Static("background")


async def _empty_exchanges_loader(
    _native_message_id: str,
) -> list[tuple[object, bool]]:
    return []


def _inspector(snapshot_factory) -> ConsoleConversationInspector:
    return ConsoleConversationInspector(
        rows=[],
        totals=ConsoleCostRowTotals(0, 0.0, False, 0),
        turns=[],
        exchanges_loader=_empty_exchanges_loader,
        snapshot_factory=snapshot_factory,
    )


@pytest.mark.asyncio
async def test_console_conversation_inspector_snapshots_do_not_leak_across_instances() -> None:
    """Two ConsoleConversationInspector instances must not share the default
    snapshot's ``current_messages``/``next_send_payload`` containers
    (task-16843).

    ``snapshot = reactive(lambda: ConsoleContextSnapshot(current_messages=[],
    next_send_payload={}))`` installs a FRESH ``ConsoleContextSnapshot`` per
    instance -- the callable-default fix this test pins. ``frozen=True`` on
    the dataclass only blocks *reassigning* ``current_messages``/
    ``next_send_payload``; it does not stop mutating the list/dict those
    fields point to in place, which is exactly what this test does.

    The snapshot factory blocks on an ``asyncio.Event`` that is never set
    until after the assertions, so both inspectors stay on their class-level
    default for the whole window under test -- the real window a user sees
    as the loading spinner between opening the Next Send tab and the
    snapshot arriving. Born red pre-fix (on the standalone modal this test
    originally targeted, retired in task-10): instance B observed instance
    A's mutation (and, with a real ``ConsoleChatMessage`` list containing
    production objects, corrupting the payload of a live in-flight modal).
    """
    never_ready = asyncio.Event()

    async def _blocking_factory() -> ConsoleContextSnapshot:
        await never_ready.wait()
        return ConsoleContextSnapshot(current_messages=[], next_send_payload={})

    app = _ConsoleContextHarness()
    async with app.run_test(size=(100, 40)) as pilot:
        modal_a = _inspector(_blocking_factory)
        modal_b = _inspector(_blocking_factory)
        await app.push_screen(modal_a)
        await pilot.pause()

        # ``next_send_loading``, not ``loading`` -- ``Widget`` already
        # declares a built-in ``loading`` reactive (a whole-widget loading
        # OVERLAY) that this pane's own Next Send fetch flag must not
        # shadow; see ``ConsoleConversationInspector``'s own comment on its
        # ``next_send_loading`` reactive declaration.
        assert (
            modal_a.next_send_loading
        ), "expected modal_a still waiting on its (blocked) factory"
        modal_a.snapshot.current_messages.append(
            ConsoleChatMessage(role=ConsoleMessageRole.USER, content="leaked-from-a")
        )
        modal_a.snapshot.next_send_payload["leaked_key"] = "leaked_value"

        await app.push_screen(modal_b)
        await pilot.pause()

        assert (
            modal_b.next_send_loading
        ), "expected modal_b still waiting on its (blocked) factory"
        assert modal_b.snapshot is not modal_a.snapshot
        assert modal_b.snapshot.current_messages == []
        assert modal_b.snapshot.next_send_payload == {}

        # Let both blocked workers resolve so the app shuts down cleanly.
        never_ready.set()
        await pilot.pause()
