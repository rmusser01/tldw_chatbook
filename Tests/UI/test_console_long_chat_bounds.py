"""A long Console chat keeps the transcript and the post-action sync bounded.

TASK-33628.5.1 and TASK-33628.5.2. Measured on dev 7d155170dc with a
file-backed ChaChaNotes database (ConsoleHarness, 160x48):

- Selecting the first message of a 1,000-message chain mounted all 1,000
  rows (8,085 widgets) and took 54 s to settle; at 3,000 it was 24,085
  widgets and 183 s. Undo of a Delete reselects the restored root and did
  the same.
- With those rows mounted, a focus change into or out of the transcript
  restyled every one of them, twice: Textual restyles a focused widget's
  whole subtree, and the focus-cue class sat on the region that holds every
  row. 29-37 s per change at 3,000 rows.
- Every post-action sync removed and re-added the control bar's height class
  (two restyles of the bar, 110 nodes, whatever changed) and ran four
  screen-wide queries for four rail labels, each one walking every mounted
  transcript row.
- Every post-action sync also captured the whole active lineage twice (one
  SQLite version read and one validated snapshot per message) to prove which
  memory applies, even for a chat that holds no memory: ~19 ms of each ~29 ms
  call at 3,000 messages, measured on the branch with GC off.

Every case drives the real ``ChatScreen`` with the real Console store over a
real file-backed database. The assertions are counts (rows planned and
mounted, nodes restyled, nodes walked, store snapshots), not wall-clock.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import contextmanager
from typing import Any, Iterator

import pytest
import textual.dom as textual_dom
from textual.css.model import CombinatorType
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.test_console_message_delete_undo import _painted, _tree_nodes
from Tests.UI.test_console_native_chat_flow import _wait_for_selector
from Tests.UI.test_destination_shells import _build_test_app
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.Chat.console_context_compaction import EffectiveMemoryKind
from tldw_chatbook.Chat.console_context_repository import (
    ConsoleMemorySelectionRecord,
    MemorySelectionKind,
)
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules.message_delete import (
    handle_console_delete_action,
)
from tldw_chatbook.Widgets.Console import ConsoleTranscript

# The real ChatScreen/store goes through config-participant admission, which
# the per-test sandbox refuses (RecoveryRequired); keep the collection-time
# profile.
pytestmark = pytest.mark.bootstrap_profile

_CHAIN = 3_000
_SIZE = (160, 48)
#: The most nodes one focus change may restyle. Dev restyled ~1,700 with
#: the 64 rows a 3,000-message chain opens with (4,965 at 200 rows); what
#: remains is the composer's own ``:focus-within`` subtree (~60 nodes).
_FOCUS_RESTYLES = 100


def _seed(db: CharactersRAGDB, rows: list[tuple[str, str, str | None]]) -> str:
    """Save ``(id, role, parent)`` rows as one conversation; return its id."""
    conversation_id = ChatConversationService(db).create_conversation(
        id="long-chat", title="Long chat", scope_type="global", state="in-progress"
    )
    with db.transaction():
        for index, (message_id, role, parent_id) in enumerate(rows):
            db.add_message(
                {
                    "id": message_id,
                    "conversation_id": conversation_id,
                    "parent_message_id": parent_id,
                    "sender": role,
                    "role": role,
                    "content": f"{message_id} text",
                    "timestamp": (
                        f"2026-09-30T{index // 3600:02d}:{index // 60 % 60:02d}:"
                        f"{index % 60:02d}.000000+00:00"
                    ),
                }
            )
    return conversation_id


def _chain(count: int, prefix: str = "m") -> list[tuple[str, str, str | None]]:
    """A linear chain of alternating user/assistant messages."""
    rows: list[tuple[str, str, str | None]] = []
    parent = None
    for index in range(count):
        message_id = f"{prefix}{index:04d}"
        rows.append((message_id, "user" if index % 2 == 0 else "assistant", parent))
        parent = message_id
    return rows


async def _until(predicate: Any, what: str, *, timeout: float = 60.0) -> None:
    """Poll on the event loop without Pilot's per-widget pause traffic."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


class _Heartbeat:
    """Record the longest gap between event-loop turns."""

    def __init__(self) -> None:
        self.max_gap = 0.0
        self.last_turn = time.perf_counter()
        self._stop = False
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        self.last_turn = time.perf_counter()
        while not self._stop:
            await asyncio.sleep(0.005)
            now = time.perf_counter()
            self.max_gap = max(self.max_gap, now - self.last_turn)
            self.last_turn = now

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def quiet(self, *, for_s: float = 0.6, timeout: float = 120.0) -> None:
        """Wait until the loop has turned freely for ``for_s`` seconds."""
        deadline = time.monotonic() + timeout
        calm_since = time.perf_counter()
        gap_seen = self.max_gap
        while time.monotonic() < deadline:
            await asyncio.sleep(0.05)
            if self.max_gap != gap_seen or time.perf_counter() - self.last_turn > 0.05:
                gap_seen = self.max_gap
                calm_since = time.perf_counter()
            elif time.perf_counter() - calm_since >= for_s:
                return
        raise AssertionError("the event loop never went quiet")

    async def stop(self) -> None:
        self._stop = True
        if self._task is not None:
            await self._task


async def _open(
    host: ConsoleHarness,
    pilot: Any,
    db: CharactersRAGDB,
    conversation_id: str,
    leaf: str,
) -> tuple[Any, Any, dict[str, str], ConsoleTranscript]:
    """Open the saved conversation in the mounted Console; wait for its rows."""
    console = host.screen_stack[-1]
    await _wait_for_selector(console, pilot, "#console-native-transcript")
    assert db.set_conversation_active_cursor(
        conversation_id, active_leaf_message_id=leaf, before_message_id=None
    )
    store = console._ensure_console_chat_store()
    session = store.restore_persisted_session(
        title="Long chat",
        workspace_id=None,
        persisted_conversation_id=conversation_id,
        all_nodes=_tree_nodes(db, conversation_id),
        active_leaf_persisted_id=leaf,
    )
    await console._sync_native_console_chat_ui()
    native = {
        node.persisted_message_id: node.id
        for node in store._nodes_by_session[session.id].values()
    }
    transcript = console.query_one("#console-native-transcript", ConsoleTranscript)
    await _until(
        lambda: native[leaf] in transcript.mounted_message_content_ids(),
        "the opened conversation's newest row to mount",
    )
    return console, store, native, transcript


def _window_rows(transcript: ConsoleTranscript) -> int:
    """The most rows a bounded window may hold: twice one load-shaped window.

    One load-shaped window is the transcript's initial line budget over the
    smallest per-message estimate in this chain: 64 of these short messages
    at 160x48. The margin covers one near extension (at most one more window)
    on top. A fixed 300 let a partial regression mounting 250 rows pass.
    Dev mounted all 3,000.
    """
    budget = transcript._initial_window_line_budget()
    smallest = min(map(transcript._estimated_message_lines, transcript._messages))
    rows = 2 * -(-budget // smallest)
    # The bound must stay far below the chain, or it proves nothing.
    assert 0 < rows <= _CHAIN // 20, rows
    return rows


def _planned(transcript: ConsoleTranscript) -> list[str]:
    """Message ids the transcript's window will mount on its next refresh."""
    return [
        message.id
        for message in transcript._messages
        if message.id not in transcript._pruned_message_ids
        and message.id not in transcript._hidden_tail_ids
    ]


async def _settled_rows(transcript: ConsoleTranscript, heartbeat: _Heartbeat) -> set:
    """Mounted message ids once the window and any jump placement have landed.

    A quiet loop alone is not enough: composing a window's rows runs as many
    short slices, a re-centered jump places its target only after them, and
    a window mounting a batch at a time adds its batches across refreshes.
    """
    await _until(
        lambda: (
            transcript._reveal_scroll_target is None
            and not transcript._suppress_boundary_hydration
            and not transcript._refresh_lock.locked()
            and getattr(transcript, "_window_fill", None) is None
        ),
        "the transcript window to land",
    )
    await heartbeat.quiet()
    return set(transcript.mounted_message_content_ids())


async def _until_painted(host: ConsoleHarness, text: str) -> None:
    """Wait until ``text`` is on screen (rendering, not just the DOM)."""
    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if text in _painted(host):
            return
        await asyncio.sleep(0.1)
    raise AssertionError(f"{text!r} was never painted")


@contextmanager
def _restyles(host: ConsoleHarness) -> Iterator[list[Any]]:
    """Record every node the stylesheet restyles inside the block."""
    applied: list[Any] = []
    stylesheet = host.stylesheet
    real = stylesheet.apply

    def apply(node: Any, *args: Any, **kwargs: Any) -> Any:
        applied.append(node)
        return real(node, *args, **kwargs)

    stylesheet.apply = apply
    try:
        yield applied
    finally:
        del stylesheet.apply


@contextmanager
def _refreshed() -> Iterator[list[Any]]:
    """Record every widget that requests a refresh inside the block."""
    refreshed: list[Any] = []
    real = Widget.refresh

    def refresh(self: Any, *args: Any, **kwargs: Any) -> Any:
        refreshed.append(self)
        return real(self, *args, **kwargs)

    Widget.refresh = refresh
    try:
        yield refreshed
    finally:
        Widget.refresh = real


@contextmanager
def _walked() -> Iterator[list[Any]]:
    """Record every node a DOM walk (query, query_one, restyle) visits."""
    visited: list[Any] = []
    real_walk = textual_dom.DOMNode.walk_children
    real_search = textual_dom.walk_breadth_search_id

    def walk_children(self: Any, *args: Any, **kwargs: Any) -> Any:
        nodes = list(real_walk(self, *args, **kwargs))
        visited.extend(nodes)
        return nodes

    def search(node: Any, node_id: str, *, with_root: bool = True) -> Any:
        queue = [node]
        while queue:
            current = queue.pop(0)
            visited.append(current)
            if (with_root or current is not node) and current.id == node_id:
                return current
            queue.extend(current._nodes)
        return None

    textual_dom.DOMNode.walk_children = walk_children
    textual_dom.walk_breadth_search_id = search
    try:
        yield visited
    finally:
        textual_dom.DOMNode.walk_children = real_walk
        textual_dom.walk_breadth_search_id = real_search


def _lazy_receipt_sheet(host: ConsoleHarness) -> str:
    """Unload the Delete receipt's sheet, as the real app has it until used.

    ``ConsoleHarness`` reads every split sheet at boot; ``TldwCli`` reads
    this one only when a receipt first opens.
    """
    from tldw_chatbook.Widgets.Console.console_message_delete_receipt import (
        ConsoleMessageDeleteReceiptModal,
    )

    path = ConsoleMessageDeleteReceiptModal.CSS_PATH
    stylesheet = host.stylesheet
    stylesheet.source.pop((path, ""), None)
    stylesheet.reparse()
    stylesheet.update(host)
    assert not stylesheet.has_source(path, "")
    return path


def _inside(node: Any, ancestor: Any) -> bool:
    return node is not ancestor and ancestor in node.ancestors


@pytest.mark.asyncio
async def test_selecting_the_first_message_of_a_3000_message_chat_mounts_a_window(
    tmp_path,
):
    """5.1 AC#1: an early selection mounts a window, not every later row."""
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        window_rows = _window_rows(transcript)
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            opened = await _settled_rows(transcript, heartbeat)
            assert 0 < len(opened) <= window_rows
            first, last = native[rows[0][0]], native[rows[-1][0]]

            transcript.select_message(first)

            planned = _planned(transcript)
            assert first in planned
            assert len(planned) <= window_rows, (
                f"selecting the first of {_CHAIN} messages planned {len(planned)} rows"
            )
            await _until(
                lambda: first in transcript.mounted_message_content_ids(),
                "the selected first message to mount",
            )
            mounted = await _settled_rows(transcript, heartbeat)
            assert first in mounted
            assert last not in mounted
            assert len(mounted) <= window_rows
            region = console.query_one("#console-transcript-region")
            assert sum(1 for _ in region.walk_children()) <= 10 * window_rows
            assert transcript.selected_message_id == first
            # The jump put the selected row on screen, not just in the DOM.
            await _until_painted(host, "m0000 text")
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_delete_and_undo_from_the_first_of_3000_messages_stay_bounded(tmp_path):
    """5.1 AC#1: select, Delete and Undo of a 3,000-message subtree stay windowed.

    Undo reselects the restored root through the selection handoff, which
    extended the window with no bound and mounted all 3,000 rows again.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    notices: list[str] = []
    app.notify = lambda message, **kwargs: notices.append(str(message))
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        first, last = native[rows[0][0]], native[rows[-1][0]]
        window_rows = _window_rows(transcript)
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            await heartbeat.quiet()

            transcript.select_message(first)
            await _until(
                lambda: first in transcript.mounted_message_content_ids(),
                "the selected first message to mount",
            )
            await heartbeat.quiet()
            await handle_console_delete_action(console._message, "delete", first)
            confirm = f"#console-message-action-delete-confirm-{first}"
            await _wait_for_selector(console, pilot, confirm, timeout=30.0)
            assert f"Delete {_CHAIN} messages" in str(
                console.query_one(confirm, Button).label
            )
            console.query_one(confirm, Button).press()
            await _until(
                lambda: bool(host.screen.query("#console-delete-receipt-undo")),
                "Undo to be offered",
            )
            await heartbeat.quiet()
            assert not transcript.mounted_message_content_ids()

            host.screen.query_one("#console-delete-receipt-undo", Button).press()
            await _until(
                lambda: transcript.selected_message_id == first,
                "Undo to reselect the restored first message",
            )
            planned = _planned(transcript)
            assert first in planned
            assert len(planned) <= window_rows, (
                f"Undo planned {len(planned)} of {_CHAIN} restored rows"
            )
            await _until(
                lambda: not host.screen.query("#console-delete-receipt-box"),
                "the receipt to close after Undo",
            )
            mounted = await _settled_rows(transcript, heartbeat)
            assert first in mounted
            assert last not in mounted
            assert len(mounted) <= window_rows
            assert len(store.messages_for_session(store.active_session_id)) == _CHAIN
            await _until_painted(host, "m0000 text")
            assert any(f"Restored {_CHAIN} messages" in n for n in notices), notices
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_undo_of_a_delete_from_inside_the_window_mounts_a_window(tmp_path):
    """5.1 AC#1: restoring an early message the window already showed.

    Here the restored root was mounted before the Delete, so the ingest keeps
    the window it had and the root is inside it: no reveal runs. Every row
    the Undo brought back below it mounted with it -- all 2,970 of them.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        first, early, last = (native[rows[i][0]] for i in (0, 30, -1))
        window_rows = _window_rows(transcript)
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            transcript.select_message(first)
            await _settled_rows(transcript, heartbeat)
            transcript.select_message(early)
            mounted = await _settled_rows(transcript, heartbeat)
            assert early in mounted and len(mounted) <= window_rows

            await handle_console_delete_action(console._message, "delete", early)
            confirm = f"#console-message-action-delete-confirm-{early}"
            await _wait_for_selector(console, pilot, confirm, timeout=30.0)
            console.query_one(confirm, Button).press()
            await _until(
                lambda: bool(host.screen.query("#console-delete-receipt-undo")),
                "Undo to be offered",
            )
            await heartbeat.quiet()
            host.screen.query_one("#console-delete-receipt-undo", Button).press()
            await _until(
                lambda: transcript.selected_message_id == early,
                "Undo to reselect the restored message",
            )
            planned = _planned(transcript)
            assert early in planned
            assert len(planned) <= window_rows, (
                f"Undo planned {len(planned)} of {_CHAIN} rows"
            )
            await _until(
                lambda: not host.screen.query("#console-delete-receipt-box"),
                "the receipt to close after Undo",
            )
            mounted = await _settled_rows(transcript, heartbeat)
            assert early in mounted
            assert last not in mounted
            assert len(mounted) <= window_rows
            assert len(store.messages_for_session(store.active_session_id)) == _CHAIN
            await _until_painted(host, "m0030 text")
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_a_focus_change_restyles_no_transcript_row(tmp_path):
    """5.1 AC#2: focus in and out of a long transcript restyles a bounded set."""
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, _native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            await _settled_rows(transcript, heartbeat)
            rows_mounted = sum(1 for _ in transcript.walk_children())
            assert rows_mounted > 4 * _FOCUS_RESTYLES
            composer = console.query_one("#console-native-composer")
            title = console.query_one("#console-transcript-title")
            composer.focus()
            await heartbeat.quiet()
            resting_bar_color = transcript.styles.scrollbar_color
            for target, transcript_focused in (
                (transcript, True),
                (composer, False),
                (transcript, True),
            ):
                with _restyles(host) as restyled, _refreshed() as refreshed:
                    target.focus()
                    await heartbeat.quiet()
                assert host.focused is target
                own_bars = {
                    transcript.vertical_scrollbar,
                    transcript.horizontal_scrollbar,
                    transcript.scrollbar_corner,
                }
                inside = [
                    node
                    for node in restyled
                    if _inside(node, transcript) and node not in own_bars
                ]
                assert not inside, (
                    f"a focus change restyled {len(inside)} transcript nodes "
                    f"({len(restyled)} in all)"
                )
                assert len(restyled) <= _FOCUS_RESTYLES, len(restyled)
                # Live, the restyle count hid the real cost: the focus rule's
                # scrollbar recolour made Textual refresh every mounted row
                # (colours inherit), 7.2 s with a scrolled-back window.
                repainted = {
                    node
                    for node in refreshed
                    if _inside(node, transcript) and node not in own_bars
                }
                assert not repainted, (
                    f"a focus change refreshed {len(repainted)} transcript nodes"
                )
                # The cues the restyle exists for still paint: the title row,
                # and the scrollbar accent on the transcript itself.
                assert bool(title.styles.text_style.bold) is transcript_focused
                assert bool(title.styles.text_style.underline) is transcript_focused
                assert (
                    transcript.styles.scrollbar_color != resting_bar_color
                ) is transcript_focused
        finally:
            await heartbeat.stop()


def _subject_start(selectors: list[Any]) -> int:
    """Index of the first selector in the rightmost compound."""
    start = len(selectors) - 1
    while start > 0 and selectors[start].combinator is CombinatorType.SAME:
        start -= 1
    return start


@pytest.mark.asyncio
async def test_no_rule_styles_a_transcript_descendant_by_the_transcripts_focus(
    tmp_path,
):
    """Fidelity pin for the scoped focus restyles above.

    The transcript restyles only itself when it gains or loses focus, and the
    region's focus class restyles only the title row. That is exact only
    while no stylesheet rule makes another node's style depend on either.
    Checked two ways: every loaded rule, and the computed style of every
    mounted node after a full subtree restyle. The transcript's focus
    restyle also skips refreshing its rows, which is exact only while focus
    changes nothing on it but its own frame and scrollbar paint, which no
    descendant inherits.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(40)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        region = console.query_one("#console-transcript-region")
        title = console.query_one("#console-transcript-title")
        offenders = []
        for rule in host.stylesheet.rules:
            for selector_set in rule.selector_set:
                selectors = selector_set.selectors
                subject = selectors[_subject_start(selectors)]
                for selector in selectors[: _subject_start(selectors)]:
                    if selector.pseudo_classes & {"focus", "blur"} and selector._check(
                        transcript
                    ):
                        offenders.append(selector_set.css)
                    if selector.name == "console-transcript-region-focused" and not (
                        subject.name == "console-transcript-title"
                    ):
                        offenders.append(selector_set.css)
        assert not offenders, offenders

        transcript.select_message(native[rows[-2][0]])
        await pilot.pause()
        await pilot.pause()
        composer = console.query_one("#console-native-composer")

        def styles() -> dict[int, str]:
            return {
                id(node): node.styles.css
                for node in region.walk_children(with_self=True)
            }

        own_rules = {}
        for target in (transcript, composer, transcript):
            target.focus()
            await pilot.pause()
            await pilot.pause()
            scoped = styles()
            host.stylesheet.update_nodes(region.walk_children(with_self=True))
            assert styles() == scoped
            own_rules[target is transcript] = transcript.styles.base.get_rules()
        assert title.styles.text_style.bold
        changed = {
            key
            for key in own_rules[True].keys() | own_rules[False].keys()
            if own_rules[True].get(key) != own_rules[False].get(key)
        }
        # Frame and scrollbar paint only; none of these is inherited.
        own_frame = ("scrollbar_", "outline_", "border_")
        assert "scrollbar_color" in changed, changed
        assert all(key.startswith(own_frame) for key in changed), changed


@pytest.mark.asyncio
async def test_the_first_delete_receipt_restyles_no_transcript_row(tmp_path):
    """5.1: opening the Delete receipt does not restyle the transcript.

    Its rules load lazily, and Textual's push restyled every node of every
    screen after reading them: 5.3 s live on the first Delete with a
    scrolled-back 420-row transcript, 9.3 s with 3,000 rows mounted.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        target = native[rows[-2][0]]
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            transcript.select_message(target)
            await _settled_rows(transcript, heartbeat)
            await handle_console_delete_action(console._message, "delete", target)
            confirm = f"#console-message-action-delete-confirm-{target}"
            await _wait_for_selector(console, pilot, confirm, timeout=30.0)
            await heartbeat.quiet()
            _lazy_receipt_sheet(host)
            await heartbeat.quiet()
            kept = set(transcript.walk_children())
            assert len(kept) > 4 * _FOCUS_RESTYLES
            with _restyles(host) as restyled:
                console.query_one(confirm, Button).press()
                await _until(
                    lambda: bool(host.screen.query("#console-delete-receipt-undo")),
                    "Undo to be offered",
                )
                await heartbeat.quiet()
            touched = [node for node in restyled if node in kept and node.is_attached]
            assert len(touched) <= _FOCUS_RESTYLES, (
                f"the first receipt restyled {len(touched)} transcript nodes "
                f"({len(restyled)} in all)"
            )
            # The receipt itself still wears its lazily loaded rules.
            box = host.screen.query_one("#console-delete-receipt-box")
            assert box.styles.border_top[0] != "", box.styles.css
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_no_rule_in_the_delete_receipt_sheet_styles_anything_else(tmp_path):
    """Fidelity pin for preloading the receipt's sheet without a restyle.

    Read the sheet and restyle every node of a mounted Console with a long
    transcript: no computed style may change. Fails the day build_css.py's
    split lets a rule reach a node that is not part of a receipt.
    """
    from tldw_chatbook.Widgets.Console.console_message_delete_receipt import (
        ConsoleMessageDeleteReceiptModal,
    )

    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(40)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        transcript.select_message(native[rows[-2][0]])
        await pilot.pause()
        await pilot.pause()
        stylesheet = host.stylesheet
        path = _lazy_receipt_sheet(host)

        def styles() -> dict[int, str]:
            return {id(node): node.styles.css for node in host.walk_children()}

        before = styles()
        assert len(before) > 400
        ConsoleMessageDeleteReceiptModal.preload_sheet(host)
        assert stylesheet.has_source(path, "")
        assert any(
            "console-delete-receipt" in rule.selector_set[0].css
            for rule in stylesheet.rules
        )
        stylesheet.update(host)
        assert styles() == before


@pytest.mark.asyncio
async def test_an_idle_post_action_sync_restyles_nothing_and_walks_no_row(tmp_path):
    """5.2: the sync's own work does not grow with the transcript.

    Before this task each sync restyled the control bar twice (110 nodes)
    and walked every mounted transcript row four times for four rail labels.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, _native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        heartbeat = _Heartbeat()
        heartbeat.start()
        try:
            await _settled_rows(transcript, heartbeat)
            await console._sync_native_console_chat_ui()
            await heartbeat.quiet()
            for _attempt in range(2):
                with _restyles(host) as restyled, _walked() as walked:
                    await console._sync_native_console_chat_ui()
                assert not restyled, (
                    f"an idle sync restyled {len(restyled)} nodes: "
                    f"{[node.id or type(node).__name__ for node in restyled]}"
                )
                rows_walked = [node for node in walked if _inside(node, transcript)]
                assert not rows_walked, (
                    f"an idle sync walked {len(rows_walked)} transcript nodes "
                    f"({len(walked)} in all)"
                )
            # The four rail values the scoped lookups write still arrive:
            # overwrite them, and the next sync writes every one back.
            rail = console.query_one("#console-left-rail")
            written = [
                rail.query_one(
                    f"#console-model-section-{row} .console-model-section-value"
                )
                for row in ("temperature", "max-tokens", "streaming")
            ] + [rail.query_one("#console-model-section-recovery")]
            values = [str(widget.content) for widget in written]
            assert all(values[:3]), values
            for widget in written:
                widget.update("stale")
            await console._sync_native_console_chat_ui()
            assert [str(widget.content) for widget in written] == values
        finally:
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_an_idle_sync_reads_no_lineage_for_a_chat_without_memory(tmp_path):
    """5.2: a 3,000-message chat holding no memory pays no per-message read.

    The settings summary and the transcript's memory banner each asked which
    memory applies, and each answer captured the whole active lineage: one
    SQLite version read and one validated snapshot per message, ~19 ms of
    every ~29 ms call at 3,000 messages, twice per sync. Without a /rewind
    summary or an active selection event the answer is raw history, so the
    capture is skipped; once a selection exists, it runs again.
    """
    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(_CHAIN)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, store, _native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        controller = console._ensure_console_chat_controller()
        heartbeat = _Heartbeat()
        heartbeat.start()
        captures: list[int] = []
        real_capture = controller._durable_context_snapshots

        def capture(session_id: str, **kwargs: Any) -> Any:
            snapshots = real_capture(session_id, **kwargs)
            captures.append(len(snapshots or ()))
            return snapshots

        controller._durable_context_snapshots = capture
        try:
            await _settled_rows(transcript, heartbeat)
            await console._sync_native_console_chat_ui()
            await heartbeat.quiet()
            captures.clear()
            await console._sync_native_console_chat_ui()
            assert captures == [], (
                f"an idle sync captured the lineage {len(captures)} times"
            )
            memory = controller.context_control_inputs(store.active_session_id)[2]
            assert memory.kind is EffectiveMemoryKind.RAW

            # Positive control: an active selection event makes memory
            # possible, and the full validated path runs again.
            controller._context_repository.insert_memory_selection(
                ConsoleMemorySelectionRecord(
                    sequence=1,
                    selection_id="reset-at-leaf",
                    conversation_id=conversation_id,
                    activation_message_id=rows[-1][0],
                    selected_memory_id=None,
                    event_kind=MemorySelectionKind.RESET,
                    suppresses_legacy=True,
                    created_at="2026-10-05T00:00:00Z",
                )
            )
            await console._sync_native_console_chat_ui()
            assert captures and set(captures) == {_CHAIN}, captures
            memory = controller.context_control_inputs(store.active_session_id)[2]
            assert memory.kind is EffectiveMemoryKind.RAW
            assert memory.branch_head is not None
            assert memory.branch_head.selection_id == "reset-at-leaf"
        finally:
            del controller._durable_context_snapshots
            await heartbeat.stop()


@pytest.mark.asyncio
async def test_a_selection_projects_library_activity_per_active_turn_once(tmp_path):
    """5.2: moving the selection does not project every turn of the chat.

    Each selection or active-path change re-projects Library activity. The
    per-turn footer counts projected every active turn in turn, each pass
    rebuilding the active set and re-reading every row: 1,500 projections
    for a 3,000-message chat, quadratic on the event loop. Turns with no
    activity row cannot count, so only the selected turn is projected here.
    """
    from tldw_chatbook.Chat import console_chat_store, library_activity

    db = CharactersRAGDB(tmp_path / "long-chat.db", "long-chat")
    rows = _chain(400)
    conversation_id = _seed(db, rows)
    app = _build_test_app()
    app.chachanotes_db = db
    app.notify = lambda *args, **kwargs: None
    host = ConsoleHarness(app)

    async with host.run_test(size=_SIZE) as pilot:
        console, _store, native, transcript = await _open(
            host, pilot, db, conversation_id, rows[-1][0]
        )
        projected: list[Any] = []
        patched = []
        for module in (console_chat_store, library_activity):
            real = module.project_library_activity

            def spy(rows_: Any, active: Any, selected: Any, real: Any = real) -> Any:
                projected.append(selected)
                return real(rows_, active, selected)

            patched.append((module, real))
            module.project_library_activity = spy
        try:
            first = native[rows[0][0]]
            transcript.select_message(first)
            # The real selection path re-projects for the new selection.
            await _until(
                lambda: (
                    (console._library_activity._projection_token or ())[3:4] == (first,)
                ),
                "the selection's Library activity projection",
            )
        finally:
            for module, real in patched:
                module.project_library_activity = real
        assert projected, "the selection never projected Library activity"
        assert len(projected) <= 2, (
            f"one selection projected {len(projected)} turns of a 200-turn chat"
        )


@pytest.mark.asyncio
async def test_an_armed_delete_with_3000_off_path_messages_syncs_like_none(tmp_path):
    """5.2 AC#1/#2: off-path history adds no sync work, and the sync is lean.

    The selected prompt has a 3,000-message branch beneath it that is not on
    the active path, so the transcript shows four rows. With the Delete armed,
    one sync's restyles, transcript walks and store snapshots are counted
    with and without that branch.
    """

    async def measure(branch: int) -> tuple[int, int, int]:
        db = CharactersRAGDB(tmp_path / f"off-path-{branch}.db", f"off-{branch}")
        rows: list[tuple[str, str, str | None]] = [
            ("u1", "user", None),
            ("a1", "assistant", "u1"),
            ("u2", "user", "a1"),
            ("a2", "assistant", "u2"),
        ]
        parent = "u2"
        for message_id, role, _parent in _chain(branch, prefix="x"):
            rows.append((message_id, role, parent))
            parent = message_id
        conversation_id = _seed(db, rows)
        app = _build_test_app()
        app.chachanotes_db = db
        app.notify = lambda *args, **kwargs: None
        host = ConsoleHarness(app)
        async with host.run_test(size=_SIZE) as pilot:
            console, store, native, transcript = await _open(
                host, pilot, db, conversation_id, "a2"
            )
            target = native["u2"]
            heartbeat = _Heartbeat()
            heartbeat.start()
            try:
                transcript.select_message(target)
                await _until(
                    lambda: transcript.selected_message_id == target, "selection"
                )
                await handle_console_delete_action(console._message, "delete", target)
                await _wait_for_selector(
                    console,
                    pilot,
                    f"#console-message-action-delete-confirm-{target}",
                    timeout=30.0,
                )
                await heartbeat.quiet()
                snapshots = 0
                real_snapshot = store._snapshot

                def counted(message: Any) -> Any:
                    nonlocal snapshots
                    snapshots += 1
                    return real_snapshot(message)

                store._snapshot = counted
                try:
                    with _restyles(host) as restyled, _walked() as walked:
                        await console._sync_native_console_chat_ui()
                finally:
                    del store._snapshot
                rows_walked = [node for node in walked if _inside(node, transcript)]
                return len(restyled), len(rows_walked), snapshots
            finally:
                await heartbeat.stop()

    without = await measure(0)
    with_branch = await measure(_CHAIN)
    assert with_branch == without, (without, with_branch)
    assert with_branch[:2] == (0, 0), with_branch
