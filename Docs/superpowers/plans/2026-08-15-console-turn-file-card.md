# Console Turn File Card Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Console transcript's one-line change-summary marker with a stacked per-file card: header counts, one expandable row per changed file, inline colored unified diffs.

**Architecture:** Pure presentation over TASK-1972's shipped subsystem. The transcript row whose message carries `change_review_run_id` renders as `ConsoleTurnFileCard` instead of a plain text row; the card lazily loads file rows (async, off the UI thread) from `AgentRunsChangeReviewProvider` and per-file diffs on expand. No data-path, persistence, or emit-seam changes; a `[console] turn_file_cards` switch (default ON) falls back to today's marker row.

**Tech Stack:** Python ≥3.11, Textual 8.x, pytest. Existing: `AgentRunsChangeReviewProvider` / `ReviewTurn` / `ChangedFile` (change_review_screen.py, change_tracking.py), `bridge.change_review_provider(conversation_id)`, `resolve_glyph`.

**Spec:** `Docs/superpowers/specs/2026-08-15-console-turn-file-review-design.md` (read the re-scoped sections; the tail is retired history).

## Global Constraints

- Never type-query message rows in tests; query by id/class only.
- Widget geometry tests run on the real app CSS stack (bundle loaded).
- All new rows always mounted, display-managed — never conditionally composed.
- Text-not-color-only for status/counts; glyphs through `resolve_glyph`.
- Kill switch OFF must render **byte-identical** marker text to today.
- Off-UI-thread for every shadow-repo read (`changed_files`, `diff_text`).
- Worktree: create via `superpowers:using-git-worktrees` off `origin/dev`; branch `feat/console-turn-file-card`.

---

### Task 1: Pure file-entry assembly

**Files:**
- Modify: `tldw_chatbook/Chat/console_display_state.py` (append near the other builders)
- Test: `Tests/Chat/test_console_turn_file_entries.py` (new)

**Interfaces:**
- Consumes: `ChangedFile` (`tldw_chatbook.Workspaces.change_tracking`; fields `path, status, adds=0, dels=0, old_path=None, binary=False` — verified), snapshot-row dicts (keys: `root`, `kind`, `tracking_error`, `files_changed`, `adds`, `dels`).
- Produces: `TurnFileEntry` dataclass and `turn_file_entries(rows, changed_by_root) -> list[TurnFileEntry]` — Task 2 renders these verbatim.

- [ ] **Step 1: Write the failing tests**

```python
# Tests/Chat/test_console_turn_file_entries.py
from tldw_chatbook.Chat.console_display_state import TurnFileEntry, turn_file_entries
from tldw_chatbook.Workspaces.change_tracking import ChangedFile


def _row(root="/ws", kind="turn", tracking_error=None):
    return {"root": root, "kind": kind, "tracking_error": tracking_error,
            "files_changed": 1, "adds": 3, "dels": 1}


def test_single_root_entries_use_bare_relpaths():
    rows = [_row()]
    changed = {"/ws": [ChangedFile(path="a/b.py", status="M", adds=3, dels=1)]}
    entries = turn_file_entries(rows, changed)
    assert entries == [TurnFileEntry(
        label="a/b.py", path="a/b.py", root="/ws",
        status="M", adds=3, dels=1)]


def test_multi_root_entries_prefix_the_root_name():
    rows = [_row(root="/ws/one"), _row(root="/ws/two")]
    changed = {
        "/ws/one": [ChangedFile(path="x.md", status="A", adds=5, dels=0)],
        "/ws/two": [ChangedFile(path="y.md", status="D", adds=0, dels=7)],
    }
    labels = [e.label for e in turn_file_entries(rows, changed)]
    assert labels == ["one/x.md", "two/y.md"]


def test_tracking_error_rows_yield_no_entries():
    rows = [_row(tracking_error="git failed")]
    assert turn_file_entries(rows, {"/ws": []}) == []
```

- [ ] **Step 2: Run to verify they fail**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/Chat/test_console_turn_file_entries.py -q`
Expected: FAIL — `ImportError: cannot import name 'TurnFileEntry'`

- [ ] **Step 3: Implement**

Append to `tldw_chatbook/Chat/console_display_state.py`:

```python
@dataclass(frozen=True)
class TurnFileEntry:
    """One changed file on a turn's transcript card (task: turn file card).

    ``label`` is what the row prints: the bare relpath for a single-root
    turn, ``<root-name>/<relpath>`` when the turn touched several roots.
    ``path``/``root`` stay separate because the diff loader needs the
    exact (row, path) pair the provider expects.
    """

    label: str
    path: str
    root: str
    status: str
    adds: int
    dels: int


def turn_file_entries(
    rows: "Sequence[Mapping[str, Any]]",
    changed_by_root: "Mapping[str, Sequence[Any]]",
) -> "list[TurnFileEntry]":
    """Assemble a turn card's file rows from its snapshot rows.

    Args:
        rows: The run's ``change_snapshots`` rows (one per root), in
            emit order. Tracking-error rows contribute nothing — the
            card degrades to the marker text for those.
        changed_by_root: ``root -> ChangedFile`` list, as returned by
            ``AgentRunsChangeReviewProvider.changed_files`` per row.

    Returns:
        Entries in row order then file order, labels root-prefixed only
        when more than one clean root contributed.
    """
    clean = [r for r in rows if not r.get("tracking_error")]
    multi_root = len({str(r["root"]) for r in clean}) > 1
    entries: list[TurnFileEntry] = []
    for row in clean:
        root = str(row["root"])
        prefix = f"{PurePath(root).name}/" if multi_root else ""
        for changed in changed_by_root.get(root, ()):  # ChangedFile
            entries.append(
                TurnFileEntry(
                    label=f"{prefix}{changed.path}",
                    path=changed.path,
                    root=root,
                    status=str(changed.status),
                    adds=int(changed.adds),
                    dels=int(changed.dels),
                )
            )
    return entries
```

Add `from pathlib import PurePath` to the module imports if absent.

- [ ] **Step 4: Run to verify they pass**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/Chat/test_console_turn_file_entries.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add Tests/Chat/test_console_turn_file_entries.py tldw_chatbook/Chat/console_display_state.py
git commit -m "feat(console): pure file-entry assembly for the turn file card"
```

---

### Task 2: ConsoleTurnFileCard widget

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_turn_file_card.py`
- Test: `Tests/UI/test_console_turn_file_card.py` (new)

**Interfaces:**
- Consumes: `TurnFileEntry` / `turn_file_entries` (Task 1);
  `AgentRunsChangeReviewProvider` duck-type: `.turns() -> list[ReviewTurn(run_id, label, rows)]`, `.changed_files(row) -> list[ChangedFile]`, `.diff_text(row, path) -> str`, `.diff_display_max_lines: int`.
- Produces: `ConsoleTurnFileCard(marker_text, run_id, provider_factory, id=...)` — Task 3 constructs exactly this.

- [ ] **Step 1: Write the failing widget tests**

```python
# Tests/UI/test_console_turn_file_card.py
"""Turn file card: header, async rows, expandable capped diffs.

Runs on the REAL app CSS stack (screen css + bundle): geometry measured
without the bundle is not measured (task-15110's lesson).
"""
from pathlib import Path

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.css import build_css
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    ConsoleTurnFileCard,
)

_CSS_DIR = Path(build_css.__file__).parent
_SELF, _SCOPED = build_css.screen_css_paths(_CSS_DIR)

MARKER = "✎ Edited 2 files  +8 −3 — review with `v`"


class _FakeProvider:
    diff_display_max_lines = 2000

    def __init__(self):
        from tldw_chatbook.UI.Screens.change_review_screen import ReviewTurn

        self._row = {"root": "/ws", "kind": "turn", "tracking_error": None,
                     "files_changed": 2, "adds": 8, "dels": 3,
                     "baseline_sha": "b", "end_sha": "e", "run_id": "run-1"}
        self._turn = ReviewTurn(run_id="run-1", label="t", rows=(self._row,))

    def turns(self):
        return [self._turn]

    def changed_files(self, row):
        from tldw_chatbook.Workspaces.change_tracking import ChangedFile

        assert row is self._row
        return [ChangedFile(path="a.py", status="M", adds=5, dels=3),
                ChangedFile(path="b.md", status="A", adds=3, dels=0)]

    def diff_text(self, row, path):
        assert path in ("a.py", "b.md")
        return "--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old line\n+new line\n"


class _Host(App):
    CSS_PATH = [str(_SELF), str(_CSS_DIR / "tldw_cli_modular.tcss"), str(_SCOPED)]

    def compose(self) -> ComposeResult:
        yield ConsoleTurnFileCard(
            MARKER, "run-1", lambda: _FakeProvider(),
            id="card-under-test",
        )


async def _settled_card(pilot):
    card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
    for _ in range(60):
        if card.query(".console-turn-file-row"):
            break
        await pilot.pause(0.02)
    return card


@pytest.mark.asyncio
async def test_header_shows_marker_and_rows_load_async():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        header = card.query_one(".console-turn-file-header")
        assert MARKER.split(" — ")[0] in str(header.render())
        rows = list(card.query(".console-turn-file-row"))
        assert len(rows) == 2
        assert "a.py" in str(rows[0].render())
        assert "+5" in str(rows[0].render()) and "−3" in str(rows[0].render())


@pytest.mark.asyncio
async def test_expand_shows_capped_scrolling_diff():
    async with _Host().run_test(size=(120, 40)) as pilot:
        card = await _settled_card(pilot)
        row = card.query(".console-turn-file-row").first()
        row.focus()
        await pilot.press("enter")
        body = None
        for _ in range(60):
            bodies = card.query(".console-turn-file-diff")
            if bodies and bodies.first().display:
                body = bodies.first()
                break
            await pilot.pause(0.02)
        assert body is not None, "diff body never displayed"
        assert "+new line" in str(body.render())
        assert str(body.styles.overflow_y) == "auto"
        assert body.styles.max_height is not None
        # collapse again: display-managed, never unmounted
        row.focus()
        await pilot.press("enter")
        await pilot.pause()
        assert not body.display and body.is_mounted


@pytest.mark.asyncio
async def test_provider_failure_degrades_to_marker_only():
    class _Broken(_FakeProvider):
        def turns(self):
            raise RuntimeError("shadow repo unavailable")

    class _BrokenHost(_Host):
        def compose(self) -> ComposeResult:
            yield ConsoleTurnFileCard(
                MARKER, "run-1", lambda: _Broken(), id="card-under-test"
            )

    async with _BrokenHost().run_test(size=(120, 40)) as pilot:
        await pilot.pause(0.3)
        card = pilot.app.query_one("#card-under-test", ConsoleTurnFileCard)
        assert not list(card.query(".console-turn-file-row"))
        assert MARKER.split(" — ")[0] in str(
            card.query_one(".console-turn-file-header").render()
        )
```

- [ ] **Step 2: Run to verify they fail**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/UI/test_console_turn_file_card.py -q`
Expected: FAIL — `ModuleNotFoundError: ... console_turn_file_card`

- [ ] **Step 3: Implement the widget**

```python
# tldw_chatbook/Widgets/Console/console_turn_file_card.py
"""Stacked per-file change card for one agent turn (turn file card spec).

Pure presentation over TASK-1972's snapshots: the card receives the
marker text (counts precomputed at emit), a run id, and a ZERO-ARG
provider factory (late-binding, the transcript-region builder
convention). File rows load asynchronously on mount; each row's diff
loads asynchronously on FIRST expand and is cached. Every shadow-repo
read runs off the UI thread. A provider failure of any kind degrades to
the marker header alone -- the card must never break the transcript.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

from loguru import logger
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    TurnFileEntry,
    turn_file_entries,
)
from tldw_chatbook.Widgets.glyph_fallback import resolve_glyph

_CHEVRON_CLOSED = "▸"
_CHEVRON_OPEN = "▾"


class ConsoleTurnFileCard(Vertical):
    DEFAULT_CSS = """
    ConsoleTurnFileCard {
        height: auto;
        min-height: 1;
    }
    ConsoleTurnFileCard .console-turn-file-header {
        height: 1;
        text-style: bold;
    }
    ConsoleTurnFileCard .console-turn-file-row {
        height: 1;
        min-height: 1;
        width: 100%;
        text-align: left;
    }
    ConsoleTurnFileCard .console-turn-file-diff {
        height: auto;
        max-height: 20;
        overflow-y: auto;
        overflow-x: hidden;
        scrollbar-size: 1 1;
    }
    """

    def __init__(
        self,
        marker_text: str,
        run_id: str,
        provider_factory: Callable[[], Any],
        *,
        id: str | None = None,
    ) -> None:
        super().__init__(id=id, classes="console-turn-file-card")
        self._marker_text = marker_text
        self._run_id = run_id
        self._provider_factory = provider_factory
        self._entries: list[TurnFileEntry] = []
        self._row_for_entry: dict[int, dict] = {}
        self._diff_cache: dict[int, str] = {}

    def compose(self) -> ComposeResult:
        # Header keeps the marker's counts but drops its "review with v"
        # trailer -- the rows ARE the review affordance now; `v` still
        # works and stays documented in the F1 help.
        head = self._marker_text.split(" — ")[0]
        yield Static(
            head,
            classes="console-turn-file-header",
            markup=False,
        )
        yield Vertical(classes="console-turn-file-rows")

    def on_mount(self) -> None:
        self.run_worker(
            self._load_rows(),
            group="console-turn-file-card-load",
            exit_on_error=False,
        )

    async def _load_rows(self) -> None:
        try:
            provider = self._provider_factory()
            if provider is None:
                return
            def _read() -> tuple[list[TurnFileEntry], dict[int, dict]]:
                turn = next(
                    (t for t in provider.turns() if t.run_id == self._run_id),
                    None,
                )
                if turn is None:
                    return [], {}
                changed_by_root = {
                    str(row["root"]): provider.changed_files(row)
                    for row in turn.rows
                    if not row.get("tracking_error")
                }
                entries = turn_file_entries(turn.rows, changed_by_root)
                row_by_root = {
                    str(row["root"]): row for row in turn.rows
                }
                mapping = {
                    idx: row_by_root[entry.root]
                    for idx, entry in enumerate(entries)
                }
                return entries, mapping

            entries, mapping = await asyncio.to_thread(_read)
        except Exception:
            logger.opt(exception=True).warning(
                "Turn file card row load failed; keeping marker-only header."
            )
            return
        if not self.is_mounted or not entries:
            return
        self._entries = entries
        self._row_for_entry = mapping
        rows_box = self.query_one(".console-turn-file-rows", Vertical)
        for idx, entry in enumerate(entries):
            chevron = resolve_glyph(_CHEVRON_CLOSED)
            row = Button(
                f"{chevron} {entry.status}  {entry.label}  "
                f"+{entry.adds} −{entry.dels}",
                classes="console-turn-file-row",
                compact=True,
            )
            row.entry_index = idx
            diff_body = VerticalScroll(classes="console-turn-file-diff")
            diff_body.display = False
            await rows_box.mount(row)
            await rows_box.mount(diff_body)

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        idx = getattr(event.button, "entry_index", None)
        if idx is None:
            return
        event.stop()
        bodies = list(self.query(".console-turn-file-diff"))
        rows = list(self.query(".console-turn-file-row"))
        body = bodies[idx]
        row = rows[idx]
        entry = self._entries[idx]
        if body.display:
            body.display = False
            row.label = (
                f"{resolve_glyph(_CHEVRON_CLOSED)} {entry.status}  "
                f"{entry.label}  +{entry.adds} −{entry.dels}"
            )
            return
        if idx not in self._diff_cache:
            snapshot_row = self._row_for_entry.get(idx)
            provider = self._provider_factory()
            if provider is None or snapshot_row is None:
                return
            try:
                text = await asyncio.to_thread(
                    provider.diff_text, snapshot_row, entry.path
                )
            except Exception:
                logger.opt(exception=True).warning(
                    "Turn file card diff load failed for %s", entry.label
                )
                return
            cap = int(getattr(provider, "diff_display_max_lines", 2000))
            lines = text.splitlines()
            if len(lines) > cap:
                hidden = len(lines) - cap
                lines = lines[:cap] + [f"… {hidden} more lines (diff capped)"]
            self._diff_cache[idx] = "\n".join(lines)
            if not body.is_mounted:
                return
            await body.mount(
                Static(
                    self._styled_diff(self._diff_cache[idx]),
                    classes="console-turn-file-diff-text",
                    markup=False,
                )
            )
        body.display = True
        row.label = (
            f"{resolve_glyph(_CHEVRON_OPEN)} {entry.status}  "
            f"{entry.label}  +{entry.adds} −{entry.dels}"
        )

    @staticmethod
    def _styled_diff(text: str):
        from rich.text import Text

        out = Text()
        for line in text.splitlines(keepends=False):
            if line.startswith("+") and not line.startswith("+++"):
                out.append(line + "\n", style="green")
            elif line.startswith("-") and not line.startswith("---"):
                out.append(line + "\n", style="red")
            elif line.startswith("@@"):
                out.append(line + "\n", style="dim")
            else:
                out.append(line + "\n")
        return out
```

- [ ] **Step 4: Run to verify they pass**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/UI/test_console_turn_file_card.py -q`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_turn_file_card.py Tests/UI/test_console_turn_file_card.py
git commit -m "feat(console): ConsoleTurnFileCard — stacked expandable per-file diffs"
```

---

### Task 3: Row-factory branch, provider factory, kill switch

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_transcript.py` (row factory, the `row.kind == "message"` branch, ~line 3311; region constructor kwargs)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (extract the provider recipe used by the `v` opener at ~17755–17784 into `_console_change_review_provider()`; pass the factory kwarg at the `ConsoleTranscriptRegion(` construction, ~line 13858)
- Test: `Tests/UI/test_console_turn_file_card_factory.py` (new)

**Interfaces:**
- Consumes: `ConsoleTurnFileCard(marker_text, run_id, provider_factory, id=...)` (Task 2); `ConsoleChatMessage.change_review_run_id`; `get_cli_setting("console", "turn_file_cards", True)`.
- Produces: `ConsoleTranscriptRegion(change_review_provider_factory=...)` kwarg (zero-arg callable returning a provider or `None`); `ChatScreen._console_change_review_provider() -> provider | None`.

- [ ] **Step 1: Write the failing factory tests**

```python
# Tests/UI/test_console_turn_file_card_factory.py
"""The transcript renders a card for change-summary rows -- switch-gated.

OFF must be byte-identical to today's marker row: the kill switch is a
pure presentation toggle (spec §2 re-scoped)."""
```

Test bodies (same file):

```python
import pytest

from tldw_chatbook.Chat.console_chat_models import (
    ConsoleChatMessage,
    ConsoleMessageRole,
)
from tldw_chatbook.Widgets.Console.console_turn_file_card import (
    ConsoleTurnFileCard,
)

MARKER = "✎ Edited 2 files  +8 −3 — review with `v`"


def _summary_message():
    return ConsoleChatMessage(
        role=ConsoleMessageRole.TOOL,
        content=MARKER,
        status="complete",
        change_review_run_id="run-1",
    )


@pytest.mark.asyncio
async def test_summary_row_renders_card_when_enabled(monkeypatch):
    from Tests.UI.test_console_native_chat_flow import (
        ConsoleHarness,
        _build_test_app,
        _wait_for_selector,
    )

    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        store.append_message(_summary_message())
        await console._sync_native_console_chat_ui()
        for _ in range(40):
            if console.query(ConsoleTurnFileCard):
                break
            await pilot.pause(0.02)
        assert console.query(ConsoleTurnFileCard)


@pytest.mark.asyncio
async def test_summary_row_stays_plain_marker_when_disabled(monkeypatch):
    from Tests.UI.test_console_native_chat_flow import (
        ConsoleHarness,
        _build_test_app,
        _wait_for_selector,
        _wait_for_text,
    )
    import tldw_chatbook.Widgets.Console.console_transcript as transcript_mod

    monkeypatch.setattr(
        transcript_mod,
        "get_cli_setting",
        lambda section, key, default=None: (
            False
            if (section, key) == ("console", "turn_file_cards")
            else default
        ),
    )
    app = _build_test_app()
    host = ConsoleHarness(app)
    async with host.run_test(size=(160, 48)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-native-composer")
        store = console._ensure_console_chat_store()
        store.append_message(_summary_message())
        await console._sync_native_console_chat_ui()
        # The plain marker text still renders, byte-identical, and no
        # card mounts -- the switch is a pure presentation toggle.
        await _wait_for_text(console, pilot, MARKER)
        assert not console.query(ConsoleTurnFileCard)
```

(`_wait_for_text` is imported from `Tests.UI.test_console_native_chat_flow`
alongside the harness imports already shown.)

- [ ] **Step 2: Run to verify they fail**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/UI/test_console_turn_file_card_factory.py -q`
Expected: FAIL — no `ConsoleTurnFileCard` renders (branch absent)

- [ ] **Step 3: Implement the branch and factory**

In `console_transcript.py`, inside the row factory's `row.kind == "message"` branch, BEFORE the markdown/plain dispatch:

```python
        if row.kind == "message" and row.message is not None:
            review_run_id = getattr(row.message, "change_review_run_id", None)
            if (
                review_run_id
                and self._change_review_provider_factory is not None
                and bool(
                    get_cli_setting("console", "turn_file_cards", True)
                )
            ):
                return ConsoleTurnFileCard(
                    str(row.message.content),
                    str(review_run_id),
                    self._change_review_provider_factory,
                    id=f"console-turn-file-card-{row.message.id}",
                )
```

Region constructor: add `change_review_provider_factory: Callable[[], Any] | None = None` keyword, store on `self._change_review_provider_factory` (default `None` keeps every existing harness working). Import `ConsoleTurnFileCard` and `get_cli_setting` at module top.

In `chat_screen.py`: extract the `v` opener's provider recipe (bridge lookup → `controller._agent_conversation_id(active)` → `bridge.change_review_provider(conversation_id)` → attach the `run_active` probe) into:

```python
    def _console_change_review_provider(self):
        """The v-opener's provider recipe, shared with the turn file card.

        Returns None whenever any collaborator is missing -- the card
        degrades to the marker header; only the v opener toasts.
        """
```

…returning the provider or `None`, with the opener now calling it (opener keeps its own toast on `None`). Pass `change_review_provider_factory=self._console_change_review_provider` at the `ConsoleTranscriptRegion(` construction site.

- [ ] **Step 4: Run the new tests plus the opener's existing tests**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest Tests/UI/test_console_turn_file_card_factory.py Tests/UI/test_console_change_review_opener.py -q` (adjust the second path to wherever the `v`-opener wiring tests live — grep `change_review` under Tests/UI)
Expected: all pass

- [ ] **Step 5: Mutation check, then commit**

Temporarily delete the new factory branch; run the factory tests — the enabled-path test must FAIL. Restore.

```bash
git add tldw_chatbook/Widgets/Console/console_transcript.py tldw_chatbook/UI/Screens/chat_screen.py Tests/UI/test_console_turn_file_card_factory.py
git commit -m "feat(console): render change-summary rows as turn file cards (kill-switched)"
```

---

### Task 4: Whole-module verification, docs, backlog seeds

**Files:**
- Modify: `Docs/User_Guide/` Console page ("Verified against" stamp + a short "Reviewing a turn's file changes" paragraph)
- Create: two backlog tasks via the Backlog.md CLI (IDs swept across ALL remotes first, leapfrogged)

- [ ] **Step 1: Run the touched modules whole**

Run: `VIRTUAL_ENV=.venv .venv/bin/python -m pytest -p no:randomly -q Tests/Chat/test_console_turn_file_entries.py Tests/UI/test_console_turn_file_card.py Tests/UI/test_console_turn_file_card_factory.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_console_internals_decomposition.py`
Expected: all pass (native flow + internals guard the transcript's existing contracts)

- [ ] **Step 2: Update the User Guide Console page**

Add a short section: the card appears after agent turns that changed files; Enter/click a row to expand its diff; `v` still opens the full review screen (revert lives there). Update the page's "Verified against" stamp.

- [ ] **Step 3: File the backlog seeds**

Two tasks (sweep remote refs for free IDs first — the collision lesson):
1. "Turn file card: annotate/feedback loop and Review affordance" (V1.5 — card button opening `ChangeReviewScreen` at the turn, hunk-level feedback fed back to the agent).
2. "Change review: git commit/push/PR modes" (V2 — the `current`/`commit`/`push`/PR contextual actions when the workspace is a git repo).

- [ ] **Step 4: Commit and open the PR**

```bash
git add -A
git commit -m "docs(console): user-guide stamp + backlog seeds for the turn file card"
```

PR body summarizes: pure presentation over TASK-1972; OFF-switch byte-parity; no data-path changes.
