# Console Top-Area Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the Console screen's three-line identity header into a single full-width line, and move the status pills (Provider/Model/Assistant/RAG/Sources/Tools/Approvals) from the top control bar down to a full-width strip directly above the composer.

**Architecture:** The header change is CSS-only plus one class and one string edit — the shared `DestinationHeader` widget is untouched; a Console-scoped `console-header-inline` class flips its instance to a horizontal one-row layout. The pills move by extracting a new `ConsoleStatusChips` widget (in its own file) that owns the seven chips, their sync, and the approvals-review action; `ConsoleControlBar` keeps only the action row and shrinks to one row; `ChatScreen` composes the chips strip just before the composer and syncs it alongside the control bar.

**Spec:** `Docs/superpowers/specs/2026-07-21-console-top-area-layout-design.md` — read it first.

**Tech Stack:** Python ≥3.11, Textual 8.2.7 (`layout: horizontal`, `text-overflow: ellipsis`, `text-wrap: nowrap` all supported), pytest + pytest-asyncio (`app.run_test()` harnesses).

## Global Constraints

- Work happens in a **git worktree off `origin/dev`** (Task 1) — other agents mutate this checkout's branches concurrently.
- pytest runs from the **worktree's own venv** (Task 1 creates it). Verify `import tldw_chatbook` resolves to the worktree path before trusting any result.
- **Never hand-edit** `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate it with `.venv/bin/python tldw_chatbook/css/build_css.py` after editing `css/components/_agentic_terminal.tcss`.
- Console-only change. Do **not** modify `DestinationHeader`'s Python or any other screen.
- Chip widget **ids are unchanged**: `#console-provider-chip`, `#console-model-chip`, `#console-persona-chip`, `#console-rag-chip`, `#console-sources-chip`, `#console-tools-chip`, `#console-approvals-chip`. Action ids and `#console-control-bar` are unchanged.
- All inline-header CSS rules are prefixed with the id+class (`#console-workbench-header.console-header-inline …`) so they beat both `#console-workbench-header` (id) and `.density-compact .workbench-header-subtitle` (two-class) rules.
- Commit after every task; commit messages end with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- Known dev-tip baseline: some `Tests/UI/scheduling` failures + shell/snapshot failures pre-exist, and a fresh venv lacks `pytest-mock` (6 `Tests/Chat` errors: "fixture 'mocker' not found"). Judge "no regressions" against that baseline, not zero.

## File Structure

- `tldw_chatbook/Widgets/Console/console_status_chips.py` — NEW: `ConsoleChip`, `ConsoleApprovalsChip`, the `_chip` builder, and `ConsoleStatusChips` (compose + `sync_state` + approvals-review handler). Single responsibility: render and sync the status pills.
- `tldw_chatbook/Widgets/Console/console_control_bar.py` — loses the chip row, chip classes, chip sync, and the approvals handler; keeps the action row, hidden compat statics, `CompactModelBar`, `sync_actions`, `_summary_line`. Height 2 → 1.
- `tldw_chatbook/UI/Screens/chat_screen.py` — add `console-header-inline` class to the header; compose `ConsoleStatusChips` before the composer; control-bar height wrapper 2 → 1; sync the chips in `_sync_console_control_bar`.
- `tldw_chatbook/Widgets/Console/console_workbench_state.py` — prepend `"— "` to the Console subtitle.
- `tldw_chatbook/css/components/_agentic_terminal.tcss` (+ regenerated bundle) — inline-header rules and the chips-strip rule.
- Tests: new `Tests/UI/test_console_status_chips.py`; updates to `Tests/UI/test_console_workbench_contract.py`.

---

### Task 1: Worktree, environment, docs, backlog task

**Files:**
- Create: worktree at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-console-top` on new branch `feat/console-top-area-layout`
- Copy in: the spec + this plan from branch `chore/harness-review-tasks-320-334`

**Interfaces:**
- Produces: a green-baseline worktree every later task runs inside. All later paths are relative to the worktree root.

- [ ] **Step 1: Create the worktree off origin/dev**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git fetch origin
git worktree add ../tldw_chatbook-console-top -b feat/console-top-area-layout origin/dev
cd ../tldw_chatbook-console-top
```

- [ ] **Step 2: Bring the spec and plan onto the branch**

```bash
git checkout chore/harness-review-tasks-320-334 -- \
  "Docs/superpowers/specs/2026-07-21-console-top-area-layout-design.md" \
  "Docs/superpowers/plans/2026-07-21-console-top-area-layout.md"
git add Docs/superpowers
git commit -m "docs: import Console top-area layout spec and plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Create the worktree venv (takes a few minutes)**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import tldw_chatbook, pathlib; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

Expected: the printed path starts with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-console-top/`. If it points at the main checkout, STOP.

- [ ] **Step 4: Baseline the affected test files**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_internals_decomposition.py -q -p no:cacheprovider 2>&1 | tail -5
```

Expected: pass (record any pre-existing failures verbatim as this branch's baseline).

- [ ] **Step 5: Create the backlog task**

```bash
backlog task create "Console top area: one-line header + status pills above composer" \
  -d "Collapse the Console 3-line identity header into one full-width line and move the status pills from the top control bar to a full-width strip directly above the composer. Spec: Docs/superpowers/specs/2026-07-21-console-top-area-layout-design.md" \
  --ac "Console header renders on a single row with title, subtitle, and Ready badge" \
  --ac "Subtitle ellipsizes when narrow and the Ready badge stays flush to the right edge" \
  --ac "Status pills render in a full-width strip directly above the composer" \
  --ac "The action row (New tab/Settings/...) stays at the top under the header" \
  --ac "No other screen's header changes and all chip ids are preserved" \
  -s "In Progress" --plan "Docs/superpowers/plans/2026-07-21-console-top-area-layout.md"
git add backlog && git commit -m "docs(backlog): file Console top-area layout task

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Note: assign the ID against origin/dev (just fetched) and **re-verify it is still free at PR time** — dev mints IDs concurrently (recent collisions on task-404). Use repeated `--ac` flags, one per criterion (a single `--ac "a,b,c"` collapses into one AC).

---

### Task 2: One-line header

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workbench_state.py:127` (subtitle)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (~line 6970, add class to the `DestinationHeader`)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (add inline-header rules near the existing `#console-workbench-header` block ~line 1893)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss`
- Test: `Tests/UI/test_console_workbench_contract.py` (append a header mount test)

**Interfaces:**
- Consumes: nothing from other tasks.
- Produces: a Console header that renders on one row; no new Python symbols other tasks depend on.

- [ ] **Step 1: Write the failing mount test**

Append to `Tests/UI/test_console_workbench_contract.py` (it already imports `pytest`, `_build_test_app`, `_configure_native_ready_console`, `ConsoleHarness`, `_wait_for_selector`, `_is_displayed`, `_widget_text`):

```python
@pytest.mark.asyncio
async def test_console_header_renders_single_inline_row():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(120, 40)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workbench-header")
        header = console.query_one("#console-workbench-header")
        assert header.has_class("console-header-inline")
        # One row tall, not the old three-line stack.
        assert header.region.height == 1
        title = console.query_one("#workbench-header-title")
        subtitle = console.query_one("#workbench-header-subtitle")
        status = console.query_one("#workbench-header-status")
        assert _widget_text(title).strip() == "Console"
        # Subtitle keeps the em-dash lead and its copy.
        assert "source handoffs" in _widget_text(subtitle)
        # The status badge sits to the RIGHT of the subtitle on the same row.
        assert status.region.y == subtitle.region.y
        assert status.region.x >= subtitle.region.x + subtitle.region.width
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py::test_console_header_renders_single_inline_row -q -p no:cacheprovider
```

Expected: FAIL — `header.has_class("console-header-inline")` is False (and height is 2-3).

- [ ] **Step 3: Add the em-dash to the Console subtitle**

In `tldw_chatbook/Widgets/Console/console_workbench_state.py` line 127, change:

```python
            subtitle="Chat, source handoffs, live runs, and control actions.",
```

to:

```python
            subtitle="— Chat, source handoffs, live runs, and control actions.",
```

- [ ] **Step 4: Add the class at the Console header compose site**

In `tldw_chatbook/UI/Screens/chat_screen.py`, find the `DestinationHeader(...)` yield (~line 6970) and change its `classes`:

```python
            yield DestinationHeader(
                workbench_state.header,
                id="console-workbench-header",
                classes="workbench-header console-header-inline",
            )
```

- [ ] **Step 5: Add the inline-header CSS**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, immediately after the existing `#console-workbench-header { … }` block (~line 1900), add:

```tcss
#console-workbench-header.console-header-inline {
    layout: horizontal;
    height: 1;
    min-height: 1;
    border: none;
}

#console-workbench-header.console-header-inline .workbench-header-title {
    width: auto;
    height: 1;
    min-height: 1;
}

#console-workbench-header.console-header-inline .workbench-header-subtitle {
    width: 1fr;
    height: 1;
    min-height: 1;
    text-wrap: nowrap;
    text-overflow: ellipsis;
    margin: 0 1;
}

#console-workbench-header.console-header-inline .workbench-header-status {
    width: auto;
    height: 1;
    min-height: 1;
}
```

- [ ] **Step 6: Regenerate the bundle**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
git diff --stat tldw_chatbook/css/
grep -c "console-header-inline" tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: diff touches only `_agentic_terminal.tcss` and `tldw_cli_modular.tcss`; grep prints ≥ 4. If the bundle diff shows unrelated churn, STOP.

- [ ] **Step 7: Run the test and the file's suite**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py -q -p no:cacheprovider 2>&1 | tail -5
```

Expected: the new test passes; the rest of the file is unchanged (any pre-existing failure matches the Task-1 baseline).

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workbench_state.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_workbench_contract.py
git commit -m "feat(console): collapse the identity header to a single inline row

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Extract the ConsoleStatusChips widget (unmounted)

**Files:**
- Create: `tldw_chatbook/Widgets/Console/console_status_chips.py`
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py` (import the chip classes from the new module instead of defining them locally)
- Test: `Tests/UI/test_console_status_chips.py` (new)

**Interfaces:**
- Consumes: `ConsoleControlState` (from `tldw_chatbook.Chat.console_display_state`).
- Produces (Task 4 relies on these):
  - `class ConsoleStatusChips(Horizontal)` with `id` set by the caller; `sync_state(self, state: ConsoleControlState) -> None`.
  - `class ConsoleChip(Static)`, `class ConsoleApprovalsChip(ConsoleChip)` (moved here; `console_control_bar` re-imports them).

- [ ] **Step 1: Write the failing unit test**

Create `Tests/UI/test_console_status_chips.py`:

```python
"""Unit tests for the extracted Console status-chips strip."""

import pytest
from textual.app import App, ComposeResult

from tldw_chatbook.Chat.console_display_state import ConsoleControlState
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
    ConsoleStatusChips,
)


def _state(**overrides) -> ConsoleControlState:
    base = dict(
        provider_label="Provider: Anthropic",
        model_label="Model: claude-3-haiku",
        persona_label="Assistant: General",
        rag_label="RAG: off",
        sources_label="Sources: 0 staged",
        tools_label="Tools: 0 ready",
        approvals_label="Approvals: 0 pending",
        sources_active=False,
        tools_active=False,
        approvals_active=False,
    )
    base.update(overrides)
    return ConsoleControlState(**base)


class _ChipsApp(App):
    def __init__(self, state: ConsoleControlState) -> None:
        super().__init__()
        self._state = state

    def compose(self) -> ComposeResult:
        yield ConsoleStatusChips(self._state, id="console-status-chips")


@pytest.mark.asyncio
async def test_status_chips_render_all_seven_labels():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        for selector, expected in (
            ("#console-provider-chip", "Provider:"),
            ("#console-model-chip", "Model:"),
            ("#console-persona-chip", "Assistant:"),
            ("#console-rag-chip", "RAG:"),
            ("#console-sources-chip", "Sources:"),
            ("#console-tools-chip", "Tools:"),
            ("#console-approvals-chip", "Approvals:"),
        ):
            chip = app.query_one(selector)
            assert expected in str(chip.render())


@pytest.mark.asyncio
async def test_status_chips_sync_updates_labels_and_emphasis():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chips = app.query_one("#console-status-chips", ConsoleStatusChips)
        chips.sync_state(
            _state(
                model_label="Model: gpt-4o",
                sources_label="Sources: 3 staged",
                sources_active=True,
            )
        )
        await pilot.pause()
        assert "gpt-4o" in str(app.query_one("#console-model-chip").render())
        sources = app.query_one("#console-sources-chip")
        assert sources.has_class("console-chip-alert")
        assert not sources.has_class("console-chip-dim")
        # A zero counter stays dim.
        assert app.query_one("#console-tools-chip").has_class("console-chip-dim")


@pytest.mark.asyncio
async def test_approvals_chip_posts_review_requested():
    app = _ChipsApp(_state())
    async with app.run_test(size=(160, 6)) as pilot:
        await pilot.pause()
        chip = app.query_one("#console-approvals-chip", ConsoleApprovalsChip)
        posted: list[object] = []
        chip.post_message = lambda message: posted.append(message)  # type: ignore[assignment]
        chip.action_review_approval()
        assert any(
            isinstance(m, ConsoleApprovalsChip.ReviewRequested) for m in posted
        )
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest Tests/UI/test_console_status_chips.py -q -p no:cacheprovider
```

Expected: FAIL — `ModuleNotFoundError: tldw_chatbook.Widgets.Console.console_status_chips`.

- [ ] **Step 3: Create the new widget module**

Create `tldw_chatbook/Widgets/Console/console_status_chips.py`:

```python
"""Console status-pill strip (provider/model/persona/RAG/source/tool/approval).

Extracted from ConsoleControlBar so the pills can render in their own strip
directly above the composer. The widget owns the chip classes, the chip
builder, chip labelling + emphasis sync, and the approvals-review action.
"""

from __future__ import annotations

from typing import Any

from textual import events, on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Button, Static

from tldw_chatbook.Chat.console_display_state import (
    CONSOLE_INSPECTOR_NO_APPROVAL_REASON,
    ConsoleControlState,
)
from tldw_chatbook.Widgets.Chat_Widgets.chat_approval_card import ChatApprovalCard


class ConsoleChip(Static):
    """Focusable Console readiness chip.

    Chips ellipsize at 22 cells; focusing a chip lifts that cap (see
    ``.console-control-chip:focus`` in ``_agentic_terminal.tcss``) so the full
    label is reachable from the keyboard, while the tooltip keeps carrying the
    same full text on hover.
    """

    can_focus = True


class ConsoleApprovalsChip(ConsoleChip):
    """Approvals readiness chip that doubles as an approval-review action.

    Activating it (Enter/Space while focused, or click) asks the strip to
    focus the pending approval card in the transcript.
    """

    BINDINGS = [
        Binding("enter", "review_approval", "Review pending approval", show=False),
        Binding("space", "review_approval", "Review pending approval", show=False),
    ]

    class ReviewRequested(Message):
        """Posted when the approvals chip is activated from keyboard or mouse."""

    def action_review_approval(self) -> None:
        self.post_message(self.ReviewRequested())

    def _on_click(self, event: events.Click) -> None:
        self.post_message(self.ReviewRequested())


class ConsoleStatusChips(Horizontal):
    """Full-width strip of the seven Console readiness pills.

    Exposes ``sync_state`` so ``ChatScreen`` can refresh the pill labels and
    counter emphasis after provider/model/source/tool/approval state changes.
    """

    def __init__(self, state: ConsoleControlState, **kwargs: Any) -> None:
        """Initialize the strip.

        Args:
            state: Display-state snapshot for the readiness labels.
            **kwargs: Additional Textual widget arguments (id/classes).
        """
        classes = kwargs.pop("classes", "")
        # Reuse the existing chip-row class so its CSS continues to apply.
        super().__init__(
            classes=f"console-control-chip-row console-status-chips {classes}".strip(),
            **kwargs,
        )
        self.state = state
        self.styles.height = 1
        self.styles.min_height = 1
        self.styles.max_height = 1

    @staticmethod
    def _chip(
        label: str,
        *,
        id: str,
        emphasis: bool | None = None,
        chip_class: type[ConsoleChip] = ConsoleChip,
    ) -> ConsoleChip:
        """Build one readiness chip. Mirrors the former ConsoleControlBar._chip."""
        classes = "console-control-chip"
        if emphasis is False:
            classes += " console-chip-dim"
        elif emphasis is True:
            classes += " console-chip-alert"
        chip = chip_class(label, id=id, classes=classes)
        chip.tooltip = label
        return chip

    def compose(self) -> ComposeResult:
        yield self._chip(self.state.provider_label, id="console-provider-chip")
        yield self._chip(self.state.model_label, id="console-model-chip")
        yield self._chip(self.state.persona_label, id="console-persona-chip")
        yield self._chip(self.state.rag_label, id="console-rag-chip")
        yield self._chip(
            self.state.sources_label,
            id="console-sources-chip",
            emphasis=self.state.sources_active,
        )
        yield self._chip(
            self.state.tools_label,
            id="console-tools-chip",
            emphasis=self.state.tools_active,
        )
        yield self._chip(
            self.state.approvals_label,
            id="console-approvals-chip",
            emphasis=self.state.approvals_active,
            chip_class=ConsoleApprovalsChip,
        )

    def sync_state(self, state: ConsoleControlState) -> None:
        """Refresh pill labels and counter emphasis from a new snapshot."""
        if state == self.state:
            return
        self.state = state
        label_values = {
            "#console-provider-chip": state.provider_label,
            "#console-model-chip": state.model_label,
            "#console-persona-chip": state.persona_label,
            "#console-rag-chip": state.rag_label,
            "#console-sources-chip": state.sources_label,
            "#console-tools-chip": state.tools_label,
            "#console-approvals-chip": state.approvals_label,
        }
        for selector, label in label_values.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.update(label)
            chip.tooltip = label
        chip_emphasis = {
            "#console-sources-chip": state.sources_active,
            "#console-tools-chip": state.tools_active,
            "#console-approvals-chip": state.approvals_active,
        }
        for selector, active in chip_emphasis.items():
            try:
                chip = self.query_one(selector, Static)
            except NoMatches:
                continue
            chip.set_class(not active, "console-chip-dim")
            chip.set_class(active, "console-chip-alert")

    @on(ConsoleApprovalsChip.ReviewRequested)
    def on_approval_review_requested(
        self, event: ConsoleApprovalsChip.ReviewRequested
    ) -> None:
        """Focus the pending approval card in the transcript.

        Falls back to a notification when no approval is pending so the
        keyboard-only path never dead-ends silently.
        """
        event.stop()
        self._focus_pending_approval_card()

    def _focus_pending_approval_card(self) -> None:
        """Scroll the displayed approval card into view and focus its action."""
        try:
            cards = list(self.screen.query("#chat-approval-card"))
        except Exception:
            cards = []
        card = next(
            (
                candidate
                for candidate in cards
                if isinstance(candidate, ChatApprovalCard) and candidate.display
            ),
            None,
        )
        if card is None:
            self.app.notify(CONSOLE_INSPECTOR_NO_APPROVAL_REASON, severity="warning")
            return
        try:
            card.scroll_visible(animate=False)
        except Exception:
            pass
        try:
            batch_visible = card.query_one("#approval-batch-body").display
        except NoMatches:
            batch_visible = False
        target_id = "#approval-submit" if batch_visible else "#approval-allow-once"
        try:
            card.query_one(target_id, Button).focus()
        except NoMatches:
            pass
```

- [ ] **Step 4: Point ConsoleControlBar at the moved chip classes**

In `tldw_chatbook/Widgets/Console/console_control_bar.py`, delete the local `class ConsoleChip(Static):` and `class ConsoleApprovalsChip(ConsoleChip):` definitions (currently ~lines 80-111, right after `_summary_line`) and import them instead. Add to the imports block:

```python
from tldw_chatbook.Widgets.Console.console_status_chips import (
    ConsoleApprovalsChip,
    ConsoleChip,
)
```

Leave everything else in `console_control_bar.py` unchanged in this task — the control bar still composes its own chip row from the imported classes and still renders identically. (Its own copy of the approvals handler and `_focus_pending_approval_card` also stay for now; Task 4 removes them.)

- [ ] **Step 5: Run the new unit test and confirm the control bar still imports**

```bash
.venv/bin/pytest Tests/UI/test_console_status_chips.py -q -p no:cacheprovider 2>&1 | tail -5
.venv/bin/python -c "import tldw_chatbook.Widgets.Console.console_control_bar as m; print('control_bar import OK')"
```

Expected: 3 passed; control-bar import OK (no circular-import error — `console_status_chips` does not import `console_control_bar`).

- [ ] **Step 6: Run the control-bar contract suite (still green — app unchanged)**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py -q -p no:cacheprovider 2>&1 | tail -5
```

Expected: unchanged from Task 2 (the chips still live in the control bar this task).

- [ ] **Step 7: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_status_chips.py \
  tldw_chatbook/Widgets/Console/console_control_bar.py \
  Tests/UI/test_console_status_chips.py
git commit -m "feat(console): extract ConsoleStatusChips widget (not yet mounted)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Flip the wiring — pills above the composer

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_control_bar.py` (remove chip row + chip sync + approvals handler; height 2 → 1)
- Modify: `tldw_chatbook/UI/Screens/chat_screen.py` (import + compose the strip; control-bar height wrapper 2 → 1; sync wiring)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss` (chips-strip rule) + regenerate bundle
- Modify: `Tests/UI/test_console_workbench_contract.py` (split the `walk_children` coupling test; add placement test)

**Interfaces:**
- Consumes: `ConsoleStatusChips` (Task 3).
- Produces: no new symbols; final DOM order header → control bar (actions) → workspace grid → `#console-status-chips` → composer.

- [ ] **Step 1: Update the coupling test and add the placement test**

In `Tests/UI/test_console_workbench_contract.py`, the test
`test_console_has_one_canonical_visible_state_action_strip` (~line 206) walks
`control_bar.walk_children()` (~line 222) and asserts the combined text has
"Provider:", "Model:", "Settings", "Library RAG". Keep its setup; replace its
assertion block so chips are asserted over the new strip and actions over the
control bar:

```python
        control_bar = console.query_one("#console-control-bar")
        assert _is_displayed(control_bar)
        action_text = " ".join(
            _widget_text(child)
            for child in control_bar.walk_children()
            if _is_displayed(child)
        )
        assert action_text.count("Settings") == 1
        assert action_text.count("Library RAG") == 1
        # Pills moved out of the control bar into their own strip.
        assert "Provider:" not in action_text
        chips = console.query_one("#console-status-chips")
        chip_text = " ".join(
            _widget_text(child)
            for child in chips.walk_children()
            if _is_displayed(child)
        )
        assert chip_text.count("Provider:") == 1
        assert chip_text.count("Model:") == 1
```

Then append a placement test:

```python
@pytest.mark.asyncio
async def test_console_status_chips_sit_above_composer():
    app = _build_test_app()
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    async with host.run_test(size=(150, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-status-chips")
        chips = console.query_one("#console-status-chips")
        grid = console.query_one("#console-workspace-grid")
        composer = console.query_one("#console-native-composer")
        # Below the chat/rail grid, above the composer.
        assert chips.region.y >= grid.region.y + grid.region.height
        assert chips.region.y + chips.region.height <= composer.region.y
        assert _is_displayed(chips)
```

- [ ] **Step 2: Run both to verify they fail**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py -q -p no:cacheprovider \
  -k "canonical_visible_state_action_strip or above_composer" 2>&1 | tail -8
```

Expected: the placement test FAILs (`#console-status-chips` not found), and the edited coupling test FAILs (`"Provider:" not in action_text` is False — chips still in the control bar).

- [ ] **Step 3: Slim ConsoleControlBar to actions only**

In `tldw_chatbook/Widgets/Console/console_control_bar.py`:

3a. Change the height constant:

```python
CONSOLE_CONTROL_BAR_HEIGHT = 1
```

3b. In `compose`, delete the entire `with Horizontal(id="console-control-chip-row", …):` block and its seven `yield self._chip(...)` lines (the first block, ~lines 316-338). Keep the `#console-control-action-row` block and everything after it (hidden compat statics, summary line, `CompactModelBar`).

3c. Delete the now-unused `_chip` staticmethod (~lines 226-244).

3d. In `sync_state`, delete the chip entries from the `label_values` dict (the seven `"#console-*-chip"` keys) and delete the entire `chip_emphasis` block (~lines 213-224). Keep the `#console-*-label` (hidden) and `#console-control-status-line` entries.

3e. Delete the `@on(ConsoleApprovalsChip.ReviewRequested)` handler `on_approval_review_requested` and the `_focus_pending_approval_card` method (moved to `ConsoleStatusChips` in Task 3).

3f. Remove now-unused imports: the `ConsoleApprovalsChip, ConsoleChip` import added in Task 3, and — only if nothing else in the file still uses them — `events`, `Binding`, `Message`, `ChatApprovalCard`, `CONSOLE_INSPECTOR_NO_APPROVAL_REASON`. Verify each with `grep` before removing; leave any that another symbol still references.

- [ ] **Step 4: Compose the strip and drop the control-bar height wrapper to 1**

In `tldw_chatbook/UI/Screens/chat_screen.py`:

4a. Add the import near the other Console widget imports (with `ConsoleControlBar` ~line 217):

```python
from ...Widgets.Console.console_status_chips import ConsoleStatusChips
```

4b. Change the control-bar height wrapper (~line 7021) from `height=2` to `height=1`:

```python
            yield self._compact_console_workbench_widget(
                ConsoleControlBar(
                    control_state,
                    self.app_instance,
                    actions=workbench_state.actions,
                    on_sidebar_toggle_requested=self._toggle_console_chat_sidebar,
                    id="console-control-bar",
                    classes="console-control-bar",
                ),
                height=1,
            )
```

4c. Compose the strip immediately before the composer. Find the composer yield (~line 7448) and insert the chips just above it:

```python
                yield self._frame_console_region(right_handle, variant="quiet")
            yield ConsoleStatusChips(
                control_state,
                id="console-status-chips",
                classes="ds-panel",
            )
            yield self._frame_console_region(
                ConsoleComposerBar(
                    id="console-native-composer",
                    classes="ds-panel",
                    collapse_large_pastes=self._console_collapse_large_pastes_enabled(),
                    paste_collapse_threshold=self._console_paste_collapse_threshold(),
                )
            )
```

(The `ConsoleStatusChips` yield is a direct child of `#console-shell`, a sibling after `#console-workspace-grid` and before the framed composer.)

- [ ] **Step 5: Sync the strip alongside the control bar**

In `tldw_chatbook/UI/Screens/chat_screen.py`, in `_sync_console_control_bar` right after the `control_bar.sync_state(...)` call (~line 10898), add:

```python
            if control_bar is not None:
                control_bar.sync_state(control_state, actions=workbench_state.actions)
            try:
                status_chips = self.query_one(
                    "#console-status-chips", ConsoleStatusChips
                )
            except QueryError:
                status_chips = None
            if status_chips is not None:
                status_chips.sync_state(control_state)
```

- [ ] **Step 6: Add the chips-strip CSS and regenerate the bundle**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, near the existing `.console-control-chip-row` rule, add:

```tcss
#console-status-chips {
    width: 100%;
    min-width: 0;
    height: 1;
    min-height: 1;
    layout: horizontal;
    border: none;
    padding: 0 1;
}
```

Then:

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
git diff --stat tldw_chatbook/css/
grep -c "console-status-chips" tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: only the two CSS files change; grep ≥ 1.

- [ ] **Step 7: Run the affected suites**

```bash
.venv/bin/pytest Tests/UI/test_console_workbench_contract.py Tests/UI/test_console_status_chips.py Tests/UI/test_console_internals_decomposition.py -q -p no:cacheprovider 2>&1 | tail -8
```

Expected: all pass. If a test in `test_console_internals_decomposition.py` asserts chips inside the control bar or asserts the control bar's height is 2, update it to the new structure (chips via `#console-status-chips`; control-bar height 1) — never re-add the chip row to the control bar. List every such change in the report.

- [ ] **Step 8: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_control_bar.py \
  tldw_chatbook/UI/Screens/chat_screen.py \
  tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_workbench_contract.py
git commit -m "feat(console): move status pills to a strip above the composer

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: Full sweep, live verification, close-out

**Files:**
- Modify: backlog task file (AC checkboxes + implementation notes)

**Interfaces:**
- Consumes: everything prior. No new interfaces.

- [ ] **Step 1: Full UI + Chat sweep**

```bash
.venv/bin/pytest Tests/UI -q -p no:cacheprovider 2>&1 | tail -15
.venv/bin/pytest Tests/Chat -q -p no:cacheprovider 2>&1 | tail -8
```

Classify every failure: (a) documented dev-tip baseline (scheduling/shell/snapshot), (b) missing-`pytest-mock` env gap (`Tests/Chat` `mocker` errors), or (c) NOVEL. For any suspected-novel failure in a file this branch did not touch, re-run it in isolation and compare against `origin/dev` in a throwaway worktree before calling it a regression. Any novel regression must be fixed before proceeding.

- [ ] **Step 2: Live verification (REQUIRED — invoke the `verify` skill or the SVG-screenshot recipe)**

Drive the real Console at wide and narrow widths. Checklist:
1. Header is a single row: `Console` + em-dash + subtitle + `Ready` badge at the far right.
2. Narrow terminal (~90 cols): subtitle ellipsizes; `Ready` stays visible, flush right.
3. The action row (New tab / Settings / Attach context / Run Library RAG / Save Chatbook / Help) is directly under the header at the top.
4. The seven pills render in a full-width strip directly above the composer.
5. Sending / changing provider still updates the pills (they sync).

Capture one wide and one narrow screenshot for the PR (SVG → PNG via the `console_rail.svg` recipe: `app.save_screenshot(path)` in a `run_test` driver, then `qlmanage -t -s 2300 -o <dir> <svg>`).

- [ ] **Step 3: Close out the backlog task and commit**

Check off all ACs, add Implementation Notes (approach, files, the header-CSS specificity/border rationale, the chip-widget extraction and approvals-handler move), set status Done via `backlog task edit <id> -s Done`, then:

```bash
git add backlog
git commit -m "docs: close out Console top-area layout task

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 4: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`. PR targets `dev`; body includes the wide/narrow screenshots and links the spec. Re-verify the backlog task ID is still free on origin/dev at PR time. CI is intentionally cancelled in this repo — verify locally, don't block on CI.

---

## Self-Review Notes (already applied)

- Spec coverage: one-line header CSS + class + em-dash (Task 2); `ConsoleStatusChips` extraction with chip ids preserved and chip-row class reused (Task 3); pills mounted above composer, control bar actions-only at height 1, sync wired (Task 4); `walk_children` coupling test split + placement/header mount tests (Tasks 2/4); bundle regenerated from component file (Tasks 2/4); live verification (Task 5). No gaps.
- The approvals-review handler + `_focus_pending_approval_card` are added to the new widget in Task 3 and removed from the control bar in Task 4 — a one-task transitional duplication, called out so review expects it.
- Type consistency: `ConsoleStatusChips(state, id=…)` and `sync_state(state: ConsoleControlState)` are used identically in the widget (Task 3), the compose site, and the sync site (Task 4); `CONSOLE_CONTROL_BAR_HEIGHT` and the `height=1` wrapper move together (Task 4).
