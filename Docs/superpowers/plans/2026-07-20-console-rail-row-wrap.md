# Console Rail Conversation Row Wrap Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Console left-rail conversation rows show flush-left names that wrap to up to two width-aware lines instead of hard-truncating at 20 characters, with the metadata line kept as the row's final line.

**Architecture:** All rendering changes live in `ConsoleWorkspaceContextTray` (`tldw_chatbook/Widgets/Console/console_workspace_context.py`): a pure cell-aware wrap helper feeds both label building and the precomputed list heights, so they cannot disagree. The wrap budget is measured from the tray's `content_region.width` inside `_fit_height_to_content` (the choke point that runs after mount, resize, and every `sync_state` recompose) with a guarded recompose only when the width actually changes. Two CSS additions (`scrollbar-gutter: stable` on the rail body; `height: auto` on row lines) break the two layout feedback loops the design identified.

**Spec:** `Docs/superpowers/specs/2026-07-20-console-rail-conversation-row-layout-design.md` — read it before starting.

**Tech Stack:** Python ≥3.11, Textual 8.2.7, Rich (`rich.cells.cell_len`), pytest + pytest-asyncio (`app.run_test()` harnesses).

## Global Constraints

- Work happens in a **git worktree off `origin/dev`** (Task 1) — other agents mutate this checkout's branches concurrently.
- pytest must run from the **worktree's own venv** (Task 1 creates it). Verify `import tldw_chatbook` resolves to the worktree path before trusting any test result.
- **Never hand-edit** `tldw_chatbook/css/tldw_cli_modular.tcss` — regenerate it with `.venv/bin/python tldw_chatbook/css/build_css.py` after editing `css/components/_agentic_terminal.tcss`.
- Rows stay `Button` widgets. Do not change widget ids (`console-workspace-conversation-{index}`, `console-conversation-star-{index}`) or the attributes attached to them (`conversation_id`, `row_key`, `native_session_id`, `scope_type`, `workspace_id`, `starred`).
- `format_console_conversation_row_label(text, *, subagent_count)` keeps its exact signature and behavior (escape whole text, append badge on its own line).
- Wrap/truncate operate on **raw** text; markup-escaping happens afterward (inside `format_console_conversation_row_label`). Never escape before measuring.
- Commit after every task. Commit messages end with `Co-Authored-By:` line per repo convention.
- Known dev-tip test baseline: ~12 `Tests/UI/scheduling` failures + 2 shell/snapshot failures pre-exist. Judge "no regressions" against that baseline, not zero.

## File Structure

- `tldw_chatbook/Widgets/Console/console_workspace_context.py` — all Python changes (wrap helpers, label building, heights, budget measurement).
- `tldw_chatbook/css/components/_agentic_terminal.tcss` — two CSS additions; bundle regenerated from it.
- `tldw_chatbook/Workspaces/display_state.py` — comment only (nominal row-height constant).
- `Tests/UI/test_console_conversation_row_wrap.py` — NEW: unit tests for the wrap/truncate helpers.
- `Tests/UI/test_console_agent_rail.py`, `Tests/UI/test_console_native_chat_flow.py`, `Tests/UI/test_console_workspace_context_rail.py` — updated assertions.

---

### Task 1: Worktree, environment, docs, backlog task

**Files:**
- Create: worktree at `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-rail-wrap` on new branch `feat/console-rail-row-wrap`
- Copy in: the spec + this plan from branch `chore/harness-review-tasks-320-334`

**Interfaces:**
- Produces: a green-baseline worktree every later task runs inside. All later paths are relative to the worktree root.

- [ ] **Step 1: Create the worktree off origin/dev**

```bash
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook
git fetch origin
git worktree add ../tldw_chatbook-rail-wrap -b feat/console-rail-row-wrap origin/dev
cd ../tldw_chatbook-rail-wrap
```

- [ ] **Step 2: Bring the spec and plan onto the branch**

```bash
git checkout chore/harness-review-tasks-320-334 -- \
  "Docs/superpowers/specs/2026-07-20-console-rail-conversation-row-layout-design.md" \
  "Docs/superpowers/plans/2026-07-20-console-rail-row-wrap.md"
git add Docs/superpowers
git commit -m "docs: import Console rail row-wrap spec and plan

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 3: Create the worktree venv (takes a few minutes)**

```bash
python3 -m venv .venv && .venv/bin/pip install -q -e ".[dev]"
.venv/bin/python -c "import tldw_chatbook, pathlib; print(pathlib.Path(tldw_chatbook.__file__).resolve())"
```

Expected: the printed path starts with `/Users/macbook-dev/Documents/GitHub/tldw_chatbook-rail-wrap/`. If it points at the main checkout, STOP — tests would exercise the wrong code.

- [ ] **Step 4: Baseline the three affected test files**

```bash
.venv/bin/pytest Tests/UI/test_console_agent_rail.py Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: all pass (record any pre-existing failures verbatim; they define this branch's baseline).

- [ ] **Step 5: Create the backlog task**

```bash
backlog task create "Console rail: flush-left width-aware two-line conversation names" \
  -d "Left-rail conversation names are hard-truncated at 20 chars and indented by a marker prefix. Implement the approved design: flush-left names wrapping to up to 2 width-aware lines, metadata line kept, guarded relabel on width change. Spec: Docs/superpowers/specs/2026-07-20-console-rail-conversation-row-layout-design.md" \
  --ac "Conversation names render flush left with no marker prefix,Names wrap to at most 2 lines at the rail's measured width and ellipsize only when 2 lines are insufficient,Metadata line renders as the row's final line and is cell-truncated to the row budget,Precomputed list heights match rendered row heights for mixed wrapped/badge rows,No recompose oscillation when the rail scrollbar toggles or the terminal resizes" \
  -s "In Progress" --plan "Docs/superpowers/plans/2026-07-20-console-rail-row-wrap.md"
git add backlog && git commit -m "docs(backlog): file Console rail row-wrap task

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

Note: backlog IDs must be assigned against origin/dev (just fetched). Re-verify the ID is still free at merge time (see memory: six past collisions).

---

### Task 2: Cell-aware wrap and truncate helpers (pure functions)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py` (module level, after the `_STATUS_DETAIL_LABELS` block, ~line 52)
- Test: `Tests/UI/test_console_conversation_row_wrap.py` (new)

**Interfaces:**
- Produces:
  - `wrap_console_conversation_title(title: str, budget: int) -> tuple[str, ...]` — 1 or 2 raw text lines, each ≤ `budget` cells; normalizes blank titles to `"Untitled conversation"`; clamps budget to ≥ 10.
  - `truncate_console_row_cells(text: str, budget: int) -> str` — raw text ≤ `budget` cells, `…`-terminated only when truncation occurred.
  - Module constants `_TITLE_WRAP_MAX_LINES = 2`, `_MIN_TITLE_WRAP_BUDGET = 10`.
- Consumes: nothing from other tasks.

- [ ] **Step 1: Write the failing tests**

Create `Tests/UI/test_console_conversation_row_wrap.py`:

```python
"""Unit tests for the Console rail title wrap/truncate helpers."""

from rich.cells import cell_len

from tldw_chatbook.Widgets.Console.console_workspace_context import (
    truncate_console_row_cells,
    wrap_console_conversation_title,
)


def test_short_title_is_single_line() -> None:
    assert wrap_console_conversation_title("Quick test", 30) == ("Quick test",)


def test_exact_fit_is_single_line_without_ellipsis() -> None:
    assert wrap_console_conversation_title("0123456789", 10) == ("0123456789",)


def test_long_title_wraps_at_word_boundary() -> None:
    lines = wrap_console_conversation_title("Debugging the RAG splat", 20)
    assert lines == ("Debugging the RAG", "splat")


def test_overflowing_title_ellipsizes_second_line() -> None:
    lines = wrap_console_conversation_title(
        "Debugging the RAG splat bug in retrieval", 20
    )
    assert lines == ("Debugging the RAG", "splat bug in retrie…")
    assert all(cell_len(line) <= 20 for line in lines)


def test_cut_landing_on_word_boundary_keeps_full_head() -> None:
    # "aaaa bbbb " is exactly 10 cells; the whole first two words must
    # survive on line 1 rather than breaking back to "aaaa".
    assert wrap_console_conversation_title("aaaa bbbb cccc", 10) == (
        "aaaa bbbb",
        "cccc",
    )


def test_spaceless_token_hard_breaks() -> None:
    lines = wrap_console_conversation_title("A" * 50, 20)
    assert lines == ("A" * 20, "A" * 19 + "…")


def test_budget_floor_clamps_to_ten_cells() -> None:
    lines = wrap_console_conversation_title("aaaa bbbb cccc", 3)
    assert lines == ("aaaa bbbb", "cccc")
    assert all(cell_len(line) <= 10 for line in lines)


def test_wide_characters_measure_in_cells() -> None:
    lines = wrap_console_conversation_title("日" * 8, 10)
    assert lines == ("日" * 5, "日" * 3)
    assert all(cell_len(line) <= 10 for line in lines)


def test_blank_title_falls_back_to_untitled() -> None:
    assert wrap_console_conversation_title("   ", 30) == ("Untitled conversation",)


def test_truncate_returns_short_text_unchanged() -> None:
    assert truncate_console_row_cells("saved - 2d", 20) == "saved - 2d"


def test_truncate_ellipsizes_long_text() -> None:
    result = truncate_console_row_cells("Workspace A - saved chat - 2d", 20)
    assert result == "Workspace A - saved…"
    assert cell_len(result) <= 20


def test_truncate_is_cell_aware_for_wide_characters() -> None:
    result = truncate_console_row_cells("日" * 12, 10)
    assert cell_len(result) <= 10
    assert result.endswith("…")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest Tests/UI/test_console_conversation_row_wrap.py -q
```

Expected: FAIL — `ImportError: cannot import name 'truncate_console_row_cells'`.

- [ ] **Step 3: Implement the helpers**

In `console_workspace_context.py`, add to the imports:

```python
from rich.cells import cell_len
```

Then add after the `_ROW_BUTTON_HEIGHT_WITH_BADGE` block (these constants get removed in Task 3; placement near the top of the module is what matters):

```python
_TITLE_WRAP_MAX_LINES = 2
_MIN_TITLE_WRAP_BUDGET = 10
_ROW_ELLIPSIS = "…"


def _cut_prefix_cells(text: str, budget: int) -> str:
    """Return the longest prefix of ``text`` that fits within ``budget`` cells."""
    used = 0
    for index, char in enumerate(text):
        width = cell_len(char)
        if used + width > budget:
            return text[:index]
        used += width
    return text


def truncate_console_row_cells(text: str, budget: int) -> str:
    """Truncate raw row text to at most ``budget`` terminal cells.

    Cell-aware (CJK/emoji safe). Appends an ellipsis only when truncation
    actually occurred. Operates on raw text -- markup escaping happens later
    in ``format_console_conversation_row_label``.
    """
    budget = max(1, int(budget))
    text = str(text)
    if cell_len(text) <= budget:
        return text
    keep = _cut_prefix_cells(text, budget - cell_len(_ROW_ELLIPSIS))
    return f"{keep.rstrip()}{_ROW_ELLIPSIS}"


def wrap_console_conversation_title(title: str, budget: int) -> tuple[str, ...]:
    """Word-wrap a raw conversation title into at most two budget-width lines.

    Widths are measured in terminal cells, not characters. Spaceless tokens
    longer than one line hard-break at the budget. When two lines are still
    insufficient the second line is ellipsized. The budget is clamped to
    ``_MIN_TITLE_WRAP_BUDGET`` to avoid degenerate wraps on absurdly narrow
    rails. Blank titles normalize to "Untitled conversation" (keep in sync
    with ``ConsoleWorkspaceContextTray._conversation_title``).
    """
    budget = max(_MIN_TITLE_WRAP_BUDGET, int(budget))
    remaining = str(title).strip() or "Untitled conversation"
    lines: list[str] = []
    while remaining:
        if len(lines) == _TITLE_WRAP_MAX_LINES - 1:
            lines.append(truncate_console_row_cells(remaining, budget))
            break
        if cell_len(remaining) <= budget:
            lines.append(remaining)
            break
        head = _cut_prefix_cells(remaining, budget)
        on_boundary = head.endswith(" ") or (
            len(head) < len(remaining) and remaining[len(head)] == " "
        )
        if on_boundary:
            lines.append(head.rstrip())
            remaining = remaining[len(head) :].lstrip()
            continue
        break_at = head.rfind(" ")
        if break_at > 0:
            lines.append(remaining[:break_at].rstrip())
            remaining = remaining[break_at + 1 :].lstrip()
        else:
            lines.append(head)
            remaining = remaining[len(head) :].lstrip()
    return tuple(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest Tests/UI/test_console_conversation_row_wrap.py -q
```

Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
git add Tests/UI/test_console_conversation_row_wrap.py \
  tldw_chatbook/Widgets/Console/console_workspace_context.py
git commit -m "feat(console): add cell-aware title wrap/truncate helpers for rail rows

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Rework row labels and heights (drop marker, wrap names, truncate metadata)

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py`
- Modify: `Tests/UI/test_console_agent_rail.py:98-158`
- Modify: `Tests/UI/test_console_native_chat_flow.py:80-96`

**Interfaces:**
- Consumes: `wrap_console_conversation_title`, `truncate_console_row_cells` (Task 2).
- Produces (Task 4 and tests rely on these exact names):
  - `_conversation_row_render_height(name_line_count: int, subagent_count: int) -> int` (module function, new signature).
  - `_conversation_button(text, *, id, conversation_id, tooltip_label=None, selected=False, subagent_count=0, name_line_count=1) -> Button` (staticmethod; new keyword `name_line_count`).
  - `_conversation_browser_rows_height(rows, budget) -> int`, `_conversation_browser_list_height(browser, budget) -> int`, `_legacy_conversation_list_height(section, budget) -> int` (staticmethods; new `budget` parameter).
  - Instance methods `_browser_title_budget() -> int`, `_legacy_title_budget() -> int`; instance attribute `_row_content_width: int` (initialized to `_FALLBACK_ROW_CONTENT_WIDTH = 20` in `__init__`).
  - Constants: `_FALLBACK_ROW_CONTENT_WIDTH = 20`, `_BROWSER_ROW_CHROME_WIDTH = 6`, `_LEGACY_ROW_CHROME_WIDTH = 2`, `_ROW_BOTTOM_MARGIN = 1`.
  - REMOVED: `_MAX_CONVERSATION_ROW_TITLE`, `_conversation_visible_title`, `_ROW_BUTTON_HEIGHT`, `_ROW_BUTTON_HEIGHT_WITH_BADGE`, the `GLYPH_ACTIVE` import, and the marker prefix in both compose paths.

- [ ] **Step 1: Update the tests that pin the old behavior (write them to the new contract first)**

In `Tests/UI/test_console_agent_rail.py`, replace `test_ellipsized_title_still_pairs_with_full_badge` (lines 98-116):

```python
def test_wrapped_title_still_pairs_with_full_badge():
    """Long titles now wrap to two budget-width lines; that wrapping must
    never interact with -- or swallow -- the badge, which lives on an
    entirely separate line."""
    from rich.cells import cell_len

    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
        wrap_console_conversation_title,
    )

    name_lines = wrap_console_conversation_title("A" * 40, 20)
    assert name_lines == ("A" * 20, "A" * 19 + "…")
    assert all(cell_len(line) <= 20 for line in name_lines)

    composed = "\n".join((*name_lines, "saved chat - 2m"))
    label = format_console_conversation_row_label(composed, subagent_count=5)
    lines = label.splitlines()
    assert lines[0] == name_lines[0]
    assert lines[-1] == "[dim]\\[5 Sub-Agents][/dim]"
    assert "Sub-Agents" not in "\n".join(lines[:-1])
```

Replace `test_short_title_without_badge_is_unchanged` (lines 119-135):

```python
def test_short_title_without_badge_is_unchanged():
    """Badge-less rows and short titles get no extra lines or ellipsis."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        format_console_conversation_row_label,
        wrap_console_conversation_title,
    )

    assert wrap_console_conversation_title("Short title", 20) == ("Short title",)

    composed = "Short title\nsaved chat - 2m"
    label = format_console_conversation_row_label(composed, subagent_count=0)
    assert label == composed
    assert label.count("\n") == 1
```

Replace `test_conversation_row_height_grows_only_when_badge_present` (lines 138-158) — the existing calls keep working because `name_line_count` defaults to 1; extend it to pin the wrapped case:

```python
def test_conversation_row_height_tracks_name_lines_and_badge():
    """Row height = name lines + metadata line, plus one line only when a
    badge will actually render."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        ConsoleWorkspaceContextTray,
    )

    badge_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary",
        id="row-badge",
        conversation_id="c1",
        subagent_count=2,
    )
    plain_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title\nsecondary",
        id="row-plain",
        conversation_id="c2",
        subagent_count=0,
    )
    wrapped_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title line one\nline two\nsecondary",
        id="row-wrapped",
        conversation_id="c3",
        subagent_count=0,
        name_line_count=2,
    )
    wrapped_badge_button = ConsoleWorkspaceContextTray._conversation_button(
        "Title line one\nline two\nsecondary",
        id="row-wrapped-badge",
        conversation_id="c4",
        subagent_count=1,
        name_line_count=2,
    )
    assert int(badge_button.styles.height.value) == 3
    assert int(plain_button.styles.height.value) == 2
    assert int(wrapped_button.styles.height.value) == 3
    assert int(wrapped_badge_button.styles.height.value) == 4
```

In `Tests/UI/test_console_native_chat_flow.py`, replace the two tests at lines 80-96:

```python
def test_console_workspace_conversation_titles_wrap_instead_of_truncating():
    """Long workspace conversation titles wrap at the rail budget."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        wrap_console_conversation_title,
    )

    assert wrap_console_conversation_title("Console UAT Workspace Chat", 20) == (
        "Console UAT",
        "Workspace Chat",
    )


def test_console_workspace_conversation_title_preserves_duplicate_suffix():
    """Duplicate-title disambiguators should remain visible in rail labels."""
    from tldw_chatbook.Widgets.Console.console_workspace_context import (
        wrap_console_conversation_title,
    )

    title = "Chat [deadbeef]"
    assert ConsoleWorkspaceContextTray._conversation_title(title) == title
    assert wrap_console_conversation_title(title, 20) == (title,)
```

(Keep the existing `ConsoleWorkspaceContextTray` import that file already has.)

- [ ] **Step 2: Run the updated tests to verify they fail**

```bash
.venv/bin/pytest Tests/UI/test_console_agent_rail.py -q \
  Tests/UI/test_console_native_chat_flow.py -q -k "wrap or badge or height or duplicate_suffix"
```

Expected: FAIL — `_conversation_button` rejects `name_line_count`; wrapped-height assertions fail (currently 3, not 4).

- [ ] **Step 3: Rework `console_workspace_context.py`**

3a. Imports: remove `GLYPH_ACTIVE` from the `console_glyphs` import (keep `GLYPH_COLLAPSED`, `GLYPH_EXPANDED`). Remove `CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT` from the `display_state` import (keep the other two names).

3b. Delete these module members: `_MAX_CONVERSATION_ROW_TITLE`, `_ROW_BUTTON_HEIGHT`, `_ROW_BUTTON_HEIGHT_WITH_BADGE`, the old `_conversation_row_render_height`, and the class staticmethod `_conversation_visible_title`.

3c. Add module constants next to the Task 2 helpers:

```python
# Pre-measurement fallback for the tray's usable row width. Only the first
# frame before `_fit_height_to_content` measures the real width renders with
# it; the guarded relabel pass corrects it immediately (see
# `_maybe_relabel_for_width`).
_FALLBACK_ROW_CONTENT_WIDTH = 20
# Grouped-browser rows share their line with the star control (width 3 +
# 1 margin) and carry 1 cell of button padding per side.
_BROWSER_ROW_CHROME_WIDTH = 6
# Legacy-section rows have no star column; only button padding.
_LEGACY_ROW_CHROME_WIDTH = 2
# Every row button carries a 1-line bottom margin (see the row CSS).
_ROW_BOTTOM_MARGIN = 1


def _conversation_row_render_height(
    name_line_count: int, subagent_count: int
) -> int:
    """Return the button height for a row: name lines + metadata line,
    plus a dedicated badge line when this conversation has historical
    sub-agent runs (see `format_console_conversation_row_label`)."""
    height = max(1, int(name_line_count)) + 1
    if subagent_count > 0:
        height += 1
    return height
```

3d. In `ConsoleWorkspaceContextTray.__init__`, after `self.show_heading = show_heading`, add:

```python
        self._row_content_width = _FALLBACK_ROW_CONTENT_WIDTH
```

3e. Add budget accessors to the class (near `_conversation_title`):

```python
    def _browser_title_budget(self) -> int:
        """Cells available to grouped-browser row text."""
        return max(
            _MIN_TITLE_WRAP_BUDGET,
            self._row_content_width - _BROWSER_ROW_CHROME_WIDTH,
        )

    def _legacy_title_budget(self) -> int:
        """Cells available to legacy-section row text (no star column)."""
        return max(
            _MIN_TITLE_WRAP_BUDGET,
            self._row_content_width - _LEGACY_ROW_CHROME_WIDTH,
        )
```

3f. Replace `_conversation_button` (keep it a staticmethod):

```python
    @staticmethod
    def _conversation_button(
        text: str,
        *,
        id: str,
        conversation_id: str,
        tooltip_label: str | None = None,
        selected: bool = False,
        subagent_count: int = 0,
        name_line_count: int = 1,
    ) -> Button:
        # Escaped-then-markup rendering round-trips plain text unchanged while
        # letting `format_console_conversation_row_label` safely append a dim
        # "[N Sub-Agents]" badge when this conversation has historical runs.
        label = format_console_conversation_row_label(
            text, subagent_count=subagent_count
        )
        button = Button(
            Text.from_markup(label),
            id=id,
            classes="console-workspace-conversation-row",
            compact=True,
        )
        button.conversation_id = conversation_id
        fallback_tooltip = text.splitlines()[0].strip() if text else text
        button.tooltip = f"Switch to {tooltip_label or fallback_tooltip}"
        button.set_class(selected, "console-workspace-conversation-row-selected")
        row_height = _conversation_row_render_height(name_line_count, subagent_count)
        button.styles.height = row_height
        button.styles.min_height = row_height
        return button
```

3g. Replace the three height staticmethods:

```python
    @staticmethod
    def _legacy_conversation_list_height(
        section: ConsoleWorkspaceConversationSectionState,
        budget: int,
    ) -> int:
        """Return the full content height for legacy conversation rows."""
        if not section.rows:
            return _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
        return max(
            _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT,
            sum(
                _conversation_row_render_height(
                    len(wrap_console_conversation_title(row.title, budget)),
                    0,
                )
                + _ROW_BOTTOM_MARGIN
                for row in section.rows
            ),
        )

    @staticmethod
    def _conversation_browser_rows_height(
        rows: tuple[ConsoleConversationBrowserRow, ...],
        budget: int,
    ) -> int:
        """Total height for a row sequence: per-row button height (from the
        same wrap the labels use, so the two cannot disagree) plus margin."""
        return sum(
            _conversation_row_render_height(
                len(wrap_console_conversation_title(row.title, budget)),
                row.subagent_count,
            )
            + _ROW_BOTTOM_MARGIN
            for row in rows
        )

    @staticmethod
    def _conversation_browser_list_height(
        browser: ConsoleConversationBrowserState,
        budget: int,
    ) -> int:
        """Return the full content height for the grouped browser rows."""
        height = 0
        for section in browser.sections:
            height += _CONVERSATION_BROWSER_HEADER_HEIGHT
            if section.collapsed:
                continue
            if section.groups:
                for group in section.groups:
                    height += _CONVERSATION_BROWSER_HEADER_HEIGHT
                    if group.collapsed:
                        continue
                    if group.rows:
                        height += ConsoleWorkspaceContextTray._conversation_browser_rows_height(
                            group.rows, budget
                        )
                    elif group.empty_copy:
                        height += _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
                continue
            if section.rows:
                height += ConsoleWorkspaceContextTray._conversation_browser_rows_height(
                    section.rows, budget
                )
            elif section.empty_copy:
                height += _CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT
        return max(_CONVERSATION_BROWSER_EMPTY_COPY_HEIGHT, height)
```

3h. In `_compose_legacy_conversation_section`, update the list-height call and the row loop (currently ~lines 683-704):

```python
            conversation_list = Vertical(id="console-workspace-conversations")
            legacy_budget = self._legacy_title_budget()
            conversation_list.styles.height = self._legacy_conversation_list_height(
                section, legacy_budget
            )
            conversation_list.styles.min_height = 0
            with conversation_list:
                if section.rows:
                    for index, row in enumerate(section.rows):
                        title = self._conversation_title(row.title)
                        name_lines = wrap_console_conversation_title(
                            row.title, legacy_budget
                        )
                        status = self._conversation_status(row.status)
                        detail = self._conversation_detail_status(row.status)
                        status_suffix = f" [{status}]" if status else ""
                        secondary = truncate_console_row_cells(
                            detail or "conversation", legacy_budget
                        )
                        yield self._conversation_button(
                            "\n".join((*name_lines, secondary)),
                            id=f"console-workspace-conversation-{index}",
                            conversation_id=row.conversation_id,
                            tooltip_label=f"{title}{status_suffix}",
                            selected=row.selected,
                            name_line_count=len(name_lines),
                        )
                else:
```

(The `else:` branch and everything after it stay as they are.)

3i. In `_compose_conversation_browser`, update the list-height call (~line 792):

```python
        conversation_list = Vertical(id="console-workspace-conversations")
        conversation_list.styles.height = self._conversation_browser_list_height(
            browser, self._browser_title_budget()
        )
```

3j. Replace the row/star body of `_compose_conversation_browser_row` (~lines 919-961; everything from `title = ...` through the two `star_button.styles` lines — the tooltip/attribute assignments after them stay):

```python
        with Horizontal(classes="console-conversation-browser-row-line"):
            budget = self._browser_title_budget()
            title = self._conversation_title(row.title)
            name_lines = wrap_console_conversation_title(row.title, budget)
            status = self._conversation_status(row.status)
            detail = self._conversation_detail_status(row.status)
            secondary_parts = [
                part
                for part in (row.workspace_label, detail, row.updated_label)
                if str(part or "").strip()
            ]
            secondary = truncate_console_row_cells(
                " - ".join(secondary_parts) or "conversation", budget
            )
            status_suffix = f" [{status}]" if status else ""
            row_button = self._conversation_button(
                "\n".join((*name_lines, secondary)),
                id=f"console-workspace-conversation-{index}",
                conversation_id=row.conversation_id or row.row_key,
                tooltip_label=f"{title}{status_suffix}",
                selected=row.selected,
                subagent_count=row.subagent_count,
                name_line_count=len(name_lines),
            )
            row_button.row_key = row.row_key
            row_button.native_session_id = row.native_session_id
            row_button.scope_type = row.scope_type
            row_button.workspace_id = row.workspace_id
            row_button.styles.width = "1fr"
            row_button.styles.min_width = 0
            yield row_button

            star_disabled = not marks_available or not row.star_enabled
            star_button = Button(
                "*" if row.starred else ".",
                id=f"console-conversation-star-{index}",
                classes="console-workspace-action console-conversation-star",
                compact=True,
                disabled=star_disabled,
            )
            # Match the row button's height so the star control still spans
            # the full row whatever the name-line and badge count.
            star_row_height = _conversation_row_render_height(
                len(name_lines), row.subagent_count
            )
            star_button.styles.height = star_row_height
            star_button.styles.min_height = star_row_height
```

3k. In `_conversation_section` (the legacy summary fallback, ~line 385), nothing changes — it uses `_conversation_title`, which stays.

- [ ] **Step 4: Run the affected suites**

```bash
.venv/bin/pytest Tests/UI/test_console_conversation_row_wrap.py \
  Tests/UI/test_console_agent_rail.py \
  Tests/UI/test_console_workspace_context_rail.py -q
.venv/bin/pytest Tests/UI/test_console_native_chat_flow.py -q
```

Expected: all pass. If `test_console_workspace_context_rail.py` has failures, they will be label-content or geometry assertions still expecting the marker/two-space prefixes — update those assertions to the new flush-left labels (names have no leading spaces; metadata is the last pre-badge line), never by re-adding prefixes to the widget.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py \
  Tests/UI/test_console_agent_rail.py Tests/UI/test_console_native_chat_flow.py \
  Tests/UI/test_console_workspace_context_rail.py
git commit -m "feat(console): flush-left wrapped conversation names in rail rows

Drop the marker prefix and 20-char truncation; wrap names to two
budget-width lines, cell-truncate the metadata line, and derive row and
list heights from the same wrap result.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Width measurement and guarded relabel in the fit pass

**Files:**
- Modify: `tldw_chatbook/Widgets/Console/console_workspace_context.py` (`_fit_height_to_content` plus one new method)
- Test: `Tests/UI/test_console_workspace_context_rail.py` (append two tests)

**Interfaces:**
- Consumes: `_row_content_width`, budget accessors (Task 3).
- Produces: `_maybe_relabel_for_width() -> bool` — measures `content_region.width`; on change stores it and schedules the recompose path; returns True when a relabel was scheduled. Called at the top of `_fit_height_to_content`.

- [ ] **Step 1: Write the failing tests**

Append to `Tests/UI/test_console_workspace_context_rail.py` (it already imports `ConsoleWorkspaceContextTray`, `pytest`, and the `_browser_row`/`_base_grouped_workspace_state` fixtures; also add `from rich.cells import cell_len` to its imports):

```python
_LONG_ROW_TITLE = (
    "A very long conversation title that overflows the rail width easily"
)


def _long_title_grouped_state():
    return _base_grouped_workspace_state(
        rows=(
            _browser_row(
                "conv-long",
                _LONG_ROW_TITLE,
                selected=True,
                updated_sort="2026-06-27T09:00:00",
            ),
        )
    )


def _first_row_name_lines(console) -> list[str]:
    row_button = console.query_one("#console-workspace-conversation-0", Button)
    lines = str(row_button.label).splitlines()
    # Last line is the metadata line; badge rows are not used in this fixture.
    return lines[:-1]


@pytest.mark.asyncio
async def test_console_rail_titles_wrap_at_measured_width() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_long_title_grouped_state())
        await pilot.pause()
        await pilot.pause()

        # The fit pass replaced the pre-measurement fallback with the real
        # measured width.
        assert tray._row_content_width == tray.content_region.width
        budget = tray._browser_title_budget()
        name_lines = _first_row_name_lines(console)
        assert 1 <= len(name_lines) <= 2
        assert all(cell_len(line) <= budget for line in name_lines)
        # Flush left: no marker prefix on the name.
        assert not name_lines[0].startswith(" ")

        # Stability: further fit passes must not flap the labels (guarded
        # relabel -- no recompose oscillation).
        settled = str(
            console.query_one("#console-workspace-conversation-0", Button).label
        )
        await pilot.pause()
        await pilot.pause()
        assert (
            str(
                console.query_one(
                    "#console-workspace-conversation-0", Button
                ).label
            )
            == settled
        )


@pytest.mark.asyncio
async def test_console_rail_list_height_matches_rendered_rows() -> None:
    app = _build_test_app()
    host = ConsoleHarness(app)

    async with host.run_test(size=(160, 44)) as pilot:
        console = host.screen_stack[-1]
        await _wait_for_selector(console, pilot, "#console-workspace-context")
        tray = console.query_one(
            "#console-workspace-context", ConsoleWorkspaceContextTray
        )
        tray.sync_state(_long_title_grouped_state())
        await pilot.pause()
        await pilot.pause()

        conversation_list = console.query_one("#console-workspace-conversations")
        # Scope every query to the list: `console-workspace-empty-copy` is
        # also used by status statics OUTSIDE the conversation list, which
        # would overcount the expected height.
        row_buttons = list(
            conversation_list.query(
                ".console-workspace-conversation-row"
            ).results(Button)
        )
        assert row_buttons
        rows_height = sum(
            int(button.styles.height.value) + 1 for button in row_buttons
        )
        header_count = len(
            conversation_list.query(".console-conversation-browser-section-header")
        ) + len(
            conversation_list.query(".console-conversation-browser-group-header")
        )
        empty_copies = len(
            conversation_list.query(".console-workspace-empty-copy")
        )
        assert (
            int(conversation_list.styles.height.value)
            == rows_height + header_count + empty_copies
        )


@pytest.mark.asyncio
async def test_console_rail_wrap_budget_tracks_terminal_width() -> None:
    """Spec: the same long title must wrap at different budgets at different
    terminal widths (the rail is 3fr, not fixed)."""
    budgets: dict[str, int] = {}
    for label, size in (("wide", (200, 44)), ("narrow", (100, 44))):
        app = _build_test_app()
        host = ConsoleHarness(app)
        async with host.run_test(size=size) as pilot:
            console = host.screen_stack[-1]
            await _wait_for_selector(console, pilot, "#console-workspace-context")
            tray = console.query_one(
                "#console-workspace-context", ConsoleWorkspaceContextTray
            )
            tray.sync_state(_long_title_grouped_state())
            await pilot.pause()
            await pilot.pause()
            budget = tray._browser_title_budget()
            budgets[label] = budget
            name_lines = _first_row_name_lines(console)
            assert all(cell_len(line) <= budget for line in name_lines)
    assert budgets["narrow"] < budgets["wide"]
```

Note: `_base_grouped_workspace_state` already accepts `rows=` and passes it through `_grouped_browser_state`; if the local helper signature differs, thread `rows` through the same way the existing fixtures do. Empty sections contribute a 1-line header plus (when present) a 1-line empty-copy `Static` carrying class `console-workspace-empty-copy` — the height cross-check counts them from the DOM, not from assumptions.

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest Tests/UI/test_console_workspace_context_rail.py -q \
  -k "measured_width or matches_rendered_rows"
```

Expected: FAIL — `tray._row_content_width` still equals the fallback (20), not the measured width (no relabel pass exists yet).

- [ ] **Step 3: Implement the guarded relabel**

In `ConsoleWorkspaceContextTray`, add after `_restore_parent_scroll`:

```python
    def _maybe_relabel_for_width(self) -> bool:
        """Rewrap row labels when the measured content width has changed.

        The check lives in the fit pass rather than ``on_resize`` because the
        tray's frame variant (solid <-> none) changes the *content* width
        without changing the outer size, so no resize event fires for it.
        The equality guard is what prevents recompose feedback loops --
        steady-state passes are free. Returns True when a relabel recompose
        was scheduled (the caller should skip fitting; the scheduled passes
        re-fit after the recompose).
        """
        region = getattr(self, "content_region", None)
        if region is None or region.width <= 0:
            return False
        measured = int(region.width)
        if measured == self._row_content_width:
            return False
        self._row_content_width = measured
        scroll_parent = self._nearest_scroll_parent()
        parent_scroll_y = getattr(scroll_parent, "scroll_y", None)
        restore_scroll_y = (
            int(parent_scroll_y) if parent_scroll_y is not None else None
        )
        self.refresh(recompose=True)
        if self.is_mounted:
            self._schedule_recomposed_content_fit(
                restore_scroll_y=restore_scroll_y
            )
        return True
```

In `_fit_height_to_content`, immediately after the existing `region is None or region.height <= 0` early return, add:

```python
        if self._maybe_relabel_for_width():
            return
```

- [ ] **Step 4: Run the file's full suite (not just the new tests)**

```bash
.venv/bin/pytest Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: all pass — including the pre-existing scroll-stability test
(`test_console_workspace_sync_while_scrolled_keeps_scroll_range_stable`),
which now also exercises the relabel path.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Widgets/Console/console_workspace_context.py \
  Tests/UI/test_console_workspace_context_rail.py
git commit -m "feat(console): rewrap rail row labels when measured width changes

Guarded relabel in the fit pass (mount, resize, and frame-variant width
changes all funnel through it); equality guard prevents recompose loops.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: CSS — scrollbar gutter and auto-height row lines (plus bundle rebuild)

**Files:**
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:2149-2153` (row-line block) and `:2176` (`#console-left-rail-body` block)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` (script only — never by hand)
- Test: `Tests/UI/test_console_workspace_context_rail.py:553` (`test_console_workspace_context_grouped_browser_styles_are_declared`)

**Interfaces:**
- Consumes: nothing from other tasks (independent of Tasks 2-4, but ordered here so the style test names the final selectors).
- Produces: the two CSS declarations the spec mandates, present in both the component file and the regenerated bundle.

- [ ] **Step 1: Extend the style-declaration test (it checks BOTH css files)**

In `test_console_workspace_context_grouped_browser_styles_are_declared`, after the existing `list_block` assertions inside the same `for css_path ...` loop, add:

```python
        # Row lines must size to their explicitly-heighted buttons; Textual's
        # Horizontal defaults to `height: 1fr`, which divides the list height
        # equally and breaks mixed wrapped/badge row heights.
        row_line_block = css.split(
            ".console-conversation-browser-row-line {", 1
        )[1].split("}", 1)[0]
        assert "height: auto" in row_line_block
        # Reserve the scrollbar cell permanently so row-wrap width does not
        # depend on scroll state (scrollbar toggle <-> rewrap feedback loop).
        rail_body_block = css.split("#console-left-rail-body {", 1)[1].split(
            "}", 1
        )[0]
        assert "scrollbar-gutter: stable" in rail_body_block
```

- [ ] **Step 2: Run it to verify it fails**

```bash
.venv/bin/pytest Tests/UI/test_console_workspace_context_rail.py -q -k "styles_are_declared"
```

Expected: FAIL — `AssertionError` on `height: auto`.

- [ ] **Step 3: Edit the component CSS**

In `tldw_chatbook/css/components/_agentic_terminal.tcss`, change the row-line block (~line 2149):

```tcss
.console-conversation-browser-row-line {
    width: 100%;
    min-width: 0;
    height: auto;
    layout: horizontal;
}
```

And add to the `#console-left-rail-body` block (~line 2176), alongside its existing declarations:

```tcss
    scrollbar-gutter: stable;
```

- [ ] **Step 4: Regenerate the bundle**

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
git diff --stat tldw_chatbook/css/
grep -c "scrollbar-gutter: stable" tldw_chatbook/css/tldw_cli_modular.tcss
```

Expected: diff touches only `_agentic_terminal.tcss` and `tldw_cli_modular.tcss`; grep prints ≥ 1. If the bundle diff contains unrelated churn, STOP and reconcile (see memory: the bundle briefly carried styles that existed nowhere else; PR #723 made it reproducible — unexpected churn means that assumption broke again).

- [ ] **Step 5: Run the style test and the row suites**

```bash
.venv/bin/pytest Tests/UI/test_console_workspace_context_rail.py -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/css/components/_agentic_terminal.tcss \
  tldw_chatbook/css/tldw_cli_modular.tcss \
  Tests/UI/test_console_workspace_context_rail.py
git commit -m "fix(css): auto-height rail row lines + stable scrollbar gutter

height:auto fixes latent 1fr equal-division of the conversation list;
scrollbar-gutter:stable decouples row-wrap width from scroll state.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: Row-height constant comment, full sweep, live verification

**Files:**
- Modify: `tldw_chatbook/Workspaces/display_state.py:56`
- Modify: backlog task file (AC checkboxes + implementation notes)

**Interfaces:**
- Consumes: everything prior. No new interfaces.

- [ ] **Step 1: Document the nominal row height**

In `display_state.py`, replace the bare constant at line 56 with:

```python
# Nominal (minimum) rail conversation-row height: one name line + the
# metadata line + the row's bottom margin. Rows whose names wrap to two
# lines render one line taller. This constant intentionally stays at the
# minimum: it only feeds the visible-row-count heuristic below, where a
# slight overestimate merely loads a row or two more than fits (the list
# scrolls); it must NOT be used for layout math (the tray derives real
# heights from the wrap result -- see console_workspace_context.py).
CONSOLE_WORKSPACE_CONVERSATION_ROW_HEIGHT = 3
```

- [ ] **Step 2: Full UI + Chat sweep**

```bash
.venv/bin/pytest Tests/UI Tests/Chat -q 2>&1 | tail -15
```

Expected: failures limited to the known dev-tip baseline (~12 `scheduling` + 2 shell/snapshot). Any failure in a file this branch touched must be fixed before proceeding.

- [ ] **Step 3: Live verification (REQUIRED — invoke the `verify` skill)**

Use the project `verify` skill (tmux recipe) to drive the real TUI. Checklist, from the spec:

1. Wide terminal (~200 cols): Console rail shows long conversation names flush left, wrapped to 2 lines, metadata line beneath, star column aligned.
2. Narrow terminal (~100 cols): same rows re-wrapped tighter; no clipping, no overlap with the star column.
3. Resize across a budget boundary: labels rewrap once; no flicker loop (watch for continuous recompose — the rail should go quiet within a beat).
4. Rail populated enough to scroll: scrollbar visible, no oscillation (the `scrollbar-gutter: stable` guard); scroll position survives a `sync_state` (send a message so the rail refreshes while scrolled).
5. Selected row: background highlight reads clearly with no marker glyph.
6. A conversation with sub-agent history: badge on its own line below the metadata line.

Capture at least one wide and one narrow screenshot for the PR.

- [ ] **Step 4: Close out the backlog task and commit**

Check off all ACs in the backlog task file, add Implementation Notes (approach, files touched, the two CSS guards and why), set status Done via `backlog task edit <id> -s Done`, then:

```bash
git add tldw_chatbook/Workspaces/display_state.py backlog
git commit -m "docs: nominal rail row-height comment + close out row-wrap task

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Finish the branch**

Invoke `superpowers:finishing-a-development-branch`. PR targets `dev`. PR body includes the wide/narrow screenshots and links the spec. Remember: CI checks are intentionally cancelled by another build — verify locally, don't block on CI.

---

## Self-Review Notes (already applied)

- Spec coverage: marker removal + flush-left (Task 3), width-aware wrap + ellipsis + cell measurement (Task 2), metadata truncation raw-then-escape (Tasks 2/3), heights from the same wrap result (Task 3), fit-pass relabel choke point + guard (Task 4), `scrollbar-gutter: stable` + row-line `height: auto` (Task 5), row-height constant note (Task 6), tooltip lstrip simplification + `GLYPH_ACTIVE` removal (Task 3), live verification (Task 6). No gaps found.
- The old `_conversation_visible_title`/`_MAX_CONVERSATION_ROW_TITLE` tests are rewritten in the same task that removes them (Task 3) so every commit is green.
- Type consistency: `name_line_count` (int, default 1) and `budget` (int) names match across Tasks 3-4; wrap helper returns `tuple[str, ...]` and every consumer takes `len(...)` of it.
