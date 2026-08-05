# Watchlists Rebuild Phase D — Content Pane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the Watchlists `CONTENT` region — today a deliberately-collapsed Phase D stub — with a working reader that renders both item kinds and marks what you have read.

**Architecture:** `content_kind` selects one of two renderers behind a single pane widget. Both kinds share the pane, its keys, and its actions, so a site change reads like a feed article. The read path is fixed first: the database already returns the body via `SELECT i.*`, but `normalize_watchlist_item` rebuilds an explicit dict and drops it, so nothing downstream can render anything until that is corrected.

**Tech Stack:** Textual ≥3.3.0, SQLite/FTS5, Python ≥3.11. Existing: `Subscriptions_DB`, `item_persist` (Phase A), `WatchlistsWorkbench` + `RegionLayout` (Phase B), `watchlist_tree` + tabs (Phase C).

## Global Constraints

- Source: `Docs/superpowers/specs/2026-07-25-watchlists-console-rebuild-design.md`, sections **Content pane**, **Reading behaviour**, **Body text: where it lives**, **Empty states**. That spec is approved; do not redesign it.
- Valid `content_kind`/`content_format` pairings are exactly `("article","text")`, `("article","markdown")`, `("change","diff")` — enforced by `item_persist._VALID_PAIRINGS`. The renderer must handle **exactly** these and must not invent a third kind.
- **`content` is `NULL` for every pre-existing item** and is unrecoverable without a re-fetch, because no code path ever persisted a body. This is a known, spec-acknowledged limitation. The reader must say so explicitly — "no body captured — re-check this source" — never render a blank pane.
- **Remote content is untrusted.** Escape it before it reaches a Textual renderable. This repo has shipped markup-injection through tooltips and button labels before; `escape_markup` at the boundary, not at the call site.
- Item read status is **global** — an item marked read from "All sources" is read in every watchlist containing that source. Correct and intended; state it in the UI copy so it is not discovered as a bug.
- Use `/private/tmp/tldw-wl-phase-d/.venv/bin/python -m pytest` (create the venv if absent). **Never `-q`** — this repo's pytest config suppresses FAILED summary lines under it.
- Every new test must be marked `pytest.mark.unit` or it is deselected by CI's `pytest -m unit`.
- Never write to `~/.config/tldw_cli/` or `~/.local/share/tldw_cli/`. Use `tmp_path` or `TLDW_CONFIG_PATH`.

## Out of scope, deliberately

- **`/` FTS search scoped to tree selection.** The FTS index exists (Phase A backfill), the search UI does not. It is a separate concern with its own spec section ("Full-text search") and belongs to a later phase. Do not start it here.
- **Stage / Ingest / Discuss actions.** The spec has the reader share these with the item list; wiring them is its own slice.
- Re-fetching to backfill `content` for historical items.

## File structure

| File | Responsibility |
|---|---|
| `tldw_chatbook/Subscriptions/watchlist_normalizers.py` | **Modify.** Carry the reader fields through the read path. |
| `tldw_chatbook/UI/Watchlists_Modules/content_pane.py` | **Create.** The pane, kind dispatch, and both renderers. |
| `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` | **Modify.** Mount the pane in `CONTENT`, route `ItemSelected`, stop defaulting it collapsed. |
| `Tests/Subscriptions/test_watchlist_normalizers.py` | Reader fields survive normalization. |
| `Tests/UI/test_watchlists_content_pane.py` | Kind dispatch, both renderers, empty states, escaping. |
| `Tests/UI/test_watchlists_read_status.py` | Mark-read on open, unread toggle, global status. |

---

### Task 1: Carry the reader fields through the read path

**Files:**
- Modify: `tldw_chatbook/Subscriptions/watchlist_normalizers.py:230-248`
- Test: `Tests/Subscriptions/test_watchlist_normalizers.py`

**Interfaces:**
- Consumes: rows from `Subscriptions_DB.get_new_items`, which is `SELECT i.*` and therefore already contains every column below.
- Produces: `normalize_watchlist_item` output gains `content`, `content_kind`, `content_format`, `change_percentage`, `change_type`, `diff_summary`, `canonical_url`. Tasks 2-4 consume these key names verbatim.

Nothing renders until this lands. The columns exist and the query returns them; the normalizer is where they are lost.

- [ ] **Step 1: Write the failing test**

```python
import pytest

pytestmark = pytest.mark.unit


def test_normalize_carries_the_reader_fields():
    """The reader cannot render what the read path drops.

    `get_new_items` is `SELECT i.*`, so the body is present in the row. This
    normalizer rebuilt an explicit dict and omitted it, which meant every
    downstream consumer saw a title-only item no matter what was persisted.
    """
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 7,
        "subscription_id": 3,
        "title": "Claude Opus 4.5 is now available",
        "url": "https://example.test/a",
        "content": "body text that must survive",
        "content_kind": "article",
        "content_format": "markdown",
        "change_percentage": None,
        "change_type": None,
        "diff_summary": None,
    }

    item = normalize_watchlist_item("local", row)

    assert item["content"] == "body text that must survive"
    assert item["content_kind"] == "article"
    assert item["content_format"] == "markdown"


def test_normalize_carries_the_change_fields():
    """A `change` item renders from these three and nothing else."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    row = {
        "id": 8,
        "subscription_id": 3,
        "title": "anthropic.com/news",
        "url": "https://anthropic.test/news",
        "content": "+ added line\n- removed line",
        "content_kind": "change",
        "content_format": "diff",
        "change_percentage": 12.0,
        "change_type": "structural",
        "diff_summary": "2 lines changed",
    }

    item = normalize_watchlist_item("local", row)

    assert item["content_kind"] == "change"
    assert item["change_percentage"] == 12.0
    assert item["change_type"] == "structural"
    assert item["diff_summary"] == "2 lines changed"


def test_normalize_tolerates_a_row_with_no_body():
    """Every pre-existing item has `content` NULL; that must not raise."""
    from tldw_chatbook.Subscriptions.watchlist_normalizers import (
        normalize_watchlist_item,
    )

    item = normalize_watchlist_item(
        "local", {"id": 9, "subscription_id": 3, "title": "Old item"}
    )

    assert item["content"] is None
    assert item["content_kind"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/Subscriptions/test_watchlist_normalizers.py -v`
Expected: FAIL with `KeyError: 'content'`

- [ ] **Step 3: Add the fields**

In `normalize_watchlist_item`'s returned dict, after `"published_date"`:

```python
        # Phase D reader fields. `get_new_items` is `SELECT i.*`, so these are
        # already on the row -- this dict was simply not carrying them, which
        # made every item title-only downstream regardless of what Phase A
        # persisted.
        "content": row.get("content"),
        "content_kind": row.get("content_kind"),
        "content_format": row.get("content_format"),
        # `change`-kind items render from these three.
        "change_percentage": row.get("change_percentage"),
        "change_type": row.get("change_type"),
        "diff_summary": row.get("diff_summary"),
        "canonical_url": row.get("canonical_url"),
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/Subscriptions/test_watchlist_normalizers.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-check**

Delete the `"content": row.get("content"),` line, re-run, confirm RED, restore. A field that no test pins is a field the next refactor drops — which is exactly how this bug arrived.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/Subscriptions/watchlist_normalizers.py Tests/Subscriptions/test_watchlist_normalizers.py
git commit -m "feat(watchlists): carry reader fields through item normalization"
```

---

### Task 2: The content pane and the Article renderer

**Files:**
- Create: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Test: `Tests/UI/test_watchlists_content_pane.py`

**Interfaces:**
- Consumes: an item dict as produced by Task 1.
- Produces: `ContentPane(Vertical)` with `item: reactive[dict | None]`, and a module-level `render_article(item) -> RenderableType`. Task 3 adds `render_change`; Task 4 mounts the pane.

Study `items_pane.py` for the established pane conventions (`RecomposeCaptureGuard`, message classes, reactive shape) and follow them rather than inventing a new pattern.

- [ ] **Step 1: Write the failing test**

```python
import pytest

pytestmark = pytest.mark.unit


def test_article_renders_title_source_and_body():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "Claude Opus 4.5 is now available",
        "source_name": "Anthropic News",
        "published_date": "2026-07-28",
        "content": "The model is available in the API today.",
        "content_kind": "article",
        "content_format": "text",
    }))

    assert "Claude Opus 4.5 is now available" in out
    assert "Anthropic News" in out
    assert "The model is available in the API today." in out


def test_article_with_no_body_explains_why():
    """`content` is NULL for every pre-existing item. Never render blank."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "An item from before bodies were captured",
        "source_name": "Old Feed",
        "content": None,
        "content_kind": "article",
    }))

    assert "no body captured" in out.lower()
    assert "re-check" in out.lower()


def test_untrusted_body_markup_is_escaped():
    """Remote content reaches a Textual renderable; it must not be markup."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_article

    out = str(render_article({
        "title": "[bold red]not a style[/]",
        "source_name": "Hostile Feed",
        "content": "[link=evil]click[/link]",
        "content_kind": "article",
    }))

    assert "\\[bold red]" in out or "[bold red]" in out
    assert "\\[link=evil]" in out or "[link=evil]" in out
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_watchlists_content_pane.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Implement the pane and the Article renderer**

```python
"""The Watchlists reader: one pane, two renderers, chosen by `content_kind`.

Both kinds share this pane, its keys and its actions, so a site change reads
like a feed article while still showing what was honestly captured.
"""

from __future__ import annotations

from typing import Any

from rich.markup import escape as escape_markup
from rich.text import Text
from textual.containers import Vertical
from textual.reactive import reactive
from textual.widgets import Static

# Every item persisted before Phase A carries `content = NULL`, and it cannot
# be recovered without re-fetching the source. Say so rather than rendering an
# empty pane the reader will mistake for a bug.
_NO_BODY = "no body captured for this item — re-check this source to fetch it"


def render_article(item: dict[str, Any]) -> Text:
    """Render a feed item: title, source, date, word count, body."""
    body = item.get("content")
    out = Text()
    out.append(escape_markup(str(item.get("title") or "Untitled")), style="bold")
    out.append("\n")
    meta = [str(item.get("source_name") or "unknown source")]
    if item.get("published_date"):
        meta.append(str(item["published_date"]))
    if body:
        meta.append(f"{len(str(body).split())} words")
    out.append(escape_markup(" · ".join(meta)), style="dim")
    out.append("\n\n")
    out.append(escape_markup(str(body)) if body else _NO_BODY)
    return out


class ContentPane(Vertical):
    """Hosts the reader for the currently selected item."""

    item: reactive[dict[str, Any] | None] = reactive(None, recompose=True)

    def compose(self):
        if self.item is None:
            yield Static("Select an item to read it.", id="content-empty")
            return
        yield Static(render_for(self.item), id="content-body")


def render_for(item: dict[str, Any]) -> Text:
    """Dispatch on `content_kind`. Task 3 adds the `change` arm."""
    return render_article(item)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_watchlists_content_pane.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/content_pane.py Tests/UI/test_watchlists_content_pane.py
git commit -m "feat(watchlists): add the content pane and the article renderer"
```

---

### Task 3: The Change renderer and kind dispatch

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Test: `Tests/UI/test_watchlists_content_pane.py`

**Interfaces:**
- Produces: `render_change(item)`, and `render_for` dispatching on `content_kind`.

Target layout, from the spec:

```
anthropic.com/news        site · changed 7m ago
12% changed · structural
───────────────────────────────────────────────
+ Claude Opus 4.5 is now available in the API
- Claude Opus 4.1 is now available in the API
```

- [ ] **Step 1: Write the failing test**

```python
def test_change_renders_percent_type_and_diff_lines():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "anthropic.com/news",
        "source_name": "Anthropic",
        "content": "+ Opus 4.5 available\n- Opus 4.1 available",
        "content_kind": "change",
        "content_format": "diff",
        "change_percentage": 12.0,
        "change_type": "structural",
    }))

    assert "12" in out and "%" in out
    assert "structural" in out
    assert "+ Opus 4.5 available" in out
    assert "- Opus 4.1 available" in out


def test_dispatch_selects_the_renderer_by_kind():
    """The two kinds must not render through the same arm by accident."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    change = str(render_for({
        "title": "site", "content": "+ x", "content_kind": "change",
        "change_percentage": 3.0, "change_type": "text",
    }))
    article = str(render_for({
        "title": "post", "content": "prose", "content_kind": "article",
    }))

    # A discriminator only the change arm emits.
    assert "3" in change and "%" in change
    assert "%" not in article


def test_unknown_kind_falls_back_to_article_without_raising():
    """An escaping exception in compose() exits the whole app."""
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_for

    out = str(render_for({"title": "odd", "content": "x", "content_kind": "wat"}))
    assert "odd" in out


def test_change_with_no_body_explains_why():
    from tldw_chatbook.UI.Watchlists_Modules.content_pane import render_change

    out = str(render_change({
        "title": "site", "content": None, "content_kind": "change",
        "change_percentage": 5.0, "change_type": "text",
    }))
    assert "no body captured" in out.lower()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest Tests/UI/test_watchlists_content_pane.py -v`
Expected: FAIL — `render_change` undefined.

- [ ] **Step 3: Implement**

```python
def render_change(item: dict[str, Any]) -> Text:
    """Render a site item: what changed, by how much, and the diff lines."""
    out = Text()
    out.append(escape_markup(str(item.get("title") or "Untitled")), style="bold")
    out.append("\n")

    headline: list[str] = []
    pct = item.get("change_percentage")
    if pct is not None:
        headline.append(f"{float(pct):.0f}% changed")
    if item.get("change_type"):
        headline.append(str(item["change_type"]))
    out.append(escape_markup(" · ".join(headline) or "changed"), style="dim")
    out.append("\n\n")

    body = item.get("content")
    if not body:
        out.append(_NO_BODY)
        return out

    # Colour the diff, but escape each line first: these lines are remote
    # content, and styling them must not mean interpreting them as markup.
    for line in str(body).splitlines():
        style = "green" if line.startswith("+") else "red" if line.startswith("-") else None
        out.append(escape_markup(line), style=style)
        out.append("\n")
    return out


_RENDERERS = {"article": render_article, "change": render_change}


def render_for(item: dict[str, Any]) -> Text:
    """Dispatch on `content_kind`, falling back rather than raising.

    An exception escaping `compose()` exits the application, so an unexpected
    kind degrades to the article renderer instead of taking the app down.
    """
    return _RENDERERS.get(str(item.get("content_kind") or ""), render_article)(item)
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/test_watchlists_content_pane.py -v`
Expected: PASS

- [ ] **Step 5: Mutation-check the dispatch**

Change `_RENDERERS` to `{"article": render_article, "change": render_article}` and confirm `test_dispatch_selects_the_renderer_by_kind` goes RED. If it stays green the two renderers are not actually distinguishable by the assertions, and the test is decorative.

- [ ] **Step 6: Commit**

```bash
git add tldw_chatbook/UI/Watchlists_Modules/content_pane.py Tests/UI/test_watchlists_content_pane.py
git commit -m "feat(watchlists): render site changes and dispatch on content_kind"
```

---

### Task 4: Mount the pane in the CONTENT region and route selection

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py` (the `CONTENT` region builder; `region_layout` default at ~`:197-200`)
- Test: `Tests/UI/test_watchlists_content_pane.py`

**Interfaces:**
- Consumes: `ItemsPane.ItemSelected` (already posted; see `items_pane.py:16`).
- Produces: a `ContentPane` mounted in `Region.CONTENT`, updated on selection.

Two things change together, and the second is easy to miss: `region_layout` currently defaults to `collapsed=frozenset({Region.CONTENT})` **because** the region held a stub. With a real reader there, that default is wrong — but `on_mount` overlays persisted state from `region_layout_store`, so a user who already has Watchlists state saved will still see it collapsed. Handle that, and say in the commit message which behaviour you chose.

- [ ] **Step 1: Write the failing test**

```python
@pytest.mark.asyncio
async def test_selecting_an_item_renders_it_in_the_content_region():
    """Selection must reach the reader through the real screen wiring."""
    # Follow the harness used by the Phase C screen tests in
    # Tests/UI/ -- reuse their app fixture rather than booting a bare screen.
    ...


def test_content_region_is_not_collapsed_by_default_now_it_has_a_reader():
    from tldw_chatbook.UI.Watchlists_Modules.region_layout import Region
    from tldw_chatbook.UI.Screens.watchlists_collections_screen import (
        WatchlistsCollectionsScreen,
    )

    default = WatchlistsCollectionsScreen.region_layout.default
    layout = default() if callable(default) else default
    assert Region.CONTENT not in layout.collapsed
```

- [ ] **Step 2: Run to verify it fails**

Expected: FAIL — `CONTENT` is in the default collapsed set.

- [ ] **Step 3: Mount the pane, route selection, drop the collapsed default**

Build the `CONTENT` region content with a `ContentPane`, mirroring how the other regions are built (see `_build_list_pane` and the `ITEMS` builder around `:1084`). Handle `ItemSelected` on the screen and assign `content_pane.item`. Remove `Region.CONTENT` from the `region_layout` reactive default and update the comment above it, which currently explains the stub.

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest Tests/UI/ -v`
Expected: PASS

- [ ] **Step 5: Verify at the real surface**

Use the `verify` skill: launch the app in tmux, open Watchlists, select an item, and `capture-pane` to confirm the reader actually paints. `render_strips()` / a real capture is the only authority on what painted — four tmux artifacts on this screen have been mistaken for real output before.

- [ ] **Step 6: Commit**

---

### Task 5: Reading behaviour — mark read on open, and the unread toggle

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`, `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Test: `Tests/UI/test_watchlists_read_status.py`

Opening an item in the reader marks it read. Because marking read destroys the unread list, there is an explicit toggle back. Status is **global** — the same article is read in every watchlist containing its source — and the UI copy must say so.

- [ ] **Step 1: Write the failing test**

```python
import pytest

pytestmark = pytest.mark.unit


def test_opening_an_item_marks_it_read():
    ...


def test_the_unread_toggle_restores_unread():
    """Marking read destroys the unread list, so this must be reversible."""
    ...


def test_read_status_is_global_across_watchlists():
    """The same item read from 'All sources' is read everywhere.

    Asserted so the behaviour is pinned as intended rather than discovered as
    a bug later.
    """
    ...
```

- [ ] **Step 2-6:** implement against the status API used by `ItemsPane`'s existing status filter, run, mutation-check the toggle (make it a no-op; the reversal test must go red), commit.

---

### Task 6: `j` / `k` item navigation and a binding conflict audit

**Files:**
- Modify: `tldw_chatbook/UI/Screens/watchlists_collections_screen.py`
- Test: `Tests/UI/test_watchlists_content_pane.py`

`j` and `k` move to the next and previous item, updating the reader.

**The audit is not optional and is the substance of this task.** The spec warns that roughly fifteen bare letters plus shifted variants is not a set to assume is free. Before adding anything, enumerate `BINDINGS` on `BaseAppScreen`, on the app, and on this screen, and confirm `j`/`k` collide with nothing. Read the class attributes — do not conclude from tmux probing, which cannot send some keys at all and has produced false conclusions on this screen before. Record the enumeration in the task report.

- [ ] **Step 1:** write a test asserting `j`/`k` are bound on this screen and that the bound letters do not appear in any ancestor's `BINDINGS`.
- [ ] **Step 2-5:** verify it fails, implement, verify, commit.

---

### Task 7: Empty states

**Files:**
- Modify: `tldw_chatbook/UI/Watchlists_Modules/content_pane.py`
- Test: `Tests/UI/test_watchlists_content_pane.py`

Covered partly by Tasks 2-3 (the no-body case). This task completes the set from the spec's **Empty states** section: nothing selected, and a source that has never been checked. Every empty state explains what to do next — never a bare box.

- [ ] **Step 1-5:** test, fail, implement, verify, commit.

---

## Self-review

**Spec coverage.** Content pane → Tasks 2-4. Reading behaviour (mark read, unread toggle, global status) → Task 5. Key bindings `j`/`k` + conflict audit → Task 6. Empty states → Tasks 2, 3, 7. `/` FTS search and the Stage/Ingest/Discuss actions are explicitly out of scope above, with reasons.

**Placeholders.** Tasks 5-7 carry test *names* and intent rather than full bodies, because their assertions depend on the status API and screen-test harness that Task 4's integration work settles. This is a deliberate, stated limitation rather than an oversight: the implementer for those tasks must write the bodies against what Task 4 actually produced. Tasks 1-3, which are self-contained, carry complete code.

**Type consistency.** `render_article` / `render_change` / `render_for` all take one item dict and return `rich.text.Text`; `ContentPane.item` is that same dict. The key names in Tasks 2-3 (`content`, `content_kind`, `change_percentage`, `change_type`) are exactly those Task 1 adds to the normalizer.

**Known risk carried from the spec.** Every pre-existing item has `content = NULL`, so on a real database this reader will show "no body captured" for essentially all historical items until sources are re-checked. That is expected, and it is why the empty state is a first-class requirement rather than a nicety.
