# Retire the dead notes unsaved-indicator CSS/watchers — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax. Spec: `Docs/superpowers/specs/2026-07-12-notes-unsaved-indicator-removal-design.md`. Branch `claude/followups-unsaved-indicator` off dev `e7a3e60b`. Line numbers are exact at the branch point; grep the symbols if they drift.

**Goal:** Delete the fully-dead notes "unsaved / auto-save status" indicator machinery — two reactives, two watchers, and a CSS family — leaving no dead grep-live references.

**Architecture:** Pure dead-code removal. The `#notes-unsaved-indicator` widget is never composed, the `notes_unsaved_changes`/`notes_auto_save_status` reactives are never mutated by any live path (only self-referenced inside the two watchers), and the Library notes editor already has its own save/conflict/auto-sync UI. So nothing to reconnect — remove it all together and prove it with a grep-clean guard test.

**Tech Stack:** Python ≥3.11, Textual reactives/watchers, the `css/build_css.py` fragment concatenator, pytest.

## Global Constraints

- **Removal only** — no new features, no widget changes, no reconnect. Nothing ever set the reactives, so removing them changes no runtime behavior.
- **Surgical scope** — remove ONLY `notes_unsaved_changes`, `notes_auto_save_status`, `watch_notes_unsaved_changes`, `watch_notes_auto_save_status`, and the `.unsaved-indicator*` CSS. Do NOT touch the sibling reactives in the same blocks (`notes_sort_by`, `notes_preview_mode`, `notes_auto_save_enabled`, `notes_auto_save_timer`, `notes_last_save_time`) — `notes_auto_save_enabled`/`notes_auto_save_timer` are LIVE (set at `app.py:7695`/`7698-7700`). Do NOT touch `auto_save_delay_ms` config or `library_notes_canvas.py`.
- **CSS is generated** — `css/tldw_cli_modular.tcss` (the file the app loads via `CSS_PATH`) is built from `css/features/*.tcss` (and siblings) by `css/build_css.py`. Edit the source fragment `css/features/_notes.tcss`, then rebuild. A fresh build reproduces the committed modular file exactly except its `Generated:` timestamp line, so the modular diff is just the removed block + that timestamp.
- **AC:** no dead grep-live references to any of the six identifiers remain in `tldw_chatbook/` source (`*.py` + `*.tcss`).
- **Staging:** explicit paths only. Every commit ends with `Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>`.
- **Test command** (venv, isolated HOME):
  ```
  HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
    /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> \
    -q -p no:cacheprovider -o addopts="" --timeout=120
  ```

---

### Task 1: Remove the dead unsaved-indicator machinery

**Files:**
- Create: `Tests/UI/test_notes_unsaved_indicator_removed.py`
- Modify: `tldw_chatbook/app.py` (delete 2 reactives + 2 watchers)
- Modify: `tldw_chatbook/css/features/_notes.tcss` (delete the `.unsaved-indicator` block)
- Regenerate: `tldw_chatbook/css/tldw_cli_modular.tcss` (via `build_css.py`)
- Modify: `backlog/tasks/task-173 - Retire-or-reconnect-the-unreachable-notes-unsaved-indicator-CSS-watchers.md`

- [ ] **Step 1: Write the failing guard test (+ boot smoke)**

Create `Tests/UI/test_notes_unsaved_indicator_removed.py`:
```python
"""Task 173: the notes unsaved-indicator machinery (reactives + watchers + CSS)
was fully dead (widget never composed, reactives never mutated). These guard
against its presence / reintroduction and confirm the app still boots after
its removal."""
from pathlib import Path

import pytest

# Scope the scan to shipped source only — NOT this test file (which necessarily
# names the identifiers), Docs/, or backlog/.
_PKG = Path(__file__).resolve().parents[2] / "tldw_chatbook"
_DEAD_IDENTIFIERS = [
    "notes_unsaved_changes",
    "notes_auto_save_status",
    "watch_notes_unsaved_changes",
    "watch_notes_auto_save_status",
    "notes-unsaved-indicator",
    "unsaved-indicator",
]


def _source_files():
    for pattern in ("**/*.py", "**/*.tcss"):
        for path in _PKG.glob(pattern):
            if "__pycache__" in path.parts:
                continue
            yield path


@pytest.mark.parametrize("identifier", _DEAD_IDENTIFIERS)
def test_dead_unsaved_indicator_identifier_removed(identifier):
    """No dead unsaved-indicator reference remains in shipped tldw_chatbook source."""
    hits = [
        str(p.relative_to(_PKG.parent))
        for p in _source_files()
        if identifier in p.read_text(encoding="utf-8", errors="ignore")
    ]
    assert not hits, f"{identifier!r} still present in: {hits}"


@pytest.mark.asyncio
async def test_app_boots_after_removal():
    """The app mounts and the modular stylesheet loads with the CSS rules gone."""
    from tldw_chatbook.app import TldwCli
    app = TldwCli()
    async with app.run_test() as pilot:
        await pilot.pause()
```

- [ ] **Step 2: Run the guard test to verify it FAILS**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  "Tests/UI/test_notes_unsaved_indicator_removed.py::test_dead_unsaved_indicator_identifier_removed" \
  -q -p no:cacheprovider -o addopts="" --timeout=120
```
Expected: FAIL — every identifier is still present (in `app.py` and/or the two `.tcss` files). (The `test_app_boots_after_removal` case passes both before and after; it's a regression net, not RED-first.)

- [ ] **Step 3: Delete the two reactives in `app.py`**

Delete the line at `app.py:2280`:
```python
    notes_unsaved_changes: reactive[bool] = reactive(False)
```
and the line at `app.py:2289`:
```python
    notes_auto_save_status: reactive[str] = reactive("")  # Status: "", "saving", "saved"
```
Leave every other reactive in those two blocks untouched (`notes_sort_by`, `notes_sort_ascending`, `notes_preview_mode`, `notes_auto_save_enabled`, `notes_auto_save_timer`, `notes_last_save_time`).

- [ ] **Step 4: Delete the two watcher methods in `app.py`**

Delete both methods in full (contiguous, currently `app.py:6623-6666`), leaving a single blank line before the next method (`watch_conv_char_sidebar_left_collapsed`):
```python
    def watch_notes_unsaved_changes(self, has_unsaved: bool) -> None:
        """Update the unsaved changes indicator."""
        if not self._ui_ready:
            return
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            # Don't update if we're showing auto-save status
            if self.notes_auto_save_status:
                return
            if has_unsaved:
                indicator.update("● Unsaved")
                indicator.add_class("has-unsaved")
            else:
                indicator.update("")
                indicator.remove_class("has-unsaved")
        except QueryError:
            pass  # Indicator might not exist yet

    def watch_notes_auto_save_status(self, status: str) -> None:
        """Update the indicator based on auto-save status."""
        if not self._ui_ready:
            return
        try:
            indicator = self.query_one("#notes-unsaved-indicator", Label)
            if status == "saving":
                indicator.update("⟳ Auto-saving...")
                indicator.remove_class("has-unsaved")
                indicator.add_class("auto-saving")
            elif status == "saved":
                indicator.update("✓ Saved")
                indicator.remove_class("has-unsaved", "auto-saving")
                indicator.add_class("saved")
                # Clear the saved status after 2 seconds (keeping this one timer for UX feedback)
                self.set_timer(2.0, lambda: setattr(self, 'notes_auto_save_status', ''))
            else:
                # Empty status - let the unsaved changes watcher handle it
                indicator.remove_class("auto-saving", "saved")
                # Re-evaluate unsaved changes
                if self.notes_unsaved_changes:
                    indicator.update("● Unsaved")
                    indicator.add_class("has-unsaved")
                else:
                    indicator.update("")
                    indicator.remove_class("has-unsaved")
        except QueryError:
            pass  # Indicator might not exist yet
```
(If the surrounding whitespace differs slightly, delete from the `def watch_notes_unsaved_changes` line through the SECOND `except QueryError:\n            pass  # Indicator might not exist yet`, and normalize to one blank line between the preceding method and `watch_conv_char_sidebar_left_collapsed`.)

- [ ] **Step 5: Delete the CSS block in the source fragment**

In `css/features/_notes.tcss`, delete lines `15-40` — the "Auto-save status indicators" comment plus the four rules — leaving line 14 (blank) and line 42's `T169: .auto-save-label pruned` comment in place:
```
/* --- Auto-save status indicators ---
 * app.py's watch_notes_unsaved_changes/watch_notes_auto_save_status still
 * reference #notes-unsaved-indicator and these has-unsaved/auto-saving/
 * saved classes (both guarded by `except QueryError: pass`, since no
 * widget currently composes #notes-unsaved-indicator either) -- kept per
 * the grep-based live/dead adjudication rather than pruned, since Python
 * still names them.
 */
.unsaved-indicator {
    margin: 0 1;
    text-style: bold;
}

.unsaved-indicator.has-unsaved {
    color: $error;
}

.unsaved-indicator.auto-saving {
    color: $primary;
    text-style: italic;
}

.unsaved-indicator.saved {
    color: $success;
    text-style: bold;
}
```

- [ ] **Step 6: Regenerate the modular stylesheet**

Run:
```
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python tldw_chatbook/css/build_css.py
```
Then confirm the modular diff is only the removed block + the `Generated:` timestamp line (no unrelated churn):
```
git diff --stat tldw_chatbook/css/tldw_cli_modular.tcss
# added CONTENT lines (single leading +, not the +++ header) that are NOT the timestamp:
git diff tldw_chatbook/css/tldw_cli_modular.tcss | grep -E "^\+[^+]" | grep -v "Generated:"
```
Expected: the stat shows a modest deletion; the second command prints **nothing** (the rebuild only removed the `.unsaved-indicator` lines and rewrote the `Generated:` timestamp — it added no other content). The removed rules appear as `-` lines, which is correct.

- [ ] **Step 7: Run the tests to verify GREEN + import smoke**

Run:
```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_notes_unsaved_indicator_removed.py -q -p no:cacheprovider -o addopts="" --timeout=180
HOME=/private/tmp/tldw-chatbook-test-home PYTHONPATH=$(pwd) \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "import tldw_chatbook.app; print('import ok')"
```
Expected: all guard-test params PASS (identifiers gone), the boot smoke PASSES (app mounts, CSS loads), and `import ok`. If `test_app_boots_after_removal` cannot run in this environment (Textual harness limitation) rather than failing on the removal itself, report DONE_WITH_CONCERNS with the exact error — the import smoke + build-determinism check + grep guard still establish correctness.

- [ ] **Step 8: Mark backlog task 173 Done**

```bash
perl -0pi -e 's/- \[ \] #1/- [x] #1/' "backlog/tasks/task-173 - Retire-or-reconnect-the-unreachable-notes-unsaved-indicator-CSS-watchers.md"
perl -0pi -e 's/^status: .*/status: Done/mi' "backlog/tasks/task-173 - Retire-or-reconnect-the-unreachable-notes-unsaved-indicator-CSS-watchers.md"
```
Add a short `## Implementation Notes` section: adjudicated remove (nothing live to reconnect — widget never composed, reactives never mutated, Library editor already has its own save UI); deleted 2 reactives + 2 watchers + the `.unsaved-indicator` CSS; rebuilt the modular stylesheet; grep-guard + boot-smoke tests added. Confirm the change landed (`grep -n "status:\|\[x\]" "backlog/tasks/task-173 "*.md`).

- [ ] **Step 9: Commit**

```bash
git add Tests/UI/test_notes_unsaved_indicator_removed.py tldw_chatbook/app.py \
  tldw_chatbook/css/features/_notes.tcss tldw_chatbook/css/tldw_cli_modular.tcss \
  "backlog/tasks/task-173 - Retire-or-reconnect-the-unreachable-notes-unsaved-indicator-CSS-watchers.md"
git commit -m "refactor(notes): remove dead unsaved-indicator reactives, watchers, and CSS; task 173 done (173)

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Final gate (after Task 1)

```
HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share \
  /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/UI/test_notes_unsaved_indicator_removed.py -q -p no:cacheprovider -o addopts="" --timeout=180
```
Plus `import tldw_chatbook.app` and the `git diff` determinism check on `tldw_cli_modular.tcss`. Then the whole-branch review and finishing-a-development-branch.
