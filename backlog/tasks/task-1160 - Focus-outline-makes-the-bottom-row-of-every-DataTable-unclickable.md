---
id: TASK-1160
title: >-
  The global focus outline makes the bottom row of every DataTable unclickable
status: Done
assignee: []
created_date: '2026-07-28 12:00'
labels:
  - bug
  - ui
  - css
  - a11y
priority: high
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`*:focus { outline: solid $ds-focus-accent }` in `css/core/_reset.tcss` paints Textual's focus outline **over** a widget's outermost lines. Those segments lose the `{"row", "column"}` meta that `DataTable._on_click` reads to resolve which cell was hit — so a click on the bottom row of any focused table does nothing.

Isolated in a bare Textual app during the TASK-1105 fix: a six-row table with the outline leaves the cursor at row 0 when the last row is clicked, and lands correctly on row 5 without it. It is masked on first interaction because `MouseDown` focuses the table before the `Click` is resolved, so the very first click still works — which is why it reads as intermittent rather than broken.

TASK-1105 fixed this for the Watchlists tables. **Every other `DataTable` in the app still has it**, and the app has many.

The fix likely belongs in `components/_lists.tcss` — moving the focus affordance off the outermost line so it stops consuming the click target. That relocates a focus indicator app-wide, which is why it was not folded into 1105: it wants its own regression pass across every screen that shows a table.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The bottom row of a focused `DataTable` is clickable
- [x] #2 Focused widgets still show a visible focus affordance
- [x] #3 A test clicks the last row of a focused table and asserts the cursor moved, proven to fail against current code
- [x] #4 Screens with tables outside Watchlists are checked, and the ones affected listed here
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**The premise holds. Re-confirmed from scratch before changing anything**, in a
bare Textual 8.2.7 app with no `tldw_chatbook` import, no config and no DB — the
outline rule the only variable between the two runs. Six-row table, region
height 7:

```
--- no outline ---            --- *:focus { outline: solid red } ---
click y=5 -> row 4            click y=5 -> row 4
click y=6 -> row 5            click y=6 -> row 0     <- bottom line dead
```

It is the whole perimeter, not only the last row. Probing x as well as y (two
columns, cursor parked at r0c1 before each click, `->` = where it landed):

```
                    x=0     x=1..7   x=8..13
no outline  y=1     r0c0    r0c0     r0c1
            y=3     r2c0    r2c0     r2c1
            y=6     r5c0    r5c0     r5c1        (every cell reachable)
outline     y=1     dead    r0c0     r0c1        (x=0 gone)
            y=3     dead    r2c0     r2c1
            y=6     dead    dead     dead        (entire bottom line gone)
```

`outline` costs no geometry — `Region(height=7)` in both runs — so it does not
push content aside, it paints over the table's own outermost rendered lines.
Those segments lose the `{"row", "column"}` Rich style metadata, and
`DataTable._on_click` resolves the clicked cell from nothing else:

```python
meta = event.style.meta
if "row" not in meta or "column" not in meta:
    return
```

Three consequences, all app-wide: the bottom line (the last row of any table
whose content reaches its region edge), column x=0 on every row, and the top
line — which for a `DataTable` is the column header, overwritten outright.
That last one is TASK-1034's "focused results grid loses its header", the same
rule biting a different edge.

**Fix — one type selector, in `css/components/_lists.tcss`.**

```
DataTable:focus                        { outline: none; }
DataTable:focus > .datatable--cursor   { background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline; }
DataTable:focus > .datatable--header   { background: $ds-focus-bg; color: $ds-focus-fg; text-style: bold underline; }
```

A bare type selector, so no call site opts in; `DataTable:focus` is (0,1,1)
against the reset's (0,1,0) and Textual type selectors match subclasses.
AC#2 is met by recolouring cells the table already draws, which is why they
cost no segment and no meta: the cursor cell and the column header both take
the sanctioned `$ds-focus-bg`/`$ds-focus-fg` pair. Two cues rather than one
because the cursor can be switched off (`cursor_type="none"`) or scrolled out
of view while the header is always on screen. Textual's own
`DataTable:focus { background-tint: $foreground 5%; }` still layers on top —
that is why the live header measures rgb(88,109,130) and the cursor row
rgb(81,103,126) (`$ds-focus-bg` = `#51677e`).

`core/_reset.tcss`'s own comment now records that "non-obscuring" is an intent
rather than a guarantee, and points here.

**Both screen-local workarounds removed, not left alongside.** TASK-1105's
`.watchlists-region DataTable:focus` and TASK-1034's `#evals-grid-table:focus`
were strict subsets of the rules above (1105 = outline + cursor, 1034 =
outline + header), so both were deleted from their feature modules and
replaced with a comment pointing at the shared rule. There is one mechanism.

**AC#4 — every `DataTable` outside Watchlists.** 41 files reference
`DataTable`; 30 actually construct one once the `tldw_api` schema modules
(unrelated `DataTable` name) are dropped, and 6 of those are Watchlists. That
leaves these 24, all of which carried the defect unless noted:

| File | Table(s) | Judgement |
|---|---|---|
| `UI/MCP_Modules/mcp_permissions_mode.py` | `#mcp-perm-table` | **Affected — reproduced live, before and after** |
| `UI/MCP_Modules/mcp_tools_mode.py` | `#mcp-tools-table` | **Affected — verified fixed live** |
| `UI/MCP_Modules/mcp_servers_mode.py` | `#mcp-servers-table` | Affected (`height: auto`) |
| `UI/MCP_Modules/mcp_audit_mode.py` | `#mcp-audit-table`, findings | Affected (`height: auto`) |
| `UI/Evals/results_grid.py` | `#evals-grid-table` | Affected; header half already fixed by 1034, the click half was not |
| `UI/Screens/scheduling/schedules_workbench.py` | schedule queue | Affected |
| `UI/Screens/scheduling/conflicts_tab.py` | conflicts | Affected |
| `UI/ChatbookExportManagementWindow.py` | 2 tables | Affected (reached via `Screens/chatbooks_screen.py`) |
| `UI/Views/RAGSearch/search_rag_window.py` | `#search-history-table` | Affected. Its `#index-stats-table` sibling never holds more than one row, so nothing was reachable to lose |
| `UI/Voice_Cloning_Window.py` | `#profile-table` | Affected |
| `Widgets/Persona_Widgets/personas_dictionary_detail.py` | entries, versions, attachments | Affected |
| `Widgets/Persona_Widgets/personas_lore_detail.py` | entries, attachments | Affected |
| `Widgets/Persona_Widgets/personas_character_editor_widget.py` | greetings | Affected |
| `Widgets/Persona_Widgets/personas_character_dictionaries.py` | attached dictionaries | Affected; display-only (no row handlers), so the cost was the header and x=0 |
| `Widgets/Persona_Widgets/personas_character_world_books.py` | attached world books | Same as above |
| `Widgets/CCP_Widgets/ccp_dictionary_editor_widget.py` | dictionary entries | Affected |
| `Widgets/chunk_preview_modal.py` | chunk list | Affected (reached from `media_details_widget.py`) |
| `Widgets/TTS/chapter_editor_widget.py` | chapters | Affected (reached from `STTS_Window.py`) |
| `Widgets/TTS/character_voice_widget.py` | voice assignments | Affected (reached from `STTS_Window.py`) |
| `UI/SiteConfigSettings.py` | `#site-list-table` | Would be affected, but **unreachable** — zero importers |
| `Widgets/voice_command_dialog.py` | `#command-table` | Same — zero importers |
| `Widgets/file_extraction_dialog.py` | extracted files | Same — zero importers |
| `Widgets/transcription_history_viewer.py` | `#history-table` | Same — zero importers |
| `css/Themes/theme_tester.py` | preview table | Developer tool, not shipped UI |

One honest qualifier on "affected": the *click* loss bites hardest on tables
whose content reaches their region edge — every MCP table is `height: auto`
(`components/_agentic_terminal.tcss`, "hugs its own row count"), so their last
row sat exactly on the outline. A `1fr` table with spare room below its rows
loses only its header line and its x=0 column until it fills or is scrolled to
the end; then it loses the last row too. All of them lost the header on focus.

**Live captures** — real app, scratch `TLDW_CONFIG_PATH` profile
(`users_name = "verify_1160"`, deleted afterwards), 235x52, click columns
computed by character (`line.find(label) + 1`), never `awk index()`.

MCP ▸ Permissions, 15-row matrix, **before the fix** (bundle rebuilt from the
stashed sources at app boot):

```
click line 16 "chat_with_llm"        (table blurred)  -> cursor moves       [MouseDown focuses first: masked]
click line 27 "get_current_datetime" (BOTTOM row)     -> NO VISUAL CHANGE
click line 26 "calculator"           (2nd-to-last)    -> cursor moves       [control: clicks do work]
click line 27 "get_current_datetime" (BOTTOM row)     -> NO VISUAL CHANGE   [retried]
```

Same table, same clicks, **after the fix**:

```
click line 16 "chat_with_llm"        -> cursor row = line 16
click line 27 "get_current_datetime" -> cursor row = line 27   <- bottom row, table already focused
```

MCP ▸ Tools, 10-row catalog (second non-Watchlists table), after the fix:

```
click line 15 "create_note"  -> cursor row = line 15
click line 22 "search_rag"   -> cursor row = line 22   <- bottom row, table already focused
```

Focus affordance confirmed on the live render rather than from CSS: with the
table focused, header line bg = rgb(88,109,130) and cursor row bg =
rgb(81,103,126), against rgb(30,30,30)/rgb(18,18,18) for the table body; with
the table blurred, no line carries either.

**Tests** — `Tests/UI/test_datatable_focus_outline_click.py`, a plain
`DataTable` with no id and no classes (so no screen-scoped rule can reach it)
under the **production bundle** as `CSS_PATH`. A bare `App` with no CSS cannot
reproduce this at all, since the outline is the cause. Four behavioural tests
plus a guard that fails if the `*:focus` fallback ever leaves the bundle, which
would otherwise let the other four pass vacuously. Proven red against the
pre-fix CSS (`git stash push -- tldw_chatbook/css`):

```
FAILED test_clicking_the_last_row_of_a_focused_table_moves_the_cursor
        - clicking the bottom line (y=6) ... left the cursor on row 0; assert 0 == 5
FAILED test_every_row_of_a_focused_table_is_clickable
        - landed on [0, 0, 0, 0, 0, 0], expected [0, 1, 2, 3, 4, 5]
FAILED test_a_focused_table_still_shows_a_visible_focus_affordance
        - focused cursor identical to blurred (Color(30, 30, 30))
FAILED test_a_focused_table_still_renders_its_column_header
        - header line rendered as '┌───...┐'
4 failed, 1 passed
```

**Suite runs.** `Tests/Watchlists` 167 passed, `Tests/Subscriptions` 109
passed, the new file 5 passed. `Tests/UI` as a whole: **5529 passed, 187
failed, 20 skipped** in 1:00:25 (`test_chat_shell_bar.py` additionally fails
to *collect* on this branch — `ImportError: cannot import name 'TabState'` —
and was `--ignore`d).

Those 187 are the branch's baseline, and that is measured rather than
asserted. My change is CSS plus one new test file, so the only tests it can
reach are ones that read a `.tcss` file or drive a `DataTable`: 69 files,
2891 tests. Running exactly those on HEAD and again on HEAD~1 (`origin/dev`,
in a separate worktree so nothing was stashed under a live run):

```
HEAD  (with the fix)   73 failed, 2818 passed, 1 skipped   33:24
HEAD~1 (origin/dev)    73 failed, 2813 passed, 1 skipped   33:34
diff of the two failure sets: empty
```

Identical failures on both sides; the +5 passes are exactly this task's new
file. Nothing here regressed. The nine `test_non_obscuring_focus_contract.py`
failures in that set are typical of the rest — they demand
`.preset-button`/`.sidebar-resize-button` blocks that `layout/_sidebars.tcss`
does not contain and a `features/_chat_tabs.tcss` that was deleted upstream,
none of which this task touches.

**Known gap, out of scope, worth a follow-up.** In the live MCP run the cursor
moves onto the clicked row but the Inspector still reads "Select an item to
inspect" — the MCP panes act on `RowSelected` (activation: Enter, or a second
click) and not on `RowHighlighted`. That is the *other* half of TASK-1100/1105,
which was fixed for the Watchlists panes only. This task's change is the CSS
one; the event-handling gap remains on MCP and probably elsewhere.

**Files:** `css/components/_lists.tcss`, `css/core/_reset.tcss` (comment),
`css/features/_watchlists.tcss` (1105's rules removed), `css/features/_evals.tcss`
(1034's rules removed), regenerated `css/tldw_cli_modular.tcss`,
`Tests/UI/test_datatable_focus_outline_click.py` (new).
<!-- SECTION:NOTES:END -->
