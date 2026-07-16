# MCP Hub Phase 4 — QA evidence (2026-07-16)

Branch: `claude/mcp-hub-phase4`, HEAD `1b106a46` ("fix(mcp-hub): guard
`MCPPermissionsMode.update_matrix` against duplicate row keys"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase4`.

Captured live from textual-serve (real app CSS, worktree code) in headless
bundled Chromium (Playwright, CDP-attached), viewport **2050×1240**. Same
methodology as
[`Docs/superpowers/qa/mcp-hub-phase3-2026-07/README.md`](../mcp-hub-phase3-2026-07/README.md):
DOM-rendered xterm (`.xterm-rows`, real text nodes — `Web_Server/serve.py`'s
own WebGL/Canvas-addon strip means no `window.__drv` patch is needed), plain
Playwright DOM text search + `Range`-based click-coordinate resolution,
route-abort non-localhost, a persistent headless Chromium launched with
`--remote-debugging-port` and attached to across steps via
`connect_over_cdp` from short `python3` snippets, two-click DataTable row
selection.

## This round's deltas from Phase 3

- Served on port **9191** (Phase 3 used 9189/9190); driver Chrome's own CDP
  port **9322** (a fresh headless instance, not reused across rounds).
- Isolated HOME **`/private/tmp/tldw-qa-mcp-hub-p4-20260716`** is a
  `cp -R` of Phase 3's evidence HOME
  (`/private/tmp/tldw-qa-mcp-hub-p3-20260714`, left untouched on disk),
  giving Permissions mode the same `docs-server` discovery snapshot Phase 3
  used for Tools mode: `search_docs` (renderable form schema), `read_file`
  (nested/raw schema), `list_files` (no schema at all — also renders raw).
  `profile_id` for that profile is `docs-server`, confirmed by reading
  `local_mcp_store.json` directly; per `hub_tool_catalog.py:83`
  (`server_key = f"local:{profile_id}"`), the server key is
  **`local:docs-server`**.
- New file this round: `mcp_permissions.json`
  (`tldw_chatbook/MCP/permission_store.py` schema, sibling of
  `local_mcp_store.json` per `unified_control_plane_service.py:2038`
  — `Path(store.path).with_name("mcp_permissions.json")`), seeded by a
  direct Python write (schema documented at the top of
  `permission_store.py`):
  - `global_default: "ask"`, `kill_switch: false`.
  - `local:docs-server.tools.search_docs` — explicit **allow**, with the
    definition hash computed live via
    `definition_hash(description, input_schema)` against the exact
    description/schema strings read out of `local_mcp_store.json`'s
    `discovery_snapshots.docs-server.tools[2]` entry (`"Search the docs
    tree for matching content."` + its object schema). Computed value:
    `33c51126dc2c94c688578cbd67375bc04e780aca84e6dac4331e381a8f2254e7`.
  - `local:docs-server.tools.read_file` — explicit **allow** with a
    deliberately WRONG hash (`"deadbeef"`) — the rug-pull case.
  - `local:docs-server.tools.list_files` — explicit **deny**.
  - No `local:docs-server` server-level `default` (so the pinned
    "Server default — docs-server" row falls through to the global
    default, "Ask").
  - Verified the seed resolves as intended *before* touching the browser,
    by constructing the same three `HubTool`s the app's own
    `local_tools_from_record()` would produce and running them through
    `resolve_effective_state()` directly: `search_docs → Allow •`,
    `read_file → Ask ⚠` (`config_changed=True`), `list_files → Off •`.
- `[mcp] enabled = true` and `hub_lifecycle_timeout_seconds = 8` carried
  over unchanged from Phase 2/3's config.

## Captures

All at 2050×1240, real app CSS, MCP nav destination via the top nav bar's
`MCP` chip (this app version's screen-based nav renders a `MainNavigationBar`
per screen, not a tab bar — clicking the `MCP` label text navigates
directly). Files:
`Docs/superpowers/qa/mcp-hub-phase4-2026-07/mcp-p4-<slug>-2026-07-16.png`.

1. **`permissions-matrix`** — Permissions mode, fresh mount, no selection.
   Pinned **`Global default → Ask`** row, pinned
   **`Server default — docs-server → Ask`** row (falls through to global,
   as seeded), then `docs-server`'s three tool rows
   (`list_files → Off •`, `read_file → Ask ⚠`, `search_docs → Allow •`),
   then the pinned **`Server default — tldw_chatbook → Ask`** row and all
   ten built-in tools as plain, unmarked **`Ask`** (inherited, no override)
   — `chat_with_character`, `chat_with_llm`, `create_note`,
   `export_conversation`, `get_conversation_history`, `ingest_media`,
   `list_characters`, `search_conversations`, `search_notes`, `search_rag`.
   Kill-switch checkbox unchecked (verified by span color, see the driver
   gotcha below — DOM text alone can't tell checked from unchecked here).
   Policy-preview strip below the table: **"Global default: Ask."**.
   Verified: DOM text search for all of `Global default`,
   `Server default — docs-server`, `Allow •`, `Ask ⚠`, `Off •`,
   `list_files`, `read_file`, `search_docs`, `MCP tools in chat`,
   `Global default: Ask.` all hit. PNG 154KB.

2. **`matrix-space-cycle`** — cursor moved onto the built-in
   `list_characters` row (single click — DataTable cursor-move, not a
   row-select) and `Space` pressed once. Row re-renders
   **`Allow •`** in place (`cycle_ui_state(None) == "allow"` per
   `permission_store.py`), matrix resynced in place (cursor preserved on
   the same row per `update_matrix()`'s row-key restore). Persisted to
   `mcp_permissions.json`: a new
   `servers."builtin:tldw_chatbook".tools.list_characters` entry with a
   freshly computed `definition_hash` and `state: "allow"`. Verified: DOM
   text search for `list_characters` + `Allow •` on the same row hit. PNG
   147KB. (This tool's permission state was later reset back to Inherit —
   see capture 10's note — so it doesn't collide with the
   "plain inherited Ask" contract captures 1/7 rely on elsewhere.)

3. **`kill-switch-toggled`** — clicked the `MCP tools in chat` checkbox
   once. DOM text alone shows `▐X▌` either way (Textual's `Checkbox`
   `BUTTON_INNER` is the literal glyph `"X"` unconditionally — see the
   driver-gotcha section), so verified the **color** flip via
   `getComputedStyle`: unchecked → `X` rendered
   `color: rgb(0, 15, 24)` on `background-color: rgb(36, 47, 56)` (near-
   identical values, effectively invisible); checked → `color: rgb(138,
   212, 161)` (bright green) on the same dark background (high contrast,
   clearly visible) — confirmed against this capture's own DOM dump before
   saving. Cross-checked against disk: `mcp_permissions.json`'s
   `kill_switch` flipped `false → true` at click time. PNG 149KB. **Toggled
   back to `false` immediately after this capture** (confirmed via a
   second disk read) so the seed's documented kill-switch state
   (`false`) holds for the rest of the round.

4. **`inspector-explanation-override`** — `search_docs` row selected (two
   clicks). Inspector: **`Permission: Allow`** +
   **"From this tool's override."** (`_ORIGIN_SENTENCES["tool_override"]`,
   `mcp_inspector.py`). Verified: DOM text search for both exact strings
   hit on the same inspector pane. PNG 151KB.

5. **`inspector-reallow`** — `read_file` row selected instead. Inspector:
   **`Permission: Ask`**, **"From this tool's override."**, then the
   config-changed notice **"Definition changed since you allowed it."**
   (`_CONFIG_CHANGED_NOTICE`) and a **`Re-allow`** button
   (`mcp-inspector-reallow`). Verified: DOM text search for
   `Permission: Ask`, `Definition changed since you allowed it.`, and
   `Re-allow` all hit. PNG 156KB.

6. **`reallow-applied`** — pressed `Re-allow`. Matrix re-renders
   `read_file` as **`Allow •`** (⚠ gone); inspector now shows
   `Permission: Allow` / `From this tool's override.` with no
   config-changed notice. Confirmed on disk: `read_file`'s
   `definition_hash` in `mcp_permissions.json` was rewritten to the tool's
   *current* live hash
   (`c70c320dabad24764b2c472dc23bebdb41c53e64630eb6338def6af32c659742`,
   distinct from both the original correct-at-seed-time hash and the
   deliberately-wrong `deadbeef`) and the entry has no `config_changed`
   marker. Verified: DOM text search for the exact row substring
   `read_file                       Allow` (spacing-scoped, to rule out
   also matching `search_docs`'s own `Allow •`) hit. PNG 151KB.

7. **`tools-state-column`** — Tools mode, no filter, no selection. `State`
   column present with the identical labels/markers Permissions mode uses:
   `list_files → Off •`, `read_file → Allow •` (post-Re-allow),
   `search_docs → Allow •`, `list_characters → Allow •` (post capture 2's
   Space-cycle), all other built-in tools plain `Ask`. **Required a
   manual `r` (refresh) press first** — see Defect 1 below; the State
   column does not resync on its own after a Permissions-mode mutation.
   Verified: DOM text search for `State`, `Off •`, `Allow •` all hit
   (no `Ask ⚠` remained in the catalog at this point, since `read_file`
   had already been re-allowed — expected, not a gap). PNG 159KB.

8. **`test-tool-denied`** — Tools mode, `list_files` selected, Test Tool
   opened (raw-JSON fallback, `{}`), Run pressed. Result: **`Failed ·
   0ms`** + **"Blocked — this tool is set to Off in Permissions."**
   (`_TOOL_TEST_BLOCKED_TEXT`, `mcp_workbench.py`) — no execution attempt,
   confirmed by the 0ms duration and the gate message itself (a real
   execution against a stale/unreachable `docs-server` would show a
   nonzero duration and a connection-failure message instead, per Phase
   3's own `tool-test-failed` capture). Verified: DOM text search for the
   exact blocked sentence hit. PNG 127KB.

9. **`test-tool-ask-armed`** — `ingest_media` selected (plain inherited
   `Ask`, global-default origin — confirmed via the inspector's own
   "Inherited from the global default." line before opening Test Tool),
   Test Tool opened, first `Run` press. Button relabels to **`Confirm
   run`** (`require_confirm()`, `mcp_inspector.py`) — no result yet.
   Verified: DOM text search for `Confirm run` + `ingest_media` (name
   still shown in the inspector header) both hit. PNG 124KB.

10. **`test-tool-ask-run`** — **used `list_characters`, not the
    `ingest_media` tool armed in capture 9**, per the task brief's own
    instruction to use a known-good executable for this specific capture.
    `list_characters` had been left in the `Allow` state by capture 2's
    Space-cycle, which would have skipped the ask-gate arm/confirm flow
    entirely (an explicit `Allow` runs on the first press) — so it was
    reset back to **Inherit** first: three more `Space` presses on its
    Permissions-mode row (`allow → ask → deny → None`, `cycle_ui_state`'s
    full rung order), confirmed by DOM text (`list_characters ... Ask`,
    no marker) and by disk (the `builtin:tldw_chatbook.tools
    .list_characters` entry removed entirely from `mcp_permissions.json`,
    matching `set_tool_state(..., state=None)`'s documented behavior).
    Then, in Tools mode (after an `r` refresh so the catalog picked up the
    reset — same staleness gap as capture 7): selected `list_characters`
    (confirmed inspector shows `Permission: Ask` / global-default origin),
    Test Tool, first `Run` press → armed `Confirm run`, second press →
    actual execution: **`OK · 6ms`** +
    `{"source": "local", "tool_name": "list_characters", "result":
    [{"id": 1, "name": "Default Assistant", "description": "A
    general-purpose assistant.", "message_count": 0}], "governance":
    {...}}` — a real character row read from the seeded ChaChaNotes DB
    through the full ask-gate arm → confirm → execute path. Verified: DOM
    text search for `OK · 6ms` and `Default Assistant` both hit. PNG
    149KB.

11. **`shortcut-footer`** — **the "space cycle permission" hint is not
    reachable anywhere in the running app; this capture instead documents
    that absence.** See Defect 2 below for the full root-cause analysis.
    Captured Permissions mode's actual bottom bar for the record: it shows
    only Textual's own built-in `Footer` widget rendering the app's global
    `Binding`s (`^q Quit App`, `^p Palette Menu`, `f1 Help`,
    `f6 Next Pane`) — none of `MCP_SHORTCUTS`' five entries (`1-4 mode`,
    `a add server`, `r refresh`, `t test tool`,
    **`space cycle permission`**) appear on screen. Verified: DOM text
    search for `cycle permission`, `add server`, `test tool`, and
    `refresh` all **failed** (not found anywhere in the 82-row terminal
    buffer); a parallel search for the visible Footer's own labels
    (`Quit App`, `Palette Menu`, `Help`, `Next Pane`) all hit, confirming
    the capture and search methodology were sound and the shortcut text
    is genuinely absent, not just mis-searched. PNG 155KB.

### Bonus capture (defect evidence, not on the requested list)

- **`defect-tools-mode-stale-state`** — Tools mode showing `chat_with_llm`
  as plain **`Ask`** immediately after Permissions mode had just resolved
  the same tool to **`Allow •`** (a fresh Space-cycle, not yet followed by
  a refresh) — the direct before/after evidence for Defect 1. PNG 164KB.

## Driver gotchas (methodology, not app defects)

- **A Textual `Checkbox`'s `X` glyph is always present in the DOM text,
  checked or not.** `ToggleButton.BUTTON_INNER = "X"` unconditionally
  (`textual/widgets/_toggle_button.py`); only the `toggle--button` visual
  style (a `color`/`background-color` pair from CSS, resolved per
  checked/unchecked pseudo-state) changes. A plain DOM-text search for
  `"X"` or even the assembled `"▐X▌"` glyph sequence therefore cannot
  distinguish checked from unchecked — this round used a `getComputedStyle`
  read of the `X` span's `color` vs. its `background-color` (near-identical
  = invisible = unchecked; high-contrast = visible = checked) instead, and
  cross-confirmed every checkbox toggle against the on-disk
  `mcp_permissions.json`.
- **xterm's DOM renders inter-word spaces as U+00A0 (non-breaking space)
  in some cells and a plain U+0020 space in others** (empirically:
  freshly-painted static text tends to use U+00A0; text inside certain
  bordered/focused containers can differ) — a plain-space `indexOf()`
  search silently missed rows it should have matched partway through this
  round (the kill-switch checkbox's own row, after it gained a focus
  border). Fixed by normalizing every U+00A0 to a plain space on both the
  haystack (`row.textContent`) and the needle before every `indexOf()`
  call; every capture in this round was re-verified after that fix.
- **A lingering hover tooltip can supply a false DOM match for a text
  search**, same lesson as Phase 3: this round's own `"Run"` search
  initially matched the *Test Tool button's own tooltip* text ("Run this
  tool with test arguments.") rather than the actual `Run` button, because
  the tooltip was still rendered from a previous hover. Worked around by
  locating the button via its *isolated* row (the row whose trimmed text
  is exactly `Run`/`Close` with nothing else on it, found by scanning for
  the `Close` row and stepping back one row) rather than a bare substring
  search, and by moving the mouse to a neutral corner before every
  verification dump.
- **DataTable single-click vs. two-click semantics were used
  deliberately differently across captures**: a single click only moves
  the cursor (needed before a `Space`-cycle, which reads
  `table.cursor_row` — a second click would instead fire `RowSelected`
  and open/refresh the inspector, which is harmless here but unnecessary);
  a genuine row *selection* (to populate the inspector) used the
  documented two-click pattern throughout.

## Defects / observations found

### Defect 1 (High, functional/UX) — Tools mode's cross-server catalog State column does not refresh after a Permissions-mode mutation — FIXED

**Reproduction:** MCP Hub → Permissions mode → Space-cycle any row (or
toggle the kill switch, or press Re-allow) → switch to Tools mode (or it
was already open — `ContentSwitcher` keeps both canvases mounted) without
pressing `r` (refresh) or otherwise triggering a full workbench reload.
Tools mode's `State` column for the tool just changed still shows its
*previous* label. Fully deterministic, reproduced twice this round (once
for `list_characters` after capture 2's Space-cycle, once for `read_file`
after capture 6's Re-allow, and a third time deliberately for
`chat_with_llm` to capture clean before/after evidence — see the bonus
capture above).

**Root cause:** `MCPWorkbench._sync_children()`
(`tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:836-895`) is the only path
that resolves `_collect_hub_tools()` + `_resolve_effective_states()` ONCE
and threads the result into *both* `_sync_tools_mode()` (Tools mode's
State column) and `_sync_permissions_mode()` (the matrix). But the three
event handlers that mutate the permission store directly — the
Space-cycle handler, the kill-switch-toggle handler, and the Re-allow
handler — are documented in `_sync_children()`'s own docstring as
"STANDALONE callers of `_sync_permissions_mode()`" that "deliberately do
NOT go through this path" and "resolve fresh" for the Permissions canvas
alone, precisely because "each of those handlers just mutated the store
itself" (an intentional perf optimization to avoid a redundant full store
reload + resolve pass for a change that, per that comment's reasoning,
only the Permissions canvas needs to see immediately). The trade-off this
optimization makes is not free, though: `MCPToolsMode`'s own `_last_
hub_tools`/State-column data is never told about the mutation, so it
keeps showing the pre-mutation `EffectiveToolState` until the next full
`_sync_children()` pass (a manual `r` refresh, a lifecycle action, a
runtime-backend-change event, or a fresh navigation to the screen) — and
`set_mode()` (`mcp_workbench.py:1459-1489`, the handler for clicking the
Tools/Permissions/Servers/Audit chips) does not trigger one; it only
swaps the `ContentSwitcher`'s visible pane and clears the inspector's Test
Tool panel.

**Impact:** A user who changes a tool's (or the kill switch's, or a
Re-allow's) permission state in Permissions mode and then switches to
Tools mode to double-check it (or to run Test Tool against it) sees a
stale, contradictory State column — the exact "same labels/markers"
capture 7 was written to demonstrate assumes cross-mode consistency that
the app does not actually guarantee without an intervening refresh. This
is not merely cosmetic: `MCPToolsMode`'s catalog is also what feeds the
Test Tool gate's *display* (though not its actual enforcement — `gate_
tool_test()` re-resolves from the live store at Run time regardless of
what the State column shows, so a stale "Ask" next to an actually-Allowed
tool would not let anything unsafe through; the risk is purely a
misleading/confusing display, not a permission bypass).

**Verified two independent ways:** (1) live reproduction with disk
cross-checks (`mcp_permissions.json`'s tool entries already reflected the
new state while the Tools-mode DOM still showed the old label); (2)
source read of `_sync_children()`'s own docstring, which documents this
exact trade-off by name ("The three STANDALONE callers of `_sync_
permissions_mode()`... deliberately do NOT go through this path").

**Fix (2026-07-16):** `MCPWorkbench._sync_permissions_mode()` now tracks
whether it was called standalone (`effective is None`, i.e. by one of the
three mutation handlers above) versus as part of a full `_sync_children()`
pass. On a standalone call, it hands the SAME freshly-resolved
`EffectiveToolState` batch it just computed for its own matrix resync to a
new narrow `MCPToolsMode.update_states()` setter, which refreshes the
widget's cached `_states` and re-renders its rows (`_apply_filter()`)
without touching the cached tool list or rebuilding the server-filter
Select. This adds zero additional `effective_tool_states()` calls,
governance fetches, or full `_sync_children()` passes — the T8/T10
counting test (`test_sync_children_resolves_effective_states_exactly_once`)
still asserts exactly one `effective_tool_states()` call per full sync.
RED-first: `Tests/UI/test_mcp_workbench.py::
test_space_cycle_propagates_fresh_states_to_tools_mode_without_full_resync`
and `::test_reallow_propagates_fresh_states_to_tools_mode_without_full_resync`
reproduced the stale-column bug pre-fix, pass post-fix. Commit:
`412431c0` — fix(mcp-hub): propagate fresh permission states to the
tools-mode column on standalone mutations.

### Defect 2 (Medium, UX/dead-code) — the footer shortcut-hint system (`MCP_SHORTCUTS`, `AppFooterStatus.set_workbench_shortcuts`) never renders in the running app

**Reproduction:** Navigate to any screen that calls
`AppFooterStatus.set_workbench_shortcuts()` (MCP, Chat/Console, Personas —
confirmed by `grep` across `UI/Screens/`) and look at the bottom of the
screen. The hint text those calls register (for MCP: `1-4 mode`,
`a add server`, `r refresh`, `t test tool`, `space cycle permission`) is
never visible anywhere. Confirmed this round via exhaustive DOM text
search across the full 82-row terminal buffer on the Permissions-mode
screen: `cycle permission`, `add server`, `test tool`, and `refresh` all
came back "not found", while a parallel search for the text that *is*
visible at the bottom (`Quit App`, `Palette Menu`, `Help`, `Next Pane`)
all hit — ruling out a search-methodology miss.

**Root cause:** `AppFooterStatus` is mounted exactly once, as a sibling of
`TitleBar`/`TabLinks`/`Container(id="screen-container")`, inside `TldwCli
._create_main_ui_widgets()` (`app.py:4359-4389`) — i.e. it lives on the
App's own default/base Textual `Screen`. But `app.py`'s own comment at
that call site says "ALWAYS use screen-based navigation now", and every
feature screen (`MCPScreen`, `ChatScreenEnhanced`, `PersonasScreen`, ...
all subclass `BaseAppScreen`) is a *separate* Textual `Screen` that gets
pushed onto the app's screen stack via the navigation system — and
`BaseAppScreen.compose()` (`UI/Navigation/base_app_screen.py:87-100`)
yields its *own* `MainNavigationBar` (the "Home Console Library... MCP..."
bar actually visible in every capture in this round) and its own
`Footer(show_command_palette=False)` — Textual's built-in Footer widget,
which is what actually renders the `Quit App`/`Palette Menu`/`Help`/`Next
Pane` bar seen at the bottom of every capture in this round (those are
literal `Binding` descriptions with `show=True` on the `TldwCli` App class
itself, e.g. `Binding("ctrl+q", "quit", "Quit App", show=True)` at
`app.py:2181`, auto-rendered by Textual's `Footer`). Once any such screen
is pushed, it fully occupies the viewport; the App's own base-screen
content — including the one `AppFooterStatus` instance — is not part of
the visible screen at all. `MCPScreen._register_footer_shortcuts()`
(`UI/Screens/mcp_screen.py:215-223`) still finds the widget via
`self.app.query_one(AppFooterStatus)` (Textual's `query_one` searches the
whole App's DOM, not just the active screen, so this does not raise
`QueryError` and silently no-op the way the method's own `try/except`
suggests it might for a *genuinely* absent widget) and successfully calls
`set_workbench_shortcuts()` on it — the call "succeeds", but updates a
widget that is never composited into any visible screen.

**Impact:** The entire footer context-shortcut feature — not just MCP's
`space cycle permission` hint, but every screen's own registered hints
(Console's `F6 next pane` / `Shift+F6 previous pane` / `Enter send`, etc.)
— is dead in the current screen-based-navigation build. This reads as a
regression from an earlier tab-bar-based navigation model (the
`_create_main_ui_widgets()` code still has a live, non-dead-code branch
that mounts `TabBar`/`TabDropdown`/`TabLinks` conditionally, suggesting
`AppFooterStatus`'s placement predates or was never updated for the
"ALWAYS use screen-based navigation now" change documented right above it
in the same function).

**Verified two independent ways:** (1) live DOM search across the full
terminal buffer, both on this round's Permissions-mode screen and (per the
session's very first capture, taken while still on the Console/Chat
screen before navigating to MCP) confirming the visible footer bar's own
literal source is `Binding` descriptions on `TldwCli`, not `AppFooterStatus`
or any screen-specific `WORKBENCH_SHORTCUTS` tuple; (2) source read
tracing `AppFooterStatus`'s single mount point
(`_create_main_ui_widgets()`) against `BaseAppScreen.compose()`'s own,
separate `Footer()` yield and Textual's documented screen-stack semantics
(a pushed `Screen` fully replaces what's visually composited).

Not fixed — reported per the task's own "capture, document, report; do not
fix code" instruction.

### Observations (not filed as defects)

- **Capture 9/10 required swapping tools mid-plan.** The task brief
  suggested `ingest_media` *or* `list_characters` for capture 9 and named
  `list_characters` specifically for capture 10. Because capture 2's own
  Space-cycle demo (which the brief left unspecified which tool to use)
  had already moved `list_characters` to an explicit `Allow`, capture 10
  needed that tool reset back to Inherit first (three more Space presses)
  to reproduce the ask-gate arm → confirm flow the capture is meant to
  show — documented in capture 10's own note above, not filed as a defect
  (the reset behaved exactly as `cycle_ui_state()`'s documented rung
  order predicts).
- **Everything else driven this round** (Permissions matrix grouping/
  pinning/markers, kill-switch checkbox mechanics, inspector explanation
  sentences and the Re-allow flow's hash rewrite, the Tools-mode State
  column's label vocabulary once refreshed, the Test Tool gate's
  Off/Ask-arm/Ask-confirm/Allow-through behaviors) matched the spec
  exactly with no further defects observed.

## Isolated HOME

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-p4-20260716`** (port 9191).
Final `mcp_permissions.json`: `local:docs-server` unchanged from the seed
except `read_file`'s hash (rewritten by capture 6's Re-allow) and a
`list_files`/`search_docs` pair otherwise as seeded;
`builtin:tldw_chatbook` carries one surviving override,
`chat_with_llm → allow` (from the Defect-1 repro capture, intentionally
left in place as it was never meant to be reset); `list_characters`'s
override was added then removed back to Inherit (captures 2 → 10, net
no-op on disk). `kill_switch` is `false` (toggled on for capture 3, then
back off). `local_mcp_store.json`'s `runtime_activity` gained one new
`list_characters` execution-log entry from capture 10's real Test Tool
run; `profile_runtime_state` is unchanged from the Phase 3 seed
(`docs-server`/`slow-server` failed-attempt records carried over, not
re-triggered this round). Both the `run_web_server` process and the
driver's persistent headless Chromium were killed at the end of this
round.
