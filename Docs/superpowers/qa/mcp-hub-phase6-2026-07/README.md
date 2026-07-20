# MCP Hub Phase 6 — QA evidence (2026-07-20, finale)

Branch: `claude/mcp-hub-phase6`, HEAD `0d6b15e4f31b7933401e2111c2573cd4a5a4113c`
("fix(mcp-hub): clear finding pane on server switch, route refresh to its
own target, retire phase-promise copy"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase6`.

**All 12 requested captures are present.** 11 of 12 exercise the real,
live-running app (real CSS, real DataTable/Select/Collapsible widgets, real
config/store files) against genuine seeded data. The 12th (Findings
remediation) could not be driven live — no `tldw_server` target is
configured in this QA HOME (same as every prior phase's HOME), and
governance findings are server-source-only data with no local equivalent;
the server-only empty state was captured instead, per the task's own
documented fallback.

## Recipe

Same base methodology as
[`Docs/superpowers/qa/mcp-hub-phase5-2026-07/README.md`](../mcp-hub-phase5-2026-07/README.md):
`textual-serve` (real app CSS, worktree code), headless bundled Playwright
Chromium via CDP, viewport **2050×1240**, DOM-rendered xterm text search
(NBSP-normalized), `Range`-based click-coordinate resolution, route-abort
non-localhost, two-click `DataTable` row selection.

- Served on port **9197** (fresh port). Started via:
  `HOME=/private/tmp/tldw-qa-mcp-hub-p6-20260720 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "from tldw_chatbook.Web_Server.serve import run_web_server; run_web_server(host='127.0.0.1', port=9197)"`
  from this worktree, using the shared repo `.venv` (no per-worktree venv).
  No stale serve process was found running beforehand.
- Driver Chromium: bundled Playwright build launched standalone with
  `--headless=new --remote-debugging-port=9326
  --user-data-dir=/private/tmp/tldw-qa-driver-p6/profile
  --window-size=2050,1240`, driven via
  `playwright.chromium.connect_over_cdp("http://127.0.0.1:9326")` from a
  forked driver module, `/private/tmp/tldw-qa-driver-p6/driver.py` (same
  action set as Phase 5's driver — `open`, `screenshot`, `text`, `find`,
  `key`, `type`, `click <needle> [occurrence]`, `dclick`, `clickxy`,
  `corner`, `html` — **plus a new `color <needle> [occurrence]` action**,
  see below). Non-localhost requests aborted at the route layer.
- **New this round — DOM color verification.** `act_color()` walks the DOM
  text nodes the same way `act_click_text()` does, then reads
  `window.getComputedStyle(el).color` off the matched node's parent
  element. Textual-serve's xterm DOM renderer paints one `<span
  style="color: rgb(...)">` per distinct-color run within a row, so this
  reliably isolates the actual painted color of a specific word/glyph, not
  just its presence. Used on every colored-state capture below.

### Isolated HOME

`cp -R /private/tmp/tldw-qa-mcp-hub-p5-20260717 /private/tmp/tldw-qa-mcp-hub-p6-20260720`
(Phase 5's HOME on disk left untouched). Carries forward: the `docs-server`
discovery snapshot (`search_docs`/`read_file`/`list_files`), the three
stale/failing local profiles (`docs-server` connect-failed, `weather-api`
missing-env-var, `slow-server` timeout), and Phase 3-5's execution-log
history.

- **`mcp_permissions.json` matched Phase 5's documented seed exactly**
  (verified byte-for-byte before touching the browser): `kill_switch:
  false`, `global_default: "ask"`, `local:docs-server` — `list_files` deny,
  `read_file` allow with the deliberately wrong hash `"deadbeef"` (the
  rug-pull case), `search_docs` allow with the correct hash. No
  `builtin:tldw_chatbook` entries.
- **`mcp.hub_state.advanced_visible` confirmed ABSENT from `config.toml`**
  at session start (only `advanced_open = true` was present, a
  Phase-5-era leftover from before this key existed — harmless, since
  `advanced_open` only matters once `advanced_visible` gates the
  collapsible into existence at all). `get_cli_setting(...,
  default=False)` therefore resolved the opt-in default correctly.
- **Seeded `resources`/`prompts` onto the `docs-server` discovery
  snapshot** in `local_mcp_store.json` (`discovery_snapshots.docs-server`),
  which Phase 5 had left as empty `[]` arrays (no capture back then
  exercised them):
  ```json
  "resources": [
    {"uri": "docs://readme", "name": "README"},
    {"uri": "docs://guide"}
  ],
  "prompts": [{"name": "summarize"}]
  ```
- `[mcp] enabled = true`, `hub_lifecycle_timeout_seconds = 8` carried over
  unchanged.

## Per-capture notes

All at 2050×1240, real app CSS. Files:
`Docs/superpowers/qa/mcp-hub-phase6-2026-07/mcp-p6-<slug>-2026-07-20.png`.
MCP Hub reached via the top nav bar's `MCP` label; mode tabs `Servers /
Tools / Permissions / Audit` underneath.

1. **`servers-colored-states`** — LIVE. Servers mode, default view (nothing
   selected). Table: `tldw_chatbook (built-in)` **Ready** (cursor-row on
   load — see Driver gotcha 1), `docs-server` **Needs attention**,
   `weather-api` **Needs setup**, `slow-server` **Needs attention**. Rail
   (left "Servers" list) independently colors all four the same way.
   Verified via `color`: table `Needs attention` = `rgb(244, 0, 95)`
   (red), table `Needs setup` = `rgb(253, 151, 31)` (amber), table `Ready`
   = `rgb(152, 224, 36)` (green, re-checked after moving the DataTable
   cursor off that row — see gotcha 1). Rail: `tldw_chatbook` (Ready) =
   `rgb(78, 191, 113)` green, `weather-api` (Needs setup) = `rgb(254, 166,
   43)` amber, `slow-server` (Needs attention) = `rgb(185, 60, 91)` red —
   three distinct states confirmed in both the table and the rail. PNG
   97788 bytes.

2. **`permissions-colored-matrix`** — LIVE. Permissions mode, `docs-server`
   tool rows visible: `list_files` **Off •**, `read_file` **Ask ⚠** (the
   rug-pulled tool), `search_docs` **Allow •**; global/server-default rows
   all **Ask**. Preview strip: `global default: ask · 3 overrides across 1
   server`. Legend: `• override · ⚠ definition changed · ⚑ high-risk floor
   · Space cycles Inherit → Allow → Ask → Off`. Verified via `color`:
   `Allow •` = `rgb(152, 224, 36)` green, `Off •` = `rgb(244, 0, 95)` red,
   every `Ask` occurrence (13 hits enumerated) = `rgb(253, 151, 31)` amber
   including the one on the `read_file` row, and the row's own `⚠` glyph =
   `rgb(253, 151, 31)` amber too (glyph stays the colorblind-safe channel,
   consistent with the warning color). PNG 112623 bytes.

3. **`permissions-filter`** — LIVE. Typed `search` into
   `#mcp-perm-filter-text` (`Filter tool or server…`). Table narrowed to
   `search_docs`, `search_conversations`, `search_notes`, `search_rag`;
   `list_files`/`read_file` (no match) dropped; `Global default` and
   `Server default — docs-server` / `Server default — tldw_chatbook`
   pinned rows stayed visible (docs-server's default row correctly stayed
   pinned because `search_docs` is still a visible child under it).
   Verified: DOM search for `list_files` NOT FOUND while `search_docs` and
   `Server default — docs-server` both hit. PNG 95936 bytes.

4. **`permissions-cascade`** — LIVE. Selected the rug-pulled `read_file`
   row. Inspector renders the three-rung cascade exactly as specced:
   `▸ Tool override: Allow ⚠` (the **winning** rung, ▸-prefixed), `Server
   default: —`, `Global default: Ask`, with `Permission: Ask` (the
   downgraded *effective* state) shown above them and `Definition changed
   since you allowed it.` + a `Re-allow` button below. Verified via
   `color`: the winning `Tool override: Allow` text = `rgb(254, 166, 43)`
   (amber/warning) — **not** ready-green despite its raw stored value
   being `"Allow"` — confirming the downgrade-aware rendering the plan
   called for; the two non-winning rungs (`Server default: —`, `Global
   default: Ask`) both = `rgb(165, 165, 165)` (dimmed gray). PNG 125999
   bytes.

5. **`mutation-echo`** — LIVE. Selected the `list_files` row (state `Off
   •`) and pressed `Space` (see Driver gotcha 3 — case matters). Row
   cycled `Off → Inherit` (wrap-around per the legend); preview strip's
   first line became `list_files → Inherit · global default: ask · 2
   overrides across 1 server` — the exact `"{tool_name} → {ui_label} · "`
   echo format from the plan. Verified via `find`. PNG 121955 bytes. (This
   mutation was **not** reverted before the round ended — see "Isolated
   HOME after this round" below.)

6. **`change-in-permissions-jump`** — LIVE. Tools mode → selected
   `search_docs` → its permission block's `Change in Permissions` button →
   clicked. Landed on the **Permissions** tab (now the active/underlined
   mode) with the `search_docs` row already the DataTable cursor row;
   Inspector immediately shows the matching cascade detail
   (`search_docs — docs-server`, `Permission: Allow`, `▸ Tool override:
   Allow •`, `Server default: —`, `Global default: Ask`) — no stale
   Tools-mode content (no leftover `Test Tool` button, no stale preview
   echo). PNG 214402 bytes.

7. **`audit-colored-decisions`** — LIVE. Audit mode, Executions table, over
   the real Phase 3-5 execution-log history (11 rows: 3 `agent`-initiated
   `list_characters`/`search_notes` records from Phase 5's own live run,
   plus older `test`/`system` seed rows). Decision column: `denied`,
   `approved` (×3), `downgraded` (×3), `allowed` (×6, wrapping); Outcome
   column: `Blocked`, `OK` (×6), `Downgraded` (×3), `Failed` (×2). Selected
   the top `search_notes`/`denied` row — Inspector shows its full JSON
   detail (`"decision": "denied"`, `"ok": false`, `"arguments": null`)
   plus `Open tool` / `Adjust permission` drill links. Verified via
   `color` (cursor moved off row 1 first — gotcha 1): `denied` = `Blocked`
   = `rgb(244, 0, 95)` red; `approved` = `allowed` = `OK` = `rgb(152, 224,
   36)` green; `downgraded` = `Downgraded` = `rgb(253, 151, 31)` amber;
   `Failed` = `rgb(244, 0, 95)` red. Full three-color palette (ready-green
   / warning-amber / error-red) confirmed consistent across both the
   Decision and Outcome columns. PNG 265261 bytes.

8. **`advanced-hidden-default`** — LIVE. Reused the exact frame from
   capture 1 (Servers mode, nothing selected, before Advanced was ever
   pressed this session) — its Inspector panel already shows exactly what
   this capture needs: a small `Advanced…` button at the bottom, **no**
   collapsible content above it. Copied to its own filename rather than
   re-deriving a fresh identical screenshot; see Driver gotcha 2 for why a
   *second* attempt (in Permissions mode, after the Space-cycle mutation)
   was discarded — it collided byte-for-byte with capture 5. PNG 97788
   bytes (identical to capture 1's file — same underlying frame,
   deliberately).

9. **`advanced-revealed`** — LIVE. From the Servers-mode `docs-server`
   detail, clicked `Advanced…` → the collapsible mounted **expanded**:
   `▼ Advanced (legacy control plane)`, `Showing: Local control plane`,
   a `Section` Select (chose **Inventory** from its populated option list:
   `Overview` / `Inventory` / `External Servers` / `Governance`). The
   Inventory section's own JSON payload (prompts/resources/tools/
   server_id/server_label for the builtin `tldw_chatbook` server) is long
   enough that its `Action` Select and `Payload (JSON)` box needed a
   scroll to reach (see Driver gotcha 4); once reached, `Action` = a
   populated Select defaulted to **`Execute Local Tool`**, with `Payload
   (JSON)` showing a live example payload
   `{"tool_name":"search_notes","arguments":{"query":"example"}}` — proof
   the six spec-protected actions (of which `Execute Local Tool` is one
   family) are still reachable post-retirement. PNG 196564 bytes (captured
   scrolled to the bottom, where `Action` is visible; the `Inventory`
   Section-select label itself had scrolled just above the frame at that
   point — its selection was independently confirmed via `find` before
   scrolling).

10. **`resources-prompts-listing`** — LIVE. Servers mode, `docs-server`
    detail panel. New compact sections render directly beneath the
    existing `Tools · 3: list_files, read_file, search_docs` line:
    **`Resources · 2: docs://readme, docs://guide`** and **`Prompts · 1:
    summarize`** — reading exactly the two resource URIs (one with a
    `name`, one without — the `_named_items_text()` fallback-to-`uri`
    path) and one prompt name I seeded into
    `local_mcp_store.json`'s `discovery_snapshots.docs-server`. Verified
    via exact-string `find`. PNG 92487 bytes.

11. **`settings-no-mcp-pane`** — LIVE. Routed via the top nav's `Settings`
    label — **this build's Settings screen is the redesigned (H1/L1-era)
    Settings screen**, not a `Tools_Settings_Window` tab strip; its own
    header already states the boundary in copy: `Mode: Overview | Runtime
    controls stay in MCP and ACP`. Full category list (`Core`: Overview,
    Providers & Models; `Interface`: Appearance, Console Behavior; `Data &
    Privacy`: Storage, Privacy & Security; `Troubleshooting`: Diagnostics;
    `Expert`: Advanced Config; `Domain Defaults`: Library & RAG,
    Artifacts, Personas, Skills, Schedules, Watchlists, Workflows, **MCP
    Defaults**, ACP Defaults) has **no `Unified MCP` entry anywhere** —
    confirmed via DOM search for the literal string `Unified MCP`: NOT
    FOUND on the Overview page. PNG 309892 bytes.

    Note: there **is** a `MCP Defaults` entry under `Domain Defaults` —
    opened it to verify it is *not* a resurrected legacy panel: its detail
    pane reads `State: Read-only contract | MCP owns workflow actions and
    setup`, `Writes allowed: No — destination ownership must be
    implemented before mutation`, `Recovery: open MCP for workflow actions
    and setup`. It's a separate, already-shipped, intentionally
    inert ownership-boundary stub for the still-unstarted task-88 (recorded
    in the Phase 6 plan's own "Out of scope" list); its only mention of
    "Unified MCP panel" is a historical data-lineage citation
    (`Source 1: Unified MCP panel`) inside its own read-only prose, not a
    functioning panel. Not a defect — the retirement is clean.

12. **`findings-remediation`** — Local-source empty state only (feasibility
    fallback used, as the task explicitly allowed). Audit mode → `Findings`
    sub-view. `Source` Select's dropdown offers **only `Local`** in this QA
    HOME (`mcp_server_targets.json` has one target,
    `http://127.0.0.1:8000`, but it was never connected/live this round —
    same as every prior phase's HOME). With `Source: Local`, the pane
    renders exactly `Findings come from a tldw_server target.` (the
    documented local-source empty copy; governance findings are
    inherently server-source-only data — there's no local equivalent to
    seed). Standing up a fake `tldw_server`-shaped HTTP backend serving a
    governance/findings endpoint (to get a *real* finding with remediation
    buttons rendered) would require building new mock-server
    infrastructure for a protocol surface no prior QA round has touched —
    a materially larger lift than this pass's scope, unlike Phase 5's fake
    LLM server (which reused an existing, simple OpenAI-compatible HTTP
    shape). Captured the honest empty state instead. PNG 173664 bytes.

## Driver gotchas (methodology, not app defects)

1. **A `DataTable`'s cursor-highlighted row overrides its own cells'
   semantic foreground color** with Textual's standard reverse/bold/
   underline cursor style — confirmed live twice (Servers-mode `Ready` row,
   Audit-mode `denied`/`Blocked` row): while that row is the keyboard
   cursor, `color` reads `rgb(225, 225, 225)` (near-white) for the *entire
   row's* text run instead of the semantic color. Clicking a different row
   to move the cursor away restores the true per-cell color immediately.
   This is standard Textual `DataTable` behavior (not introduced by Phase
   6) and doesn't hide the state glyph/word text itself — only its
   distinct color while it's under keyboard focus — so it was not filed as
   a defect, just worked around for clean color reads.
2. **A byte-identical screenshot is a legitimate/expected outcome, not a
   bug in the driver** — capture 8's first attempt (in Permissions mode,
   right after capture 5's mutation) produced an MD5-identical PNG to
   capture 5's own file, because nothing on screen had changed between the
   two `screenshot` calls (only `find`/`color` calls ran in between, which
   don't mutate the DOM). Re-shot from a genuinely different, earlier
   frame (Servers mode, nothing selected) instead of shipping a duplicate
   under two names.
3. **Playwright key names are case-sensitive against this
   CDP session, and Textual's own keybindings care**: `driver.py key
   "space"` (lowercase) was silently a no-op against the Permissions
   matrix's Space-cycle binding; `key "Space"` (Playwright's canonical
   name) worked immediately. `ctrl+a` did not act as "select all" inside
   the `#mcp-perm-filter-text` `Input` either (no visible effect) — repeated
   `Backspace` presses were used to clear it instead.
4. **A long `Static`/JSON payload nested inside a `Collapsible` needs `End`
   to reach its bottom, not repeated scrollbar-track clicks** — clicking
   the visible in-DOM scrollbar track below the thumb advanced the view a
   little, then visibly stalled (six more identical clicks produced
   byte-identical screenshots, confirmed via `md5`) well before the
   `Inventory` section's `Action` Select came into view. Clicking inside
   the payload area and pressing `End` jumped straight to the true bottom
   in one step, revealing `Action` immediately. Mouse-wheel scrolling
   (`page.mouse.wheel`) had no visible effect at all in this environment.

## Observations (not filed as defects)

- **No app defects were reproduced this round.** Every requested behavior
  matched source/spec on first try: three-way semantic coloring
  (ready-green / warning-amber / error-red) across the Servers table+rail,
  the Permissions matrix (including the rug-pulled row's own `⚠` staying
  warning-colored), and the Audit Decision/Outcome columns; the
  downgrade-aware cascade winner (raw `Allow` rendered amber, not
  false-green, because it's stale); the permissions text filter (narrows +
  keeps relevant pinned rows); the mutation echo's exact copy shape; the
  cross-mode `Change in Permissions` jump (clean landing, zero stale
  panels); Advanced's opt-in default (hidden until explicitly revealed,
  and the six protected legacy actions still reachable once revealed); the
  new read-only resources/prompts listing; and the full legacy-panel
  retirement (`Unified MCP` is gone from Settings' nav with no trace, and
  the one surface that still name-drops it — the unrelated, separately-
  shipped `MCP Defaults` read-only stub — does so only as a historical
  citation, never as a live panel).
- **The Settings screen has moved on from the `Tools_Settings_Window`
  design this plan's own "verified seam facts" describe** (`ts-nav-*`
  id-prefixed tab strip). The redesigned Settings screen reached via the
  same `Settings` nav label already has zero MCP-specific interactive
  surface of its own — Phase 6's retirement work landed cleanly against
  whichever Settings implementation is live today.
- Same observation as Phase 5: the Audit table's `When` column is UTC, not
  local wall-clock — unchanged this round, not re-verified in depth since
  no new capture depended on it.

## Isolated HOME (after this round)

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-p6-20260720`** (port 9197).
Deltas from the seed, both intentional side effects of driving captures 5
and 9 live (not reverted, in keeping with prior rounds' practice of
documenting rather than scrubbing the final state):

- `mcp_permissions.json`: the `local:docs-server.list_files` entry is
  **gone** (Space-cycling it to `Inherit` in capture 5 removes the
  override entirely rather than storing an explicit "inherit" state) —
  it now correctly falls back to the server default (`ask`, since
  `docs-server` has no explicit server-default override either) rather
  than its old seeded `deny`. `read_file`/`search_docs` unchanged from the
  seed. `kill_switch` still `false`.
- `config.toml`: `mcp.hub_state.advanced_visible` is now `true` (was
  absent/default-`false` at the start of the round) — the real, persisted
  effect of pressing the `Advanced…` reveal button in capture 9.
- `local_mcp_store.json`: `discovery_snapshots.docs-server` carries the
  two seeded `resources` entries and one seeded `prompts` entry added at
  the start of this round (see "Isolated HOME" setup above) — this is
  QA-added seed data, not a live-driven side effect.

Both the `run_web_server` process (port 9197) and the driver's headless
Chromium (CDP port 9326) were killed at the end of this round.
