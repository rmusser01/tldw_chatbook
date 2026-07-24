# MCP Hub Phase 1 — QA evidence (2026-07-13)

Branch: `claude/mcp-hub-phase1`, HEAD `f2596b55` ("fix(mcp-hub): single-source
mode-change chip sync in set_mode"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase1`.

Captured live from textual-serve (real app CSS, worktree code) in headless
bundled Chromium (Playwright, CDP-attached), viewport **2050×1240**, isolated
HOME **`/private/tmp/tldw-qa-mcp-hub-20260713`** (left on disk — see bottom).
No live model/provider needed for this destination; all data is local (seeded
`local_mcp_store.json` + `config.toml`).

## Capture methodology note (not an MCP Hub defect)

`python3 -m tldw_chatbook.app --serve --host … --port …` **does not start the
web server** — it silently launches the plain interactive TUI instead. Root
cause: the `if __name__ == "__main__":` block at the bottom of `app.py`
(≈line 8174) unconditionally runs `TldwCli().run()` before any argument
parsing; the `--serve`/`--host`/`--port` argparse logic that would dispatch to
`Web_Server.serve.run_web_server()` lives entirely inside a separate
`main_cli_runner()` function (defined ≈line 8395) that **is never called**
anywhere in the module. This is a real, general-purpose defect (unrelated to
MCP Hub) that burned time in this QA round; worked around by calling
`tldw_chatbook.Web_Server.serve.run_web_server(host=..., port=...)` directly
via `python3 -c`. Flagging for awareness, not fixing (out of this task's
scope).

## Data seeding

- `~/.config/tldw_cli/config.toml`: full default `CONFIG_TOML_CONTENT` with
  `[mcp] enabled = true` (default is `false`) so the built-in row reads Ready.
- `~/.local/share/tldw_cli/default_user/local_mcp_store.json`: 3 external
  profiles chosen for distinct `tldw_chatbook/MCP/readiness.py` outcomes:
  - `docs-server` — has a `discovery_snapshot` (3 tools, 1 resource) and no
    live client session → **STALE** (`runtime_unavailable`, "3 tools
    discovered; not currently connected."). Command args include
    `--api-key sk-qa-test-redact-0001` to exercise `redact_args()`, and
    `env_placeholders: {"WORKSPACE_ROOT": "$HOME"}` (a *set* var, so it
    doesn't also trip `auth_missing`) to exercise the Env row.
  - `weather-api` — `env_placeholders: {"API_KEY": "$QA_MISSING_KEY"}`,
    `$QA_MISSING_KEY` deliberately never set anywhere in the process
    environment → **NEEDS SETUP** (`auth_missing`, "Missing environment
    variables: QA_MISSING_KEY.").
  - `git-tools` — no discovery snapshot, no env → **NEEDS SETUP**
    (`discovery_not_run`, "Not validated yet — connect or test to discover
    tools.").
  - Plus the built-in `tldw_chatbook (built-in)` row → **READY** (from
    `[mcp].enabled = true`).
  - Verified the derivation directly against `readiness.py` before capturing
    (see below) — table/aggregate line match exactly.

```
local:docs-server   -> STALE          (runtime_unavailable)  3 tools discovered; not currently connected.
local:weather-api    -> NEEDS_SETUP    (auth_missing, discovery_not_run)  Missing environment variables: QA_MISSING_KEY.
local:git-tools      -> NEEDS_SETUP    (discovery_not_run)   Not validated yet — connect or test to discover tools.
builtin:tldw_chatbook -> READY         ()                    Served over stdio when an MCP client launches chatbook.
```

## Captures

All at 2050×1240, real app CSS, MCP nav destination (`nav-mcp` → route `mcp`).

1. **`mcp-servers-overview-mixed-2026-07-13.png`** — Servers mode overview.
   Aggregate line "1 of 4 servers ready — 1 stale, 2 needs setup.", table with
   4 rows in mixed badge states (● Ready / ◌ Stale / ○ Needs setup ×2), and 3
   recovery callouts under the table (one per non-ready server, each showing
   that server's readiness message). Look for: badge glyphs/labels matching
   the state, callouts only for non-ready rows.

2. **`mcp-local-profile-detail-docs-server-2026-07-13.png`** — Local profile
   detail for `docs-server` (rail row click). Command line shown **redacted**
   (`npx -y @modelcontextprotocol/server-filesystem --api-key *** /Users/qa/docs`
   — the raw `sk-qa-test-redact-0001` never appears), an `Env · WORKSPACE_ROOT
   (set)` row, `Tools · 3: list_files, read_file, search_docs`, `Resources ·
   1: README.md`, `Prompts · 0`. Inspector mirrors the readiness message with
   `[runtime_unavailable]` and renders two stacked action buttons ("Connect",
   "View details") — confirmed via DOM dump these are two separate rows, not
   overlapping text (a low-res visual read that looked like overlap turned
   out to just be tight vertical spacing). Look for: redaction, env row,
   counts, correct action set for the STALE/runtime_unavailable reason.

3. **`mcp-builtin-server-detail-2026-07-13.png`** — Built-in
   `⌂ tldw_chatbook (built-in)` row selected (from a fresh, no-prior-selection
   session — see Defect 1 below for why). Shows "● Ready", "Served over stdio
   when an MCP client launches chatbook.", the stdio launch line
   (`python3 -m tldw_chatbook.MCP`), `expose_tools/resources/prompts · True`,
   and the **"Copy client config"** button in the canvas. Inspector shows the
   3 READY_ACTIONS buttons: "Open tool catalog", "Refresh tools", "View
   details" (confirmed present via DOM dump — "Refresh tools" reads faint at
   thumbnail scale but is there). Look for: Ready badge, stdio snippet, all
   three expose flags, Copy client config button, 3 action buttons.

4. **Placeholder modes** — chip-follows-active-mode fix (commit `f2596b55`)
   verified visually and via DOM style inspection (only the active chip's
   `<span>` carries `xterm-bold xterm-underline-1` at the point of
   verification, matching whichever canvas placeholder is showing):
   - `mcp-mode-tools-2026-07-13.png` — "Tools" chip active; canvas: "Tools
     mode arrives in a later phase. Until then, a server's tools are listed
     in its Server detail, and tool actions run via Advanced in the
     inspector."
   - `mcp-mode-permissions-2026-07-13.png` — "Permissions" chip active;
     canvas: "Permissions mode arrives in a later phase. MCP tools are not
     yet callable from chat, so there is nothing to permit yet."
   - `mcp-mode-audit-2026-07-13.png` — "Audit" chip active; canvas: "Audit
     mode arrives in a later phase. Action results appear inline in the
     inspector's Advanced section for now."
   - First two attempts (not kept) showed the mode's own Button **tooltip**
     text overlapping the left rail's Source/Servers panel — a hover artifact
     from the mouse resting on the just-clicked chip (real, expected Button
     tooltip behavior), not an app bug. Recaptured after moving the mouse
     away; not something a real click-and-glance user would ever see this
     way, so not filed as a defect.

5. **`mcp-advanced-external-servers-2026-07-13.png`** — Inspector's Advanced
   section, Section select switched to "External Servers": the legacy panel
   loads and shows all 3 profiles, the Action select shows "Save Profile"
   (one of the `profile.*` action set — `profile.save/delete/connect/
   disconnect/test/refresh` per `unified_control_plane_service.py`
   `available_actions()`), and the Payload (JSON) TextArea is seeded with
   that action's template. **See Defect 2 below — this capture is also the
   evidence for a real secret-redaction bypass.**

## Defects found (documented, not fixed, per task scope)

### Defect 1 (P0, crash) — selecting a second server crashes the whole MCP Hub session

Reproduction: select any server in the rail/table whose readiness `allowed_actions`
includes `VIEW_DETAILS` (i.e. almost every server, since it's in
`READY_ACTIONS` and in nearly every `REASON_TO_ACTIONS` entry), then select a
**different** server whose `allowed_actions` also includes `VIEW_DETAILS`.
Concretely: select `docs-server` (STALE → actions `Connect`, `View details`),
then select `tldw_chatbook (built-in)` (READY → actions `Open tool catalog`,
`Refresh tools`, `View details`). The app raises an unhandled
`textual.errors.DuplicateIds` and the whole textual-serve session dies
("Session ended. — Restart" screen; captured as
`mcp-defect-duplicate-ids-crash-2026-07-13.png`).

Root cause: `MCPInspector.update_readiness()`
(`tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:131-157`) calls
`actions.remove_children()` then, in the same synchronous method, loops over
the new snapshot's `allowed_actions` and calls `actions.mount(button)` for
each — button ids are `f"mcp-inspector-action-{action.value}"`, **not** scoped
per-server. `remove_children()` schedules removal asynchronously; it isn't
awaited, so the old `mcp-inspector-action-view_details` button (from the
previous server) can still be mounted when the new one with the *same id* is
mounted, raising `DuplicateIds`. Full traceback excerpt:

```
tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:407 in on_mcp_rail_server_selected
  self._sync_children()
tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:296 in _sync_children
  self.query_one(MCPInspector).update_readiness(selected)
tldw_chatbook/UI/MCP_Modules/mcp_inspector.py:158 in update_readiness
  raise DuplicateIds(...)
DuplicateIds: Tried to insert a widget with ID 'mcp-inspector-action-view_details',
but a widget already exists with that ID
(Button(id='mcp-inspector-action-view_details', ...)); ensure all child
widgets have a unique ID.
```

Impact: this is not an edge case — clicking through more than one server row
in a single session (the single most obvious thing to do in Servers mode) is
very likely to crash the app, since `view_details` is in nearly every
reason's action set. Worked around for capture 3 by starting a fresh session
and selecting the built-in row first (no prior action buttons mounted).

### Defect 2 (P1/P2, security-adjacent) — Advanced ▸ External Servers panel leaks raw, unredacted secrets

The Servers-mode detail panel correctly redacts secret-looking CLI args via
`redact_args()` (`tldw_chatbook/MCP/redaction.py`) — e.g. `docs-server`'s
command line shows `--api-key ***`. The Inspector's "Advanced (legacy control
plane)" ▸ "External Servers" section does **not**: it dumps the full raw
profile dict verbatim for every local profile, including the literal secret
value `sk-qa-test-redact-0001` and the raw env placeholder value
`$QA_MISSING_KEY`, visible in `mcp-advanced-external-servers-2026-07-13.png`.

Root cause: `render_external_servers_section()`
(`tldw_chatbook/UI/MCP_Modules/unified_mcp_sections.py:137-154`) picks the
list key by `key = "name" if external_servers else "profile_id"`. For a
local-source `external_servers` payload (the shape `_AdvancedSectionShim`
always produces for local profiles — see `mcp_workbench.py`'s
`_AdvancedSectionShim.load_section()`), `key` resolves to `"name"`, but
`LocalExternalMCPProfile.to_dict()` never has a `"name"` field — only
`profile_id`. `_render_named_items()`
(`unified_mcp_sections.py:287-298`) then does
`item.get(key) or item`; since `item.get("name")` is always falsy for local
profiles, it always falls through to printing the **entire raw item dict**
instead of just the profile name/id, defeating the redaction contract that
the rest of the Hub UI enforces.

Impact: any local MCP profile with a secret embedded in its command args or
raw env placeholder value (the store schema explicitly supports/expects
`$NAME` placeholders precisely so raw secrets are never persisted — but this
panel reprints the placeholder string and any non-placeholder literal/arg
value verbatim) has that value exposed in this "escape hatch" panel even
though the primary UI redacts it. Given `redact_args`/`redact_mapping`
already exist and are used elsewhere in this same file's family
(`mcp_servers_mode.py`), this looks like an integration gap rather than a
missing capability.

### Observation (not filed as a defect) — stray focus highlight on the Audit chip at first load

On a **fresh** MCP screen mount, before any user interaction with the mode
strip, the "Audit" chip's rendered `<span>` carries the same
`xterm-bold xterm-underline-1` classes as the genuinely-active "Servers"
chip, with a distinct (slightly lighter) background — i.e. it looks like two
chips are highlighted at once. Confirmed via DOM inspection this is a
**focus** ring (Textual's default focused-widget styling), not the
`.is-active` CSS class — the canvas content stayed correctly on Servers mode
both times this was observed (once on the very first MCP nav in this
session, once again after a fresh session was started following the Defect 1
crash). Once any mode chip is actually clicked, only that chip shows the
highlight and the stray Audit styling disappears for the rest of the
session. Likely just Textual defaulting initial focus to the last-composed
`DestinationModeStrip` child on mount. Cosmetic and momentary, but worth a
follow-up since it's the first thing rendered when a user opens MCP for the
first time and could read as "Audit is selected."

## Fix verification re-capture (2026-07-13, commit `f900e7a9`)

Branch `claude/mcp-hub-phase1` advanced from this round's HEAD (`f2596b55`)
through one intermediate docs commit (`ecb42feb`, this README) to
**`f900e7a9`** ("fix(mcp-hub): async-serialize inspector readiness refresh,
redact Advanced external-server secrets"), which fixes both defects
documented above. Re-captured the two affected screenshots live against
`f900e7a9`, same isolated HOME (`/private/tmp/tldw-qa-mcp-hub-20260713`,
left unchanged), same methodology (textual-serve + Playwright bundled
Chromium over CDP, 2050×1240, real app CSS, route-abort non-localhost,
`HOME=/private/tmp/tldw-qa-mcp-hub-20260713 PYTHONPATH=.` from the worktree,
`Web_Server.serve.run_web_server()` called directly per the capture
methodology note above). Originals kept; new files use a `-redacted` /
`-refresh` suffix so both rounds are visible side by side.

### Defect 1 (P0 crash) — fixed → `mcp-two-click-inspector-refresh-2026-07-13.png`

Root cause was `MCPInspector.update_readiness()` calling
`remove_children()`/`mount()` without awaiting either, so a second server
selection could mount a same-id `view_details` action button before the
first selection's removal had finished, raising an unhandled `DuplicateIds`
(captured pre-fix as `mcp-defect-duplicate-ids-crash-2026-07-13.png` — kept
above). `f900e7a9` makes `MCPInspector.update_readiness()` and the full
`MCPWorkbench._sync_children()` call chain genuinely `async`/awaited
end-to-end.

Re-verified with the **exact original repro**, from a fresh session: clicked
`docs-server` (STALE → actions `Connect`, `View details`), then
re-located and clicked `tldw_chatbook (built-in)` (READY → actions
`Open tool catalog`, `Refresh tools`, `View details`) as the very next
action — no manual settle-wait injected between the two clicks. (Row
coordinates were re-resolved via a live DOM text search between the two
clicks rather than reused, since selecting `docs-server` shifts the rail's
row layout by one line — a capture-script detail, not app behavior.)

Result: **no crash**. The session stayed alive, no "Session ended." screen,
and the canvas/Inspector both correctly refreshed to the *second* selection's
(built-in's) readiness — "● Ready tldw_chatbook (built-in)", the stdio
launch line (`python3 -m tldw_chatbook.MCP`), all three `expose_*` flags
`True`, the "Copy client config" button, and all three READY_ACTIONS buttons
present in the Inspector ("Open tool catalog" / "Refresh tools" / "View
details" — confirmed via text dump; "Refresh tools" renders faint at
thumbnail scale, same known cosmetic note as the original round's capture 3).
This proves the two-click crash — arguably the single most obvious thing a
user does in Servers mode (select one server, then another) — is gone.

Incidental observation: the mode strip's already-documented "stray focus
highlight on the Audit chip" (see Observation above, not a defect) recurred
once during this re-capture's two-click sequence and is visible in this
screenshot's mode strip (the "Audit" label instead of "Servers" carries the
underline/bold focus styling). Canvas and Inspector content stayed correctly
on Servers-mode/built-in data throughout — confirmed via text dump — so this
remains the same cosmetic focus-ring artifact previously observed, not a
regression and not related to Defect 1.

### Defect 2 (P1 security leak) — fixed → `mcp-advanced-external-servers-redacted-2026-07-13.png`

Root cause was `_AdvancedSectionShim.load_section()` handing the legacy
`render_external_servers_section()` renderer raw local-profile dicts; that
renderer keys records by `"name"` (which local profiles never have — only
`"profile_id"`), so its `item.get(key) or item` fallback always printed the
full raw dict, secrets included. `f900e7a9` redacts each record via
`redact_args()`/`redact_mapping()` (`tldw_chatbook/MCP/redaction.py`) at the
`_AdvancedSectionShim.load_section()` integration seam, before the renderer
ever sees it — on both the bare-list local path and any dict payload
carrying an `external_servers` list.

Re-verified the identical navigation as the original leak capture: Inspector
→ "Advanced (legacy control plane)" → Section select → "External Servers".
Same 3 profiles, same raw-dict-dump rendering style (unchanged, as
expected — only the values are now redacted), directly comparable line by
line against the kept original `mcp-advanced-external-servers-2026-07-13.png`:

- `docs-server` args: was `[..., '--api-key', 'sk-qa-test-redact-0001', ...]`
  → now `[..., '--api-key', '***', ...]`.
- `weather-api` env / env_placeholders: was `{'API_KEY': '$QA_MISSING_KEY'}`
  in both fields → now `{'API_KEY': '***'}` in both (the `API_KEY` key name
  itself matches `redact_mapping`'s secret-key pattern, so the env
  *placeholder name* `$QA_MISSING_KEY` is redacted too, not just literal
  secrets — a stricter, fail-closed result than the minimum ask).
- `git-tools`: no secrets in its record either before or after (unchanged).
- `docs-server`'s non-secret `env: {'WORKSPACE_ROOT': '$HOME'}` is correctly
  left **unredacted** (key doesn't match the secret pattern) — confirms this
  isn't a blanket "hide everything" regression, just the secret-keyed
  fields.

Confirmed programmatically against the rendered page text (not just visual
read): `sk-qa-test-redact-0001` — **0 occurrences** anywhere on screen;
`QA_MISSING_KEY` (the raw placeholder value) — **0 occurrences**; exactly
**3** `***` redaction markers present (docs-server's `--api-key` arg,
weather-api's `env.API_KEY`, weather-api's `env_placeholders.API_KEY`). This
proves the "escape hatch" Advanced panel no longer bypasses the redaction
contract the rest of the Hub UI already enforces.

### Nothing else observed wrong in this re-capture round

No new defects surfaced while driving either flow. The pre-existing
"Observation (not filed as a defect)" stray-focus-ring item above is the
only cosmetic artifact seen, and it recurred exactly as previously
characterized (momentary, content-accurate, not a regression from
`f900e7a9`).

## Fixed-defect re-capture (2026-07-13, commit `530cf9df`)

Verified in every capture above and re-checked directly in the two flows
that select a rail row: the SELECTED left-rail row (`Button.mcp-rail-row`
with `.is-active` — e.g. "All servers" or a selected server) rendered as a
**blank, blue-bordered box** with no visible label. Non-selected rows
rendered fine.

**Root cause**: `Button.mcp-rail-row` is fixed at `height: 1` (see
`MCPRail.DEFAULT_CSS`), but the generic `.is-active` rule in
`_agentic_terminal.tcss` sets `border: round $ds-action-focus`. A round
border needs at least 2 lines to render; on a height-1 button it consumed
the row's only line, collapsing the label's content area to height 0. The
more-specific `#mcp-hub-rail Button.mcp-rail-row.is-active` rule already
existed but only set `text-style: bold`, so it never overrode `border` and
the generic rule's border kept winning. The sibling `.mcp-mode-chip.is-active`
rule (`MCPScreen.DEFAULT_CSS` and its bundle mirror) already carries the
correct `border: none` override for the same height-1 hazard and never had
this bug — that's what the fix mirrors.

**Fix** (`530cf9df`): add `border: none;` to
`#mcp-hub-rail Button.mcp-rail-row.is-active` in `_agentic_terminal.tcss`,
rebuilt `tldw_cli_modular.tcss`. No new colors/tokens — the row still uses
`.console-action-subdued`'s existing `color`/`background`, now with
`text-style: bold` actually visible instead of hidden behind a border that
ate the whole row.

Re-captured all three affected screenshots live (same textual-serve +
Playwright/bundled-Chromium-over-CDP methodology, same isolated HOME,
2050×1240), replacing the originals in place:

- `mcp-servers-overview-mixed-2026-07-13.png` — the default-selected "All
  servers" row now shows its bold label instead of an empty bordered box.
- `mcp-local-profile-detail-docs-server-2026-07-13.png` — selecting
  `docs-server` keeps its rail label visible/bold; canvas detail (redacted
  command, Env/Tools/Resources) unaffected/unchanged.
- `mcp-builtin-server-detail-2026-07-13.png` — selecting
  `tldw_chatbook (built-in)` keeps its rail label visible/bold; canvas
  detail (Ready badge, stdio line, Copy client config) unaffected/unchanged.

Also added a regression test,
`Tests/UI/test_mcp_rail.py::test_rail_active_row_label_is_not_blank_with_bundled_css`,
which mounts `MCPRail` under an `App` loading the real bundled stylesheet
and asserts the active row's border is empty, its height stays >= 1, and
its rendered label text is non-empty.

## Remaining two re-captures (2026-07-13, HEAD `a0018a56`)

The `530cf9df` pass above re-captured three of the five screenshots that
select a rail row. The other two — `mcp-two-click-inspector-refresh-2026-07-13.png`
(Defect 1 fix-verification) and `mcp-advanced-external-servers-redacted-2026-07-13.png`
(Defect 2 fix-verification) — still showed the pre-fix blank-selected-row
rail from their earlier capture pass and needed the same re-shoot. Re-ran
live against current HEAD (`a0018a56`, `530cf9df` fix included; no code
changes made in this pass — captures only), same methodology (textual-serve
+ Playwright bundled Chromium over CDP, isolated HOME
`/private/tmp/tldw-qa-mcp-hub-20260713`, 2050×1240, route-abort non-`127.0.0.1:9870`
traffic, `body.-first-byte` readiness gate), replacing both files in place:

- `mcp-two-click-inspector-refresh-2026-07-13.png` — repeated the exact
  Defect 1 repro (click `docs-server`, then immediately click
  `tldw_chatbook (built-in)`, no wait between): no crash, canvas/Inspector
  correctly show the built-in server's readiness (`● Ready`, stdio launch
  line, all three `expose_*` · True, "Copy client config", all three
  READY_ACTIONS buttons — confirmed "Refresh tools" present via text dump,
  faint at thumbnail scale as previously noted), and the selected rail row
  (`● ⌂ tldw_chatbook (buil...`) now renders its bold label instead of a
  blank bordered box.
- `mcp-advanced-external-servers-redacted-2026-07-13.png` — with the
  built-in row still selected (rail label legible), opened Inspector ▸
  Advanced ▸ Section ▸ "External Servers": same redacted rendering as the
  `f900e7a9` verification (docs-server `--api-key`/`***`, weather-api
  `env.API_KEY`/`***` and `env_placeholders.API_KEY`/`***`, git-tools
  unchanged, docs-server's non-secret `WORKSPACE_ROOT` left unredacted).
  Confirmed programmatically against the rendered page text: `sk-qa-test-redact-0001`
  — 0 occurrences; `QA_MISSING_KEY` (raw value) — 0 occurrences; exactly 3
  `***` markers present.

All five captures in this evidence set that show a selected rail row now
render on `530cf9df`/current HEAD and display a legible, non-blank selected
row label. No new defects observed in this pass.

## UX A-batch re-capture (2026-07-13, HEAD `6de76924`)

Branch `claude/mcp-hub-phase1` advanced from the prior round's HEAD (`a0018a56`)
through `530cf9df`'s doc round, then **`f78cf10e`** ("fix(mcp-hub): apply five
approved UX fixes (A1-A5) to MCP Hub Phase 1") and a docs-only commit
**`6de76924`** (Phase 2 UX-input notes, no app code). Re-captured all eight
primary screenshots live against `6de76924` in the same worktree
(`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase1`),
same isolated HOME (`/private/tmp/tldw-qa-mcp-hub-20260713`, unchanged seed
data), same viewport (2050×1240), route-abort non-`127.0.0.1` traffic.

**Driver note (methodology addendum, not an app defect):** a bare
`Web_Server.serve.run_web_server()` process (as used in every prior round)
serves textual-serve's stock JS, which never exposes the xterm terminal/driver
instance to the page (no `window.__drv` or equivalent) — earlier rounds'
`window.__drv`-based buffer reads only worked because that JS had been
hand-patched for a *different*, unrelated QA session's worktree. For this
round the QA driver was rebuilt from `tldw_chatbook.Web_Server.serve`'s own
public building blocks (`_load_textual_serve_server_class()` +
`build_chatbook_web_server_class()` — the exact `ChatbookWebServer` subclass
`run_web_server()` itself instantiates, so the app's own viewport-resize/
first-byte JS patches still apply) with one QA-only addition: a `statics_path`
override pointing at a local copy of textual-serve's bundled
`static/js/textual.js` with a single extra line (`window.__drv=t;`) inserted
right after the terminal wrapper object is constructed. No `tldw_chatbook`
source file was modified — only a QA-local copy of the third-party
textual-serve static bundle, used solely so the Playwright driver could read
the live xterm buffer as text/cell-attributes instead of parsing screenshots.
Screenshot pixels are unaffected either way (confirmed byte-for-byte visual
match against the pre-existing `mcp-servers-overview-mixed` capture before
any A-batch interaction). Flagging as a methodology detail for the next QA
round, not an app defect.

### Per-fix visual + programmatic confirmation

Each fix was checked two ways: (1) objectively, by reading the live xterm
buffer's per-cell style attributes (`bold`/`underline`/`fg`/`bg`) via the
`window.__drv` hook above — renderer-agnostic, not a screenshot guess: and
(2) visually, in the corresponding screenshot.

- **A1 (mode-chip focus vs. active-mode underline) — confirmed.** On a fresh
  MCP mount (`mcp-servers-overview-mixed-2026-07-13.png`), cell-attribute
  read of the mode-strip row showed **only** "Servers" (the true active mode)
  carrying `underline=True`; "Tools"/"Permissions"/"Audit" all read
  `underline=False`. Later in the session, the already-documented (see
  "Observation", pre-A1) stray-focus artifact recurred — after clicking a
  rail row, the "Audit" chip picks up focus as a side effect (unrelated to
  A1's scope, not re-triggered by this fix) — but it now renders with a
  **background/foreground color swap and bold, with no underline**
  (`bg`/`fg` shift from the baseline `#1e1e1e`/`#6f6f6f`-family values to
  `#272727`/`#e2e2e2`; `underline` stays `False` throughout), objectively
  distinct from "Servers"'s bold-underline active style. Zoomed crop of
  `mcp-two-click-inspector-refresh-2026-07-13.png`'s mode strip confirms this
  visually: "Servers" is plain text with an underline; "Audit" has a filled
  background box and bold text, no underline. Before this fix, both states
  rendered identically (bold+underline), which is the bug A1 fixes. The
  recurring stray-focus landing spot itself is unchanged pre-existing
  behavior, not filed as a new defect (see "Observation" above).

- **A2 (disabled inspector action buttons legible) — confirmed.** On
  `mcp-local-profile-detail-docs-server-2026-07-13.png`, the disabled
  "Connect" button reads cell fg `#6f6f6f` / bg `#1e1e1e` — a dim but
  distinctly non-background gray, clearly legible in a zoomed crop right next
  to the enabled "View details" button (bright white). Previously this
  combination (50% opacity stacked on `$text-disabled`) read as functionally
  invisible.

- **A3 (no internal reason-code/config-flag vocabulary in user copy) —
  confirmed.** Full-page text search on the docs-server detail capture found
  zero occurrences of `[runtime_unavailable]` or any other bracketed
  `ReasonCode` value. The built-in detail capture
  (`mcp-builtin-server-detail-2026-07-13.png`) shows `Exposes · tools,
  resources, prompts` in place of the old raw `expose_tools · True` /
  `expose_resources · True` / `expose_prompts · True` flag dump (verified
  `"expose_tools"` is absent from the rendered text).

- **A4 (rail rows left-aligned, one hard left edge; built-in label
  untruncated) — confirmed.** Column analysis of the rail's five rows on
  fresh mount: every row's Button content starts flush at the same left
  column immediately after the border + 1-col padding (no more
  center-justified Button text sitting at different x-offsets per label
  length). The local-profile rows' labels ("docs-server", "weather-api",
  "git-tools") and "All servers" (now padded with a matching 2-col gutter)
  all start their label text at the same column; the built-in row's label
  starts 2 columns further right only because it carries an extra
  `⌂` built-in icon before the label (by design, not a misalignment). `⌂
  tldw_chatbook (built-in)` renders in full with no ellipsis in both the rail
  and the servers table, in every capture that shows it.

- **A5 (inspector "Why" line, not a repeat of the canvas message) —
  confirmed.** `mcp-local-profile-detail-docs-server-2026-07-13.png`'s
  Inspector reads `Why · Not connected` (the humanized `REASON_LABELS`
  phrase for `RUNTIME_UNAVAILABLE`), distinct from the canvas detail's own
  `docs-server: 3 tools discovered; not currently connected.` — previously
  the Inspector just repeated the canvas's `snapshot.message` verbatim with a
  bracketed reason code appended.

### Defect 1 / Defect 2 fixes still hold

Re-ran both original repros against `6de76924` as part of this pass:

- **Two-click crash (Defect 1):** `mcp-two-click-inspector-refresh-2026-07-13.png`
  — clicked `docs-server`, then immediately (~150 ms, row coordinates
  re-resolved live) clicked `tldw_chatbook (built-in)`. No crash, no "Session
  ended." screen; canvas/Inspector correctly show the built-in server's Ready
  state, stdio launch line, all three `expose_*` flags (now rendered as
  `Exposes · tools, resources, prompts`), and all three READY_ACTIONS
  buttons.
- **Advanced-panel secret leak (Defect 2):**
  `mcp-advanced-external-servers-redacted-2026-07-13.png` — Inspector ▸
  Advanced ▸ Section ▸ "External Servers" with the built-in row selected.
  Confirmed programmatically against rendered page text: `sk-qa-test-redact-0001`
  — 0 occurrences; exactly 3 `***` redaction markers (docs-server's
  `--api-key` arg, weather-api's `env.API_KEY`, weather-api's
  `env_placeholders.API_KEY`); docs-server's non-secret
  `env.WORKSPACE_ROOT: $HOME` correctly left unredacted; git-tools unchanged
  (no secrets).

### Recaptured files (all 8 primary views, same filenames, overwritten in place)

1. `mcp-servers-overview-mixed-2026-07-13.png`
2. `mcp-local-profile-detail-docs-server-2026-07-13.png`
3. `mcp-builtin-server-detail-2026-07-13.png`
4. `mcp-mode-tools-2026-07-13.png`
5. `mcp-mode-permissions-2026-07-13.png`
6. `mcp-mode-audit-2026-07-13.png`
7. `mcp-advanced-external-servers-redacted-2026-07-13.png`
8. `mcp-two-click-inspector-refresh-2026-07-13.png`

`mcp-defect-duplicate-ids-crash-2026-07-13.png` (historical, pre-fix) and
`mcp-advanced-external-servers-2026-07-13.png` (historical, pre-redaction-fix)
were left untouched, as instructed.

### Nothing else observed wrong in this pass

No new defects surfaced while driving all eight views and the two repro
flows. The only anomaly is the pre-existing, already-documented stray
mode-strip focus artifact (see "Observation" above and the A1 write-up),
which now renders with the fixed, visually-distinct focus style rather than
impersonating the active-mode indicator.

## Isolated HOME

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-20260713`** (config +
`local_mcp_store.json` seed data as described above).
