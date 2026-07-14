# MCP Hub Phase 2 — QA evidence (2026-07-14)

Branch: `claude/mcp-hub-phase2`, HEAD `aa2cddcd` ("fix(mcp-hub): surface args
secret-lint warning as toast on save success"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase2`.

Captured live from textual-serve (real app CSS, worktree code) in headless
bundled Chromium (Playwright, CDP-attached), viewport **2050×1240**, isolated
HOME **`/private/tmp/tldw-qa-mcp-hub-p2-20260714`** (left on disk — see
bottom). No live model/provider needed; all data is local (seeded
`local_mcp_store.json` + `config.toml`).

## Driver notes (methodology, not app defects)

- Same `--serve` dead-code trap as Phase 1: served via
  `HOME=<isolated> PYTHONPATH=. python3 -c "from
  tldw_chatbook.Web_Server.serve import run_web_server;
  run_web_server(host='127.0.0.1', port=9187)"` directly, from this worktree.
- textual-serve's `/static/js/textual.js` route is already patched by the
  app's own `Web_Server/serve.py::patch_textual_serve_viewport_js()` to strip
  xterm's WebGL/Canvas renderer addons (`this.webglAddon=null,
  this.canvasAddon=null`), so the served page uses xterm's default **DOM**
  renderer (`.xterm-rows` with real text nodes). That made it possible to
  drive this round with plain Playwright DOM text search + `Range`-based
  click-coordinate resolution — no `window.__drv` xterm-buffer patch needed
  this time (unlike Phase 1's A-batch round, which needed it for cell-level
  style-attribute reads; this round only needed text + pixels).
- **DataTable rows need two clicks to select**, not one: Textual's
  `DataTable._on_click()` only posts `RowSelected` when the click coordinate
  equals the *already-highlighted* cell (`highlight_click = new_coordinate ==
  self.cursor_coordinate`) — the first click on an unfocused row only moves
  the cursor there. Every row-selection action in this round's driver clicks
  twice.
- **Input/TextArea placeholders can look like real values in a text-only
  buffer dump.** The Add-server form's `Input(value="", placeholder="docs-
  server")` renders `docs-server` in the exact screen position a filled
  value would use; first pass of this round misread the empty Add form as
  pre-filled with `docs-server`/`npx` before realizing those are
  `mcp_profile_form.py`'s own placeholder strings (`id_input = Input(...,
  placeholder="docs-server")`), not stale state. No bug — a QA-driver gotcha
  worth flagging for the next round.
- **Textual `TextArea`/`Input` "select all" keys are not `Ctrl+A`.**
  `Ctrl+A`/`Home` both map to `cursor_line_start` in both widgets; the actual
  select-all bindings are `F7` (`TextArea.action_select_all`) and
  `Ctrl+Shift+A` (`Input.action_select_all`, not used here). Used `F7` +
  retype to replace TextArea contents; used `End` + repeated `Backspace` for
  `Input` fields.
- **`Select` (dropdown) options must be clicked precisely inside the open
  overlay**, not via a generic text search for the option label — "Server"
  is a substring of the ever-present "Servers" rail-section heading, so a
  naive first-match click landed on that inert `Static` instead of the
  dropdown's "Server" option. Resolved by clicking the option row's exact
  screen coordinates read from a fresh DOM dump taken while the overlay was
  open.

## Data seeding

- `~/.config/tldw_cli/config.toml`: full default `CONFIG_TOML_CONTENT` with
  `[mcp]` `enabled = true` and `hub_lifecycle_timeout_seconds = 8` (short
  timeout so lifecycle failures resolve fast), and `[splash_screen]`
  `enabled = false` (skip the startup animation for faster, deterministic
  captures — cosmetic-only, not part of the MCP surface).
- `~/.local/share/tldw_cli/default_user/local_mcp_store.json`: 3 external
  profiles, chosen per the task brief for distinct `readiness.py` outcomes:
  - `docs-server` — `discovery_snapshot` present (3 tools: `list_files`,
    `read_file`, `search_docs`; 0 resources/prompts) **and** a
    `profile_runtime_state` entry (`{"ok": false, "last_error": "Timed out
    after 45s", "last_action": "connect", "last_attempt_at":
    "2026-07-14T00:00:00Z", "last_ok_at": null}`). Per `readiness.py`'s
    `local_profile_readiness()`, a recorded failed attempt (`runtime_state
    .ok is False`) takes precedence over the generic "discovered but not
    connected" signal — reason resolves to `DISCOVERY_FAILED` →
    **NEEDS_ATTENTION**, message = the stored `last_error` verbatim
    ("Timed out after 45s"). (The task brief's own item (4) — "→ NEEDS
    ATTENTION with stored error" — is what actually renders; item (1)'s "→
    STALE" describes what the *discovery_snapshot alone* would produce
    without the runtime-state entry layered on top. Confirmed by direct
    `readiness.py` derivation before capturing — see below.)
  - `weather-api` — `env_placeholders: {"API_KEY": "$QA_MISSING_KEY"}`,
    `$QA_MISSING_KEY` never set in the process environment, no discovery
    snapshot → **NEEDS_SETUP** (`auth_missing` wins priority over
    `discovery_not_run`), "Missing environment variables: QA_MISSING_KEY.",
    Auth column "1 env var".
  - `slow-server` — command `python3 -c "import time; time.sleep(120)"`, no
    discovery snapshot, no runtime state → **NEEDS_SETUP**
    (`discovery_not_run`), actions include `Connect` — used to catch the
    in-flight `CHECKING` state (the 120 s sleep guarantees it stays pending
    past the 8 s `hub_lifecycle_timeout_seconds`, which is what eventually
    fails it out from under a real capture attempt if you're not fast).
  - Plus the built-in `tldw_chatbook (built-in)` row → **READY** (from
    `[mcp].enabled = true`).
  - Verified directly against `readiness.py` before capturing:

```
local:docs-server   -> needs_attention (discovery_failed)  Timed out after 45s
local:weather-api   -> needs_setup (auth_missing, discovery_not_run)  Missing environment variables: QA_MISSING_KEY.
local:slow-server   -> needs_setup (discovery_not_run)  Not validated yet — connect or test to discover tools.
builtin:tldw_chatbook -> ready ()  Served over stdio when an MCP client launches chatbook.
aggregate: 1 of 4 servers ready — 1 needs attention, 2 needs setup.
hub_lifecycle_timeout_seconds: 8
```

## Captures

All at 2050×1240, real app CSS, MCP nav destination (`nav-mcp` → route
`mcp`). Files: `Docs/superpowers/qa/mcp-hub-phase2-2026-07/mcp-p2-<slug>-
2026-07-14.png`.

1. **`overview-colored`** — Servers overview (Local source, no
   selection). Aggregate line "1 of 4 servers ready — 1 needs attention, 2
   needs setup." rendered in the aggregate's severity color (pink/red, since
   `worst_state()` picks NEEDS_ATTENTION); table has 4 rows with colored
   status text/glyphs (● green Ready, ! pink/red Needs attention, ○ orange
   Needs setup ×2) mirrored in the left rail's server list; **no Scope
   column** in the table header (`Name Transport Status Tools Auth` only —
   `_TABLE_COLUMNS_NO_SCOPE` for local source); weather-api's Auth column
   reads "1 env var"; 3 compact one-line actionable callouts under the table
   (one per non-ready server: `! docs-server: Timed out after 45s`, `○
   weather-api: Missing environment variables: QA_MISSING_KEY.`, `○
   slow-server: Not validated yet — connect or test to discover tools.`).
   Look for: badge colors matching state, no Scope column, "1 env var"
   copy, callouts present only for non-ready rows.

2. **`detail-breadcrumb`** — docs-server detail (two clicks on the rail
   row — see DataTable two-click note above). Shows `← All servers  !
   Needs attention  docs-server` breadcrumb, detail toolbar (`Edit` /
   `Delete`), canvas body leads with the stored error **"Timed out after
   45s"** verbatim, then `Command · npx -y @modelcontextprotocol/server-
   filesystem /Users/qa/docs` (no secrets in this command, so nothing to
   redact), `Tools · 3: list_files, read_file, search_docs`, `Resources ·
   0`, `Prompts · 0`. Inspector mirrors the state badge and reads `Why ·
   Tool discovery failed` (the humanized `REASON_LABELS[DISCOVERY_FAILED]`
   phrase, distinct from the canvas's own stored-error message — matches
   Phase 1's A5 "Why line is not a repeat of the canvas message" contract),
   action buttons `Refresh tools` / `View details` (`REASON_TO_ACTIONS
   [DISCOVERY_FAILED]`). Look for: breadcrumb, Edit/Delete toolbar, stored
   error text in both canvas and inspector Why line.

3. **`add-form`** — Add-server form (Local source, "Add server" button),
   fields filled: Profile id `notes-server`, Command `npx`, Args `-y` /
   `@example/notes-mcp`, Env `TOKEN=$MY_TOKEN`. The secrets guidance line is
   visible above the Env box: **"Secrets are never stored — reference them
   as KEY=$ENV_VAR and export the variable before connecting."** Not saved
   (this capture only demonstrates the empty→filled form). Look for: title
   "Add local server (stdio)", all 4 fields populated, guidance copy
   present and legible.

4. **`form-error`** — Same form, Profile id `bad-server`, Env line
   `API_KEY=hunter2secret`, Save clicked. The in-form `#mcp-form-error`
   Static renders the store's own validation rejection verbatim:
   **"Secret-bearing env key 'API_KEY' cannot be stored as a literal"**
   (from `local_store.py`'s `_sanitize_env_literals()` — `API_KEY` matches
   `_SECRET_KEY_PATTERN`, so a literal is refused outright regardless of its
   value; only `$ENV_VAR`-style placeholders are accepted for secret-shaped
   keys). Save button re-enabled for retry. Look for: exact rejection
   sentence rendered in the form's error Static (red), Save button not
   stuck disabled.

5. **`secret-lint-toast`** — Same form flow, profile id
   `cli-secret-server-3`, args `-y` / `@example/tool` / `--api-key` /
   `sk-test-1234567890`, Save clicked; captured with minimal delay so both
   toasts are visible stacked bottom-right: green **"Saved cli-secret-
   server-3."** success toast, and an orange **"Warning: args line 4 looks
   like a secret — it will be visible in process listings; pass it via env
   ($VAR) instead."** warning toast (`MCPProfileForm._args_secret_warning()`
   → surfaced as a toast by `MCPWorkbench._save_local_profile()`'s `warning`
   re-notify, the exact fix this HEAD's own commit — `aa2cddcd` — added).
   **Methodology note, not a defect:** the task brief's literal example
   value `sk-test-123` does **not** actually match
   `_looks_like_raw_secret_value()`'s `^sk-[A-Za-z0-9._-]{12,}$` pattern (only
   8 chars follow `sk-`, the pattern requires 12+), so it would silently
   fail to trigger the warning if used as-is — confirmed by direct regex
   test before capturing. Used `sk-test-1234567890` (15 chars after `sk-`)
   instead, which does match, to actually exercise the capture's intent.
   Not filing this as a product defect (a 12-char floor is a reasonable
   anti-false-positive threshold for a heuristic lint), but worth noting
   since short fake/test-shaped keys like the brief's example won't trip it.
   Look for: both toasts present together, warning names the correct line
   number (args line 4: `-y`=1, `@example/tool`=2, `--api-key`=3,
   `sk-test-1234567890`=4).

6. **`import-preview`** — Import panel (Local source, "Import…" button),
   pasted `{"mcpServers": {"search-server": {"command": "npx", "args":
   ["-y", "search-mcp"], "env": {"API_KEY": "sk-live-abcdef1234567890"}}}}`,
   Preview clicked. Preview list shows `search-server — npx` with the
   placeholder-conversion warning: **"API_KEY: value can't be stored — will
   reference $API_KEY; export it before connecting."** (`parse_mcp_servers
   _json()`'s literal-not-storable fallback path — `API_KEY` fails
   `_literal_is_storable()`'s round-trip through the store's own validation,
   same secret-key-pattern rejection as capture 4, so it's silently
   converted to a `$API_KEY` placeholder candidate instead of being
   imported as a raw literal). "Import 1 server" / "Cancel" buttons present;
   **not actually imported** — cancelled after the capture to avoid
   polluting the seed data. Look for: guidance copy ("Secret-shaped values
   are never imported as literals..."), pasted JSON visible in the source
   TextArea, warning text under the candidate naming the placeholder it
   will use.

7. **`lifecycle-checking`** — slow-server detail, Connect clicked,
   captured ~0.3–0.4 s later (before the 8 s `hub_lifecycle_timeout_seconds`
   elapses — this state is genuinely transient and the driver has to be
   fast; a first attempt at this capture landed after the timeout had
   already fired and had to be redone). Breadcrumb/rail/table all show
   `◐ Checking` in the state-info color; canvas reads "Working —
   connect…"; Inspector shows the same badge/message and a single
   **`Cancel`** button (`REASON_TO_ACTIONS` is bypassed entirely while
   `_in_flight` holds the key — `_display_snapshot()` overlays `as_checking
   ()` regardless of the underlying reason). Look for: CHECKING badge
   (orange/info color, not the alarm red used for NEEDS_ATTENTION), single
   Cancel action, "Working — connect…" message.

8. **`delete-armed`** — docs-server detail, Delete clicked once (arms,
   does not delete). Detail toolbar swaps to **`Confirm delete`** /
   **`Keep`** (the task brief called this the "Keep pair" informally; the
   actual button label is just `Keep` — `mcp_servers_mode.py`'s
   `_detail_toolbar_widgets()`, id `mcp-detail-delete-cancel`). Rest of the
   detail canvas (stored error, command, tool counts) and Inspector
   unchanged underneath. Clicked `Keep` immediately after this capture to
   disarm without deleting — docs-server is still present in the final seed
   state left on disk. Look for: two-button arm-then-confirm toolbar in
   place of the normal Edit/Delete pair, rest of the pane unaffected.

9. **`builtin-toggles`** — built-in `tldw_chatbook (built-in)` row
   selected. Four checkboxes (`Enabled`, `Expose tools`, `Expose
   resources`, `Expose prompts`), all checked (config default), the note
   **"Applies to the next client launch — the built-in server reads config
   at start."**, and the **"Copy client config"** button — all present,
   directly under the note with no gap between them. (This capture was
   retaken after Defect 1's fix landed — see Defect 1 below; the original
   2026-07-14 01:34 capture showed a large, unintended empty gap between the
   note and the button.)

10. **`advanced-collapsed`** / **`advanced-open-object-label`** — built-in
    server still selected. First capture: Inspector's Advanced disclosure
    collapsed (`▶ Advanced (legacy control plane)`). Second: clicked to
    expand (`▼ Advanced (legacy control plane)`), showing the **"Showing:
    Local control plane"** object label (T12's UX-inputs #1 contract — the
    Advanced pane names the object its content describes), Section select
    on "Overview", the Unified MCP Overview summary (Source/Server/Scope/
    Tools/Resources/Prompts/External server profiles/Governance rules
    counts), Action select "No actions available", Payload (JSON) TextArea,
    "Run Action" button (disabled — no action selected). Look for: object
    label text, disclosure glyph state (▶ vs ▼), Overview summary fields
    populated.

11. **`server-gated`** — Source switched from Local to Server (via the
    rail's Source `Select` — a configured `127.0.0.1:8000` target is
    present from the default seeded config, `display_state: needs_setup`
    since it's never been probed this session). Both overview buttons are
    disabled: **`Add server`** and **`Import…`** render dimmed/gray.
    Hovered `Add server` and captured its native Textual tooltip visibly:
    **"Requires team, org, or system-admin scope."** (`_MUTATIONS_GATED_
    TOOLTIP`, from `_update_add_server_button()` — scope gate takes
    precedence over the no-active-target gate for server source, matching
    the "Personal" scope the rail defaults to). `Import…`'s tooltip was
    **not** captured visually in the same shot (a follow-up hover attempt
    didn't land before time ran out on this round) — verified instead by
    direct source read: `_IMPORT_GATED_TOOLTIP = "Import creates LOCAL
    server profiles — switch Source to Local."`, set unconditionally
    whenever `self._source == "server"` in `_update_import_server_button()`,
    so it is exercised by the same Source-switch this capture already
    performs even though the tooltip pixel wasn't grabbed. Also visible:
    Scope/Scope Entity selects appear in the rail (server-source-only
    fields), and the table **does** show a Scope column here (opposite of
    capture 1 — local source hides it, server source shows it, matching
    `_TABLE_COLUMNS` vs `_TABLE_COLUMNS_NO_SCOPE`). Look for: both buttons
    visually dimmed, Add-server tooltip text, Scope column present.

## Defects / observations found (documented, not fixed, per task scope)

### Defect 1 (P3, cosmetic layout) — built-in detail: large dead gap before "Copy client config" — FIXED

Fixed by the `fix(mcp-hub): give built-in detail toggles container a
bounded height` commit (this same commit also carries this README update
and the re-capture below, so it has no fixed SHA to cite from within
itself — see `git log --oneline -- tldw_chatbook/UI/MCP_Modules/
mcp_servers_mode.py` for the actual hash).

Reproduction: select the built-in `tldw_chatbook (built-in)` server (any
session, this is fully deterministic, not a race). The four expose/enable
checkboxes and the "Applies to the next client launch…" note render
correctly at the top of the detail pane, but the **"Copy client config"**
button then renders roughly 800px further down, near the very bottom of the
2050×1240 viewport — visible in `mcp-p2-builtin-toggles-2026-07-14.png` and
`mcp-p2-advanced-open-object-label-2026-07-14.png` (in the latter, the
"Advanced" collapsible panel and Inspector actions render at their normal
position on the right while the canvas's own copy button sits far below its
neighboring content on the left, with a mostly-empty scroll region between
them).

Root cause: `MCPServersMode.compose()`
(`tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py:214-223`) mounts three
widgets inside `VerticalScroll(id="mcp-detail-scroll")`: the body `Static`,
`Vertical(id="mcp-detail-builtin-toggles")`, then the copy-snippet `Button`.
Every *other* sized container in this same file's `DEFAULT_CSS`
(`#mcp-detail-scroll`, `#mcp-servers-form`, `#mcp-detail-header`) explicitly
sets `height: auto` or `height: 1fr; min-height: 0` to avoid Textual's
container defaults — but **no CSS rule anywhere targets
`#mcp-detail-builtin-toggles`** (confirmed via `grep -rn
"mcp-detail-builtin-toggles"` across `UI/MCP_Modules/*.py` and every
`css/**/*.tcss` file — only the two `.py` references that construct/query
it). Textual's base `Vertical` container (`textual/containers.py`) defaults
to **`height: 1fr`** — an *expanding* container — so this toggles `Vertical`
silently inherits that and stretches to consume all the scroll region's
remaining vertical space, with its four checkboxes + note top-anchored
inside it and the "Copy client config" `Button` (the toggles container's
sibling, not its child) pushed down below that expanded box.

Fix would be a one-line CSS addition (e.g. `#mcp-detail-builtin-toggles {
height: auto; }` in `MCPServersMode.DEFAULT_CSS`), mirroring the pattern
already used for every other container in the same file — not applied at
capture time, per that round's no-fix scope.

Impact: purely cosmetic (the button is still present, clickable, and
functionally correct — confirmed via DOM text search, not just a visual
guess), but on a real terminal window shorter than ~80 rows the button could
scroll out of view or require scrolling to reach, and the huge empty gap
reads as a broken/incomplete panel at a glance.

**FIXED**: added `#mcp-detail-builtin-toggles { height:
auto; min-height: 0; }` in both layers, matching this file's existing
`#mcp-servers-form` lockstep pattern — the baseline copy in
`MCPServersMode.DEFAULT_CSS` (`mcp_servers_mode.py`) and the bundle-source
copy in `tldw_chatbook/css/components/_agentic_terminal.tcss` (rebuilt into
`tldw_cli_modular.tcss`). Regression coverage added:
`Tests/UI/test_mcp_servers_mode.py::
test_builtin_toggles_container_does_not_expand_past_content` asserts the
toggles container renders at content height (`< 12` rows, not the 18 rows
an unbounded `1fr` produced pre-fix in the default 80×24 test harness) and
that the copy button sits within 2 rows of the container's bottom edge;
confirmed red (`18 < 12` failing) against the pre-fix widget and green
after. **Re-captured**: `mcp-p2-builtin-toggles-2026-07-14.png` was
retaken in place against the same isolated HOME
(`/private/tmp/tldw-qa-mcp-hub-p2-20260714`, `local_mcp_store.json`
unchanged) and the same recipe (textual-serve, Playwright bundled Chromium,
2050×1240, non-localhost routes aborted) — the "Copy client config" button
now renders one row (15px) below the "Applies to the next client
launch…" note, i.e. immediately adjacent, with no dead gap. Verified via
DOM `Range` rects (`note` bottom `y=315` → `copy button` top `y=330`) in
addition to the visual diff.
`mcp-p2-advanced-open-object-label-2026-07-14.png` (the other capture that
showed this same gap) was left as originally captured — the built-in
toggles/button region visible in its left-hand canvas is unaffected by
Advanced-panel state and would show the same fix, but that capture's
purpose was the Advanced disclosure, not this pane, so it was not
re-taken.

### Observations (not filed as defects)

- **QA-brief example value doesn't trigger the secret-lint it's meant to
  demonstrate.** See capture 5's note above — `sk-test-123` is 4 characters
  short of `_looks_like_raw_secret_value()`'s 12-character minimum. Not a
  product bug (arguably the right anti-false-positive tradeoff for a
  heuristic), but worth fixing in future QA briefs/fixtures for this
  surface.
- **Import tooltip not visually re-confirmed** (see capture 11) — verified
  by source read instead, per the task's own fallback instruction.
- Everything else driven this round (form validation, secret redaction in
  the primary command line, toast stacking, lifecycle checking/cancel,
  arm-then-confirm delete, built-in toggles' functional behavior, Advanced
  disclosure persistence/object-label, Source-switch gating) matched its
  Phase 1/Phase 2 spec exactly with no further defects observed.

## Isolated HOME

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-p2-20260714`** — final
`local_mcp_store.json` state matches the documented seed exactly (the three
throwaway profiles created while exercising captures 3-6, `cli-secret-
server` / `-2` / `-3`, were deleted from the store file and the Source
selector was switched back to Local before leaving the session, so a fresh
MCP nav on this HOME reproduces capture 1's exact 4-row overview).
