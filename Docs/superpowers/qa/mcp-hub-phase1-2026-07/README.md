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

## Isolated HOME

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-20260713`** (config +
`local_mcp_store.json` seed data as described above).
