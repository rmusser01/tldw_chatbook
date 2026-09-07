# MCP Hub Phase 3 — QA evidence (2026-07-14)

Branch: `claude/mcp-hub-phase3`, originally captured at HEAD `808ce4d3`
("fix(mcp-hub): reject colon/whitespace-bearing local profile ids at save
time (I3)"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase3`.

**Re-capture round (2026-07-16, branch at `892bf041`):** both defects this
round originally found were since fixed on the branch — Defect 1 by
`d19d1adc` (bundle-layer width rule for the tools filter Select) and
Defect 2 by `4fd1e908` + `892bf041` (built-in tool import cascade, plus a
follow-on fix replacing a nonexistent DB method call). All nine Tools-mode
PNGs were re-taken **in place** (same filenames) against the same two HOMEs
with the fixed code+bundle served fresh: `tools-catalog`,
`tools-filter-text`, `tools-filter-server`, `tools-empty-diagnostic`,
`tool-inspector-detail`, `tool-test-form`, `tool-test-raw`, `tool-test-ok`,
`tool-test-failed`. The two Servers-mode captures (`checking-bound`,
`hugging-layout`) are unaffected by either fix and were left as originally
taken. Per-capture notes below are updated where the re-capture changed
what the PNG shows; the two defect sections at the bottom now carry
"FIXED" verdicts with the verification evidence.

Captured live from textual-serve (real app CSS, worktree code) in headless
bundled Chromium (Playwright, CDP-attached), viewport **2050×1240**. Same
methodology as
[`Docs/superpowers/qa/mcp-hub-phase2-2026-07/README.md`](../mcp-hub-phase2-2026-07/README.md):
DOM-rendered xterm (`.xterm-rows`, real text nodes — `Web_Server/serve.py`'s
own WebGL/Canvas-addon strip means no `window.__drv` patch is needed), plain
Playwright DOM text search + `Range`-based click-coordinate resolution,
route-abort non-localhost, a persistent headless Chromium launched with
`--remote-debugging-port` and attached to across steps via
`connect_over_cdp` from short `python3 -c` snippets, two-click DataTable row
selection.

**This round's deltas from Phase 2:**
- Served on port **9189** (main HOME) and, for one capture only, a second
  instance on port **9190** (minimal empty-store HOME) — Phase 2 used 9187.
- Isolated HOME **`/private/tmp/tldw-qa-mcp-hub-p3-20260714`** is a `cp -R`
  of Phase 2's evidence HOME (`tldw-qa-mcp-hub-p2-20260714`, left untouched
  on disk per instructions), with `local_mcp_store.json`'s `docs-server`
  discovery snapshot extended: `search_docs` got a renderable object schema
  (`query` required string, `max_results` integer default 5, `fuzzy` boolean
  default false, `scope` enum `["docs","api","all"]`), `read_file` got an
  unrenderable schema (nested `options` object property — `parse_schema()`
  fails the whole schema on any unrenderable property), `list_files` was
  left with no `inputSchema` at all (also renders "raw" — falsy schema is
  itself an unrenderable case). All three outcomes confirmed by direct
  `parse_schema()` invocation before capturing.
- A second, minimal HOME **`/private/tmp/tldw-qa-mcp-hub-p3-empty-20260714`**
  (config copied from the main HOME, `local_mcp_store.json` seeded with zero
  profiles) was used briefly on port 9190, purely for capture 4 (see its own
  section below for why a special HOME was necessary).
- `[mcp] enabled = true` and `hub_lifecycle_timeout_seconds = 8` carried over
  unchanged from Phase 2's config (the 8s bound renders as "(up to 8s)").

## Captures

All at 2050×1240, real app CSS, MCP nav destination → Tools mode (chip click
or the `2` keybinding; `MCPScreen.BINDINGS` maps `"2"` to
`action_mcp_mode('tools')`, confirmed by reading `mcp_screen.py`). Files:
`Docs/superpowers/qa/mcp-hub-phase3-2026-07/mcp-p3-<slug>-2026-07-14.png`.

1. **`tools-catalog`** — Tools mode, no filter, no selection. Catalog table
   grouped implicitly by server (sorted by server label then tool name):
   `docs-server`'s 3 tools first (`list_files`/`read_file` "raw",
   `search_docs` "form"), then `tldw_chatbook`'s 10 built-in tools (all
   "raw" — built-in tools never carry an `input_schema`, see
   `hub_tool_catalog.py:149`). `Server` column reads `docs-server (stale)`
   for all three (the profile has no active session this round). Columns:
   `Tool | Server | Tags | Schema`. Verified: DOM text search for
   `search_docs` + `form` and `read_file` + `raw` on the same row both hit.
   **Re-taken 2026-07-16 (post-`d19d1adc`):** the server-filter Select now
   renders visibly at the right end of the filter bar (`All servers ▼`),
   width verified numerically — its full box-frame segment
   (`▊  All servers          ▼  ▎`) measures **202.05px = exactly 28
   terminal cells** (2045px canvas / 284 cols × 28), matching the intended
   `width: 28` rule; the bundled-CSS harness pilot independently reports
   `select.region == Region(x=255, y=0, width=28, height=3)` and
   `styles.width == 28` (was 0×0 / `100w` pre-fix). Rail note:
   `slow-server` shows `!` (NEEDS_ATTENTION) instead of the original
   round's `○` — the documented persisted side effect of capture 10's
   connect-timeout, not a regression. PNG 105KB.

2. **`tools-filter-text`** — typed `search` into `#mcp-tools-filter-text`.
   Table narrows to exactly the 4 tools whose name/description contains
   "search": `search_docs` (docs-server), `search_conversations`,
   `search_notes`, `search_rag` (all tldw_chatbook). Verified via DOM row
   count read straight from the live buffer. **Re-taken 2026-07-16:** same
   4-row narrowing re-verified (column-scoped row count == 4, zero
   non-matching rows), now with the server Select visible alongside the
   text filter. PNG 79KB.

3. **`tools-filter-server`** — **re-taken 2026-07-16 as a working filter**
   (the original 2026-07-14 capture documented Defect 1 — the Select
   rendered at 0×0 and was unclickable; see the FIXED defect section
   below). Post-`d19d1adc`: clicked the Select (prompt "All servers"), its
   overlay opened listing `docs-server` / `tldw_chatbook`; clicked
   `docs-server` via a row-scoped `Range` click inside the overlay. Result:
   the Select displays **docs-server** as its value and the table filters
   to **exactly the 3 docs-server rows** (`list_files`, `read_file`,
   `search_docs`), the 10 built-in rows gone — verified by column-scoped
   row count before capturing. Reset to "All servers" afterwards (full
   13-row catalog re-verified) so later captures see the unfiltered
   catalog. PNG 77KB.

4. **`tools-empty-diagnostic`** — the *true* zero-tool diagnostic state:
   **"No servers configured — add one to see its tools."** with an **"Add
   server"** button. This is **not** reachable by filtering the main HOME
   to zero rows (a text/server filter narrowing to zero matches is just an
   empty table — `_apply_filter()`'s own docstring: the diagnosis is driven
   by whether the *whole* catalog is empty, not the filtered view) — nor is
   it reachable under **Local** source at all, even with zero local
   profiles, because the built-in `tldw_chatbook` server's 10 tools are
   *always* included in the Local-source catalog regardless of `[mcp]
   enabled` (`local_control_service.get_inventory()` reads a static
   manifest unconditionally; confirmed by direct call). The only reachable
   path is **Server** source with zero configured server targets — but even
   an intentionally profile-less HOME auto-seeds one target
   (`http://127.0.0.1:8000`) from `[tldw_api].base_url` into
   `~/.local/share/tldw_cli/default_user/mcp_server_targets.json` the first
   time the MCP screen is visited (`ConfiguredServerTargetStore
   .ensure_legacy_config_target()`-style bootstrap; not itself a bug — a
   reasonable default-target convenience — but it means "no server source
   targets" isn't achievable purely by omitting a targets file). Used the
   second, minimal HOME (`tldw-qa-mcp-hub-p3-empty-20260714`, port 9190):
   blanked `[tldw_api] base_url = ""` and deleted the already-auto-created
   targets file, then opened a **fresh** browser tab/session (textual-serve
   spawns one app instance per session, so a stale session wouldn't have
   re-read the edited config) and switched Source to Server. Result:
   `target_store.list_targets()` empty → `_empty_tools_diagnosis()`'s
   `if not relevant: return ("No servers configured…", "add_server")`
   branch. Verified: DOM text search for the exact message and the "Add
   server" button both hit. **Re-taken 2026-07-16:** the filter bar renders
   in the empty state too, and the Select is now visible in it (same
   202.05px = 28-cell segment measurement as capture 1, re-measured in this
   state at capture time); the diagnostic message, Add-server button, and
   filter Input all re-verified via DOM text search. The blanked
   `[tldw_api].base_url` held — no `mcp_server_targets.json` was re-seeded
   on the fresh session. PNG 75KB.

5. **`tool-inspector-detail`** — `search_docs` row selected (two clicks,
   docs-server). Inspector's tool-detail block: `search_docs —
   docs-server` name/server line, description ("Search the docs tree for
   matching content."), `Tags: —`, `Parameters: form`, "Stale — not
   currently connected." (docs-server has no active session), `Test Tool`
   button. Verified via DOM text search for all five lines. **Re-taken
   2026-07-16:** all five lines re-verified, now with the visible filter
   Select above the table. PNG 110KB.

6. **`tool-test-form`** — Test Tool pressed for `search_docs`. Schema-driven
   form rendered exactly per the seeded schema: `query *` label (required
   star) + Input placeholder "Search query", `max_results` label + Input
   pre-filled `5` (the schema's `default`), `fuzzy` label + unchecked
   Checkbox, `scope` label + `Select` with prompt "Select" (optional enum,
   no default), `Run`/`Close` buttons. A lingering hover tooltip
   ("Run this tool with test arguments.", the Test Tool button's own
   tooltip, left over from the click that opened this panel) briefly
   obscured the `query *` label in the first dump — moved the mouse to a
   neutral screen corner and re-verified the label text was present in the
   DOM before capturing. Verified: DOM text search for `query *`,
   `max_results`, `fuzzy`, `scope` all hit. **Re-taken 2026-07-16:** all
   seven field/button texts (`query *`, `Search query`, `max_results`,
   `fuzzy`, `scope`, `Run`, `Close`) re-verified via DOM text search before
   capturing. PNG 121KB.

7. **`tool-test-raw`** — `read_file` selected instead, Test Tool pressed.
   `Parameters: raw JSON` in the detail block, then the raw-fallback panel:
   note **"This tool's parameters can't be rendered as a form — edit raw
   JSON."** + a `{}` TextArea + `Run`/`Close`. Matches `read_file`'s seeded
   schema being unrenderable (nested `options` object property fails
   `parse_schema()`'s "no nested object/array" rule, the whole schema falls
   back to raw per that function's documented all-or-nothing contract).
   Verified: DOM text search for the exact fallback note. **Re-taken
   2026-07-16:** fallback note, `Parameters: raw JSON`, and the `{}`
   TextArea all re-verified. PNG 117KB.

8. **`tool-test-ok`** — the task brief's suggested "datetime tool" does not
   exist in this hub's built-in registry
   (`describe_local_mcp_capabilities()`'s actual 10 tools are
   `chat_with_llm, chat_with_character, search_rag, search_conversations,
   create_note, search_notes, list_characters, get_conversation_history,
   export_conversation, ingest_media`; "DateTimeTool"/"CalculatorTool" are a
   *different* system — the agent tool-calling registry used by Chat,
   `Tools/tool_executor.py` — not exposed through the MCP hub at all).
   The original 2026-07-14 round additionally hit Defect 2 (see below) on
   `list_characters` — every DB-touching built-in tool was broken — and had
   to fall back to the DB-free `ingest_media` (`OK · 3ms`, placeholder
   payload). **Re-taken 2026-07-16 (post-`4fd1e908`+`892bf041`) with
   `list_characters`, exercising the real DB path end to end:** raw-JSON
   args `{}`, Run → **`OK · 10ms`** +
   `{"source": "local", "tool_name": "list_characters", "result": [{"id":
   1, "name": "Default Assistant", "description": "A general-purpose
   assistant.", "message_count": 0}], "governance": {...}}` — a real
   character row read from the seeded ChaChaNotes DB, not a placeholder.
   Verified: DOM text search for `OK · 10ms` and `Default Assistant`.
   PNG 142KB. (Driver note: the first Run press this round was swallowed —
   likely landed while the panel's own tooltip overlay still hovered the
   button — the result Static stayed empty for >5s with no OK/Failed; a
   second, rect-resolved Run click executed normally. Not reproducible as
   a product defect — the same press pattern works on every other run —
   filed as a driver flake, not an app issue. Side effect, expected: run
   persisted to `runtime_activity` in `local_mcp_store.json`.)

9. **`tool-test-failed`** — `search_docs` on docs-server, form filled
   `query = "driver test"` (defaults left for the rest), Run. docs-server's
   `command` is `npx -y @modelcontextprotocol/server-filesystem
   /Users/qa/docs`, which fails to spawn/connect in this sandbox — result:
   original round **`Failed · 3357ms`**, re-capture **`Failed · 681ms`**
   (**re-taken 2026-07-16**), both + **"Failed to connect profile:
   docs-server"** (well under the 8s `hub_lifecycle_timeout_seconds` bound —
   it fails fast here rather than timing out; see the `checking-bound`
   capture's own note about this same server resolving too quickly to catch
   mid-flight). Verified: DOM text search for the exact failure line both
   rounds. PNG 126KB.

10. **`checking-bound`** — **could not use docs-server** as the task brief
    suggested: repeated attempts (both "Refresh tools", its actual wired
    lifecycle action — docs-server's `DISCOVERY_FAILED` reason has no
    literal "Connect" action, only `slow-server`/`weather-api`'s
    `NEEDS_SETUP` reason does) resolved to "Failed to connect profile:
    docs-server" in under 60ms every time (confirmed: a click-then-dump at
    150ms, then again at 60ms, both already showed the failure — the
    in-flight CHECKING window is sub-frame here, not a capturable state).
    Switched to **`slow-server`** instead (`python3 -c "import time;
    time.sleep(120)"` — the same substitution Phase 2's own README made for
    the identical reason). Servers mode, `slow-server` selected, `Connect`
    clicked, captured ~0.5s later: breadcrumb/rail/table show `◐ Checking`,
    canvas reads **"Working — connect (up to 8s)…"** (the
    `hub_lifecycle_timeout_seconds` bound baked into the message, read
    straight from config per `_display_snapshot()`'s docstring), Inspector
    shows the same badge/message and a single **`Cancel`** button.
    Verified: DOM text search for `◐ Checking`, `Working — connect (up to
    8s)…`, and `Cancel` all hit; PNG 79KB. Left to resolve on its own after
    capture (attempting `Cancel` a moment later missed the window — it had
    already timed out server-side, leaving `slow-server` in
    `NEEDS_ATTENTION` / "Timed out after 8s", an expected, harmless final
    state for this profile).

11. **`hugging-layout`** — Servers mode overview, Local source, no
    selection (captured incidentally while switching back from Tools mode
    for capture 10, before selecting `slow-server`). Table hugs its 4 rows
    (`tldw_chatbook (built-in)`, `docs-server`, `weather-api`,
    `slow-server`) with the three one-line recovery callouts immediately
    beneath, no dead space — matches Phase 2's own "Task 7 layout" note
    (the `#mcp-servers-table { height: auto; max-height: 70%; }` fix).
    Verified: DOM text search for the aggregate line and all three callout
    lines; PNG 97KB.

12. **`non-executable-note`** — **skipped**, per the task brief's own
    fallback instruction. Server-source tool testing requires a live
    `tldw_server` backend (Phase 4 scope per `hub_tool_catalog.py`'s own
    docstring: `server_tools_from_inventory()` sets `executable=False`
    unconditionally, "server-source execution ships in Phase 4"); no such
    backend is available in this sandbox, and faking the response would
    misrepresent an unimplemented code path. Not captured.

## Defects / observations found (documented at capture time; both defects since FIXED on the branch — see each section's verdict)

### Defect 1 (High, functional) — Tools mode server-filter Select renders at 0×0, completely invisible and unusable — FIXED (`d19d1adc`)

**Reproduction:** Navigate to MCP → Tools mode with more than one server
present in the catalog (so the filter Select would have real options). The
`Select` immediately right of the "Filter tools…" `Input` never appears —
the Input's own box border runs to the canvas panel's right edge with no
gap for it. Fully deterministic, not a race.

**Root cause:** `MCPToolsMode.DEFAULT_CSS` (`tldw_chatbook/UI/MCP_Modules/
mcp_tools_mode.py:84-86`) sets `#mcp-tools-filter-server-slot Select {
width: 28; }`, intending to give the dynamically-mounted server-filter
`Select` a fixed 28-column width inside its `width: auto` slot container so
the sibling `Input` (`width: 1fr`) only claims the remaining space. This
rule is correctly parsed and registered (confirmed via direct
`Stylesheet.rules` inspection) with the highest selector specificity of any
rule touching `Select` in the whole app (`(1, 0, 1)` — one ID, one type —
versus every competing rule's `(0, 0, *)`). It still loses, because
`tldw_chatbook/css/features/_conversations.tcss:344-348` declares a
**global, unscoped** rule:

```
/* Select widget styling */
Select {
    width: 100%;
    margin-bottom: 1;
}
```

(compiled into the app-wide bundle, `tldw_cli_modular.tcss`). Textual's
cascade treats **every** rule sourced from the app's `CSS_PATH` stylesheet
as unconditionally higher-priority than **any** widget's own `DEFAULT_CSS`,
regardless of selector specificity (`Stylesheet.apply()`'s
`is_default_rules` flag — confirmed by direct source read of
`textual/css/stylesheet.py`). So this bare, low-specificity `Select {
width: 100%; }` — clearly intended for a completely different screen's
sidebar forms (the file is `_conversations.tcss`) — silently wins over
`MCPToolsMode`'s own, far more specific override, no matter how the latter
is written. The Select then computes a genuine 100% width against its own
`width: auto` parent slot, which (per Textual's fr/auto sizing order, with
the sibling `Input` at `width: 1fr` already claiming the full bar) resolves
the whole thing to zero.

**Verified three independent ways:**
1. **Live DOM** (real running app, this round's main HOME): searched every
   row of the rendered Tools-mode filter bar for `All servers` or any
   server label — no match anywhere on screen.
2. **Harness pilot** (`App(CSS_PATH=tldw_cli_modular.tcss)`, real bundled
   CSS, `MCPToolsMode` mounted standalone at the served terminal's actual
   284×73 cell geometry): `select.region == Region(x=283, y=0, width=0,
   height=0)`, `select.styles.width == Scalar(100.0, WIDTH_FRACTION...)`
   ("100w") instead of the intended `28`. Re-running the *same* pilot
   **without** `CSS_PATH` (DEFAULT_CSS alone) produces the correct `region
   == Region(x=256, y=0, width=28, height=3)` — isolates the bundle as the
   sole cause.
3. **Live click test:** computed the Select's expected screen coordinates
   from the live DOM and issued a real mouse click there — no dropdown
   opened, no DOM change at all (before/after dumps identical), confirming
   it is genuinely unclickable, not just visually thin.

**Impact:** Users cannot filter the Tools catalog by server through the UI
at all — the entire per-server filtering feature (capture item 3's whole
purpose) is unreachable. This is the same defect *class* as Phase 2's
Defect 2 (`Button.mcp-callout` losing `content-align` to a Textual
default) but the inverse direction and a different mechanism: there, a
project rule was missing a property Textual's own default supplied; here, a
different project screen's own bundled rule (not a Textual default at all)
unconditionally clobbers this screen's override because of how Textual
weighs `DEFAULT_CSS` vs. `CSS_PATH` rules, independent of specificity.

**FIXED** by commit `d19d1adc` ("fix(mcp-hub): bundle-layer rules for tools
filter select"): a scoped copy of the `#mcp-tools-filter-server-slot Select
{ width: 28; }` rule was added to the bundle-source component file that
already carries the other MCP hub lockstep rules
(`_agentic_terminal.tcss`), and the bundle regenerated — exactly the
follow-up shape this section originally recommended. Regression coverage
was added in `Tests/UI/test_mcp_tools_mode.py` (rule-presence assertion +
a bundled-CSS harness asserting non-zero computed geometry, confirmed to
reproduce the 0×0 collapse pre-fix per the commit message).
**Re-verified at re-capture time (2026-07-16, branch `892bf041`), both
numerically and interactively:**
- Bundled-CSS harness pilot (same 284×73 geometry as the served terminal):
  `select.region == Region(x=255, y=0, width=28, height=3)`,
  `styles.width == 28` — was `Region(x=283, y=0, width=0, height=0)` /
  `100w` pre-fix.
- Live DOM: the Select's full on-screen box segment
  (`▊  All servers          ▼  ▎`) measures **202.05px = exactly 28 cells**
  (2045px / 284 cols × 28).
- Live interaction: clicking the Select opens its overlay
  (docs-server / tldw_chatbook options); selecting `docs-server` filters
  the table to exactly its 3 tools and the Select displays the chosen
  value; selecting "All servers" restores the full 13-row catalog — the
  whole per-server filter feature now works end to end
  (`mcp-p3-tools-filter-server-2026-07-14.png` re-taken as the working
  filter).

### Defect 2 (High, functional) — most built-in MCP tools crash with an ImportError when actually executed — FIXED (`4fd1e908` + `892bf041`)

**Reproduction:** MCP Hub → Tools mode → select any of `chat_with_character`,
`search_conversations`, `create_note`, `search_notes`, `list_characters`,
`get_conversation_history`, or `export_conversation` (7 of the 10 built-in
tools) → Test Tool → Run (any arguments, including none). Every one fails
immediately (~5ms) with:

```
Failed · 5ms
cannot import name 'ChaChaNotes_DB' from 'tldw_chatbook.DB.ChaChaNotes_DB'
(/Users/.../tldw_chatbook/DB/ChaChaNotes_DB.py)
```

**Root cause:** `tldw_chatbook/MCP/tools.py:17`, `tldw_chatbook/MCP/
resources.py:16`, `tldw_chatbook/MCP/prompts.py:13`, and `tldw_chatbook/
MCP/server.py:130` all do:

```python
from ..DB.ChaChaNotes_DB import ChaChaNotes_DB
```

but `tldw_chatbook/DB/ChaChaNotes_DB.py` defines the database class as
**`CharactersRAGDB`** (`class CharactersRAGDB:` at line 117) — there is no
class, function, or alias named `ChaChaNotes_DB` anywhere in that module
(confirmed via `grep -n "^class "` across the file). Any code path that
imports `MCPTools`/`MCPResources`/`MCPPrompts` (which all construct via
this broken import at module load, per `local_runtime_delegate.py`'s
lazily-imported `_get_tools()`) raises `ImportError` the first time it's
actually touched.

**Verified live:** ran `list_characters` (no required args) via the Hub's
Test Tool panel — `Failed · 5ms` with the exact ImportError text above,
captured and confirmed via DOM text search before switching to a
DB-independent tool (`ingest_media`) for capture 8.

**Scope of impact:** every built-in tool whose implementation reaches
`MCPTools`, `MCPResources`, or `MCPPrompts` is currently non-functional
when invoked through the Hub's Test Tool runner (and, by the same import
path, through the real stdio MCP server client entry point too — this is
not specific to the QA harness). Only tools that avoid this import path
work: `ingest_media` (verified OK, no DB access), and `chat_with_llm`
(would fail differently — `local_runtime_delegate.py`'s own
`_tool_chat_with_llm()` unconditionally raises `RuntimeError("...not
available through the direct local runtime delegate yet.")`, a separate,
already-documented stub, not this bug). `search_rag` was not tested this
round.

**FIXED** by commit `4fd1e908` ("fix(mcp): correct ChaChaNotes_DB imports
in built-in MCP server modules" — the four call sites now import
`CharactersRAGDB`, the first of the two fix shapes this section originally
suggested), plus follow-on `892bf041` ("fix(mcp): replace nonexistent
get_conversation_messages calls with get_messages_for_conversation" — a
second latent nonexistent-name bug in the same modules that the ImportError
had been masking).
**Re-verified at re-capture time (2026-07-16, branch `892bf041`), live and
end to end:** `list_characters` (the exact tool that reproduced the
original failure) run through the Hub's Test Tool panel now returns
**`OK · 10ms`** with a *real* row read from the seeded ChaChaNotes DB —
`"result": [{"id": 1, "name": "Default Assistant", "description": "A
general-purpose assistant.", "message_count": 0}]` — captured as the new
`mcp-p3-tool-test-ok-2026-07-14.png` (replacing the original round's
`ingest_media` placeholder-path workaround). Other DB-touching tools were
not individually re-run this round; they share the identical import path,
and `list_characters` exercises `_get_tools()` → `MCPTools` construction →
a real DB query, the full previously-broken chain.

### Observations (not filed as defects)

- **Task brief's "datetime tool" doesn't exist in this registry.** See
  capture 8's note — the MCP hub's built-in tool set has no datetime tool;
  that name belongs to a separate, unrelated tool-calling system
  (`Tools/tool_executor.py`, used by Chat's own tool-calling feature, not
  exposed through MCP at all).
- **docs-server's connect/refresh resolves in well under 100ms**, not
  anywhere near the 8s `hub_lifecycle_timeout_seconds` bound — its `npx`
  command fails to spawn/connect essentially immediately in this sandbox.
  This makes it unusable for capturing an in-flight CHECKING state (capture
  10 had to substitute `slow-server`, same as Phase 2). Not a product bug —
  a QA-environment characteristic (no real filesystem-MCP npm package
  reachable) worth noting for future rounds.
- **An intentionally "empty" HOME isn't actually server-target-free by
  default.** See capture 4's note — the app auto-seeds one server target
  from `[tldw_api].base_url` into `~/.local/share/tldw_cli/default_user/
  mcp_server_targets.json` (a *different* path than
  `ConfiguredServerTargetStore`'s own hardcoded default,
  `~/.config/tldw_cli/mcp_server_targets.json` — confirmed by direct
  `find`) the first time the MCP screen is visited in a session. A
  reasonable convenience default, not a bug, but worth knowing for anyone
  trying to reproduce a true zero-target state: blank `[tldw_api].base_url`
  *and* delete the file *and* start a fresh session (config is read once
  per textual-serve session, not re-read on navigation).
- **"Server" is a three-way substring collision this round**, worse than
  Phase 2's own documented two-way version: the literal text "Server"
  appears in (a) the mode-strip's "Servers" chip (row 5, always present),
  (b) the rail's "Servers" section heading, and (c) the Source dropdown's
  actual "Server" option — a naive first-match click reliably lands on (a),
  switching MODE instead of Source (confirmed: this happened once during
  this round's driving, silently reverting to Servers mode with Source
  still "Local"). Resolved the same way Phase 2 did: read a fresh DOM dump
  while the dropdown was open, located the exact row containing the
  dropdown's own box-drawn option (`│  Server` inside the overlay's border
  characters, distinguishable from both other matches), and clicked a
  `Range` computed strictly within that row element.
- Everything else driven this round (catalog grouping/columns, text
  filtering, tool inspector detail rendering, schema-form field
  types/defaults/required-star, raw-JSON fallback note, OK/Failed result
  rendering and timing, CHECKING badge/message/Cancel action, overview
  table+callout layout) matched Phase 2/3's spec exactly with no further
  defects observed.

## Isolated HOMEs

Left on disk per instructions:
- **`/private/tmp/tldw-qa-mcp-hub-p3-20260714`** — main HOME (port 9189).
  Final `local_mcp_store.json` state: `docs-server`/`weather-api` unchanged
  from the seed; `slow-server`'s `profile_runtime_state` now reflects the
  capture-10 connect attempt (`ok: false`, `last_error: "Timed out after
  8s"`, readiness now `NEEDS_ATTENTION` instead of the original
  `NEEDS_SETUP`) — an expected, harmless side effect of driving that
  capture, not reverted (mirrors Phase 2's own practice of leaving
  capture-induced state as the final on-disk record). One
  `runtime_activity` entry was also persisted from capture 8's `ingest_media`
  run. The 2026-07-16 re-capture round appended further expected
  `runtime_activity`/`profile_runtime_state` entries (the `list_characters`
  OK run, the re-run `search_docs` failure, and the original round's
  DB-broken `list_characters` attempt).
- **`/private/tmp/tldw-qa-mcp-hub-p3-empty-20260714`** — minimal HOME (port
  9190), used only for capture 4 (both rounds). `[tldw_api].base_url` is
  blanked (`""`) and `mcp_server_targets.json` deleted, by design, to keep
  it reproducing the true zero-server-target empty state on a fresh
  session — re-confirmed at the 2026-07-16 re-capture: no targets file was
  re-seeded.

Both `run_web_server` processes and the driver's persistent headless
Chromium were killed at the end of each QA round (original and
re-capture).
