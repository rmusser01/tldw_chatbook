# MCP Hub Phase 5 — QA evidence (2026-07-17)

Branch: `claude/mcp-hub-phase5`, HEAD `e0c27697` ("docs(mcp-hub): clarify
invoke()'s docstring ordering after the I1/Minor-5 fixes"). Worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/mcp-hub-phase5`.

**All 10 requested captures are LIVE** — a real Console agent turn, through
a real `ConsoleAgentBridge` → `ConsoleProviderGateway` → real HTTP call to a
scripted local LLM, dispatched through the real `MCPToolProvider` /
permission-store / approval-card round trip. No seeded-JSONL fallback was
needed anywhere in this round.

## Recipe

Same base methodology as
[`Docs/superpowers/qa/mcp-hub-phase4-2026-07/README.md`](../mcp-hub-phase4-2026-07/README.md)
and phase3: `textual-serve` (real app CSS, worktree code), headless bundled
Playwright Chromium, viewport **2050×1240**, DOM-rendered xterm text search
(NBSP-normalized), `Range`-based click-coordinate resolution, route-abort
non-localhost, two-click `DataTable` row selection.

- Served on port **9195** (a fresh port, not reused from prior rounds).
  Started via:
  `HOME=/private/tmp/tldw-qa-mcp-hub-p5-20260717 PYTHONPATH=. /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -c "from tldw_chatbook.Web_Server.serve import run_web_server; run_web_server(host='127.0.0.1', port=9195)"`
  from this worktree. **Do not set `PYTHON_KEYRING_BACKEND=fail`** for this
  invocation — see Driver gotcha 1 below; the shared repo `.venv` at
  `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv` was used (no
  per-worktree venv exists).
- Driver Chromium: bundled Playwright build
  (`~/Library/Caches/ms-playwright/chromium-1208/.../Google Chrome for
  Testing`) launched standalone with `--headless=new
  --remote-debugging-port=9324 --user-data-dir=/private/tmp/tldw-qa-driver-p5/profile
  --window-size=2050,1240`, then driven across steps by re-connecting via
  `playwright.chromium.connect_over_cdp("http://127.0.0.1:9324")` from short
  `python3` invocations of a small reusable driver module,
  `/private/tmp/tldw-qa-driver-p5/driver.py` (actions: `open`, `screenshot`,
  `text`, `find`, `key`, `type`, `click <needle> [occurrence]`, `dclick`,
  `clickxy`, `corner`, `html`). `context.route("**/*", ...)` aborts any
  request whose URL doesn't contain `127.0.0.1`/`localhost` — this allows
  BOTH the textual-serve websocket (`ws://127.0.0.1:9195/ws`) and the fake
  LLM server (`http://127.0.0.1:8899`, called server-side by the Python app
  process, not the browser, so this rule was actually moot for it, but kept
  for methodology consistency with prior rounds).
- Isolated HOME **`/private/tmp/tldw-qa-mcp-hub-p5-20260717`** — a `cp -R`
  of Phase 4's evidence HOME (`/private/tmp/tldw-qa-mcp-hub-p4-20260716`,
  left untouched on disk), giving this round the same `docs-server`
  discovery snapshot (`search_docs`/`read_file`/`list_files`) and the same
  three stale/failing local profiles (`docs-server` connect-failed,
  `weather-api` missing-env-var, `slow-server` 8s-timeout) Phase 3/4 used.
  **`mcp_permissions.json` was reset to Phase 4's own documented ROUND-1
  seed** (Phase 4's HOME had drifted to its own final, re-allowed state by
  the end of that round): `kill_switch: false`, `global_default: "ask"`,
  `local:docs-server` — `search_docs` allow (correct hash
  `33c51126dc2c94c688578cbd67375bc04e780aca84e6dac4331e381a8f2254e7`),
  `read_file` allow with the deliberately WRONG hash `"deadbeef"` (the
  rug-pull case — confirmed still live this round: the app itself
  re-flagged `read_file` with `config_changed: true` on first
  `compose_catalog()` during this round's very first Console send), `list_files`
  deny. No `builtin:tldw_chatbook` entries at all — this is the important
  deviation from Phase 4's *final* state (which had `list_characters`
  pinned to `allow` from a Permissions-mode Space-cycle): this round needed
  `list_characters` to resolve to the pinned **global default (`ask`)**,
  not an explicit override, so the very first agent tool-call would
  genuinely hit the ask-gate and render the approval card instead of
  silently allow-through.
- `[mcp] enabled = true` and `hub_lifecycle_timeout_seconds = 8` carried
  over unchanged. `[console] agent_runtime` was left unset — it defaults to
  `true` in code (`ChatScreen._console_agent_runtime_enabled`,
  `chat_screen.py:2190-2192`).

### Config keys added for the fake-LLM provider

```toml
[chat_defaults]
provider = "custom"
model = "fake-model"

[api_settings.custom]
api_key_env_var = "CUSTOM_API_KEY"
api_key = "dummy-qa-key"
api_url = "http://127.0.0.1:8899"
model = "fake-model"
# (all other keys — temperature, timeout, retries, streaming=false, ... —
# left as the HOME's existing defaults)
```

Why `provider = "custom"` specifically, traced via source read before
touching the browser:
- `Agents/native_tools.py`'s `NATIVE_TOOLS_PROVIDERS` includes
  `"custom-openai-api"` — one of the execution keys that forwards `tools=`
  and returns the raw OpenAI response shape needed for native tool-calls.
- `Chat/console_provider_support.py`'s `resolve_console_provider_identity`:
  the raw config string `"custom"` IS ALREADY the readiness key (no alias
  needed on the way in); `_READINESS_TO_EXECUTION_ALIASES["custom"] =
  "custom-openai-api"` resolves the execution key, and
  `"custom-openai-api" in API_CALL_HANDLERS` confirms it's supported.
- `Chat/provider_readiness.py`'s `KEYLESS_PROVIDER_KEYS` includes
  `"custom"` — no API key is actually *required* for readiness (an
  `api_key` was still set per the task brief's own instruction, and to
  match `LLM_API_Calls_Local.py:chat_with_custom_openai`'s `Authorization:
  Bearer` header path).
- `LLM_Calls/LLM_API_Calls_Local.py:chat_with_custom_openai` reads
  `[api_settings.custom].api_url`/`.model` directly from the process-global
  config and POSTs (non-streaming, since this HOME's `custom` section
  already had `streaming = false`) to `{api_url}/v1/chat/completions`,
  returning `response.json()` **completely unmodified** — the exact raw
  OpenAI-compatible shape a native tool-call round-trip needs.
- `Chat/console_session_settings.py`'s `build_default_console_session_settings`
  derives the Console's *default* session provider/model straight from
  `[chat_defaults]` when no prior session exists — so setting `provider =
  "custom"` / `model = "fake-model"` there made a fresh Console session
  boot already pointed at the fake LLM, no UI provider-picker interaction
  needed.

### `fake_llm_server.py` design

Written to this directory (`fake_llm_server.py`), stdlib-only
(`http.server.ThreadingHTTPServer`), no repo imports. Started with
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python
fake_llm_server.py 8899`.

**The task brief's own MVP description — "first call returns a native
tool-call response, subsequent calls return plain text" — turned out to be
unusable literally**, for two reasons discovered live, before the first
successful capture:

1. **Progressive tool disclosure.** This build's `Agents/tool_catalog.py` /
   `agent_runtime.py` only discloses the full tool catalog directly when
   it's small. This HOME's local-source catalog (10 builtin MCP tools + 3
   docs-server tools) is over `DIRECT_DISCLOSE_THRESHOLD`, so the Console
   instead offers just `find_tools`, `load_tools`, and `spawn_subagent` —
   confirmed live via the fake server's own request log (first attempt,
   before the redesign): `offered=['find_tools', 'load_tools',
   'spawn_subagent']`, no MCP tool anywhere, causing the naive
   "first-call-does-a-tool-call" script to call nothing useful and the turn
   to finish with "No matching MCP tool was offered." The model must call
   `find_tools(query=...)`, then `load_tools(ids=[...])`, and only THEN
   does the real tool (e.g. `mcp__tldw_chatbook__list_characters`) appear
   in a `tools=` list it can call — 3-4 `/v1/chat/completions` round trips
   per turn, not 1.
   - `MCPToolProvider.list_catalog()` (`Agents/mcp_tool_provider.py:206-217`)
     sets a catalog entry's `id` equal to its own llm-facing `name` (the
     same `mcp__tldw_chatbook__list_characters` string tool-calling uses),
     so `load_tools(ids=[...])` needs no separate id-lookup step.
2. **Multi-turn captures.** Capture 3 (session-approval-no-reprompt) needs
   TWO separate fresh turns calling the same tool. A global "first call
   ever" counter can't tell "first call of turn 2" from "some later call of
   turn 1".

**Final decision policy** (stateless — re-derived from the request body on
every call, no server-side session state): resolve a target MCP tool +
arguments from the newest `role: "user"` message via a trigger-phrase table
(`"list the characters"` → `list_characters`, `"search the notes"` →
`search_notes`; this is also how captures 1-3 vs. capture 4 got two
*different* MCP tools in independent "ask" states without touching the
HOME's config between captures — see capture 4's note). Then:

- If the request's LAST message is not `role: "tool"` (a fresh step): call
  the target tool directly if it's already in this request's own `tools=`
  list; else call `find_tools` if that's offered; else (neither offered —
  the kill-switch capture) reply with plain fallback text.
- If the last message IS `role: "tool"`: read the immediately preceding
  assistant message's own `tool_calls[0].function.name` to see which call
  this result answers, and step forward through
  `find_tools → load_tools → <target tool> → final plain-text answer`
  accordingly (full state machine and rationale documented as a comment
  block at the top of `fake_llm_server.py` itself).

Verified empirically live (request log excerpt, capture 1's turn):
```
last_role=user   offered=[find_tools, load_tools, spawn_subagent]                              -> tool_calls find_tools(query="list_characters")
last_role=tool   offered=[find_tools, load_tools, spawn_subagent]                              -> tool_calls load_tools(ids=["mcp__tldw_chatbook__list_characters"])
last_role=tool   offered=[find_tools, load_tools, mcp__tldw_chatbook__list_characters, spawn…] -> tool_calls mcp__tldw_chatbook__list_characters({})
```
(the 4th call, the plain "Done — listed the characters." finisher, only
happens after the human approves/denies via the card — see capture 2.)

## Per-capture notes

All at 2050×1240, real app CSS. Files:
`Docs/superpowers/qa/mcp-hub-phase5-2026-07/mcp-p5-<slug>-2026-07-17.png`.
Console reached via the top nav bar's `Console` label; MCP Hub via `MCP`.

1. **`approval-card-batch`** — LIVE. Sent "List the characters please." (the
   first two identical sends, before the fake server's progressive-
   disclosure redesign landed, finished immediately with "No matching MCP
   tool was offered" — visible in the transcript as the first two
   `User`/`Assistant` pairs; harmless, left in place as an honest record
   rather than starting a fresh conversation). The third send, after the
   redesign, drove the real `find_tools → load_tools → list_characters`
   handshake and rendered the batch approval card: **"Approval required"**,
   row **`tldw_chatbook · list_characters`** with args `{}`, a per-row
   **`Approve once ▾`** Select, and **`Approve all` / `Submit` / `Deny
   all`** buttons; header chip **`Approvals: 1 pending`**. Verified: DOM
   text search for all of `Approval required`, `tldw_chatbook ·
   list_characters`, `Approve once`, `Approve all`, `Submit`, `Deny all`,
   `1 pending` hit. PNG 152KB.

2. **`approval-approved-result`** — LIVE. Clicked `Submit` with the row's
   default decision (`Approve once`). The turn resumed, the tool executed
   for real, and a 4th LLM call produced the final answer. Transcript now
   shows the real tool-result marker
   `⚙ mcp__tldw_chatbook__list_characters → {"source": "local", "tool_name":
   "list_characters", "result": [{"id": 1, "name": "Default Assistant", ...`
   (a genuine ChaChaNotes DB row) followed by **`Assistant  Done — listed
   the characters.`**. Verified: DOM text search for both hit on the live
   transcript. PNG 159KB.

3. **`session-approval-no-reprompt`** — LIVE, two separate fresh turns. Sent
   "List the characters please, session test one." → approval card
   reappeared (a fresh turn, `list_characters` still resolves to the
   pinned global `ask` — no override was ever written); opened the row's
   decision Select (a Textual `Select` — needed an explicit click to open
   the overlay, see driver gotcha 4), picked **`Approve for session`**
   (label wraps across two xterm rows, `"Approve for"` / `"session"` — see
   driver gotcha 5), `Submit`. Turn completed normally. Then sent a
   **second, independent** fresh message, "List the characters please,
   session test two." — DOM text search for `Approval required` came back
   **NOT FOUND** for the whole buffer at the point the turn finished, while
   the transcript shows the full `find_tools → load_tools → list_characters
   → Done` chain completed anyway (all 4 fake-LLM calls fired
   back-to-back with no pause), and the real tool result
   (`"name": "Default Assistant"`) is present — the tool executed directly,
   no re-prompt. PNG 219KB.

4. **`approval-denied-result`** — LIVE, same HOME/session, a **different
   tool** (per the task brief's own fallback: "fresh HOME state or
   different tool") — `search_notes`, triggered by sending "Search the
   notes for onboarding docs please." (matches the fake server's own
   `"search the notes"` trigger phrase). `search_notes` had never been
   touched by captures 1-3 (only `list_characters` gained a session
   approval), so it still resolves to the plain global `ask` default and
   the card reappeared: row `tldw_chatbook · search_notes`, args
   `{"query":"onboarding"}`. Opened the Select, picked **`Deny`** (second
   DOM occurrence of the substring "Deny" — the first is the batch's own
   "Deny all" button; see driver gotcha 6), `Submit`. Transcript shows the
   real refusal: **`⚙ mcp__tldw_chatbook__search_notes → ERROR: blocked by
   MCP permissions (set to Off)`** — the literal
   `DENY_REFUSAL` constant from `Agents/mcp_tool_provider.py:56` — followed
   by the model's own plain "Done — searched the notes." wrap-up reply (the
   model doesn't see WHY it was denied, only the refusal text as its tool
   result; its own reply text is scripted/generic, not itself part of
   what's being verified here). Verified: DOM text search for the exact
   refusal string hit. PNG 236KB.

5. **`console-mcp-inspector-row`** — LIVE, the **blocked variant** (not just
   the ready one — this HOME's `docs-server` profile is already stale
   from Phase 3/4's seed, giving this for free). The Console's right-side
   **Inspector** panel is itself a collapsible that needed a click to
   populate (see driver gotcha 3); once open, the **Tools** section reads
   exactly:
   ```
   Tools: 0 ready
   MCP: 1 server enabled, not connected
   ```
   (`console_display_state._mcp_inspector_row`'s "a stale server still
   contributes tools, so a non-zero not-connected count always wins over
   the ready-count row" branch, `console_display_state.py:63-72` —
   confirmed live, not just by source read). Verified: DOM text search for
   the exact string `MCP: 1 server enabled, not connected` and `Tools: 0
   ready` both hit. A lingering hover-tooltip box ("Ask Library sources
   before sending") from an earlier "Run Library RAG" hover survived into
   this capture, overlapping unrelated "Live work sources" rows well below
   the MCP row — cosmetic, doesn't obscure anything being verified, not
   re-captured (see driver gotcha 7). PNG 358KB.

6. **`audit-executions-agent`** — LIVE. MCP Hub → Audit mode, Executions
   table. Top rows are this round's own live agent-initiated records,
   newest first:
   ```
   2026-07-17 09:42:06  builtin:tldw_chatbook::search_notes     agent  denied    0ms  Blocked
   2026-07-17 09:40:28  builtin:tldw_chatbook::list_characters  agent  approved  0ms  OK
   2026-07-17 09:40:15  builtin:tldw_chatbook::list_characters  agent  approved  1ms  OK
   2026-07-17 09:37:48  builtin:tldw_chatbook::list_characters  agent  approved  8ms  OK
   ```
   (times are UTC — see the Observations section) below which the older
   `test`/`system` records from Phase 3/4's own seeded history are still
   present unchanged, exactly as the task brief expects ("alongside any
   test records"). Two-click-selected the top `search_notes`/`denied` row;
   inspector populated with the identity line `search_notes —
   builtin:tldw_chatbook` and pretty-printed JSON (`"ts"`, `"tool"`,
   `"initiator": "agent"`, `"decision": "denied"`, `"ok": false`,
   `"duration": "0ms"`, `"error": null`, `"arguments": null`,
   `"result_excerpt": null` — this entry's `arguments`/`result_excerpt` are
   genuinely `null` because a denial never reaches `execute_hub_tool()`;
   `record_tool_decision()` never receives them for a refusal), plus
   **`Open tool`** / **`Adjust permission`** drill links. Verified: all
   fields + both drill labels hit via DOM text search. PNG 214KB.

7. **`audit-filter-decision`** — LIVE. Opened the `All decisions` Select,
   picked **`Approved`**. Table narrowed from 11 rows to exactly the 3
   agent-approved `list_characters` rows above; the `denied` `search_notes`
   row and every `test`/`system`/`downgraded` row disappeared from the
   table (the still-selected row's own inspector detail JSON below it
   still shows `"decision": "denied"` from the prior selection — that's the
   inspector retaining its last selection across a filter change, not the
   table itself, and is expected/harmless). Verified via the screenshot's
   own row count and a DOM search confirming no `denied` row remains in
   the table body. PNG 167KB.

8. **`audit-drill-open-tool`** — LIVE. Clicked **`Open tool`** from the
   still-selected `search_notes` audit entry's detail. Navigated to Tools
   mode with `search_notes` selected (highlighted row at the bottom of the
   catalog); Inspector now shows the Tools-mode detail (`search_notes —
   tldw_chatbook`, description "Search notes by content or title.", `Tags:
   —`, `Parameters: raw JSON`, `Test Tool` button, `Permission: Ask`,
   "Inherited from the global default.") with **zero** trace of the prior
   audit-entry detail — confirmed by DOM text search: `"decision"` and
   `Adjust permission` (both present in capture 6/7) are **NOT FOUND**
   anywhere in the buffer after the drill. PNG 165KB.

9. **`audit-findings-empty`** — LIVE. Back to Audit mode, clicked the
   **`Findings`** sub-view button (Source is `Local` throughout this whole
   round — never switched to Server). Findings pane shows exactly
   **"Findings come from a tldw_server target."** (the local-source empty
   copy, `_FINDINGS_LOCAL_EMPTY_MESSAGE`), Executions table/filters hidden.
   Verified: exact string hit via DOM text search. PNG 116KB.

10. **`kill-switch-blocks`** — LIVE both halves. MCP Hub → Permissions →
    clicked the **`block MCP tools in chat: no ▸`** toggle button once →
    `no ▸` → `yes ▸`; confirmed on disk (`mcp_permissions.json`'s
    `kill_switch: false → true`) before touching Console. Sent "List the
    characters please, kill switch test." from a fresh Console send: the
    fake server's own request log shows `offered=[calculator,
    get_current_datetime, spawn_subagent]` — **no MCP tool, and no
    `find_tools`/`load_tools` either** (with the MCP provider never even
    constructed this run, the remaining catalog is small enough to fall
    UNDER the direct-disclosure threshold, so the meta-tools aren't offered
    at all this time either) — so the fake LLM's own fallback branch fired
    immediately (`target not in offered` AND `find_tools not in offered` →
    plain text), and the turn finished in a single LLM call: **"No matching
    MCP tool was offered for this request."** Confirmed the Inspector's
    **MCP row is absent**: with the panel still expanded from capture 5,
    the Tools section now reads only `Tools: 0 ready` with no MCP line
    at all — DOM text search for `server enabled` and `tools ready`
    (the two substrings the row's two possible renderings always contain)
    both came back NOT FOUND; the ONLY `MCP:` match left in the whole
    buffer is the unrelated `Live work sources` list's own
    `MCP: Not wired - MCP servers.` line. PNG 359KB. **Flipped the kill
    switch back to `no` immediately after** (confirmed via a second disk
    read: `kill_switch: false`).

## Driver gotchas (methodology, not app defects)

1. **`PYTHON_KEYRING_BACKEND=fail` crashes the textual-serve-spawned app
   subprocess outright** — `"fail"` is not an importable module path, and
   `keyring.core.load_keyring()` raises `ValueError: Empty module name`
   inside `TldwCli.__init__` → `_wire_server_context_provider` →
   `build_default_server_credential_store` → `keyring.get_keyring()`,
   which crashes the per-session app process before it ever renders a
   frame (confirmed via `run_web_server`'s own log — a full traceback, not
   a hang). This env var is fine for `native_gate.py`-style direct script
   invocations (no keyring init on that path) but must NOT be set for
   `run_web_server`. Simply omitting it worked fine — no keyring prompt or
   hang was observed either way in this sandboxed HOME.
2. **The terminal is genuinely blank (just the intro splash) until
   `document.body.classList` gains `-first-byte`** — textual-serve's own
   intro-dialog CSS (`body.-first-byte .textual-terminal { opacity: 1 }`)
   gates the actual app content's visibility; a fixed sleep before this
   class appears (cold Python import + DB init genuinely takes a few
   seconds) captures a false-empty page. Poll for the class instead (this
   round: 3-30s depending on cold vs. warm process).
3. **The Console's right-side "Inspector" panel, and its own left-rail
   "Agent"/"Details" sections, are collapsibles that must be clicked open**
   — none of captures 1-4's transcript-only content needed this, but
   capture 5 (the MCP row) lives inside the Inspector panel and was
   invisible until clicked. Collapsing/expanding one section shifts every
   later section's own collapse-arrow glyph position — an occurrence-index
   click plan computed before the first expand goes stale after it;
   re-`find` the target text fresh after each expand rather than
   pre-computing all click occurrences up front.
4. **A Textual `Select`'s current-value display needs an explicit click to
   open its options overlay** — clicking the collapsed value text alone
   (as opposed to, e.g., a DataTable row) does open it on a clean click,
   but a same-coordinate double-click (this round's own first attempt, out
   of caution) opens then immediately re-closes it, net no-op. One click,
   dump-and-check before the next action, exactly like Phase 3/4's own
   Select-interaction precedent.
5. **A Select option's label can wrap across two xterm rows** —
   `"Approve for session"` rendered as `"Approve for"` on one row and
   `"session"` on the next; a DOM search for the full phrase as one
   contiguous string fails even though the option is genuinely present and
   clickable. Search for a substring guaranteed to stay within one row
   (`"Approve for"`) instead.
6. **The literal substring `"Deny"` has two legitimate DOM matches on the
   same approval-card screen** — the batch's own always-present `Deny all`
   button, and (once a row's decision Select is open) the `Deny` option
   inside the overlay. This round's own driver supports an `occurrence`
   index specifically for this (`click "Deny" 1` selects the second hit);
   confirmed by screenshot which hit was actually clicked before
   `Submit`-ing.
7. **Lingering hover tooltips survive into later screenshots**, same
   lesson as every prior round — moving the mouse to `(5, 5)` (literal
   top-left corner) is not neutral in this build: it hovers the `Home` nav
   chip and produces its OWN tooltip ("Open dashboard, notifications, and
   active work."), polluting the very next capture. Moved to a point deep
   inside the empty transcript area (`(1000, 700)`) instead for every
   "neutral point" move this round.

## Observations (not filed as defects)

- **The Audit table's "When" column renders in UTC, not local wall-clock
  time.** `mcp_audit_mode._format_when()` (`datetime.fromisoformat(ts)` →
  `.strftime(...)`, no timezone conversion) renders the stored ISO-UTC
  timestamp's own digits verbatim. Cross-checked live: capture 2's actual
  approval was submitted at local time ~02:37 (this machine's system
  clock, matching the fake LLM server's own `datetime.now()` log lines),
  and the corresponding Audit row shows `2026-07-17 09:37:48` — a 7-hour
  offset consistent with this machine's local UTC−7 offset. Not filed as a
  defect: showing UTC in an audit/execution log is a defensible, common
  design choice (unambiguous across timezones/deployments), and no other
  Console/MCP-Hub surface driven this round displayed an absolute
  timestamp to compare it against for an actual inconsistency — but it IS
  a real, user-visible 7-hour-off reading if someone assumes the Audit
  table shows local time, worth a product decision either way.
- **No app defects were reproduced this round.** Everything driven —
  native tool-call round-trip (find_tools/load_tools/target-tool
  disclosure sequencing), the batch approval card (row copy, per-row
  Select, Approve all/Submit/Deny all), the approve-once/approve-for-
  session/deny verdict paths and their exact transcript renderings
  (including the real `DENY_REFUSAL` copy), the MCP inspector row's
  ready/blocked branching, the Audit mode table/filter/drill/Findings
  sub-view, and the kill switch's effect on both tool disclosure and the
  inspector row — matched source/spec exactly.

## Isolated HOME (after this round)

Left on disk at **`/private/tmp/tldw-qa-mcp-hub-p5-20260717`** (port 9195).
`mcp_permissions.json`: unchanged from this round's own seed except
`local:docs-server.read_file` gained a `"config_changed": true` marker
(the app's own live rug-pull detection against the seeded wrong hash,
expected) — **no `builtin:tldw_chatbook` entries were ever persisted to
the store**, because `approve_session` is an in-memory-only cache
(`MCPToolProvider`/`unified_control_plane_service.approve_for_session`),
not a store write; `kill_switch` is `false` (toggled on for capture 10,
then back off, confirmed via disk reads both times). `mcp_execution_log.jsonl`
gained this round's 4 new agent-initiated records (3 `approved` for
`list_characters`, 1 `denied` for `search_notes`) on top of Phase 3/4's
existing `test`/`system` history. `local_mcp_store.json`'s
`runtime_activity` gained the corresponding `list_characters`/
`search_notes` entries. Both the `run_web_server` process and the
driver's headless Chromium (and the fake LLM server) were killed at the
end of this round; the fake-LLM's own port (8899) and the driver's CDP
port (9324) were confirmed free afterward.
