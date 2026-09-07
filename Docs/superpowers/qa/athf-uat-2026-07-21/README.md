# ATHF (Agentic Threat Hunting Framework) — third-party MCP server UAT via chatbook (2026-07-21)

User-acceptance test of the full "install → configure → connect → discover →
test → gate → drive via chat → audit" flow for a **real, independent,
third-party MCP server** (not chatbook's own builtin tools, not a scripted
stub) added through the MCP Hub UI shipped in the six MCP Hub phases
(`Docs/superpowers/qa/mcp-hub-phase1-2026-07/` … `phase5-2026-07/`). Real app
(`dev` branch, main checkout), real CSS, headless browser, real ATHF stdio
process, real hunt data on disk. This round drove the app through
`textual-serve` + a standalone headless Chromium over CDP, same methodology
as the MCP Hub Phase 5 round — no app code was modified.

**Verdict up front: yes.** A threat hunter can install ATHF, add it as a
local stdio MCP server, connect, discover all 21 tools, test one with real
data, gate it behind an approval, and drive it from a natural-language
Console turn — entirely through the chatbook UI, no source-code or raw-JSON-file
editing required. The one env-var friction point below is real and worth a
product decision, but has a clean workaround already in the product (env
placeholders), and the form surfaces it with a legible (if terse) error.

## Setup (for reproducibility)

- ATHF: isolated venv `/private/tmp/athf-uat-20260721/athf-venv/` (binary
  `athf-mcp`, stdio, 21 tools, needs `ATHF_WORKSPACE`). Hunt workspace at
  `/private/tmp/athf-uat-20260721/hunts` (4 hunts, incl. `H-0001` "macOS
  Data Collection via AppleScript Detection").
- Isolated HOME: `/private/tmp/athf-uat-home` — a `cp -R` of the existing
  `/private/tmp/tldw-qa-mcp-hub-p6-20260720` QA HOME (already had `[mcp]
  enabled = true` and a `[api_settings.custom]` block pointed at a fake
  local LLM on port 8899 from prior MCP Hub QA). Its seeded `docs-server`
  profile was removed (`local_mcp_store.json` deleted and rebuilt) and
  `mcp_permissions.json` reset to a clean `kill_switch: false,
  global_default: "ask"`, no per-server overrides — so ATHF is the only
  external server and starts at the global default, letting the very first
  agent tool-call genuinely hit the approval gate.
- The `athf` profile was seeded directly via
  `LocalMCPStore` + `LocalMCPControlService.save_external_profile({"profile_id":
  "athf", "command": ".../athf-venv/bin/athf-mcp", "args": [],
  "env_placeholders": {"ATHF_WORKSPACE": "$ATHF_WORKSPACE"}, "env_literals":
  {}})`, mirroring the exact call the task's own controller had already
  confirmed works end-to-end — this establishes the *working* baseline
  profile; the UI's own Add-server form was then used separately (see
  capture 1) to test what a user typing a raw path looks like.
- Served from the main checkout with the workspace var **exported into the
  server process's own environment** (this is what the placeholder actually
  resolves against):
  `HOME=/private/tmp/athf-uat-home ATHF_WORKSPACE=/private/tmp/athf-uat-20260721/hunts
  PYTHONPATH=. .venv/bin/python -c "from tldw_chatbook.Web_Server.serve import
  run_web_server; run_web_server(host='127.0.0.1', port=9199)"`.
- Fake LLM: `/private/tmp/athf-uat-20260721/fake_llm_server.py`, the P5 QA
  harness (progressive-disclosure-aware `find_tools`/`load_tools` state
  machine), extended with one more trigger: the substring `"hunts in the
  workspace"` in the user's message now resolves to
  `mcp__athf__athf_hunt_list` (verified against `MCP/tool_naming.py`:
  `llm_tool_name("local:athf", "athf_hunt_list")` strips the `local:` label
  prefix and joins with `__`, giving exactly `mcp__athf__athf_hunt_list`),
  with the second-turn completion text `"I found 4 hunts in the workspace,
  including the macOS AppleScript collection hunt."` — the exact line the
  task brief specified. Run on port 8899, `.venv/bin/python fake_llm_server.py 8899`.
- Browser: bundled Playwright Chromium (`chromium-1208`, "Google Chrome for
  Testing"), headless, `--remote-debugging-port=9330`, driven from short
  `python3 driver.py <action>` invocations of
  `/private/tmp/athf-uat-driver/driver.py` (forked from the Phase 5/6 QA
  driver — `open`/`screenshot`/`find`/`click`/`type`/`key`/`clickxy`/`corner`),
  viewport 2050×1240, `context.route("**/*", ...)` aborting any non-
  `127.0.0.1`/`localhost` request.

## Captures — all 10 LIVE, none fallback

All 10 requested pieces of evidence were captured against the real running
app; no seeded-JSONL or screenshot-mockup fallback was needed anywhere.
(Capture 1 in the brief was naturally two screenshots — the filled form and
its rejection — since the interesting UAT moment *is* the state transition
between them.)

| # | File | What it shows |
|---|------|----------------|
| 1a | `athf-add-server-form-2026-07-21.png` | Add-server form filled for ATHF: profile id `athf-demo`, command the real `athf-mcp` path, and — deliberately, to test the friction point — the env line typed as a **plain literal** `ATHF_WORKSPACE=/private/tmp/athf-uat-20260721/hunts` rather than a placeholder. |
| 1b | `athf-add-server-env-rejected-2026-07-21.png` | Clicking Save on that form: the store's validation rejects it in-form, in red, directly under the env `TextArea`. |
| 2 | `athf-servers-connected-2026-07-21.png` | Servers mode, the seeded `athf` profile after Connect: `Ready`, "Connected — 21 tools available." |
| 3 | `athf-tools-catalog-athf-2026-07-21.png` | Tools mode: all 21 `athf_*` tools listed (plus the 10 builtin `tldw_chatbook` tools below them), each with its `raw`/`form` Schema column. |
| 4 | `athf-test-tool-hunt-list-2026-07-21.png` | Test Tool on `athf_hunt_list` (`{}` raw JSON) → confirm-run (permission is `Ask`) → `OK · 40ms`, real result: `"count": 4`, `H-0001` "macOS Data Collection via AppleScript Detection". |
| 5 | `athf-test-tool-schema-form-2026-07-21.png` | Test Tool on `athf_attack_lookup` — a schema-driven form rendering its one parameter, `technique_id *`. |
| 6 | `athf-permissions-athf-2026-07-21.png` | Permissions mode: all 21 ATHF tools in the matrix (`athf: 1 allow · 20 ask · 0 off`) — `athf_hunt_stats` was cycled to an explicit `Allow` override to demonstrate the affordance, while `athf_hunt_list` was deliberately left at the inherited `Ask` default so the next capture's approval card would be genuine. |
| 7 | `athf-agent-approval-card-2026-07-21.png` | Console: sent "Please list the hunts in the workspace." → the agent (through the real `find_tools`→`load_tools`→tool-call handshake, unattended) hit the `Ask` gate → **"Approval required" / `athf · athf_hunt_list`**, an `Approve once` decision Select, `Approve all` / `Submit` / `Deny all`; header chip `Approvals: 1 pending`. |
| 8 | `athf-agent-result-2026-07-21.png` | After `Submit`: the transcript shows the real tool call `⚙ mcp__athf__athf_hunt_list → {"result": [...H-0001 "macOS Data Collection via AppleScript Detection"...` and the assistant's final line, **exactly** `"I found 4 hunts in the workspace, including the macOS AppleScript collection hunt."` |
| 9 | `athf-audit-athf-execution-2026-07-21.png` | MCP Hub → Audit → Executions, drilled into the top row: `2026-07-21 02:07:16  local:athf::athf_hunt_list  agent  approved  124ms  OK`, detail JSON `"initiator": "agent"`, `"decision": "approved"`, real `result_excerpt`, plus `Open tool` / `Adjust permission` drill links. The row directly below it (`test  allowed  38ms`) is this same round's own Test Tool run from capture 4 — both genuinely present, alongside the pre-existing `test`/`system` history carried over from the HOME template. |

## The user story, step by step

**Install.** Out of scope for the UI (ATHF is a separate `pip`-installed
package with its own venv) — the app only ever needs a command path, no
different from any other stdio MCP server.

**Configure.** The Add-server form (Servers mode → `Add server`) is a plain
four-field form: profile id, command, args (one per line), env (`KEY=value`
per line). It correctly distinguishes an env **placeholder** (`$VAR` /
`${VAR}`) from a **literal** purely by regex on the value — no separate UI
toggle to get wrong. Typing the correct value (`ATHF_WORKSPACE=$ATHF_WORKSPACE`)
would have saved cleanly; see the dedicated finding below for what happens
if a user instead pastes a real path.

**Connect / discover.** Selecting the profile row and clicking `Connect`
spawned the real `athf-mcp` process, and — genuinely, not simulated —
returned 21 tools in well under a second: "Connected — 21 tools available."
The Servers-mode detail view lists all 21 tool names inline.

**Test.** Tools mode lists every tool with its own `Ask`/`Allow`/`Off`
state and a `raw`/`form` schema indicator. `Test Tool` on `athf_hunt_list`
correctly gated behind the `Ask` permission with a one-more-click confirm
("This tool is set to Ask — press again to run; anything else cancels."),
then ran the real process and returned genuine hunt data in 40ms.

**Gate.** Permissions mode is a flat table of all 21 ATHF tools plus the
server-default and global-default rows; Space cycles a selected row through
Inherit → Allow → Ask → Off. This matches the existing builtin-tools
Permissions UX exactly — no ATHF-specific quirk.

**Drive via chat.** A single natural-language Console message ("Please
list the hunts in the workspace.") triggered the full real agent pipeline —
tool discovery, the `Ask`-gated approval card naming the tool by its
human-readable `athf · athf_hunt_list` identity (not the raw
`mcp__athf__athf_hunt_list` LLM-facing name), and, after one click to
approve, a transcript showing both the raw tool result and the model's own
natural-language summary.

**Audit.** The Audit tab's Executions table picked up the agent-initiated
call immediately, correctly tagged `initiator: agent`, alongside the
manually-run Test Tool call from earlier in the same session tagged
`initiator: test` — the two are visually and structurally indistinguishable
except for that one field, which is exactly the audit trail's job.

## Finding: env-literal path rejection (confirmed, as predicted)

Typing a plain filesystem path into an env field of the Add-server form —
`ATHF_WORKSPACE=/private/tmp/athf-uat-20260721/hunts` — and clicking Save
produces, verbatim, in the form's own error `Static` (red text, directly
under the env box, no toast, no navigation away):

> **Literal env key 'ATHF_WORKSPACE' must use an explicit safe operational
> literal or an env placeholder**

Root cause (`MCP/local_store.py:_is_safe_literal_value`): a "safe literal"
is deliberately restricted to booleans/`enabled`/`disabled`/log-levels
(`_SAFE_LITERAL_VALUES`), small integers/decimals, or an `http(s)://` URL —
never a filesystem path, by design (paths are exactly the kind of thing
that leaks machine-specific/PII-shaped detail into a shared config file).
The *only* way to store a filesystem-shaped env value is as a `$VAR`
placeholder resolved from the process's own environment at connect time —
which is also the officially-working pattern this whole UAT run used.

**Legibility assessment:** the error message itself is accurate but
self-contained — it names the mechanism ("safe operational literal", "env
placeholder") without restating *how* to fix it in the moment. A security
analyst unfamiliar with the app's env-placeholder convention would need to
scroll back up to the form's own static hint text above the env `TextArea`
("Secrets are never stored — reference them as `KEY=$ENV_VAR` and export
the variable before connecting.") to know what to type instead. That hint
is present and correct, just not repeated alongside the error. **Product
recommendation:** append the fix to the error itself, e.g. `... must use an
explicit safe operational literal or an env placeholder — try
'ATHF_WORKSPACE=$ATHF_WORKSPACE' and export the value before connecting.`
Low-cost, would remove the only real "stuck" moment in this entire UAT run.
Not filed as a defect — the current behavior is safe-by-default and
correct, just terser than ideal at the exact moment a new user needs more.

## Observation: many ATHF tools render as raw JSON, not a form

Of ATHF's 21 tools, 10 rendered as a schema-driven form (`athf_attack_lookup`,
`athf_attack_techniques`, `athf_hunt_get`, `athf_hunt_search`,
`athf_hunt_stats`, `athf_hunt_validate`, `athf_investigate_search`,
`athf_research_search`, `athf_research_stats`, `athf_research_view`) and 11
fell back to raw JSON (`athf_agent_run_hypothesis`,
`athf_agent_run_researcher`, `athf_context`, `athf_hunt_coverage`,
`athf_hunt_list`, `athf_hunt_new`, `athf_investigate_list`,
`athf_investigate_new`, `athf_research_list`, `athf_research_new`,
`athf_similar`). Read `UI/MCP_Modules/mcp_schema_form.py:parse_schema()` to
confirm why: it's a deliberate, documented, honest design choice — "if ANY
declared property can't be rendered faithfully..., the WHOLE parse fails...
rather than silently dropping that property." ATHF's tools are built with
Pydantic v2, whose `Optional[str] = None` parameters serialize as JSON
Schema `anyOf: [{"type": "string"}, {"type": "null"}]` — a pattern
`parse_schema()` doesn't support, so any tool with even one optional filter
argument (which is most of them: `athf_hunt_list`'s four filters are all
optional) falls back to raw JSON, even though every individual field is
otherwise a plain string. This is not a bug (the fallback is honest and the
raw-JSON path works fine, as capture 4 shows), but it is a real,
reproducible UX cost specifically for third-party servers using this very
common modern Python typing idiom — worth a follow-up ticket to add `anyOf`-
of-`[T, null]` support to `parse_schema()` (treat it as an optional `T`,
same as today's `required` bookkeeping already distinguishes) if the product
wants more third-party MCP tools to get the friendlier form.

## App defects

**None.** Every surface driven — Add-server form and its validation,
Servers/Tools/Permissions/Audit modes, the Test Tool raw-JSON and
schema-form paths, the Console agent pipeline (discovery →
`find_tools`/`load_tools` handshake → approval card → approved execution →
transcript → audit trail) — matched source/spec exactly against a genuine
third-party server it had never seen configuration for before. The
env-literal rejection and the raw/form schema split above are real UX
frictions worth product decisions, not defects.

## Overall UAT verdict

**Yes** — a real threat hunter can install ATHF and drive it via chatbook
today, entirely through the UI: add the server, connect, see all 21 tools
discovered live, test one against real hunt data, gate it behind an
approval, ask a Console agent to use it in plain English, approve the one
prompt it asks for, and find the whole thing logged in the audit trail
afterward. The only moment requiring outside knowledge (the env-placeholder
convention for anything path-shaped) is enforced safely and explained
correctly, just not as helpfully as it could be at the point of failure.

## Cleanup

The fake LLM server, the standalone headless Chromium, and the `run_web_server`
process were all killed at the end of this round; ports 8899, 9330, and 9199
were confirmed free afterward. Both HOMEs (`/private/tmp/athf-uat-home` and
the untouched `/private/tmp/tldw-qa-mcp-hub-p6-20260720` template) and the
ATHF install/workspace under `/private/tmp/athf-uat-20260721/` were left on
disk for follow-up, unmodified beyond this round's own writes (the seeded
`athf` profile, the `athf_hunt_stats` permission override, and this round's
audit/execution-log entries).
