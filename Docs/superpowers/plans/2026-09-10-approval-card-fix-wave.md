# Approval Card + MCP Permissions Fix Wave Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the twenty backlog tasks 32272-32291 filed from the 2026-09-10 live review of the Console tool-approval card and the MCP Permissions/Tools/Servers/Audit surfaces.

**Architecture:** Four independent lanes, each its own worktree and branch off `docs/approval-card-ux-review-2026-09-10` (origin/dev + the task files): **A** dev blockers (readiness validator, provider payload serialisation, subscriptions schema, first-send stall); **B** the approval card and Console honesty (card widget, chat screen binding, activity/run/inspector copy, transcript marker, wizard copy); **C** the MCP hub (built-in inventory, audit denials, exact-input rules UI, session approvals, Tool gates, Permissions/Tools polish); **D** docs. Tasks inside a lane run sequentially; lanes run in parallel. Each lane becomes one PR to `dev`.

**Tech Stack:** Python 3.12, Textual 8.x, pytest (`.venv/bin/python -m pytest`), SQLite, TCSS bundle via `python3 tldw_chatbook/css/build_css.py`.

**Spec:** the backlog task files `backlog/tasks/task-32277 … task-32291` plus `task-32341 … task-32345` (acceptance criteria are the binding contract) plus the review snapshot `.impeccable/critique/2026-09-10T17-31-53Z__hatbook-widgets-chat-widgets-chat-approval-card-py.md`. Tasks 1-4 and 10 were filed as `task-32272 … task-32276` and renumbered to `task-32341 … task-32345` on 2026-09-10 after `dev` landed its own tasks on those ids; see each task file's `## Renumbering provenance`.

## Global Constraints

- Every task ends with its task file updated: AC boxes ticked, `## Implementation Notes` added, status set Done via `backlog task edit <id> -s Done --notes "..."` run from the worktree root (the CLI is `backlog`, ~15 s per call). Verify the CLI printed the expected file path.
- TDD: write the failing test first, run it red, implement, run it green. Never verify with a `-k`-filtered whole-suite run; run the named test files.
- Always run tests with the repo venv: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider`.
- Before changing any behaviour, grep the tests for a defect-PINNING test (a test whose name asserts the current behaviour). If one exists, the "bug" may be a decision: keep the behaviour and report it as a concern instead of flipping the test.
- Editing `tldw_chatbook/css/components/*.tcss` requires `python3 tldw_chatbook/css/build_css.py` and committing the regenerated `tldw_chatbook/css/tldw_cli_modular.tcss` and `widget_defaults_{self,scoped}.tcss` together with the source. Never hand-edit the bundle. Never write `SomeWidget > Vertical`-style ancestor-scoped bare-type rules in `DEFAULT_CSS`.
- Copy rules (PRODUCT.md): state must be carried in text, never colour alone; blocked and waiting states are explicit; keyboard-first.
- Vocabulary (locked for this wave): a tool that will not run because the USER said no is **denied by you**; because policy is Off it is **blocked (Off)**; because the kill switch is on it is **blocked (kill switch)**. Card decision labels (locked): `Approve once`, `Approve for session`, `Always allow this exact input`, `Always allow`, `Deny` stay as the option VALUES; display labels may be shortened per Task 7.
- Stage explicit paths only (`git add <files>`); never `git add -A`.
- Commit messages end with `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01PN1gyDYViLmXzgkwGHVN5J`.
- Do not touch files outside the task's listed files except tests and the task file, unless the fix's root cause lives elsewhere; then say so in the report.

---

## Lane A — dev blockers (branch `fix/approval-wave-a-blockers`)

### Task 1: Keyless provider with a stored api_key no longer crashes Console readiness (task-32341, filed as task-32272)

**Files:**
- Modify: `tldw_chatbook/Chat/console_session_settings.py:1384-1394` (credential facet derivation)
- Test: `Tests/Chat/test_console_session_settings.py`

**Interfaces:**
- Consumes: `get_provider_readiness(...)` returning `.requires_api_key`, `.api_key_source`, `.ready`.
- Produces: `ConsoleSettingsReadiness(credential="not_required", credential_source="none")` for every keyless provider, whether or not a key is stored.

- [ ] **Step 1: Write the failing test** next to the existing keyless test (around line 630, which pins `credential == "not_required"` and `credential_source == "none"`). Build the same fixture but with `[api_settings.custom] api_key = "dummy"` (and `api_key_env_var` unset) in the config passed to the readiness builder:

```python
def test_keyless_provider_with_stored_key_reports_not_required_without_source(...):
    # same arrangement as the neighbouring keyless test, plus a stored api_key
    readiness = build(...)  # the function the neighbouring test calls
    assert readiness.credential == "not_required"
    assert readiness.credential_source == "none"
    assert readiness.operability == "ready_to_send"
```

- [ ] **Step 2: Run it** — expected: `ValueError: Console credential source conflicts with its facet.`
- [ ] **Step 3: Fix** — in the `if not readiness.requires_api_key:` branch set `credential_source = "none"` (the key is irrelevant to readiness for a keyless provider; the validator at L566-568 already encodes that rule).
- [ ] **Step 4: Run** `Tests/Chat/test_console_session_settings.py` and `Tests/Chat/test_provider_test_evidence.py` — all green.
- [ ] **Step 5: Live check** — launch with a scratch HOME (`HOME=<scratch> python -m tldw_chatbook.app` from the worktree root; config `[chat_defaults] provider="custom" model="x"` and `[api_settings.custom] api_key="dummy" api_url="http://127.0.0.1:1"`), confirm the app reaches Console without a traceback; quit. Record the command and result in the report.
- [ ] **Step 6: Close the task file, commit.**

### Task 2: Tool-call continuation serialises immutable arguments; client-side serialisation failures are not "provider HTTP 400" (task-32342, filed as task-32273)

**Files:**
- Modify: the payload builder that emits the assistant `tool_calls` entries for re-send — trace from `tldw_chatbook/Chat/console_agent_bridge.py:3272` (`message["tool_calls"] = native_calls`) back to where each entry's `function.arguments` is set; `tldw_chatbook/Agents/agent_models.py:372` makes `ToolCall.arguments` a `MappingProxyType`.
- Modify: `tldw_chatbook/LLM_Calls/LLM_API_Calls_Local.py` (`_chat_with_openai_compatible_local_server`, the `except` that maps a data-processing error to `ChatBadRequestError(status=400)`).
- Modify: `tldw_chatbook/Chat/console_provider_gateway.py:1080-1105` (`safe_provider_error_copy`) and the Console copy "The provider rejected this request. Confirm the model is still available…" (grep it in `tldw_chatbook/Chat/`).
- Test: `Tests/Agents/test_native_tools.py` or `Tests/Chat/test_console_agent_bridge.py` (payload), `Tests/LLM_Calls/` (handler), `Tests/Chat/test_console_provider_gateway.py` (copy).

**Interfaces:**
- Produces: on the wire, `function.arguments` is always a JSON **string** (OpenAI shape); any local `TypeError` during request preparation raises `ChatConfigurationError` (no status code) whose user copy says the app could not build the request.

- [ ] **Step 1: Failing test (payload):** build a `ToolCall` with arguments `{"query": "x"}`, run it through the bridge/history projection that produces the outbound `messages` list for the next provider call, and assert `json.dumps(messages)` succeeds and `messages[-1]["tool_calls"][0]["function"]["arguments"] == '{"query": "x"}'` (a string).
- [ ] **Step 2: Run red** — expected `TypeError: Object of type mappingproxy is not JSON serializable` (or a dict instead of a string).
- [ ] **Step 3: Fix at the builder** (one place, every handler benefits): serialise `json.dumps(dict(call.arguments))` when building the entry. Do not patch individual handlers.
- [ ] **Step 4: Failing test (error copy):** feed a `TypeError` raised inside request preparation through the handler's error path and assert the raised exception is `ChatConfigurationError` with `status_code is None`; assert `safe_provider_error_copy("custom", exc)` does not contain "Status:" or "rejected".
- [ ] **Step 5: Fix** the handler's `except` so non-HTTP local failures do not become `ChatBadRequestError(status=400)`; adjust the Console copy so it is only used when an HTTP status came back from the provider.
- [ ] **Step 6: Run** the four test files above plus `Tests/Chat/test_console_agent_bridge.py` — green (compare failing-test NAME sets against a baseline run taken before your change; pre-existing reds are not yours).
- [ ] **Step 7: Live check** with the repo's fake LLM: `python Docs/superpowers/qa/mcp-hub-phase5-2026-07/fake_llm_server.py 8899`, scratch HOME config `[chat_defaults] provider="custom" model="fake-model"`, `[api_settings.custom] api_url="http://127.0.0.1:8899" streaming=false`, `[mcp] enabled=true`; send "List the characters please."; the approval card must appear (three fake-server calls: find_tools, load_tools, list_characters). Record the fake server's log lines in the report.
- [ ] **Step 8: Close the task file, commit.**

### Task 3: Subscriptions DB tolerates a schema_version table holding older rows beside the current one (task-32343, filed as task-32274)

**Files:**
- Modify: `tldw_chatbook/DB/Subscriptions_DB.py:740-760` (version check) and the create script's `INSERT OR IGNORE INTO schema_version (version) VALUES (2)`.
- Test: `Tests/DB/test_subscriptions_db.py`

- [ ] **Step 1: Failing tests:** (a) create a temp DB file, execute `CREATE TABLE schema_version(version INTEGER PRIMARY KEY NOT NULL); INSERT INTO schema_version VALUES (1),(2);` plus the minimal v2 tables the constructor expects (copy the v2 DDL the existing tests use), then `SubscriptionsDB(path)` must open and `SELECT version FROM schema_version` must return `[2]`. (b) rows `[3]` (unknown future) must raise `SubscriptionError` whose message names the file path and the found/supported versions.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Fix:** in `_initialize_schema`, if `_CURRENT_SCHEMA_VERSION in versions` → delete the other rows and continue; if `versions == [1]` → migrate (existing); else raise `SubscriptionError(f"Unsupported subscriptions schema version {versions} in {path}; this build supports {_CURRENT_SCHEMA_VERSION}")`.
- [ ] **Step 4: Reproduce the trigger:** write a test that opens a v1 DB with the constructor twice concurrently-ish (second constructor after the first's migration committed) and one that interrupts between `DELETE FROM schema_version` and the `INSERT` (simulate by calling the migration's SQL statements manually with an exception between them) — assert the normalisation in Step 3 recovers on the next open. Record which sequence yields `[1, 2]`; if none does, say so in the report.
- [ ] **Step 5: Run** all `Tests/DB/test_subscriptions_db*.py` and `Tests/Subscriptions/test_subscriptions_db_connection_lifecycle.py`.
- [ ] **Step 6: Close the task file, commit.**

### Task 4: First agent send after restart: trace the stall, bound it, and show it (task-32344, filed as task-32275)

**Files:**
- Investigate: `tldw_chatbook/Agents/mcp_tool_provider.py` (`compose_catalog`, built-in server connect), `tldw_chatbook/MCP/unified_control_plane_service.py` (`hub_lifecycle_timeout_seconds`, connect), `tldw_chatbook/MCP/client.py`, `tldw_chatbook/Chat/console_agent_bridge.py` (pre-provider setup), `tldw_chatbook/UI/Console_Modules/agent.py:246-260, 385-430` (activity line).
- Test: `Tests/Agents/test_mcp_tool_provider.py`, `Tests/Chat/test_console_agent_bridge.py`.

- [ ] **Step 1: Reproduce with timing:** using the Task 2 live recipe, quit and relaunch the app, send once, and read the app log (`<scratch HOME>/.local/share/tldw_cli/<user>/tldw_cli_app.log`) for the gap between "console agent reply start" and "Chat API Call - Routing to endpoint". Add temporary INFO timing logs around catalog composition / MCP connect if needed to locate the wait. Report the culprit with numbers.
- [ ] **Step 2: Failing test:** the located wait must be bounded by the configured timeout (`[mcp] hub_lifecycle_timeout_seconds`, default in code) — write a test that fakes a never-completing connect and asserts the compose returns (with the server marked not connected) within that timeout.
- [ ] **Step 3: Fix** the unbounded wait (or the retry loop) so it honours the timeout.
- [ ] **Step 4: Failing test (visibility):** `format_turn_activity`-style function in `UI/Console_Modules/agent.py` must render `Connecting tools… · <elapsed>` when the run is in the pre-provider setup phase (a new snapshot state; name it `setup` and document it in the four-state table in that docstring, making it five). Implement by having the bridge mark the phase before catalog composition and clear it before the first provider call.
- [ ] **Step 5: Run** the test files and re-run the live check; report the new gap.
- [ ] **Step 6: Close the task file, commit.**

---

## Lane B — approval card and Console honesty (branch `fix/approval-wave-b-card`)

### Task 5: Decision labels fit the Select and state their scope; risk reason visible (task-32278)

**Files:**
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py` (`_DECISION_OPTIONS` L59-68, `_RAW_SHELL_DECISION_OPTIONS`, `_REASON_TOOLTIPS` L167-172, `set_batch` row construction L830-980, `_on_batch_row_select_changed`)
- Modify: `tldw_chatbook/css/components/_agentic_terminal.tcss:10337` (`.approval-row-decision { width: 26 }`) + rebuild bundle
- Test: `Tests/UI/test_chat_approval_card.py`, `Tests/UI/test_approval_row_information_budget.py`

**Interfaces:**
- Produces: option VALUES unchanged (`approve_once`, `approve_session`, `allow_matching`, `always_allow`, `deny`). New module constant `DECISION_SCOPE_COPY: dict[str, str]` and a per-row `Static` with class `approval-row-scope` whose text is `DECISION_SCOPE_COPY[select.value]`. Reason text rendered as a `Static` with class `approval-row-reason` (not a tooltip).

- [ ] **Step 1: Failing tests:**
  - every display label in `_DECISION_OPTIONS` and `_RAW_SHELL_DECISION_OPTIONS` is at most 18 characters (26-cell Select minus 8 cells of chrome);
  - a mounted card with one MCP row has a `.approval-row-scope` Static reading `DECISION_SCOPE_COPY["approve_once"]`; after `select.value = "always_allow"` it reads `DECISION_SCOPE_COPY["always_allow"]`;
  - a row with `reason == "risk_floored"` renders a `.approval-row-reason` Static containing "asks before running" and the header has no tooltip requirement.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement.** Labels (locked): `Approve once` → `Once`; `Approve for session` → `This session`; `Always allow this exact input` → `Always · these args`; `Always allow` → `Always`; `Deny` → `Deny`. Raw shell: `Run once` stays; `Allow all raw shell commands for this Console session` → `All shell · session`. Scope copy (locked):
  - approve_once: `This call only.`
  - approve_session: `Every call to this tool until Chatbook exits (Task 32291 adds revoke).` — ship as `Every call to this tool until Chatbook exits.`
  - allow_matching: `Remembered for exactly these arguments. Remove it under MCP ▸ Tools ▸ this tool.`
  - always_allow: `Remembered for this tool. Change it under MCP ▸ Permissions.`
  - deny: `This call only; the model is told not to retry.`
  Reason copy: risk_floored reads → `High risk: this tool reads local data and always asks first.`; when the entry's `effects` contains `mutates_local` → `High risk: this tool changes local data and always asks first.`; config_changed → `Definition changed since you last allowed it; review the arguments.`
  Keep the fast buttons' labels `Approve once`/`Deny`. Widen `.approval-row-decision` to 22 only if any label still clips (it should not).
- [ ] **Step 4: Rebuild CSS if touched; run** `Tests/UI/test_chat_approval_card.py Tests/UI/test_approval_row_information_budget.py Tests/UI/test_approval_context_lines.py Tests/UI/test_approval_argument_budget.py Tests/Chat/test_approval_payload_summary.py`.
- [ ] **Step 5: Close the task file, commit.**

### Task 6: Bulk Approve all skips raw-shell rows; needs-decision is text (task-32282)

**Files:**
- Modify: `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py` (`_set_all_batch_decisions` L1158-1190, `_on_batch_row_select_changed`, `_format_row_header`)
- Test: `Tests/UI/test_chat_approval_card.py`, `Tests/Chat/test_console_raw_shell_approval.py`

- [ ] **Step 1: Failing tests:** (a) batch with an MCP row and a raw-shell row: after `approval-approve-all` the raw-shell Select is still `deny` and its header Static text starts with `needs decision · `; (b) after the user changes that Select, the prefix is gone; (c) `approval-deny-all` sets both to deny and clears the prefix.
- [ ] **Step 2: Run red.**
- [ ] **Step 3: Implement:** in the bulk loop, `if _is_raw_shell_row(entry) and "approve_once" in candidates: skip + mark`; marking updates the header Static to `NEEDS_DECISION_PREFIX + base_header` (store the base header per row in a list alongside `_batch_rows`); clearing restores it. Keep the CSS class too.
- [ ] **Step 4: Run** the two test files. **Step 5: Close, commit.**

### Task 7: Deadline countdown ticks (task-32288)

**Files:** `tldw_chatbook/Widgets/Chat_Widgets/chat_approval_card.py` (`set_batch` deadline block L763-770, `format_approval_deadline`); `Tests/UI/test_chat_approval_card.py`.

- [ ] **Step 1: Failing test:** mount with `timeout_seconds=90`, advance the app clock (pilot pause 1.1 s) and assert the `#approval-deadline` text changed from `Auto-denies in 1:30` to `Auto-denies in 1:29` or lower; with `timeout_seconds=0` no timer is armed and the Static is hidden.
- [ ] **Step 2: Red. Step 3:** store `self._deadline_at = time.monotonic() + total`, arm `self.set_interval(1.0, self._tick_deadline)` once per batch (stop the previous timer first), `_tick_deadline` re-renders and stops at 0 and when the card hides (`set_batch([])` / `display=False`).
- [ ] **Step 4: Run** `Tests/UI/test_chat_approval_card.py`. **Step 5: Close, commit.**

### Task 8: Approval card height matches its content (task-32287)

**Files:** `tldw_chatbook/Widgets/Chat_Widgets/chat_task_cards.py`, `tldw_chatbook/Widgets/Console/console_session_surface.py:267`, `tldw_chatbook/css/components/_agentic_terminal.tcss` (add rules for `#console-task-surface` / `ChatTaskCards` / `#approval-batch-actions` as needed) + bundle; `Tests/UI/test_chat_approval_card.py`.

- [ ] **Step 1: Measure:** mount `ChatTaskCards` inside a 200×50 app with the real bundle CSS (see `Tests/UI/test_chunking_templates_widget_parity.py` for a real-bundle harness pattern), set a one-row batch, read `card.size.height` and the rows/actions heights; find which container reserves the blank rows (suspects: `ChatTaskCards` default `height: 1fr`, or `#approval-batch-rows` `max-height: 15` combined with a fixed parent).
- [ ] **Step 2: Failing test:** one-row batch card height ≤ 12 lines; three-row batch ≤ 22; in an 80×24 app the `#approval-submit` button's region is within the screen.
- [ ] **Step 3: Fix** with `height: auto` on the reserving container (bundle CSS, not `DEFAULT_CSS` only — note memory: app CSS beats `DEFAULT_CSS`). Rebuild the bundle.
- [ ] **Step 4: Run** the test file; **Step 5: Close, commit.**

### Task 9: Keyboard route to the approval card (task-32277)

**Files:** `tldw_chatbook/UI/Screens/chat_screen.py` (BINDINGS near L1881; the handler that already forwards the inspector's Review approval button at L20144; the tab-strip needs-approval marker widget); `Tests/UI/` (a mounted ChatScreen test file that already exercises bindings, e.g. grep `open_trajectory_view`).

- [ ] **Step 1: Failing tests:** (a) `Binding("alt+a", "review_pending_approval", "Approval", show=True)` exists; (b) with a pending batch mounted, `action_review_pending_approval()` leaves focus on the row's `.approval-row-decision` Select (use `ChatApprovalCard.focus_first_decision`); (c) with nothing pending it notifies `CONSOLE_INSPECTOR_NO_APPROVAL_REASON` ("No approval is pending."); (d) clicking the tab's `◆` marker calls the same action.
- [ ] **Step 2: Red. Step 3: Implement** by routing the new action through the same code path as `handle_console_inspector_review_approval`. Add the key to the footer legend string (grep `Alt+I inspect` in the footer text and add `Alt+A approval`).
- [ ] **Step 4: Run** the test file(s) plus `Tests/UI/test_console_agent_steering*.py` if they mount the screen. **Step 5: Close, commit.**

### Task 10: Waiting-for-approval state on the activity line, run chip and inspector (task-32345, filed as task-32276)

**Files:**
- Modify: `tldw_chatbook/UI/Console_Modules/agent.py:246-260, 385-430` (activity line; add state `waiting_approval` → `Waiting for your approval · <elapsed>`), `tldw_chatbook/Chat/console_chat_controller.py:22509` (run state copy — add `ConsoleRunState(ConsoleRunStatus.STREAMING, "Waiting for your approval.")` when `has_pending_approval_round(session_id)`), `tldw_chatbook/Widgets/Console/console_send_authority_summary.py:144` (already has `run = "Waiting for approval"` — make it the branch that fires for a pending approval, and make `Live work`/`Setup`/`Blocked impact` lines not say Generating / Recovery required / Provider configuration required in that case), `tldw_chatbook/Chat/console_display_state.py` (status chip).
- Test: `Tests/Chat/test_console_display_state.py`, `Tests/UI/test_console_send_authority_summary*.py` (grep), `Tests/UI/test_console_turn_activity*.py` (grep `CONSOLE_TURN_ACTIVITY_THINKING`).

- [ ] **Step 1: Failing tests:** (a) snapshot with a pending approval renders `Waiting for your approval · 5s` not `Thinking… · 5s`; (b) the run chip reads `Run: Waiting for your approval.`; (c) the authority summary with `pending_approvals=1` reads `Run: Waiting for approval` and `Live work: Waiting for your approval`, and never `Recovery required` unless a real blocker exists.
- [ ] **Step 2: Red. Step 3: Implement.** The pending flag already exists: `ConsoleChatController.has_pending_approval_round(session_id)`; thread it into the snapshot the activity formatter reads (add `pending_approval: bool` to the live snapshot or pass it as a parameter) — do not infer from the tool name.
- [ ] **Step 4: Run** the named tests + `Tests/Chat/test_console_chat_controller.py` (compare name sets against baseline). **Step 5: Close, commit.**

### Task 11: Transcript shows "denied by you" and labels model-facing text (task-32279)

**Files:** `tldw_chatbook/Chat/console_agent_bridge.py:1440-1482` (`classify_activity_status`), `tldw_chatbook/UI/Console_Modules/agent.py` (`format_agent_step_marker` status words), the expanded tool-box label "Full output" (grep in `tldw_chatbook/Widgets/Chat_Widgets/` / `UI/Console_Modules/`); `Tests/Chat/test_console_agent_bridge.py`, `Tests/UI/test_console_tool_markers*.py` (grep `· blocked`).

- [ ] **Step 1: Failing tests:** (a) `classify_activity_status(error=USER_DENY_REFUSAL)` → `"denied"`; kill-switch and Off refusals stay `"blocked"`; (b) the marker line for a denied step reads `· denied by you`; for policy Off `· blocked (Off)`; for kill switch `· blocked (kill switch)`; (c) the expanded box header for a refused step reads `Sent to the model` instead of `Full output`.
- [ ] **Step 2: Red. Step 3: Implement** (new status value `denied` flows through every consumer of the status; grep `"blocked"` consumers and add the sibling).
- [ ] **Step 4: Run** the test files. **Step 5: Close, commit.**

### Task 12: First-run tools copy (task-32289)

**Files:** `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py` (`ToolsStep._TOOL_COPY` L6561-6575, Quick-setup `SummaryStep` tools line "Tools — all off (default)"); `Tests/UI/test_first_run_setup_wizard*.py` (grep `_TOOL_COPY`).

- [ ] **Step 1: Failing tests:** (a) `_TOOL_COPY["read_file"][1]` mentions "asks each time" ; (b) the Quick-setup summary tools line reads `Tools — all off; turn them on under MCP ▸ Servers ▸ Tool gates`.
- [ ] **Step 2: Red. Step 3: Implement:** read-class descriptions gain ` Asks you each time before running.`; summary line as above.
- [ ] **Step 4: Run** the wizard tests. **Step 5: Close, commit.**

---

## Lane C — MCP hub (branch `fix/approval-wave-c-hub`)

### Task 13: Built-in server tools appear in Tools mode and the Permissions matrix (task-32283)

**Files:** `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:1680-1712` (built-in inventory read), `tldw_chatbook/MCP/hub_tool_catalog.py:153-190` (`builtin_tools_from_inventory`), `tldw_chatbook/Agents/mcp_tool_provider.py:345-365` (the Console's read of the same inventory, for parity); `Tests/UI/test_mcp_tools_mode.py`, `Tests/UI/test_mcp_permissions_mode.py`.

- [ ] **Step 1: Trace:** in a mounted MCP screen with the real `unified_mcp_service`, log what `getattr(service, "local_service", None)` and `get_inventory()` return in the workbench versus in `MCPToolProvider`. Report the divergence (hypotheses: the workbench holds a different service object; the inventory `tools` list is empty until the local client session connects; `_require_allowed` raises and is swallowed).
- [ ] **Step 2: Failing tests:** with an inventory of two tools, the Tools-mode table has rows `list_characters` and `search_notes` under server `tldw_chatbook (built-in)`, and the Permissions matrix has a `Server default — tldw_chatbook (built-in)` group with those rows; a Space-cycle on `list_characters` writes `builtin:tldw_chatbook`/`list_characters` to the permission store and `resolve`-ing that key returns the new state.
- [ ] **Step 3: Fix** so the workbench uses the same inventory source the Console provider uses. If the inventory is genuinely empty until connect, `Refresh tools` must trigger the connect/discovery and the empty state must read `Built-in tools appear after the first Console tool call or Refresh tools.`
- [ ] **Step 4: Run** the two test files + `Tests/UI/test_mcp_servers_mode.py`. **Step 5: Live check** with the Task 2 recipe from lane A (or without the fake LLM: open MCP ▸ Tools and confirm the built-in rows). **Step 6: Close, commit.**

### Task 14: Card denials are recorded in Audit, distinct from policy Off (task-32280)

**Files:** `tldw_chatbook/Chat/console_chat_controller.py:1603-1950` (the `review_tool_calls` hooks applying verdicts before dispatch), `tldw_chatbook/Agents/mcp_tool_provider.py:790-815, 1000-1012` (`_record_decision_safe` call sites), `tldw_chatbook/UI/MCP_Modules/mcp_audit_mode.py:78-96` (`_DECISION_OPTIONS`); `Tests/Agents/test_mcp_tool_provider.py`, `Tests/UI/test_mcp_audit_mode.py`, `Tests/Chat/test_console_agent_bridge.py`.

- [ ] **Step 1: Trace** why a fast-button Deny on an MCP row left no execution-log row (the review found 3 approvals, 0 denials): follow the verdict from `ApprovalDecided` → `resolve_pending_approval` → the hook's returned verdict map → whether `MCPToolProvider.invoke` runs at all for a denied call.
- [ ] **Step 2: Failing tests:** (a) a hook-level `deny` verdict for an MCP call produces `record_tool_decision(..., decision="denied")` exactly once; (b) the policy-Off path records `decision="denied-policy"`; (c) `_DECISION_OPTIONS` contains `("Denied by you", "denied")` and `("Blocked (Off)", "denied-policy")` and the Audit filter narrows on each.
- [ ] **Step 3: Implement.** Record at the point the denial becomes final (the hook), through the same service method, with `initiator="agent"`. Rename the Off path's decision to `denied-policy` and add the label. Keep `denied-timeout`/`denied-unresolved`.
- [ ] **Step 4: Run** the three test files. **Step 5: Close, commit.**

### Task 15: Exact-input allow rules: list and remove; honoured wherever offered (task-32281)

**Files:** `tldw_chatbook/MCP/permission_store.py:1483` (`add_tool_arg_rule` — add `list_tool_arg_rules(server_key, tool_name)` and `remove_tool_arg_rule(server_key, tool_name, rule_id)`), `tldw_chatbook/MCP/unified_control_plane_service.py:5162` (mirror the two methods), `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (`_render_permission_container`: a `ds-field-row` per rule `Exact-input allow · {args summary ≤60 chars} — Remove`), `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py` (State cell marker ` ≡` when the tool has arg rules; add to `_LEGEND_TEXT`: `≡ exact-input allows`), `tldw_chatbook/Agents/virtual_cli_provider.py:325-346, 470-502` (offer `options=("approve_once","approve_session","always_allow","deny")` OR honour `allow_matching` via the store's `add_tool_arg_rule` — choose honour if the Virtual CLI rows have stable argument shapes, else narrow; state the choice in the report); `Tests/MCP/test_permission_store.py`, `Tests/UI/test_mcp_permissions_mode.py`, `Tests/Chat/test_console_virtual_cli_approval.py`.

- [ ] **Step 1: Failing tests:** store list/remove round-trip; inspector renders one row per rule and Remove deletes it; matrix marker; Virtual CLI: a `allow_matching` decision either persists a rule that makes the next identical call resolve allow, or the option is absent from its rows.
- [ ] **Step 2: Red. Step 3: Implement. Step 4: Run** the four test files + `Tests/UI/test_mcp_tools_mode.py`. **Step 5: Close, commit.**

### Task 16: Session approvals can be reviewed and revoked (task-32291)

**Files:** `tldw_chatbook/MCP/unified_control_plane_service.py:4769-4860` (`approve_for_session`, `is_session_approved`, `clear_session_approvals` — add `list_session_approvals()` returning `[(server_key, tool_name)]` and `revoke_session_approval(server_key, tool_name)`), `tldw_chatbook/Agents/builtin_tool_gate.py:322` (same for built-ins: list + revoke), `tldw_chatbook/UI/MCP_Modules/mcp_inspector.py` (a `Session approvals` group under the permission container: one row per entry with `Revoke`), `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py` (State cell suffix ` (session)` when session-approved); tests in `Tests/MCP/`, `Tests/Agents/test_builtin_tool_gate*.py`, `Tests/UI/test_mcp_permissions_mode.py`.

- [ ] **Step 1: Failing tests:** list/revoke round-trip on both services; after revoke `is_session_approved` is False; matrix suffix; inspector row + Revoke.
- [ ] **Step 2: Red. Step 3: Implement. Step 4: Run. Step 5: Close, commit.**

### Task 17: Tool gates pane: text state, humanised labels, accurate restart note (task-32284)

**Files:** `tldw_chatbook/Agents/tool_catalog.py:824-890` (`GateableTool` gains `title: str` and `blurb: str`; move the wizard's `_TOOL_COPY` values into the table), `tldw_chatbook/UI/Wizards/FirstRunSetupWizard.py:6561` (read titles from `GateableTool`, delete the local table), `tldw_chatbook/Agents/builtin_tool_gate.py` (`all_tool_gates()` entries carry `title`, `description`, `restart_required: bool`), `tldw_chatbook/UI/MCP_Modules/mcp_servers_mode.py:1215-1235` (replace the compact `Checkbox` with the Library/kill-switch toggle-Button pattern `"{title}: on ▸"` / `"{title}: off ▸"`, tooltip = blurb; the note under the group lists which gates need a restart), `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py` (gate breadcrumb text names the pane: `… under MCP ▸ Servers ▸ built-in row ▸ Tool gates`); `Tests/UI/test_mcp_servers_mode.py`, `Tests/UI/test_first_run_setup_wizard*.py`, `Tests/Agents/test_tool_catalog*.py`.

- [ ] **Step 1: Failing tests:** every `_GATEABLE_BUILTINS` row has a non-empty title; the wizard renders `GateableTool.title`; the Servers pane renders `Read file: off ▸` for a gate that is off and `Read file: on ▸` after a click that persisted `read_file_enabled = true`; the restart note names only construction-time gates.
- [ ] **Step 2: Red. Step 3: Implement. Step 4: Run. Step 5: Close, commit.**

### Task 18: Permissions mode polish (task-32285)

**Files:** `tldw_chatbook/UI/MCP_Modules/mcp_workbench.py:2900-2935, 3080-3145` (preview recompute after a cycle/kill-switch echo), `tldw_chatbook/UI/MCP_Modules/mcp_permissions_mode.py:54-57, 548-561, 585-589, 676` (legend `height: auto` so it wraps; kill-switch hint), `tldw_chatbook/Chat/console_chat_controller.py:1271` and `tldw_chatbook/Agents/mcp_tool_provider.py:95`, `tldw_chatbook/Agents/local_tool_provider.py:120`, `tldw_chatbook/Chat/console_agent_bridge.py:1391` (one kill-switch refusal wording: `tool call blocked: the chat tool kill switch is on`); `Tests/UI/test_mcp_permissions_mode.py`, `Tests/Chat/test_console_agent_bridge.py`.

- [ ] **Step 1: Failing tests:** after cycling a row to Allow the preview reads `1 allow · 29 ask · 0 off`; the legend Static wraps (height ≥ 2 at width 100) and its full text is present; the hint reads `Also blocks the app's own built-in tools (calculator, date/time, file and note tools).`; all four refusal constants equal the locked wording.
- [ ] **Step 2: Red. Step 3: Implement. Step 4: Run. Step 5: Close, commit.**

### Task 19: Tools-mode master control layout (task-32286)

**Files:** `tldw_chatbook/UI/MCP_Modules/mcp_tools_mode.py:200-245`, bundle TCSS rules for the control's container; `Tests/UI/test_mcp_tools_mode.py` (real-bundle harness).

- [ ] **Step 1: Measure** the control's region at 80/120/250 columns in a real-bundle harness; **Step 2: failing test** asserting the checkbox and the label `Enabled` are fully within the region (width ≥ 12) at those widths; **Step 3: fix** (likely `width: auto` / `min-width` on the wrapping container or removing a `Select`-shaped frame); rebuild bundle. **Step 4: Run. Step 5: Close, commit.**

---

## Lane D — docs (branch `fix/approval-wave-d-docs`, dispatched after lanes B and C finish)

### Task 20: Docs match shipped behaviour (task-32290)

**Files:** `Docs/User_Guide/console/agent-runs-and-tools.md:214-300`, `Docs/User_Guide/mcp.md`, `Docs/User_Guide/images/console/approval-card.svg` (regenerate with the lane-B card: mount the card in a 200×50 app and `app.export_screenshot()`-style SVG export, or `textual` Rich export as the existing file was produced), stamps.

- [ ] **Step 1:** read lane B's `chat_approval_card.py` and lane C's inspector/permissions modules in their worktrees (paths given in the dispatch) for the final strings.
- [ ] **Step 2:** rewrite the Approvals section: five decisions with scope copy verbatim; the path-warning string verbatim from `_PATH_PRECHECK_SUFFIX`; "Always allow" applies to MCP and local workspace tools, not built-ins; `denied by you` vs `blocked (Off)` vs `blocked (kill switch)`; Alt+A route; waiting state copy.
- [ ] **Step 3:** mcp.md: replace the stub notice for Permissions mode with: Inherit/Allow/Ask/Off definitions, Space cycling, kill switch label and blast radius, risk floor and the explicit tool-level Allow bypass, exact-input rules and session approvals with their revoke locations, Tool gates pane names.
- [ ] **Step 4:** regenerate the SVG; update both "Verified against" stamps with the lane-B/C branch heads; **Step 5: Close, commit.**

---

## Self-review notes

- Coverage: 32272→T1, 32273→T2, 32274→T3, 32275→T4, 32278→T5, 32282→T6, 32288→T7, 32287→T8, 32277→T9, 32276→T10, 32279→T11, 32289→T12, 32283→T13, 32280→T14, 32281→T15, 32291→T16, 32284→T17, 32285→T18, 32286→T19, 32290→T20.
- Shared files across lanes: `console_chat_controller.py` (A/T4 may touch the bridge; C/T14 and T18 touch controller constants) — lanes merge in order A, C, B, D; conflicts are expected only in the CSS bundle (regenerate) and the controller constants (trivial).
- Interfaces named in later tasks: `DECISION_SCOPE_COPY` (T5, used by T20 docs); `denied` status + `denied-policy` decision (T11, T14, T20); `GateableTool.title/blurb` (T17, used by T12's wizard only if lane C merges first — T12 keeps its local table if `GateableTool` has no `title` attribute at its base; T17 deletes the table).
