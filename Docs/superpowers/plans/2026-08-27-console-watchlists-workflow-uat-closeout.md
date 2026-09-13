# Console-Driven Watchlists Workflow UAT Closeout Implementation Plan

> Execution: use `superpowers:test-driven-development` for the deterministic
> UAT harness and `superpowers:verification-before-completion` before closing
> the programme task.

**Goal:** Prove on the then-current `origin/dev` that a fresh-profile user can
drive the complete Watchlists-to-briefing workflow from Console, inspect the
same durable state in product surfaces, let the agent consume the briefing, and
understand the external-MCP and Library-skill boundaries.

**Architecture:** One deterministic QA harness drives the real Console agent
loop, provider/catalog/permission layers, Watchlists command/query services,
SQLite migration, operation coordinator, scheduler projection, and local HTTP
feed fixture. Only model planning and briefing prose generation are scripted,
so the test is repeatable without vendor credentials. A separate production
Textual run uses a disposable profile for human/HCI inspection at 160x42 and a
normal terminal. Evidence records the exact Git SHA and redacts private data.

**Tech stack:** Python 3.11+, pytest/pytest-asyncio, real temporary SQLite,
local HTTP fixtures, Textual pilot/detached PTY, JSON/SVG evidence.

**Backlog task:** TASK-22868 after TASK-613 and TASK-22859 through TASK-22867.

**ADR required:** no

**ADR path:** N/A

**Reason:** This task verifies and documents the previously approved contracts;
it introduces no new storage, scheduler, security, or runtime boundary.

## TASK-22868 — Close out the Console-driven Watchlists workflow UAT

### Files

- Create: `Tests/QA/test_console_watchlists_workflow_uat.py`
- Create: `Docs/UAT/2026-08-27-console-watchlists-workflow-round-trip.md`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/README.md`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/evidence.json`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/automated-transcript.txt`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/console-160x42.svg`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/watchlists-160x42.svg`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/library-skill-classification-160x42.svg`
- Create: `Docs/superpowers/qa/console-watchlists-workflow-2026-08/redaction-scan.txt`
- Modify: `Docs/User_Guide/console/agent-runs-and-tools.md`
- Modify: `Docs/User_Guide/watchlists.md`
- Modify: `Docs/User_Guide/library/skills.md`
- Modify: `Docs/User_Guide/mcp.md`
- Modify: `Docs/User_Guide/schedules.md`
- Modify: `Docs/User_Guide/workflows.md`
- Modify: `Tests/Wizards/test_first_run_setup_integration.py`
- Modify: `Tests/UI/test_first_run_wizard_live_contract.py`
- Modify: `Tests/Chat/test_console_onboarding_state.py`

### Step 1: Pin the joined Console workflow in a deterministic harness

Write the QA test as a user-observable sequence, not a chain of private helper
calls. Build an isolated profile directory, config, migrated Subscriptions DB,
permission store, app loop/coordinator, and local HTTP server serving at least
three small threat-news RSS/Atom fixtures. Script model turns that request the
real tool names in this order:

1. `watchlists_create_sources`
2. `watchlists_create_collection`
3. `watchlists_check_sources`
4. `watchlists_get_operation_status` until every check is terminal
5. `watchlists_generate_briefing`
6. `watchlists_get_operation_status` until the briefing is terminal
7. `watchlists_set_briefing_schedule` with `every_24_hours`
8. `watchlists_list_briefings`
9. `watchlists_get_briefing`
10. final assistant synthesis grounded in the returned briefing/provenance

Use the real `ConsoleAgentBridge`, `ConsoleChatController` composition,
`LocalToolProvider`, permission resolver, tool transcript, command/query
services, coordinator, and database. Script only the model's tool choices and
briefing text. Persist definition-hashed Allow for the exact tools in the
fixture; assert this is explicit authorization, not catalog exposure.

Pin canonical source/collection/receipt IDs, at-most-four check concurrency,
completed ordered provenance, redacted URLs, exact 86,400-second cadence,
reload acknowledgement, tool-result visibility to the next model turn, and
the final answer's use of briefing content. Verify the operation survives
Console navigation and that Watchlists/Settings read the same rows.

Run:

```bash
pytest -q Tests/QA/test_console_watchlists_workflow_uat.py -k "console_round_trip"
```

Expected RED before the programme tasks: the command, receipt, schedule, and
full-briefing tools do not all exist.

### Step 2: Prove the external MCP privacy boundary

In the same QA file, compose the standalone local MCP projection against the
existing temporary database and an explicit stored Allow. Assert metadata and
receipt tools are listed and callable, including bounded briefing receipt
metadata. Assert these names are absent from tool discovery and direct dispatch
is rejected:

- `watchlists_create_sources`
- `watchlists_create_collection`
- `watchlists_update_collection_sources`
- `watchlists_check_sources`
- `watchlists_generate_briefing`
- `watchlists_set_briefing_schedule`
- `watchlists_search_items`
- `watchlists_get_item`
- `watchlists_get_briefing`

Inventory and inspect every external serialization surface exercised by the
test: discovery payloads, successful metadata/receipt results, rejected direct
dispatch, validation failures, and external-MCP audit records. Search each for
a unique marker present only in the briefing Markdown and ordered provenance
snapshots; it must be absent. The local Console transcript is expected to hold
the marker after `watchlists_get_briefing`, so also assert that it is not copied
into external audit or receipt records. Reopen the DB read-only and assert the
external call created or modified no file, schema, row, or migration version.

Run:

```bash
pytest -q Tests/QA/test_console_watchlists_workflow_uat.py -k "external_mcp_boundary"
```

### Step 3: Retain latest-dev First Run as a regression prerequisite

Add only missing assertions to the existing targeted tests. At 100x24 and a
normal size, pin detected/selected/configured vocabulary, atomic provider/model
commit plus readback, focus traversal, responsive containment, and blocked
Console intent when setup is incomplete. Do not reopen TASK-21142 through
TASK-21149 or TASK-22281 and do not create a second setup path.

Run:

```bash
pytest -q Tests/Wizards/test_first_run_setup_integration.py Tests/UI/test_first_run_wizard_live_contract.py Tests/Chat/test_console_onboarding_state.py
```

### Step 4: Verify generic skill/framework behavior

Extend the QA test with local fixtures only:

- one root installable skill reaches trust-pending Review;
- one repository with two skill subdirectories requires exact selection;
- one valid non-skill framework returns the generic framework classification
  and supported recovery actions;
- one delayed import refuses a superseding submit and later reports its actual
  result.

Never fetch or install ATHF in the deterministic suite. Its repository may be
used in the manual UAT only to confirm the generic classification, with no
repository-specific code, shell execution, product integration, or hunt
handoff.

Run:

```bash
pytest -q Tests/QA/test_console_watchlists_workflow_uat.py -k "skill or framework or single_flight"
```

### Step 5: Update user documentation with executable vocabulary

Document two concise Console prompt examples: initial setup through an accepted
briefing receipt, and “Read the latest completed briefing for collection X and
summarize it using its cited provenance.” List each approval effect before the
call, the poll tool/arguments returned by accepted work, and the distinction
between accepted, running, complete, empty, failed, and cancelled.

Document bulk authoring and selection keys, partial-result confirmation,
category-specific feed recovery, last-good Artifacts behavior, every-24-hours
interval semantics, app-open/global-gate limitations, and the existing selected
provider/model used by the collection's briefing preset. State that an absent
preset falls back through the existing app/provider defaults and never borrows
the current Console conversation model. Do not call the interval an LLM
“model.”

Document that external MCP sees only approved metadata/receipts, never command
tools or full briefing content. Document generic skill/framework classification
and trust review. Do not suggest that Chatbook creates a hunt hypothesis,
installs an external framework CLI, or hands a briefing to a hunt feature.

### Step 6: Run production-shaped live UAT in disposable state

Fetch/reconcile the branch with the then-current `origin/dev`, record both SHAs,
and use a new disposable config/data directory plus local fixture feeds. Never
open the user's normal subscriptions database on the schema-bump branch.

Drive the real app at 160x42 and a normal terminal through:

- the First Run prerequisite and Console entry;
- Console source/collection/check/generate/schedule/read workflow;
- approval cards and durable receipt cards;
- Watchlists Sources/Runs/Artifacts inspection and Settings schedule gate;
- bulk source modal, multi-selection, partial result, stale refresh, failed
  refresh, Retry, and storage-mismatch recovery;
- Library root skill, multi-skill choice, framework classification, and
  single-flight behavior;
- external MCP discovery/call boundary.

Capture the three named SVGs, a chronological text transcript, and
`evidence.json` containing Git SHA, Python/Textual versions, terminal sizes,
fixture hashes, commands, result counts, and canonical receipt IDs. Do not
record credentials, custom headers, signed queries, full private article text,
database paths, or user profile paths. Run a final pattern scan and store its
zero-match output in `redaction-scan.txt`. Include the same unique briefing-only
marker in that scan and enumerate the evidence files and external serialization
surfaces scanned, so a zero match cannot result from checking the wrong files.

### Step 7: Publish the UAT report and close the task

Write the UAT report with scope, persona walkthroughs, evidence map, resolved
issues, remaining limitations, and explicit out-of-scope statement. Separate
automated deterministic evidence from manual visual/HCI observations. State
that scheduled runs require the application scheduler to be running and that
“every 24 hours” is interval-based from the latest attempt.

Run the full targeted gate:

```bash
pytest -q Tests/QA/test_console_watchlists_workflow_uat.py
pytest -q Tests/Wizards/test_first_run_setup_integration.py Tests/UI/test_first_run_wizard_live_contract.py Tests/Chat/test_console_onboarding_state.py
pytest -q Tests/Tools/test_watchlists_tool_service.py Tests/Tools/test_watchlists_command_service.py Tests/Agents/test_local_tool_provider.py Tests/MCP/test_local_server_tools.py
pytest -q Tests/Subscriptions/test_watchlists_operation_coordinator.py Tests/Subscriptions/test_briefing_service.py Tests/Scheduling/test_scheduler_loop.py Tests/Scheduling/test_briefing_projection.py
pytest -q Tests/Watchlists/test_watchlists_bulk_source_authoring.py Tests/Watchlists/test_watchlists_artifacts_refresh_states.py Tests/Skills/test_skill_package_inspection.py Tests/Skills/test_skills_import.py
ruff check Tests/QA/test_console_watchlists_workflow_uat.py
python tldw_chatbook/css/check_bundle_sync.py
git diff --check
```

Do not run the full repository test sweep unless the user explicitly opts in.
If a targeted command encounters an unrelated latest-dev baseline failure,
record the exact isolated reproduction and do not relabel it as a task failure.

Commit boundary:

```bash
git add Tests/QA/test_console_watchlists_workflow_uat.py Tests/Wizards/test_first_run_setup_integration.py Tests/UI/test_first_run_wizard_live_contract.py Tests/Chat/test_console_onboarding_state.py Docs/UAT/2026-08-27-console-watchlists-workflow-round-trip.md Docs/superpowers/qa/console-watchlists-workflow-2026-08 Docs/User_Guide/console/agent-runs-and-tools.md Docs/User_Guide/watchlists.md Docs/User_Guide/library/skills.md Docs/User_Guide/mcp.md Docs/User_Guide/schedules.md Docs/User_Guide/workflows.md backlog/tasks/task-22868\ -\ Close-out-the-Console-driven-Watchlists-workflow-UAT.md
git commit -m "test: close Console Watchlists workflow UAT"
```

## Plan-level self-review gate

- The harness drives public Console/catalog/domain seams; it does not simulate
  success by writing final rows directly.
- Scripted model turns make the UAT repeatable while tool execution, receipts,
  migration, scheduling, and provenance remain real.
- The agent receives and consumes full briefing content only inside Console.
- External MCP absence is tested in discovery and dispatch, even after Allow.
- Dedicated Watchlists, Settings, and Library screens independently corroborate
  Console receipts.
- First Run is regression-only and no already-landed work is duplicated.
- No hunt workflow or framework-specific product integration is introduced.
- All persistent/live evidence uses disposable state and passes the redaction
  scan.
