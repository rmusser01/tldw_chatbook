# Phase 6.2 Power-User Workflow Replay

Date: 2026-05-05
Task: `TASK-7.2`
Parent Task: `TASK-7`
Branch: `codex/unified-shell-phase6-power-user-replay`
Base: `origin/dev` at `1261b1cb`

<!-- PHASE_6_2_POWER_USER_REPLAY_METADATA:BEGIN -->
```json
{
  "task": "TASK-7.2",
  "parent_task": "TASK-7",
  "persona": "power-user",
  "decision": "power_user_workflows_recorded",
  "entry_path": "running-app-fast-repeat-use",
  "verified_workflows": [
    "home-next-action-to-console",
    "console-live-work-readiness",
    "library-search-rag",
    "library-import-export",
    "console-live-work-follow-through"
  ],
  "speed_paths": [
    "home-primary-action",
    "top-nav",
    "ctrl-p-affordance"
  ],
  "red_contract_result": {
    "passed": 1,
    "failed": 1
  },
  "final_focused_replay_result": {
    "passed": 2,
    "failed": 0
  },
  "final_broader_shell_replay_result": {
    "passed": 67,
    "failed": 0
  }
}
```
<!-- PHASE_6_2_POWER_USER_REPLAY_METADATA:END -->

## Environment

- Runtime: running Textual app through the app test harness with splash disabled.
- Python: project-local virtual environment using Python 3.13.
- Home/config boundary: temporary app test user-data directory created by the harness; no live user database was required.
- Test command: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_power_user_replay.py -q`

## Workflow Matrix

| Workflow | Goal | Steps Attempted | What Worked | Friction | Severity |
| --- | --- | --- | --- | --- | --- |
| Home next action to Console | Start work fast from a configured dashboard. | Launch Home in returning-user mode with model and Library content ready; activate `Start in Console`. | Home routes to Console and Console renders live-work source readiness. | This is fast after setup, but first-run users still get setup guidance instead. | none |
| Console live-work readiness | Confirm Console is a control surface, not just chat. | Inspect Console after Home launch. | Source readiness shows W+C, Workflows, Schedules, RAG, and Artifacts as connected, with ACP and MCP honest unavailable states. | Readiness is visible, but deeper source event streams remain future work. | recoverability |
| Library Search/RAG | Reach RAG from the source hub. | Navigate to Library and activate `Search/RAG`. | Search/RAG opens through the Library action path. | Search/RAG is a Library subroute, so the top-nav Library item stays active and is not a back-to-hub affordance. | workflow-degradation |
| Library Import/Export | Reach source ingestion/export from the source hub. | Navigate to Library and activate `Import/Export Sources`. | Import/Export opens the ingestion route through the Library action path. | Same Library-subroute back affordance limitation as Search/RAG. | workflow-degradation |
| Console live-work follow-through | Follow an active W+C live-work item into details. | Stage a W+C live-work payload, verify the Console status card, activate `Open W+C run`. | Console renders source/title/status/action and opens the subscriptions watchlist-runs context. | The replay uses a staged local payload, not a live watchlist execution. | none |

## Repeated-Use Findings

- Home supports fast repeated use once the user has model setup and Library content: the primary action becomes `Start in Console` rather than setup guidance.
- Top-level nav supports fast hopping between Console and Library, and the visible `More: Ctrl+P` affordance plus registered `ctrl+p` binding preserve command-palette discovery.
- Library subroutes for Search/RAG and Import/Export are reachable, but they inherit the Library top-nav active state. A power user who wants to return to the Library hub from a subroute needs another route path, such as Console/Home then Library or the command palette.
- Console live-work follow-through works for the W+C run-detail action and consumes staged context into the subscription screen.

## Visual Usability Notes

- The Home configured-state primary action is direct and concise.
- Console status cards use explicit source, title, status, recovery, action, and payload rows, which supports recognition over recall.
- Library exposes Search/RAG and Import/Export as sibling action buttons, making the source model visible.
- The Library subroute active-nav behavior is logically consistent with the product model, but it weakens wayfinding for fast repeated source-hub use.

## Keyboard Path Result

The replay verifies the `ctrl+p` command-palette binding exists and the shell exposes `More: Ctrl+P` in the running app. It does not complete a full keyboard-only Tab/Enter sweep; that remains part of the Nielsen closeout.

## Functional Result

Power-user replay completed for five core workflows: configured Home-to-Console launch, Console source-readiness review, Library-to-Search/RAG, Library-to-Import/Export, and W+C live-work Console follow-through into run details.

## Defect Severity

| Finding | Severity | Result |
| --- | --- | --- |
| Missing Phase 6.2 durable evidence before this slice | recoverability | Captured by the red contract and fixed by this document. |
| Library subroutes lack an explicit return-to-Library-hub affordance | workflow-degradation | Recorded for Nielsen closeout; not a P0/P1 because top-level navigation and command palette paths remain available. |
| Full keyboard-only sweep not replayed in this slice | recoverability | Deferred to Nielsen closeout. |

## Deferred Service-Depth Work

- The replay uses local shell contracts and staged live-work payloads; it does not run a live W+C job.
- Search/RAG route reachability is verified, not retrieval quality or embedding execution.
- Import/Export route reachability is verified, not a full import/export transaction.
- MCP and ACP source-readiness rows remain honest unavailable or future-work states for deeper live event production.

## Residual Risk

- Nielsen heuristic closeout is not closed by this evidence.
- A live terminal/manual screenshot sweep can still find layout, clipping, focus-order, or keyboard traversal issues not captured by mounted app harness checks.
- Live server/API, optional dependency, and remote-auth paths are represented by recovery states from earlier phases, not re-exercised here.

## Verification

- Red contract after test creation and test harness correction: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_power_user_replay.py -q`
- Red result: `1 passed, 1 failed`; the failure was `FileNotFoundError` for this missing evidence file.
- Final focused replay after evidence/tracking updates: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_power_user_replay.py -q`
- Final focused replay result: `2 passed`.
- Final broader shell/product-model replay: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_qa_protocol.py Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_screen_navigation.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_unified_shell_phase234_maturity_gate.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_unified_shell_phase6_power_user_replay.py -q`
- Final broader shell/product-model replay result: `67 passed`.
