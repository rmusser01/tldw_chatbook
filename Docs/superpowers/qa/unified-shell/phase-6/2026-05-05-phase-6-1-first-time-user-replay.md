# Phase 6.1 First-Time User Replay

Date: 2026-05-05
Task: `TASK-7.1`
Parent Task: `TASK-7`
Branch: `codex/unified-shell-phase6-first-time-audit`
Base: `origin/dev` at `2a8dd1e1`

<!-- PHASE_6_1_FIRST_TIME_REPLAY_METADATA:BEGIN -->
```json
{
  "task": "TASK-7.1",
  "parent_task": "TASK-7",
  "persona": "first-time-user",
  "decision": "first_time_walkthrough_recorded",
  "entry_path": "clean-first-run-home",
  "verified_routes": [
    "home",
    "console",
    "library",
    "personas",
    "skills"
  ],
  "orientation_paths": [
    "library",
    "personas",
    "skills"
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
    "passed": 65,
    "failed": 0
  }
}
```
<!-- PHASE_6_1_FIRST_TIME_REPLAY_METADATA:END -->

## Environment

- Runtime: running Textual app through the app test harness with splash disabled and first-run routing enabled.
- Python: project-local virtual environment using Python 3.13.
- Home/config boundary: temporary app test user-data directory created by the harness; no live user database was required.
- Test command: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q`

## Entry Path

Clean first-run launch from configured `chat` default. The shell correctly routes first-time startup to Home instead of dropping the user directly into Console.

## Steps Attempted

1. Launch the app from a first-run state with `app.app_config["_first_run"] = True` and `_initial_tab_value = "chat"`.
2. Wait for the active screen to become `HomeScreen`.
3. Verify top-level navigation order and labels: Home, Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, Settings.
4. Verify Home explains dashboard/status/notifications/active-work/next-action purpose and exposes the next action `Set up Console model`.
5. Activate Console from the top nav and verify live-work source readiness copy is visible.
6. Activate Library and verify source orientation paths, including Import/Export and Search/RAG.
7. Activate Personas and verify behavior-profile orientation plus Console attachment affordance.
8. Activate Skills and verify Agent Skills orientation plus `SKILL.md` model copy.

## Visual Usability Notes

- Home has a clear title, purpose line, status sections, active-work summary, next-best action, and recent-work empty state.
- The top-level product model is visible without prior knowledge: Console, Library, Artifacts, Personas, W+C, Schedules, Workflows, MCP, ACP, Skills, and Settings.
- `W+C` remains compact but still appears in a stable top-nav position; existing navigation metadata keeps its full meaning available through tooltip/full-label tests.
- Console reads as a control surface because live-work source readiness appears before generic chat controls.
- Library, Personas, and Skills each explain ownership boundaries in plain product language rather than exposing only implementation labels.

## Keyboard Path Result

The focused replay verifies deterministic button activation, top-nav order, and `More: Ctrl+P` visibility. It does not close the full keyboard-only Tab/Enter sweep; that remains a Phase 6 power-user and Nielsen closeout check.

## Functional Result

First-time orientation workflow completed for Home, Console, Library, Personas, and Skills. The user can understand where to begin, how Console relates to live work, where source/RAG actions live, where persona behavior belongs, and where Agent Skills are managed.

## Defect Severity

| Finding | Severity | Result |
| --- | --- | --- |
| Missing Phase 6.1 durable evidence before this slice | recoverability | Captured by the red contract and fixed by this document. |
| First-time Home-to-Console orientation | none | Home shows status and next action; Console exposes live-work readiness. |
| Library, Personas, Skills orientation | none | Each destination has visible owner/purpose copy and stable buttons. |

## First-Time Onboarding Gaps

- No P0 blocker was found in this focused replay.
- The app still relies on destination purpose copy rather than a dedicated guided tour. This is acceptable for Phase 6.1 because Home provides next-best action and the nav model is visible, but the later Nielsen closeout should decide whether skippable guidance is needed.
- Keyboard-only discoverability is not fully closed by this slice; it must be replayed in the remaining Phase 6 power-user/Nielsen work.

## Deferred Service-Depth Work

- This walkthrough intentionally verifies first-time orientation and destination affordances, not full service CRUD parity.
- Library detail views, embedded Import/Export depth, and embedded Search/RAG completion remain service-depth work outside this first-time replay.
- Personas import/export, archetypes, exemplars, dictionaries, lore, and edit flows remain service-depth work.
- Skills import, validation, editing, execution, and server skill flows remain service-depth work.
- ACP complete runtime operation remains outside this first-time replay.

## Residual Risk

- Power-user repeated workflows are not closed by this evidence.
- Nielsen heuristic closeout is not closed by this evidence.
- A live terminal/manual screenshot sweep can still find layout or focus issues that mounted app harness checks do not capture.
- Live server/API, optional dependency, and remote-auth paths are represented by recovery states from earlier phases, not re-exercised here.

## Verification

- Red contract after test creation: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q`
- Red result: `1 passed, 1 failed`; the failure was `FileNotFoundError` for this missing evidence file.
- Final focused replay after evidence/tracking updates: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_phase6_first_time_replay.py -q`
- Final focused replay result: `2 passed`.
- Final broader shell/product-model replay: `.venv/bin/python -m pytest Tests/UI/test_unified_shell_qa_protocol.py Tests/UI/test_shell_product_model_visibility.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_screen_navigation.py Tests/UI/test_shell_chrome_contract.py Tests/UI/test_unified_shell_phase234_maturity_gate.py Tests/UI/test_unified_shell_phase5_recovery_taxonomy.py Tests/UI/test_unified_shell_phase6_first_time_replay.py -q`
- Final broader shell/product-model replay result: `65 passed`.
