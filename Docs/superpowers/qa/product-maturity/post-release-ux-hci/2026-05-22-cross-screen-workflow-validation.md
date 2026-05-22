# Post-Release UX/HCI Walkthrough Evidence: Cross-Screen Workflows

## Metadata

- Task: `TASK-60.3`
- Screen or workflow: Cross-screen workflow validation
- Date: 2026-05-22
- Branch: `codex/continue-screen-qa`
- App command: mounted Textual app regressions against current `origin/dev`; prior screen visuals captured through textual-web/CDP in `actual-screenshots/`
- Evidence method: focused mounted Textual workflow tests plus existing actual screenshot evidence; no visible UI changed in this slice
- Actual screenshot path: existing screen captures under `Docs/superpowers/qa/product-maturity/post-release-ux-hci/actual-screenshots/`
- Screenshot approval: no new screenshot required for this docs/test-only workflow closeout
- Reviewer: workflow evidence pending PR review

## User Goal

Verify that the top-level destinations work together as a product system, especially when handing context into Console as the primary agentic control surface.

## Steps Attempted

1. Refreshed current `origin/dev` after PR #344 merged.
2. Reviewed existing post-release screen evidence and stale cross-screen follow-up notes.
3. Ran focused mounted Textual regressions for Home, Library, Artifacts, Personas, Skills, ACP, Watchlists, Schedules, Workflows, and Console source-readiness paths.
4. Classified each workflow as complete, recoverably blocked, or future service-depth work with visible recovery copy.

## Workflow Matrix

| Workflow | Status | Evidence | Friction And Recovery | Severity |
| --- | --- | --- | --- | --- |
| Home primary action opens target route | Verified | `test_home_primary_action_opens_target_route` | Fast path is one action; clean dashboard remains sparse but routes to Library instead of dead-ending. | P2 residual orientation only |
| Home active work opens Console with payload | Verified | `test_app_console_hook_opens_console_with_adapter_launch_payload` | Adapter payload preserves source/status/recovery/action metadata for Console. | None |
| Home mixed Watchlists plus Chatbook controls | Verified | `test_home_mixed_active_work_exposes_chatbook_artifact_resume_controls` | Chatbook remains reachable even when Watchlists active work exists. | None |
| Library Search/RAG mode and source handoff | Verified/recoverable | `test_library_search_action_switches_to_search_mode_without_route_handoff`, `test_library_use_in_console_uses_source_snapshot_context` | Empty source state blocks safely; populated source snapshot can stage Console context. | None |
| Artifacts/Chatbook resume to Console | Verified | `test_artifacts_destination_reopens_console_saved_chatbook_with_provenance` | Saved Chatbook provenance is preserved in the Console launch payload. | None |
| Personas and Skills attach to Console | Verified | `test_personas_attach_to_console_uses_listed_behavior_context`, `test_skills_attach_to_console_uses_listed_skill_context` | Attach paths use selected/listed context and avoid fake handoffs when context is unavailable. | None |
| ACP session follow into Console | Verified with fixture-backed session payload | `test_acp_session_payload_enables_console_follow_live_work_handoff` | Missing runtime remains an honest blocked state; session payload enables follow. | Future runtime depth |
| Watchlists, Schedules, Workflows run follow | Verified with adapter-backed run context | `test_watchlists_destination_routes_latest_active_run_to_console`, `test_schedules_destination_routes_latest_active_run_to_console`, `test_workflows_destination_routes_latest_active_run_to_console` | Empty states remain disabled; populated adapter context routes to Console. | None |
| MCP source readiness in Console | Classified future service-depth | `test_console_renders_source_readiness_summary_without_pending_launch` | Console states MCP is not embedded and directs server management to MCP; no fake Console tool run exists. | Future MCP runtime depth |

## Nielsen Norman Heuristic Findings

- Visibility of system status: strong for handoff readiness; Console shows source readiness and destination screens disable unavailable follow actions.
- Match between system and real world: source, run, artifact, persona, skill, and ACP session language maps to user-visible product objects.
- User control and freedom: blocked states keep users in the current destination and explain the recovery path instead of navigating unexpectedly.
- Consistency and standards: handoff payloads now use shared Console live-work contracts instead of per-screen ad hoc launches.
- Error prevention: empty, unavailable, or missing-runtime states disable unsafe Console follow actions.
- Recognition rather than recall: recovery copy names the missing source, runtime, run, or artifact prerequisite.
- Flexibility and efficiency of use: repeated workflows are single-action once a source/run/artifact exists; service-depth gaps remain explicit future work.
- Aesthetic and minimalist design: no new UI was introduced; this pass validates existing approved destination shells.
- Error recognition, diagnosis, and recovery: missing-provider/source/runtime states are visible and preserve user work.
- Help and documentation: evidence now distinguishes verified workflow support from future MCP/runtime/server depth.

## Power-User Repetition Findings

| Repeated Workflow | Qualitative Timing | Shortcut/State Notes |
| --- | --- | --- |
| Home -> Library next action | Fast | One primary action; useful for empty profile onboarding and repeated source import starts. |
| Home active work -> Console | Fast when adapter context exists | Launch payload preserves action label, recovery, and target route. |
| Library source -> Console | Moderate | Requires source availability/selection; empty Library blocks honestly. |
| Artifacts Chatbook -> Console | Fast once artifact exists | Latest/requested Chatbook selection is deterministic. |
| Watchlists/Schedules/Workflows -> Console | Fast with active adapter item | Empty states are safe but require upstream run creation. |
| Personas/Skills -> Console | Fast with valid local context | Invalid/empty service states block honestly. |

## Severity Decisions

| Finding | Severity | Follow-Up Task | Decision |
| --- | --- | --- | --- |
| MCP configured-server tool execution remains service-depth future work | P2 | `TASK-60.4` | Defer to feature tranche planning because the UI is honest and non-actionable rather than misleading. |
| ACP real runtime launch remains future service-depth work | P2 | `TASK-60.4` | Fixture-backed session payload proves the Console handoff contract; runtime launch planning remains deferred. |
| Empty Library Search/RAG cannot produce evidence until sources exist | P2 | `TASK-60.4` | Accept as recoverable blocked state; deeper source/citation/snippet carry-through belongs to deferred feature planning. |

No unresolved P0/P1 findings remain for `TASK-60.3`.

## Verification

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_home_screen.py::test_home_primary_action_opens_target_route Tests/UI/test_home_screen.py::test_app_console_hook_opens_console_with_adapter_launch_payload Tests/UI/test_home_screen.py::test_home_mixed_active_work_exposes_chatbook_artifact_resume_controls Tests/UI/test_destination_shells.py::test_library_use_in_console_uses_source_snapshot_context Tests/UI/test_destination_shells.py::test_library_search_action_switches_to_search_mode_without_route_handoff --tb=short
```

Result: 5 passed.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_destination_shells.py::test_personas_attach_to_console_uses_listed_behavior_context Tests/UI/test_destination_shells.py::test_skills_attach_to_console_uses_listed_skill_context Tests/UI/test_destination_shells.py::test_acp_session_payload_enables_console_follow_live_work_handoff Tests/UI/test_console_live_work_handoffs.py::test_schedules_destination_routes_latest_active_run_to_console Tests/UI/test_console_live_work_handoffs.py::test_workflows_destination_routes_latest_active_run_to_console Tests/UI/test_console_live_work_handoffs.py::test_watchlists_destination_routes_latest_active_run_to_console Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_provenance Tests/UI/test_console_live_work_handoffs.py::test_console_renders_source_readiness_summary_without_pending_launch --tb=short
```

Result: 8 passed.

## Acceptance Decision

- Accepted: yes for `TASK-60.3` workflow-validation scope.
- Reason: required handoffs are verified or honestly classified as recoverable/future service-depth work with user-visible copy.
- Remaining follow-up: `TASK-60.4` plans deferred feature tranches for MCP/ACP runtime depth, Workspaces/Library depth, write sync, and citation/snippet carry-through.
