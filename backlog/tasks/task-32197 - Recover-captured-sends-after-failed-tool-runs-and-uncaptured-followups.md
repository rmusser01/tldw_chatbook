---
id: TASK-32197
title: Recover captured sends after failed tool runs and uncaptured followups
status: Done
assignee:
  - '@codex'
created_date: '2026-09-09 19:56'
updated_date: '2026-09-10 04:25'
labels: []
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Restore Capture On in conversations stranded by a failed empty tool-run reply, including conversations used through Send without capture afterward, while preserving historical call evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 After a durably failed empty tool run, subsequent captured sends succeed with project instructions on or off, including after successful uncaptured followups and cold reload.
- [x] #2 Recovery proves exact saved message revisions and parent ownership, closed response state, no active checkpoints or intervening captured calls, matching policy and bounded prior tool suffix.
- [x] #3 Active or uncertain delivery, partial failed responses, sidecars, ambiguous siblings, changed saved values and unproven or over-limit history fail closed.
- [x] #4 Earlier trace records remain unchanged and no provider calls or successful outcomes are invented for uncaptured turns.
- [x] #5 Real SQLite/controller/gateway regressions verify recovery, following sends, cold factory and final-binding rejection controls.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes, amendment to existing ADR097.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md
Reason: permit exact durable failed-empty closure and bounded untraced saved history in the existing tool-suffix replacement proof, without changing storage or representing absent calls as captured.
1. Preserve four real-controller reproductions covering project context and uncaptured followups.
2. Amend ADR097 before implementation; generalize the existing bounded closure witness with exact saved revision evidence, avoiding permissive history replacement.
3. Revalidate closed original owner, every uncaptured followup row, absence of live checkpoints/captured intervening calls, policies and suffix ownership at preparation and final binding.
4. Test immutable original calls, successor sends, cold recovery and negative ownership/state/value cases.
5. Run targeted trace tests and independent review; document verified limits.
5. PR review: reproduce explicit Discard followed by a completed Capture Off send and a Capture On successor; retain the narrowly verified RESPONSE_STARTED allowance for the original discarded owner and verify immutable history plus failed-empty negative controls.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Recovered captured sends after a durably failed empty tool run, including exact mixed discarded and complete uncaptured followups. The existing bounded tool-suffix witness now carries transient closed-owner/saved-revision evidence. Shared ownership validation is rechecked at final binding; exact parent/revision/value, closure, policy and span checks reject active or uncertain delivery, partial answers, sidecars, unknown metadata, changed values, ambiguous owners and intervening captures/checkpoints. Earlier captures and uncaptured delivery history are never rewritten or invented. Pending response recovery still uses Discard.

Amended ADR097 before implementation; changed console_trace_runtime.py, console_trace_service.py and console_trace_final_values.py, with one new controller regression module. Basic four cases failed before the fix. GREEN: 95 failed-history/Discard/rendered-system cases, plus four independent RUN_STUCK integrations. Root final standalone recovery module:39 passed; broader final bridge/request/provenance/project/service/runtime/witness selection:611 passed, three cases excluded under two documented pre-existing display-test functions. Real local llama.cpp greeting/calculator/followup capture passed again after combined repair. Independent source/spec review accepted. Owned-source Ruff, scoped formatting and diff checks pass. User guide, main repair plan and live-verification lesson updated.

Validation limits are documented in Docs/superpowers/plans/2026-09-09-local-model-send-repairs.md: existing baseline display/Settings/filesystem tests and an unrelated Mermaid check requiring unavailable pinned Python3.12.11. No full suite, new dependency, schema change, live data mutation or credential change. Private snapshot metadata matches the reproduced failed-empty and mixed untraced chain; complete private-profile replay was not possible without its original workspace/Library environment.
Maintainer review found and fixed the Discard -> successful Capture Off -> Capture On sequence across project on/off and warm/cold cases. Re-read the durable original Discard marker at every validator invocation, including final binding, while preserving all exact owner/closure checks and immutable old captures. Four mutation controls changing only Discard to failed at final binding remain blocked. Independent review: 11 passed; broader run completed the entire failed-run and discarded-run modules without failure. ADR097 and the incident lesson were updated.
<!-- SECTION:NOTES:END -->
