---
id: TASK-32762
title: Preserve permission summary settings through save recovery and navigation
status: Done
assignee:
  - '@codex'
created_date: '2026-09-17 23:33'
updated_date: '2026-09-17 23:59'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Permission-summary preferences must accurately report what is saved and retain the latest edit when users navigate away or recover from a failed write.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The latest permission summary mode, provider and model survive overlapping saves and leaving or reopening Settings.
- [x] #2 Failed saves show an inline result, preserve the edited values and offer a keyboard-accessible retry without silently enabling summaries.
- [x] #3 Opening or revisiting unchanged settings does not write configuration; unrelated permission summary keys and the external-content disclosure are preserved.
- [x] #4 Targeted private-profile regression tests and compact/wide visual evidence cover the changed workflow; task and review ledger record the exact limits.
- [x] #5 The runtime permission-summary consumer reads the same saved mode, provider and model as Settings, without changing consent or invoking a model during verification.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/090-permission-request-context-summaries.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: repair existing immediate-save behavior within the approved settings and consent boundaries.

1. Reproduce late-write, failed-save and Settings-recreation defects using private-profile mounted UI tests.
2. Serialize app-lifetime writes, preserve pending or failed values, and add an inline result and Retry using existing component classes.
3. Verify no-op viewing, unrelated config preservation, cache-refresh failure reporting, and compact/wide keyboard access. Verify the saved runtime projection used by the approval consumer.
4. Record native visual evidence, targeted checks and review results; update the completion ledger and draft PR without merging.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Permission-summary mode/provider/model now use an app-lifetime serialized writer with retained failed edits, inline result/Retry, correct focus guidance, and post-replacement cache recovery. The runtime configuration now exposes the saved section to both Settings and the existing approval resolver. A transaction-bound publication generation prevents a later config writer from leaving stale saved form values.

Modified settings_screen.py, the config projection, focused UI regressions, Settings user guidance, testing lessons, and the completion/conflict reports. Existing token-backed classes suffice; consent disclosure, default Off, unrelated config keys and provider execution boundaries remain unchanged. ADR required: no; implements ADR-090 permission summaries and ADR-150/161.

Verification: 15 private-profile UI cases pass; 52 related service/governance/bundle/boot cases pass. Six old approval-wiring failures reproduce at saved baseline 5ac787ce6a; no assertion was removed or skipped, and these failures remain explicitly outside this Settings qualification. All seven derived-artifact checks pass, inventory is unchanged, no new lint findings, and the new test/runner format checks pass. Independent review is clear after two cache-recovery regressions. No full suite was run.

Final native run 003 (PID 31992) passes dark/light at 190x55 and 80x24: real private-path save refusal, retained opt-in with runtime Off, keyboard retry, model save, navigation and restoring Off. Twelve SVGs are inspected; app exits cleanly, 11 private DBs pass integrity checks, default fingerprints and seven live DB identities remain unchanged. Four expected PrivatePathError records are retained; no external model call occurred. Evidence and exact limits: Docs/superpowers/qa/2026-09-17-settings-permission-summary/README.md.

Plan expansion: mounted tests exposed the missing runtime projection, added as AC5 before implementation. Native inspection exposed Retry guidance falling back to generic staged copy; fixed and reverified. The broader migration review remains active; PR #2704 remains draft and cannot merge without the user's explicit visual approval.
<!-- SECTION:NOTES:END -->
