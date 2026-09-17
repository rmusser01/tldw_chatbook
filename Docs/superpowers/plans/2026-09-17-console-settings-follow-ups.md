# Remaining Console investigation issues

The user authorized all remaining identified issues: three stale regression
failures and synchronous subscription readiness in Settings, first-run setup,
and persona handoff. Preserve the preceding repairs and unrelated local edits.

ADR required: yes (amend existing ADR-012)
ADR path: backlog/decisions/012-provider-credential-settings-boundary.md
Reason: final review found inactive API-key deletion during subscription setup.
Clarify the shared writer's sparse no-change credential operation under the
existing ownership boundary. ADR-052, ADR-095 and ADR-149 also govern the
regression fixtures; no new storage, dependencies or external service contracts.

## TASK-32710: regression evidence

- [x] Reproduce identity-field, compaction-close fixture and first-persistence
  failures; verify each expectation against the existing production contract.
- [x] Repair fixtures/assertions without dropping their ownership or lifecycle
  checks; run the affected tests and relevant neighboring cases.

## TASK-32711: responsive provider UI

- [x] Reproduce slow credential I/O from canonical Settings, first-run provider
  setup and persona handoff readiness using controlled thread gates.
- [x] Use bounded background snapshots for UI; preserve credential resolution
  for actual requests and mutation ownership for saves.
- [x] Refresh pending completion and expiry automatically, guarding current
  selection and mounted ownership. Keep recovery copy accurate and secret-free.
- [x] Run mounted regressions, request/commit checks and scoped static checks.
- [x] Preserve inactive API-key fields and credential-source configuration on
  unchanged subscription setup; retain explicit Clear/replacement behavior and
  the shared writer's issued-mutation and compare-and-swap protections.

## Integration

- [x] Run all three formerly failing cases and the new UI regressions together.
- [x] Review diffs, update the investigation and task records, and close only
  after targeted verification. The user subsequently authorized committing and
  publishing a PR against dev. Do not merge or run the full suite.

Independent agent ownership: regression agent owns the two Console test files;
Settings agent owns settings_screen.py and new Settings responsiveness tests;
persona agent owns persona controller/screen and new responsiveness tests;
primary agent owns wizard/state and final integration.

Final review added the sparse subscription-preservation repair (regression
agent, shared writer/state) and a header/inspector completion race (persona
agent), with failing regressions before each fix. Combined final run: 73 passed;
writer/state/readiness/preservation: 446 passed. Full details and overlapping
neighbor runs are recorded in the investigation report and TASK-32711 notes.

PR integration uses an isolated branch based on current dev and preserves its
endpoint identity helpers, sampling enum ownership, cleanup, and modal geometry.
Real-config regressions use dev's private-profile child-process harness without
changing production source-selection guards. Final integration also covers
idle credential expiry/TTL renewal and discarded discovery results after modal
dismissal. Fresh evidence and baseline limitations are recorded in the report.
