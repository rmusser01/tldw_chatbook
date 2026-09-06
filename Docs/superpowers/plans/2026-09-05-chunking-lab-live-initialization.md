# Chunking Lab live initialization fix

> Execute inline with systematic-debugging and test-driven-development; verify each gate before continuing.

**Goal:** Restore the accepted local authoring workflow when Textual yields during mounting.
**Architecture:** Keep the existing app-owned coordinator and lazy screen worker. Schedule initialization after the first refresh, when mounting has completed; retain teardown guards.
**Tech stack:** Python 3.12, Textual 8, SQLite.
**Spec:** `Docs/superpowers/specs/2026-09-04-chunking-lab-design.md`; TASK-31645 AC18.

ADR required: no new ADR.
ADR path: backlog/decisions/118-chunking-lab-local-execution-and-recovery.md
Reason: correct lifecycle scheduling within the accepted UI/coordinator boundary; no new policy or interface.

## Constraints

Keep v1 single-sample A/B, local execution, lossless configuration, and automatic recovery. Do not change runtime admission, storage formats, dependencies, or global startup. Work in the existing isolated worktree; targeted tests only. Publication/merge is not part of this follow-up.

## One correction and acceptance gate

- [x] Trace the actual live load boundary. Temporary tracing observed `_load` enter and exit with `mounted=False`, message present, and no coordinator.
- [x] Add `test_initial_load_survives_yielding_mount_handler` in `Tests/UI/test_chunking_lab_screen.py`. A yielding inherited mount handler permits worker scheduling before Textual finishes mounting; assert a real coordinator and enabled sample editor after bounded readiness.
- [x] Run that exact test against unchanged production code; require the readiness failure.
- [x] In `tldw_chatbook/UI/Screens/chunking_lab_screen.py`, wrap the existing lazy `run_worker(self._load, ...)` dispatch in `self.call_after_refresh(...)`. Retain existing teardown checks and worker flags. Adjust the existing deferred-worker test to wait for the refresh callback before observing it.
- [x] Rerun the regression, then the full Lab screen/recovery/results selections and scoped Ruff/format/whitespace checks.
- [x] Launch the uninstrumented real app with a disposable profile. Exercise sample entry, advanced JSON/control round trips, B preview, pin A, alter B, Run both, inspect execution metadata, local template save, reopen and process-crash recovery. Preserve actual terminal captures and state evidence. Any newly observed defect requires diagnosis before repair.
- [x] Update verification documentation and the incident lesson; complete TASK-31645 only on actual acceptance evidence. Review the scoped diff before handoff.

Evidence and qualifications: `Docs/Chunking_Lab_UAT_2026-09-05.md`. Independent review accepted the scheduling/test diff. Two non-blocking presentation findings are documented, not repaired. No publication performed.

## User-authorized publication follow-up

The user subsequently requested a PR against dev. This supersedes the publication
exclusion above, not the merge exclusion. Create a dedicated
`codex/chunking-lab-uat-fix` branch, replay only this correction onto fetched dev,
rerun the targeted Lab tests and derived-artifact preflight on the integrated
tree, and publish a PR including the UAT qualifications. Preserve the worktree.
Existing ADR-118 still applies; no new architectural decision.

Publication gate: replay onto dev `e49a7a16d32053434053895ba3559b970ec06289`
completed without conflicts; the production/test diff is unchanged. Fresh
integrated screen/recovery/results selection passed66 in72.88s with the existing
Requests compatibility warning (`pr-integrated-targeted.xml` in the local UAT
evidence directory). All six derived-artifact preflight checks, scoped Ruff,
format and whitespace checks pass. The earlier live UAT remains tied to its
recorded source state, not retroactively attributed to this dev replay.
