# Evaluation Private Overrides Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Execution is inline in the current chat, as authorized by the user.

**Goal:** Load current shipped Eval defaults and persist private sparse overrides.

**Architecture:** Adapt the existing YAML loader and raw participant. Keep the
package selector separate from a config-owned override selector and reuse the
current merge, publication and recovery lifetimes.

**Tech Stack:** Python >=3.12, PyYAML, existing private-file and recovery helpers.

**Spec:** `Docs/superpowers/specs/2026-10-04-evaluation-private-overrides-design.md`

ADR required: yes
ADR path: `backlog/decisions/220-evaluation-defaults-and-private-overrides.md`
Reason: change configuration ownership while preserving existing boundaries.

## Global Constraints

- Python >=3.12; retain YAML; add no dependencies or UI.
- Preserve private path and maintenance gates and per-run overrides.
- Targeted tests only; disposable native test roots stay private.

## Review Focus

- Explicit values equal to defaults remain dirty until saved and stay pinned.
- Mutable dictionaries returned by get must not freeze unrelated defaults.
- Exports and failed saves must not publish a false persisted baseline.
- An absent override must neither block recovery nor create a settings file.
- Legacy retained definitions keep authenticated provenance and inactive state.

### Task 1: Loader and selector split

**Files:** Evals/__init__.py, Evals/config_loader.py,
Backup_Recovery/settings_file_participants.py,
Backup_Recovery/raw_participants.py,
Tests/Backup_Recovery/test_eval_private_overrides.py.

**Interfaces:** `_default_config_path() -> Path` stays packaged;
`_override_config_path(config_selector: Path | None = None) -> Path` selects
private state. Default loader getters keep returning effective values.

- [x] Add and run regressions for all five review focus cases applicable to loading.
- [x] Adapt selection, merging and sparse save bookkeeping using existing helpers.
- [x] Run new regressions and existing Eval raw lifetime tests.

### Task 2: Recovery and documentation

**Files:** Evals/recovery.py, existing Eval retained-definition tests,
Evals/README.md and the Backlog task.

- [x] Add failing discovery assertions for private overrides and normal absence.
- [x] Change discovery and canonical-retention checks; preserve legacy provenance.
- [x] Run targeted native recovery tests; attempt and report installed-product checks.
- [x] Run touched-file lint/format, review the diff and record verification limits.

## Execution Record

Verification initially left the implementation in the checkout for review; the
owner subsequently requested a PR against dev. The original defect was reproduced by the first new test against
the actual shipped YAML: private admission refused the checkout and returned
four task types instead of twelve. Initial regressions: 5 failed, 4 passed.

Final command (with a disposable private native test root):

```text
python -m pytest Tests/Backup_Recovery/test_eval_private_overrides.py
  Tests/Backup_Recovery/test_settings_file_participant_lifetimes.py
  Tests/Backup_Recovery/test_eval_retained_definitions.py::test_ordinary_profile_does_not_create_retained_authority
  Tests/Backup_Recovery/test_domain_owners.py::test_eval_default_definition_selector_is_shared_and_import_light
  Tests/Backup_Recovery/test_domain_owners.py::test_yaml_definition_schema_refuses_unsafe_or_nonportable_values
  -q --no-cov --tb=short
  -k "(eval or yaml_definition_schema or installed_default_selector) and not independent_maintenance"
```

Result: 29 passed, 35 unrelated/platform-incompatible cases deselected. Four
native ACL policy baseline tests also passed before implementation. New loader,
selector and regression files pass Ruff lint/format. Comparing all touched Python
files to HEAD finds no new lint diagnostics (54 inherited diagnostics).

Fresh-context whole-change review found two Important issues. Both were reproduced
RED, corrected and verified GREEN in the final run:

- Initial private-read failure must retain the shipped baseline so saving one
  change does not pin unrelated defaults.
- Full rollback must omit only the exact canonical `unused` declaration, while
  continuing to reject noncanonical missing legacy sources.

Task 1: complete. Task 2: complete for the targeted change; installed-product
release verification remains subject to the limit recorded below.

Final review ruling: leave the unchanged Windows publication/subprocess issues
outside this Eval change. Full generation fixtures fail before Eval assertions
at directory-flush handle reopening (`WinError 5`); old maintenance helpers use
POSIX `select(pipe)`, and POSIX mode assertions do not describe Windows ACLs.
The cost is unverified end-to-end restore/rollback on this host; the new rollback
selection regression uses real strict receipt parsing without substituting the
private safety boundary. No guards or durability calls were weakened.

Final review ruling: scope static analysis to new diagnostics and formatted new
code, preserving unrelated inherited formatting/lint. The cost is that the
repository's existing diagnostics remain. No minor review findings were deferred.

### Subsequent Windows Verification — TASK-34408

The user authorized fixing the two demonstrated Windows blockers separately.
The corrected publication creation/receipt protocol and pipe helpers now permit
the deeper native fixtures to finish without relaxing native checks.

Final native recovery result: **124 passed, 2 skipped, 1 deselected**. This includes
all private override cases, retained/selective/rollback/later-snapshot cases and
the fresh-process present/absent settings recovery-and-owner-reopen matrix for
finish and rollback. Final pipe/admission/lifetime/selector/schema result:
**100 passed**. The two runs contain 211 distinct passing cases. The two skipped
file-symlink cases require a Windows privilege this account lacks; the separate
complete-rebackup fixture is blocked by unavailable/overlapping owner inventory
before publication on this host.

The installed-wheel two-profile flow installed and entered restore, but exceeded
its existing 300-second timeout; that wider release gate remains unverified.
No full sweep or POSIX execution was performed. The combined 21-file static
comparison adds zero lint diagnostics (134 inherited); new files and changed
production ranges pass formatting, all files parse, and whitespace checks pass.
Fresh review has no remaining substantive findings. Full evidence and the
ADR-126 amendment are linked from TASK-34408 and
`Docs/superpowers/plans/2026-10-04-windows-recovery-verification-fixes.md`.

PR preparation against dev a7d9bca5da: 38 focused regressions passed again in 9.11s. Task IDs were voluntarily renumbered to TASK-34407/TASK-34408 to avoid published peer claims; task records retain provenance. The diagnostic inventory was regenerated after statement review: only the existing selected-path loaded message changes from info to debug with revised wording; five calls and sinks are unchanged. Clean tracked-source profile census passes.
