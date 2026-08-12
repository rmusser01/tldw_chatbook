# Effective Config Startup Isolation Design

## Context

TASK-15674 was created after a long-running generated-video UAT observed that the
default user config file changed while the app was launched with
`TLDW_CONFIG_PATH` pointing at a disposable profile. A controlled reproduction on
current `dev` used a disposable `HOME`, all relevant XDG directories, an effective
scratch profile, and a distinct decoy default config. The real Textual startup
normalized only the effective profile and left the decoy byte-for-byte unchanged.

The original cross-profile attribution is therefore not reproduced. The useful
product contract is still worth pinning at the real app boundary, and the merged
UAT/task/lesson wording must distinguish an observed fingerprint change from proof
that this app launch caused it.

## Goal

Close TASK-15674 by adding a regression that proves real app startup honors the
effective config profile, and by correcting the historical documentation to match
the controlled evidence.

## Non-goals

- Change config loading, normalization, or persistence production code.
- Investigate unrelated concurrent writers that may have changed the original file.
- Redesign the config-profile model or introduce a new persistence abstraction.
- Duplicate broad bootstrap coverage already present in the config test suite.

## Required Behavior

1. A subprocess starts the real `TldwCli` Textual lifecycle after establishing a
   fully disposable environment.
2. `TLDW_CONFIG_PATH` identifies an effective scratch config that is distinct from
   the default path derived from the subprocess's scratch `HOME`.
3. The default-path file is a decoy with known bytes. Those bytes are identical
   before and after the app lifecycle.
4. If startup normalization persists defaults, the change occurs only in the
   effective scratch profile. The regression proves this using file identity and
   change/count assertions, not by printing config values.
5. Existing no-override tests remain the control for default-path behavior and are
   run in the focused verification set.
6. Test diagnostics expose only booleans, counts, and sanitized phase labels; they
   do not expose config contents, credentials, or real user paths.

## Test Isolation and Safety

The parent pytest process creates every file beneath `tmp_path`. The subprocess
receives scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and
`TLDW_CONFIG_PATH` before importing application modules. The effective profile also
points `[paths].data_dir` at scratch storage so startup cannot reach the real data
profile.

The subprocess enters and exits `TldwCli.run_test()` to exercise mounted startup
and teardown. The parent process retains the before bytes and performs the final
file comparisons. No real network or user configuration is used.

## Verification

- Capture RED with a focused named regression before any production change.
- Keep production unchanged if current `dev` satisfies the contract.
- Prove the guard is load-bearing by temporarily disabling the effective-profile
  override in the isolated subprocess path; the named regression must fail because
  the decoy changes or the effective profile does not receive normalization. Restore
  the mutation exactly.
- Run only tests related to touched files: the new lifecycle regression and existing
  config import/bootstrap controls.
- Run Ruff on the touched Python test, `py_compile` to a temporary output directory,
  and `git diff --check`.

## Documentation Corrections

Update the generated-video UAT, TASK-3401.14 notes, TASK-15674, and the live
verification lesson to state:

- the original post-run fingerprint drift was real and the precautionary restore
  was appropriate;
- controlled current-`dev` reproduction did not attribute that drift to isolated
  app startup;
- fingerprint drift alone does not identify the writer; use a decoy default profile
  and an isolated effective profile before assigning causality;
- the regression now locks the verified effective-profile boundary.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: this is a regression-only characterization of an existing config-path
boundary. It introduces no new storage, security, provider, runtime, or cross-module
contract decision.
