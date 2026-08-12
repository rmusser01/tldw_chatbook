# Effective Config Startup Isolation Design

## Context

TASK-15674 was created after a long-running generated-video UAT observed that the
default user config file changed while the app was launched with
`TLDW_CONFIG_PATH` pointing at a disposable profile. A controlled reproduction on
current `dev` used a disposable `HOME`, all relevant XDG directories, an effective
scratch profile, and a distinct decoy default config. The real Textual
startup-to-approved-quit lifecycle persisted only the effective profile and left
the decoy byte-for-byte unchanged.

The original cross-profile attribution is therefore not reproduced. The useful
product contract is still worth pinning at the real app boundary, and the merged
UAT/task/lesson wording must distinguish an observed fingerprint change from proof
that this app launch caused it.

## Goal

Close TASK-15674 by adding a regression that proves the real app's
startup-to-approved-quit lifecycle honors the effective config profile, and by
correcting the historical documentation to match the controlled evidence.

## Non-goals

- Change config loading, normalization, or persistence production code.
- Investigate unrelated concurrent writers that may have changed the original file.
- Redesign the config-profile model or introduce a new persistence abstraction.
- Duplicate broad bootstrap coverage already present in the config test suite.

## Required Behavior

1. A subprocess starts the real `TldwCli` Textual lifecycle after establishing a
   fully disposable environment, then completes the approved quit path that owns
   shutdown config persistence.
2. `TLDW_CONFIG_PATH` identifies an effective scratch config that is distinct from
   the default path derived from the subprocess's scratch `HOME`.
3. The default-path file is a decoy with known bytes. Those bytes are identical
   before and after the app lifecycle.
4. A test-local wrapper records that `persist_cli_config_for_shutdown()` ran and
   returned successfully during approved quit. The decoy must always remain
   byte-identical. If the effective profile bytes change, only that effective path
   may change; an idempotent/no-op persistence remains valid.
5. The existing no-override default-path controls
   `test_default_application_config_directory_is_created_as_0700` and
   `test_existing_default_config_directory_is_hardened_before_read` remain the
   focused control for default-path creation and hardening. This work does not claim
   they are a second real-app lifecycle test.
6. Test diagnostics expose only booleans, counts, and sanitized phase labels; they
   do not expose config contents, credentials, or real user paths.

## Test Isolation and Safety

The parent pytest process creates every file beneath `tmp_path`. The subprocess
receives scratch `HOME`, `XDG_CONFIG_HOME`, `XDG_DATA_HOME`, `XDG_CACHE_HOME`, and
`TLDW_CONFIG_PATH` before importing application modules. The effective profile also
points `[paths].data_dir` at scratch storage so startup cannot reach the real data
profile.

The effective scratch config sets `[model_catalog] auto_refresh_enabled = false`
before application imports, so the unconditional startup worker cannot perform a
provider refresh. Other configured storage paths remain scratch-only.

The subprocess enters `TldwCli.run_test()` to exercise mounted startup, then drives
`_confirm_and_quit()` so the production approved-quit flow reaches
`_run_blocking_quit_persistence()` and exits normally. After importing `app.py`, a
test-local wrapper replaces `tldw_chatbook.app.persist_cli_config_for_shutdown`
(the symbol actually called by `_run_blocking_quit_persistence()`), delegates to
the real `tldw_chatbook.config.persist_cli_config_for_shutdown`, and records only
whether it ran and whether it returned successfully. The parent process retains the
before bytes and performs the final file comparisons. No real network or user
configuration is used.

## Verification

- Keep production unchanged if current `dev` satisfies the contract. The required
  RED evidence is the temporary production lookup mutation below; unmodified
  current `dev` is expected to pass the characterization.
- Prove the guard is load-bearing by temporarily mutating the production
  `_get_effective_config_path()` / `get_cli_config_path()` lookup so it ignores
  `TLDW_CONFIG_PATH` before the subprocess imports application modules. The named
  lifecycle regression must fail because the wrong profile is selected. Restore the
  source mutation exactly; merely deleting the test's environment variable does not
  count.
- Run only tests related to touched files: the new lifecycle regression and existing
  config import/bootstrap controls.
- Run Ruff on the touched Python test, `py_compile` to a temporary output directory,
  and `git diff --check`.

## Documentation Corrections

Update the generated-video UAT, TASK-3401.14 notes, TASK-15674, and the live
verification lesson to state:

- the original post-run fingerprint drift was real and the precautionary restore
  was appropriate;
- controlled current-`dev` reproduction did not attribute that drift to the
  isolated startup-to-approved-quit lifecycle;
- fingerprint drift alone does not identify the writer; use a decoy default profile
  and an isolated effective profile before assigning causality;
- the regression now locks the verified effective-profile boundary.

## Architecture Decision Record

ADR required: no

ADR path: N/A

Reason: this is a regression-only characterization of an existing config-path
boundary. It introduces no new storage, security, provider, runtime, or cross-module
contract decision.
