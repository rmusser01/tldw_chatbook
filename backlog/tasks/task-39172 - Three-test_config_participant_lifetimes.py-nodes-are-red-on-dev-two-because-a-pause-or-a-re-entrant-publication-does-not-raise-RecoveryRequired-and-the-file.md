---
id: TASK-39172
title: "Three test_config_participant_lifetimes.py nodes are red on dev, two because a pause or a re-entrant publication does not raise RecoveryRequired, and the file runs on no PR lane"
status: Done
assignee:
- '@claude'
updated_date: '2026-11-10 18:40'
created_date: 2026-11-08 06:54
labels:
- follow-up
- tests
- backup
- config
priority: low
references:
- Tests/Backup_Recovery/test_config_participant_lifetimes.py
- tldw_chatbook/Backup_Recovery/bootstrap.py
- tldw_chatbook/config.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 12 of the 2026-10-23 burn-down (settings stream, on base fu-base-r12). Checked on dev b590a15afd and on the round-12 stack 9091820bf2, 2026-11-08 (run.sh, `-n 4`, plain, load ~21-27). The same three of the file's nodes fail on both trees:
- `test_two_config_processes_preserve_each_others_keys`: `assert 'serializing' == 'saved'`.
- `test_reentrant_publication_from_derived_scope_refuses_all_effects`: `Failed: DID NOT RAISE <class 'tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired'>`.
- `test_pause_refuses_cached_bootstrap_without_erasing_cache`: `Failed: DID NOT RAISE <class 'tldw_chatbook.Backup_Recovery.bootstrap.RecoveryRequired'>`.

The file carries no profile marker, and its other nodes pass plain. No PR lane runs it.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each node passes on dev, or is retired with the commit that changed the behaviour it pinned named; each red is shown first with the same command
- [x] #2 Each is classified as test drift or a product defect; a pause or re-entrant publication that no longer refuses as designed is a product defect and gets a test that is red before its fix
- [x] #3 The file runs on a PR lane (never the fast PR contract), with three timings recorded
<!-- AC:END -->

## Implementation Plan

1. Re-run the file plain at `-n 4` on base ae9f9e8d21 and record the three reds with full tracebacks.
2. For each, find the commit that changed the pinned behaviour (git log -S on the refusal path) and decide drift or defect against its design note.
3. Defect: fix production with the node red first. Drift: re-pin to the shipped behaviour, keeping the refusal the node was written to prove.
4. Lane the file on a required PR lane (not pr-fast-lane), update the CI contract pin, and time the step three times.

## Implementation Notes

**Base.** ae9f9e8d21. Ran the file plain at `-n 4` through run.sh: 3 failed, 26 passed, the three described nodes (load ~23). After the change, the same command gives 29 passed (load ~70). For each node, an archive of the named commit and of its parent was run through run.sh to place the red.

| Node | Red since | Class | Re-pin |
|---|---|---|---|
| `test_pause_refuses_cached_bootstrap_without_erasing_cache` | 5e56aac410 (TASK-32804.1); green on its parent | drift | `load_cli_config_and_ensure_existence` now serves a warm hit ahead of the admission handshake, on purpose ("the handshake protects the file/build lifecycle, not a pure in-memory cache read"; TASK-35645's `config_read_would_wait` docstring repeats that a pause refuses only cold reads). The node asserts the guarded `_load_cli_config_bootstrap()` still raises `RecoveryRequired` under the pause, the public warm read returns the same cached dict, and the cache is never erased. |
| `test_reentrant_publication_from_derived_scope_refuses_all_effects` | 6ae1ddd69f (task-32901); green on its parent | drift (the refusal is intact) | The default model-cache directory is now made by `secure_private_directory`, natively on POSIX, so the test's `Path.mkdir` hook never fired and nothing re-entered. The hook now also wraps `source.secure_private_directory`. The re-entrant `replace_cli_config` still raises `RecoveryRequired("raw_path_outside_scope")`, with no directory, no config bytes and no cache/generation change. |
| `test_two_config_processes_preserve_each_others_keys` | its own introduction, b5251e9a6e (red there with the same `'serializing' == 'saved'`) | test defect | A save serializes twice under the lock: `expected_raw = tomllib.loads(toml.dumps(...))`, then the write (the child's recorded stacks, `config.py:6270` and `:4855`; the same line was at b5251e9a6e `:7733`). The first child blocked again on the second serialization. It now holds only its first serialization, which is still under the lock. Both keys are kept, and the second process is still shown waiting at the lock. 3 of 3 green. |

AC#2: neither the pause nor the re-entrant publication is a product defect. The pause still refuses the bootstrap as designed, and since 5e56aac410 the design deliberately exempts warm reads. The re-entrant publication is still refused.

Negative controls (temporary, reverted):
- no pause: the pause node fails "DID NOT RAISE" at the guarded bootstrap;
- the re-entry made a no-op: the re-entrant node fails "DID NOT RAISE".

**Lane (AC#3).** The file joins `recovered-suites`' sandboxed block (after the prompt batch suites, before the first bootstrap_profile suite), because it reads config in-process. `RECOVERED_SUITES_TARGETS` is updated to match. The step runs under `--storage-lease-guard`. Step timings (that step's whole pytest command, serial, through run.sh): `recovered-suites`' whole pytest command, serial through run.sh: 943 s, 1088 s and 1020 s wall, 871 passed and 1 skipped each (1-minute load 32-44). In CI the same step before this round ran 733 passed in 307 s (run 42841528703). This round's additions take about 330 s serial locally at most, so the job should stay near half its 20-minute cap.

**Files.** `Tests/Backup_Recovery/test_config_participant_lifetimes.py`, `.github/workflows/derived-artifacts.yml`, `Tests/CI/test_ci_queue_pressure_contract.py`.
