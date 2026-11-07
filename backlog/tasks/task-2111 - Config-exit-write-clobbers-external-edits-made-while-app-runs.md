---
id: TASK-2111
title: Config exit-write clobbers external edits made while app runs
status: Done
assignee: []
created_date: '2026-08-03'
labels: [config, data-integrity]
dependencies: []
priority: high
---

## Description (the why)

During the hands-free live gate (2026-08-03), an `[app_tts]` section appended to
`~/.config/tldw_cli/config.toml` while an app instance was running was silently deleted:
the instance (which had loaded the config before the edit) wrote its startup-era in-memory
config back to disk around exit/settings-save, producing a reordered round-trip of the file
with the externally-added sections absent. A second, forgotten instance from three days
earlier would have done the same with a three-day-old view. Any externally-managed config
(hand edits, setup scripts, other tools) is at risk whenever an instance is open.

Evidence: live file at 14:09 was a section-reordered rewrite missing the 13:40 append;
mtime matched the app-exit window; reproduced timeline in the hands-free SDD ledger
(`.superpowers/sdd/2026-08-02-hands-free-loop/progress.md`).

## Acceptance Criteria (the what)

- [x] Identify every code path that writes the whole config file and document which state
      it serializes (startup snapshot vs freshly-read file).
- [x] A config write that would DROP sections present on disk but absent from the writer's
      in-memory view either preserves them (read-merge-write) or refuses with a logged
      conflict — external edits are never silently deleted.
- [x] A regression test pins the scenario: edit-on-disk while "running" (snapshot held),
      then app-triggered save → the on-disk-only section survives.

## Closeout audit 2026-10-11

Verdict: DONE (verified against origin/dev @ a3c47c6c72).

Evidence:
- Every legacy save path now delegates to the atomic read-merge-write primitive `apply_settings_mutation_to_cli_config` (`tldw_chatbook/config.py:8955`, "Atomically apply exact config sets/deletes"): `save_setting_to_cli_config` → `save_settings_to_cli_config` → mutation (`config.py:9440-9494`); `delete_settings_from_cli_config` likewise. No path serializes the startup-era in-memory config any more.
- `Tests/test_config_delete_settings.py::test_other_sections_and_keys_are_untouched` pins that on-disk sections outside the exact mutation survive; `Tests/test_config_persistence_owner.py` pins a single shutdown persistence owner; TASK-26038's `(mtime_ns, size)` stamp (`config.py:6621-6632`, `Tests/test_config_hot_reload.py`) picks up external edits on the read path so the cache never serves a stale view to the writer.
- Primitive introduced in `1f69b124cc`; delete-preserving pin in `a3c039499c`.
- All three ACs ticked.
