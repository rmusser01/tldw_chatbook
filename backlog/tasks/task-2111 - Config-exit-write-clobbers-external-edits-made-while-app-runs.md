---
id: TASK-2111
title: Config exit-write clobbers external edits made while app runs
status: To Do
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

- [ ] Identify every code path that writes the whole config file and document which state
      it serializes (startup snapshot vs freshly-read file).
- [ ] A config write that would DROP sections present on disk but absent from the writer's
      in-memory view either preserves them (read-merge-write) or refuses with a logged
      conflict — external edits are never silently deleted.
- [ ] A regression test pins the scenario: edit-on-disk while "running" (snapshot held),
      then app-triggered save → the on-disk-only section survives.
