---
id: TASK-22216
title: >-
  Move the actor-pack staging sweep out of TldwCli.__init__
status: To Do
assignee: []
created_date: '2026-08-24'
labels:
  - performance
  - startup
priority: medium
dependencies: []
---

## Description

Source: holistic performance review of dev `a71e62e4b` (2026-08-24). Evidence, measurements,
and full file:line cites: `Docs/Design/2026-08-24-holistic-perf-review.md` (finding 22216).

PR #1998 (`ac1037732`) put synchronous filesystem work back into construct — the class
task-21106 removed: `app.py:7322` constructs `ActorPackImportService`, whose `__init__`
ends with `self.sweep_staging()` (`Actor_Packs/importer.py:216`). The sweep runs
`secure_private_directory(..., create=True)` — a per-component walk from `/` with
`os.open`+`fstat` owner/mode checks (`Utils/private_paths.py:995-1030`) — then `os.scandir`
over up to 32 staging candidates, each with lstat + two O_NOFOLLOW opens (`importer.py:
218-255`, `:1313`), every boot, before the event loop exists. Small-medium warm; medium on
network/FUSE homes or with residue. The boot-files guard cannot see it (it asserts six DB
filenames, and this is a directory).

## Acceptance Criteria

- [ ] `TldwCli.__init__` performs no staging filesystem I/O (probe or guard asserts the sweep is not reached from construct)
- [ ] The sweep runs on first import-feature use or a deferred worker; crash-recovery semantics preserved (staging residue still cleaned within the session)
- [ ] The boot-time guard is extended to cover this class (construct-time filesystem side effects), with its blind spots stated
