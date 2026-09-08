# SQLite orderly-exit qualification gate

Date: 2026-09-07
Task: TASK-31942, implementation Task5
Product baseline: `e8f7ae2ce3c3b6b0b3be69992412d72f7d11d48e`
Status: Implementation paused; independent gate review supports STOP. No alternative
shutdown policy or production workaround is approved by this record.

## Finding

The approved ADR125 terminal-retention contract requires foreign WAL/SHM cohorts
to remain unchanged through both orderly interpreter shutdown and abrupt owned
process exit. An owned actual-repository experiment fails that orderly-exit gate:
both substituted foreign names disappear, and externally held observer FDs have
link count0 (previously1). Their bytes remain readable through those FDs; that
does not undo the unlink or satisfy foreign-file preservation.

Both baseline and helper-shaped ownership reproduce the failure. The helper-shaped
control starts the actual fixed proof helper, removes legacy local immutable SQL
and main/WAL/SHM raw pins only while the original namespace is healthy, rechecks
proof, then kills/reaps that captured helper while retaining its real admission
permit. The actual live SQL handle, repository, directory FD, SHARED store lease
and worker remain owned. Repository close refuses before SQL cleanup.

An atexit observation confirms non-null owner references and both foreign names
still exist; it does not establish worker-thread liveness. After ordinary interpreter exit0, both foreign names are absent and
observer nlink is0. Abrupt captured-child SIGKILL controls preserve names/inodes/
nlink/bytes. Both exit modes release the cooperative store lease. The failure is
localized after that atexit callback and before completed process exit, not after
every possible callback; there is no captured native C stack or syscall trace.

This is an ownership-shape characterization, not an implemented replacement
wrapper, restart_required mapping, Textual shutdown test or complete Task5
qualification. It demonstrates that simply moving raw proof pins to the helper
does not make retained ordinary Python SQLite objects safe at interpreter exit
in this tested ownership control.
Closing legacy pins while SQL is already open also differs from never opening
them, and may affect native lock history. This control does not prove every future
helper-first wrapper must fail; pristine helper-first integration remains unqualified.

## Preserved reproduction and evidence

Exact diagnostic source is adjacent in
`2026-09-07-sqlite-orderly-exit-gate-diagnostic.py.txt`, Git blob
`acf4a816ba75fabb1cfe91e405ea0d2ea6db6094`. It is archived as text so normal test
collection does not silently adopt a knowingly failing design diagnostic.
The original development copy was
`Tests/TTS/test_profile_sqlite_helper_lifecycle.py`; it was removed from collection
after this byte-identical archive and independent review. Reproduction requires restoring
the archived source to that Tests path, preserving repository pytest pre-import
isolation and using only owned temporary data/processes.

Implementer final command:

```sh
../../.venv/bin/python -m pytest -q Tests/TTS/test_profile_sqlite_helper_lifecycle.py -k retained_repository_exit
```

Result: **2 failed, 2 passed, 1 deselected**, one existing dependency warning,
3.02s. Both orderly arms fail foreign preservation; both abrupt arms pass.
Scope: Python3.12.11 / SQLite3.49.1 / macOS. No Windows qualification claimed.
The initial sibling-writer RED in the same source has a cleanup failure and did
not reach its post-close exclusion assertion; do not run the whole diagnostic
file as a qualification suite or count that RED as a demonstrated writer oracle.
Ruff check/format passed for the diagnostic. Product code remains unchanged.

Detailed execution/provenance record during development:
`.superpowers/sdd/2026-09-07-sqlite-lock-safe-private-validation-implementation/task-5-report.md`.
No product forced exit, foreign-path restoration, journal/dependency change,
user-data access, host cleanup, full test suite or PR/merge action occurred.
All reported owned execution sessions and captured children finished.

Independent reviewer reproduced the helper-shaped pair: **1 failed orderly,
1 passed abrupt, 3 deselected** in1.66s with the existing dependency warning.
The review found no observed fixture/cleanup confound invalidating STOP, while
preserving the phase and integration-equivalence limits above. See
[independent review](2026-09-07-sqlite-orderly-exit-gate-review.md).

## Required next decision

Revisit the approved terminal-loss ownership/shutdown contract before implementing
Task5. This record does not authorize forced application exit, unsafe SQL close,
foreign-cohort deletion, moving live transactions into a process, or another
replacement design. Tasks1–4 remain reviewed; Task5–7 and Canvas admission remain
incomplete. Existing semaphore and inventory qualification gaps are unwaived.
