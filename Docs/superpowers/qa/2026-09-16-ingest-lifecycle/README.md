# Local Import lifecycle checkpoint — TASK-32700

Baseline: `856cbf355d` on `feat/component-pattern-library`.

**Incomplete.** The real app cannot start its local parse pool on this host:
`multiprocessing` semaphore construction raises errno 28. The real-spawn test
fails at the same allocation both inside and outside the sandbox. The volume
has 75 GiB available and the reported semaphore limit is 10,000. This evidence
does not identify which process or leaked resource owns the exhausted capacity.
The task remains In Progress with all acceptance criteria unchecked.

## Completed evidence

The isolated native app clears an unsubmitted source without creating a job or
media. Submitting a real text file reaches the production pool-creation path
and produces an honest retryable failure. A fresh app process restores that
failure; the mounted Retry control creates a new attempt with durable lineage,
which fails at the same allocation. No parser, pool or writer is substituted.
No successful import is claimed.

Four native captures cover the failure row at 80×24 and 170×48 in textual-dark
and textual-light. The error summary, Show details, Retry and Dismiss are
readable, and Retry has visible focus. Compact capture intentionally scrolls
the queue into view; it does not qualify all surrounding navigation layout.
Final run-003 receipts: [submit](submit-result.json), [restart](restart-result.json),
[inspection](inspection.json) and [lifecycle](lifecycle.json).
[Persistence](persistence.json) records ten healthy private databases, two failed
attempts with lineage, zero media/messages, and unchanged fixture/default-profile
hashes. Earlier run-001 stopped before app startup because the harness had not
created its private data directory; run-002 first established the pool-allocation
blocker and failure/restart behavior. Their raw failures remain in ignored scratch.

[Verification](verification.json) records 76 passing targeted checks covering
registry/database restoration, duplicate resolution, local persistence,
permanent failure, pool-creation failure, source entry and Recent imports.
The runner tests use their existing controlled-pool harness; they are not proof
that the real native pool succeeded. The separate real-spawn test remains
blocked. No full suite ran.

The module documentation now describes the implemented optional durable store,
explicit retry after interruption, atomic retry lineage and supported local
transcription cancellation. This is a documentation correction only: no runtime,
UI, database schema, worker, or dependency behavior changed.
[Static comparison](static-comparison.json) records the same 21 inherited Ruff
findings in the registry before and after this documentation edit. The native
runner and its guard tests are lint-clean; all three edited Python files pass
formatting. Six additional guard tests pass (82 targeted passes in total). They
reject absent/misconfigured profiles and symlink escapes before app imports,
and accept the recorded run-003 profile. The guard was added after the native
confirmation; the native journey itself did not change or rerun. Production AST
is unchanged after excluding docstrings.

[Independent review](review.json) found the missing profile guard and overstrong
persistence wording; both corrections passed its follow-up review.

## Reproduction and remaining work

[native_check.py](native_check.py) is a failure-specific runner. Prepare a fresh
private profile with `data/`, `data/db/` and `config/` directories, every configured
database/data path redirected beneath it, catalog auto-refresh disabled, one
parse worker, and a disposable `sources/alpha.txt`. The runner sets isolated
config/XDG paths and a null keyring, primes the terminal probe, disables analysis
and embeddings through mounted controls, and requires exclusive profile access.
The runner validates these data/database path preconditions before importing
the app; XDG paths alone are insufficient because the app retains its historic
default data location. Run `submit`, wait for its app process to exit, then run `restart` in a new
process against the same profile and owned tmux session. Each phase writes a
unique evidence directory; preserve separate process exit receipts.

This runner expects the observed errno-28 failure. When host allocation recovers,
it must fail its expectation; then execute the remaining successful lifecycle
journeys from the task plan. Do not count that expectation failure as an app
regression. The pending scope is real text/Markdown import and Open in Library,
duplicate imports without extra content, permission failure followed by a
successful Retry, quitting during a real parse, and fresh-process interruption
reconciliation followed by successful Retry.

Resume after a host resource change or on a clean qualification host. The
existing [semaphore diagnosis](../../reviews/2026-09-08-semaphore-allocation-diagnosis.md)
explains the same host failure and the attribution limits. A coordinated Mac
restart is a recovery option after saving other work; no reboot, global resource
cleanup, kernel setting change or termination of unrelated processes was attempted.

ADR required: no. Existing lifecycle/UX contracts are governed by
[ADR-014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[ADR-065](../../../../backlog/decisions/065-active-ingest-source-admission-and-override.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
