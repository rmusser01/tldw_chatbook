# Local Import lifecycle qualification — TASK-32700

Baseline: `856cbf355d` on `feat/component-pattern-library`.

**Completed: real local success and restart recovery now pass.** See the
[resumed qualification](resume/README.md) for four native processes, text/Markdown
imports, duplicate resolution, successful Retry, interruption reconciliation,
fresh-process Open, 16 inspected captures and 83 passing targeted checks. The
resumed record includes the unrelated caught Chat sidebar startup error and
explicit scope limits. No additional application code change was needed.

The historical checkpoint below preserves the earlier failure evidence.
The semaphore blocker cleared on 2026-09-16 at 19:31 UTC.
A fresh isolated spawn-context Lock probe and the previously failing real
spawn-pool parser test both passed outside the sandbox (one test in 0.70s).
[Recovery check](semaphore-recheck.json) records this evidence. At that point,
the full native success/restart journeys were still pending; they are now
qualified in the resumed record above.

At the earlier checkpoint, the real app could not start its local parse pool:
`multiprocessing` semaphore construction raised errno 28 inside and outside the
sandbox. The volume then had 75 GiB available and the reported semaphore limit
was 10,000. Neither the exhausted resource owner nor the recovery cause was
established; this recheck changed no host settings or unrelated processes.

## Historical failure evidence

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
that the real native pool succeeded. The separate real-spawn test was blocked
at that checkpoint; it passes in the recovery check above. No full suite ran.

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

## Historical failure runner

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

This runner expects the earlier errno-28 failure. Allocation now passes the
recheck, so successful parsing should fail that old expectation. Do not count
that expectation failure as an app regression. Use the resumed qualification's
success runner for text/Markdown import, duplicate resolution, permission Retry
and actual interruption/restart recovery.

The allocation gate and resumed native qualification now pass.
The existing [semaphore diagnosis](../../reviews/2026-09-08-semaphore-allocation-diagnosis.md)
explains the earlier host failure and the attribution limits. No reboot, global
resource cleanup, kernel setting change or termination of unrelated processes
was attempted during this qualification or recheck.

ADR required: no. Existing lifecycle/UX contracts are governed by
[ADR-014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[ADR-065](../../../../backlog/decisions/065-active-ingest-source-admission-and-override.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
