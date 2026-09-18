# Recent imports continuity and grouped outcomes — TASK-32696

Baseline: `2e79851916` on `feat/component-pattern-library`.

A queue transition collapsed Recent imports and moved its focused title to
Keywords in the next draft. The queue panel now snapshots the mounted disclosure
before rebuilding and restores its title through the existing focus callback.
Open and closed states survive queue updates; newer user focus still wins. The
state belongs to the current panel and starts collapsed on a fresh panel.

ADR required: no. This is a repair under
[ADR-014](../../../../backlog/decisions/014-library-ingest-service-authority-and-recovery.md),
[ADR-150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [ADR-161](../../../../backlog/decisions/161-component-pattern-library.md).
No CSS, token values, consent rules, queue ownership or execution boundaries changed.

## Verification

[Verification](verification.json) records **94 passing targeted checks**: 68 UI,
queue and governance cases plus 26 state cases. Eight new production-CSS journeys
cover wide/dark and compact/light terminals, explicit expansion and collapse,
newer focus, grouped Show/Hide, Retry all, Dismiss all, full Clear confirmation,
Recent ledger and unsaved draft preservation. Retry exercises the real controller
and in-memory registry through an explicit replacement app seam; it starts no
worker. Real temporary SQLite and local preflight are used.

[Regressions](regressions.json) records the pre-fix failures. Four cases reproduced
collapse or focus loss before that red run was interrupted; all six Recent cases
then passed. The grouped test initially sent its second Enter during Textual's
press flash; it now waits for the flash to end as well as passing the confirmation
dead zone. This required no product change.

[Static comparison](static-comparison.json) shows zero new Ruff diagnostics;
six inherited canvas diagnostics remain. New Python files and modified ranges
are formatted. [Review](review.json) found no actionable issues. LibraryScreen
and the ingest controller are unchanged, so their previously recorded size-ceiling
failures were not rerun or claimed repaired; [sizes](size-comparison.json) records
that comparison. No full repository suite was run. Existing pytest cleanup and
component-governance SyntaxWarnings remain.

## Native evidence

[native_check.py](native_check.py) runs actual TldwCli/LinuxDriver with an exclusive
private profile, private database paths and null keyring. Its synthetic registry
has neither a store nor a runner. Retry all creates four replacement registry
entries through a UI-only seam; Dismiss and Clear update the session ledger.
No import, model, provider, server or installation operation is requested.

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Group controls after expansion | [capture](group-expanded-170.svg) | [capture](group-expanded-80.svg) |
| Recent focused after queue transition | [capture](recent-transition-170.svg) | [capture](recent-transition-80.svg) |
| Full Clear confirmation | [capture](clear-consent-170.svg) | [capture](clear-consent-80.svg) |

Final run-003 passes both sizes and exits normally with status 0:
[result](result.json), [isolation](isolation.json), [lifecycle](lifecycle.json).
Six final SVG captures were rendered and inspected as the confirmation batch;
[inspection](inspection.json) records the observations and limits. Group member
rows below the fold are asserted mounted, not all pictured simultaneously.
[Persistence](persistence.json) checks ten healthy private databases, zero
media/messages/ingest jobs, unchanged source bytes and default-profile hashes,
and no app ERROR/CRITICAL log lines. The check uses the project interpreter;
system Python's SQLite could not open several WAL-mode databases read-only.

Run-001 passed wide but its second-size fixture wrongly expected older cleared
entries to be marked dismissed. Run-002 passed both sizes; its first visual batch
revealed that waiting for app-level Toasts missed the active screen's notices.
Run-003 waits on that screen and supplies the final unobscured captures. Earlier
runs, private databases and full logs remain in ignored scratch.

The real optional image terminal probe runs before app startup, as in TASK-32667.
This qualifies probe-primed startup. Resizing happens at the Library hub followed
by re-entry. Live resize with focus still in Import and actual provider-specific
recovery remain for review. No restart, push or dev integration was performed.
