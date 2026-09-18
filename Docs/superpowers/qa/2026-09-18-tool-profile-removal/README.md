# TASK-32782 — Tool Profile removal outcomes

Removal now distinguishes an uncertain outcome from a definite refusal and
explains recovery. Every terminal attempted-removal outcome refreshes the
listing, including stale revisions, refusals and unknown results. It never
retries a mutation automatically. Existing revision, lease, reference and
permanent-tombstone authority is unchanged.

Removal feedback belongs to its profile row, above the action controls. When
the row disappears, feedback moves to the Import continuation. Its plain-text
widget rechecks wrapping and reveals only while the originating control or its
replacement continuation still owns focus. A newer category, dialog or profile
action retains focus; other import/export feedback keeps its global location.

## Targeted evidence

167 distinct targeted cases pass; no full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| New removal outcomes, current row facts, complete paint, literal IDs, newer navigation, non-removable continuation | 18 | [Focused regressions](focused.txt) |
| Existing focus and action/loading lifecycle | 31 | [Focused regressions](focused.txt) |
| Review lifetime, existing workflows and export recovery | 62 | [Other regressions](regressions.txt) |
| Token/component governance | 26 | [Other regressions](regressions.txt) |
| Generated CSS synchronization | 5 | [CSS](css.txt) |
| Real removal service boundaries and reconciliation | 25 | [Service](service.txt) |

All six original outcome/copy/paint cases failed before repair and passed after
it. [Pre-fix summary](red-summary.txt). An initial follow-up captured scroll
position while a focus animation was still running; the fixture now settles
scheduled animations before comparing positions. No production change was
needed for that fixture timing correction. The complete 49-case focused run
then passed, including that exact newer-profile navigation case.

Mounted UI cases deliberately control service outcomes for stale, referenced,
in-use, non-removable, uncertain, unexpected-exception, invalid-return and
successful removal. They qualify presentation and refresh, not injected storage
behavior. The separate real-service cases cover reference/lease races, stale
revisions, permanent Deny semantics and strict uncertain-outcome reconciliation.
Independent review found no introduced blocker and inspected an overlapping
render/outcome probe. [Probe receipt](review-race.txt).

## Native visual qualification

The real app uses LinuxDriver and both TTY streams in a private profile. Each
cell imports a real service-exported fixture through the UI, opens removal,
then holds an actual exact-profile runtime lease. The real removal service
refuses; policy bytes remain unchanged, the listing shows current eligibility,
and the full outcome and keyboard continuation stay visible. After releasing
the lease, the journey follows the displayed guidance, reopens Tool Profiles,
and explicitly confirms a new removal. The stored profile becomes a permanent
Deny tombstone; the full result and Import continuation remain visible.

| Theme / viewport | Refusal and continuation | Explicit retry review | Removed result |
| --- | --- | --- | --- |
| Dark 80×24 | [View](textual-dark-80x24-refused.svg) | [View](textual-dark-80x24-retry-review.svg) | [View](textual-dark-80x24-removed.svg) |
| Dark 170×48 | [View](textual-dark-170x48-refused.svg) | [View](textual-dark-170x48-retry-review.svg) | [View](textual-dark-170x48-removed.svg) |
| Light 80×24 | [View](textual-light-80x24-refused.svg) | [View](textual-light-80x24-retry-review.svg) | [View](textual-light-80x24-removed.svg) |
| Light 170×48 | [View](textual-light-170x48-refused.svg) | [View](textual-light-170x48-retry-review.svg) | [View](textual-light-170x48-removed.svg) |

All twelve captures were rendered and visually inspected. Compact feedback
wraps in full above the two-row action layout; wide feedback stays beside the
one-row actions. Disabled Remove is distinct from the focused Export fallback.
No native service or dialog substitutions are used. Uncertain outcomes have
mounted/service evidence above; the native matrix specifically qualifies a
real in-use refusal and confirmed retry.

[Native result](native-result.json), [capture hashes](capture-manifest.json),
and [lifecycle receipt](lifecycle.json) pin the final source and runner. Normal
Ctrl+Q exit returned 0. Before terminal closure the app process was absent, the
instance lock was reacquired, all eleven private databases passed integrity
checks, conversations/messages remained empty, and default-profile fingerprints
were unchanged. No error or faulthandler output was recorded.

Scoped Ruff introduces no diagnostics (Settings 114→114; panel 1→0). New tests,
panel, native runner and changed Settings methods pass formatting; CSS sync,
backlog, diagnostic inventory and diff guards pass. Existing ADR-107 and ADR-150
apply; no new ADR or design token is needed. [Verification](verification.json).
The [review ledger](../../reports/2026-09-18-tool-profiles-review.md) retains the
remaining management handoffs and concurrent-workflow review.
