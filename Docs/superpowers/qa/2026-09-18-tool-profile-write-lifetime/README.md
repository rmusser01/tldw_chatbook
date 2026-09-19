# TASK-32787 — Tool Profile writes survive Settings recreation

Leaving Settings used to cancel its observer while an admitted thread could
still commit. Returning then showed stale profile rows and no outcome. The app
now owns admitted Import, Export and Remove tasks; each Settings visit observes
bounded pending/terminal state and refreshes facts after completion. Observation
cancels promptly, duplicate same-kind writes remain guarded across visits, and
normal shutdown closes admission and drains writes before resource teardown.
The existing process-owned exit watchdog is armed before the first drain.

## Targeted evidence

229 distinct targeted cases pass. No full suite or provider requests were run.

| Scope | Distinct cases | Evidence |
| --- | --- | --- |
| Mounted overlap/recreation, review lifetime, Settings actions, export recovery, removal outcomes and initialization lifecycle | 126 | [Broad run](targeted.txt), [six additional malformed-result cases](malformed-ui.txt) |
| Coordinator admission, ordering, cancellation, malformed results and drain | 21 | [Final owner run](owner.txt) |
| App admission/drain/watchdog ordering and embedded-app isolation | 4 | [Owner/app run](owner-app.txt) |
| Real publication and removal service boundaries | 52 | [Service run and initial isolation failure](services-and-isolation-attempt.txt) |
| Design-token and component-pattern governance | 26 | [Governance](governance.txt) |

Counts are deduplicated: the broad run contains 120 UI cases plus three app
cases; the owner/app run repeats twenty owner cases and contains all four app
cases. The service run passed its 52 service and fourteen earlier owner cases,
but its raw embedded-app invocation failed at config-source bootstrap. The
unchanged embedded-app assertion passes in the established private-profile
wrapper. That mixed run is not reported as an entirely passing command.

After explicit lambda argument binding and unused-import cleanup, the affected
Import/Export checks were repeated: [31 passed](final-import-export.txt). The
final native run also pins that production source. [Red/green and fixture
history](red-summary.txt) records the actual recreation failure and subsequent
review corrections. [Independent review](independent-review.txt) found no
remaining blocker after watchdog ordering, malformed-result and callable-origin
cancellation fixes. [Preflight](preflight.txt) records unchanged existing Ruff
diagnostic counts, clean new/changed support files, Backlog IDs and diagnostic
inventory guards. No CSS changed.

## Native visual review

The real app ran with LinuxDriver, TTY streams, a private profile and the actual
Tool Pack services. Four theme/size journeys imported a real service-exported
archive, confirmed Remove, and held the existing lifecycle mutation lock. A
repeated action before departure retained one write. F3 destroyed Settings; F4
created a replacement while the admitted operation remained pending. The
replacement displayed the pending receipt and loading state because listing
lease counts need that same real lock. Releasing it produced the tombstone,
exactly one revision increment and refreshed rows without taking newer category
focus. Explicit focus on the now-enabled Import button provided continuation;
another departure/return replayed the terminal receipt.

| Theme / size | Confirmation | Recreated, pending | Removed | Return after completion |
| --- | --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-review.svg) | [View](textual-dark-80x24-pending.svg) | [View](textual-dark-80x24-removed.svg) | [View](textual-dark-80x24-restored.svg) |
| Dark 170x48 | [View](textual-dark-170x48-review.svg) | [View](textual-dark-170x48-pending.svg) | [View](textual-dark-170x48-removed.svg) | [View](textual-dark-170x48-restored.svg) |
| Light 80x24 | [View](textual-light-80x24-review.svg) | [View](textual-light-80x24-pending.svg) | [View](textual-light-80x24-removed.svg) | [View](textual-light-80x24-restored.svg) |
| Light 170x48 | [View](textual-light-170x48-review.svg) | [View](textual-light-170x48-pending.svg) | [View](textual-light-170x48-removed.svg) | [View](textual-light-170x48-restored.svg) |

A fifth confirmed removal remained pending at normal Ctrl+Q
([capture](shutdown-pending.svg)). The fixture released the service lock only
after observing closed operation admission, a still-pending owned task and the
actual armed exit-watchdog thread. Shutdown drained the write; its tombstone and
single revision increment were checked after `app.run()` returned.

All seventeen captures were rendered and visually inspected. Confirmation
actions, wrapped pending/completion receipts and keyboard continuation are
readable in both themes and sizes. [Native result](native-result.json),
[capture hashes](capture-manifest.json) and [lifecycle receipt](lifecycle.json)
pin the runner and 25 source files. Exit was 0; the PID was absent before terminal
closure, the instance lock was reacquired, all eleven private databases passed
integrity checks, conversations/messages stayed empty, default fingerprints
were unchanged and error/faulthandler output was empty.

The first two native attempts exposed fixture assumptions, not product failures:
waiting for lock-blocked rows before release, then attempting to focus disabled
Import. Their result/lifecycle receipts are retained as `failed-run001-*` and
`failed-run002-*`. Final run003 corrects both and passes on the pinned source.

Native coverage uses real Import/Remove services and the removal shutdown
journey. Cross-instance duplicate admission, all three operation outcomes,
cancellation and concurrent operation ordering are qualified by the targeted
mounted/coordinator cases. The receipt is session-only and bounded, not a
persistent job history. Whole-MCP compact layout and broader feature review
remain open in the [ledger](../../reports/2026-09-18-tool-profiles-review.md).
[ADR-167](../../../../backlog/decisions/167-tool-profile-write-lifetime.md)
records ownership; ADR-107 service authority and ADR-150 presentation still apply.
Draft PR2707 requires its separate visual review and merge approval.
