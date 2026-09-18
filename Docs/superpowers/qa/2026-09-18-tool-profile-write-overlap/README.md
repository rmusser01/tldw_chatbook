# TASK-32786 — Preserve admitted Tool Profile write outcomes

Starting another Import, Export or Remove could cancel the exclusive UI worker
while its admitted thread continued writing. The mounted reproduction removed
the actual profile but retained its old row and displayed `Remove cancelled`.
Settings now guards same-operation dispatch while the write is pending, shows
local progress, and observes its terminal result. Preparation remains
supersedable; actual worker cancellation is unchanged. Import uncertainty now
refreshes current facts and is described as uncertain, not a definite failure.

## Targeted evidence

173 distinct targeted cases pass. No full suite or provider requests were run.

| Scope | Cases | Evidence |
| --- | --- | --- |
| Three operations: repeated action, success/refusal/uncertainty, newer modal/focus, cancellation; existing review lifetime, Settings actions, export recovery and removal outcomes | 95 | [UI run](targeted.txt) |
| Real Tool Pack publication and removal boundaries | 52 | [Services](services.txt) |
| Design-token and component-pattern governance | 26 | [Governance](governance.txt) |

The UI run includes 15 new overlap cases. Controlled services hold the admitted
write; actual mounted review/confirmation and production CSS qualify its UI
ownership. [Red/green record](red-summary.txt) separates the initial nine failing
cases, the remaining uncertainty-copy failure and the final expanded run.

A separate real-publication boundary probe confirms that cancellation before
publication prevents the destination, while cancellation after its last poll can
legitimately complete publication. Both leave no temporary archive. This is why
cancelling a UI observer is not proof that its thread stopped writing.
[Probe receipt](publication-cancellation.txt).

Independent source review found no introduced blocker: all guards run before
the exclusive worker wrapper, write flags clear in `finally`, destination-change
recovery releases the flag before reopening the picker, exact reviewed arguments
remain intact, and export keeps its existing cancellation probe. No shield or
cancellation-swallowing loop was introduced. [Review summary](independent-review.txt).
Scoped Ruff remains 114 existing Settings diagnostics before and after; changed
methods, the new test, adjusted lifetime test and native runner are formatted.
Backlog-ID and persistent-diagnostic inventory guards pass.

## Native visual review

The real app used LinuxDriver, TTY streams, a held instance lock and real private
Tool Profile services. Every size/theme cell imported a real service-exported
pack through the UI. A fixture thread held the existing lifecycle mutation lock
while a confirmed removal waited. Repeating Remove kept the first worker alive,
opened no second review, and displayed fully painted progress. Releasing the
lock produced one revision increment, the permanent tombstone, a refreshed list
and a visible Import continuation. No service implementation was replaced.

| Theme / size | Confirmation | Write pending | Removed |
| --- | --- | --- | --- |
| Dark 80x24 | [View](textual-dark-80x24-review.svg) | [View](textual-dark-80x24-pending.svg) | [View](textual-dark-80x24-removed.svg) |
| Dark 170x48 | [View](textual-dark-170x48-review.svg) | [View](textual-dark-170x48-pending.svg) | [View](textual-dark-170x48-removed.svg) |
| Light 80x24 | [View](textual-light-80x24-review.svg) | [View](textual-light-80x24-pending.svg) | [View](textual-light-80x24-removed.svg) |
| Light 170x48 | [View](textual-light-170x48-review.svg) | [View](textual-light-170x48-pending.svg) | [View](textual-light-170x48-removed.svg) |

All twelve captures were rendered and inspected. Compact progress and completion
text wrap fully above their actions. [Native result](native-result.json),
[capture hashes](capture-manifest.json) and [lifecycle receipt](lifecycle.json)
pin the runner/source and final profile. Ctrl+Q returned 0; the PID was absent
before terminal closure, the instance lock was reacquired, all eleven private
databases passed integrity checks, conversations/messages stayed empty, default
fingerprints were unchanged, and no error/faulthandler output was emitted.

This qualifies overlapping actions within the current Settings instance.
Screen-destruction/app-shutdown outcome ownership is a separate review boundary;
the fix deliberately preserves existing cancellation behavior. Broader MCP
layout and feature review remain open in the [Tool Profiles ledger](../../reports/2026-09-18-tool-profiles-review.md).
ADR-107 and ADR-150 apply; no new architecture decision is introduced. Draft
PR2707 still requires its own visual review and merge approval.
