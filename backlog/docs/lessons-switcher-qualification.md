# Lessons: source-bound switcher qualification

## A settled flag snapshot does not survive a later await

**TASK-31245, 2026-10-04.** The real-owner latency probe waited for a blank
Character modal's pending flag to clear, yielded through `pilot.pause()`, then
asserted the ready prompt. Repeated frozen-head diagnostic runs failed that
assertion before collecting samples. The retained compositor frame showed
`Loading local chats…`, pending true, and no mounted empty-state row: live
activity hydration had reconciled again during the pause. The untimed setup
boundary now waits for the actual current ready frame after that yield.
Measured windows and limits remain unchanged; five RED cases reject loading,
pending, nonblank-query and nonempty-result snapshots before 47 covering tests
pass. See `Docs/QA/task-31245/fixture-rebuild-2026-10-04.md` for failed attempts
and provenance. A yielded readiness check must be re-established at the
boundary where its result is consumed; arbitrary pause is not a readiness proof.

## Wizard completion is not Console first-send completion

**Same fixture rebuild.** A fresh offline profile with saved synthetic chats
set `[first_run] setup_completed=true`, but Ctrl+K correctly refused to open
under Console's provider-setup overlay. A read-only action observer proved
setup blocking, first-send false, and no modal push. The navigation fixture now
explicitly sets `[console.onboarding] first_send_completed=true`; it still has
no credentials, no send readiness and no permitted network. That represents
existing-chat navigation only, never application/provider onboarding evidence.
