# TASK-32823 — preserved inspector refresh work

This is unfinished follow-up work, excluded from PR2707 at the user-approved
closeout boundary. The prototype reconciles selected tool definitions after
catalog refresh and keeps an unchanged argument form mounted. The original four
cases have three intended failures and one unchanged-catalog pass; all four now
pass. These narrow results do not qualify the change for merging.

Independent read-only review found an unresolved P2 race: a pending preview mint
can finish during show_tool's awaited child removal, after the old preview was
cleared. The same server/name and still-queryable pruning panel can satisfy
_test_panel_is_current without advancing the generation, allowing the late old
preview to survive panel retirement. The current tests complete the mint before
refresh and do not cover this interval.

Next: add a deterministic delayed-mint/removal regression and fix preview
retirement before any await; cover concurrent selection/focus changes and raw
forms, run targeted adjacent tests, then perform bounded native verification.
No native verification or completion claim has been made. Continue after PR2707
merges, from merged dev on a fresh follow-up branch; port this preserved commit
rather than starting duplicate work. Service admission policy remains unchanged.
