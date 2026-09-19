# Focused review — TASK-32666

Reviewed the working diff against `b1303615d6` using an independent reviewer.

The review reproduced programmatic event feedback, obsolete native events after
Reset, events from replaced controls, incompatible controls during a pending
backend recompose, a forwarded edit overtaking Reset, and a sibling edit lost
when two genuine edits arrived in their emitted order. Mounted regressions
cover each boundary. It also identified a detached reveal callback; attachment
admission protects leaving Import before its deferred work runs.

The resulting update path distinguishes dependency refresh from explicit reset.
Ordinary edits retain live editor values and cached emitted values. Reset writes
defaults while suppressing programmatic echoes. Both native and forwarded user
messages verify their current sender/value; a pending backend compose receives
the newest snapshot. Focus reveal reads current focus after layout and scrolls
immediately, without adding another delayed target.

The final read-only review found no remaining actionable issue. Group-local
capability dependencies and existing recursive-summary guidance were checked.
Actual extraction, model/provider actions and queue execution were not reviewed
in this slice.

Final native inspection also found the compact shell's one-row action cap
clipping the install explanation. Two real-shell regressions reproduce it; the
scoped two-ID selector and existing full-height token make both cases pass.
The independent follow-up review found no actionable issue in that final CSS
and test adjustment.
