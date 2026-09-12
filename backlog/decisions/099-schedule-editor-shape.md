# ADR-099: Schedule creation and editing stay modal

Status: Proposed
Date: 2026-08-28
Related Task: TASK-23103 (spike; spawned by the 2026-08-28 Settings+Schedules UX critique)

## Decision

Schedule create/edit remains a modal (`ReminderForm(ModalScreen)`), upgraded in
place rather than replaced by a detail-pane editor. The modal is the durable
shape because it is the only one of the candidate shapes that works at every
supported terminal size without a second code path. The idiom gaps that made
the modal feel foreign to the workbench are fixed inside the modal
(TASK-23100 scrolling/height-robustness, TASK-23102 vocabulary; footer-style
key hints and theme-token styling fold into that work), not by relocating the
form.

## Context

The 2026-08-28 critique found the creation flow to be the weakest part of an
otherwise product-native workbench, with a P0 rooted in the modal's fixed
height (`ReminderForm` stacked fields in a plain `Vertical` inside a
`max-height: 55` container with no scrolling), and asked the structural
question: every peer surface (Workflows, Watchlists, Settings) edits in the
detail pane of a persistent workbench — why is Schedules different? A
pane-based editor would inherit screen scrolling, the state banner, and
footer-key consistency structurally, and would keep the queue (existing tasks
and their next runs) visible while scheduling.

Three shapes were compared against the workbench idiom, keyboard flow, and
terminal-height/width constraints:

1. **Modal, patched.** Works at every terminal size; creation stays reachable
   from the empty state (`c`) even at the 80×24 floor. Costs: the height
   robustness had to be built (TASK-23100), and workbench conventions
   (footer-visible key hints, shared styling, state text) must be duplicated
   inside the modal deliberately.
2. **Detail-pane editor.** Best idiom fit and context visibility — but the
   workbench already hides the detail and inspector panes at narrow widths
   ("Detail and inspector hidden — widen the window …", the responsive path in
   `schedules_workbench.py`), and creation must work there. A pane editor
   therefore requires a modal fallback at narrow widths, i.e. two code paths
   for the single most important interaction. It also moves form lifecycle
   (dirty-state guarding across queue selection changes) into the workbench,
   which the modal currently provides for free by construction.
3. **Pushed full editor screen.** Height/width-safe like the modal, but it is
   still navigationally modal (the queue is not visible), costs the most to
   build, and abandons the discard-guard and focus behavior the modal already
   has.

The deciding constraint is the width cliff: shape 2 is only better than
shape 1 on terminals wide enough to show the detail pane, and strictly worse
below that — the fallback modal it needs *is* shape 1. Under the standing
stability-over-quick-wins ruling, one shape that works everywhere beats a
better shape that needs a second shape as a safety net.

## Consequences

- TASK-23100 and TASK-23102 invest in the modal and are not throwaway.
- The modal owns idiom parity deliberately: key hints rendered visibly inside
  the form (not only in bindings), theme-token styling, text-carried state.
  These ride with TASK-23102's form rework; no new task is spawned.
- The pane-editor question is closed, not deferred. Reopen only if a future
  schedule editor must show live queue context while editing (e.g. a
  conflict-aware picker), and then reconsider all three shapes against the
  width floor rather than defaulting to the pane.
- The queue-as-forecast question from the same critique (lead with a
  next-24-hours timeline) is orthogonal to editor shape and remains open in
  the critique snapshot; this ADR neither adopts nor rejects it.

## Amendment (2026-09-03)

The modal remains the durable shape for creation, for multi-field surgery
("Edit in full…" — question text, custom cron, scope rework), and as the
way to reach every field at every width — including any width at which
the redesign's PR-4 responsive floor later hides the detail pane (today's
`schedules-workbench-compact` rule only narrows the panes, it never hides
one). The width-cliff argument above still stands for all three,
unchanged.

A narrower carve-out ships alongside the redesign program's inspector-pane
work: single-value rows in the detail pane (Repeat/At/Timezone, and —
recurring-question definitions only — Notifications/Model/Generation/
Finding policy/Sources, plus the header pause/resume affordance and the
owner row's transfer dropdown) now edit in place via
`DetailValueRow`'s activation/edit-swap API, committing or cancelling
(Escape) without leaving the row. This does not reopen the three-shape
comparison this ADR closed — a fixed-height row swap has none of the
scrolling, discard-guard, or 80×24-floor costs that made the modal the only
shape that worked everywhere for the *whole* form, so it was never inside
the width-cliff argument to begin with. See
`116-schedules-inspector-editing.md` for the full decision and rationale.
