---
id: TASK-2702
title: 'Library: an unsaved prompt silently blocks screen navigation'
status: Done
assignee:
  - '@codex'
created_date: '2026-07-31'
updated_date: '2026-08-13 09:45'
labels: [library, bug, ux]
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
With unsaved edits in the Library prompt editor, clicking another
destination in the nav bar does nothing at all — no screen change, no
toast, no banner. The user is stuck on Library with no explanation and no
stated way out.

Mechanism (dev @ 207053253): `flush_pending_work` returns False whenever
`_library_prompt_dirty` is set (`library_screen.py:10713-10731`), and the
app-level navigation veto **only logs** — the screen is responsible for
telling the user. `flush_pending_work` does exactly that for the skills
editor (`self._notify_skill_dirty_veto()`, `library_screen.py:1796-1801`)
but deliberately not for prompts; the comment there says "notes show their
own conflict banner and prompts predate this pattern, so only the skill
veto reports here." Notes' banner covers their case, so prompts are the
one dirty state with no feedback.

Reproduced live (G3 user-guide session, 2026-07-31): typed into a new
prompt, clicked "5 Roleplay" in the nav bar twice — stayed on Library,
no notification either time. Made worse by task-2701 (the editor's Save
button renders below the viewport at standard heights, so the fix the
veto is asking for isn't visibly available either).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Attempting to navigate away with an unsaved prompt tells the user why the switch was refused and what to do (same shape as the skills dirty veto)
- [x] #2 The message names the resolution (Save, or discard/leave the editor) and matches whatever affordances actually exist after task-2701
- [x] #3 A test covers the notify-on-veto path so it cannot regress to silence
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no

ADR path: N/A

Reason: routine UX bug fix applying the existing Library dirty-veto/discard
pattern without changing persistence, service contracts, state ownership,
security, or long-lived application structure.

1. Strengthen the mounted Prompt dirty-navigation test and add clean/save plus
   compatibility-only Discard regressions, then capture the intended RED.
2. Mirror the existing Skill warning/discard pattern in the current Prompt canvas
   and screen seams without new state, CSS, workers, modals, or abstractions.
3. Mutation-check both the warning and live Discard enablement, run the bounded
   affected/static/UI-hardening gates, document evidence, and close the task.

Detailed execution plan:
`Docs/superpowers/plans/2026-08-13-task-2702-prompt-dirty-navigation.md`.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a fixed, content-free warning when dirty Prompt state vetoes navigation:
  `Unsaved Prompt changes — Save or Discard changes first.` Clean Prompt flushes
  remain silent.
- Added one stable `Discard changes` action to every Prompt editor state. It is
  disabled with a literal explanation while clean, updates in place when fields
  become dirty or save successfully, and returns to the current Prompt list
  without persisting the working copy. Compatibility-only editors therefore
  retain an explicit escape even when Update and Convert are unavailable.
- Mounted real-SQLite regressions cover clean/dirty/save transitions, the exact
  warning, compatibility discard without persistence, current-scope refresh,
  first-row focus, normal/conflict DOM order, and action containment across four
  terminal sizes. RED evidence was the absent warning/action, stale disabled
  state, and no-op discard; notifier, live-patch, and handler mutations each made
  their exact tests fail before restoration.
- Focused behavior verification passed 6 tests and the final normal/conflict
  geometry matrix passed 8 tests. The one full Prompt-canvas run produced 274
  passes and 5 failures: two stale action-order expectations were corrected and
  passed exactly; the other three PromptBlockEditor mount-race failures reproduced
  unchanged at pre-implementation commit `9bc505b72` and were not weakened or
  worked around.
- Ruff lint, changed-range Ruff formatting, production `py_compile`, cumulative
  diff checking, and the final Impeccable detector passed. The detector returned
  no findings. No dependency, CSS, diagnostic, persistence, or service-contract
  changes were introduced.
- ADR required: no. This is a routine application of the existing Library dirty
  veto/discard pattern and changes no architectural boundary. No new lesson was
  added because the implementation surfaced no new generalizable incident.
<!-- SECTION:NOTES:END -->
