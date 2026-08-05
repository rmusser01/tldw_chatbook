---
id: TASK-1642
title: 'Library: an unsaved prompt silently blocks screen navigation'
status: To Do
assignee: []
created_date: '2026-07-31'
labels: [library, bug, ux]
dependencies: []
---

## Description (the why)

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
no notification either time. Made worse by task-1641 (the editor's Save
button renders below the viewport at standard heights, so the fix the
veto is asking for isn't visibly available either).

## Acceptance Criteria (the what)

- [ ] Attempting to navigate away with an unsaved prompt tells the user
      why the switch was refused and what to do (same shape as the skills
      dirty veto).
- [ ] The message names the resolution (Save, or discard/leave the
      editor) and matches whatever affordances actually exist after
      task-1641.
- [ ] A test covers the notify-on-veto path so it cannot regress to
      silence.
