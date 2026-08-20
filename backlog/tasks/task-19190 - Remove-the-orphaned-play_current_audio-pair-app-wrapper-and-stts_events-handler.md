---
id: TASK-19190
title: Remove the orphaned play_current_audio pair (app wrapper + stts_events handler)
status: To Do
assignee: []
created_date: '2026-08-20'
labels:
  - dead-code
  - stts
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Third-wave burn-down residue (sibling of merged TASK-19043, which removed the
`export_current_audio` pair the same way). At dev `7877defba`,
`tldw_chatbook/app.py:11409` defines `async def play_current_audio` — it lazily
initializes the S/TT/S handler and awaits
`Event_Handlers/STTS_Events/stts_events.py:2767`'s `play_current_audio`
handler. A whole-tree grep (production + `Tests/`) finds exactly three hits:
the wrapper def, the wrapper's internal call (`app.py:11416`), and the handler
def. Zero callers of the wrapper anywhere; the handler's only caller is the
wrapper; zero test references. TASK-19043's reviewer independently confirmed
the orphan with a grep that included dynamic-dispatch shapes
(`getattr`/string-built names).

Use merged TASK-19043 as the template, including its security-coverage-map
discipline: before retiring anything, check whether the handler performs
validation that needs a live-path equivalent (here the handler only does a
path-existence check on `_current_playground_audio_path()` and notifies on
failure — no unique validation identified, and there are no tests to retire —
but the check must be recorded, not assumed). Two knock-ons the removal must
chase: (1) the handler contains one `logger.error` call, so the persistent
diagnostic inventory's `stts_events.py` row must be hand-edited in the same PR
(the exact playbook step both 19042 and 19043 initially missed — see the
2026-08-20 lesson in `backlog/docs/lessons-testing-evidence.md`); (2)
`_current_playground_audio_path` (`stts_events.py:2786`) has this handler as
its ONLY caller — decide its fate in the same PR (its underlying
`_current_playground_artifact`/`_current_audio_file` attributes have ~25 other
usages and stay).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The `play_current_audio` wrapper in `app.py` and the `play_current_audio` handler in `stts_events.py` no longer exist, and a whole-tree grep (including dynamic-dispatch shapes) finds no remaining reference.
- [ ] #2 The security-coverage-map check from TASK-19043's playbook is performed and its outcome recorded in Implementation Notes: any validation the handler performed either has a live-path equivalent or is explicitly noted as not security-relevant.
- [ ] #3 The persistent diagnostic inventory row for `stts_events.py` is hand-edited in the same PR to reflect the removed `logger.error`, and `scripts/check_persistent_diagnostic_inventory.py` does not regress further because of this change (it is already red on dev for unrelated drift — see TASK-19191).
- [ ] #4 Any helper left caller-less by the removal (`_current_playground_audio_path`) is either removed with it or its retention justified; no new orphan is created.
- [ ] #5 STTS-affected suites (`Tests/UI/test_stts_profile_library.py` and any suite importing the touched modules) pass.
<!-- AC:END -->
