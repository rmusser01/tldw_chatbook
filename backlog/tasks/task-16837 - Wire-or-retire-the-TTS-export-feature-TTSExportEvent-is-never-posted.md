---
id: TASK-16837
title: 'Wire or retire the TTS export feature (TTSExportEvent is never posted)'
status: To Do
assignee: []
created_date: '2026-08-16'
labels:
  - dead-code
  - tts
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The TASK-16199 review (PR #1676, its F1) established — and it still holds at dev
`ee741cf10` — that the TTS export path is dead code in the shipped app:

- `TTSExportEvent` (`Event_Handlers/TTS_Events/tts_events.py:430`) is **never constructed
  or posted anywhere in production** — repo-wide grep finds only the class definition,
  `handle_tts_export` (`:3694`), and `on_tts_export_event` (`:3850`), plus tests.
- Even if something posted it, it would not arrive: `TTSEventHandler` is **not a
  `MessagePump`** — it is a plain class instantiated at `app.py:11127` and driven by
  direct method calls, so Textual name-dispatch never reaches `on_tts_export_event`.

Meanwhile three merged tasks (15471, 16194-era claim work, 16199) have invested real
concurrency engineering in protecting exports that cannot currently happen. Decide
whether export gets a UI affordance (a Save/Export action on TTS-bearing messages posting
through a real dispatch path) or gets retired along with its claim machinery's
export-only half.

If wired, two known residuals from the same review become live and belong to the wiring
task: **F6** — the check-then-act window between the shutdown sweep's single up-front
union read (`:3928`) and its per-file deletes across awaits (demonstrated by probe: an
export claiming in that gap loses its source); and **F4** — the union's live half
(exports admitted after the cancel pass) is untested: mutating the union to
snapshot-only left all 26 tests green, so a refactor can delete that half for free.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 An explicit wire-or-retire decision is recorded (owner call)
- [ ] #2 If wired: a user-reachable affordance posts the export through a dispatch path that actually reaches the handler, and the F6 residual window is closed or bounded with evidence
- [ ] #3 If wired: a test pins the shutdown-sweep union's live half (F4) so it cannot be silently removed
- [ ] #4 If retired: the event, both handlers, and the export-only claim machinery are removed with reachability evidence, and the 15471/16199 tests are re-scoped accordingly
<!-- AC:END -->
