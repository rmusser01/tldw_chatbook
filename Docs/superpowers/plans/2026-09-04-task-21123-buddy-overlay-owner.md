# Persona Buddy presentation ownership implementation plan

**Goal:** One app-owned coordinator manages Buddy presentation across navigation,
screen rebuilds, controller changes, and shutdown, with the existing UI contract.
**Architecture:** Native screen-change signal plus generic ContentsRebuilt messages;
thread-safe generation notifications; one coalescing reconciliation worker.
**Tech stack:** Existing Python, Textual 8, asyncio, pytest/Pilot.
**ADR required:** yes, amendment to backlog/decisions/074-portable-actor-packs-and-local-persona-visual-runtime.md.
**Reason:** Presentation lifetime and cross-module ownership change.
**Spec:** ../specs/2026-09-04-task-21123-buddy-overlay-owner.md

## Steps

- [x] Read current task, ADR-074, lifecycle code, and relevant Textual/testing lessons;
  confirm TASK-21122 merged and no relocation PR is active.
- [x] Run the existing lifecycle and disabled-import baseline (21 passed).
- [x] Add failing behavior tests: no Buddy reconciliation workers while disabled,
  controller changes from a worker thread mount without a manual reconcile, and
  screen rebuilds retain exactly one current view without per-screen Buddy state.
- [x] Add UI/Navigation/persona_buddy_overlay.py: content-free change message and
  app-owned coordinator. Keep all mount/currentness/cleanup ownership here.
- [x] In UI/Navigation/base_app_screen.py replace Buddy fields and methods with a
  generic post-recompose ContentsRebuilt notification; retain unrelated mouse safety.
- [x] Wire app.py screen-change subscription, generation message handler, lazy owner,
  unavailable confirmation, and geometry drain. Keep App's public reconcile entry point.
- [x] Add the optional content-free controller generation callback. Its production
  target posts a message; it must never await or enter UI logic under the controller lock.
- [x] Wire widget unavailable callbacks to the app-owned boundary. Retain widget
  timers, render authority, geometry persistence, style, and input behavior.
- [x] Adapt existing lifecycle tests to the owner; verify cancellation, stale mounts,
  modal resume, unavailable fences, and shutdown durability on the production path.
- [x] Run targeted Buddy widget/controller/lifecycle/Workbench/import guards, Ruff,
  compilation, diff checks, and derived-artifact preflight. Inspect any failures before
  widening the run. No full repository test sweep.
- [x] Review final diff; record exact results, complete task criteria, and prepare the verified commit.

Commands use the shared interpreter from the isolated worktree:
`/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q <target files>`.
Use unique /private/tmp/task21123-* basetemp directories.
Expected failing tests must demonstrate the missing behavior before implementation;
existing lifecycle regressions must pass after adapting ownership references.
