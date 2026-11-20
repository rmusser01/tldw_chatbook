---
id: TASK-42091
title: "Boot worker census flakes on the Console skill-trust setup worker"
status: Done
assignee: []
created_date: 2026-11-19 23:55
labels:
- follow-up
- tests
- performance
priority: medium
references:
- Tests/Performance/test_boot_worker_census.py
- tldw_chatbook/app_skill_wiring.py
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`Tests/Performance/test_boot_worker_census.py::test_boot_worker_and_thread_starts_stay_within_the_allowlist` fails intermittently on dev 6a5a332f95: `workers started during boot that are not on the reviewed allowlist: [('ensure_local_skill_trust_service', 'console-skill-trust-setup')]`. It runs only in Perf Guard, which is not a required check.

The worker is mount discovery's skill-trust preparation (`app_skill_wiring._prepare_console_skill_trust_service`). It is a coroutine worker whose build hops to the executor. It is not new: real boots have started it since 3e618d7119 (2026-10-15).

The census only began to see it after TASK-41642 (0d64811b8d). Before that, discovery compared the captured `WorkerManager._new_worker`'s `__globals__["Worker"]` with Textual's `Worker`. The census probe patches `_new_worker` before importing the app, so the captured function was the probe's recorder, whose globals hold no `Worker`. The check failed, and discovery fell back to building trust on the UI loop, where a census that records starts cannot see it. TASK-41642 removed that code-identity check, so the census boot now behaves like a real boot. The worker starts inside the census window only when the preparation lands there.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The census decides the skill-trust setup worker explicitly, with its owning feature named
- [x] #2 The census passes repeatedly on a boot where the worker starts inside the window
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no (allowlist row for an existing off-loop worker).

1. Reproduce on dev.
2. Choose between deferring the worker past first paint and allowlisting it.
3. Pin the choice and re-run the census and the stagger-policy cross-check.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- **Decision: allowlisted, not deferred.** The worker is not new boot work, and it is no regression. Real boots have run it since 3e618d7119; only the census's view changed, with TASK-41642. It belongs to the already-allowlisted mount discovery (`_refresh_console_skill_candidates`, `console-skill-discovery`), which awaits it inline. Deferring it would mean deferring discovery itself. The row's comment records that history and says the worker is allowlisted, not asserted present, because it only sometimes lands in the window.
- **Reproduction:** on dev 6a5a332f95, 2 of 4 runs failed with exactly that message.
- **After the row:**
  - 6 of 6 runs pass.
  - A temporary probe of `started_workers` showed the worker recorded inside the window in 5 of 6 further runs, all of which passed. That is AC #2's condition.
  - `Tests/Performance/test_boot_worker_census.py` and `Tests/App/test_boot_worker_stagger_policy.py` pass 22/22. The stagger policy cross-checks policy rows against this allowlist, so an extra allowlist row needs no policy row.
- **Review (local).** No Critical or Important findings. Its two Minor findings are fixed:
  - Root cause: my first version said TASK-41642 had moved this work off the UI loop. In fact the census's own instrumentation had hidden the worker until then. The comment, this description and the commit now say so.
  - AC #2: the probe evidence above shows the worker recorded in the window.
- **Lesson:** recorded in `backlog/docs/lessons-testing-evidence.md`.
<!-- SECTION:NOTES:END -->
