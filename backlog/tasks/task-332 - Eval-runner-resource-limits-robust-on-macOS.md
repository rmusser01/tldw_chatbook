---
id: TASK-332
title: Make eval-runner resource limits robust on platforms lacking RLIMIT
status: Done
assignee: []
created_date: '2026-07-20 18:45'
updated_date: '2026-07-24 12:00'
labels: [security, evals]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The evals code-runner (`Evals/specialized_runners.py:281`) sandboxes model-generated code well (static AST safety scan, temp cwd, minimal `PATH`/empty `PYTHONPATH`, disabled dangerous builtins, timeout). However `RLIMIT_AS`/`RLIMIT_NPROC` are wrapped in `try/except: pass`, so on macOS (and other OSes lacking them) the memory and fork limits silently do not apply. It is process-level, not container-level, sandboxing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 When `RLIMIT_AS`/`RLIMIT_NPROC` are unavailable, the runner either applies an equivalent bound or surfaces that the limit is not enforced (no silent gap)
- [x] #2 The macOS limitation is documented near the runner
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
On macOS, `RLIMIT_AS` silently no-ops (the resource is not supported); `RLIMIT_NPROC` works. Parent-side check `_memory_limit_enforced()` returns False on Darwin and logs a one-time WARNING, recording the gap in `results["sandbox_warnings"]`. Docstring updated to clarify that memory limits may not be enforced on all platforms. The pre-existing memory-exhaustion test was vacuous (static AST safety scan blocks its payload) and remains so; deployment users rely on timeout as the primary limit on unsupported platforms.
<!-- SECTION:NOTES:END -->
