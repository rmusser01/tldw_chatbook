---
id: TASK-332
title: Make eval-runner resource limits robust on platforms lacking RLIMIT
status: To Do
assignee: []
created_date: '2026-07-20 18:45'
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
- [ ] #1 When `RLIMIT_AS`/`RLIMIT_NPROC` are unavailable, the runner either applies an equivalent bound or surfaces that the limit is not enforced (no silent gap)
- [ ] #2 The macOS limitation is documented near the runner
<!-- AC:END -->
