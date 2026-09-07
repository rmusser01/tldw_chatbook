---
id: TASK-19874
title: >-
  git_status advertises a path scope it does not apply
status: To Do
assignee: []
created_date: '2026-08-22'
labels:
  - tools
  - agents
  - owner-decision
priority: low
dependencies:
  - TASK-19632
---

## Description

Source: a pre-existing minor finding noted by **TASK-19632**'s reviewer while
verifying the pathspec-exclusion work. Re-verified at `3605bd52d`.

The `git_status` agent tool takes a `path` argument. It is used **only** to
discover which repository to report on — `_prepare_for_path` walks up to the
repository root and the argv that follows carries no positive pathspec for it
(`Tools/git_tool_impls.py:632-648`). Asking for one subdirectory's status
therefore returns every changed file in the whole repository.

The Python docstring is now accurate: TASK-19632 rewrote it to say so
explicitly (`git_tool_impls.py:614-624`, "It is NOT applied as a scoping
pathspec … asking for a subdirectory's status still returns every changed file
in the repo"). But that docstring is not what the model reads. The tool schema
handed to the model still says:

> "path": "Path inside the repository, relative to the workspace root
> (default: the workspace root)."

(`Agents/local_tool_provider.py:1258-1261`.) To a model, that reads as a scope
— particularly next to `git_diff` and `git_log`, whose `path` argument on the
same surface **is** applied as a literal pathspec. An agent narrowing to a
subdirectory to keep its context small silently receives the whole repository
instead, and has no signal that it did.

This is not a correctness or safety problem post-TASK-19632: the denylist
exclusions are applied to `git_status` regardless of `path`, so no sensitive
file becomes reachable through the gap. It is a truthfulness problem in the
interface the model plans against.

**This wants a decision rather than a default.** Either is defensible:

- **Give it real scoping** — mirror `git_diff` / `git_log` and append a
  `:(literal)` positive pathspec, so the parameter means what both its name and
  its description suggest. Note that TASK-19632's headline finding applies
  directly here: a bare value spliced after `--` is read as pathspec *magic*,
  so any positive pathspec must be `:(literal)`-prefixed and covered by that
  task's AST tripwire.
- **Keep the current behaviour** and make the model-facing description say what
  the docstring now says, so the parameter is honestly advertised as
  repository-discovery only.

## Acceptance Criteria

- [ ] A decision is recorded: `git_status`'s `path` either scopes the output or
      is described to the model as discovery-only
- [ ] The model-facing tool description in `Agents/local_tool_provider.py`
      matches the implementation's actual behaviour
- [ ] The description no longer implies a scoping contract that `git_diff` and
      `git_log` honour and `git_status` does not
- [ ] If scoping is added, the positive pathspec is `:(literal)`-prefixed and
      covered by TASK-19632's AST tripwire, and a test proves a
      pathspec-magic filename cannot invert the scope
- [ ] If scoping is added, the denylist exclusions still apply and a test pins
      that they are not weakened by the new positive pathspec
- [ ] A test asserts the tool description and the implementation agree, so the
      two cannot drift again

## Notes

Worth doing precisely because the docstring was already fixed. Half the drift
was closed and the half the model actually reads was not — which is the version
of this problem that still costs an agent a wasted turn.
