---
id: TASK-19874
title: >-
  git_status advertises a path scope it does not apply
status: Done
assignee:
  - rmusser01
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

- [x] A decision is recorded: `git_status`'s `path` either scopes the output or
      is described to the model as discovery-only
- [x] The model-facing tool description in `Agents/local_tool_provider.py`
      matches the implementation's actual behaviour
- [x] The description no longer implies a scoping contract that `git_diff` and
      `git_log` honour and `git_status` does not
- [x] If scoping is added, the positive pathspec is `:(literal)`-prefixed and
      covered by TASK-19632's AST tripwire, and a test proves a
      pathspec-magic filename cannot invert the scope
- [x] If scoping is added, the denylist exclusions still apply and a test pins
      that they are not weakened by the new positive pathspec
- [x] A test asserts the tool description and the implementation agree, so the
      two cannot drift again

## Implementation Plan

1. Record the decision: REAL SCOPING -- mirror `git_diff`/`git_log` (the
   parameter means what its name, its description, and its two siblings on
   the same surface already mean), which also fixes the actual cost the
   review found (an agent narrowing to a subdirectory to keep its context
   small silently received the whole repository).
2. Splice `_literal_pathspec(_repo_relative_path(...))` into git_status's
   argv ahead of the denylist pathspecs; `None`/`.` keep whole-repo
   behaviour (`:(literal).` is identity, so no special case).
3. Rewrite the model-facing description to the scoping contract, phrased
   like git_diff's.
4. Tests: scoping positive+negative, magic-inversion, denylist choke-point
   refusal, and a description/implementation agreement pin.

## Implementation Notes (2026-10-11, dev `9f8533686c`)

- **Decision**: real scoping (the task's first option). Either option was
  defensible; scoping was chosen because it makes the surface internally
  consistent (`git_diff`/`git_log` apply `path` literally) and removes the
  real cost (silent whole-repo context bloat for a narrowing agent).
- **Implementation** (`Tools/git_tool_impls.py::git_status`): the argv now
  carries `_literal_pathspec(_repo_relative_path(workspace_root, repo_root,
  path, ...))` ahead of `_denylist_pathspecs(...)` when `path` is neither
  `None` nor `"."` (`:(literal).` is identity, so the default and an
  explicit `.` keep today's whole-repo behaviour). The positive pathspec
  rides the same local `pathspecs` list the denylist extends, which is
  exactly the shape TASK-19632's AST tripwire
  (`test_every_pathspec_this_family_builds_carries_explicit_magic`)
  validates -- only `_literal_pathspec` appends and `_denylist_pathspecs`
  extends may build it, and the tripwire run is green with the new call
  site in place.
- **Description** (`Agents/local_tool_provider.py`): "Limit the status to
  one path (file or directory), relative to the workspace root; also
  selects the repository. Omit for the whole repository." -- phrased like
  git_diff's and no longer implying the discovery-only contract.
- **Tests**: `Tests/Tools/test_git_tool_sensitive_paths.py` gains
  `test_git_status_scopes_to_the_requested_path` (subdir entries in,
  outside entries out, default still whole-repo),
  `test_git_status_magic_in_the_path_argument_cannot_invert_the_scope`
  (the `:(exclude)notes.txt` filename scopes literally; the assertion is
  on the REAL file's exact entry shape because the magic file's own name
  contains the substring), and
  `test_git_status_positive_pathspec_does_not_weaken_the_denylist`
  (naming `.ssh/id_rsa` as `path` is refused by the same choke point
  TASK-19551 pinned for git_diff; the denylist pathspecs riding alongside
  the positive one is guaranteed structurally by the tripwire).
  `Tests/Tools/test_git_tool_impls.py` gains
  `test_git_status_description_matches_the_scoping_implementation`
  (bootstrap_profile): the provider schema must state the Limit contract
  and must not carry the old phrasing, AND git_status's AST must still
  contain the `_literal_pathspec`/`_denylist_pathspecs` calls -- either
  half drifting fails it.
- **No regression**: both Tools test files 56 passed;
  `Tests/Agents/test_local_tool_provider.py` shows the same 209
  pre-existing shared-profile `RecoveryRequired` failures at base and
  after (`diff` of sorted FAILED lists empty).
- **ADR required: no** -- truthfulness fix inside an existing tool's argv,
  on the pattern its two siblings already ship.

## Notes

Worth doing precisely because the docstring was already fixed. Half the drift
was closed and the half the model actually reads was not — which is the version
of this problem that still costs an agent a wasted turn.
