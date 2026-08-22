---
id: TASK-19632
title: >-
  git_diff leaks denylisted file content when path is omitted
status: Done
assignee: []
created_date: '2026-08-21 12:05'
labels:
  - security
  - agents
  - tools
priority: high
dependencies: []
---

## Description

Found and reproduced during TASK-19551 (which closed the same class of hole for
the `fs_*` tools), then re-verified by that task's reviewer. TASK-19551 made
`Tools/local_tool_impls.py::resolve_workspace_path` enforce the sensitive-path
denylist (`Utils/sensitive_paths.py`) for every path a model NAMES, and made the
three enumerating `fs_*` tools (`fs_list`/`fs_glob`/`fs_grep`) filter the
entries they present but the model never named.

The `git_*` tools share that choke point for their path argument, but have no
equivalent output filter — and on three of them `path` is OPTIONAL. When it is
omitted, no candidate path reaches the denylist at all: the seam is explicit at
`Agents/local_tool_provider.py:444-445`, where `path_targets` returns the
repository root and stops:

```
if raw_path is None:
    return (ToolPathTarget(path=repo_root, kind="repository"),)
```

git then enumerates the whole repository on the tool's behalf and the tool
returns its output verbatim. Under the shipped `workspace_root` default (the
app's cwd at startup), an app launched from `$HOME` inside a git repository puts
`~/.ssh/id_rsa` in that enumeration.

Measured on this branch with an isolated `$HOME` that is a git repo containing a
synthetic `.ssh/id_rsa` (probe re-run twice; second run added the clean-tree
case):

| call | leaks |
| --- | --- |
| `git_diff(commit_range="HEAD~1..HEAD")`, no `path` | **file CONTENT**, on a CLEAN worktree |
| `git_diff()`, no `path`, dirty worktree | **file CONTENT** |
| `git_diff(stat=True)`, no `path` | file NAME only |
| `git_status()`, no `path`, dirty worktree | file NAME only |
| `git_log()`, no `path` | nothing — commit metadata only |
| any of the above WITH `path=".ssh/id_rsa"` | nothing; refused "protected path" (TASK-19551) |

Two properties make this worse than the `fs_grep` gap TASK-19551 closed:

1. **No write primitive is required.** The `commit_range` form reads the
   credential out of history, so a read-only agent on a clean checkout is
   enough. An earlier draft of this finding claimed a dirty worktree was
   needed; that is wrong.
2. It is reachable by **prompt injection** from fetched web content, exactly
   like the hole TASK-19551 closed — the model only has to call a read-only,
   `reads`-tagged git tool with no arguments.

Scope note: `git_log` is clean and `git_status` discloses names only. The fix
should not be described or built as "the three git tools leak".

## Acceptance Criteria

- [x] `git_diff` never returns the content of a path `is_sensitive_path` refuses,
      whether that path is reached via the worktree, the index, or a
      `commit_range`, and whether or not the caller supplied `path`
- [x] `git_status` never names such a path
- [x] The behaviour is enforced by construction (e.g. denylisted paths inside the
      repository are excluded from git's own output via pathspec, or the output
      is filtered before it is returned) rather than by asking the model to pass
      `path`
- [x] A born-red test reproduces the clean-worktree `commit_range` content leak
      and the `stat=True`/`git_status` name leaks, and each is refused after
      the fix
- [x] A test pins that `git_log` output is unchanged, and that ordinary
      (non-denylisted) diffs, stats and statuses are unchanged — the fix must not
      silently truncate legitimate multi-file diffs
- [x] `Utils/sensitive_paths.py`'s module docstring and
      `Tools/local_tool_impls.py`'s drop the TASK-19632 exception once it no
      longer applies (both currently state it explicitly)

## Notes

The two obvious implementations both need care, which is why this is its own
task rather than a rider on TASK-19551:

* **Pathspec exclusion** (`git diff -- . ':(exclude)<relpath>'`) keeps git the
  authority and never parses diff text, but the exclusions must be computed
  per call from the resolved denylist, restricted to paths inside the
  repository root, and rendered repo-relative; `run_git`'s argv allowlist
  (`_validate_argv`) has to accept them.
* **Output filtering** means parsing unified diff / porcelain v2 text, and a
  half-parsed diff is worse than none.

`Tests/Tools/test_local_tool_sensitive_paths.py::
test_every_workspace_rooted_function_uses_the_choke_point` already covers
`git_tool_impls` structurally and carries a NOTE that reaching the choke point
proves the path ARGUMENT is checked, not that output is filtered — that note
points here.


## Implementation Plan

1. Reproduce all five measured cases against an isolated `$HOME` that is a git
   repository, and confirm which of them actually leak (the table in the
   description, re-measured on this branch).
2. Choose between pathspec exclusion and output filtering, and record the
   argument.
3. Give `Utils/sensitive_paths.py` a structural accessor for its own denials so
   the git renderer cannot become a second, hand-maintained copy of the
   denylist.
4. Render those denials as git exclude pathspecs in `git_tool_impls`, one
   pathspec form per denial rule so no rule is widened or narrowed in
   translation.
5. Prove the pathspec-injection case is closed rather than assuming `--`
   handles it.
6. Born-red tests for every leak; hard negative pins for `git_log` and for
   ordinary diffs/stats/statuses.
7. Remove the TASK-19632 exception paragraphs from both module docstrings.

## Implementation Notes

**Chosen: pathspec exclusion.** git stays the authority on what matches, no
unified-diff or porcelain text is ever parsed, and the exclusions are
recomputed from the live denylist on every call (so a `TLDW_CONFIG_PATH`
switch or a relocated database is observed immediately). Output filtering was
rejected on the task's own grounds -- a half-parsed diff is worse than none --
and it would have had to parse two unrelated formats, one of them NUL-delimited
porcelain v2.

The seam matters as much as the mechanism. `Utils/sensitive_paths.py` gained
`sensitive_exclusions_under(root)`, which returns this module's denials as
structured `SensitiveExclusion(kind, value)` entries relative to a root;
`git_tool_impls._denylist_pathspecs` renders them. So the denylist stays the
one place that knows WHAT is denied and the git module the one place that knows
HOW to say it to git. Each kind maps to the pathspec form that expresses
exactly that rule: a container's `direct_children` becomes
`:(exclude,glob,icase)<dir>/*`, where `*` does not cross `/` under glob magic,
so `tool_sandbox/` stays diffable while a loose file beside it does not -- the
same line `is_sensitive_path`'s own container rule draws. An unrecognized kind
raises rather than being skipped.

**Argv injection: found live, and closed.** While building the fix I measured a
vector the task did not list. `:(exclude)notes.txt` is a legal POSIX filename,
so `git_diff(path=":(exclude)notes.txt")` passes the choke point as an ordinary
confined path -- and git then reads it as MAGIC and inverts the diff's scope,
returning the rest of the repository with `~/.ssh/id_rsa`'s content in it. `--`
does NOT stop this: `--` ends OPTION parsing, not magic parsing. Every pathspec
this family builds is now rendered with explicit magic. The proof is
`test_pathspec_magic_in_the_path_argument_cannot_invert_the_scope` (red at base
with the key's content in the failure message) plus an AST tripwire,
`test_every_pathspec_this_family_builds_carries_explicit_magic`, which fails on
any bare value spliced after `--` (red at base, naming `git_log` and
`git_diff`).

`git_blame` is deliberately untouched: `git blame` takes a plain PATH, not a
pathspec, and rejects magic outright (`fatal: no such path ':(literal)a.txt' in
HEAD`, verified) -- so it never interprets a magic-shaped filename either.

**Deliberate non-changes.** `git_log` is not given exclusions: its `--format`
emits commit metadata only, it was measured leaking nothing, and excluding
denied paths would delete commits from a legitimate history while protecting
nothing. A test asserts a commit touching ONLY `~/.ssh/id_rsa` still appears.
And nothing announces that an exclusion took effect: the only honest note would
state that this repository contains a protected path, which is the same
disclosure `stat=True`/`git_status` were leaking. A model that NAMES the path
still gets the "protected path" refusal, which is the actionable case.

**Case folding (composed with TASK-19800, merged mid-task).** Every exclusion
carries `icase`, for the reason 19800 gives for folding the denylist itself:
git records whatever spelling a path was added under, and a denial that misses
`.SSH/id_rsa` is a leak. Folding an exclusion only ever removes more, so it
fails in the cheap direction -- unlike the SCOPING pathspec, which is
deliberately left case-sensitive because folding it would ADD files to what the
model gets back. `_relative_within` decides containment through 19800's
`_is_within` rather than `Path.relative_to`'s exact-case parts, so the git side
has no second, differently-normalized comparison path.

**A trap left for the next reader.** TASK-16801's lesson recommends
`GIT_LITERAL_PATHSPECS=1` as blanket hardening for git argv. It is incompatible
with this fix and fails SILENTLY: under it `:(exclude,literal)<path>` is taken
as a literal filename, matches nothing, and every `git_diff`/`git_status`
returns empty output with exit 0 (verified, git 2.39) -- breaking the feature
and every exclusion at once. `_git_environment` carries a comment and
`test_the_runner_environment_never_sets_a_pathspec_mode` asserts no
`*_PATHSPECS` variable appears there.

**Evidence.** Born-red at `origin/dev` (`d4f3f9776`, i.e. including TASK-19800):
10 of the 19 tests in the new file fail, each failure message carrying the
leaked bytes -- `git_diff(commit_range=..., no path)`, `stat=True`,
`staged=True`, dirty worktree, `git_status` (including an UNTRACKED denylisted
file), the pathspec injection, the AST tripwire, the container rule, the name
rule, and the case variant. The remaining 9 pass at base and after, which is
the point: they are the negative pins (`git_log` unfiltered; ordinary diff and
stat byte-identical to raw git run with the same flags; ordinary status and
scoped diffs unchanged). All 19 pass on this branch.

Modified: `tldw_chatbook/Utils/sensitive_paths.py`,
`tldw_chatbook/Tools/git_tool_impls.py`,
`tldw_chatbook/Tools/local_tool_impls.py` (docstring),
`backlog/docs/lessons-testing-evidence.md`. Added:
`Tests/Tools/test_git_tool_sensitive_paths.py`.
