---
id: TASK-19700
title: 'Security: agent write tools can modify .git/ inside a workspace root'
status: Done
assignee: []
created_date: '2026-08-21'
labels:
  - security
  - tools
  - git
dependencies:
  - TASK-16801
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Neither `Tools/workspace_file_roots.py` nor `Tools/file_operation_tools.py`
excludes `.git/` from the paths an agent's write tools may touch, so an agent
operating in a workspace root that is a real git repository can create or
rewrite `.git/config`, `.git/HEAD`, and `.gitattributes`.

This was surfaced (not introduced) by TASK-16801's git modes arc, where it
sets the threat model for four separate defects found during review. Each was
reachable purely by repository-supplied content, with no dangerous flag
anywhere in the application's own argv:

- a remote named `--force`/`--mirror` reaching git's argv as an option;
- a branch named `--mirror`/`--all` via `.git/HEAD` doing the same;
- `remote.origin.push`, `remote.origin.mirror` and `push.default=matching`
  turning an ordinary push into a forced update or a ref deletion;
- `diff.external` and textconv drivers making a review pane render a
  fabricated diff, or a blank one for a genuinely changed file.

All four are fixed defensively inside the git-modes engine. This task is about
the upstream cause: an agent that can write `.git/` can reconfigure git's
behaviour for every feature that shells out to git, not only change review.
Deciding the policy is the work here — a blanket refusal is the obvious
candidate, but legitimate flows (a user asking an agent to fix a
`.gitignore`, or tooling that writes `.git/info/exclude`) need a considered
answer rather than an accidental one.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A policy decision is recorded (in the task or an ADR) for whether agent write tools may modify anything under `.git/`, and for `.gitattributes`, with the reasoning for legitimate exceptions
- [x] #2 The chosen policy is enforced at the path-resolution boundary shared by the write tools, not per-tool
- [x] #3 A write attempt that the policy refuses fails with an explanation naming the path, never silently
- [x] #4 Tests cover: a refused `.git/config` write, a refused `.git/HEAD` write, and whatever exceptions the policy deliberately allows
- [x] #5 The audit states which other features shell out to git against agent-writable roots, so the blast radius of this gap is recorded rather than assumed
<!-- AC:END -->

## Policy decision (AC #1)

**Writes to anything under a `.git` component: REFUSED.** No legitimate
agent flow needs them. The two candidate exceptions named when this task was
filed both dissolve on inspection: a user asking an agent to "fix the
gitignore" edits `.gitignore`, an ordinary tracked file at the repository
root, not `.git/`; and the app's own tooling that writes `.git/info/exclude`
is the shadow tracker, which owns a private `GIT_DIR` under the app data
directory and never routes through these tools.

**`.gitignore`, `.gitattributes` and `.github/`: ALLOWED.** They are ordinary
tracked files that share only a name prefix with `.git`, and refusing them
would break routine coding work. `.gitattributes` is the one with a security
history — it can assign a textconv/diff driver — but that vector was closed
defensively in TASK-16801 by passing `--no-ext-diff --no-textconv
--no-color` at every `git diff` site, so blocking the file as well would cost
real usability for no remaining gain.

**Reads are deliberately UNCHANGED.** ADR-032 adopted `allow_hidden` so a
coding agent can read a repository's dotfile configuration; making the guard
read-side too would undo that for no benefit against this threat, which is
about reconfiguring git rather than observing it. One read-side question is
worth separating rather than smuggling in here: `.git/config` can embed a
credential in a remote URL (`https://user:TOKEN@host/...`), so an agent
reading it can see a token. That is a disclosure question with a different
shape and a different fix, and it is NOT addressed by this task.

## Blast-radius audit (AC #5)

Every feature that shells out to git against a root an agent can write:

| Feature | Reads the user's `.git/config`? | Status |
|---|---|---|
| `Workspaces/git_workspace.py` (change-review git modes) | Yes, by design | Hardened in TASK-16801: option-shaped remote/branch names refused, explicit fully-qualified push refspec, `GIT_LITERAL_PATHSPECS=1`, machine-safe diff flags |
| `Tools/git_tool_impls.py` (read-only agent git tools) | Yes | Fixed-argv allowlist, sanitized env, and already passes `--no-ext-diff --no-textconv --no-color` |
| `Workspaces/change_tracking.py` (shadow tracker) | **No** — explicit `--git-dir` to an app-owned directory and every `GIT_*` scrubbed | Residual, narrow: it diffs the user's WORKING TREE without machine-safe diff flags, so a worktree `.gitattributes` naming a driver defined in the user's GLOBAL `~/.gitconfig` could still colour its output. Not reachable through this gap (an agent cannot write `~/.gitconfig`), recorded for completeness |
| `Notes/file_notes_git_*` | Operates on the Notes sync repository, not an agent workspace root | Out of scope |

`UI/CodeRepoCopyPasteWindow.py` and `Media/local_media_reading_service.py`
match on the string "git" but shell out to no git binary (the former uses the
GitHub HTTP API, the latter names a `git_repository` source type).

With this task's guard in place, an agent can no longer plant the
`.git/config` or `.git/HEAD` that every one of TASK-16801's four vectors
depended on — the defensive fixes there are now belt to this braces.

## Implementation Notes

Adds `is_git_metadata_write(path)` to `Utils/sensitive_paths.py` — the module
both write families already import — matching `.git` as an exact path
COMPONENT so prefix lookalikes stay writable, and covering the `.git` FILE a
linked worktree carries as well as the directory.

**The gap was in one family, not two.** The task text assumed both. Verified
by mutation while implementing: `Tools/file_operation_tools.py` refuses every
hidden component outright ("Access to hidden files/directories is not
allowed"), so `.git/` was already unreachable there — disabling the new check
leaves its test passing. The real gap was `Tools/local_tool_impls.py`, whose
`resolve_workspace_path` sets `allow_hidden=True` per ADR-032 precisely so a
coding agent can read dotfiles; that is where the guard is load-bearing, and
where the five refusal tests are red-first. The check is still wired into the
other family as defense-in-depth against a future `allow_hidden` adoption
there, and its tests say plainly which mechanism does the work today rather
than crediting the new one for a refusal it did not produce.

One test-quality note worth recording: the first version of the `patch_files`
test used a different tool's `*** Begin Patch` envelope, which this parser
rejects as `invalid_diff` — so it passed without ever reaching the guard. It
now uses a real unified diff and is paired with a control proving the same
payload APPLIES to an ordinary path, so the refusal is the guard's doing.

**Files:** `tldw_chatbook/Utils/sensitive_paths.py`,
`tldw_chatbook/Tools/local_tool_impls.py`,
`tldw_chatbook/Tools/file_operation_tools.py`,
`Tests/Tools/test_local_tool_sensitive_paths.py`,
`Tests/Tools/test_file_tool_sandbox.py`.
