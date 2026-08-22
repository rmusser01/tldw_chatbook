---
id: TASK-19700
title: 'Security: agent write tools can modify .git/ inside a workspace root'
status: To Do
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
- [ ] #1 A policy decision is recorded (in the task or an ADR) for whether agent write tools may modify anything under `.git/`, and for `.gitattributes`, with the reasoning for legitimate exceptions
- [ ] #2 The chosen policy is enforced at the path-resolution boundary shared by the write tools, not per-tool
- [ ] #3 A write attempt that the policy refuses fails with an explanation naming the path, never silently
- [ ] #4 Tests cover: a refused `.git/config` write, a refused `.git/HEAD` write, and whatever exceptions the policy deliberately allows
- [ ] #5 The audit states which other features shell out to git against agent-writable roots, so the blast radius of this gap is recorded rather than assumed
<!-- AC:END -->
