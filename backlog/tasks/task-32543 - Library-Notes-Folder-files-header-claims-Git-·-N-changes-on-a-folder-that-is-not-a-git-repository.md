---
id: TASK-32543
title: >-
  Library Notes: Folder files header claims "Git · N change(s)" on a folder that
  is not a git repository
status: In Progress
assignee:
  - '@claude'
created_date: '2026-09-13 06:46'
updated_date: '2026-09-13 15:13'
labels:
  - library
  - notes
  - critique-3
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #3 (dev 5fd502dbac), both assessors saw the string, persona solo operator, Folder files workflow. A P2 #8.

**What happened.** Fresh vault fixture with no `.git` (`git rev-parse` fails). After one edit the header reads "Folder files · Folder: vault · Git · 1 change" at 100x30 (A 57) and at 235x52 (B 40, 50). The solo operator reads "Git" as "this folder is version-controlled". Captures: A 57; B 40, 50.

**Cause.** INFERRED: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py:230-232` builds `f"Git · {git_changes} {change_word}"` from the session change count with no repository gate in the renderer; the git failure/uncertain/running branches above it only fire when a git probe ran. Docs contradicted: file-notes.md's header is "Linked · Local folder: <folder>" — the suffix is undocumented and false.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The "Git · …" suffix renders only after a successful repository check
- [x] #2 A non-repository folder with session changes reads "N session change(s)" (or nothing), never "Git"
- [x] #3 A test renders the header for a non-repository folder with one change and asserts "Git" is absent; file-notes.md documents the suffix
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce live on a non-git vault (fn-00): header says Git · 1 change after one edit\n2. RED: resolver test with repository_confirmed=False, git_changes=1 asserts no Git; second test asserts Git only when confirmed\n3. Fix: thread repository_confirmed into resolve_file_note_status_channels; the workspace derives it from a rev-parse discovery per session binding (git -C root rev-parse), or trust\n4. Update the pins that asserted the old suffix without a repository; add a real-git pin for the confirmed side\n5. GREEN, live captures on vault-plain and vault-git, guide header paragraph + stamp
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The header's change count is no longer allowed to imply version control.
`resolve_file_note_status_channels` gained a `repository_confirmed` flag: with
it the suffix is the old "· Git · N change(s)"; without it "· N session
change(s)". The flag defaults to False, so a caller that does not know cannot
accidentally claim a repository.

The workspace supplies the fact rather than guessing at it: one `git
rev-parse` discovery per session binding (`_ensure_repository_probe` ->
`_SessionGitService.discover`), plus the already-granted
`snapshot.trusted_repository`. The probe is a worker, so the header renders
"N session change(s)" first and upgrades when the answer lands -- the honest
order.

Trade-off: the probe costs one `git rev-parse` per linked folder per session.
The alternative (probing on every status render) was rejected as chatty, and
caching per binding means a folder that BECOMES a repository mid-session keeps
the session wording until it is re-linked. That is the safe direction of
error.

Pins: `Tests/UI/test_library_file_notes_workspace.py::test_header_never_says_
git_for_a_non_repository_with_changes` / `::test_header_says_git_only_after_a_
confirmed_repository` (the resolver, both ways) and
`Tests/UI/test_library_notes_w4_file_notes.py::test_the_header_says_git_only_
when_the_folder_is_a_repository[False/True]`, which runs the REAL Git service
against a real throwaway repository and a plain folder beside it, so the fact
is `git rev-parse` truth rather than a flag a test set. Three existing pins
that asserted "Git · 1 change" on a folder with no Git service attached now
assert the new copy.

RED proofs (each by patching the fix out of a scratch copy of the file, never
`git stash`): restoring the ungated `status_copy` line fails all three
resolver pins and the `[False]` route pin; removing the
`_ensure_repository_probe()` call fails the `[True]` route pin.

Live (captures under `wave4-caps/file-notes/`): `fn-32` /
`fn-41-header-plain-session-change-{235x52,100x30}` read "Folder files ·
Folder: vault-plain · 1 session change" after one Ctrl+S on a vault where
`git rev-parse` fails; `fn-51` / `fn-52-header-git-{235x52,100x30}` read
"· Git · 1 change" on the real repository.

Files: `tldw_chatbook/Widgets/Library/library_file_notes_workspace.py`,
`Tests/UI/test_library_file_notes_workspace.py`,
`Tests/UI/test_library_notes_w4_file_notes.py`,
`Docs/User_Guide/library/file-notes.md`.
<!-- SECTION:NOTES:END -->
