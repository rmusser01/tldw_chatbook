---
id: task-32927
title: Shell metacharacter blacklist rejects legitimate user filenames
status: To Do
assignee: []
labels:
  - bug
  - usability
created_date: '2026-09-23'
---

## Description

`Utils/path_validation.validate_path_simple` refuses any path containing
`|`, `;`, `&&`, `||`, `` ` ``, `$(` or `${`. Those are **command-injection**
guards, and they are all legal characters in a POSIX filename. A user who
selects a file called `Q3 P&L; final.txt` — which the OS creates happily — is
told the path "contains a dangerous pattern" and the operation fails before
the file is opened.

**They protect nothing here.** The repo contains **zero** `shell=True` and no
`os.system`; every validated path goes to `open()`/`stat()`, never to a shell.

PR #2823 fixed this for the two ingest boundaries by adding
`reject_shell_metacharacters=False` (default unchanged, so no other caller's
behaviour moved). The remaining callers still carry the defect:

- `UI/Evals/snippet_editor.py:629` — importing a user-chosen snippet file
- `UI/Evals/results_grid.py:889` — choosing an export destination

`config.py` and `app.py` also call it, but on app-owned paths rather than
user-selected ones, so they are lower priority.

## Acceptance Criteria

- [ ] A file whose name contains `;`, `|`, `` ` ``, `&`, or `${` can be
      selected for snippet import and for results export
- [ ] Traversal (`../..`), NUL bytes and `~` expansion are still refused on
      every one of those paths — verified by a test that is **seen red** with
      the guard removed
- [ ] A decision is recorded on whether the shell patterns should remain in
      `validate_path_simple` at all, given there is no shell call site in the
      repo; if they stay, the reason is written down next to the list
- [ ] The default value of `reject_shell_metacharacters` is not changed
      without auditing all five current call sites
