# File Notes full-app UAT — 2026-07-30

## Verdict

**Accepted after TASK-1411 remediation; guarded commit behavior remains
passed.**

The test launched the real Chatbook Textual application in a PTY and exercised
the Library navigation, native folder picker, process-only Git trust prompt,
File Notes editor, autosave, session staging, commit review, commit execution,
and unrelated-staged-content block against a disposable Git repository and
SQLite replica.

The initial run found two reachability defects:

1. At `120x40` and `160x45`, Library > Notes rendered the source strip as
   `Database |`; the Files choice was not visible or operable.
2. At `40x20`, a realistic linked-root path consumed enough vertical space
   that Prepare session for commit clipped its status and staging/commit
   actions below the viewport. Keyboard traversal did not make a usable Stage
   action available.

The focused remediation is tracked by
[TASK-1411](../../../../backlog/tasks/task-1411%20-%20Restore-File-Notes-entry-and-compact-terminal-Prepare-usability.md).
TASK-1411 repaired both defects. A second full-app PTY run used the current
application directly, with no diagnostic launcher or runtime override:

- At `120x40`, Library > Notes rendered
  `Database (selected) | Files`; selecting Files changed it to
  `Database | Files (selected)` and mounted the retained workspace.
- At `40x20`, the linked-root summary stayed on one line as
  `Linked —...o/notes`. The visible Details action opened the exact root and
  warning text in a keyboard-focused, read-only dialog.
- Prepare session for commit scrolled from status to the selected-path,
  Stage/Unstage, Stage All/Unstage All, and Commit staged controls.
- In an actionless Prepare state, Shift+Tab focused the scroll surface and End
  revealed `Commit staged (0)` plus the “Stage at least one…” guidance.
- A new autosaved session edit was staged from the compact UI. After
  `unrelated.txt` was staged externally, review remained fail-closed and
  displayed the safe recovery instruction.

Guarded push remains subsequent work.

## Build and environment

- Feature source HEAD:
  `c07fef609f9b6eec0732b8ab28ec13d086515f1d`
- Merged by PR #1098 into `dev`:
  `665ef1c01a48130b8da3bd80eea17054de54e976`
- App launch: real `python -m tldw_chatbook.app` process in a tmux PTY
- App configuration, user data, Git repository, and SQLite replica were
  isolated under `/private/tmp/chatbook-fullapp-uat.z0StBm`
- Git fixture:
  `/private/tmp/chatbook-fullapp-uat.z0StBm/repo`
- Notes root:
  `/private/tmp/chatbook-fullapp-uat.z0StBm/repo/notes`
- SQLite replica:
  `/private/tmp/chatbook-fullapp-uat.z0StBm/data/uat_operator/file_notes.sqlite`

The first run used the unmodified full app and exposed the missing Files
choice. A process-local launcher then applied only a diagnostic source-strip
layout override and selected Files at startup so the remaining workflow could
be tested. It did not modify repository source. The override made
`Database | Files` visible, confirming that the blocker was presentational,
not missing feature registration.

## Acceptance matrix

| Scenario | Result | Evidence |
|---|---|---|
| Open Library > Notes and select Files at normal width | **Pass after remediation** | Direct full-app retest passed at `120x40`; mounted geometry and keyboard coverage passed at both `120x40` and `160x45`. |
| Select root through the native picker | Pass after entry-only diagnostic bypass | The disposable `notes` folder was linked. |
| Decline/reopen/accept process-only Git trust | Pass | Cancel remained the safe default; `Trust and check status` enabled Session Git for this process only. |
| Edit Markdown body and autosave | Pass | Both notes reached `Saved`; disk bytes and replica hashes matched. |
| Preserve frontmatter exactly while editing body separately | Pass | `one.md` retained its original frontmatter bytes and received only the intended body edit. |
| Stage all current-session paths | Pass | Exactly two notes were staged; unrelated worktree content remained unstaged. |
| Review and cancel back to the form | Pass | Subject/body and staging state were preserved. |
| Review and commit current-session paths | Pass | Commit `540f5e02da489216a3b9bc1bd87c538b03d67c21` contains only the two session notes. |
| Promise plus count after success | Pass | `Committed 2 session notes; unrelated changes untouched.` remained explicit. |
| Navigator/editor switching at `40x20` | Pass | The selected note could be opened, edited, saved, and returned to Navigator. |
| Prepare session actions at `40x20` with long root | **Pass after remediation** | The one-line root summary preserved content height; the Prepare surface scrolled to every staging and commit action. |
| Block commit when unrelated staged content exists | Pass | Review was refused; HEAD and the complete staged state were unchanged; the UI explained how to recover safely. |

## Git proof

Successful Chatbook commit:

```text
commit  540f5e02da489216a3b9bc1bd87c538b03d67c21
parent  a6dac2e9bdc19b0db0c9e0a0dfc549b4b09d48a7
subject UAT: update two session notes
author  Chatbook UAT <chatbook-uat@example.invalid>
files   notes/one.md
        notes/study/two.md
```

After externally staging `unrelated.txt` and staging one changed session note,
Chatbook refused review with:

```text
The complete staged state does not exactly match this session. If Git has
unrelated staged changes, commit or unstage them outside Chatbook; then
Refresh and review this session again.
```

HEAD remained `540f5e02da489216a3b9bc1bd87c538b03d67c21`,
and the index still contained both `notes/one.md` and `unrelated.txt`. The
commit subject draft remained available.

## Disk and replica proof

The disk and SQLite `raw_bytes` SHA-256 values matched after autosave:

| Path | SHA-256 |
|---|---|
| `notes/one.md` | `c740cfaca39684b68a49d9085a7b903cd097e66f8416c457531e7c312ed6fdad` |
| `notes/study/two.md` | `1f7c3a6f8b2b7dbcbd1a5f6ebd4d0ea19a2abc028a38388d49c03737d9a2526f` |

The corresponding raw-byte lengths were 77 and 50 bytes.

## Focused remediation checks

After the no-bypass full-app retest, the affected mounted UI files passed:

```text
Tests/UI/test_library_file_notes_workspace.py
Tests/UI/test_library_file_notes_git.py
169 passed, 1 dependency warning in 69.38s
```

The complete-staged-state integration regression passed:

```text
Tests/Notes/test_file_notes_git_commit_integration.py::test_complete_commit_proof_blocks_unrelated_staged_without_disclosure
1 passed in 0.69s
```

Targeted Ruff and Python compilation passed. The Impeccable layout detector
reported `[]` for the three changed UI modules. The rendered source-choice
test uses real Tab/Shift+Tab/Enter input, asserts viewport containment at
`120x40` and `160x45`, and verifies the explicit selected state in both
directions. The compact Prepare regression uses real Tab/Enter input at
`40x20`, verifies focus-driven scrolling in both actionable and actionless
states, and traverses Stage, Unstage, and guarded commit entry. The exact
linked-root state is also opened and closed entirely through keyboard input.

No full suite or broad CI run was performed.
