# File Notes full-app UAT — 2026-07-30

## Verdict

**UX acceptance blocked; guarded commit behavior passed.**

The test launched the real Chatbook Textual application in a PTY and exercised
the Library navigation, native folder picker, process-only Git trust prompt,
File Notes editor, autosave, session staging, commit review, commit execution,
and unrelated-staged-content block against a disposable Git repository and
SQLite replica.

Two reachability defects prevent full UX acceptance:

1. At `120x40` and `160x45`, Library > Notes rendered the source strip as
   `Database |`; the Files choice was not visible or operable.
2. At `40x20`, a realistic linked-root path consumed enough vertical space
   that Prepare session for commit clipped its status and staging/commit
   actions below the viewport. Keyboard traversal did not make a usable Stage
   action available.

The focused remediation is tracked by
[TASK-1411](../../../../backlog/tasks/task-1411%20-%20Restore-File-Notes-entry-and-compact-terminal-Prepare-usability.md).
Guarded push remains subsequent work; it should not be added on top of an
unreachable commit workflow.

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
| Open Library > Notes and select Files at normal width | **Fail** | Only `Database |` was visible at `120x40` and `160x45`. |
| Select root through the native picker | Pass after entry-only diagnostic bypass | The disposable `notes` folder was linked. |
| Decline/reopen/accept process-only Git trust | Pass | Cancel remained the safe default; `Trust and check status` enabled Session Git for this process only. |
| Edit Markdown body and autosave | Pass | Both notes reached `Saved`; disk bytes and replica hashes matched. |
| Preserve frontmatter exactly while editing body separately | Pass | `one.md` retained its original frontmatter bytes and received only the intended body edit. |
| Stage all current-session paths | Pass | Exactly two notes were staged; unrelated worktree content remained unstaged. |
| Review and cancel back to the form | Pass | Subject/body and staging state were preserved. |
| Review and commit current-session paths | Pass | Commit `540f5e02da489216a3b9bc1bd87c538b03d67c21` contains only the two session notes. |
| Promise plus count after success | Pass | `Committed 2 session notes; unrelated changes untouched.` remained explicit. |
| Navigator/editor switching at `40x20` | Pass | The selected note could be opened, edited, saved, and returned to Navigator. |
| Prepare session actions at `40x20` with long root | **Fail** | Status/actions were clipped; keyboard traversal did not expose a usable staging action. |
| Block commit when unrelated staged content exists | Pass | Review was refused; HEAD and the complete staged state were unchanged. |

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
The complete staged state does not exactly match this session.
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

## Focused automated check

The existing mounted source-switch regression passed before any remediation:

```text
Tests/UI/test_library_file_notes_workspace.py::test_library_database_files_switch_retains_workspace_and_database_canvas
1 passed in 6.02s
```

That test proves direct widget switching, but not that both source controls are
rendered and reachable in the full application. TASK-1411 therefore requires
rendered-geometry and keyboard-operation coverage. No full suite or broad CI
run was performed.
