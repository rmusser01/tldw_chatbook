# TASK-21139 Windows-Safe Backlog Paths Design

## Status

Approved for implementation planning on 2026-08-23.

## Problem

Windows CI fetches the repository successfully but cannot check out commit
`1a81909eed20ffaa64220f9d2eb16930dd151710`. Git for Windows exits 128 before
any project setup or test because this tracked path contains `>`:

`backlog/tasks/task-21130 - TTS profile v3->v4 migration snapshots the entire reference-BLOB table into memory twice.md`

The file arrived in commit `46cb7bc1f531e76162d8dfa86993882d75f691c4`.
The existing Backlog guard checks filename/frontmatter ID uniqueness but does
not check whether task paths are representable on Windows. Because the guard
runs on Ubuntu, it can inspect and reject such names as a merge gate without
needing a Windows checkout; GitHub may still schedule Windows jobs concurrently.

## Scope

This task covers files directly inside the three Backlog task buckets already
owned by `scripts/check_backlog_task_ids.py`:

- `backlog/tasks`
- `backlog/completed`
- `backlog/archive/tasks`

It does not establish a repository-wide filename policy, enforce checkout-root
dependent path-length limits, change task IDs/content, or alter Windows runner
configuration.

## Selected approach

Extend `scripts/check_backlog_task_ids.py` rather than adding a second checker.
The script is already stdlib-only, scans all three task buckets, runs from local
preflight, Backlog Guard, and Derived Artifacts, and is covered by focused
architecture tests. Reusing it gives every existing entry point the same rule
without workflow changes or dependencies.

Rename only TASK-21130's path, replacing `v3->v4` with `v3-to-v4`. Preserve its
frontmatter, title, description, acceptance criteria, and ID byte-for-byte.
Update every resolvable link, command, or path-valued reference to the old path.
Historical incident evidence may retain the literal failing path when it is
clearly quoted as past evidence rather than presented as a live repository path.

## Validation contract

For each file directly inside a configured task bucket, the guard reports a
Windows-incompatible path when its basename:

- contains `<`, `>`, `:`, `"`, `/`, `\`, `|`, `?`, or `*`;
- contains an ASCII control character (`U+0000` through `U+001F`);
- ends in a dot or space; or
- has a Windows device-name stem (`CON`, `PRN`, `AUX`, `NUL`, `COM1` through
  `COM9`, `COM¹` through `COM³`, `LPT1` through `LPT9`, or `LPT¹` through
  `LPT³`), case-insensitively. The device stem is the portion before the first
  period, so every extension depth remains invalid (`NUL.txt` and
  `NUL.tar.gz`).

These rules follow Microsoft's Win32 file-naming contract:
<https://learn.microsoft.com/en-us/windows/win32/fileio/naming-a-file>.

The implementation exposes a small pure basename classifier plus a bucket
scanner that returns each invalid repo-relative path and reason. Existing
`duplicate_ids()` callers and return values remain unchanged. `main()` reports
both duplicate-ID and incompatible-path failures in one invocation and exits 1
if either class exists.

Directories are not recursively scanned: the Backlog CLI's task records live
directly in the three existing buckets, matching the current guard contract.

## Failure reporting

The report must identify every incompatible path and the reason it failed, then
state that task content may retain punctuation while the filename must use a
Windows-safe spelling. This makes the repair actionable without needing a
Windows machine to rediscover the rule.

## Testing

Focused TDD will add tests to
`Tests/Architecture/test_derived_artifact_checkers.py` before implementation:

1. Invalid-character/control/trailing/reserved cases fail with their paths and
   reasons.
2. Ordinary task filenames pass, including punctuation that Windows allows.
3. Duplicate-ID reporting continues to work when path validation is enabled.
4. The real repository inventory passes after TASK-21130 is renamed.

Filesystem fixtures on POSIX cover names Linux can create. `/` and NUL are
exercised against the pure basename classifier because POSIX cannot create a
single basename containing either character. Reserved-name tests include
superscript COM/LPT aliases and a multi-extension `NUL.tar.gz` case.

Verification is limited to the checker tests, local uniqueness gate, checker
script, Ruff/format, and affected CI. On the repair PR, both previously failing
job types must show `actions/checkout` completing successfully before their
project steps:

- `GGUF source evidence - windows-latest` (failed in run `32617893248`); and
- `Artifact leases - Python 3.11 - windows-latest` (failed in Tests run
  `32617893237`).

No broad local suite is needed.

## Alternatives rejected

- **Rename only:** restores checkout but permits immediate recurrence and does
  not satisfy TASK-21139's prevention criterion.
- **Separate Windows-path checker:** duplicates the same bucket discovery,
  workflow triggers, preflight wiring, and tests for no independent benefit.
- **Repository-wide path policy:** substantially broadens ownership and edge
  cases beyond the observed Backlog failure.

## ADR check

ADR required: no

ADR path: N/A

Reason: this is a focused CI portability bugfix inside an existing guard and
does not change storage, runtime, security, dependency, or cross-module
architecture.
