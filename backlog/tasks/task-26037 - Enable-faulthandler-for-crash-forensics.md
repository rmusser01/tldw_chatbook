---
id: TASK-26037
title: Enable faulthandler for crash forensics
status: Done
assignee:
  - '@claude'
created_date: '2026-08-31 15:47'
updated_date: '2026-08-31 19:15'
labels:
  - ops
  - reliability
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A hard crash or hang leaves no evidence. Verified on origin/dev: a named grep for excepthook and faulthandler across tldw_chatbook returns three hits, all comments - nothing is installed. Targeted crash guards exist (Utils/text_selection_crash_guard.py, Utils/fd_protection.py) but they catch known cases; an unexpected segfault, a C-extension crash or a deadlock produces nothing to diagnose from. Hermes enables faulthandler to a dedicated log with all-threads dumps plus a signal handler for on-demand dumps. The private log directory already resolves at config.py:7849.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 faulthandler is enabled at startup, writing to a file in the existing private log directory
- [x] #2 Dumps include all threads, so a deadlock is diagnosable and not just a crash
- [x] #3 A signal handler allows dumping stacks on demand from a hung process, on platforms that support it
- [x] #4 The dump file is created with the same restrictive permissions as other private logs
- [x] #5 The dump file is size-bounded or rotated so it cannot grow without limit
- [x] #6 Enabling this adds no measurable startup cost - measured and recorded
- [x] #7 Tracebacks are treated as potentially sensitive: the dump lives under the private log path and is not included in any shareable output
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no. Installs a stdlib diagnostic at startup; no new dependency, storage format, or seam.

1. Write to the private log directory, not stderr -- a TUI owns the screen and a dump printed there dies with the alternate buffer.
2. Reuse the existing private-path helpers so the dump inherits the same directory hardening and 0600 mode as every other private log.
3. all_threads=True, so a deadlock is diagnosable and not just a crash.
4. Register SIGUSR2 for on-demand dumps from a process that is still hung, guarded for platforms without it.
5. Bound the file at startup rather than adding rotation machinery.
6. Never raise -- but log the failure class, so a misconfiguration is not invisible.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added `Logging_Config.enable_crash_forensics()` and called it at the top of `configure_application_logging`, so a segfault, native-extension crash or deadlock leaves evidence.

**Why a file and not stderr.** A TUI owns the screen; a traceback printed to stderr dies with the alternate buffer. The dump goes to `faulthandler.log` in the private log directory, created through the existing `secure_private_directory` helper so it inherits the same hardening as every other private log, and opened with `O_CREAT|O_APPEND|O_WRONLY|O_NOFOLLOW` at mode 0600. The stream is held for the process lifetime because faulthandler writes to the descriptor from a fault or signal context and cannot reopen it lazily.

**Bounding (AC#5).** Dumps append, so a process repeatedly hitting the same fault could grow the file without limit. Reset at startup once past 1 MiB, which preserves the most useful artifact -- the dump from the crash that just happened -- while capping the worst case. Marked with a `ponytail:` comment naming the ceiling: this is truncate-at-startup, not real rotation, and a generation scheme is the upgrade path if one file proves insufficient.

**Verified end-to-end, not just by flag assertions.** A live process with a deliberately stuck worker thread was sent SIGUSR2: the dump contained 2 thread headers including the `worker_that_is_stuck` frame, and the process kept running afterwards. That is AC#2 and AC#3 demonstrated rather than asserted from the arguments passed.

**AC#6 measured:** median 0.105 ms over 30 runs (max 0.206 ms). No meaningful startup cost.

**AC#7:** the dump lives under the private log path at 0600. It is not reachable by any shareable output because chatbook has no diagnostics-bundle or upload command at all -- confirmed by named grep during the 2026-08-31 parity pass, and recorded there as a deliberate privacy stance rather than a gap.

**One thing worth keeping.** The `except Exception -> return None` guard, added so a diagnostic aid can never become a boot failure, silently swallowed a real `TypeError` during development (`secure_private_directory` takes keyword-only `application_owned`). It now logs the exception *class* at debug -- not the message, which could carry a path -- so the same mistake is visible next time instead of presenting as "forensics just doesn't work".

**Verification:** 8 tests in the new file; 365 pass across `Tests/Metrics/`, `Tests/App/` and the MCP redaction files.

**Files:** `tldw_chatbook/Logging_Config.py`, `Tests/Metrics/test_crash_forensics.py` (new).

## Review round — three tests that could not fail

**AC#1/#2/#3 were pinned by tests that would survive the feature being deleted.**
- `test_crash_forensics_enables_faulthandler` asserted `faulthandler.is_enabled()`, but pytest's own faulthandler plugin enables it before any test runs, so the assertion held regardless. It now asserts on what this function actually produces — the returned path and the installed stream.
- The `all_threads` spies defaulted the very keyword under test (`def spy_enable(file=None, all_threads=True)`), so the assertion passed even if production stopped passing it. They now capture `**kwargs` with no defaults.

Verified by mutation: gutting `enable_crash_forensics` to `return None` now fails 6 of 9 tests; before these fixes it failed 3.

**Two hardening fixes.** `os.truncate(path, 0)` follows symlinks, so a link planted at `faulthandler.log` would have had its *target* truncated before the `O_NOFOLLOW` open refused it — bounding now happens through `os.ftruncate` on the already-opened descriptor. The failure path also leaked the file descriptor, and now closes it.

`configure_application_logging` runs twice in a normal boot, so installation is now explicitly idempotent with a test pinning it; the test fixture resets the module-level stream between cases.

Dropped `raising=False` from the `get_cli_log_file_path` patch: the name exists, so it only disabled the guard — if the call were ever refactored, these tests would have silently written into the user's real private log directory and still passed.
<!-- SECTION:NOTES:END -->
