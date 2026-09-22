---
id: TASK-32900
title: Explain private-path startup failures to first-time users
status: Done
assignee:
  - '@Robert'
created_date: '2026-09-22 03:31'
updated_date: '2026-09-22 04:11'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A fresh PyPI install on a machine where an ancestor of the config/data directories (e.g. ~/.config at 0775) is group/world-writable crashes during module import with a raw PrivatePathError traceback. First-time users have no clue why the app failed or what to do. Keep the fail-closed private-path posture (ADR-029/ADR-127) but surface a plain-language, actionable diagnostic at startup.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Startup prints a plain-language diagnostic naming the blocked directory, its mode/owner, and the exact chmod command
- [x] #2 The packaged tldw-cli command exits 1 with the diagnostic only (no raw traceback) for startup private-path failures
- [x] #3 python -m tldw_chatbook.app and other importers print the same diagnostic before the exception propagates
- [x] #4 PrivatePathError carries the offending directory path and mode/owner for walk-based refusals (diagnostics only; no security semantics change)
- [x] #5 Reason-specific guidance covers shared-writable, sticky-shared HOME, ownership, and ambiguous data roots
- [x] #6 Refusals still fail closed; no permissions are changed automatically
- [x] #7 Targeted tests cover the formatter, enriched errors, and the CLI exit path; existing private-path tests still pass
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR check: no new ADR — diagnostics-only change implementing the existing ADR-029/ADR-127 fail-closed posture; both linked in Implementation Notes.

1. Enrich PrivatePathResult/PrivatePathError in Utils/private_paths.py with best-effort offender path + mode/owner at the walk refusal sites (shared_writable_parent, sticky variants, untrusted/wrong owner; ambiguous roots keeps its existing result).
2. Add Utils/startup_errors.py (leaf, stdlib + private_paths only): reason-mapped plain-language formatter + print-once emitter, mirroring startup_logging.py conventions.
3. Wrap the import-time load_cli_config_and_ensure_existence() at config.py:9608: on PrivatePathError emit the diagnostic, then re-raise (covers python -m and tldw-serve).
4. In cli.py catch PrivatePathError around the heavy import + run: ensure the diagnostic was emitted, exit 1 without a traceback (packaged tldw-cli path).
5. TDD: RED tests for formatter mapping, enriched str(exc), cli SystemExit(1) via stubbed app module, and an isolated-subprocess diagnostic test modeled on test_config_import_hygiene._run_isolated_python.
6. Targeted test runs: Tests/Utils/test_private_paths.py, new startup-errors/cli tests, Tests/test_database_path_privacy.py, Tests/DB/test_private_sqlite.py (message consumers), plus a live editable-install subprocess verification.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**Summary.** A fresh install whose `~/.config` (or any ancestor of the config/data
directories) is group/world-writable now explains itself in plain language instead of
dying with a bare `unsafe_parent: shared_writable_parent` traceback. The refusal is
unchanged and still fails closed; only the reporting changed.

**Approach.**
- `Utils/private_paths.py`: `PrivatePathResult` gains diagnostics-only
  `offender_path`/`offender_detail` fields (defaults `None`, frozen dataclass stays
  backward compatible); the three walks (`_open_verified_parent`,
  `secure_private_directory`, `verify_trusted_directory`) track the traversed
  components (reset on an absolute trusted-symlink splice) and attach the offending
  directory plus a `stat.filemode`+uid/gid description (best-effort pwd/grp names) at
  each mode/owner refusal. `PrivatePathError.__str__` appends `[path (detail)]`, so
  every existing surface that stringifies the error (TTS/DB diagnostics, logs) gets
  the offender for free. No trust decision reads the new fields.
- `Utils/startup_errors.py` (new leaf, mirrors `startup_logging.py` import rules):
  `format_private_path_error()` renders a 72-column diagnostic — why the app refuses
  (secrets), the location, the blocked directory with mode/owner, the reason, and a
  reason-mapped HOW TO FIX block (`chmod g-w,o-w <dir>` for shared-writable; real
  HOME guidance for sticky-shared; ownership guidance for sudo/foreign-owner;
  explicit `[paths] data_dir` for ambiguous roots; generic `namei -l` otherwise) —
  plus the `TLDW_CONFIG_PATH` escape hatch and the technical detail line.
  `emit_private_path_startup_error()` prints once per process.
- `config.py` (import-time bootstrap, was line 9608): emits the diagnostic on
  `PrivatePathError`, then re-raises — so `python -m tldw_chatbook.app` and
  `tldw-serve` explain themselves and still show a traceback for bug reports.
- `cli.py`: catches `PrivatePathError` around the heavy import and the runner call,
  ensures the diagnostic was emitted (no-op when config already reported it), and
  exits 1 with no traceback — the packaged `tldw-cli` shows the diagnostic alone.

**Files.** `tldw_chatbook/Utils/private_paths.py`,
`tldw_chatbook/Utils/startup_errors.py` (new), `tldw_chatbook/config.py`,
`tldw_chatbook/cli.py`, `Tests/Utils/test_private_paths.py`,
`Tests/Utils/test_startup_errors.py` (new), `Tests/test_cli_startup_private_path.py`
(new), `backlog/docs/lessons-testing-evidence.md` (uv-venv/pip trap).

**Verification.** TDD throughout: RED confirmed before each implementation step.
Targeted suites green: test_private_paths (74), test_startup_errors +
test_cli_startup_private_path (12), test_database_path_privacy,
test_data_root_concurrency (ADR-127 fallback), test_prompt_dump_storage,
test_config_import_hygiene, TTS profile-sqlite/kokoro (173 combined),
DB/test_private_sqlite (399 combined incl. shared suites), Packaging
test_installed_distribution (164 pass; 16 initial failures were the throwaway
uv venv lacking the `pip` module — all green after installing pip/build/setuptools;
one `ScreenStackError` flake passed on re-run with and without this change).
Live verification from the editable install: bad-HOME scenarios print the
diagnostic and exit 1 with no traceback; the sticky-HOME variant prints the
relocation guidance; a clean HOME still boots; and following the printed
`chmod g-w,o-w` command makes the very next start succeed. mypy: one pre-existing
error in `private_paths.py` (reproduced with the change stashed), none introduced.

**Decisions/trade-offs.** No config-directory fallback was added (unlike the data
root's ADR-127 fallback): relocating where keys live silently is a bigger behavioral
change than this UX task, and ADR-127's alternatives explicitly rejected auto-chmod
of user-owned sharing policy. ADRs: none required — diagnostics-only; implements the
existing posture of [ADR-029](../decisions/029-local-private-data-boundary.md) and
[ADR-127](../decisions/127-fresh-install-private-data-root-recovery.md).
<!-- SECTION:NOTES:END -->

## Review Remediation (PR #2793, Qodo)

All nine Qodo findings addressed in the PR revision:

1. (High) The `TLDW_CONFIG_PATH=$HOME/...` alternative is now reason-aware: sticky-shared-HOME refusals get a `HOME=/home/you` relocation hint instead (relocating the config under the same shared HOME would refuse again), and ambiguous-data-root refusals get no config alternative at all.
2. Google-style docstrings added to `PrivatePathResult` and `PrivatePathError`.
3. `format_private_path_error`/`emit_private_path_startup_error` docstrings gained Args/Returns.
4. `cli.main_cli_runner` docstring gained a `Raises: SystemExit` section.
5. All pasted repair commands shell-quote paths (`shlex.quote`) with `--` end-of-options markers, and the `TLDW_CONFIG_PATH` example quotes `"$HOME"` — verified live with a hostile path containing spaces and `;`.
6. The default alternative now says config relocation does not fix data-side refusals and points at `[paths] data_dir`.
7. Ownership guidance appends a `chmod g-w,o-w` step when the captured `offender_mode` has group/world write bits (new diagnostics-only `offender_mode` field on `PrivatePathResult`).
8. Symlink refusals now carry the offender: `_read_trusted_symlink` and the hop-limit raise attach the component path (plus lstat detail for rejected links).
9. The blanket "No existing files or permissions were changed" claim was narrowed to "Nothing outside Chatbook's own config/data directories was created or changed" (hardening an app-owned directory before a postcondition failure can still occur).
