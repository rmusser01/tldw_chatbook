# Task 3 report

## Summary

Implemented the platform-neutral persistent terminal contracts: ADR-099 resource
limits, lifecycle and reason vocabularies, immutable value objects, lifecycle
event projection, cleanup receipt timing, reservation retention, and the abstract
backend protocol. No subprocess, Windows, UI, or manager machinery was added.

## Files changed

- `tldw_chatbook/Terminal/__init__.py`
- `tldw_chatbook/Terminal/contracts.py`
- `tldw_chatbook/Terminal/backend.py`
- `Tests/Terminal/test_contracts.py`

## RED evidence

Command:

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
```

Result: exit code 1; 7 failed and 6 passed. The first failure was an assertion
failure, `assert 0 == 4`, in `test_terminal_limits_match_adr_099`; the remaining
failures exercised neutral transition, reservation, event, and Retry
placeholders. The test collected successfully, so this was not an import or
collection failure.

## GREEN/static verification

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
```

Result: exit code 0; `13 passed, 1 warning in 1.02s`. The warning is the
environment's existing `RequestsDependencyWarning` plus pytest temporary
cleanup warnings; no test failed.

```text
../../.venv/bin/python -m ruff check tldw_chatbook/Terminal Tests/Terminal/test_contracts.py
```

Result: exit code 0; `All checks passed!`

```text
git diff --check
```

Result: exit code 0; no output.

## Commit hash

`0ad25add15` (`feat: define persistent terminal contracts`)

## Self-review notes and remaining risks

- The diff is limited to the four owned implementation/test paths; no process
  handles, raw bytes, environment mappings, Windows imports, or mutable UI
  objects appear in projections.
- Lifecycle validation is pure and rejects forbidden transitions; cleanup Retry
  is the sole operation that creates a new cleanup T0.
- Remaining risk is intentional scope: concrete launch, parser, I/O actor,
  manager, and POSIX/Windows backend implementations are deferred to later
  tasks and are not represented here.

## Review-fix evidence

### Production running projection

The test-local `running_projection()` helper was removed and the test now
imports the production function. Before the fix, the focused test run reached
an assertion failure in `test_shell_exit_drains_and_nonzero_exit_is_ordinary`:
production returned `TerminalLifecycle.RESERVED` instead of
`TerminalLifecycle.DRAINING`. After changing the production helper to return a
running projection, the focused file passed with `13 passed, 1 warning`.

### Admission failure transition

Added `test_admission_failure_closes_and_releases_reservation`. Before the
production change, its focused run failed at the lifecycle assertion because
`apply_event()` left `ADMITTING` unchanged (`1 failed, 13 deselected`). The
minimal `admission_failure` branch now returns `CLOSED` with
`TerminalReason.ADMISSION_FAILED`; the focused test passed with
`1 passed, 13 deselected` and verifies the reservation is released.

### Cleanup schedule coverage and final verification

The limits test now pins `deadline_seconds`, `hangup_no_later_than`,
`terminate_no_later_than`, `force_kill_no_later_than`, and
`proof_reserve_seconds` to the brief's exact values.

Final focused/static checks:

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
14 passed, 1 warning in 1.33s

../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
All checks passed!

../../.venv/bin/python -m ruff format --check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
4 files already formatted

git diff --check
exit code 0; no output
```

The test warning is the environment's existing `RequestsDependencyWarning`,
along with pytest temporary-cleanup warnings; no test failed.

## Whole-branch review fixes

### Retry-only cleanup transition

Added adversarial `close` and `parser_failure` cases starting from
`cleanup_unproven`. The focused RED run reached both lifecycle assertions and
reported `2 failed, 14 deselected`: each ordinary event incorrectly produced
`closing`. After excluding the retained state from those generic event paths,
the same run reported `2 passed, 14 deselected`.

The explicit Retry contract was then tightened. Its RED run reported
`2 failed, 3 passed, 13 deselected`: `retry_cleanup()` returned only a receipt
instead of an atomic projection/receipt pair and accepted a running projection
without retained cleanup authority. The minimal implementation now accepts
only `cleanup_unproven` and atomically returns a `closing` projection plus a
fresh Retry receipt T0; the focused run reported
`5 passed, 13 deselected`. Joining or observing an existing receipt still
returns that receipt unchanged and retains its original T0.

### Parser invariant classification

Added literal no-reason and mismatched-reason parser failure cases. The RED run
reported `1 failed, 1 passed, 17 deselected`: a supplied `io_failed` reason
overrode the parser invariant category. `apply_event()` now always classifies
this event as `terminal_protocol_failed`; the focused GREEN run reported
`2 passed, 17 deselected`.

### Exact public-contract coverage and YAGNI cut

The tests now exhaust all lifecycle source/target pairs against the approved
14-pair transition set, pin all nine Task 3 dataclasses as frozen and slotted,
and compare all five `TerminalBackend` method signatures and annotations to
the brief. These contract-shape tests passed with
`11 passed, 15 deselected` against the existing value/backend definitions.

The earlier review-fix production-helper section is superseded: the test-only
`running_projection()` API was removed from `contracts.py` and package exports.
The focused tests now construct a running `TerminalProjection` locally.

Final scoped verification:

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q
26 passed, 1 warning

../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
All checks passed!

../../.venv/bin/python -m ruff format --check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
4 files already formatted

git diff --check
exit code 0; no output
```

## Event-source authorization review fix

Added an adversarial cross-product over `admission_failure` and
`cleanup_proven` against every lifecycle source. The test pins the authorized
source directly and requires the complete projection to remain unchanged for
every unauthorized source; the existing shell-exit, cleanup-failure,
close/parser-failure, and Retry tests remain in the focused file.

The assertion-level RED run was:

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q -k event_authorized_source --tb=short -p no:warnings
6 failed, 12 passed, 26 deselected in 1.27s
```

`admission_failure` incorrectly closed `reserved`, `creating`, and `closing`;
the `closing` case demonstrated reservation release without cleanup proof.
`cleanup_proven` incorrectly closed `reserved`, `creating`, and `admitting`
through the generic launch-failure edges. All failures reached
`assert projection == original`.

After replacing the generic CLOSED-transition checks with exact source guards
(`ADMITTING` for `admission_failure`, `CLOSING` for `cleanup_proven`), the
focused authorization run reported:

```text
18 passed, 26 deselected, 1 warning in 1.17s
```

Final scoped verification on the formatted tree:

```text
../../.venv/bin/python -B -m pytest Tests/Terminal/test_contracts.py -q --basetemp=/private/tmp/task22512-contracts.vY4mqH
44 passed, 1 warning in 1.13s

../../.venv/bin/python -m ruff check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
All checks passed!

../../.venv/bin/python -m ruff format --check tldw_chatbook/Terminal/__init__.py tldw_chatbook/Terminal/contracts.py tldw_chatbook/Terminal/backend.py Tests/Terminal/test_contracts.py
4 files already formatted
```

The warning remains the environment's existing `RequestsDependencyWarning`;
no focused test failed.
