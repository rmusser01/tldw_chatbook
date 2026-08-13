# TASK-15742 — Windows-safe collection for POSIX-only tests

## Purpose

Restore repository-wide pytest collection on Windows without changing media
playback or TTS materialization production behavior. The two failures are test
collection defects: the media suite evaluates POSIX-only signal constants while
building parameters, and the TTS suite imports the POSIX-only `fcntl` module at
module import time.

## Decision

Use narrow test-side capability gates.

- Keep the media playback module cross-platform. Mark only the tests that
  assert real `SIGSTOP`/`SIGCONT` delivery as requiring both constants, and
  parameterize portable signal names rather than evaluating absent constants
  during collection. Resolve the expected signal only inside a supported test.
  Add a platform-neutral regression for the documented no-signal behavior by
  replacing the player's imported signal seam with a private stub; do not
  delete attributes from Python's process-global `signal` module. The test must
  prove pause/resume clock state still updates while `os.kill` is never called.
- Treat the profile-reference materialization module as a POSIX contract suite.
  Skip the module with an explicit reason when `fcntl` is unavailable. The
  existing simulated non-POSIX test continues to exercise the product's
  explicit `unsupported` outcome on POSIX CI.
- Make no production, dependency, schema, configuration, or runtime changes.

## Alternatives considered

1. **Recommended: capability-gate the tests.** Smallest truthful change;
   preserves useful Windows media coverage and acknowledges that the TTS
   implementation is intentionally POSIX-only.
2. Mark every TTS test individually. This is noisy, easier to apply
   inconsistently, and provides no additional Windows coverage because the
   production materializer rejects the platform before those POSIX contracts.
3. Add Windows implementations for process suspension and materialization file
   locks. This changes production architecture and belongs in a separate
   feature, not a collection-portability fix.

## Verification

- Retain the current Windows collection errors as the genuine RED evidence:
  media parameter construction raises `AttributeError`, and TTS import raises
  `ModuleNotFoundError`. The no-signal media regression is additional
  characterization coverage, not a fabricated product RED.
- Verify both modules with `pytest --collect-only` on Windows after the gates.
- Run both focused modules; expect media behavior tests to pass and the TTS
  POSIX module to report one clear module skip on Windows.
- Run static checks and `git diff --check`.
- Re-run repository-wide `pytest --collect-only` to prove global collection
  advances beyond the former `SIGSTOP` and `fcntl` errors, then start the full
  suite with the repository-local temp root. Any later unrelated failures are
  reported separately rather than attributed to this task.

## ADR disposition

ADR required: no

ADR path: N/A

Reason: this is a test-only portability correction that preserves existing
runtime boundaries and the existing POSIX-only materialization contract.
