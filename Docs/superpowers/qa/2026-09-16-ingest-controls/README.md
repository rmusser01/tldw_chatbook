# Import option editing — TASK-32666

Reviewed on `feat/component-pattern-library`, based on `b1303615d6`.

Checkbox and selector changes previously replaced the form, moving an unrelated
title selection from 2–7 to the end of its text. Options now refresh their
existing controls, dependencies, validation and receipts. Reset applies defaults
to its own group; other editors keep their text, selection and widget identity.
Retry confirmation wording also returns to its unarmed state after an edit.

The retained update path needed explicit event admission: programmatic snapshots
must not emit new user edits, obsolete native and forwarded messages must not
undo Reset, and an older accepted option must not overwrite a newer sibling
editor awaiting delivery. Pending backend replacements consume the newest state
without querying the outgoing backend's incompatible fields. Deferred focus
reveal checks attachment/current focus and scrolls immediately after layout.

Long checkbox explanations wrap using measured content height. The Parakeet
install explanation also wraps, including inside the real compact Library shell
whose toolbar rule otherwise capped it at one row. Existing tokens and component
patterns are used; no token values or import/provider contracts changed.

ADR required: no. Existing
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
and [161](../../../../backlog/decisions/161-component-pattern-library.md) apply.

## Automated evidence

[Verification commands and log hashes](verification.json) record **355 passing
targeted checks** across the final focused option/event-order, existing canvas/consent/keyboard/directory/retry,
template/cache, token/bundle/component-governance and boot-CSS checks. No full
suite was run. The two new test files include both-theme compact/wide editor
journeys, all seven type groups, actual painted explanations, reset scope,
dependency transitions, consent labels, delayed events and backend replacement.

Earlier failing runs are preserved. They caught context loss, clipped labels,
programmatic event feedback, stale queued changes, pending-backend lookups and
a later sibling edit overwritten by an earlier dependency refresh. The compact
Browse neighbor caught a second deferred scroll moving the newly focused field
below the viewport. Native inspection then exposed the install explanation's
shell-specific height cap; two real-shell tests failed before its correction.

Three old tests were updated at their obsolete whole-form-refresh seam while
retaining consent, external-worker cancellation and scoped-update assertions.
The Retry footer test failed with the base methods as well: direct route entry
still held its entry registration. It now settles the first registry projection
before measuring identical subsequent ticks. A mutable parameter shared between
themes was deep-copied, and focus is awaited before assigning a test selection.

[Static comparison](static-comparison.json) records zero new Ruff diagnostics.
New Python passes full Ruff and formatting; changed existing ranges are
[formatted](format-ranges.json). [Sizes](size-comparison.json) record boot CSS at
615,643/634,050 bytes and the two inherited architecture ceiling failures: LibraryScreen and
the unchanged ingest controller already exceed their limits at base. The screen
shrinks by four lines and gains one helper method; no budget was increased.
The final [independent review](review.md) has no actionable findings.

## Native inspection

[native_check.py](native_check.py) runs actual TldwCli/LinuxDriver with an
[exclusive private profile](isolation.json), six synthetic local files and real
classification/preflight. It changes PDF engine/OCR/language, generic analysis,
chunking/encoding and audio provider controls, resets PDF, checks unrelated
metadata/prompt selections, and reaches the optional GGUF action by keyboard.
It never activates installation, a model, extraction or an import submission.

The environment lacks PDF processing, audio processing and Parakeet extras.
The real unavailable copy is captured first; only widget availability is then
simulated to qualify enabled controls. Web controls are covered by render-only
automated cases, without fetching a URL. Normal Input re-entry selects all by
Textual policy; the retained off-focus selection is checked before that action.

Eight run-002 captures were rendered and inspected together. The clipped install
explanation prompted the final correction; the two run-004 install captures were
inspected as confirmation. All linked captures and results are from run-004:

| State | 170×48 dark | 80×24 light |
|---|---|---|
| Missing-package explanation | [capture](availability-170.svg) | [capture](availability-80.svg) |
| Full install explanation | [capture](install-reason-170.svg) | [capture](install-reason-80.svg) |
| Chunk off, invalid draft retained | [capture](chunk-gate-170.svg) | [capture](chunk-gate-80.svg) |
| PDF reset, other draft retained | [capture](pdf-reset-170.svg) | [capture](pdf-reset-80.svg) |
| Optional GGUF action focus | [capture](gguf-control-170.svg) | [capture](gguf-control-80.svg) |

[Results](result.json) record both viewport journeys.
[Read-only persistence](persistence.json), using the project interpreter,
confirms ten healthy SQLite databases, zero media/messages/ingest jobs, unchanged
bytes in all six source fixtures and no ERROR/CRITICAL app log lines. Final Quit
returned from `app.run` with exit 0 before the owned shell was closed. The first
queued Quit left Hub displayed; another normal terminal Quit completed shutdown.

Run-001's compact check scrolled a nonfocused label before the pending focus
reveal settled; the runner now waits before that separate inspection scroll.
Run-002 passed UI checks but its session was closed before its exit receipt,
so it is not shutdown evidence. The process was confirmed gone; run-003 then
recorded normal exit and healthy persistence, and run-004 repeated after the
final CSS repair. These evidence limits are retained in the verification record.

## Remaining review

Continue queue activity and recovery. An attempted fully expanded-form Retry
focus check painted the docked “more” hint over the focused Retry row. This slice
verifies Retry consent wording via its actual button event, not queue keyboard
traversal; the overlap remains for that next review. Actual import execution,
installation, remote/provider authorities and restart remain unqualified.
No push or integration into `dev` was performed.
