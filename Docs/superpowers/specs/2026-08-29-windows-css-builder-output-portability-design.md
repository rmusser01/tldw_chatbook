# Windows CSS Builder Output Portability Design

**Date:** 2026-08-29
**Status:** Revised after adversarial review; awaiting approval

## Problem

`tldw_chatbook/css/build_css.py` prints checkmarks, emoji, and other Unicode
glyphs as progress decoration. On Windows processes using a strict CP1252
standard-output stream, the first checkmark raises `UnicodeEncodeError` before
the modular CSS bundle is written. Startup logs the build failure and may then
continue with stale checked-in generated stylesheets.

## Goals

- The complete CSS builder runs successfully when standard output uses strict
  CP1252 encoding.
- Progress and completion output remains readable on every supported platform.
- Missing modules, source/build races, and filesystem failures retain their
  current fail-loud behavior.
- Generated CSS contents and manifest semantics remain unchanged.

## Non-goals

- Forcing a process-wide standard-output encoding.
- Mutating `PYTHONIOENCODING`, Windows code pages, or parent-process
  environment.
- Adding a console-output abstraction or dependency.
- Changing CSS generation, ordering, hashing, or staleness policy.

## Design

Use ASCII-only text for every direct `print` emitted by `build_css.py`. Replace
decorative glyphs with compact words or punctuation such as `Processing CSS
module 3 of 75`, `CSS build complete`, and `Total size:`. Printed dynamic data
is limited to numeric counts. Filesystem paths and source/module names are not
interpolated because a valid checkout may itself contain characters that the
active console encoding cannot represent. ASCII literals and numeric text are
valid under UTF-8, CP1252, and other supported locale encodings without changing
global stream policy.

The change covers the whole script rather than only the first observed
checkmark, so a later completion glyph cannot reproduce the same failure after
the module loop succeeds.

## Failure Handling

Only output decoration changes. The builder continues to raise for substantive
input, race, and write failures. Existing guarantees that invalid input does
not bless or silently replace generated output remain in force.

## Verification

Adapt the existing end-to-end manifest/staleness builder test instead of adding
a parallel integration harness. Remove its `builtins.print` no-op monkeypatch
and run the complete builder entry path against a scratch CSS/package tree
whose path contains a character not representable in CP1252. Redirect standard
output through an encoding-enforcing `io.TextIOWrapper` configured with
`encoding="cp1252"` and `errors="strict"`, flush that wrapper, and inspect its
captured bytes.

The test must assert ASCII progress markers from all four builder phases:
module progress, bundle completion, widget-default completion, and screen-CSS
completion. That prevents a vacuous pass caused by suppressing or deleting all
progress output. It must also retain the current manifest and staleness
assertions, validate all expected generated stylesheets, and assert that a
distinctive rule from the scratch CSS input appears in the generated bundle.
This proves that the actual build ran through the strict stream rather than
merely creating empty output files.

Retain existing missing-module, manifest-race, output-preservation, and bundle
integrity tests. Run only the focused CSS build/staleness suites plus scoped
lint and compilation checks.

## Delivery and ADR Check

This is one atomic portability task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: this is a portable diagnostic-output correction with no runtime,
dependency, tooling, or generated-artifact contract change.
