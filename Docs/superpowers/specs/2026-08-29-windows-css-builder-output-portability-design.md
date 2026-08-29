# Windows CSS Builder Output Portability Design

**Date:** 2026-08-29
**Status:** Approved for implementation planning

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
decorative glyphs with compact words or punctuation such as `Processing:`,
`CSS build complete:`, and `Total size:`. ASCII is valid under UTF-8, CP1252,
and other supported locale encodings and avoids changing global stream policy.

The change covers the whole script rather than only the first observed
checkmark, so a later completion glyph cannot reproduce the same failure after
the module loop succeeds.

## Failure Handling

Only output decoration changes. The builder continues to raise for substantive
input, race, and write failures. Existing guarantees that invalid input does
not bless or silently replace generated output remain in force.

## Verification

Add a cross-platform test that runs the complete builder entry path against a
scratch CSS/package tree while standard output is a strict CP1252 text stream.
The test must reach successful completion and validate the expected generated
files rather than merely proving that one progress string is encodable.

Retain existing missing-module, manifest-race, output-preservation, and bundle
integrity tests. Run only the focused CSS build/staleness suites plus scoped
lint and compilation checks.

## Delivery and ADR Check

This is one atomic portability task and one PR-sized change.

ADR required: no
ADR path: N/A
Reason: this is a portable diagnostic-output correction with no runtime,
dependency, tooling, or generated-artifact contract change.
