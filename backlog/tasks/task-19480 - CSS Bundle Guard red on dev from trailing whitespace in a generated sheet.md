---
id: TASK-19480
title: >-
  CSS Bundle Guard red on dev from trailing whitespace in a generated sheet
status: Done
assignee:
  - '@claude'
created_date: '2026-08-21 16:30'
updated_date: '2026-08-21 16:30'
labels:
  - css
  - ci
priority: high
dependencies: []
---

## Description

The **CSS Bundle Guard** check has been failing on `dev` (and therefore on
every open PR) with:

> `widget_defaults_self.tcss is out of sync with the BUNDLED_CSS declarations
> in the Python sources.`

Root cause: `widget_css.py`'s sheet renderer emitted each lifted
`BUNDLED_CSS`/`BUNDLED_SCREEN_CSS` block verbatim. A block whose closing
`"""` is indented ends with a whitespace-only line (the quote's own
indentation), so the *generated* file contained trailing whitespace — which
any editor or linter with strip-on-save silently removes. Once the committed
sheet is stripped but the builder still emits the spaces, the guard fails for
everyone until someone regenerates. It went live with
`PersonaVisualCustomStateDialog` (commit `6d8a11fb2`, Persona Visual packs),
whose block contributed exactly one 4-space line.

Found while triaging CI on PR #1885 (unrelated batch): its failure list was
~110 UI tests across ~20 files plus this guard; the guard was the only one
that turned out to be a real, actionable dev red.

## Acceptance Criteria

- [x] `check_bundle_sync.py` reports all five generated sheets in sync on a clean checkout of `dev`
- [x] The generator no longer emits trailing whitespace at block boundaries, so a strip-on-save editor cannot desync a committed sheet again
- [x] The main `tldw_cli_modular.tcss` bundle is untouched by the fix (it is built by a different path; keeping it out avoids conflicts with in-flight branches)
- [x] CSS generation/contract suites stay green (`test_widget_css_consolidation`, `test_non_obscuring_focus_contract`, `test_compact_focus_outline_render`, `test_checkbox_height_render`)

## Implementation Notes

One-line mechanism fix in `tldw_chatbook/css/widget_css.py`: each block's text
is `rstrip()`-ed and given exactly one trailing newline before it is appended
to a stream, so generated output is canonical. Regenerating then dropped 283
whitespace-only/blank-separator lines across the four widget/screen sheets;
the main bundle changed only its generation timestamp. CSS semantics are
unaffected (trailing whitespace and inter-block blank lines are inert in
TCSS), and the four CSS suites (137 tests) stay green.

Deliberately NOT done: a repo-wide trailing-whitespace normalization of every
generated artifact. That would churn ~321 lines including
`tldw_cli_modular.tcss`, this repo's most conflict-prone generated file, while
many branches are in flight — the cost outweighs the marginal robustness, and
this fix already closes the mechanism that produced the incident.

**Evidence hygiene lesson (added to `lessons-testing-evidence.md`):** every
local `check_bundle_sync.py` run in the preceding CSS work piped through
`tail -1`, which shows only the LAST of its five per-file lines — the error
line sits in the middle, so a red guard would have read as green. Verifications
that emit one line per checked item must be read in full (or grepped for the
failure token), never tailed.
