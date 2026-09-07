---
id: TASK-3304
title: >-
  Ingest structural fixes: disabled-state legibility, receipt into view, picker table, clipped install command
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-07/08/15/17 of the 2026-08-07 Media Ingestion review. (1) All `enabled_when_values`-disabled option fields (Parakeet model folder, transcription model under `default`, web max pages/depth under `individual`) and the Parakeet install button render identically to enabled controls with no stated reason — violating the dense-form Inert-actions rule and sitting on the documented all-themes-below-3:1 disabled-contrast trap. (2) After Start, the outcome rows land below the fold on every submit and the `VerticalScroll` canvas has no fold indicator (task-1623 convention). (3) The file picker table has no column headers, raw unitless sizes, a `..` row showing "512" as an apparent size, and an unlabeled filename input. (4) The missing-dependency warning line clips the pip command at the canvas edge, and the only copy button lives in the guardrail modal.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Schema-disabled fields and the install button read as visibly inert with the reason at the control (e.g. "— needs Parakeet provider"), at ≥3:1 label contrast in a running terminal
- [x] #2 After Start, the queue's outcome area is brought into view (or a pinned live status adjacent to Start shows run state until terminal); a fold indicator appears while canvas content overflows
- [x] #3 Picker shows column headers, humanized sizes, no size on directory rows, and a labeled filename input
- [x] #4 The full install command is readable (wrapped or truncated-with-copy affordance outside the modal)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-verify each on the worktree; RED tests where assertable (reason-annotation presence, post-start scroll target, picker header row).
2. Disabled: reason suffix from the schema's enabled_when metadata; app-tier disabled styling (Legible Disabled rule — app stylesheet, not DEFAULT_CSS).
3. Receipt: scroll-to-queue on submit + overflow fold row (display-managed, owned by the in-place updater — never conditionally composed).
4. Picker + warning-line copy affordance.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD: 15 new tests written first; 13 went RED on the exact live defects
("no reason at the model-folder control", `opacity=0.7` on the disabled
field, `#library-ingest-fold-hint` never mounted, "queue heading (virtual
y=36) still out of view after submit (scroll y=0, viewport=21)",
`#ingest-preflight-copy-command-0` never mounted, "raw unitless size:
'… 512 2026-08-07 …'", header/label NoMatches) and 2 deliberately pinned
already-correct behavior (no double-annotation on gate-hint checkboxes;
the warning line already WRAPS the command at width 80 — see MI-17 note).
Four mutations run, each RED then restored via Edit: CSS disabled rule
removed → paint test RED at opacity=0.7; scroll-on-submit call removed →
receipt test RED; fold overflow gate hardwired True → no-overflow leg
RED; curated reason swallowed in the helper → 2 schema + 2 canvas tests
RED (proving the canvas routes through the helper).

**AC#1 (MI-07), two parts.** (a) Reason at the control: new
`OptionField.disabled_reason` (curated copy: "needs the parakeet-onnx
provider", "needs the faster-whisper provider", "needs the docext
engine", "single-page fetch selected", "needs Chunk content on", "needs
Enable OCR on") + `field_disabled_state(field, cap, values,
is_installed=)` in `ingest_capabilities.py` — the ONE computation for
both the disabled flag and the annotation, replacing the canvas's inline
gate logic (the canvas passes its module-global `_is_installed` so the
12+ incumbent tests patching `library_ingest_canvas._is_installed` keep
working). Rules: depends_on → "needs <feature> installed" (new
`diarization` label — the group-extra fallback said "Audio ingestion and
transcription", which is not the gate); gate closed → curated reason,
else generic derivation, SUPPRESSED when the field's static hint already
names the gate (3303's "docling or docext engines only" labels are never
double-annotated — pinned). A meta-test forces every future value-gated,
hint-less field to carry curated copy. The Parakeet install button's
label ends "— needs the parakeet-onnx provider" while gated. Labels are
env-dependent now (depends_on annotations), so
`test_select_fields_carry_visible_labels` accepts `label` or
`label — <reason>` pinned to the exact separator.
(b) Legible Disabled paint: app-tier `LibraryIngestCanvas
Input:disabled / Select:disabled / Checkbox:disabled` (+
`SelectCurrent` and its inner `Static#label` — SelectCurrent paints its
own opaque $surface, the "set but painted over by a child" family) in
`css/components/_agentic_terminal.tcss`, same recipe as the incumbent
task-2130 Button rule: opacity/text-opacity 100% (kills Textual's
app-default `*:disabled:can-focus { opacity: 0.7 }` fade), `color:
$text-muted`, `background: $surface-darken-1`. Measured via the
compositor's own strips under a CSS-true canvas host: disabled Input
value **3.31:1**, disabled Select value **3.31:1** (before: the 0.7 fade
only — no floor, no state colour), enabled sibling stays distinct ink
(asserted). Bundle rebuilt via build_css.py; sources only.

**AC#2 (MI-08).** `_do_submit_ingest` now `call_after_refresh`-scrolls
`#library-ingest-queue-heading` to the viewport top
(`_scroll_library_ingest_queue_into_view`; the submit already recomposes,
so the scroll rides the same settle). Fold indicator per the task-1623
convention: `#library-ingest-fold-hint` ("▼ more — scroll for the rest")
is a `dock: bottom` canvas child — docked children of a scroll container
never scroll, so it is pinned chrome (pinned by a scroll-end region test)
— always mounted, display-managed by `LibraryIngestCanvas.sync_fold_hint`
(virtual vs container height, mirroring Settings'
`_update_inspector_overflow_hint`), synced on canvas mount/resize and
re-derived by the screen's gate updater after in-place height changes
(`_update_library_ingest_fold_hint`, reached from
`_update_library_ingest_dynamic_regions` via the gate call). Identity
across the hot path asserted with `is`, not a re-query.

**AC#3 (MI-15) — ours vs library.** The picker is NOT an installed
third-party package: `textual_fspicker` is VENDORED under
`tldw_chatbook/Third_Party/` and already carries repo commits (TASK-378,
task-1479, task-2160, task-2222), so the row rendering was reachable
without forking anything. Fixed in the vendored copy:
`DirectoryEntry._size` humanizes (512 B / 2.4 MB… — the upstream
`# TODO: format well for a file browser` comment) and returns "" for
directories (kills the `..` row's fake "512"); `base_dialog` composes a
`#file-dialog-column-headers` Static (Name / Size / Modified) built from
the exact `_as_renderable` grid recipe (pad 1/icon 3/name 1fr/size
10/time 20/pad 1) with `padding: 0 1` matching the listing's
`border: blank` inset; `BaseFileDialog._input_bar` gains a "File name:"
Label + placeholder (before the Input, so the task-1479 scoped
first-Input lookups still resolve). NOT reachable without deeper
surgery, accepted and documented in-code: when the list overflows, the
OptionList's vertical scrollbar shifts the data columns left of the
header by its width; on Windows the DriveNavigation pane offsets the
listing; SelectDirectory shares the base compose so it shows the same
headers (Size column simply stays empty there).

**AC#4 (MI-17).** Investigated first: at canvas widths the RED harness
can produce, the warning Static WRAPS the command (pinned by a paint
test at width 80) — the live clip could not be reproduced from this
code, so the "recoverable at the warning" guarantee is the copy
affordance: new `LibraryIngestCanvasState.warning_commands` (deduped,
order-preserving, via shared `preflight_install_commands()` — several
features often resolve to one extra) renders one compact "Copy install
command" button per distinct command directly under the "⚠" lines
(`#ingest-preflight-copy-command-{i}`, disambiguated by extra name when
plural), handled ON the canvas with the guardrail modal's exact
copy/notify seam. The modal's button is unchanged.

**Test-helper repairs in passing** (same stale-`__new__`-helper family
3300/3303 repaired): `_do_submit_ingest` now schedules a callback, so
the three unmounted-screen helpers
(`test_library_ingest_guardrail_modal._minimal_library_screen`,
`test_library_screen._minimal_ingest_screen`, both flow-test screens)
seed `call_after_refresh`.

Files: `Library/ingest_capabilities.py`, `Library/library_ingest_state.py`,
`Widgets/Library/library_ingest_canvas.py`, `UI/Screens/library_screen.py`,
`css/components/_agentic_terminal.tcss` (+ rebuilt bundle),
`Third_Party/textual_fspicker/{base_dialog,file_dialog,parts/directory_navigation}.py`,
`Docs/User_Guide/library/import-and-export.md` (+ task-3304 stamp).
Tests: new `Tests/UI/test_library_ingest_structural.py` (11) and
`Tests/UI/test_fspicker_listing_columns.py` (4); extended
`test_ingest_capabilities.py` (+6), `test_library_ingest_state.py` (+2);
label test loosened as above.

Final battery (six prior-phase suites + new + picker suites): **491
passed, 1 failed**. Coordinator correction: that failure was NOT
pre-existing — `test_installed_feature_produces_no_tooling_warning` was
one of this task's own six new tests, and it patched only
`OPTIONAL_FEATURES` while the probe's curated
`_FEATURE_REQUIRED_PACKAGES["pdf_processing"]` branch short-circuits
first, leaving the outcome hostage to whether pymupdf is installed.
Repaired to patch `_FEATURE_REQUIRED_PACKAGES` (the branch production
takes); capabilities suite 64 passed, full ingest battery 417 passed
after repair, fspicker suites 46 passed. `--collect-only` over Tests/:
31,401 collected, 27 errors, all in the numpy/audio/transcription/
confluence families this venv is known to lack — none in touched areas.
xhigh review + live-verify round (2026-08-09): two MI-07/MI-17 regressions found live and fixed.
(a) Disabled TEXT fields dropped their hint. The disabled branch rebuilt the label as
`f"{field.label} — {disabled_note}"`, discarding the `(hint)` the enabled branch adds -- so on a
stock install the cookies field's "Netscape cookies.txt · video URLs only" and the trim fields'
"HH:MM:SS or seconds" were invisible exactly while the control was inert. The checkbox branch had
always appended correctly; the text branch now does the same (`f"{label_text} — {disabled_note}"`).
(b) MI-17's per-command disambiguation never reached the screen. `_command_short_name` returned
`.[audio]`, and a Textual `Button` label is parsed as content markup, which ate `[audio]` as a style
tag -- live, six or seven stacked buttons all read "Copy install command (.)". It now returns the
bare extra name ("audio", "video", "transcription_faster_whisper"), which says the same thing and
survives the renderer. Files: tldw_chatbook/Widgets/Library/library_ingest_canvas.py; new tests in
Tests/UI/test_library_ingest_canvas.py (enabled+disabled hint pair; a distinct-labels assertion that
is RED with the bracketed spelling -- captured output: three identical "Copy install command (.)").
<!-- SECTION:NOTES:END -->
