---
id: TASK-14824
title: >-
  Ingest accessibility residue: Select focus, unreachable disabled reasons,
  placeholder contrast
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 21:00'
updated_date: '2026-08-10 21:42'
labels:
  - library
  - ingest
  - accessibility
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
P2 of the 2026-08-10 re-critique — three gaps the structural-focus and disabled-reason work missed, each measured live.

1. **`#opt-generic-encoding` focus is colour-only.** A per-focusable Tab walk found the Select's focused and unfocused plain-text captures byte-identical; only the background changes, at 1.12:1 between the two. Every other canvas focusable shows a glyph-level change (`┏━┓` on inputs, `┃label┃` on buttons, a `┃` marker on collapsible titles). The 13 global nav tabs are colour-only too, but those are out of this surface's scope.

2. **Disabled fields are keyboard-unreachable, hiding the reasons written for them.** The Audio & video group contributes exactly 2 tab stops (its collapsible title and Reset to defaults) because all 13 option fields and the Parakeet install button are disabled. A keyboard-only user can therefore never land on any of them to read the `— needs X installed` annotation that task-3304 added specifically so inert controls explain themselves. The explanation is currently mouse-and-eyes-only.

3. **Input placeholders measure ~3.5:1 in both states.** Enabled placeholder 3.52:1, disabled placeholder 3.49:1 — below AA for normal text, and a 0.03 delta means a placeholder-only field has effectively no colour cue for its disabled state. Related: the path Input has NO visible label at all — its identity is placeholder-only and vanishes once populated, the same defect task-2012 fixed for option fields, still present on this surface's primary control.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every focusable on the ingest canvas, including Selects, shows a glyph-level focus change asserted by a plain-text render diff
- [x] #2 A keyboard-only user can reach the explanation for a disabled option without a mouse (either the control stays focusable-but-inert, or its reason is surfaced somewhere keyboard-reachable)
- [x] #3 The path field carries a persistent visible label rather than a placeholder that disappears on input
- [x] #4 Placeholder text meets the contrast floor, or placeholders are not the sole carrier of a field's identity or state
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Reproduce (1) with a COMPOSITED capture, not ``Widget.render_lines``: the existing focus tests read the widget in isolation, which is why a rule painted over by a child passed them.
2. Root-cause and fix the Select focus in the app-tier CSS sources; mutation-check the rule; add a sweep over EVERY canvas focusable so the next covered-by-a-child control cannot ship.
3. (3) Give the path Input a persistent label ``Static`` above it, dense-form convention.
4. (4) Measure the placeholder contrast in both states from the compositor, add a token that clears AA against BOTH the resting and the darkened disabled surface, and assert it.
5. (2) Establish what Textual actually supports for a focusable-but-inert control, then pick between that and surfacing the group's reason on a control that IS a tab stop.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
**(1) The Select focus rule had been dead since task-2014.**
``LibraryIngestCanvas Select:focus { outline: heavy $accent }`` was
declared and never reached a cell: ``Select`` composes an opaque
``SelectCurrent`` whose border+padding make it exactly the parent's size,
and the compositor paints a child over its parent -- the same
"set but painted over by a child" family as the ``Select:disabled
SelectCurrent`` pair two rules below it. The structural cue moved to the
child: ``LibraryIngestCanvas Select:focus > SelectCurrent { border: heavy
$ds-input-focus-accent }`` swaps ``SelectCurrent``'s own ``tall`` border
for ``heavy`` -- a glyph change (``▊▔▎▁`` -> ``┏━┓┃┗━┛``) at identical
thickness, so no row is consumed (the task-3302 trap) and the value stays
readable. RED evidence: composited captures byte-identical before the fix.
Mutation: replacing the ``border`` with a ``background`` reproduces the
byte-identical capture exactly.

**The measurement lesson (generalises).** ``Widget.render_lines`` renders
a widget WITHOUT its children, so it showed the outline happily while the
screen showed nothing -- a focus test written that way is vacuous for any
widget with an opaque child. The new
``test_every_canvas_focusable_changes_at_the_glyph_level_on_focus`` sweeps
every focusable the canvas offers using compositor strips; removing the
new rule makes it name ``Select#opt-generic-encoding`` specifically.

**(2) Textual cannot make a disabled control focusable, so the reason
moved to a control that IS a tab stop.** ``Widget.focusable`` is
``can_focus and visible and not self._self_or_ancestors_disabled`` -- no
flag or CSS reaches that, and ``Input`` in Textual 8.2 has no
``read_only``. A focusable-but-inert presentation would mean leaving the
13 fields ENABLED and intercepting every key/Changed event to revert the
value: that re-opens the task-673 recompose storm (a reverted value posts
another ``Changed``), loses the app-tier ``:disabled`` legibility styling,
and lies about affordance. The tradeoff taken instead: the group's
``CollapsibleTitle`` -- a real tab stop -- carries the blocked count and
reason (``Audio & video — 13 options unavailable — needs faster-whisper
installed``), with packaging gates preferred over within-form ones
because those are the ones a user must act on outside the app. The test
asserts BOTH halves: the title is focusable and carries the reason, AND
the fields themselves really are unfocusable, which is what makes the
title the only honest place for it.

**(3)** ``#library-ingest-path-label`` ("File, folder or URL to import"),
composed above the Input -- the task-2012 fix finally applied to the
canvas's primary control. Asserted with the field POPULATED, which is when
the placeholder is gone.

**(4)** Textual's stock ``$text-disabled`` placeholder measured 3.52:1
enabled / 3.49:1 disabled (reproduced exactly). New token
``$ds-text-placeholder: #8a8a8a`` clears AA against both the resting field
surface (rgb 30) and the darkened disabled surface (rgb 13) -- 4.94:1 /
5.63:1 -- while staying clearly dimmer than a real value. Applied via the
``input--placeholder`` component class, which is the only selector that
reaches placeholder paint. Identity is additionally no longer
placeholder-borne anywhere on the canvas after (3).

Files: ``css/components/_agentic_terminal.tcss``, ``css/core/_variables.tcss``
(+ rebuilt bundle), ``Widgets/Library/library_ingest_canvas.py``,
``Tests/UI/test_library_ingest_structural.py``,
``Docs/User_Guide/library/import-and-export.md``.

**xhigh review round: (2) labelled healthy panels broken.** The blocked
clause this task put on the collapsible title counted EVERY disabled
field, so an ordinary closed within-form gate -- a field greyed because a
sibling toggle is off -- read as a broken panel. Measured on the shipped
schemas with every package installed: ``Web pages — 2 options unavailable
— single-page fetch selected`` (the DEFAULT panel), ``PDF documents — 3
options unavailable — needs Enable OCR on``, ``Audio & video — 3 options
unavailable — needs the parakeet-onnx provider``. Three fully working
panels leading their receipt with a failure.

The clause was for PACKAGING gates -- work the user must do outside the
app, and the reasons a keyboard user cannot reach at all because Textual
drops a disabled widget from the tab order. ``build_type_group_title``
now counts only those (``_is_packaging_gate``: a ``depends_on`` feature
that is not installed, the same first branch ``field_disabled_state``
evaluates and in the same order). Disabled fields still contribute no
value pairs, whatever gate closed them (task-14825 #7 is unchanged);
what changes is only what counts as "unavailable". The gated cases are
untouched: nothing installed still reads ``PDF documents — 3 options
unavailable — needs PDF processing installed`` (was 4: ``ocr_backend`` is
sibling-gated, not packaged) and ``Audio & video — 13 options unavailable
— needs Audio processing installed``. ``_preferred_blocked_reason`` is
gone with it: the preference it applied over a mixed list is now
structural, because the list only ever holds packaging reasons.

Pinned by ``test_a_healthy_panel_does_not_lead_with_options_unavailable``
and ``test_a_packaging_gate_still_leads_the_panel_receipt``
(``Tests/UI/test_library_ingest_canvas.py``); forcing
``_is_packaging_gate`` true fails both plus
``test_option_panel_title_reads_as_plain_language``. AC#2 still holds --
a genuinely blocked group still states its reason on a tab stop.
<!-- SECTION:NOTES:END -->
