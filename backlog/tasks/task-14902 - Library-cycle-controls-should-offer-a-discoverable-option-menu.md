---
id: TASK-14902
title: Library cycle controls should offer a discoverable option menu
status: Done
assignee:
  - '@claude'
created_date: '2026-08-10 17:20'
updated_date: '2026-08-11 15:26'
labels:
  - library
  - ux
  - recritique-2026-08-09
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Filed from task-4023 AC#5. The batch's bounded fix gave every Library value-cycler
its own glyph (`⇄` via `library_cycle_label`) and an option-enumerating tooltip, so
the option set is no longer invisible — but a tooltip is hover/focus-gated and the
control still only ADVANCES; a user cannot jump to a specific option (re-critique
heuristic #6, "cycle-buttons hide their option space"; persona note "cycle-buttons
can't be jumped"). The Notes canvas's Sort control already shows the discoverable
pattern: pressing it swaps in a one-row choice strip with a ✓ on the active option.
Converge the cyclers (media type, prompts sort/collection, skills sort + editor
toggles, export quality, Search/RAG mode) on that choice-strip pattern or a shared
popover, and retire the per-press cycle where it no longer earns its place.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every Library cycle control can show its full option set on screen (not only in a tooltip) and lets the user pick an option directly
- [x] #2 The active option carries a non-colour marker consistent with the Library marker vocabulary
- [x] #3 The footer/F1 advertise the interaction where it is keyboard-reachable
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Precedent read first: the Notes Sort chooser (`library_notes_canvas.py` — press
`#library-notes-sort` → browse-actions row hides, a one-row `ds-toolbar`
Horizontal of choice Buttons appears with `✓ ` on the active option; a pick
applies + closes; Escape closes + refocuses the opener; footer advertises
"enter choose sort / esc cancel"). The sync panel's direction/conflict groups
are a second conforming precedent (always-visible choice group, `✓` active,
`choice_value` attr).

Per-site disposition (10 cycler sites audited):

| Site | Options | Decision |
|---|---|---|
| Media `#library-media-type-filter` | dynamic (All + present types) | converge → choice strip; must work in both task-14900 layouts (the strip sits above the workbench split, so one mechanism serves both) |
| Prompts `#library-prompts-sort` | 2 | converge → strip. 2 options is toggle-sized, but Sort is a control FAMILY: Notes Sort already opens choices, and a same-named control that silently mutates on the next canvas is the exact one-grammar regression this batch exists to close |
| Skills `#library-skills-sort` | 2 | converge → strip (same family argument) |
| Export `#library-export-quality` | 3 | converge → strip rendered under the button (form has vertical room; the opener stays visible, so a second press also closes) |
| Prompts `#library-prompts-collection` | unbounded user data | ALREADY direct-pick: press opens `PromptCollectionManagerModal` (browse lane, full set, direct pick) — the "shared popover" divergence the task text sanctions; a one-row strip cannot fit an unbounded set. Work here = fix the now-dishonest vocabulary: drop the `⇄` (press does not cycle) via the shared chooser label, retire the "Cycles the prompt scope" tooltip |
| Search/RAG `#library-rag-mode-toggle` | 2 | KEPT one-press toggle (genuine two-state mode flip with retrieval-state reset; a strip adds a press to the most common action for zero information). AC#1 satisfied at the label: both options rendered with `✓` on the active one (`mode: ✓ Search ⇄ RAG Answer`) |
| Skill editor user-invocable / agent-invoke / context | 2 each | KEPT toggles, same both-options-in-label treatment; context keeps its task-418 plain-language hint on the ACTIVE option only to stay 60-col-safe |
| Notes `#library-notes-sort` | 3 | already the pattern — re-pointed at the extracted shared builder (no behaviour change) |
| Notes sync direction/conflict | 3+3 | already conformant (always-visible `✓` choice groups) — no change |
| Media "Trash" | n/a | not a cycler (task-4025 made it a nav action deliberately) — no change |

Extraction (one grammar, no second mechanism):
1. `Library/library_shell_state.py`: `library_choice_label(name, value)`
   (chooser-opener label, Notes-Sort spelling, no `⇄` — press opens choices),
   `library_choice_tooltip(subject, options)` (replaces the now-false
   "Cycles …" tooltips on converged sites), `library_toggle_label(name,
   options, active_index)` (kept toggles: full 2-option set on the label,
   `✓` active, `⇄` between the options keeps its press-advances meaning).
   Vocabulary comment updated. `library_cycle_label`/`library_cycle_tooltip`
   retire if caller-free after conversion.
2. New `Widgets/Library/library_choice_strip.py`: one strip composer
   (per-option Buttons, `✓ ` active prefix, `choice_value` attr — the sync
   groups' existing attr name). Notes canvas re-pointed at it (ids/classes
   preserved); its handler reads `choice_value` instead of the one-off
   `sort_mode` attr.
3. Screen wiring mirrors Notes exactly per site: a `_…_choices_visible` bool,
   opener toggles + recompose, choice handler applies + closes, toolbar row
   swaps for the strip (media/prompts/skills; export renders it under the
   opener). Media pick preserves today's select-mode discard semantics; the
   opener is inert while the bulk-delete confirmation is armed (task-2853
   AC3's "no drift under a confirm" rule). Keyboard: opening focuses the
   active choice; Escape and picks refocus the opener.
4. Escape: front-gate `action_library_list_focus_rail` (shared list-canvas
   Escape) and `action_library_export_back` — an open strip closes first.
5. Footer/F1 via the one seam (`_library_footer_shortcuts_for_current_state`):
   an open strip advertises "enter choose <subject> / esc cancel"; gate placed
   AFTER the bulk-delete-confirm gate to match binding resolution order.
6. Per-press cycle handlers retire on converged sites (media type, prompts
   sort, skills sort, export quality); `next_media_quality` retires with its
   caller if nothing else uses it.
7. TDD in `Tests/UI/test_library_choice_strips.py` (+ rewritten pins in the
   existing suites that assert cycle labels/handlers); docs
   (`Docs/User_Guide/library.md`, `library/media-and-conversations.md`,
   `library/import-and-export.md`, `library/search-and-rag.md`) updated with
   stamps. Live-verify ≥3 strips across ≥2 canvases, media in BOTH layouts,
   one keyboard-only path.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Converged the Library cyclers on the Notes Sort choice-strip pattern via ONE
extracted mechanism; kept the genuine two-option toggles as one-press flips
with their full option set moved onto the label.

**Shared mechanism (no second strip/labelling path):**
- `Library/library_shell_state.py`: `library_choice_label` (chooser-opener,
  glyph-free `name: value`), `library_choice_tooltip` ("Press to pick …"),
  `library_toggle_label` (`name: ✓ a ⇄ b` — full option set, ✓ active, ⇄
  between the options keeps its press-advances meaning),
  `LIBRARY_CHOICE_ACTIVE_MARKER`, vocabulary comment updated.
  `library_cycle_label`/`library_cycle_tooltip`/`next_media_quality` retired
  (zero callers).
- New `Widgets/Library/library_choice_strip.py`: the one strip composer
  (✓-prefixed active option, `choice_value` attr — the sync groups'
  convention). The Notes Sort strip is re-pointed at it (ids/classes
  unchanged; its handler reads `choice_value` instead of the one-off
  `sort_mode` attr).

**Converged to strips** (press opens the full option set, direct pick,
Escape/second-press cancels, opener refocused): media type filter (works in
both task-14900 layouts — the strip sits above the workbench split; inert
while the bulk-delete confirm is armed, per task-2853 AC3; select-mode
discard semantics preserved on an actual change), prompts sort (pick maps to
the exact browse scope at page 1), skills sort, export quality (opener stays
visible — the form has room and the label anchors the bare values — so a
second press also closes; strip state resets on each fresh Export visit).

**Kept as one-press toggles (documented decisions):** Search/RAG mode (a
two-state mode flip with retrieval reset — a strip would tax the most common
action; label now `mode: ✓ Search ⇄ RAG Answer`, RAG-39 next-mode tooltip
kept) and the three skill-editor switches (user-invoke / agent-invoke /
context; context keeps its task-418 plain-language hint on the ACTIVE option
only for 60-col safety). The prompts collection control was ALREADY
direct-pick (opens the collection manager modal — the sanctioned popover for
an unbounded user-data set); its work here was vocabulary honesty: the `⇄`
and the "Cycles the prompt scope" tooltip were dropped. The notes sync
direction/conflict groups already conform (always-visible ✓ choice groups) —
untouched. Media "Trash" is a nav action, not a cycler (task-4025) — untouched.

**Keyboard/footer:** opening a strip focuses the active choice; Escape
front-gates `action_library_list_focus_rail` and
`action_library_export_back` (close + refocus opener, never navigate);
footer/F1 advertise "enter choose <subject> / esc cancel" through the one
shared seam (`_library_footer_shortcuts_for_current_state`), gated AFTER the
bulk-delete-confirm branch to match binding resolution order.

**Tests:** new `Tests/UI/test_library_choice_strips.py` (15: builders, media
strip open/pick/escape/both-layouts/keyboard-only, footer for media+export,
export second-press + helper line, skills strip, prompts exact-scope pick,
collection vocabulary). Rewritten pins in test_library_shell.py (media type
flow → `_pick_media_type` helper; mode labels), test_library_skills_canvas.py
(handler pair + shell strip flow + label pins), test_library_prompts_canvas.py
(sort handler), test_library_prompt_collections.py (6 label pins),
test_library_honesty_accessibility.py (vocabulary test),
test_library_export_state.py (`next_media_quality` pins → option-set +
state-flag pins), product-maturity mode-label pins. One gate16 failure
(`test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional`)
A/B'd at clean base 0662e09f5: fails identically there — ambient, not this
change.

**Live-verified** (isolated tmux, scratch config `sdd_lq6`, cleaned up, live
config grepped clean): media type strip at 235-col side-by-side AND 100-col
stacked (mouse pick, keyboard-only Enter/Shift-Tab/Enter, Escape-cancel,
footer "enter choose type / esc cancel"), export quality strip (pick updated
value + helper line, second Enter closed), skills sort strip, and the mode
toggle's moving ✓. Docs: library.md + library/{media-and-conversations,
import-and-export,search-and-rag,notes,prompts,skills}.md copy updated with
new stamps.
<!-- SECTION:NOTES:END -->
