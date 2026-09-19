---
id: TASK-32626
title: >-
  Library Notes: guide residual at 77eb2601a6 — three wave-4 claims critique 4
  contradicts, plus changelog clauses in user prose
status: Done
assignee:
  - '@robert'
created_date: '2026-09-15 06:45'
updated_date: '2026-09-15 19:20'
labels:
  - library
  - notes
  - critique-4
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Critique #4 (dev 77eb2601a6), assessor A section 10 and assessor B section 8, every persona. Wave 4's own sweep (task-32558, PR #2687) re-verified about 300 claims, so each item below is either NEW -- a claim the sweep itself added for a wave-4 fix -- or MISSED -- a claim checked only on its happy path.

NEW (added by wave 4, contradicted live):
- notes.md promises Receipts shows 'Wrote note to file' for a note you edited in Chatbook. No user-reachable path produces that receipt (the P0's docs half).
- 'The New note view no longer shares the canvas with an empty notes list … takes the stage': at 235 columns it shares with navigation (37) and the list (62), taking about 105 (A cap 03).
- 'While no note is open the list takes the width the empty work area would otherwise waste': the list widens to about 132 columns but a 54-column work pane remains, holding one sentence (A cap 08).
- The chooser 'bar below holds only ‹ Notes': nothing is rendered there (A cap 17).

MISSED (true on the swept path, false beside it):
- 'slash focuses the filter and selects whatever is already in it' -- true only from the navigator (A caps 41-43; B VERIFIED from the navigator).
- 'The footer names whichever editor control has focus' -- false in Info (A caps 11-13).
- 'the picker opens with that File name field already focused' -- false for the Folder-files door (A caps 27, 28).
- 'Saved appears once per view' -- true of Info, but printed twice in the editor header (A cap 06).
- 'Once a note has saved in this session the state names the time' -- on reopening, a bare 'Saved' (A cap 09).
- A warning callout 'becomes Warning' -- type and body run together (A cap 25).

DOC DEFECT: about 30 inline '(Was … — superseded by task-3xxxx)' clauses sit inside user-facing prose across notes.md. That is a commit log in a user guide, and the wave added more of them.

B's own conformance pass VERIFIED 17 further claims (Ctrl+N, Escape, footer strings, Import-once copy, Folder-files keys, Session Git end to end), so the page is mostly true -- these are the exceptions. Adjacent open rider 32589 proposes making the guide-claim string check a gated test.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each contradicted claim is corrected or removed, and the verification stamp names this critique
- [x] #2 Claims a fix introduces state the width or region they hold for, rather than generalising from the one case walked
- [x] #3 The changelog clauses move out of user-facing prose
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Locate each cited claim's exact sentence(s) in Docs/User_Guide/library/notes.md by grep.
2. For each NEW/MISSED claim, trace the underlying code (widget compose, footer-shortcut resolver, presentation-state builder, reconciler) to confirm or refute the contradiction before editing -- do not take the critique's characterization on faith where a quick code trace settles it, but do trust it where live-capture evidence is the only kind that could (width measurements, focus behavior).
3. Correct each confirmed claim in place, scoping it to the width/region it actually holds for rather than generalising (AC#2).
4. For the two claims where evidence conflicts (Receipts "Wrote note to file", the chooser's pinned-bar text) and neither side was walked live in this session, leave the guide's existing wording alone and record the conflict rather than guessing.
5. Sweep the whole page for "(Was ... -- superseded by task-NNNNN)" clauses sitting in feature-description prose (excluding "Verified against" stamps, where that framing is the intended convention) and rewrite each into the present tense of what the page already says, dropping the clause entirely where it carried no fact beyond the surrounding sentence.
6. Append one new "Verified against" stamp naming this critique, recording what was corrected, what was checked and left alone, and why.
7. Run scripts/check_guide_claim_strings.py --fail-on-miss against the page and resolve every miss it reports that traces to something this pass touched.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Ten findings processed individually (4 NEW, 6 MISSED), plus the changelog-clause sweep.

NEW, corrected: (1) New-note view width -- corrected from "no longer shares
the canvas" to the real 235-column split (list ~62, view ~105, rail ~37),
traced to LIBRARY_NOTES_READER_PROFILE's list_comfort_width=64 and the
reader_has_item branch task-32544/32547 added. (2) "empty work area would
otherwise waste" -- corrected to note the work_min_width=48 floor the empty
pane keeps for its own message. (3) The chooser's "bar below holds only
‹ Notes" -- DISPUTED, left alone (see below). (4) Receipts "Wrote note to
file" -- DISPUTED, softened rather than reversed (see below).

MISSED, corrected: slash-filter select-on-focus scoped explicitly to "the
Notes list only" (its own heading now says so, not just its section
placement); "the footer names whichever editor control has focus" scoped
to Edit/Preview, with Info's actual fixed "enter run action" tier named
(LIBRARY_NOTES_CONTEXT_SHORTCUTS); the picker's "File name field already
focused" scoped to Import once/Keep a folder synced's three doors, with an
explicit exclusion naming Folder files' own pre-filled Folder path field;
"'Saved' appears once per view" corrected to "once, in Info" plus the
editor's own repeat; "Once a note has saved... names the time" corrected
to name the reopen-resets-to-bare-Saved ceiling, traced to
LibraryNoteSessionSnapshot's status_message resetting on a fresh session
snapshot; the warning-callout sentence gained a clause on the body/type
running together, traced to render_obsidian_callouts's _header rewriting
only the header line.

DISPUTED, left unchanged (both findings lack a capture citation, unlike
every other bullet in this task -- the two are noted as concerns rather
than resolved by guessing): Receipts' "Wrote note to file" is contradicted
by this file's own adjacent wave-4 stamp (cites roots-11/roots-12) and by
the reconciler's note_changed -> UPDATE_FILE branch, which structurally
supports the claim -- softened the sentence to state the vocabulary
without asserting the specific pathway. The chooser's "holds only
‹ Notes" is contradicted by a passing pin
(test_one_back_cue_grammar_across_the_notes_surfaces, extended this
session for task-32624) that renders exactly that text -- left untouched.
**Superseded by task-32604, merged to origin/dev after this branch
forked**: that PR's own commit message states the guide's claim "no
user-reachable path produced" was true, fixes the underlying gap, and
rewrites the Receipts/Check-changes prose comprehensively -- the
controller flagged this mid-task; dev's version should win over this
branch's narrower hedge once this branch sits on top of it. Also per the
controller: task-32606 (also merged after this branch forked) fixes an
unrelated false claim in Docs/User_Guide/library/file-notes.md, out of
this page's scope.

Changelog clauses (AC#3): removed or rewrote roughly a dozen inline
"(Was ... -- superseded by task-NNNNN)" clauses sitting in feature
description prose (not the ~60 legitimate "Verified against" stamps,
where that framing is the intended convention and was left alone). Each
was either deleted outright (where it carried no fact beyond the sentence
already in front of it) or rewritten into the present tense of the live
fact it did carry (e.g. "Import once and Keep a folder synced open the
files-or-one-folder dialog... only Folder files' 'Choose File Notes
Folder' has the pre-filled 'Folder path'" survives, its "(Was ... --
superseded by task-32271...)" wrapper does not).

Verification stamp (AC#1): appended one new "Verified against" entry
naming critique #4 / task-32626, honest about its method -- a source-level
re-read against the tree and the existing test suite, not a fresh live
capture like the stamps above it, since this task's brief is copy-only
and no behaviour changed to walk.

scripts/check_guide_claim_strings.py --fail-on-miss: 413 quoted strings
checked, 145 flagged not-emitted -- read individually; none trace to a
new claim added by this pass. The genuinely new additions this task wrote
(e.g. "Select a note to edit it here.", "enter run action") were NOT
flagged, confirming they match the source verbatim. The flagged 145 are
overwhelmingly composed/illustrative examples ("Reading list · Unfiled ·
2h"), meta-commentary quoting a clause being discussed (this task's own
closing stamp quoting "(Was … — superseded by task-NNNNN)" as the pattern
it removed), and pre-existing content outside this task's ten findings --
consistent with the script's own docstring ("a guide legitimately quotes
strings no source emits"). One harmless false-positive of my own: line
387's "Keyboard on this toolbar (the Notes list only)." matched the
script's bold-run regex as if it were a quoted UI string; it is a section
lead-in I wrote, not a UI-copy claim.

No source code changed for this task; Docs/User_Guide/library/notes.md
only.
<!-- SECTION:NOTES:END -->
