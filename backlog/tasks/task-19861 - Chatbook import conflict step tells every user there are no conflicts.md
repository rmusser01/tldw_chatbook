---
id: TASK-19861
title: >-
  Chatbook import conflict step tells every user there are no conflicts
status: Done
assignee:
  - '@t1-chatbook'
created_date: '2026-08-22'
labels:
  - ux
  - honesty
  - chatbooks
  - bug
priority: high
dependencies:
  - TASK-19550
  - TASK-19734
---

## Description

Source: surfaced by the reviewer of **TASK-19734**, who drove awkward import
cases through the real wizard against real SQLite. Re-verified at `3605bd52d`
(after 19734 merged as `cf38ee6f8`).

Step 3 of the chatbook import wizard asks the user to choose how conflicts
should be resolved — Skip, Rename, Replace, or Merge. Directly beneath that
choice, the step displays:

> ✅ No conflicts detected - all content is new

This message is **unconditional**. It is a plain `Static` yielded in
`compose()` (`UI/Wizards/ChatbookImportWizard.py:639-644`) with no condition
attached, `display=True` on every render, and no code anywhere ever hides it.
**No conflict check exists anywhere in the wizard or the importer.** Driven
against a destination where all four of the chatbook's items were already
present and would all be skipped, it still says "all content is new".

Every import passes through this step, so every user picks their conflict
strategy on a premise the application invented. A user who reads it and leaves
the default in place has been told that their choice does not matter, which is
exactly backwards in the case that matters most.

The lesser half of the same defect: `#conflict-container`
(`:635-637`), the "Potential Conflicts Detected:" panel and its
`#conflict-list`, carries the `hidden` class in `compose()` and is never
unhidden — no Python outside `compose()` references either id. So the wizard
ships both the true panel (permanently invisible) and the false one
(permanently visible).

This is the same family as **TASK-19550** (a "Create backup" checkbox that
writes a ✓ and does nothing) and **TASK-19734** (per-type ticks the import did
not earn): *the app asserts an outcome it did not produce*. Given that the
false claim here is what the user's next decision is based on, it is arguably
higher-impact than either.

Either half is a valid fix — perform a real conflict check and drive both
elements from it, or say nothing at all about conflicts — but shipping a
reassurance nothing computed is not.

## Acceptance Criteria

- [x] The conflict step never claims that no conflicts were detected unless a
      check actually ran and actually found none
- [x] When the destination already contains items the chatbook would import,
      the step either names them or says nothing — it does not assert that all
      content is new
- [x] No element in the step is rendered permanently-hidden with no code path
      that can reveal it (either `#conflict-container` becomes reachable, or it
      is removed)
- [x] A test drives the step against a destination pre-loaded with every item
      in the chatbook and asserts the "no conflicts" copy is absent, and is
      mutation-checked (restoring the unconditional `Static` makes it red)
- [x] A test drives the step against a genuinely empty destination and asserts
      whatever the step now says is true of that case
- [x] The user-facing documentation page for chatbook import matches whatever
      the step now claims

## Implementation Plan

1. Add a read-only `ChatbookImporter.scan_conflicts(chatbook_path)` that reads
   each item's collision key out of the validated archive (conversation file
   `name`, note manifest title, parsed character-card name, media metadata
   URL) and looks it up in the destination databases with the same lookups
   the import path uses; check both the plain and `[Imported] `-prefixed
   name variants, because the rename option is chosen on a later step.
2. Drive both step-3 elements from that scan: findings populate and reveal
   `#conflict-container`; a completed scan with no findings reveals an earned
   no-conflicts message that states what was checked; no manifest or a scan
   error shows neither.
3. Born-red tests: pre-loaded destination (claim absent, items named), empty
   destination (earned claim), pre-scan state (no claim exists yet), scan unit
   tests (findings, prefix variants, empty, broken archive, no writes), and a
   structural lock on compose()'s shipped classes plus the panel's reveal path.
4. Mutation-verify: restoring the unconditional `Static` must turn the guard
   tests red.
5. Update the user-facing docs (`Docs/User_Guide/artifacts.md`) to match the
   step's new claims.

## Implementation Notes

- Approach taken: the real-check option from the task's "either half" note.
  `ChatbookImporter.scan_conflicts` (chatbook_importer.py) validates the
  archive through the same `_preflight_archive` used by preview/import, reads
  per-type collision keys from the archive members the importer itself reads,
  and looks them up in the destination DBs (`get_conversation_by_name`,
  `get_note_by_title`, `get_character_card_by_name`, `get_media_by_url`) --
  so the scan cannot drift from the import's own conflict semantics. It is
  scoped to the four content types the resolution strategy governs; prompts
  (which fail on collision rather than resolve) and kept briefings (matched
  by content) are excluded, and the step's copy states that coverage.
- Both prefix variants are checked because `prefix_imported` is chosen on the
  NEXT step; a finding reports the destination name that matched.
- The wizard step (`ConflictResolutionStep`) now runs the scan in a worker on
  every `on_show` (fresh on back-navigation, `exclusive=True`), resets both
  elements to hidden before each scan, and clears the reassurance copy
  outright (not merely hides it) when the scan falsifies it.
- Tests: `Tests/Chatbooks/test_chatbook_import_wizard_conflict_honesty.py`
  (10 tests, all green). Mutation check performed: removing `hidden` from the
  compose-time `#no-conflicts` classes (restoring the unconditional Static)
  turns `test_the_claim_does_not_exist_before_any_check_has_run` and
  `test_no_conflicts_element_ships_hidden_and_the_panel_has_a_reveal_path`
  red; restoring it makes all 10 green.
- The file carries `pytestmark = pytest.mark.bootstrap_profile` (the same
  escape hatch Tests/Chatbooks/test_provider_continuation_roundtrip.py uses):
  constructing real Chatbook databases through the config-participant
  admission chain fails closed with RecoveryRequired under the per-test
  config redirect otherwise (documented trap in Tests/conftest.py).
- CSS: added `ConflictResolutionStep .conflict-item` to
  `tldw_chatbook/css/features/_wizards.tcss` (token-composed) and rebuilt the
  bundle with `python tldw_chatbook/css/build_css.py`; the design-token
  governance test stays green.
- Docs: `Docs/User_Guide/artifacts.md` gained an "Importing a pack: the
  conflict step" subsection describing the three states and the coverage
  boundary.
- ADR required: no -- routine bug fix inside an existing UI flow; no schema,
  sync, boundary or contract change.
- Pre-existing reds noted (not caused by this change, present at base
  a3c47c6c72 in this environment): `test_chatbook_import_result_honesty.py`
  (12 fixture errors + 4 failures) and `test_chatbooks_smoke.py::
  test_chatbook_creator_minimal_archive_creation` all fail with
  `RecoveryRequired("raw_source_selection_changed")` -- the same
  config-participant admission trap, on files that do not carry the marker.
- Modified/added files: `tldw_chatbook/Chatbooks/chatbook_importer.py`,
  `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`,
  `tldw_chatbook/css/features/_wizards.tcss`,
  `tldw_chatbook/css/tldw_cli_modular.tcss` (rebuilt),
  `Tests/Chatbooks/test_chatbook_import_wizard_conflict_honesty.py` (new),
  `Docs/User_Guide/artifacts.md`.

## Notes

Scope note for the implementer: if a real conflict check is out of scope for
this task, removing the false claim is a complete and shippable outcome on its
own — an honest silence is strictly better than a confident lie, and the
strategy selector still works. Do not close this by making the claim *more
specific* without computing it.
