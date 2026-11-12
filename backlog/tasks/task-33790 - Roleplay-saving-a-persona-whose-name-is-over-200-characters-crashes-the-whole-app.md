---
id: TASK-33790
title: >-
  Roleplay: saving a persona whose name is over 200 characters crashes the whole
  app
status: To Do
assignee: []
created_date: '2026-10-02 04:56'
updated_date: '2026-10-04 18:52'
labels:
  - roleplay
  - ux-review-2026-10-01
  - bug
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-075 (P1, severity 3, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** The persona Name box accepts any length, but the data model allows 200 characters (tool-rule names allow 512). On Save, the model raises a validation error whose text contains square brackets. The screen passes that text to a toast, the toast parses it as markup, the parser fails, and the app exits to a Python traceback. Reproduced live on both the create and the edit path, at 160x45 and 120x36. The trigger is validation text shaped like markup ("[type=…, input_value=…]"); bracketed text such as "[Errno 2] …" renders fine.

**Who it hurts.** One over-long paste into Name (for example, a description pasted into the wrong box) kills the app and loses every open draft (character, persona, visual). The only message the user sees is a traceback.

**Evidence:**
- `personas_screen.py:16249-16269`: `_notify` calls the app's notify with Textual's default markup parsing. `:15596` and `:15711` build "Save failed: {exc}", and about 30 other `_notify` calls interpolate exception text the same way. The validation error comes from the screen's own model construction (`:15671`, `:15697`).
- `persona_profile_editor_widget.py:127, 544-568`: the Name input has no maximum length, and `validate()` checks only for a blank name. `tldw_api/character_persona_schemas.py:562, 581, 626` cap the name at 200 and the rule name at 512.
- Review log `rv-gap2` `tldw_cli_app.log:394, 409-410`: string_too_long, then `exception_type=MarkupError widget_type=PersonasScreen`.
- Capture: `review-rv-gap2-persona-name-too-long-save-160x45` rows 1-46 (the traceback).

**Prior art.** TASK-1513 (To Do) tracks the repo-wide markup-parsing gap and a convention for it; so far it has hardened only the Evals surfaces. This task fixes the Roleplay instance. The house precedent is `UI/Evals/notify_mixin.py:40-48` and `evals_screen.py:1228`, which show exception text without markup parsing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Saving a new or an existing persona with a name longer than 200 characters never exits the app; the user sees a readable message and every open draft remains.
- [ ] #2 The 200-character name limit is visible before Save: a longer name is either prevented at entry with a visible notice, or flagged inline next to the field.
- [ ] #3 The validation message names the field and its limit in plain words, not raw validation-library text.
- [x] #4 Any Roleplay toast whose text contains square brackets (exception text or user-entered names) shows them as literal text and never raises.
- [ ] #5 A tool-policy rule name longer than 512 characters is handled the same way: no crash, and a readable message.
- [ ] #6 Regression tests: a validation error on the persona save path (create and edit) produces a toast without raising, and a Pilot-driven Save with a 205-character name leaves the app running; they fail on the current code.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
2026-10-04, TASK-34400: the app-exit half is fixed. Every Roleplay toast now renders literally (PersonasScreen._notify passes markup=False), so the persona save with a 205-character name shows the pydantic text in a toast and the app keeps running (pinned by Tests/UI/test_roleplay_hostile_names.py::test_a_persona_save_that_fails_validation_keeps_the_app_running). AC#4 is delivered. Still open here: AC#1's readable message (the toast still shows raw validation text), AC#2 (visible 200-character limit), AC#3 (plain-words message), AC#5 (512-character rule name: no longer exits, but the message is raw) and AC#6's Pilot-driven create and edit paths.
<!-- SECTION:NOTES:END -->
