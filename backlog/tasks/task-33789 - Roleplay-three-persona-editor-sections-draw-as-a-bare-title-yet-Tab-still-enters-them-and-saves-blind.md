---
id: TASK-33789
title: >-
  Roleplay: three persona-editor sections draw as a bare title, yet Tab still
  enters them and saves blind
status: To Do
assignee: []
created_date: '2026-10-02 04:56'
labels:
  - roleplay
  - ux-review-2026-10-01
  - bug
  - layout
dependencies: []
references:
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md
  - Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/findings.json
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-073 (P0, severity 4, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.** Scrolled to the bottom, the persona editor shows the heading "Tool policy rules (narrowing-only)" and then Save. The rule list, Kind, Name, Allowed, Require confirmation, Max calls/turn, New/Save/Delete, the deny-by-default warning and the status line are not drawn. The Shared Visual Identity reactions browser and New Actor Pack's "Required portrait" picker collapse the same way. Keyboard focus still walks into the hidden controls (8 invisible Tab stops at 220x55; Tabs 8-15 of 16 at 160x45), so values can be typed and saved unseen: in a live run a blind key sequence created and persisted the rule "skill: web_search → allow". New Actor Pack also pre-selects the first eligible character's portrait in its invisible picker. Verified at 120x36, 160x45 and 220x55 for the policy section and at 120x36 for the portrait picker; the CSS cause is the same at every size.

**Who it hurts.** Persona managers (J4) cannot see or use tool policy, shared reaction art or the portrait choice. A blind "allow" rule silently stops every other tool of that kind being offered (spawn_subagent included), and the warning that says so is in the hidden block. An Actor Pack persona is silently bound to a portrait the user never chose.

**Cause.** The three containers sit inside `VerticalScroll#personas-editor-body` with Textual's default height (1fr) and hidden overflow. No rule gives them an automatic height, so the editor's 1fr TextAreas take the space.

**Evidence:**
- `persona_profile_editor_widget.py:109-175`; `widget_defaults_self.tcss:2120-2150`.
- `personas_policy_rules_editor.py:33-37, 52, 153-181`: the hidden block holds the deny-by-default warning and every validation message. `persona_profile_editor_widget.py:323, 333-335, 561-566`: `begin_actor_pack_creation` pre-selects the first portrait option.
- Captures: `review-rv-gap2-persona-editor-bottom-policy-clipped-160x45` row 39; `review-rv-gap2-2-persona-editor-bottom-policy-clipped-120x36` row 30; `review-rv-gap2-3-persona-editor-bottom-policy-clipped-220x55` row 49; `review-rv-gap2-policy-rule-saved-blind-while-form-unsaved-160x45` rows 19-20; `review-rv-gap2-2-new-actor-pack-portrait-select-hidden-120x36` rows 15-17.

**Constraint worth knowing.** The boot CSS bare-selector census is at its cap (274/274), so broad new selectors are not available.

**Out of scope:** each long persona field filling the whole screen (RP-076) shares the container cause but is a separate finding; the validation footer that prints internal widget ids is covered by TASK-1651's persona-editor criterion; moving Tool policy and Reactions into their own modes is redesign work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 At 120x36, 160x45 and 220x55, scrolling to the Tool policy rules section of the persona editor shows all of it: the rule list, Kind, Name, Allowed, Require confirmation, Max calls/turn, New/Save/Delete, the deny-by-default warning and the status line.
- [ ] #2 At the same sizes, the Shared Visual Identity reactions browser and New Actor Pack's Required portrait picker are fully drawn when scrolled to.
- [ ] #3 Every Tab stop in the persona editor lands on a control that is visible on screen, scrolled into view if needed.
- [ ] #4 In New Actor Pack, the portrait that will be saved is visible before Save.
- [ ] #5 A regression test asserts at 120x36 that each of the three sections renders at least as tall as its content; it fails on the current code.
<!-- AC:END -->
