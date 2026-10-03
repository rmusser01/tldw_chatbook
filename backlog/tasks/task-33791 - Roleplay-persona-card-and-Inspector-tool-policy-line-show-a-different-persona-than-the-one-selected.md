---
id: TASK-33791
title: >-
  Roleplay: persona card and Inspector tool-policy line show a different persona
  than the one selected
status: To Do
assignee: []
created_date: '2026-10-02 04:57'
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
Found by the 2026-10-01 Roleplay NN/g + HCI review: finding RP-078 and RP-079 (both P1, severity 3, effort S). Report: `Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/report.md` (section 5 and Appendix A), with the machine-readable entry in `findings.json` beside it. The review ran on origin/dev @ 84247cb843 (worktree `.worktrees/roleplay-library-ux`).

Line refs below were re-checked on origin/dev @ ab4df99959. Paths are under `tldw_chatbook/` (`personas_screen.py` is `UI/Screens/personas_screen.py`; the `personas_*` and `persona_*` widgets are in `Widgets/Persona_Widgets/`). Capture names refer to the review's capture set; those copied into the QA folder are marked `evidence/`.

**What happens.**
- **RP-078.** Selecting a persona never refreshes the Inspector's Tool policy line. It keeps whatever the last opened editor or rule save wrote, so persona B can show persona A's rules, and a persona that has rules reads "no rules" when first selected. Live: after a rule was added to Terse Code Reviewer and the edit cancelled, both Cozy Storyteller and Isolde's Bridge Officer showed "Tool policy: 1 rule(s) / skill: summarize → allow", although neither has rules.
- **RP-079.** Save keeps the persona editor open; Cancel then re-shows the card, but the card is never re-rendered after a save. After creating a persona and pressing Save then Cancel, the card shows the persona that was selected before while the Inspector names the new one, and Edit refuses with "Selection out of sync; reselect the persona." After editing an existing persona, Save then Cancel leaves the pre-edit text on the card while the rail and Inspector show the new name.

**Who it hurts.** Persona managers (J4) judge a persona's tool permissions from a line that may describe another persona, and may hand a persona to Console believing it is restricted when it is not, or the reverse. Right after the main create step, the screen shows one persona while Chat now, Export and Delete act on another (the Delete dialog names the right one, which reduces but does not remove the risk).

**Shared cause.** The persona selection, save and cancel paths each refresh only some of the displays of the selected persona, so the card, the Inspector and the rail can describe different records.

**Evidence:**
- `personas_screen.py:5383-5440`: `_select_profile` never refreshes the policy summary; only Edit-open (`:8374`), rule save (`:15600`) and after-save (`:15784`) do. `personas_inspector_pane.py:249-253, 403-431`: clearing the selection does not reset it.
- `personas_screen.py:5408`: the card is rendered only in `_select_profile`. `:15733-15810` (`_after_profile_save`) re-selects the saved id without re-rendering the card; `:15866-15882` (`_finish_cancel_profile_edit`) re-shows the stale card; `:8333-8341` makes Edit reject the mismatched id.
- Captures: `review-rv-gap2-4-stale-policy-summary-other-persona-160x45` rows 19-20; `review-rv-gap2-2-stale-card-after-create-save-cancel-120x36` rows 15-21; `review-rv-gap2-2-edit-selection-out-of-sync-120x36` rows 15-31; `review-vf-gap2-stale-card-after-edit-save-cancel-160x45` rows 15-20.

**Out of scope:** the persona editor's endless "Loading Shared Visual Identity reactions…" (RP-088) has a different cause (a background load that stops silently when its freshness check fails). Moving the policy summary into a persona Info mode is redesign work, and copy explaining the deny-by-default effect of allow rules belongs to the persona-semantics work.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Selecting any persona shows that persona's own tool-policy rules in the Inspector (or that it has none), whichever persona was edited or whichever rules were saved earlier in the session.
- [ ] #2 Clearing the persona selection clears the Inspector's tool-policy line.
- [ ] #3 After creating a persona and pressing Save then Cancel, the card shows the new persona, matching the Inspector and the rail, and Edit opens it without a "Selection out of sync" message.
- [ ] #4 After editing an existing persona and pressing Save then Cancel, the card shows the saved text.
- [ ] #5 After any sequence of select, create, edit, Save and Cancel, the card, the Inspector and the rail name the same persona.
- [ ] #6 Regression tests: select persona B after editing persona A's rules and assert B's own rules are shown; run create, Save, Cancel and edit, Save, Cancel and assert the card name equals the Inspector name. They fail on the current code.
<!-- AC:END -->
