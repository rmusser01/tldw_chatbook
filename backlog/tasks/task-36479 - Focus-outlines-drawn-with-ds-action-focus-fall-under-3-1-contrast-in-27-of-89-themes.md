---
id: TASK-36479
title: >-
  Focus outlines drawn with $ds-action-focus fall under 3:1 contrast in 27 of 89
  themes
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-23 11:02'
updated_date: '2026-10-24 14:53'
labels:
  - ui
  - accessibility
dependencies: []
references:
  - Docs/superpowers/specs/2026-10-02-roleplay-library-frame-design.md
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Exists on dev today and is not fixed by Roleplay frame B2a (TASK-33910.3): the focus edge token $ds-action-focus measures under 3:1 against $surface in 27 of 89 themes, and dev draws focus outlines with it on 23 stylesheets today, among them Library's and Settings'. B2a's own items list already uses $ds-active-fg for its focus edge (owner decision 2026-10-23, "Both (Rec.)": file this follow-up and amend the Roleplay spec to name $ds-active-fg). Fix direction: $ds-active-fg (the Settings rail precedent) or a re-tuned token.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every focus edge dev draws with $ds-action-focus clears 3:1 against its surface on every measurable theme (scope per the title; focus edges dev draws with other tokens are listed in the notes, not fixed here)
- [ ] #2 A contrast test pins it, so a token change that drops a theme under 3:1 fails
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Inventory every $ds-action-focus use in the CSS sources (tcss modules and Python CSS blocks): focus edges (border*/outline* in a :focus, :focus-within or -focused rule) versus every other use; measure each against $surface, $panel and the focus rule's own fill on the 89 measurable themes.
2. Red first: a contrast test in Tests/UI/test_theme_contrast.py that discovers every focus edge in the app's CSS sources, resolves its token in its own source's variable scope, and requires 3:1 against $surface, $panel and its rule's own fill over each, on every measurable theme. Dev-debt tokens outside this task's scope (other $primary/$accent aliases) are listed with a stale-entry guard; a vacuity guard pins known edges. Run it against dev's tokens.
3. Fix with the smallest blast radius: switch the focus-edge declarations from $ds-action-focus to $ds-active-fg (the Settings rail and Roleplay list precedent); PromptBlockEditor's DEFAULT_CSS aliases the token to $accent, so its focus edges move to $text-accent. Every non-focus use keeps $ds-action-focus. Regenerate the generated sheets with css/build_css.py.
4. Update the tests that pin the old token literally, the DESIGN.md focus convention, the _variables.tcss comment and the Roleplay spec lines that cite the Library house rule.
5. Named mutations; gates: touched tests, paired run against fu-base, ruff, preflight; boot-CSS bytes and the broad-selector ratchet measured base vs head.
<!-- SECTION:PLAN:END -->
