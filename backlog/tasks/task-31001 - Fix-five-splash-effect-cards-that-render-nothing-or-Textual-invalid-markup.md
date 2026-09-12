---
id: TASK-31001
title: Fix five splash effect cards that render nothing or Textual-invalid markup
status: To Do
assignee:
  - '@{self}'
created_date: '2026-09-02 15:41'
labels:
  - splash-screen
  - bug
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Verified 2026-09-01 while fixing splash playback stability (task-30016): five animated cards are broken independently of the animation driver. cyberpunk_glitch, hypno_swirl and phonebooths emit rich markup strings that Textual's own Content.from_markup parser (used by Static.update via visualize) rejects -- 'closing tag does not match any open tag' -- so their first real tick raises, the driver stops the animation and falls back to the static card mid-splash (cyberpunk_glitch only fails on some random draws, which is why the intro sometimes plays fine and sometimes collapses). world_map's effect crashes with AttributeError ('WorldMapEffect' object has no attribute 'height') on every frame. typewriter_news's update() has no return statement (its render() is never called), so it produces no frames at all and the splash area stays blank for the whole duration. With random card selection this is roughly a 1-in-15 chance per launch of an intro that visibly does not play. Repro sweep: construct each card's effect and parse frames through textual.content.Content.from_markup; see Tests/Widgets/test_splash_animation_playback.py KNOWN_BROKEN_EFFECT_CARDS.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 cyberpunk_glitch renders frames that pass Content.from_markup on every draw,hypno_swirl renders frames that pass Content.from_markup,phonebooths renders frames that pass Content.from_markup,world_map effect constructs and renders without AttributeError,typewriter_news update() returns rendered frames (render() wired in or inlined),Removed from KNOWN_BROKEN_EFFECT_CARDS in Tests/Widgets/test_splash_animation_playback.py and the smoke test still passes
<!-- AC:END -->


## Renumbering provenance

Second collision: the renumbered ids 30016/30017 were themselves minted on dev (Server-capture backlog batch) while this PR was open. Final ids 31000/31001 sit far beyond dev's allocation frontier (concurrent `backlog task create` sessions mint at local max+1) so an open PR cannot keep racing the frontier.

Originally created as TASK-28028; renumbered to TASK-31001 alongside
TASK-30016 (see its Renumbering provenance section) after dev independently
minted a colliding TASK-28026 and merged it first.
