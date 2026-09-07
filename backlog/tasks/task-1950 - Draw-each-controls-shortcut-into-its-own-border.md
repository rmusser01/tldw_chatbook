---
id: TASK-1950
title: Draw each control's shortcut into its own border
status: To Do
assignee: []
created_date: '2026-08-02 20:30'
labels:
  - ux
  - navigation
  - design-system
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Borrowed from [Bagels](https://github.com/EnhancedJax/Bagels) (TUI expense tracker,
Textual, 2.9k stars) after reviewing its screenshots.

Bagels prints a control's key **on the control**, in its border, rather than only in a
footer legend or a help screen. Its Templates pane carries `1 - 9` in its bottom border
and each template button carries its own digit; the period stepper carries `← . →`; the
account switcher carries `[ ]`; a Date/Person toggle carries `q w`. The footer still
exists, but it is a reminder rather than the only source of truth.

In Textual this is `border_title` (top-left) and `border_subtitle` (bottom-right) — one
line per widget, no new machinery.

We have 47 screens in `UI/Screens/` and 34 `BINDINGS` blocks. A user cannot hold that,
and today the only way to learn a screen's keys is the footer, which is truncated on
narrow terminals and cannot name a key that belongs to one specific control. This is the
cheapest discoverability win available to us, and it composes with a jump-mode
overlay (filed separately) rather than competing with it.

Scope this as a convention plus a first application, not an app-wide sweep: pick two or
three dense screens (Console and Evals are the obvious candidates), establish the rule in
`DESIGN.md`, and let the rest follow as those screens are touched.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] `DESIGN.md` states the convention: a control with a dedicated key shows it in its `border_subtitle`, and a pane whose children share a key range shows the range in its own border subtitle
- [ ] The convention says what to do when a control has no border, and when a key is configurable rather than fixed
- [ ] At least two dense screens apply it, and the keys shown are the keys that actually fire
- [ ] A test asserts the rendered hint matches the widget's real binding, so a renamed action cannot leave a lying label behind
- [ ] Hints stay legible at 80x24 as well as 160x45 and 235x52, and never push a control out of reach
<!-- AC:END -->
