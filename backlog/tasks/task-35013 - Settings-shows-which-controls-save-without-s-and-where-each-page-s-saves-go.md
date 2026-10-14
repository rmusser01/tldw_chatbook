---
id: TASK-35013
title: Settings shows which controls save without s, and where each page's saves go
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- ux
- save-model
- ux-review-2026-10-14
dependencies:
- TASK-33007
- TASK-35016
- TASK-35017
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console Behavior is a draft page, which means edits wait until the user presses s. Yet four of its controls save the moment they are touched: Show model thinking, Status row placement, Chat images and Permission summaries. A fifth group, Exchange capture, saves only through its own Apply button.

The first three lack the shared instant-apply label that ADR-033 rule 3 requires. While one of them is focused, the Scope Inspector's only Save row says edits save with s. The State line also gives no hint that the page has exceptions.

The Inspector's storage line says "Local-only: saves write your config file." on every page, including pages that save to a database and pages that save nothing at all. The User Guide's and DESIGN.md's save-model text no longer matches the badges the app shows. In particular, the guide says s works on seven categories when it works on ten.

As a result, a user cannot tell from the screen whether a change needs s, or where a save goes. This task does four things:
- makes every control that saves without s on a draft page declare itself the same way;
- has the State line summarise those controls;
- makes the storage line true for each page;
- brings the documented badge tables back in line with the app.

No commit model changes (ADR-033 rule 1).

Plan: `backlog/docs/plan-2026-10-14-settings-save-honesty.md`. Findings closed: D1, D-shell-design-M1, D5, D-shell-design-M4 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 On every category that s saves, each control that writes a saved setting without s and without its own confirmation dialog has a hint next to it. The hint uses the shared instant-apply wording, or names the control's own Apply button and says s does not save it (Exchange capture). Providers & Models' 'Save selected' is out of scope here.
- [ ] #2 Focusing one of those controls shows an Inspector Save row that matches how that control saves. Focusing a staged field, or nothing, shows the staged row. The Inspector never shows two Save rows that disagree.
- [ ] #3 At rest, Console Behavior's State line says how many of its groups save without s. Providers & Models, whose instant groups are named in their Advanced titles, keeps today's line, and so does every category with no such groups. Every guided category's clean and dirty State lines stay one row at 211x44 and 235x52.
- [ ] #4 Changing Status row placement or Chat images still writes the config at once and leaves no unsaved count. The saved value survives leaving and reopening Settings.
- [ ] #5 The Inspector's storage line says in plain words where the open category's saves go: for example your config file, the Agents database, the workspace records, your encrypted profile, a theme file, or config plus a page's own store. On pages that save nothing, it says so. Providers & Models and Console Behavior keep 'Local-only: saves write your config file.'
- [ ] #6 A test fails when a registered control that saves without s has no hint or no matching Inspector Save row. It also fails when a Settings category has no storage-line entry.
- [ ] #7 Docs/User_Guide/settings.md lists every badge the State line can show, the at-rest draft wording, and each category under the badge it shows, and what it says about which categories s saves matches the app. DESIGN.md's persistence-badge bullet lists the same badges. A test fails when a badge the app can show is missing from the guide.
- [ ] #8 The change does not raise settings_screen.py's budget in the module size ratchet, and the new tests run on every pull request.
<!-- AC:END -->
