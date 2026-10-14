---
id: TASK-35020
title: Settings search finds every rendered setting, and its drift guards fail pull requests
status: To Do
created_date: 2026-10-14 17:10
labels:
- settings
- search
- ci-guards
- ux-review-2026-10-14
dependencies:
- TASK-33007
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Typing a setting's name after '/' in Settings is meant to find it, and the User Guide says the filter knows every rendered setting's visible label. On dev, 29 rendered settings in 7 categories cannot be found. They include session summary on quit, Anthropic's 'Sign in with', the Dreams provider, model and region, the Web Search default and backend, Console reasoning and goal-budget limits, and the speech hands-free options. '/discover' opens Dreams rather than the provider's Discover models action, and 'dark mode' finds nothing. The drift guard written to stop this is red. So is the rail category-count guard, which pins 31 categories when there are now 33. Neither blocked anything, because both run only in a workflow that no pull request triggers. This task makes '/' complete again and sends the evidenced queries where users mean them. It also moves both guards into the required pull-request check, so the next drift fails the PR that causes it. It assumes PR #3057 has merged. It leaves the Console Behavior generation-fallback search to TASK-33003.14.

Plan: `backlog/docs/plan-2026-10-14-settings-keyboard-search.md`. Findings closed: A-pm-design-M1, A7, D-shell-design-M3, D-H6 (Settings UX review, 2026-10-14).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every value-editing control rendered in a Settings page can be found through '/' by its visible label, or is listed as a non-setting control with a stated reason, and the search-index drift guard passes
- [ ] #2 '/session summary' and '/summary duration' land on the session summary controls, '/sign in' lands on Sign in with, '/region' on the Dreams region field, and '/search backend' on Web Search's backend choice
- [ ] #3 '/discover' and '/model discovery' open Providers & Models with focus on Discover models, and its disclosure is open; '/discover' still lists Dreams as the next match (this assumes owner decision OD1 is (a) or (b))
- [ ] #4 '/dark mode' and '/light mode' open the Theme picker
- [ ] #5 Every '/' landing and match count that existing tests pin is unchanged
- [ ] #6 The rail category-count guard matches the number of Settings categories and passes
- [ ] #7 The search-index drift guard and the rail category-count guards run in the required pull-request check: a PR that renders a setting with no index row, or changes the number of categories without updating the guard, fails that check
- [ ] #8 The Settings User Guide's description of what '/' finds matches the shipped behaviour
<!-- AC:END -->
