---
id: TASK-19701
title: 'Change review: remaining git-config push redirects (pushurl, pushInsteadOf)'
status: To Do
assignee: []
created_date: '2026-08-21'
labels:
  - console
  - change-review
  - git
dependencies:
  - TASK-16801
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
TASK-16801 hardened change review's push against repository-supplied config
that redirects or escalates the operation: option-shaped remote and branch
names are refused, and both push forms now pass an explicit fully-qualified
refspec, which defeats `remote.<name>.push`, `remote.<name>.mirror` and
`push.default=matching`.

Two sibling redirects were deliberately left in place after review, and this
task records them rather than leaving the decision undocumented:
`remote.<name>.pushurl` and `url.<other>.pushInsteadOf` both send the push to
a different URL than the fetch URL.

They were judged materially weaker than the fixed vectors: no ref is
destroyed and no history rewritten — the push simply lands somewhere else —
and `git remote -v`'s push line already reflects the redirect, so a user
inspecting the repository in a terminal sees it. The reason they are not
simply dismissed is that the confirm modal shows only the remote NAME, never
a URL, so the app's own surface does not reveal the redirect even though a
terminal would.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A decision is recorded for each of `remote.<name>.pushurl` and `url.<other>.pushInsteadOf`: surface the effective push URL, refuse the push, or accept the redirect as normal git behaviour
- [ ] #2 If the decision is to surface it, the push confirm modal names the effective destination the push will actually reach, not only the remote's name
- [ ] #3 Tests drive a real repository configured with each redirect and assert the chosen behaviour
- [ ] #4 The User Guide's git-actions section matches whatever behaviour ships
<!-- AC:END -->
