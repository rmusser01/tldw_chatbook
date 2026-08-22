---
id: TASK-19701
title: 'Change review: remaining git-config push redirects (pushurl, pushInsteadOf)'
status: Done
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
- [x] #1 A decision is recorded for each of `remote.<name>.pushurl` and `url.<other>.pushInsteadOf`: surface the effective push URL, refuse the push, or accept the redirect as normal git behaviour
- [x] #2 If the decision is to surface it, the push confirm modal names the effective destination the push will actually reach, not only the remote's name
- [x] #3 Tests drive a real repository configured with each redirect and assert the chosen behaviour
- [x] #4 The User Guide's git-actions section matches whatever behaviour ships
<!-- AC:END -->

## Implementation Notes

**Decision (AC #1): surface, do not refuse.** Both `remote.<name>.pushurl`
and `url.<other>.pushInsteadOf` are legitimate, widely used git
configuration — fetching over https while pushing over ssh is a standard
corporate setup — so refusing them would break ordinary workflows to guard
against nothing this app is entitled to override. What was wrong was not
the redirect but the silence: the confirm dialog named the remote's ALIAS
and never its destination, so a terminal running `git remote -v` told the
user more than the dialog whose whole job is to state what a button will do
before it is pressed.

**No new git call was needed.** Verified against real git that both
settings are already resolved into `git remote -v`'s (push) line (and
`git remote get-url --push`), and detection parses exactly that line — so
the effective URL was in hand all along:

```
baseline          origin  https://fetch.example/repo.git (push)
+ pushurl         origin  ssh://git@PUSH-TARGET.example/repo.git (push)
+ pushInsteadOf   origin  ssh://git@INSTEADOF.example/repo.git (push)
```

The dialog now reads `pushes to origin (ssh://…)`. The URL is shown for
EVERY push, redirected or not, so the disclosure is ordinary copy rather
than an alarm that appears only in the unusual case — a warning that fires
only when something is odd trains people to fear the odd case; a line that
always states the destination just makes the dialog honest.

**Tests (AC #3)** drive real repositories configured each way, plus a
control asserting the plain destination is named with no redirect present;
mutation-proven (dropping the URL from the label fails all three).

**Files:** `tldw_chatbook/UI/Screens/change_review_screen.py`,
`Docs/User_Guide/console/agent-runs-and-tools.md`,
`Tests/UI/test_change_review_push_ui.py`.
