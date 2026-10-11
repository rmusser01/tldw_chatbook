---
id: TASK-2704
title: 'Console Chat Context viewer: Current-tab fold titles leak the role enum repr'
status: Done
assignee: []
created_date: '2026-07-31'
labels: [console, polish, ui]
dependencies: []
---

## Description (the why)

The "Chat Context" viewer's **Current** tab titles each message fold with
its role and status, but the role renders as the raw enum repr:
`[ConsoleMessageRole.USER] complete` / `[ConsoleMessageRole.ASSISTANT]
complete` instead of a user-facing `[user]` / `[assistant]`. Internal
identifiers leaking into a user-facing surface; observed live on dev @
ff435772c during the G1 user-guide capture session (2026-07-31), headless
and real-terminal alike.

## Acceptance Criteria (the what)

- [x] Current-tab fold titles show a human role word (e.g. `[user]
      complete`), never the enum repr.
- [x] Covered by a test that would fail if an enum repr reappears in the
      fold title.


## Closeout audit 2026-10-11

Board-hygiene flip (cohort A, Jun-Jul 2026 status sweep). **Verdict: DONE** — evidence verified against origin/dev (a3c47c6c72).

Fixed by TASK-32336: commit 8c350147e0 'fix(console): human role names in Current Context viewer titles (TASK-32336)'; the older task-2704 footnote was retired by 3f5a25b807 'docs: retire the task-2704 role-form footnote (fixed by TASK-32336)'. Boundary labels now use human role names (Console_Modules/session.py:2567).
