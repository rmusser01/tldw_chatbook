# SDD ledger — plan: Docs/superpowers/plans/2026-08-26-console-thinking-blocks-ui-settings.md

Setup: isolated worktree `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.worktrees/console-thinking-blocks`, branch `codex/console-thinking-blocks`.

Dependencies: TASK-18932.1 complete at `9906b4d4bd`; TASK-18932.2 complete at `7ec7fbc9dc`. This child consumes the reviewed envelope, live capture, replay-policy, and owner-group contracts.

Impeccable: Operate mode; incumbent Neon Workbench vocabulary; required craft floor loaded before UI implementation. Ponytail full: reuse existing disclosure, expansion, Settings, and Context seams.

Task 1: BASE `7ec7fbc9dc`; pure trusted thinking-activity projection and turn ordering only.

Task 1 initial implementation: commit `236b8a448d`; 34 focused and 171 nearest
presentation tests passed; Ruff and `git diff --check` clean.

Task 1 review fix round 1: reproduced generation identity collision and positional
round inference at RED (`3 failed, 34 deselected`). Added mandatory stable
`generation_id` to the projection/ID contract and explicit session-only
`activity_round_ordinal` ownership to activity rows. GREEN: 38 focused and 175 nearest
presentation tests passed. Task 2 must reuse an installed variant ID, frozen live
generation-attempt fact, or durable restored generation identity; the later activity
producer must stamp exact round ordinals and leave post-run rows unowned.
