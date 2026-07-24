---
id: TASK-520
title: >-
  Console branching: stronger legacy-flat fingerprint for resume root-chaining
  (all-USER-legacy phantom counter)
status: To Do
assignee: []
created_date: '2026-07-24'
labels:
  - console
  - chat
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`ConsoleChatStore._chain_legacy_flat_roots` (Phase A, amended in Phase B PR #811) chains multiple root-level threads into a linear spine on resume only when the root set is role-MIXED, because a genuine Phase-B root fork (edit-&-resend of the conversation's first user message) produces an all-USER root set that must NOT be chained. Known non-airtight edge, documented in the method's docstring: a DEGENERATE legacy conversation whose 2+ user turns each got NO assistant reply (reachable in the flat era via repeated failed/blocked sends) loads as all-USER roots and is wrongly left un-chained — it resumes showing only the last user message plus a phantom sibling counter. Non-data-loss (all messages stay reachable via swipe), and the two shapes are provably indistinguishable from the persisted tree alone, but a stronger fingerprint can close most of the gap: gate the legacy-chain decision on "the conversation contains at least one NULL-parent ASSISTANT row" (the true legacy signature — legacy wrote every row parentless, and any conversation with a reply then has a parentless assistant) instead of mere role-mixing. Note the flat-prefix case (legacy prefix + post-feature branched continuation) must keep chaining; see `Tests/UI/test_console_resume_active_path.py` for the covering tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A degenerate all-USER legacy conversation (multiple parentless user rows, no replies) resumes showing the full sequence in order with no phantom sibling counter
- [ ] #2 A genuine first-message edit-&-resend root fork is still left un-chained (both branches navigable)
- [ ] #3 Existing legacy-flat and flat-prefix chaining tests still pass
<!-- AC:END -->
