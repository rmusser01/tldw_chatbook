---
id: TASK-572
title: >-
  Console branching: stronger legacy-flat fingerprint for resume root-chaining
  (all-USER-legacy phantom counter)
status: Done
assignee:
  - '@claude'
created_date: '2026-07-24'
labels:
  - console
  - chat
  - tech-debt
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
(Renumbered from task-520 in the PR #803 rework: that ID collided with an unrelated Done task on dev, which keeps the ID per the Done-tasks-never-move rule.)

`ConsoleChatStore._chain_legacy_flat_roots` (Phase A, amended in Phase B PR #811) chains multiple root-level threads into a linear spine on resume only when the root set is role-MIXED, because a genuine Phase-B root fork (edit-&-resend of the conversation's first user message) produces an all-USER root set that must NOT be chained. Known non-airtight edge, documented in the method's docstring: a DEGENERATE legacy conversation whose 2+ user turns each got NO assistant reply (reachable in the flat era via repeated failed/blocked sends) loads as all-USER roots and is wrongly left un-chained — it resumes showing only the last user message plus a phantom sibling counter. Non-data-loss (all messages stay reachable via swipe), and the two shapes are provably indistinguishable from the persisted tree alone, but a stronger fingerprint can close most of the gap: gate the legacy-chain decision on "the conversation contains at least one NULL-parent ASSISTANT row" (the true legacy signature — legacy wrote every row parentless, and any conversation with a reply then has a parentless assistant) instead of mere role-mixing. Note the flat-prefix case (legacy prefix + post-feature branched continuation) must keep chaining; see `Tests/UI/test_console_resume_active_path.py` for the covering tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 A degenerate all-USER legacy conversation (multiple parentless user rows, no replies) resumes showing the full sequence in order with no phantom sibling counter
- [x] #2 A genuine first-message edit-&-resend root fork is still left un-chained (both branches navigable)
- [x] #3 Existing legacy-flat and flat-prefix chaining tests still pass
<!-- AC:END -->

## Implementation Plan

1. Strengthen `_chain_legacy_flat_roots`'s all-USER decision: chain when any
   root is an ASSISTANT row (true legacy signature -- a genuine tree never
   roots an assistant) OR when every root is childless (degenerate legacy;
   a genuine first-message fork always carries at least one reply subtree).
2. TDD at the real-DB resume level: degenerate all-USER legacy resumes as
   the full ordered sequence with no phantom counter; genuine fork and
   flat-prefix behaviors pinned by the existing suite.

## Implementation Notes

- Signal changed from role-homogeneity alone to
  `root_has_assistant OR all_roots_childless`; flat-prefix and mixed-role
  legacy keep chaining exactly as before, genuine all-USER forks with any
  subtree stay un-chained.
- Documented residual edge (narrower than the closed gap): a genuine
  first-message fork whose BOTH branches are childless (anchor never
  replied AND the resent reply never persisted) now chains -- non-data-loss,
  provably indistinguishable from degenerate legacy from the tree alone.
- Files: `tldw_chatbook/Chat/console_chat_store.py`,
  `Tests/UI/test_console_resume_active_path.py`.
