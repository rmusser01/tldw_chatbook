---
id: TASK-32107
title: >-
  Library hand-off consistency: Conversations 'Open in Console' is disabled when
  ineligible while the rail's Use-in-Console stays blocked-but-pressable
  (TASK-716)
status: In Progress
assignee: []
created_date: '2026-09-08 22:43'
updated_date: '2026-09-11 07:02'
labels:
  - library
  - console
  - ux
  - design-decision
  - critique-8
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
task-32056 (PR #2523) disables the conversation reader's 'Open in Console' for an unlinked conversation and offers 'Link to workspace' beside it, which its AC required; the rail's `#library-use-in-console` deliberately stays pressable-with-reason (TASK-716). Two blocked-action grammars now coexist in Library; the critique-8 improvement list asked for one 'Send to Console' verb and placement. Rider from the critique-8 fix wave reviews (plan Docs/superpowers/plans/2026-09-08-library-crit8-wave.md; wave PRs #2519 #2523 #2524 #2525 #2528 #2531).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 A recorded decision picks one blocked-action grammar for Console hand-offs in Library
- [ ] #2 The other surface is aligned to it, or the difference is documented with its reason
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Failing tests for link-on-use, Undo, and a non-linkable refusal.\n2. Reader: a link-resolvable block no longer disables Use as source; receipt + Undo ride the metadata seam.\n3. Screen: link first then proceed; Undo mirrors the link.\n4. Record the user decision under ## Decision; update the 32101 pins to the new grammar.
<!-- SECTION:PLAN:END -->

## Decision

Decided by the user, 2026-09-11 (critique-10 fix wave, branch `fix/library-crit10-layout`).

**Link-on-use.** Pressing "Use as source" on a conversation whose only block
is one a link can resolve (`not_in_active_workspace`, `cross_workspace`)
links it into the active workspace and proceeds, in one step, with a
"✓ linked · <workspace>" receipt and an "Undo link" beside it. The separate
"Link to workspace" button stays for membership without a hand-off.

What the gate protects, and keeps protecting: workspace membership decides
which items a Console turn is allowed to read, so a hand-off that widened it
silently would change what the model can see without anyone saying so — which
is why the link is a visible, reversible act with its own receipt rather than
an implicit side effect, and why `no_active_workspace` and the aggregate
`LIBRARY_GENERIC_WORKSPACE_BLOCK` fallback still refuse exactly as they do
today.

The rail's `#library-use-in-console` keeps TASK-716's pressable-with-reason
grammar: it acts on a SET whose members may be blocked for different reasons,
so there is no single link that would unblock it.
