---
id: TASK-35678
title: 'Share Settings action availability across shortcuts buttons and help'
status: Done
assignee:
  - '@codex'
created_date: '2026-10-19'
labels:
  - settings
  - textual-patterns
  - maintainability
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Priority 3 of the Textual-book follow-ups: reduce duplicated category-action eligibility in Settings without changing established interaction behavior. SettingsScreen declares the s/r/t, RAG profile and Personal Context bindings, separately derives _category_footer_shortcuts, and guards actions by category/focus. Console already uses check_action() for contextual actions.

This is a maintainability improvement, not a claim that today's footer advertises broken actions: TASK-1340 deliberately kept some unavailable shortcuts bound so they can explain what to do. Share the existing availability predicates among relevant actions, buttons and custom footer/F1 help. Use native check_action() only where it preserves that behavior; do not blindly hide or disable every unavailable action. The custom footer requires its own projection and does not gain automatic native Footer behavior.

Scope: existing Settings category Save/Revert/Test, RAG profile and Personal Context actions. Keep text-entry key consumption, explicit button/direct-call semantics, confirmation guards and mutation-time revalidation. Avoid a generic action framework, global shortcut redesign, new caching or changes to Console action gating.

Book basis: chapter 8 pp. 161-162 (dynamic actions). Existing precedent: ChatScreen.check_action.

ADR required: no new ADR.
ADR path: backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/033-settings-commit-models-three-honestly-labeled.md.
Reason: behavior-preserving consolidation of the accepted action/help and settings commit contracts.

Deduplication: TASK-1340, TASK-1580, TASK-1714 and TASK-16265 cover existing footer/help and feedback improvements; retain those contracts, do not reopen them.

Other recommendations from this book review already have owners and are explicitly outside this task: message-pump/blocking-I/O work belongs to TASK-2834 and TASK-32804; worker/post-await checking to TASK-32800.4; held-attachment keyboard responsiveness is fixed by TASK-35614, with measured follow-up under TASK-35645. Console polling/reconciliation belongs to TASK-34563.33 and TASK-34563.34 under ADR-231. That ADR's general proposal remains Proposed; this filing neither authorizes a polling rewrite nor creates a competing owner.

### Related work and decisions

- [TASK-1340](../tasks/task-1340%20-%20Settings-honest-footer-hints-and-key-feedback.md): existing owner or precedent described above.
- [TASK-1580](../tasks/task-1580%20-%20Settings-footer-s-r-hints-gated-to-draft-model-categories.md): existing owner or precedent described above.
- [TASK-1714](../tasks/task-1714%20-%20Settings-copy-action-verbs%2C-honest-dirty-banner%2C-spelled-stats.md): existing owner or precedent described above.
- [TASK-16265](../tasks/task-16265%20-%20Reconcile-Settings-help-and-theme-interaction-contracts.md): existing owner or precedent described above.
- [TASK-2834](../tasks/task-2834%20-%20Guard-against-blocking-IO-reachable-from-a-Textual-message-handler.md): existing owner or precedent described above.
- [TASK-32804](../tasks/task-32804%20-%20Work-stream-synchronous-work-on-the-Textual-event-loop.md): existing owner or precedent described above.
- [TASK-32800.4](../tasks/task-32800.4%20-%20Add-a-guard-for-the-Textual-worker-and-post-await-DOM-contract.md): existing owner or precedent described above.
- [TASK-35614 / TASK-35645 — PR #3166](https://github.com/rmusser01/tldw_chatbook/pull/3166): existing keyboard-responsiveness fix and measured native-attachment follow-up. Their task files exist in the primary checkout but are not on this branch's dev base; link their existing PR without copying unrelated work.
- [TASK-34563.33](../tasks/task-34563.33%20-%20Plan-Console-polling-and-full-state-reconciliation.md): existing owner or precedent described above.
- [TASK-34563.34](../tasks/task-34563.34%20-%20Verify-Console-polling-reconciliation-boundaries.md): existing owner or precedent described above.
- [031-tui-keybinding-and-footer-hint-conventions](../decisions/031-tui-keybinding-and-footer-hint-conventions.md)
- [033-settings-commit-models-three-honestly-labeled](../decisions/033-settings-commit-models-three-honestly-labeled.md)
- [231-console-polling-and-full-state-reconciliation](../decisions/231-console-polling-and-full-state-reconciliation.md)
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A before/after category and focus-state matrix records action execution, guidance-only behavior, button availability and footer/F1 wording for the scoped action families.
- [x] #2 Shared eligibility rules drive the relevant action, button and custom help/footer decisions without duplicated category lists that can disagree; any intentional distinction between advertised, executable and guidance-only behavior is explicit.
- [x] #3 Existing guidance-only shortcuts still provide their guidance, focused text fields still receive printable keys, button/direct-call paths retain their intended focus behavior, and destructive actions keep their confirmations.
- [x] #4 Switching category or changing relevant draft/busy state refreshes availability and help without stale results or configuration/database reads newly added to key dispatch or check_action.
- [x] #5 Targeted mounted tests retain the existing footer/help contracts and cover keyboard/button equivalence, supported and unsupported categories, text focus and narrow-width discoverability; a private-profile live keyboard check confirms behavior.
- [x] #6 No Console polling, worker ownership, screen-wide recomposition or global binding changes are included; implementation notes link the existing ADRs and record any deliberately retained duplicate presentation needed for different semantics.
<!-- AC:END -->


## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: no
ADR path: backlog/decisions/033-settings-commit-models-three-honestly-labeled.md; backlog/decisions/031-tui-keybinding-and-footer-hint-conventions.md; backlog/decisions/161-component-pattern-library.md
Reason: bounded changes under the existing Settings and approved screen-decomposition contracts; persistence and service ownership stay with their existing owners.
1. Record the current category/focus/action/footer matrix.
2. Share pure category eligibility for save/revert/test, RAG and profile actions across handlers, footer/help and relevant controls.
3. Preserve guidance-only and focused-text semantics and mutation-time safety gates.
4. Verify matrix, keyboard/button flows, narrow layouts and scoped static checks.
5. Qualify the rebased PR within the existing UI job budget: use hosted per-occurrence timings to rebalance census slots if the new journey reaches the 20-minute limit; retain every entry and assertion, the 372-entry floor, shard count and timeouts (existing CI workflow policy; ADR-103).
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Shared pure category support across Save/Revert/Test dispatch, guided buttons, footer/F1 and RAG/Personal Context action guards. Guidance-only bindings remain active; native check_action was deliberately not added because it would suppress the existing explanatory paths. The screen and category-owned widgets retain mutation-time validation, dirty/busy checks, text-focus behavior and confirmation ownership. Theme and Personal Context keep their context-specific hints. Current dev's new Tools & MCP live-mode footer and verb branches remain before generic guards, unchanged under ADR-240.

The before/after category/focus matrix and targeted mounted/keyboard evidence are in [qualification](../../Docs/superpowers/qa/2026-10-19-textual-settings-patterns.md). An existing Theme screen-binding test failed on the original source because focus remained in ThemeOptionList, whose t means Try; its setup now explicitly releases that focus, preserving all assertions. Affected mounted tests use the required private-profile process. Independent initial and post-rebase reviews found no remaining defects. New regression entries were appended to the PR UI census without moving earlier entries.

ADR required: no new ADR; [ADR-031](../decisions/031-tui-keybinding-and-footer-hint-conventions.md) and [ADR-033](../decisions/033-settings-commit-models-three-honestly-labeled.md) remain governing. Modified the Settings screen, action-availability and affected footer tests, architecture docs and test census. No worker, polling, global bindings or screen-wide recomposition changes were included. Existing performance and polling owners remain linked above.


Current-dev qualification (2026-10-20): a fresh requested Superpowers review found no issues on 1bbc145ffc..3c2b0e5601; 17 targeted Settings cases and two size checks passed. Seven additional local footer fixture failures reproduced identically on unchanged dev. Hosted UI shard 9 then exhausted its existing 20-minute budget without an assertion failure. Qualification therefore swaps three measured census pairs across shards 5/6/9, including four upstream entries as an explicit exception to initial append-only placement. All 372 entries and assertions remain; workflow limits and product source are unchanged. The 78 census/count checks and real census validation pass. The qualification report records timestamps, upstream comparison and pending hosted confirmation. This follows existing [ADR-103](../decisions/103-fast-pr-lane-and-required-gate-aggregation.md), with no new ADR. Backlog CLI is unavailable locally; this plan/note update preserves the task's existing metadata and completed criteria.
<!-- SECTION:NOTES:END -->
