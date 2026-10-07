---
id: TASK-34566
title: Simplify Console single and batch approval controls
status: In Progress
assignee:
  - '@codex'
created_date: '2026-10-06 06:41'
updated_date: '2026-10-06 20:38'
labels: []
dependencies:
  - TASK-34565
documentation:
  - >-
    Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
  - backlog/decisions/221-console-approval-interaction-and-feedback.md
  - Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Make the common permission decision immediate and make batch decisions explicit without overlapping controls.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Single requests expose one-gesture Allow once and Deny, with supported broader grants under More options.
- [x] #2 Counted bulk actions submit exactly the captured calls once; mixed choices submit one complete map with truthful counts and explicit raw-shell review.
- [x] #3 Deny all remains usable for eligible mixed batches, and deliberate selection of a displayed Deny counts independently of its default value.
- [ ] #4 Keyboard focus, Escape, resize, queued gestures and disclosures preserve the correct request without accidental approval or duplicate submission.
- [x] #5 Batch broader choices stay under More options; ineligible bulk approval directs individual review while preserving Deny all and one complete staged Apply.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 3 Replace overlapping controls with explicit decisions

**Backlog:** TASK-34566. **ADR:** ADR-221/150/161/031. **Consumes:** ApprovalBatchView and owner metadata from Task 2.

**Files:** Create `tldw_chatbook/UI/Console_Modules/approval_controls.py` and `Tests/UI/test_approval_interaction.py`. Modify `Widgets/Chat_Widgets/chat_approval_card.py`, `chat_task_cards.py`, `css/core/_variables.tcss`, `css/build_css.py`, and create `css/features/_console_approvals.tcss`. Move feature-local approval rules from `components/_agentic_terminal.tcss` at their existing cascade position; do not duplicate canonical ds-approval-card definitions. Update the existing action-ownership, compact-layout, batch-geometry and card tests.

**Interfaces:** Define `ApprovalDraft(view: ApprovalBatchView)` with `stage(verdict_key: str, decision: str, *, deliberate: bool) -> bool`, `can_apply() -> bool`, `submit_map() -> dict[str, str]`, and `summary() -> str`. This owns staged UI choices, never authorization. Add optional captured view/revision keyword arguments to `ChatApprovalCard.set_batch`, preserving legacy callers. Keep the existing ApprovalDecided decision map, denial reasons and round_id. All committing controls and disclosures retain their originating generation.

- [ ] **Step 1: Write failing user-interaction tests.** Add single Allow once/Deny one-gesture submission, More options noncommit plus explicit scope commit, counted immediate bulk once/deny, and mixed-stage Apply tests. Assert exact emitted maps and one decision per round. Add `test_reselecting_raw_default_deny_is_deliberate`, `test_programmatic_default_does_not_complete_raw_review`, `test_alt_a_enter_cannot_commit`, and `test_stale_more_options_cannot_change_replacement`. Exercise real clicks/keys and painted controls, not only labels or private handler calls.
- [ ] **Step 2: Implement draft rules.** Ordinary staged choices start Once; raw shell starts displayed Deny with a separate unresolved review flag. A deliberate same-value selection can satisfy that flag. Bulk once exists only when every captured call is eligible, never falls back to another scope, and counts hidden pages. Deny all remains an explicit whole-batch negative action. Unknown tools do not acquire new review floors. Mixed row controls are selections with the exact hint **Choices apply when you press Apply**.
- [ ] **Step 3: Build the compact card.** Single calls show Allow once, Deny and More options; grouped multiple calls use batch grammar. Remove redundant single-call Select/Submit/bulk chrome. Add a focusable request summary as Alt+A's first target; it consumes Enter without committing or bubbling into the screen's Send action. Task 4 makes Details the preferred neutral target once available. Keep card-owned scope disclosure, Escape, optional denial reason and the raw command viewport. Allow once uses the existing primary action treatment; Deny is a neutral choice rather than an error variant. Actual blocked/error outcomes keep their status tokens. A commit locks controls and sets Applying before posting the generation-stamped message.
- [ ] **Step 4: Apply token-backed responsive layout.** Keep actions outside the scrollable request body. Resolve inline versus stacked layout from measured available width and token-resolved control requirements, not an invented global breakpoint. When a complete counted label cannot fit, use **Apply decisions** beside a pinned complete count/scope summary; no ellipsis may conceal consent. Add any necessary feature tokens centrally, rebuild CSS, and preserve hover/focus/disabled states.
- [ ] **Step 5: Verify affected journeys and guards.** Run `Tests/UI/test_approval_interaction.py`, `test_approval_action_ownership.py`, `test_chat_approval_card.py`, `test_console_approval_compact_layout.py`, `test_approval_batch_geometry.py`, `test_console_approval_first_open_render.py`, plus `test_design_token_governance.py`, `test_component_pattern_governance.py`, and `test_css_bundle_sync_guard.py`. Revise old structural expectations only to the approved interaction; keep painted actionability, exact decisions and stale-gesture assertions. Inspect native fixture frames at the specified sizes in both themes.
- [ ] **Step 6: Close and commit.** Update task notes and evidence with exact tested outcomes. Keep existing set_batch pre-mount all-or-nothing behavior and construction-time hiding; do not reintroduce deferred mount hides or zero-delay timers.

ADR required: yes
ADR path: backlog/decisions/221-console-approval-interaction-and-feedback.md
Reason: Existing ADR-221/150/161/031 defines explicit approval interactions, token-backed layout and neutral focus without new permission policy or schema.

Execution: consume the reviewed owner capture foundation while documented Windows dispatch and transport qualifications remain open; no scope/lifetime changes or speculative speed repair.

Final review correction: implement batch scope disclosure with gesture-generation/focus protection and neutral Review individually route; cover actual batch gestures and scope/count preservation.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented explicit single and counted bulk controls, deliberate raw review and exact complete maps, neutral Alt+A focus with Enter consumed, captured action/target/profile/location/scope, and token-governed responsive action layout. Independent review round1 fixes display redaction while preserving canonical originals, captured/reused path-precheck warnings, named broader Apply and complete mixed count/tool/scope summaries, complete distinguishing path/URL/command targets versus explicitly marked parameter excerpts, distinct narrow bulk actions, UTF8 and fenced Escape opener focus. Preserved all stale/replacement/clear/finishing/duplicate and denial/round assertions. Task3-added private artifacts removed from index only; local files retained. Durable QA/report: Docs/superpowers/qa/2026-10-05-console-approval-ux/task-3/task-3-report.md. Targeted interaction20/presentation18/card32/ownership25 and focused captured painted checks pass; bundle/token and owned static checks pass. Exact BASE worker/first-open/governance failures, native timing and actual Windows virtual dispatch remain open. Subsequent reused metadata is inspected via an explicit Home gesture; no production scroll reset or initial-reused visibility claim. ADR-221 with150/161/031 applies. Status remains In Progress pending independent re-review and existing DoD limits.

Functional AC1–3 supported by reviewed captured-view/interaction and integration receipts. Invocation/native visual/full DoD qualifications remain open; status In Progress. Final qualification: Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/README.md; ADR-221.

Final bundled fix4542052fb5..9d8a1bd29a: batch broader scopes behind More options, neutral Review individually route, bounded complete-identifier preview with explicit omissions and complete captured target Details. Actual80x24 Console clipping reproduction repaired with existing batch-only4/2 viewport tokens and three-action toolbar. Interaction25/Details19/ownership25/compact1/journeys4/budget28/CSS5/token1/refinement1 passed. Independent scoped review all3 addressed, no new material issue; hashes verified. Evidence Docs/superpowers/qa/2026-10-05-console-approval-ux/task-6/final-review.md and final-fix/. Native/browser timing/fullmatrix, actual Windows dispatch and broader baseline failures remain open; In Progress, not Done. ADR-221.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

Originally TASK-34413 in the reviewed approval checkout. Renumbered to TASK-34566 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.
