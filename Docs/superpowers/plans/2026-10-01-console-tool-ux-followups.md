# Console tool UX follow-ups implementation plan

> **For agentic workers:** Use the task-specific plans below with red-first verification and independent review. Root integrates the disjoint components and owns CSS rebuilding, QA, task closeout and PR publication.

**Goal:** Finish the four Console tool/approval UX work items identified in this chat.

**Architecture:** Reuse the existing card controls, app-owned interrupt registry, shared pending copy, close-impact model and confirmation dialog. Correct their projections and responsive layout without adding execution authority, storage or dependencies.

**Tech stack:** Python 3.12+, Textual 8.x, existing token-backed TCSS.

**Spec:** TASK-33625.1, TASK-33625.2, TASK-33621.16 and TASK-32367 acceptance criteria in backlog/tasks.

## Global constraints

- Work only in the clean managed worktree on codex/console-tool-ux-followups; preserve the dirty primary checkout and never use git stash.
- Targeted tests only. No full-suite sweep without user opt-in.
- Preserve exact-session/round ownership, fail-closed cancellation, rail preferences and wide-layout behavior.
- Use design tokens and canonical confirmation controls. Edit CSS sources and rebuild generated artifacts.
- Use selected private profiles before imports, actual ChatScreen/compositor checks, and native terminal verification. Record fixture/probe limits honestly.
- Require final-head relevant CI/performance gates, review resolution and latest dev before normal merge. No bypasses.

## Stop: TASK-33625.1

- [x] Inspect latest dev: already Done and merged in PR #2934.
- [x] Re-run its focused real ChatScreen run-control checks during combined qualification; do not duplicate its implementation.

## Approval layout: TASK-33625.2

ADR required: no
ADR path: backlog/decisions/043-console-rail-compact-collapse-yields-to-explicit-toggle.md; backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: Repair existing layout/focus behavior without changing the rail or decision contract. ADR-210 preserves the approval decision surface and lists these safety repairs as prerequisites of its separately staged region migration.

Files: Widgets/Chat_Widgets/chat_approval_card.py; css/components/_agentic_terminal.tcss; Tests/UI/test_console_approval_compact_layout.py; Console approvals guide.

- [x] Mount the actual ChatScreen at 80x24, 90x30 and 100x30, explicitly open Inspect, and show a pending approval. Assert every decision control is contained, painted and hit-testable, including both Deny paths. Watch failure on unchanged dev.
- [x] Use the card's content width and the existing shell compact-height mode to switch the existing controls to a compact layout. Keep Deny before or level with Approve and preserve wide order/layout. Keep control instances, focused disclosure and generation fences.
- [x] Verify keyboard focus, narrow/wide resizing, row reuse, fast Deny and Deny all plus Submit with existing mounted decision tests.
- [x] Rebuild source CSS; verify token/bundle, startup-budget and UI responsiveness guards. Run a real Ask-gated tool through the Console at 80x24 Inspect open/closed and 235x52.

## Close session: TASK-33621.16

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/067-indefinite-human-approval-waits.md; backlog/decisions/082-console-per-chat-private-scratch-space.md; backlog/decisions/085-console-activity-receipts-and-switcher-ownership.md
Reason: Correct existing close consequences and exact-session cleanup.

Files: UI/Console_Modules/session.py; close-impact model; Chat/console_chat_controller.py cancellation method; Tests/UI/test_console_session_tab_close.py; sessions-tabs-workspaces.md.

- [x] Preserve TASK-33621.15's already merged close-runtime repair. Reproduce remaining dialog content, question cleanup and failed-confirmation recovery on actual ChatScreen with real worker rounds.
- [x] Include existing pending kind snapshot in the impact model. Reuse ConfirmationDialog and display title; omit zero consequences, name pending categories and their denial/cancellation effects, retain default Stay focus.
- [x] Report a failed confirmed at-risk close and offer fresh confirmation without automatically retrying. Cancel exact-session unanswered questions through existing revoked-result semantics only after the regression proves the gap.
- [x] Verify Close/Stay, sibling isolation, worker termination and cleared attention, including native terminal coverage. Preserve closed/stale session and idle failure behavior.
- [x] Address Qodo: include worktree-merge consequences with five-kind 80x24 geometry; validate native CLI before profile mutation; reconcile the exact provisional fleet fence before a failed Close can retry, retaining a separate failed-provisional generation if rollback cannot prove success so surviving child usage remains valid for the still-open session.

## Pending display: TASK-32367

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/067-indefinite-human-approval-waits.md; backlog/decisions/195-console-live-tool-call-presentation.md
Reason: Complete the existing kind-aware projection into Inspector.

Files: Chat/console_chat_controller.py count accessor; UI/Screens/chat_screen.py Inspector projection; Chat/console_display_state.py Live work copy; Tests/UI/test_console_pending_interrupt_projection.py; agent-runs-and-tools.md.

- [x] Reproduce Inspector mismatch with real question/approval workers and actual cards. Keep broad has_pending_approval_round semantics and existing activity copy.
- [x] Add pending_round_count(session_id, *, kind='approval') under the existing lock, counting the existing map. The labels agent writes this first, then releases controller ownership to the close agent.
- [x] Use authoritative active-session counts and shared pending copy in Inspector. Retain compatibility fallback only for unavailable legacy seams.
- [x] Verify lone question, question plus approval, approval precedence, queued count2, resolution, sibling session and detach/remount. Reuse actual bridge snapshots rather than synthetic activity stubs.
- [x] Address Qodo: scan visible decision cards in approval-first order independently of queued counts. Verify Alt+A, Inspector and attention-tab routes against a real skill-confirmation owner and queued approval.

## CI timing follow-up

ADR required: no
ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md
Reason: Consolidate existing scenarios without weakening their assertions or private-profile boundaries; apply the existing ADR-094 Close lifetime boundary to a fail-closed recovery refusal.

UI Fast Lane on head70eadf9119 exceeded its existing 20-minute limit at 78%.
Job110655142114 reports that timeout; the repeated full-app cases consumed
144s (compact layout), 131.6s (pending projection) and 389.7s (Close).

- [x] Check the five compact geometry scenarios on one isolated real screen, resizing and replacing the pending round between cases.
- [x] Loop the five pending-kind Close, two title and two rollback scenarios in their respective private children; preserve all assertions and fresh app teardown.
- [x] Preserve the failed-provisional fence and usage semantics, but raise one authored recovery refusal immediately and stop UI reconfirmation; verify the mounted RED case before the fix.
- [x] Restore unrelated multiline CSS and original rationale; use two-space property indentation in the owning sheet to pay the unchanged startup budget, prove identical ordered rules before rebuilding, and run the actual byte-cap guard.
- [x] Verify consolidated tests and source/scenario preservation with independent review; refresh native production receipts after the refusal change.
- [ ] Require the exact-head UI gate to finish within its existing budget; do not change CI settings, remove scenarios or weaken isolation.

## Fresh Qodo short-height follow-up

ADR required: no
ADR path: backlog/decisions/150-design-token-system-and-design-language.md; backlog/decisions/161-component-pattern-library.md
Reason: Repair overflow using the canonical confirmation primitive and existing native scrolling/docked-action pattern; preserve specialized body composition and decision ownership.

- [x] Reproduce the actual 22-row dialog clipped by an80x18 viewport before the fix.
- [x] Reuse VerticalScroll, native action docking and the viewport token; preserve original60-cell frame, IDs, literal text, safe focus and specialized layouts.
- [x] Verify original80x24 cases, short keyboard scrolling and resize identity; qualify17 affected literal/dismissal/callback contracts and9 actual byte-cap/token checks. Add Google-style fixture docs with executable AST unchanged.
- [x] Refresh native approval12 and Close13; independently qualify current pins, actual process exits and unchanged real profiles.
- [ ] Require fresh final-head CI and Qodo resolution before normal merge.

## Integration

- [x] Review component diffs against all acceptance criteria and perform independent correctness review.
- [x] Run focused combined tests once, build/derived guards, preflight and UI latency guardrails; repeat only for new changes/failures.
- [x] Record evidence/limits in Docs/superpowers/qa/2026-10-01-console-tool-ux-followups.md, update guides and task notes, check acceptance criteria and set Done via CLI only after all DoD requirements.
- [x] Rebase onto devab4df9995954 with all six patches unchanged; verify three mounted pending projections, thirteen warm-config safety cases and fresh native approval11/Close12 against current source hashes and private-profile isolation. Record the sanitized integration receipt.
- [ ] Create/attach PR against dev, address verified Qodo findings, rebase when needed and qualify final head, merge normally and verify MERGED.


### Latest-dev logging integration (2026-10-02)

- Rebased onto dev92a95170a540 with feature Python/CSS bytes unchanged; retained
  both independent lesson entries and verified the merged diagnostic inventory.
- Passed19 targeted combined-tree cases, native approval run13 (nine journeys),
  native Close run14 (four worker closes plus80x18 keyboard geometry), and fresh
  preflight. No full sweep, dependency, runtime boundary or CI-setting change.
- ADR required: no. Existing ADR-150/161 and lifetime decisions still govern;
  this integrates already-approved dev logging without adding feature behavior.
- Final published-head CI/Qodo resolution and a verified normal merge remain the
  integration checkpoint. See the linked QA report and receipts.

### Final retained-ID/cancellation review follow-up

Existing ADR-094 ownership and ADR-150/161 confirmation patterns apply; no new
ADR. Preserve permanent native-ID tombstones and use the authored terminal
recovery refusal before voice ownership. Extend the cancellation body selector
to preserve the existing primary border. Both defects have focused RED/GREEN
evidence;30 targeted checks, current native approval14/Close15, preflight and
baseline Ruff pass. The unrelated-error fixture retains its original privacy
assertions and actual retry. See the combined QA/receipt for scope and source
pins. Final published-head CI/Qodo/normal merge remains the PR checkpoint.
