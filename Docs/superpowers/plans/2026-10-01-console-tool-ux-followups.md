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
Reason: Repair existing layout/focus behavior without changing the rail or decision contract.

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
- [ ] Address Qodo: include worktree-merge consequences with five-kind 80x24 geometry; validate native CLI before profile mutation; reconcile the exact provisional fleet fence before a failed Close can retry, retaining a separate failed-provisional generation if rollback cannot prove success so surviving child usage remains valid for the still-open session.

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

## Integration

- [x] Review component diffs against all acceptance criteria and perform independent correctness review.
- [x] Run focused combined tests once, build/derived guards, preflight and UI latency guardrails; repeat only for new changes/failures.
- [x] Record evidence/limits in Docs/superpowers/qa/2026-10-01-console-tool-ux-followups.md, update guides and task notes, check acceptance criteria and set Done via CLI only after all DoD requirements.
- [ ] Create/attach PR against dev, address verified Qodo findings, rebase when needed and qualify final head, merge normally and verify MERGED.
