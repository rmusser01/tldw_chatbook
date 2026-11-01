---
id: TASK-35310
title: Guard Console sends while a conversation resume activation is still in flight
status: In Progress
assignee:
- Robert
created_date: 2026-10-18 00:00
labels:
- console
- send-admission
- uat
priority: high
dependencies: []
updated_date: 2026-10-31 20:34
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console UAT 2026-10-17 (`qa/console-uat-2026-10-17/`, origin/dev `6c0bde2437`) round 11: a
Send pressed shortly after clicking a workspace-rail conversation row was **silently dropped** —
no provider call, no DB write, no transcript turn, no user feedback. With a ~1.5 s settle the
send works.

Cause: `open_console_workspace_conversation`'s slow path occupies the screen pump across
multiple awaits during which `active_session_id` *and* the composer binding still name the OLD
session, so every switch-progress guard sees a consistent OLD state and waves the Send through,
pinned to OLD. The decoupled send task then dies in branches that return without any notice
(`chat_screen.py` pending-token mismatches, `send_acknowledgement.py` Enter-guard drop) or in
received-intent custody whose failure only logs. `refuse_send_from_unbound_composer` cannot
fire because bound == active == OLD throughout the window.

Governance: ADR-098 ("boundary-racing drafts never stranded or duplicated"), ADR-230, ADR-094;
the refusal UX follows the shipped TASK-33622.7 tab-switch guard.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- SECTION:AC:BEGIN -->
- [x] Enter/Send during the resume-activation window shows the existing switching-chats refusal and dispatches nothing
- [x] No silent drop remains on this path: the pending-token and Enter-guard branches notify the user
- [x] After the resume settles, Enter sends into the newly activated conversation
- [x] A `console_send_stage` breadcrumb records the switch-in-flight refusal
- [ ] Existing tab-switch / send-guard / acknowledgement tests stay green
<!-- SECTION:AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. RED test in `Tests/UI/test_console_resume_send_guard.py` cloning the Gate pattern from
   `test_console_tab_switch_binds_composer.py`: hold `load_console_conversation_tree`
   mid-resume, deliver Enter with a draft, assert refusal + no dispatch.
2. Set a conversation-activation-pending flag over the slow path of
   `open_console_workspace_conversation` (try/finally covering all rollbacks).
3. `refuse_send_from_unbound_composer` refuses while the flag is set (existing copy).
4. Louden the silent branches; add the send-stage breadcrumb.
5. GREEN + adjacent suites.
PR3173 repair (2026-10-31): preserve current dev repaint/received-intent custody while reconciling the resume guard; verify held hydration and settled sends with targeted tests; inspect actual diagnostic changes and regenerate affected inventory/census pins. ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md. Reason: routine fixes within existing owners. Owner waived size limits for this repair; retain measured pins and document any adjustment. Independent requesting-code-review replaces Qodo for this authorized backlog cleanup; the user later waived CI waiting; independent review and resolved threads still apply.

Review repair plan (2026-10-31): reproduce held hydration through the actual Library handoff and Ctrl+K choice paths, then move the reentrant activation window to their shared hydration operation. Verify refusals, draft preservation, settled dispatch and depth retirement with targeted cases only. ADR required: no. ADR path: backlog/decisions/094-console-turn-lifetime-and-navigation-boundary.md; backlog/decisions/230-console-send-preparation-and-io-ownership.md. Reason: apply the existing admission boundary consistently to all current callers.
<!-- SECTION:PLAN:END -->
## Implementation Notes

- Rebased PR3173 from `fcc6066d51857869c332e9d9779d5fb1098bb8f4` onto
  `57e330e735` without replacing current dev repaint or received-intent ownership.
  The reentrant hydration window visibly refuses Enter/Send and preserves the draft.
- The refusal now emits `console_send_stage` through the existing app-owned monitor
  with only fixed phase/status/operation values (the existing sink schema). The warning remains. Review of the
  inventory statement diff found only the intended warning (+1, 29 to 30); no
  draft, path, secret or private identifier is added to either diagnostic.
  Actual sink regressions also repaired the two original pending-send breadcrumbs:
  their unsupported reason field is now the existing operation field.
- Repaired the regression harness: focus the composer before Enter, restore the
  hydration patch, and await the detached resume before closing native owners.
  The held-resume test is RED on base 57e330e735 and GREEN on this branch.
  All five resume/refusal cases pass against the real metadata sink, including teardown.
- Registered the four resume/send regression nodes in the existing UI census. Size
  pins reflect exact measurements under the owner's 2026-10-31 size waiver;
  no size check is disabled. Existing ADR-094/098/230 apply; no new ADR required.
- Local controls: acknowledgement (9) and retained repaint (2) pass. The four
  `test_send_refuses_and_keeps_the_draft_when_the_composer_is_bound_elsewhere`
  variants fail their pre-send mounted state assertion on BOTH base and branch;
  `test_a_held_enter_never_resends_a_draft_a_spoken_send_took` also times out on
  both. These are existing limits, not an all-green adjacent-suite claim.
- Files: session/workspace/send_acknowledgement/chat_screen, resume tests,
  diagnostic inventory, exact size pins, UI census and Console architecture docs.
  Independent review/publication belongs to the coordinating agent. Status stays
  In Progress until that review; the user waived waiting for CI, not local evidence.

- Independent review found that the rail-only window missed Library and Ctrl+K.
  Both real caller regressions reproduced a missing refusal during held hydration.
  The window now belongs to the shared resume operation, including its first await;
  current claim predicates and activation rollback remain inside that window.

- Review verification: all three mounted caller variants pass with draft preservation,
  no dispatch during hydration and a settled send to the resumed chat; the other four
  resume/diagnostic cases also pass. An initial test-only draft accessor typo was
  corrected before the clean three-case rerun (review-callers-green.log).
- Diagnostic statement review against d36f7cf confirms workspace.py remains at
  54 calls: two re-indented, zero added/removed, no content or privacy change.
  Exact size checks pass at workspace 8766 and controller 29184 lines under the
  existing owner waiver. The existing census node includes all caller parameters.

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Current-dev integration (2026-10-31, base 0d05d27fa1d569f1edebbc2346d43f8c03da9911): preserved current acknowledged-press input capture and every base census entry. Independent integration review found no new production defect, but the old test recorder rejected the current _captured_inputs keyword. All four mounted controls reproduced that exact TypeError. The recorder now accepts the explicit keyword and asserts its captured session and draft equal the dispatched values; no production behavior or original owner/refusal assertion changed. All four mounted cases pass with clean teardown (153.45s), covering rail, Library, Ctrl+K and post-resume delivery. The unchanged current-base product also passed all eight shutdown/provider lifecycle tests and the real mounted background-notification control. No new Ruff findings; diff check passes. All six changed product-file diagnostic owner/sink/path projections match the inventory; current census prefix is unchanged and the four verified nodes make1178 entries. Exact size pins reflect the owner's 2026-10-31 waiver. Hosted CI waiting is waived; final independent review and oldest-first merge sequencing remain required. Native-private evidence receipts: ../test-runtime/current-base-integration-1793478189891008400 and ../test-runtime/resume-recorder-green-1793478564969789600 (JUnit+source manifests). Earlier adjacent exact-base failures remain recorded under TASK-35311; this targeted integration does not claim a full-suite pass.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
