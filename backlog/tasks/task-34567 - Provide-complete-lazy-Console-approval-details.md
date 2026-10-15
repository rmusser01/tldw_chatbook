---
id: TASK-34567
title: Provide complete lazy Console approval details
status: In Progress
assignee:
- '@codex'
created_date: 2026-10-06 06:41
updated_date: 2026-10-15 22:39
labels: []
dependencies:
- TASK-34565
- TASK-34566
documentation:
- Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md
- backlog/decisions/235-console-approval-interaction-and-feedback.md
- Docs/superpowers/plans/2026-10-05-console-approval-ux-and-responsiveness.md
priority: high
type: enhancement
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let users inspect complete captured arguments and distinguish long targets while keeping large requests responsive.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Details exposes complete redacted captured arguments in bounded pages without fresh file reads or network calls.
- [ ] #2 Collapsed requests do not eagerly build or mount full large payloads; late pages cannot update a replaced or resolved request.
- [ ] #3 Raw shell retains its complete command and required primary warnings within the existing command limit.
- [ ] #4 Long paths, large write bodies, many targets and maximum-size commands remain inspectable with stable focus and scroll at supported sizes.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
### Task 4 Add complete arguments without eager UI work

**Backlog:** TASK-34567. **ADR:** ADR-235/090/150. **Consumes:** ApprovalRowView and its semantic revision, Task 3's neutral focus target and disclosure lifecycle.

**Files:** Create `tldw_chatbook/UI/Console_Modules/approval_details.py`, `Tests/UI/test_approval_details.py`. Modify `Widgets/Chat_Widgets/chat_approval_card.py` and `css/features/_console_approvals.tcss`. Extend `Tests/UI/test_approval_argument_budget.py` and `test_approval_row_information_budget.py` where their old summary-only assumptions change.

**Interfaces:** Define frozen `ApprovalDetailsPage(index: int, text: str, has_more: bool)` and `iter_redacted_details(argument_sets: Sequence[Mapping[str, object]], *, page_chars: int = 4096) -> Iterator[ApprovalDetailsPage]`. Concatenated pages represent a JSON array of all captured argument sets after existing redaction. Define `DetailsIdentity = tuple[str, int, int, str]` for round_id, semantic revision, UI generation and verdict_key. Produce `ApprovalDetailsController.open(row: ApprovalRowView, *, round_id: str, revision: int, generation: int) -> None`, `request_page(index: int) -> None`, `deliver_page(identity: DetailsIdentity, page: ApprovalDetailsPage) -> bool`, and `close() -> None`.

Its keyword-only constructor takes `current_identity: Callable[[], DetailsIdentity | None]`, `spawn_worker: Callable[[Callable[[], None]], object]`, `post_page: Callable[[DetailsIdentity, ApprovalDetailsPage], None]`, `paint_page: Callable[[ApprovalDetailsPage], None]`, and `paint_loading: Callable[[], None]`. These are late-binding services. post_page is thread-safe; current_identity and paint services are used only on the UI thread. It stores no removable child widget instance.

- [ ] **Step 1: Write failing Details tests.** Add `test_collapsed_details_does_not_serialize_large_body`, `test_paged_arguments_reconstruct_complete_redacted_capture`, `test_page_from_old_revision_is_ignored`, `test_switching_rows_rejects_late_page`, `test_previous_page_result_cannot_replace_newer_request`, `test_escape_does_not_resolve_round`, and `test_raw_command_remains_primary`. Use a 1 MiB synthetic write body, 1,000 synthetic targets in one argument list, distinguishing long paths, CJK/emoji and existing secret-key redaction. Assert page text is at most 4,096 characters, all legitimate content remains retrievable, and neither original arguments nor permission rules use the redacted text. Establish red with the new file.
- [ ] **Step 2: Implement the pure page iterator.** Serialize captured data only, on demand, using the existing redactor and incremental JSON encoding. Never allocate or render full JSON in set_batch, on resize or on a summary patch. Use literal rendering and a fixed safe Arguments unavailable error, not raw repr or exception text. Keep at most two prepared pages per active row; release them on clear/replacement. More content is labeled explicitly; do not invent a total page count before it is known.
- [ ] **Step 3: Connect a cooperative worker.** Opening Details paints Preparing details immediately, then starts one exclusive worker. Close, row switch or replacement sets its cancellation flag; worker cancellation alone cannot stop an already running thread. Workers check the cooperative flag and post captured identities. UI-thread deliver_page rechecks round, revision, generation, verdict_key and the latest requested page index before painting. Another row or an earlier page request cannot replace the visible content. No worker reads UI identity, store, filesystem, active workspace or a remote endpoint.
- [ ] **Step 4: Keep navigation and review stable.** Render the current bounded page in a read-only viewport with Previous/Next controls and a continuation label. Preserve the committing actions outside that viewport. The raw command uses its existing dedicated complete-command viewport and MAX_RAW_COMMAND_BYTES rather than this generic JSON disclosure. Escape returns to an eligible neutral opener without clearing staged choices or pausing the existing deadline.
- [ ] **Step 5: Verify and commit.** Run the new Details tests and affected argument-budget, row-information, action-ownership and compact-layout cases plus token/bundle guards. Characterize large requests with Task 1's recorder, distinguishing first page preparation from UI feedback. Record source hashes, exact frames and memory/cancellation observations; close this task only after its complete-data and stale-page ACs hold.

ADR required: yes
ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md
Reason: Existing ADR235/090/150 governs snapshot-only lazy redacted details; no new authority or persistence.

Current-dev PR3036 integration: follow Docs/superpowers/plans/2026-10-11-pr3036-current-dev-integration.md. ADR required: yes. ADR path: backlog/decisions/235-console-approval-interaction-and-feedback.md. Reason: preserve approved interaction/noncontrolling observations and shared credential masking under existing human-decision authority. Source history/checked qualification is retained above; current acceptance is reopened for actual sequential-base invocation/privacy/UI/native/browser qualification. No cap/clock/guard relaxation or headless-as-native claim.
2026-10-15 actual-base integration: user deferred #3023; continue on dev1c72cfd48f2bd903093ce6b032684de74c511919 without private #3023 changes. Existing approval decision remaps to ADR235 after a live ref/worktree census (original source221/private229 provenance retained); no new architecture or task ID. Retain this task plan/AC/history and full original approval/privacy/currentness/geometry/timing requirements. First establish the supported private launch, passing original control and reviewed content-free measurement contract before functional edits; Root owns publication and final external qualification.
Review repair: hold real Details preparation and queue a page before actual card removal; RED must show private cancellation/identity remains live. Synchronously close/fence existing controller at actual card unmount, drop cache/capture/opener references without DOM queries/focus. Persist only bounded nonauthoritative reading position through existing pending-round presentation state; never retain pages/bodies/callbacks. Test actual removal/recreation, late delivery rejection and exact identity mismatch. ADR required:no new; existing ADR235/custody.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented snapshot-only lazy redacted Details with4096-character pages, two-page cache, cooperative cancellation and round/revision/generation/verdict-key/latest-page fences. Neutral Details focus and Escape preserve decisions/deadline; maximum raw command retains its primary viewport. Tokens/bundle retained. ADR235/090/150.
Verification: Details15, argument28, row12, ownership25, interaction21, real Console4-variant journey1, tokens3, bundle5; owned Ruff/format and diff-check pass. Component preparation receipt is unqualified for native latency.
Evidence: Docs/superpowers/qa/2026-10-05-console-approval-ux/task-4/README.md and component.json. Changed card, Console_Modules/approval_details.py, feature CSS/bundle, targeted tests, private launcher and testing lesson.
Task remains In Progress: full live matrix/native-browser qualification and independent review open; preserve existing Windows dispatch/governance limitations. Cache misses replay from capture; no authority, storage or policy changes.

Independent review found eager generic captured-body formatting and unowned navigation. Actual-card RED3 then Details18/ownership25/interaction21/Console1 GREEN. Captured rows skip eager generic formatting; nav events capture row identity and page index. Source hashes refreshed. Re-review pending; native/full matrix limits retained.
Current1c72 integration: bounded lazy Details/cache/cancellation and identity fences use existing collaborators. Actual admitted argv/OAuth privacy regressions fail before repairs and pass through existing shared redactors afterward; captured matching arguments remain untouched. Required UI census now enrolls privacy plus stale-page/stale-options nodes without dropping190 existing entries. Current core540passed84.15s; browser/native physical qualification stays open. ADR235, In Progress.
Review repair: actual card unmount cooperatively closes Details, rejects queued/held pages and drops capture/cache/opener references without retiring-DOM query or focus restoration. Existing retained round payload stores only scalar choices/review/disclosure/key/page/scroll for exact session/run/round/revision; actual host unpark/replacement retires state. Real provider/controller parked head through shipping ChatTaskCards clear/remove/recreate covers broader choice plus deliberate raw Deny and Details page/scroll, scalar-only retained state, fresh gesture fences and identity mismatch. Final affected141PASS/0guards. Existing ADR235; no new durable store/framework. Complete native/browser/timing criteria remain open; no Done.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Renumbering provenance

Originally TASK-34414 in the reviewed approval checkout. Renumbered to TASK-34567 during PR integration onto current dev because older unrelated TASK-34411/34412 already landed. The six approval records moved together to preserve dependency order; original verification hashes/commit references retain their historical context.
