# Focused CI review: Console acknowledgement handoff

Reviewed immutable PR #3283 head `cc907deb187f4e3de99e9f1bab2f73983ac1870a` with the requesting-code-review template, systematic debugging and Ponytail. Product/test/task references below are from `git show` at that head. No App or pytest execution, source/index/branch changes, remote actions or external messages were performed. The only written artifact is this report.

## Evidence and strengths

- The saved UI lane log `current-head-cc907-watch/logs/122678943800.log` reports one failure and 156 passes: `test_enter_paints_a_sending_row_before_admission_and_validation`, line 375, `gaps == [5, 6, 7, 8]`, against the unchanged `len(gaps) <= 1` bound. The other lanes' success is coordinator-supplied context, not independently recertified here.
- Acknowledgement ownership is appropriately separate from durable acceptance. `send_acknowledgement.py:207–240` removes the synthetic row only when this token was admitted and a USER id outside its custody-time snapshot exists **in the same projection**. `wiring.py:608–616` takes that snapshot before `accept_turn` and marks the same dispatch admitted. Token, sibling-row, refusal and custody tests cover these distinctions.
- `chat_screen.py:17013–17056` passes that exact replacement projection to the transcript. The source does not support blaming this failure on releasing the acknowledgement merely because admission started.

## Critical

None identified within this read-only scope.

## Important

**I1 — The acknowledgement-to-echo replacement has an unbounded visible teardown interval.** `Widgets/Console/console_transcript.py:7243–7284` removes the synthetic row's keys and awaits all native removal pumps; the echo's new keys are not built and mounted until `7286–7340`. `refresh_messages:4921–4928` serializes reconciliation but does not batch its repaint. Native Textual teardown awaits descendants and Unmount; intervening screen refreshes can therefore paint neither message. The test's own `366–375` comment and TASK-33620.5's “Known residue” already acknowledge this mechanism. Their observed one-frame maximum is empirical, not a framework guarantee. The CI four-sample interval is consistent with this product race; its exact causal timeline remains unrecorded.

Smallest candidate fix: preserve `_refresh_lock`, `finish_removal` shielding, row identities, ownership checks and order logic; use the installed Textual `App.batch_update()` only around the awaited `_reconcile_rows` operation. This is the native repaint transaction for remove/mount: Textual 8.2.8 `app.py:1061–1078` and `screen.py:1175,1239` suppress repaint while batched; `Widget.batch()` documents remove-then-mount as its use case (`widget.py:4415–4430`). Since the transcript already holds its refresh lock, a second widget lock is unnecessary. This remains a candidate pending the coordinator's native RED/GREEN witness; do not declare it fixed from source inspection. Do not raise the frame allowance, add sleeps/retries, slow polling or change acknowledgement/store ids. Generic mount-before-remove also needs extra treatment for same-key replacement's duplicate widget ids and is a larger initial change.

## Minor

**M1 — The recorder observes compositor commits, not literal terminal writes or elapsed blank time.** `Tests/UI/test_console_send_acknowledgement.py:76–115` re-renders compositor strips after each non-None `_display` invocation, while default `run_test()` is headless. Installed Textual 8.2.8 `app.py:2133–2159,3829–3838` confirms no terminal output in that mode. Active-screen refreshes do call `_display`, so this does not invalidate the continuity gate or justify discarding the failure. However four list indexes do not prove four refresh periods or a millisecond duration; the assertion collects all missing-row samples, not necessarily a consecutive duration. Add small failure-only records of timestamps, current-screen/update identity, acknowledgement activity, supplied message ids, row attachment/regions and reconcile phase. Preserve the existing assertion. A batched regression must inspect the **last committed frame**, not call `render_strips()` directly while repaint is intentionally suspended.

## Small deterministic native witness

1. Reuse the private-profile, production-CSS real `ConsoleTranscript` harness and the held-native-Unmount pattern from `test_console_transcript_poll_stop_removal.py`. Use a real `ConsoleSendAcknowledgement`: begin, project/mount its pending USER row, bind custody/admission, then project a real USER echo with a different id. Assert the supplied replacement projection already contains the exact echo and no synthetic row.
2. Hold only the acknowledged message row's real `Unmount` with Events. That point follows its descendant teardown. Trigger `set_messages` plus awaited `refresh_messages` from `set_interval`, matching the Console poll. Do not replace the reconciler or mock removal/mount completion. A one-shot timer is not equivalent here (TASK-34982's recorded lesson).
3. Request real screen refreshes (`screen.refresh(layout=True)`) while the native exit is held, giving the native update timer multiple opportunities to publish the intermediate state. Poll with bounded `asyncio.sleep`, not `pilot.pause`, since Pilot waits on the deliberately held pump. Make the original implementation commit more than one missing-message frame; record the boundary events and last committed frame. Do not wait for new commits on the fixed arm, where batching should suppress them. Release in `finally`. The candidate must retain the acknowledgement frame until the completed echo replacement can commit, then show exactly one echo and no ack. Retain the original end-to-end `<=1` assertion unchanged.
4. Keep the existing timer-cancellation/removal regression, and add only the necessary batch-cleanup control: cancellation/teardown must not leave the app's repaint batch open or revive stale rows. Re-run focused reorder/session-switch and stopped-poll controls if shared reconciliation is changed. This witness proves the shared rendering mechanism; it does not certify every send producer.

## Existing ownership and ADRs

- **TASK-33620.5 (Done)** owns this acknowledgement and explicitly records the remove/mount residue; attach the regression/follow-up to that existing ownership rather than recreating the original send-feedback task.
- **TASK-33620.15 (Done)** owns detached admission and Enter/Send/Workbench routing. Its broader performance claims do not establish this handoff's continuity.
- **TASK-34982 (Done)** owns cancellation-safe row removal; preserve `finish_removal` and its exact native timer test.
- **TASK-15453 (Done)** owns avoiding redundant row moves; preserve its order and steady-state behavior. **TASK-32522 (Done)** is precedent for compositor continuity verification of disappearing thinking content, not ownership of this distinct USER-row swap.
- ADR required: **no new ADR** for a routine rendering transaction preserving current boundaries. ADR-094 and accepted ADR-230 keep runtime/store authority separate from the disposable view. ADR-231's wider polling proposal remains Proposed outside its bounded accepted amendment; this failure does not authorize polling redesign.

## Declined boundaries and uncertainty

- No certification of incoming dev `798f8fca75a7acba56da5f523329efa52c4a78f0` or its future composition with this PR; all repository reads are pinned to `cc907deb18`.
- No attribution to PR #3283's settings changes without an original-base comparison. This source already documents the defect; a passing rerun alone cannot disprove it.
- No provider/admission latency, live terminal visibility, paid-call/deduplication, durable acceptance, received-intent or full-suite verdict. The compatibility send harness differs from the stock received-intent producer (the repository's TASK-33621.20 lesson); shared transcript coverage is not whole-send certification.
- The CI log has no per-gap phase/identity records. Native local RED/GREEN and batch duration/input responsiveness remain coordinator work. Repeated whole-lane attempts before the small witness would spend substantially more time without isolating the cause.

## Assessment

**Ready to merge: With fixes and focused verification.** Keep the CI gate's threshold intact. First establish the native replacement RED and candidate GREEN, retain cancellation/order behavior, then qualify the integrated current head through the original failing test and CI. Setup/privacy refusals are neither product RED evidence nor permission to weaken those guards.
