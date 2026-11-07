# Current-head CI fixture diagnosis

Published head: b1efc37ae8f481c1df01e2f253d5878ba68e35a4. PR Fast Lane run37229906610/job111518400864.

The first invocation passed1254/skip1. Admission invocation failed six cases/216passed/existingxfail1. All six fail before their polling assertion because `_start_console_view_after_reconciliation` binds `_session.consume_pending_vllm_console_intent`, absent from `_post_reconciliation_admission_screen`'s SimpleNamespace. Its real Session owner implements both vLLM and llama.cpp consumers, and the screen binds both methods; this fixture supplies neither. The two poll test bodies and shared fixture existed before the Task7 owner move. Production fallback/owner changes are unnecessary.

Six failed nodes are idle_reconciled_view_does_not_admit_transcript_poll and five parametrizations of reconciled_view_keeps_each_live_poll_reason_and_one_timer: viewed,other,wake,review,custody. Add only the two existing benign no_op consumers to the fixture Session namespace. Preserve every original statement/assertion, existing profile/xfail/timeout and all unrelated ASTs. Shared-helper third caller captured_attach_timer_overlap_rearms_real_sync_worker retains its original strict baseline xfail; qualify it explicitly without changing that exclusion. A local representative failure before the overlay plus all six GREEN/expectedxfail qualification closes fixture setup; previous216 passes remain historical source-specific receipts.

Latest-dev integration must precede this overlay and preserve unrelated incoming Together source. No new profile, production code, skip/xfail or timeout policy is selected.
