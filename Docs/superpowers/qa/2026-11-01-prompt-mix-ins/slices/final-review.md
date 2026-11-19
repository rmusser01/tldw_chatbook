# Prompt mix-ins — final whole-feature review

Reviewed original base `d83ca84cdbd8cb934ba747bcb7a2179b934904ed` through head `b69e9114626e21292104240cc70e912485a90c54` using the supplied `review-d83ca84cdb..b69e911462.diff`. Requirements: accepted design, implementation plan, ADR-286, and this plan's progress ledger including every `Ruling:`, deferred-minor and parked entry. Rubric: requesting-code-review/code-reviewer.md and SDD final-review rules.

**Ready to merge: With fixes.** No Critical findings; **four Important findings** below need correction. The underlying architecture is sound, but fork receipt ownership and current-choice source fencing are incomplete across entry points. The server Browse error is also reachable through the supported UI. The final fix wave should address these together and run one scoped re-review, rather than repeating nine slice reviews.

## Strengths

- Artifact identity is explicit throughout the model, v3 codec, local schema, retained versions, portable v2 record and Markdown envelope. Literal braces/whitespace are preserved. Ordinary Prompt execution remains positively gated, rather than guessing type from compatibility text.
- Local capture reads UUID-qualified revisions in one SQLite snapshot, runs under the existing finite worker/policy seam, and checks exact service/backend/database identities. Server CRUD advertisement is correctly distinguished from the unavailable coherent capture capability. Stable local authority uses a credential-free path digest plus transient object fences.
- The request composition function wraps only the identified current envelope, returns the original no-mix-in payload, and preserves multipart ordering. Native message/attempt anchors handle hook user rows and synthetic Continue input. Accepted queue/configuration values are frozen, automatic new turns clear selections, and operational records are independent of optional capture.
- The v82 storage seam uses parent transactions, scoped anchors, explicit empty records, immutable revision reuse, guarded retry markers, retention ownership and recovery schema inventory. Replacement trace proof is bound to exact call/checkpoint/result revisions, with latest-complete ownership and predecessor checks; marker/count alone is not authority. Fork proof seals are deliberately cleared.
- Provider caching is an optional prepared-request boundary, translated only into the existing Anthropic path with structure validation, converted-coordinate checks and the four-breakpoint budget. Unsupported mappings omit the optimization. No new endpoint kwargs, cache opt-in or TTL policy are invented.
- Workbench integration preserves the existing Prompts/Browse route, adds literal editing and exact-origin controls, and fences asynchronous completions against view generation and newer edits. Historical instructions use the operational reader rather than current Library text.
- Qualification is unusually explicit about what was actually exercised. Retained 249 affected, 540 predecessor and 1 packaging passes are credible targeted evidence, not a full-suite claim. Actual isolated installed-app PTY plus loopback HTTP exercised the named controls and original/current semantics. Cloud/cache and OS-font claims are correctly withheld.

## Important findings

### I1 — Copying a conversation after an in-place Retry fails

**Location:** `tldw_chatbook/DB/console_prompt_mixins.py:395` (loop starts at `:359`; `_mark_assistant` rejects conflicting attempt markers at `:142`).

`copy_conversation_mixins` selects every retained attempt whose assistant is copied, iterates that set, and stamps the destination assistant once per attempt. A normal failed attempt and its in-place Retry share one assistant row. The second stamp conflicts with the first because `_mark_assistant` receives no predecessor, and the final source-marker pass cannot be reached. Merely sorting the set is insufficient: even chronological stamping still conflicts, and the surviving selected marker must follow the source's actual current marker.

**User impact:** Fork/copy of a chat that has retried a failed response rolls back with `MixinSnapshotUnavailableError('mixin_anchor_attempt_conflict')`. This affects recorded-empty attempts too, so users need not have enabled any mix-in to encounter it.

**Independent narrow evidence:** Using the repository's bootstrap profile, real `CharactersRAGDB(':memory:')` at schema 82, and existing `Tests/DB/test_console_prompt_mixins.py` helpers `saved_turn`/`attempt`: write attempt A; write `replace(A, attempt_id='retry', retry_of_attempt_id=A.attempt_id)` with the same native assistant mapping; copy to a fresh conversation with remapped user/assistant IDs. Both `empty=False` and `empty=True` produce exactly `mixin_anchor_attempt_conflict`. These were new targeted probes of a concrete doubt, not reruns of the reported suite.

**Smallest complete correction:** Copy the retained attempts/items/revisions without repeatedly changing the destination message marker; publish each copied assistant's exact source marker once after the needed records exist. Preserve retry lineage, unavailable-new-format markers, parent rollback, distinct snapshot reuse, empty active preferences and cleared trace proof authority. Add a real DB regression combining same-row Retry with copy/fork, source purge and exact replay; test both empty/nonempty and a chain longer than one Retry.

### I2 — Live fork projection loses operational mix-in ownership, and temporary forks silently become legacy

**Locations:** `tldw_chatbook/Chat/console_chat_fork.py:350` (`ConsoleChatForkSnapshot` carries no mix-in receipts); `tldw_chatbook/Chat/console_chat_store.py:7990` (candidate staging), `:8403` (message publication), `:8450` (new session publication); integration caller `tldw_chatbook/UI/Console_Modules/session.py:2831`.

The new durable copier preserves receipts in SQLite, but the existing immutable fork snapshot and `register_fork_snapshot` publication never carry them. Published assistant messages have neither the operational marker nor verified usage count, and new sessions have no copied temporary batches. The actual UI commits, registers this snapshot, and hydrates Library policy only; it does not reload the copied messages through the changed conversation hydration code.

**User impact:** A temporary fork of a response with recorded nonempty mix-ins is subsequently classified as legacy and returns an empty exact-Retry snapshot. Its history/inspection can silently lose the original instructions, and later promotion cannot recover them. A durable fork has good stored receipts but immediately published messages have `metadata=None` and `mixin_usage_count=None`; failed-message actions offer current regeneration instead of original Retry, and direct retry lacks the predecessor marker needed to append the new record. Reopening happens to use the corrected hydration path; immediate fork use does not. Legacy and recorded-empty forks also need an explicit verified display classification, never a fabricated count or receipt authority.

**Independent narrow evidence:**

1. Real `ConsoleChatStore`, valid `ConsoleSessionSettings`, temporary chat, clean user/assistant, and `publish_prompt_mixin_attempt` with one item; then `issue_fork_fence` → `stage_fork_snapshot` → `register_fork_snapshot`. Result: source items `1`, destination attempts `0`, assistant metadata `None`, and production `retry_snapshot` returns `0` items.
2. Real v82 in-memory DB with existing native `_snapshot`/`_commit` helpers from `test_console_chat_fork_persistence`, one persisted receipt and actual `ChatPersistenceService`; commit then `register_fork_snapshot`. Result: durable read is `recorded` with `1` item, live assistant marker and count are both `None`.

**Smallest complete correction:** Extend the existing fork capture/publication contract with the exact historical operational state needed for the selected copied prefix. Remap native/durable message anchors and retain required Retry predecessors; keep new active selections empty. Qualify unavailable/partial/future markers instead of silently treating them as legacy. Temporary snapshots must remain owned after source close and promote with the destination; durable publication must expose the same marker/count truth as reopen without render-time DB reads. Clear all replacement proof authority on fork as already ruled. Use the existing fork fence/transaction/publication owners; do not add a second copy engine. Cover temporary → temporary → promotion and durable immediate publication/Retry, as well as explicit-empty and unavailable source records. This is separate from I1: fixing the SQL loop alone does not repair it.

### I3 — Current-choice replay routes discard source fences before acceptance

**Locations:** `tldw_chatbook/Chat/console_chat_controller.py:21716` (`_current_mixin_configuration`), `:18832`/`:18844` (Regenerate), corresponding Edit-and-resend capture at `:19950`; `tldw_chatbook/Chat/console_turn_resend.py:281` and `:312` (new-current replay/Continue).

Normal manual/received admission retains a `_capture_currentness` witness through awaited preparation and checks it at acceptance. `_current_mixin_configuration` instead returns only the captured content; Regenerate and Edit-and-resend then await authority/provider resolution and transforms without retaining the Prompt source object witness. Continue's current-choice branch in `_replay_message` has the same gap across hooks and provider resolution. `accept_attempt` creates/uses its current service after these awaits, so its persistence/session checks cannot prove that the earlier Prompt source still owns this preparation.

**User impact:** Replacing the selected Prompt service/backend/database while one of these operations prepares can still dispatch the old source's instructions. This contradicts the approved source-change refusal and exact-origin contract. Library body edits after a successful capture may remain frozen; a source-identity change is different and must refuse unaccepted work.

**Independent narrow evidence:** Reused `trace_harness.__wrapped__` with the legitimate bootstrap profile and existing `select`/`capture_mode(False)` helpers. After an initial accepted send, wrapped `_resolve_for_send_bounded` to replace both `Scope.source` and `Scope.database` before returning the normal provider resolution. Regenerate returned `accepted=True`, increased adapter entries from 1 to 2, and sent `body A\n\ninput`. It should have added no adapter entry.

**Smallest complete correction:** Carry the same body-free source/session capture witness through each current-choice recovery route and check it after awaited preparation and immediately before its acceptance/mutation boundary. Preserve drafts and existing exact-origin checks on refusal. Do not recapture a newer body. Exact original Retry and already accepted queue/continuation snapshots remain exempt from current Library availability. Add held-resolution/hook source-replacement tests for Regenerate, Edit-and-resend and explicit Continue, plus original-Retry/accepted-queue controls.

### I4 — Opening a server Mix-in from ordinary Browse propagates a capability exception

**Location:** `tldw_chatbook/Widgets/Console/console_prompts_modal.py:1208`–`:1218`; policy refusal at `tldw_chatbook/Prompt_Management/prompt_scope_service.py:1936`.

A server may truthfully return/advertise a valid schema-v3 Mix-in while lacking coherent snapshot reads. Browse renders that record as a selectable button. `open_artifact` branches solely on decoded v3 state, enters the local Mix-ins view, then calls `mixin_source_identity(mode='server')` outside its guarded detail-loading block. That method intentionally raises `PromptCapabilityError`. `_artifact_requested` awaits it directly without a recovery catch. The unavailable capability is therefore an unhandled UI event error, rather than bounded source-specific refusal, and the view has already moved away from the selected server source.

**Independent narrow evidence:** Invoked production `ConsolePromptsModal.open_artifact` with its real immutable state and a bounded facade for the selected server record/view, using real `PromptScopeService` and normalized server capabilities. The record is valid v3 and advertises Mix-in CRUD. Result: `PromptCapabilityError` propagates; mode is already `mixins`; message is `server prompt source does not support mix-in snapshot reads.` This was a method-level probe, not another app/UI qualification pass.

**Smallest complete correction:** Check effective source capture capability before transitioning into chat Mix-ins, retain the server Browse context and display the unavailable capability without fallback. Guard/revalidate the awaited mode/view transition so a stale or failed projection cannot turn `query_one` into another unhandled exception. Add one ordinary-Browse server Mix-in refusal regression (and the local success control). Do not implement speculative server capture or silently use Local.

## Minor findings and deferred-observation triage

### M1 — Repair the archive privacy fixture and correct its attribution

**Location:** `Tests/Chatbooks/test_chatbook_prompt_round_trip.py:323`–`:327`; ledger Task 2 deferred item at `progress.md:74`.

The controller's final original-product audit found **5 passes at original d83ca and 5 failures at pinned Task8 3c77bbb**, with the same bootstrap plugin and unchanged function. They must not remain described as original inherited failures. The cause is now established: v6's empty-table rebuild creates `sqlite_sequence('Prompts', 0)`. The test's `INSERT OR REPLACE` does not replace it because `sqlite_sequence.name` has no UNIQUE constraint. It inserts a second row with `971971970`, but the next ID is 1; searching all logs for `'1'` then matches harmless paths and `item-000001`.

Independent native probe: initial rows `[('Prompts', 0)]`; after fixture seed `[('Prompts', 0), ('Prompts', 971971970)]`; allocated ID `1`. This is an invalid sentinel fixture assumption, **not evidence that source IDs/instructions leaked**, and does not justify changing production log copy or the valid empty migration sequence state. Seed the existing row via UPDATE (INSERT only if absent), assert the intended distinctive ID before the operation, rerun only these five nodes and retain the original/current evidence plus corrected explanation. Keep the privacy and atomic-output assertions intact. Root retained the four exact logs/results under its named original-attribution temporary root for durable preservation.

### M2 — Tighten the restored-generation assertion

**Location:** `Tests/Chat/test_console_prompt_mixin_journey.py:257`–`:258`.

The original message object is mutable, so comparing against its post-selection metadata does not independently prove the original attempt ID was restored. Accepting either `None` or 1 cannot establish restored count behavior. Capture the scalar original attempt ID before mutation; distinguish verified live capture/apply from a deliberately unqualified reload. Keep count separate from Retry authority. This remains a coverage improvement; the new fork failures above are independently demonstrated, not inferred from this weak assertion.

### M3 — Expected unavailable-source Preview still claims an internal error

**Locations:** `tldw_chatbook/Chat/console_chat_controller.py:20444`–`:20454`; retained `live/captures/source-unavailable-preview.txt:36` and `live/app.log:608`.

Known missing-source capture is caught by the generic context-preview exception path, producing internal-error copy and a traceback. Workbench separately shows the affected unavailable row and recovery actions; Send is fail-closed with draft preserved, so this is not a new dispatch/security blocker. Catch the known mix-in unavailable/stale categories with bounded disable/remove/retry guidance and normal expected-failure logging. Preserve unexpected-error diagnostics and preview privacy. A narrow regression is sufficient; no new UI layout iteration is needed.

### M4 — Retiring browser read logs an expected source-change traceback

**Location:** retained `Docs/superpowers/qa/2026-11-01-prompt-mix-ins/live/app.log:640` (`console_browser_source_changed`).

The actual normal quit still completed its documented unmount/client/timer cleanup. This is expected-retirement log noise, not evidence of a leaked process, successful exit status zero, or a mix-in cleanup failure. It may remain deferred: do not broaden this final fix wave into unchanged browser lifecycle repair. If adjusted, handle only the expected retirement category at its existing owner and preserve real unexpected failures.

### M5 — Delivery status fields disagree

**Locations:** `Docs/superpowers/plans/2026-11-01-prompt-mix-ins-implementation.md:19`; design spec's opening `Implementation: Not started`; ADR-286's final pre-refinement paragraph still says no product code accompanies it.

The plan still says Tasks 3–9 are pending although the task ledger and qualification report say all nine slices passed their gates. Keep the original design provenance, but update current status annotations to implemented/pending-final-fixes as appropriate. This is documentation consistency, not a request to rewrite the approved design or claim full completion before this review's fixes.

## Ledger rulings reviewed

No parked-with-open-blocker entry was found. Deferred observations were explicitly triaged above rather than silently dropped. The following ruling groups remain reasonable, with their recorded limitations:

- Task 2: separate local/server snapshot-read capability and refuse unfenced server mutation. I4 fixes the remaining UI consumer, not the source contract.
- Task 3: actual integrated v81→v82 allocation, exact staged native-ID transaction contribution mapping, and the narrowly evidenced FTS soft-delete/purge correction. Any integration with a newer schema still requires renumber/rebase and affected migration/recovery qualification; this branch is unreleased.
- Task 4: credential-free lexical DB authority plus exact transient object fences; carry immutable custody through existing copies; failure-only settings recovery; preserve queue-edited accepted snapshots; complete caller-supplied configuration, including empty, is authoritative. I3 closes the missed current-choice entry points without changing accepted-custody behavior.
- Task 5: body-free replacement witness, exact result/call seals, bounded accepted lineage and the promotion/fork authority distinction. The final ruling retains inert replacement lineage on promotion but no historical durable proof seals; fork clears all three proof fields. I1/I2 must preserve those constraints.
- Task 6: keep existing OpenAI automatic caching/key behavior and add no speculative explicit API support; no optimization claim without real provider evidence.
- Task 7: typed Mix-in handoff in the existing one-shot channel, transient source fences, and blank raw create drafts with strict Save validation.
- Task 8: existing Browse landing, additional Mix-ins section, and transient receipt-derived count used only for display. I2 completes the live fork publication audit; M2 improves the generation test.

The historical Task5 SIM102 minor was addressed in the subsequent fix; current source combines the condition. Inherited scoped lint/formatter debt remains disclosed and does not warrant broad cleanup. Original 26-node A/B evidence and identical diagnostic-inventory blobs retain their existing qualifications; M1 is a distinct newly resolved attribution issue and does not invalidate or expand those audits.

## Evidence and review limits

- Read product/test/documentation changes in bounded passes from the supplied package and targeted current-source ranges: artifacts/portability/source capability; migration/recovery/retention; capture/custody/queue; request/trace/current replay; provider conversion/cache; Library/handoff/Workbench/inspection; integrated evidence and deferred ledger observations. No Git diff was regenerated, no subagent dispatched and no earlier per-slice review repeated.
- HEAD was verified as b69e911462 and tracked status remained clean. The six source/test files involved in the new probes/findings match the final product-source manifest SHA-256 values. Read the final-checks, live launch, source manifest and independent Task9 review; relied on its previously completed 542/543-record, 77-live-hash and link checks rather than repeating the entire evidence census. Raw ANSI/SVG/log files were consulted only at named evidence points.
- Did not rerun the reported 249/540/1 checks. Only the new narrow probes described above were executed using the existing private test bootstrap, real DB/store owners where applicable, and no external provider. An initial bare import was refused by the recovery guard; it did not run a test and was not bypassed. Subsequent probes used `Tests.conftest`'s legitimate private bootstrap. Two probe setup mistakes (DB context-manager assumption; omitted temporary-chat settings) were corrected before the reported successful reproductions. They are not product findings or counted test passes.
- Served module launch causality was resolved by source comparison: original/head `app.py` both have Git blob `ff4a7bc951ab33a82e4a03ea94bac7eae67d7fde`; `app_collections_wiring.py` both have `625b8c08d3ee56bee3559123d3315f8adfd02cec`. `app.py:5684` enters `_run_module_main()` before `_CONSOLE_SKILL_APP_SOURCE` is assigned at `:5704`, while unchanged `app_collections_wiring.py:188` accesses it during deferred startup, matching the retained traceback. This is an unchanged module-order defect, not a newly introduced mix-in eager reference. It remains a real documented launch limitation; no original runtime reproduction or fix is claimed. The normal installed console-script live run succeeded.
- Real external models/cache metrics remain unqualified; the loopback responder is not a model. ANSI-derived PNGs do not prove native OS font/border rendering. Normal teardown was observed but no detached process exit-code zero was retained. No full suite was authorized or run; existing broader failures are not called green.

## Recommended final wave and assessment

One combined fix dispatch should handle I1–I4, repair M1's fixture/attribution, and preferably tighten M2 plus current delivery status. M3 can be fixed narrowly or explicitly deferred; M4 can remain deferred with this explanation. Run only covering fork/DB/promotion/retry/admission/server-Browse tests and the five archive nodes, plus scoped lint/format. Retain actual commands/results and update evidence manifests for changed source. Re-review the complete fix diff once against these findings and any new breakage it introduces.

**Ready to merge: With fixes.** Core literal composition, captured queue behavior, operational/trace separation and provider gating are credible. The four Important gaps violate approved ownership/lifetime/unavailability behavior and should be corrected before delivery is called complete.

Skipped full-suite, cloud/cache and new live UI runs. Remaining risks are the four Important findings, existing qualified test/static-analysis debt, and the unchanged module-launch defect; none is represented as a passing check.
