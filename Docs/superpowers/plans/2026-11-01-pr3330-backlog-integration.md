# PR3330 current-base integration and repair

## Current execution authority

Actual base: 7d2fc1ea13bf3ef68e214d06abc392c3592a207b. Source: c4dcf9a5a38c3aa109e7bd2c2fe987922d3f91c9; source base: 92ecd2c0aa73fef850bbf2d0516a3f0e11c27e46. Live PR refreshed OPEN, non-draft, unarmed, zero threads immediately before local rebase. Branch codex/pr3330-resync in the existing isolated pr3298-resync worktree. Local rebase stopped at fbdd036a93 (2/19), conflicts in console_provider_gateway.py and provider_error_reason.py. Implementation owner completes the rebase, source edits, local commits and serial targeted tests. Parent owns all remote actions.

## Global Constraints

- Oldest eligible non-draft first. Never merge drafts or mark ready. PR3267/3297 are excluded from all actions. No release/main work. Main checkout is read-only; its .venv Python may be used.
- Hosted CI and size limits waived by owner; superpowers:requesting-code-review replaces Qodo. No weakening security, tests, custody/admission, census floors or size budgets. Focused tests only, no full sweep.
- Preserve current physical-contact proof, refused-stream accounting, unknown-contact replay refusal, native/received custody, PR3328 compaction/admitted-skill behavior, PR3329 blocker/rebuild ownership. No whole-file source snapshot replacement.
- Use native private fixture roots and existing windows_private_fixture_runner. Do not normalize ACLs or bypass raw_source_selection_changed. Preserve before/after provenance and record actual commands, cases, failures and evidence limits. One test/app-import/collection owner at a time in this session. Do not touch unrelated session processes.
- Test enrollment must be narrow: exact git/base identity, changed sources, selected tests, imported fixture dependency closure and applicable conftests, before/after dirty-set checks. Do not hash all Tests/**/*.py. Artifact changes use canonical artifact validation rather than unchanged generator/helper suites. Start fixture diagnosis with one failing node and --maxfail=1. Repeat only affected cases after a fix; do not rerun all passing selections routinely.
- No subagents from implementer. Parent dispatches one independent review after implementation. Source review is not final approval. Parent refreshes live head/draft/auto/dev/threads and uses exact lease before publication. Do not mutate remote or other checkouts.

ADR required: no new ADR. ADR path: backlog/decisions/211-ephemeral-provider-failure-presentation.md. Reason: repair existing transient presentation without new ownership/persistence/diagnostic contracts. Link existing094/230 where relevant to custody preservation.

## Task 1: complete rebase and the prepared coupled repairs

Treat the prepared classifier and strip/custody work below as one integration task because the same typed error/readiness interface is their shared contract. The prepared plan's preparation-only statements and stale comparison base are historical; this handoff is execution authority. Preserve current budgets/census and regenerate diagnostics only after reviewing exact changed statements if needed. Source task and follow-up ACs govern outcomes; TASK38424 narrow-width limitation remains open. Add accurate final QA with current receipt counts and historical-source boundaries.

## Prepared implementation detail (verbatim historical plan)

# PR3330 image-input classification repair implementation plan

> For the later implementation owner: use superpowers:executing-plans inline after the parent's explicit handoff. No subagents or execution during preparation.

Goal: Integrate TASK33621.11 onto the then-current base without losing its newer contracts, fix TASK38420 at the shared classifier, and apply TASK38424's model-independent strip copy within existing geometry.

Architecture: Keep ChatAPIError, RunOutcome, ConsoleBlockedReason and the existing gateway queue. Classify bounded provider evidence before projection; carry only the existing transient boolean. No new error framework, provider-name branches, dependencies, persistence or UI layout.

Tech stack: existing Python, stdlib json/re, Textual 8, pytest, real SQLite AgentRunsDB.

Spec/evidence: pr3330-preparation-brief.md; pr3330-initial-review.md; immutable pr3330-source-review/metadata.json and review-package.diff; source TASK33621.11, TASK38420 and TASK38424; existing ADR211.

## Identity and authority

- Source reviewed: c4dcf9a5a38c3aa109e7bd2c2fe987922d3f91c9; original base: 92ecd2c0aa73fef850bbf2d0516a3f0e11c27e46.
- Comparison base inspected: a9c1fa9d577f623befe0cc65f94d2a003ac6c8f1. These are immutable evidence identities, not promises about a later branch head.
- Preparation only: PR3323 owns the mutable worktree and sole test slot. Nothing below has been implemented or run.
- Parent owns eventual rebase, publication and merge in oldest eligible order. Drafts remain drafts; PR3267/3297 excluded; no release. Hosted CI and size limits waived; requesting-code-review replaces Qodo; targeted verification only; no budget raise or additional approval workflow.
- Before execution, parent must provide the actual integration head and release the test slot. Recheck source/base changes by exact objects. Do not overwrite newer code with snapshot files wholesale.

ADR required: no new ADR; existing ADR211 applies.
ADR path: backlog/decisions/211-ephemeral-provider-failure-presentation.md.
Reason: routine classification/copy repair and integration of existing transient presentation; no new ownership, schema, runtime boundary or durable error contract. Link ADR211 in TASK33621.11's Implementation Plan and Implementation Notes and in TASK38420's plan/notes. State explicitly that typed diagnostics, fallback semantics and STEP_ERROR/run-log custody remain separate from Console presentation. If implementation starts persisting the new verdict/copy or changing exception messages to transport it, stop that expansion: it is outside this plan and ADR211.

## File and interface map

Repair:
- tldw_chatbook/Chat/provider_error_reason.py: keep is_image_input_rejection(exc: BaseException) -> bool and _is_image_input_rejection; replace _IMAGE_WORDS and _body_mentions_image(body: str) -> bool's broad matching. Reuse bounded_response_body and _MAX_BODY_BYTES=65536. Use _text_rejects_image_input(text: str) -> bool for rule 6 and _is_image_input_path(path: str) -> bool for rule 5; these stay private in the same module. Update attach_provider_reason(exc, response, *, known_credentials=()) -> BaseException to retain the existing image_input_rejected boolean when this existing seam drops structured evidence.
- tldw_chatbook/Chat/console_display_state.py: console_image_blocker(refusal: str | None, model: str | None) -> ConsoleBlockedReason; change notice only to "this model can't accept images". Keep model argument for compatibility and full refusal in text.

Integration touchpoints (merge existing source behavior onto base; no redesign):
- tldw_chatbook/Chat/console_provider_gateway.py: _QueueItem.error(..., local: bool=False, image_rejected: bool=False); worker classifies original exception; queue consumer assigns .console_copy and .image_input_rejected on the projected typed error.
- Keep _provider_error_copy_with_model_recovery(copy, *, model, status_code, provider_message="", tools=None, refused_stream=False, model_unavailable=False) -> str with BOTH keyword arguments. Its refused-stream recursion must forward model_unavailable and tools, retain the tool-definition guard and add REFUSED_STREAM_ADVICE once.
- tldw_chatbook/Chat/Chat_Deps.py: existing ChatAPIError.console_copy plus source gateway_sentence and image_input_rejected defaults.
- tldw_chatbook/Agents/agent_models.py: RunOutcome.console_image_rejection remains keyword-only, repr=False, compare=False, transient like console_copy.
- tldw_chatbook/Agents/agent_service.py: catch-all writes diagnostics through describe_stream_failure and separately copies console_copy / is_image_input_rejection(exc) into RunOutcome.
- tldw_chatbook/Chat/provider_failures.py and console_chat_controller.py: retain console_failure_copy, image_refusal_hint(failure, carries_images), all direct/escaped-bridge/outcome callers; image history is the second gate. Do not make a body-only classifier claim that a chat contains an image.
- tldw_chatbook/UI/Console_Modules/prompt_queue.py and wiring.py; UI/Screens/chat_screen.py: integrate coded gate without bypassing received-intent/native custody. Keep _console_send_blocked_reason() -> str, _console_send_gate() -> ConsoleBlockedReason and _console_composer_send_blocker() -> ConsoleBlockedReason. Add _console_send_gate to the finite _CONSOLE_RECEIVED_SCREEN_SELECTORS with its actual defining function/code, alongside existing readers.
- tldw_chatbook/Widgets/Console/console_composer_bar.py: consume ConsoleBlockedReason as source does; preserve 52-cell cap, live width budget, 32-cell draft floor, hide/ellipsis threshold, tooltip and setup-link rules. No geometry/CSS changes for the long-ID fix.

Tests: extend existing source Tests/Chat/test_console_failure_copy_typed.py and Tests/UI/test_console_send_strip_codes.py. Preserve Tests/Chat/test_console_readiness_codes.py. Use existing Tests/Chat/test_console_provider_failure_copy.py for diagnostic custody and Tests/Chat/test_first_reply_failure_copy.py for attachment/sanitization seam controls; no new test framework/module required. Add stock received-custody regression in Tests/UI/test_console_received_intent_feedback.py using its existing harness.

Docs/task hygiene at execution: update TASK38420 and narrow TASK38424 notes; add ADR disposition to main task; adjust the image-strip sentence in Docs/User_Guide/console/attachments-images-voice.md only if it repeats model-ID strip copy. Architecture docs may get a short failure-behavior clarification if their present text would be false; do not rewrite unrelated docs.

## Chosen parser/classification rules

1. Preserve IMAGE_REFUSAL_STATUSES = {400, 413, 415, 422}; no 401/404/429/500 expansion. No status, unusable body, unrelated text, unsupported shape or parser/access error means False (unknown), not an inferred refusal. Keep existing cycle protection, explicit cause precedence and suppressed-context boundary; never crawl a severed context to recover a body. Do not infer a status from display text or str(exc).
2. Preserve an exact True .image_input_rejected as the explicit verdict from an earlier classified exception. It is not an invitation to accept truthy strings/integers. Adapters/attachment seam may set it only after allowed-status evidence. Raw failures without that verdict retain the status gate. Leave status/contact accounting to its existing owner.
3. Parse only bounded JSON. Support the existing dict and first-dict list envelope. For error dicts inspect error.code/type as classification-only exact scalar codes, error.param as a request field path, and error.message as text/path evidence; never recursively scan echoed request schemas, metadata, headers, tool definitions or arbitrary fields. Retain existing top-level error-string/message/detail fallback, using the same strict text/path predicate. Malformed JSON-looking bodies fail closed; genuine plain-text bodies use the text predicate.
4. Exact code/type positive initially: invalid_image_format (case-normalized whole value). Do not generalize to every *_image*, vision*, invalid_request_error, function/tool name or arbitrary code containing image. Add another code only with an actual retained schema fixture during implementation review; no provider-name special case.
5. Image field path positives are rooted message-content input paths, not arbitrary dotted strings: messages.<nonnegative index>.content.<index>.image.source[.<field>...] and messages.<index>.content.<index>.image_url[.url], allowing bracket-index equivalents. Also accept content.<index>.image.source... used in a provider's message-local error. For param require the whole value to be such a path; for message require that path at the beginning followed by ":" and its detail. Never count tools/functions/parameters/properties paths or a quoted/embedded path mentioned in schema prose. Do not broaden to any occurrence of ".image.".
6. Free text/provider_reason requires an unambiguous rejection clause at the start of the sentence, case-insensitive, with real image/vision words and rejection semantics. Cover these existing/control clause families explicitly:
   - "Image does not match the provided media type ..." / "Invalid image format ..." / "Unsupported image format ...";
   - "image exceeds ... maximum ..." / "image is too large ...";
   - "Unsupported media type for image input";
   - "This model does not support vision input" / "image input is not supported ...";
   - "You uploaded an unsupported image ...".
   Do not search arbitrary positions in a sentence for these phrases: quoted function names/schema descriptions cannot become an image rejection. Existing Anthropic path-prefixed message is accepted by rule 5. Mere "image", "vision", image/png in schema text, "generate_image", "image_generator", model IDs, "see image documentation", or an unrelated schema's image property are insufficient. Use the same predicate for explicit provider_reason; fixing only JSON would leave a second false-positive route.
7. Evidence preservation: the HTTP exception chain already carries structured response bodies for normal requests/httpx paths, so classify there before project_provider_error severs them. hosted_chat._raise_http_error instead uses attach_provider_reason then raises from None: only error.message survives today. At attach_provider_reason, read the bounded body once, compute a boolean from actual integer response.status_code plus rule 3-6, and attach the existing verdict to the same exception before the body disappears. Keep existing credential scrubbing and sensitive-request early return; never attach body/type/code/param strings. Do not change the helper's reason allowlist or diagnostic exception message.
8. Do not reclassify model_not_found by image words or alter provider_error_category. Preserve ChatModelUnavailableError semantics and 404 recovery; a plain 400 gets neither generic model advice nor image advice absent image evidence. TASK38425's category wording is separate.

## Review focus / proof matrix

- Tool schema references must be negative in every available carrier (message, code/type near-match, param, provider_reason, plain text), including a chat containing an admitted image.
- Real structured image codes/paths must survive projection, including code-only hosted errors whose chain is suppressed by design.
- 401/no-image/unknown failures cannot acquire image advice; .console_copy and visible words are never classification input.
- Refused-stream recursion/fallback and physical admission/contact proof must retain their current behavior.
- The mounted strip and refused Enter must retain draft/attachment/session custody and plain-string diagnostic/presentation boundaries.

## Task 1: integrate current contracts and repair the classifier (one RED/GREEN cycle)

- [ ] Read the actual TASK38420 file, mark In Progress by CLI at implementation handoff, add this plan and ADR211 disposition before editing. Parent performs the branch integration; re-resolve touched hunks against its exact resulting head.
- [ ] Add test_tool_schema_image_name_is_not_an_image_rejection, parameterized agent=True/False, using the existing real gateway/controller/store/bridge fixture. Send a valid tiny PNG as an admitted pending attachment on a vision-capable selected model, then fail the provider with HTTP400 error.message "Invalid schema for function 'generate_image': ...". Register/send that tool schema so existing tool-rejection recovery is exercised. Reuse the sent-tool composition pattern and _openai_tool_error fixture shape from Tests/Chat/test_console_tool_definition_rejection.py; if using _failing_call, construct ChatBadRequestError with the schema message (not its generic "bad" placeholder), because existing tool recovery reads provider_message=str(exc). Capture actual provider payload containing image content and tool name; assert visible row retains tool/schema recovery but IMAGE_HINT is absent, .image_input_rejected/RunOutcome.console_image_rejection is False, and generic model advice is absent. Keep a history-image variant because existing source coverage uses history rather than staged custody.
- [ ] RED expectation on immutable source behavior: the test reaches a failure System row and fails specifically because IMAGE_HINT is appended / image verdict is True. A missing hook/profile, image-capability preflight refusal, uncollected test or absent provider call is not this RED.
- [ ] Add compact parser controls in the same test file. Negative payloads: generate_image/image_to_text in message; type/code="generate_image"; param="tools[0].function.parameters.properties.image"; schema prose containing image/png, quoted content image path, and the exact provider_reason/plain-text tool error. Assert False. Unknown/malformed or cap-truncated JSON and broken response access do not raise or infer True; bytes beyond the cap cannot create a positive; suppressed contexts and cycles terminate.
- [ ] Positive controls (agent/direct, image present): exact error.code="invalid_image_format" with otherwise generic message; error.param="messages[0].content[1].image_url"; Anthropic message "messages.0.content.1.image.source.base64.data: Image does not match the provided media type image/png"; preserve existing 413/415/422 text cases. Assert projected verdict True and visible row == sanitized gateway sentence + exactly one IMAGE_HINT.
- [ ] Add test_attached_structured_image_verdict_survives_suppressed_chain in test_first_reply_failure_copy.py: response code-only invalid_image_format, helper -> exception -> raised from None -> real gateway projection; assert True and no response-body/field strings newly retained. Tool schema counterpart is False. Sensitive attachment keeps current early-return behavior. This control should fail on source because attach_provider_reason currently preserves only message.
- [ ] Keep/add agent/direct 401 with image in history and image words in error body: no IMAGE_HINT or model hint. Repeat true image body without an image in chat: classifier may be True but row has no IMAGE_HINT. Keep status500 no-mmproj and 429 negatives. A timeout/synthetic status with no body remains unknown.
- [ ] Keep 404 and explicit ChatModelUnavailableError(status400) recovery. Add helper-level refused_stream=True/model_unavailable=True control: model recovery survives recursion and streaming advice occurs once. Ordinary temperature400 still lacks model advice. Do not change TASK38425's category.
- [ ] Run only the added nodes on the pre-repair integrated source, record failing assertion/output; implement rules above; run the same nodes plus the three source typed suites. Actual commands are below; no run is authorized during preparation.
- [ ] Preserve both diagnostic-custody tests in test_console_provider_failure_copy.py. Adjust only source's intentional plain400 visible-copy expectations; keep exact semantic type, status/provider, retry_after, content-free str/message, SQL STEP_ERROR and run-log assertions. Add image/tool-control variants that exclude IMAGE_HINT and presentation canaries from durable diagnostics. Never solve by making projected error.message equal Console copy.
- [ ] Review the integration diff for refused_stream fallback, _note_provider_failure call, local=local, answer_noted, response closing/retirement and existing provider_error_category tail. The worker enqueue must carry BOTH local=local and image_rejected=...; do not paste source's earlier local expression over base's accounting branch.

## Task 2: same-surface long-model strip and mounted custody controls

- [ ] Extend test_console_send_strip_codes.py with test_long_model_image_strip_keeps_the_limitation_visible. Use a selected non-vision long model ID (including slash/colon/punctuation and >52 cells), actual staged PendingAttachment, real readiness and mounted composer. At (160,42) and source wide (235,52), assert painted strip content equals "Send blocked — this model can't accept images", no ellipsis, no setup link/class, disabled Send, full original model ID in Send tooltip, and visible draft floor. RED on source is the old model-ID strip/truncated image meaning, not a changed-copy setup precondition.
- [ ] Change console_image_blocker.notice only. Keep its text exactly the full model-specific refusal; keep build_console_disabled_reason's typed dispatch and no setup link. Update the existing local-model expected strip string.
- [ ] Add resize control at (80,30) and an intermediate width, using measured region/content_region and existing _send_reason_width_cap. Preserve draft visibility, expected hide/ellipsis policy, full tooltip and unchanged draft/attachment; do not claim full strip copy fits where the current budget intentionally hides/ellipsizes it.
- [ ] TASK38424 AC1 asks for whole meaning at the narrowest supported width. Existing budget permits only a 12-cell strip then hides it; its "Send blocked — " prefix alone cannot fit there. The model-independent 44-cell sentence fixes long-ID clipping at the ordinary cap without redesign, but does not fulfill that stronger AC. Record this scope limit and leave TASK38424 open unless the later owner explicitly supplies an approved narrow-width copy requirement; no silent AC rewrite or geometry expansion.
- [ ] Add test_stock_received_image_refused_enter_preserves_custody in test_console_received_intent_feedback.py, route enter/send-button using _received_console_case, _send and _held_received_record. Exercise real stock readers; do not monkeypatch _console_send_gate/_console_send_blocked_reason to fake the refusal. Stage image and actual non-vision session settings after the appropriate receipt setup. Assert stock_native_methods against the finite screen selector tuple succeeds, same session/received claim and captured image survive across held preparation, no provider calls/admissions occur, dispatch is REFUSED, exact full image-refusal System row is plain str, draft/attachment remain, no user turn is consumed, pending acknowledgement/received custody retires.
- [ ] Wait for actual painted refusal and no "Sending…" using a bounded predicate, then verify build_context_snapshot/deepcopy succeeds. Preserve current base's repaired refusal repaint. Reuse harness teardown's exact reader retirement barrier; never equate visible handoff with durable acceptance or use a fixed sleep as proof.
- [ ] Keep source's Library-evidence-before-setup refused Enter and no-key plain-string controls. If a stock receipt refuses before a held configuration reader can be entered, pin that legitimate boundary and retirement rather than forcing the harness into a nonexistent later phase.

## Targeted verification after handoff only

Run new causal nodes first, then:

    python -m pytest Tests/Chat/test_console_failure_copy_typed.py Tests/Chat/test_console_readiness_codes.py Tests/UI/test_console_send_strip_codes.py -q
    python -m pytest Tests/Chat/test_console_provider_failure_copy.py::test_typed_gateway_error_reaches_console_without_changing_diagnostics Tests/Chat/test_console_provider_failure_copy.py::test_typed_console_presentation_never_enters_durable_agent_error -q
    python -m pytest Tests/Chat/test_first_reply_failure_copy.py::test_attached_structured_image_verdict_survives_suppressed_chain Tests/UI/test_console_received_intent_feedback.py::test_stock_received_image_refused_enter_preserves_custody Tests/UI/test_console_send_acknowledgement.py::test_a_refused_acknowledged_enter_repaints_its_refusal -q
    python -m pytest Tests/Chat/test_console_provider_gateway.py -k "refused_stream" -q
    python -m pytest Tests/Chat/test_console_refused_stream_retry.py Tests/Chat/test_console_trace_retry_recovery_proof.py Tests/Chat/test_self_hosted_retry_accounting.py Tests/UI/test_console_composer_reason_width.py -q

The new node names above are planned, not existing/passing evidence. Account for source/base renamed tests at handoff without removing their assertions. For the acknowledgement test, its current-base monkeypatch of _console_send_blocked_reason may need targeting the coded gate after integration; the new stock-reader control remains unpatched.

Accounting controls must retain these current-base proofs: physical_fallback_outcomes_bound_retry_recovery; real_adapter_timeout_cannot_authorize_refusal_only_replay; final_http_refusal_cannot_hide_unknown_earlier_attempts; a_retry_after_definitive_rejections_is_recoverable; a_llamacpp_direct_answer_after_a_refusal_keeps_the_first_call_rule; only_an_http_error_status_counts_as_a_refusal. Verify actual dispatch/admission counters and response-owned outcomes, not provider_started or classifier booleans. No-body/unknown contact stays unknown; synthetic status is not replay permission.

Apply targeted formatter/linter checks to modified files using repo configuration and inspect git diff --check at implementation time. No full suite, no hosted-CI requirement, no size-waiver reversal. Record exact commands/results and source identity. Self-review and independent requesting-code-review after the repair; parent alone handles subsequent branch publication.

## Evidence and completion limits

Historical source metadata/task notes describe live Anthropic/OpenAI proxy runs, RED/mutation checks and prior counts; this preparation reproduces none of them. Current-base tests are source evidence of contracts, not newly executed PASS. Mounted tests proposed here prove headless painted/state/geometry boundaries, not real terminal flush or real-provider compatibility. Any later live verification must use authorized existing resources and report exact observed response/contact facts.

Keep all source follow-ups visible: TASK38421 (HTTP500 no-mmproj requires owner decision), TASK38422 (partial Evidence badge), TASK38423 (staged-summary equality/Retrieval-prefix classification), TASK38425 (model_not_found category), TASK38426 (historical refused Enter paint issue; base already has separate repair, preserve it), TASK38332 (default OpenAI gpt-4.1 vision patterns). Only TASK38420 is the blocker repaired here; TASK38424 receives bounded copy improvement with the AC1 limitation above. No task may be marked Done without its actual acceptance criteria and required evidence. Source review C0/I1/M1 and "with fixes" are not final merge approval.

## Current-base fix-wave ruling

The stock received image preflight can refuse before request promotion; generic runtime cleanup then has no recovery request and previously projected no reason. Repair only this checked image branch with a plain-string System row in the original captured session, using frozen configuration/capability evidence before attachment transfer or admission. Preserve cancellation/stale/closed ownership checks and existing terminal cleanup. ADR required: no new ADR; existing ADR094 (console-turn-lifetime-and-navigation-boundary), ADR211 and ADR230 apply. Cost if wrong: stale or duplicate feedback; mounted Enter/button cases pin original ownership, no promoted request/admission/contact, exact painted refusal, retained draft/image and acknowledgement retirement.

Ruling: only for110/80 resize states, replace the new unconditional32-cell floor with positive displayed draft content within row bounds. Exact44-cell ordinary160/235 paint and existing qualified32-floor controls remain. Measured110 draft13/cap18/strip-content16;80 cap0/hidden. Preserve cap/hiding/ellipsis, draft/attachment/tooltip and existing geometry. TASK38424AC1 remains open. Cost: existing narrow draft/strip limits remain.

Ruling: natural Enter/button may finish at genuine coded REFUSED or received preflight, preserving exact feedback/custody; an added original-handler route immediately awaits the actual stock Button.Pressed handler after staging and MUST reach held original configuration/image preflight. No gate/state/callback patches. At80 preserve and explicitly record draft region2/content0 (padding-only), not positive-content usability. Current/base geometry helpers and CSS comparison are recorded; no layout code changed. TASK38424 remains open, with zero-content narrow draft limitation.
