# Capture-On archive recovery correction implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this single corrective task. Steps use checkbox syntax.

**Goal:** Let a captured native archive-recovery follow-up retain its exact current-user source and reach the existing restore confirmation.

**Architecture:** At the existing authenticated archive metadata insertion seam, compose the current text with the already admitted request using the existing CURRENT_TURN_TEXT descriptor. Preserve the original saved source and frozen policy. The agent factory continues to reuse only exact frozen rows; trace runtime/service guards stay intact.

**Tech Stack:** Existing Python, SQLite, pytest and typed Console trace machinery; no new dependency.

**Spec:** [Approved archive recovery design](../specs/2026-09-17-agent-archive-recovery-design.md), [approved metadata scope](../qa/console/2026-09-27-agent-archive-recovery-followup-proposal.md), and [ADR097 current-turn transformed request ownership](../../../backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md#amendment-recorded-2026-09-08-current-turn-transformed-request-ownership).

## Global Constraints

- Scope: local saved non-deleted conversations; interactive user-originated primary runs in ordinary local Console sessions. No remote, temporary, subagent, MCP or unattended execution.
- Assistant Library access must be Allowed for agent tools; the dedicated capability works in Direct and RAG-only modes. Manual Library recovery remains available when agent access is Blocked.
- Preserve frozen per-turn authority, destination disclosures, persona restrictions, kill switches and runtime execution gates. Tool names and actor provenance are app-owned.
- Fresh native follow-ups disclose only previously model-observed set/reference/order/display title/workspace label/valid scopes, source-session envelope and omitted-page count, to the exact same resolved destination; no excerpts, archived messages, target IDs, timestamps or cursors. The rider including its wrapper remains bounded to 32 KiB.
- Restore only after one-shot exact-target UI confirmation. Neither stored Allow nor remembered approval substitutes for it. Restoration does not activate, send, resume agents, move ownership or copy messages.
- Only the accepted current user owner from the pre-transform snapshot may receive CURRENT_TURN_TEXT. Changed historical rows, unknown owners, and changes outside the current text field gain no authority.
- Use real SQLite/controller/native runtime/agent gateway/capture factory. Fake external transport only through the documented gateway seam, with a fail-closed network guard. No live model, normal profile, downloads or full sweep.
- Work only in the existing clean isolated worktree /Users/macbook-dev/.codex/worktrees/agent-archive-recovery/tldw_chatbook; leave the original dirty checkout untouched. No push, PR, merge or cleanup.
- Preserve earlier completion and qualification boundaries. This task does not explain historical unsupported_surface_change, aggregate +729 FD growth, provider errors, or the pre-correction dictionary second-send control failure. Next Send preview/estimate omission remains outside scope.

ADR required: no

ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md and backlog/decisions/166-agent-assisted-archive-recovery.md

Reason: direct correction using existing typed current-turn ownership and approved archive metadata contracts; no new schema, authority, disclosure, retention or architectural boundary.

## Task 1: Preserve current-user ownership through the authorized rider

Backlog: [TASK-33164](../../../backlog/tasks/task-33164%20-%20Preserve-captured-archive-recovery-follow-up-ownership.md), already In Progress with an Implementation Plan before source work.

**Files:**
- Modify: tldw_chatbook/Chat/console_chat_controller.py, existing rider seam near26518 before bridge.run_reply.
- Modify/test: Tests/Chat/test_console_archive_recovery_followup.py; extend its real native fixture rather than add a competing harness framework.
- Create/update evidence: Docs/superpowers/qa/console/2026-09-27-agent-archive-recovery-capture-on.md and its narrowly scoped artifacts.
- Root owns this plan, Backlog status/notes and SDD ledger.

**Interfaces:**
- Consumes: recovery_runtime.admit(turn_context, session), followup_context(authority), _prepend_evidence_context(), PreparedConsoleRequest.provenance, SavedRevisionTraceProvenance, DerivedTraceProvenance, TraceTransformKind.CURRENT_TURN_TEXT, and the existing _is_current_user_text_transform().
- Produces: matching final agent rows and admitted request for ConsoleAgentTraceRequestFactory.build; its equality rule and ConsoleTraceBoundaryFactory/runtime/service remain unchanged.

- [x] **Step 1: Read the corresponding Backlog file, narrow approved scope and exact diagnostic report.**
  Diagnosis: /private/tmp/archive-capture-on-diagnosis.md. Seed probe: /private/tmp/test_archive_capture_diagnosis.py. Final probe proved the ownership loss but exited1 with blocked-dispatch teardown error; repair test custody/cleanup before treating it as a regression. Earlier invalid transport phases and root control failure must remain recorded.
- [x] **Step 2: Establish a production-boundary RED.**
  Extend the native follow-up fixture with real ConsoleProviderGateway, ConsoleTraceBoundaryFactory and documented fake adapter. Deny socket/network access and account for pending dispatch before teardown. Parametrize Direct/RAG-only. Search, then ask “Restore the second one.”; the fake model must use the actual retained page metadata and request exact confirmation, which the existing trusted callback can explicitly accept. Assert only the intended target/scope changed and no transcript rewrite. Run the exact new test before production edits using PYTHONPATH=. and the shared installed interpreter; expect the successor trace_turn_unavailable failure at the real capture boundary.
- [x] **Step 3: Apply the minimal controller-local reseal at the existing authorized seam.**
  Compose only the actual current-user text after successful metadata admission. Validate the unmodified current row matches the admitted request, its original saved source is exact, all history/role/attachments/sidecars are unchanged, and the transformed row is text-only. An existing CURRENT_TURN_TEXT descriptor must retain its one original SavedRevisionTraceProvenance input; no nested or guessed source. Reuse existing request construction/immutable replacement patterns and frozen policy/metadata/durability/attachment handling. The existing typed shape is:
  ```python
  DerivedTraceProvenance(
      TraceTransformKind.CURRENT_TURN_TEXT,
      inputs=(saved,),
      artifact=ProviderArtifactTraceProvenance(
          TraceProvenanceSource.ACTIVE_REQUEST, policy
      ),
  )
  ```
  Here saved is the exact admitted current-user source and policy is the admitted request's frozen policy. Do not alter generic row matching, infer an older user's source, relabel old artifacts or widen surface replacement.
- [x] **Step 4: Verify covering behavior and strict refusals.**
  New targeted tests must cover real captured Direct/RAG search→natural selection→exact confirmed restore, exact source pinning/reconstruction and unchanged predecessor call rows/reconstructed requests. Include composition with an already typed current-user text transform and a subsequent ordinary captured send. Unknown/unowned sources, changed history and non-text changes must fail closed. Revalidate affected metadata admission/history gates in the existing follow-up file. Use the smallest focused descriptor controls where necessary; do not repeat prior broad qualification or mask the unrelated dictionary second-send failure. Run Ruff/check formatting on changed files, report unchanged controller debt honestly, and use scoped whitespace checks with raw logs byte-preserved.
- [x] **Step 5: Self-review, record exact evidence and commit only corrective source/tests/QA.**
  Keep the historical native Capture-Off success intact. Fake-boundary green is automated qualification, not a new live Capture-On walkthrough. Record commands/exit codes/raw phases, exact source hashes, policy pins, old trace immutability and all remaining limits. Independent spec/quality review follows using the complete baseline-to-correction diff. Do not mark Backlog Done before that review.

## Evidence baseline

HEAD before correction: 0f5e537fe33553daf9d947834f90d14315a77772. Worktree clean before this planning metadata.

Diagnostic: real captured search3 COMPLETE calls; Allowed successor with metadata blocks trace_turn_unavailable before service/witness; current Library Blocked excludes metadata and successor completes with a CompletedToolTurnWitness. Prior call records and reconstructions unchanged. Final diagnostic2 passed/1 teardown error, exit1; final guarded adapter0 socket attempts. A preliminary setup attempted remote transport and failed DNS; invalid product evidence, preserved in the report.

Root current-text control on this HEAD: test_dictionary_send_and_successors_keep_exact_saved_source[False] failed at test line162: second outgoing content alias second instead of expanded second. This is a pre-correction control failure, not a passing reference and not attributed to the rider.

## Plan self-review

One task owns the rider/request seam, regression and documentation; there is no cross-task producer/consumer overlap. The authority, trace provenance, source immutability and exact-confirmation requirements are all in Task1. No new UI, schema, dependency, consent or generic history behavior is proposed.

## Task1 review fix round1

Independent review of e8fa9b0b978 found one Important lifetime defect: the new reseal await precedes the finally that releases an authority already admitted from native user proof. Protect that exact lifetime through refusal and cancellation before bridge entry. Use the existing captured native fixture with real admit/release, controlled unknown-source refusal and a cancellation barrier; observe closure/removal, no successor reservation/transport/restore, unchanged predecessor traces, and drain held work. Revalidate only amended/covering cases, retain earlier phases and source hashes separately, then a scoped fix re-review. No new AC/ADR/export/trace-guard or unrelated Cancel/Discard change.

## Task1 completion (original automated qualification)

TASK-33164 completed. Ownership correction: `e8fa9b0b978`; lifetime fix: `03caf0a6b4bd`. The initial independent review found I1; the scoped fix review addressed it and approved the correction with no actionable findings. [Final QA and evidence](../qa/console/2026-09-27-agent-archive-recovery-capture-on.md) preserves the earlier 39-pass phase and the five covering fix checks, exact hashes, immutable predecessor evidence, static debt and qualification limits. No new ADR was needed; ADR097 and ADR166 remain the governing contracts. No full sweep, live model, push, PR, merge or cleanup was performed.

## Task 2: Correct the shared Direct Library factory call (2026-09-28)

The user’s subsequent “do so” authorizes bounded local live verification, superseding the initial no-live-model constraint for this supplemental phase. Task1 and its approved review remain completed. TASK-33164 is reopened for AC1 after actual Direct verification exposed an upstream composition defect.

The mounted app reports `capture_provider_failure`; both capture preparation and ordinary Library advertisement call `_library_provider_for_app`, which supplies `collections_service` to a constructor that does not accept it. Other UI/MCP construction sites use the supported signature. RAG-only actual search→natural second-result selection→exact workspace confirmation→restored original history succeeded:4local200/SSE-complete requests and4COMPLETE calls, exact source pins and unchanged predecessor rows/reconstruction. Direct’s2search requests completed; the follow-up did not contact the model. Native Cancel cleared it and Quit closed the disposable app.

ADR required: no
ADR path: N/A; ADR097/166 remain the existing capture/recovery context.
Reason: routine correction of an unsupported call-site argument, preserving the established service contract and all runtime/authority boundaries.

Files: `tldw_chatbook/Chat/console_runtime.py`; smallest real-service factory regression in `Tests/Chat/`; narrowly scoped QA/evidence. No trace helper/factory/runtime/service guard, LocalLibraryToolService API, Collections feature, UI/schema, dependency or permission change.

- [x] Read all shared-factory callers and supported construction references; establish real LocalLibraryToolService factory RED with an app carrying the collections attribute and a RAG branch control.
- [x] Remove only the unsupported argument; preserve supported backend/enforcer composition and frozen mode branch; run smallest covering checks, Ruff/format/whitespace and self-review.
- [x] Commit source/test/QA separately from root metadata; independently review only the supplemental range. No completed old review wave reopened.
- [ ] Root qualifies fresh Direct with the real local model and disposable native UI, at most4more requests under the shared10-attempt cap; retain failures and unchanged original capture evidence.

Live scratch: `/private/tmp/archive-capture-native-_yuzjq8a`; cases rag/direct, both quit through native Ctrl+Q. Worktree/source baseline:75456b7a2d3f8ea4f711f41c061d212abdd03add. Initial setup/terminal/read-only-query assumptions are verification harness incidents, retained separately from the actual Direct constructor failure.

Task2 source correction e4a2717b4158d178d2ec1616d9a8ebd983982c08:15covering checks pass and independent scoped review is Approved/0actionable. [Live QA](../qa/console/2026-09-28-agent-archive-recovery-capture-live.md) qualifies RAG-only. Corrected Direct discovers/loads the tool and searches successfully, consuming attempts7–10; the next request is refused at the shared cap, so the final live checkbox/AC1 remain unchecked. All apps quit0; normal fingerprints unchanged/server idle. User budget-extension request pending; no cap reset/extra request.

2026-09-28: user “yes” approves up to10additional localhost9099requests to finish Direct verification. Fresh direct-complete disposable case retains all old files/counters and uses a total20-attempt observer ceiling. This supersedes only the exhausted10-request live cap; source, existing gates and all other constraints remain unchanged. Run real native search→natural selection→exact confirmation→original-history Open, read-only capture/fixture assertions and clean shutdown. No source work or repeated review/automated sweep.

## Task 3: Compare captured history in the existing canonical projection (2026-09-28)

The extended native Direct search completes five calls (attempts11–15), persists a historical assistant with five thinking blocks, and saves the exact natural follow-up user revision. The follow-up is refused before any successor factory/transport. Native Cancel and Quit close it; five of the authorized twenty localhost requests remain unused. The read-only diagnosis reproduces archive_trace_admitted_rows: raw admitted history uses _tldw_thinking_owner, while agent rows use the existing native-owner representation. The existing canonical preflight projection compares the unchanged histories equal.

Base: 75b671c001e84bf2713c380e30de3e587bcc8ae0.
Diagnosis: .superpowers/sdd/2026-09-27-agent-archive-recovery-capture-on/task-3-diagnosis.md.
Production scope: only _reseal_archive_followup_trace_request in tldw_chatbook/Chat/console_chat_controller.py; reuse _flatten_preflight_messages and existing locator exclusion. Test scope: existing native captured fixture in Tests/Chat/test_console_archive_recovery_followup.py. QA: a narrowly scoped Task3 report and raw evidence under Docs/superpowers/qa/console/. Root owns Backlog, plan, ledger and all actual live qualification.

ADR required: no
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md and backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: routine correction of a representation comparison using the existing validated projection, preserving request/groups, source ownership, schema, authority and service contracts.

- [x] Read the complete reseal/projection/build-request functions and every named reseal/projection caller; use the actual diagnosis and scratch structural evidence. Preserve original Task1/2 qualification and limitations.
- [x] Establish the smallest real production-boundary RED with supported persisted historical thinking: native/SQLite/controller/actual corrected shared Library factory/agent capture gateway, fake only documented external transport and deny network. Search then natural second-result selection must reproduce the original reseal refusal; cleanly cancel any paused preparation and drain work. Preserve raw failures, including the diagnosis setup failure separately from product evidence.
- [x] Compare the admitted side through the existing canonical preflight projection, then exclude only the same existing native/persisted locator annotations used on the transcript side. Preserve validated thinking/continuation groups, immutable request fields/policy/provenance, exact current saved source and current-text-only composition. Retain canonical disagreement rejection. No generic factory/runtime/service relaxation, arbitrary metadata removal, new abstraction/API/dependency, optional RAG-capture repair, UI/schema or authority change.
- [x] Verify real captured thinking-history search→natural follow-up→exact confirmed restore with unchanged saved transcript, sidecar groups, predecessor rows/reconstruction and exact source pins. Include the smallest continuation-owner compatibility control and conflicting-marker refusal; retain existing unknown-owner/changed-history/non-text/refusal/lifetime controls through focused covering checks rather than repeat old broad waves. Require clean teardown and zero network. Run changed-file Ruff/format/scoped whitespace, reporting unchanged controller debt and preserving raw-output whitespace honestly.
- [x] Self-review, record exact commands/exits/raw phases/source hashes and commit only source/tests/minimal Task3 QA/evidence. Report to .superpowers/sdd/2026-09-27-agent-archive-recovery-capture-on/task-3-report.md. No root metadata stage/edit, app/model/server/network request, subagent or full sweep. Independent review receives exactly base75b→Task3HEAD; completed reviews stay closed.
- [ ] Root independently reviews the exact supplemental range, then qualifies a fresh disposable native Direct flow under the remaining five requests if the supported workflow fits. Preserve prior wire files/counters and exact confirmation/capture gates. If budget prevents honest live completion, record that limit and request an explicit extension only after the concrete correction/review is ready. No invented page state, hotpatch, Capture-Off shortcut, normal-profile/server change, integration or cleanup. AC1 stays unchecked until Direct and RAG native behavior is qualified.

Task3 handoff preserves all Global Constraints above. The later user authorization permits root-only bounded localhost live calls; it does not authorize implementer/reviewer network or relax the original feature contracts.

## Task3 scope amendment — exact shared marker handoff (2026-10-18)

The fresh real-boundary regression proves a second composition mismatch after the local reseal correction. The adapter hands the same validated historical owner to the request factory using _tldw_call_thinking_owner; admitted history uses _tldw_thinking_owner. Raw row equality loses the saved descriptor before gateway preparation. The gateway then legitimately filters optional historical replay under current-exchange mode. Marker renaming and later replay filtering are distinct; neither may be treated as permission to relabel changed history.

This amendment supersedes only Task3's reseal-only production-file limit. Production scope is console_chat_controller.py plus ConsoleAgentTraceRequestFactory.build and its _StreamingModelAdapter.chat_call caller in console_agent_bridge.py. Read every actual factory caller and connected projection/attachment functions before editing. Preserve an exact admitted saved descriptor only across the known app-owned marker representation when the same validated owner and complete provider values match. Unknown, missing, contradictory or changed owners/fields must gain no saved authority. Existing gateway selection, source/sidecar validation and generic runtime/service guards stay unchanged. Do not force historical groups back into replay, disable thinking/history, merge aggregate tool thinking into final-call thinking, drop arbitrary private fields, or match on role/content alone. Prefer existing typed projection/attachment machinery; add no new service, config, dependency, schema or authority.

ADR required: no
ADR path: backlog/decisions/090-console-thinking-block-ownership-and-replay.md; backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md; backlog/decisions/166-agent-assisted-archive-recovery.md
Reason: corrective handoff of existing validated ownership through an existing app-owned representation, preserving ADR090 replay filtering and ADR097 exact value/provenance binding. No new ownership/service contract is introduced.

Keep the existing natural-thinking restore regression asserting success. Establish/retain its exact downstream RED before the shared correction, then require full targeted GREEN with clean custody/network guards. Prove exact historical saved revision survives to gateway-bound semantic input while legitimate current-mode replay omission remains honest; predecessor traces and saved text/envelopes stay unchanged. Keep plain Direct/RAG paths, the existing typed-transform/lifetime/refusal cases, continuation compatibility, contradictory marker refusal, unknown/changed owner and non-text/multimodal changes, and active-call versus aggregate thinking separation in the smallest covering selection. Avoid broader test waves. Static-check changed source/tests and report pre-existing debt accurately.

Root owns plan/task/ledger/live; no root metadata stage/edit. Implementer owns the two production files, necessary existing tests and minimal fresh QA/evidence, all in one reviewable Task3 correction from base75b. Do not commit a knowingly failing regression as completion. Report exact raw phases including new downstream RED/setup/teardown failures; expired older tmp evidence remains a stated limitation. Independent review receives the complete supplemental range, not earlier completed review waves. Root live verification remains blocked by the unavailable local server; five model requests remain unused.

## Task3 final classification amendment (2026-10-18)

The exact validated native tool-call thinking rewrite preserves its TOOL_CALL causal input, but the service's reference producer labels the outer transformed row thinking. The unchanged completed-tool suffix validator then raises completed_tool_turn_range. Correct only the new-reference classification of the existing exact typed case; never add generic thinking to the suffix allow set or change the typed THINKING rewrite.

ADR required: no new ADR; existing ADR097 clarification required and written before code.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md#clarification-recorded-2026-10-18-thinking-serialized-tool-call-surface-identity (ADR090/166 still apply).
Reason: exact classification correction under existing typed provenance and replay contracts; no schema, migration, new disclosure, authority or service contract.

Read the new ADR clarification before implementation. This amendment adds only console_trace_service.py reference classification and the smallest necessary existing service tests to Task3's production/test scope. Read all named functions/callers first. Use one shared classification rule for _reference, _reference_key, _durable_reference_matches and _surface_reference_matches. Map to TOOL_CALL only for exact DerivedTraceProvenance MESSAGE_REWRITE, outer THINKING artifact and direct ProviderArtifactTraceProvenance TOOL_CALL first input; preserve all validated attachment/input/policy/value data. Ordinary thinking, other sources/transforms and contradictory descriptors must not be promoted. Runtime/service validation guards stay byte-for-byte unchanged. No per-node legacy inference or migration.

Add the smallest real SQLite producer/lookup/durable-match control covering repeated prefix reuse, plus ordinary-thinking and legacy-node refusal. Preserve the full Direct/RAG plain/thinking recovery success regression, exact source and predecessor evidence, and existing source/lifetime/multimodal/marker/active-thinking controls. Run a targeted selection covering only these changes and the related completed-tool witness boundaries; no full sweep or old broad review wave. Capture all RED/GREEN/static phases honestly. Do not disable thinking or weaken the success regression.

The root-authored ADR097 clarification belongs in the corrective commit and review package; this is the sole exception to the root-metadata staging exclusion. Leave plan, Backlog task and SDD ledger edits unstaged. Source/tests/ADR/minimal QA form one coherent Task3 correction from base75b. Report full status and exact evidence to task-3-report.md; stop on a concrete new load-bearing blocker rather than expand scope again. Root owns the eventual fresh live phase and remaining5-request cap; server still unavailable.

2026-10-18 Task3 source20f3f3e24b:41focused checks pass,10post-format controls pass, independent exact-range review spec compliant/Approved/0actionable. Root now qualifies a fresh captured Direct case at user-corrected localhost8010 with its existing Qwen3.8-27B server. The existing narrowing Persona keeps only2recovery tools advertised to fit the remaining5requests; all Library, current-source, capture/replay and one-shot confirmation gates apply. Counter starts at15spent and refuses beyond20. No old raw/counter reset; expired earlier temp evidence remains reported. Normal profile/server/model settings are untouched.

## Task3 reviewed correction and native8010 boundary (2026-10-18)

Source20f3f3e24b5c9ba5a58f6793af1cad0ad76034ce independently Approved/0actionable; retained41targeted checks and10post-format controls pass. Source/test bytes stayed exact during the live phase. [Fresh native QA](../qa/console/2026-10-18-archive-recovery-live.md) records two COMPLETE Qwen Direct search calls, preserved reasoning_content history and the natural follow-up blocked by ValueError at trace reservation before a successor model call. No restore confirmation/restore/Open qualification. Native Cancel refuses unavailable reservation status; Quit/PTTY exits0, normal profile exact, server healthy/idle unmeasured. Global17of20calls accounted from capture/admission/provider-entry evidence; requests-only observer missed this native HTTPX path, so no new HTTP200/SSE-wire claim. Three authorized calls remain; no reset. AC1 stays unchecked, task In Progress; exact remaining refusal diagnosis is read-only and no additional source expansion is assumed.

Read-only factory reconstruction on a private SQLite backup proves completed_tool_turn_revision: prior version1 frozen response projection requires start_anchored_think, actual Qwen terminal thinking is reasoning_content, canonical projection refuses response_thinking_source_mismatch and settlement keeps a sanitized artifact. Terminal-only thinking still fails; no archive metadata or successor streaming setting is required to reproduce this durable blocker. Producer/projector/settlement/guard match base75b. This is a new frozen-response contract boundary; a prospective future-only versioned profile and exact equality controls await user design approval. No Task4 code/plan/ADR change has begun.

## Task 4: Version newly captured local thinking response profiles

User approval on2026-10-18: Approve versioned profile repair. Spec: Docs/superpowers/specs/2026-10-18-versioned-thinking-response-profile-design.md. Existing Task1–3 implementation/reviews stay complete; the current source baseline is fcca09d3f810e035f5b89b3df09b6bda1c228efd.

ADR required: yes, amend existing ADR097 before implementation.
ADR path: backlog/decisions/097-console-reference-backed-semantic-trace-ledger.md#amendment-recorded-2026-10-18-version2-local-thinking-response-formats
Reason: prospective frozen response-profile contract; no new schema, old meaning or authority. ADR090/166 remain binding.

**Files:** service profile producer in tldw_chatbook/Chat/console_trace_service.py; shared closed format rule and version-aware canonical response projection in tldw_chatbook/Chat/console_semantic_revision.py; existing Tests/Chat/test_console_trace_thinking_settlement.py and Tests/Chat/test_console_archive_recovery_followup.py; the existing _call fixture in Tests/Chat/test_console_trace_settlement.py may accept optional provider/model keyword arguments while preserving defaults and callers. One existing gateway test may change only if necessary to prove the actual structured parser/client seam. Minimal new QA/evidence at Docs/superpowers/qa/console/2026-10-18-versioned-thinking-response-profile*. Root owns this plan, spec, ADR, Backlog and ledger.

**Interfaces:** _response_projection_profile(resolution) produces existing header JSON; _project_revision_response_envelope consumes it for settlement/native reader. Existing exact raw equality proof, typed response envelope, source/terminal selection and completed-tool validators are unchanged. Consume the approved spec verbatim for v1/v2 shapes and canonical formats; use one small shared helper in the already imported semantic-revision module.

- [x] Read the corresponding Backlog task, approved spec/ADR amendment and retained Qwen diagnostic report. Read the complete two changed functions, all callers and connected settlement/parser paths. Check current Docs/Architecture context read-only from original checkout because this worktree predates it. Preserve source20f native evidence and all earlier boundaries.
- [x] RED: extend existing actual gateway/controller/SQLite fixture at documented external adapter seam so actual typed reasoning_content reaches terminal saved-response settlement and a captured successor. Assert exact revision/verified_equal, not merely COMPLETE. Keep fail-closed socket/DNS guard and clean pending-work teardown. Add the smallest v2 shape/eligibility/refusal cases. Run the exact new case before production edits and retain failure separately from harness errors.
- [x] Implement the approved profile contract in the two existing production modules. New displayable eligible local profiles are version2 with source_formats; hosted proprietary and old/no-profile paths stay unchanged. The shared rule returns only the exact approved list for the exact provider/model. Validate complete shape, exact integer versions and ordered source list; retain literal actual source_format and every existing ownership/round/status/type/equality/privacy/completed-tool guard. No parser/replay/native-tools override or old-row rewriting.
- [x] GREEN: targeted tagged/structured response links and natural archive Direct/RAG confirmation/restore; unchanged saved transcript/envelopes and previous call rows/reconstruction; exact source pins; v1/legacy behavior; malformed/widened/reordered/unknown source/provider/model/protocol/round/status/type refusal; raw equality proof and PII/credential-mask collision controls. Extend the native archive fixture with structured reasoning parsed by the production adapter, not a service/factory double. No new test framework, full sweep or old broad review wave. Suggested covering commands use existing thinking-settlement file and selected native archive/thinking/privacy controls.
- [x] Run affected-file static/format checks and scoped whitespace, distinguishing existing baseline debt. Self-review exact two-file production diff, preserve raw RED/GREEN/commands/exits/hashes and minimal QA evidence. Commit only owned source/tests/QA; root spec/ADR/plan/task remain excluded unless explicitly included by root. Report to the extracted Task4 report path with DONE/DONE_WITH_CONCERNS/BLOCKED/NEEDS_CONTEXT, exact commits/tests and limits. No worker subagents, app/model/network, original-checkout edits, integration, cleanup or new contract expansion; escalate a new load-bearing blocker with exact evidence.
- [x] Root verifies exact artifacts/source, independently reviews only this supplemental range and records the result. Then separately prepare a fresh private native run with a tested guard on the actual requests/HTTPX/fallback/auxiliary transport. Preserve17consumed/20total/3remaining; request additional model budget only after the concrete reviewed fix if needed. No more model requests until fresh hard accounting fits the run. AC1 stays unchecked until native outcome qualifies.

## Fresh native qualification after Task4

User approved10additional localhost8010requests, totalcap30 with17already consumed and13remaining. Root uses reviewed sourcef9b8746a, fresh private Direct/RAG fixtures and the offline-tested shared HTTPX/requests cap; no counter reset. Require native search → exact natural second-result confirmation → selected restore → explicit original-history Open, preserved source draft and exact traces/history/equality. Close apps and compare normal fingerprints; keep AC1 open until both qualify. No source edit during app runs, full sweep, server/model/normal-profile change, integration or cleanup.

## Task 5: Carry observed references through the admitted fenced-tool protocol

User approval2026-10-18: Approve fenced-tool handoff repair (Recommended). This is the approved bounded change to the existing controller handoff; no new execution-choice/spec-approval prompt is needed. Source baseline before preflight metadata is ec68d8f1752fcd33b64d8e0b287dd5d0b071e4d2. Task1–4 source and reviews remain complete. Fresh actual resolver proves ready=true/native_tools=false with /props404; native search terminal revision is now verified_equal, but the native-only rider condition omits references. Optional staged-RAG capture failure co-occurs and is not claimed causal.

ADR required: yes, clarify existing ADR166 before code.
ADR path: backlog/decisions/166-agent-assisted-archive-recovery.md#clarification-recorded-2026-10-18-fenced-tool-reference-handoff
Reason: supported protocol eligibility for the existing private metadata handoff; same authority/envelope/destination/confirmation and no storage/schema change. ADR090/097 still govern capture/thinking.

**Files:** modify only tldw_chatbook/Chat/console_chat_controller.py at the rider's provider protocol condition, and Tests/Chat/test_console_archive_recovery_followup.py using its existing real-stack fixture. Minimal QA/evidence prefix Docs/superpowers/qa/console/2026-10-18-archive-fenced-followup*. Root owns this plan, approved design/ADR, Backlog, ledger and live QA. No gateway/bridge/native-tools/provider/runtime/repository/service/profile/optional-RAG production changes.

**Interfaces:** admitted recovery_runtime.followup_context(authority) and the existing reseal/prepend helpers remain unchanged. CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS is already imported in the controller and is the known supported execution-key set. The real bridge chooses native/fenced transport through its existing provider_supports_native_tools policy; do not change that policy or lie about native_tools. Replace only the rider's native-wire predicate with supported execution-key membership and remove its now-unused local import:

```python
and (getattr(resolution, "execution_key", None) or resolution.provider)
in CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS
```

All other conditions stay exact, including native_tool_calls_enabled=true, Persona advertisement/no-ask, kill-reader fail-closed and authentic recovery admission. Unsupported/unknown keys do not receive metadata. No new helper/provider list/dependency or generic gate relaxation.

- [x] Read the task, approved design/ADR clarification, retained live QA2026-10-18-versioned-profile-live.md and actual source boundary before edits. Read the complete touched function/callers and existing supported-provider/bridge protocol selector. Check Architecture console/tools read-only from original checkout if needed; verify against this older worktree.
- [x] RED: extend the existing captured_followup fixture with the exact new mode fenced-structured. Freeze llama_cpp/Qwen, ready=true, reasoning_replay.native_tools=false/server_default, LibraryAllowed and the global tool flagtrue. At the documented external adapter/client seam return real ordinary completion reasoning_content plus content containing the existing fence protocol; no native tool_calls response. The actual gateway/adapter/runtime must parse it. Reuse the existing natural restore test in Direct/RAG, so new parameter cases fail before production on missing metadata/confirmation. Assert the external request has no native tools schema, the native capability staysfalse and socket/DNS attempts remainzero. Clean held work on both RED/GREEN.
- [x] Implement only the shown controller eligibility correction using the already imported constant, with the unused local import removed. Preserve exact rider bytes/size/privacy/destination/current-user ownership, all strict capture guards and every other admission/cleanup condition. No source changes outside the two named files without an exact new blocker report and root scope ruling.
- [x] GREEN: real fenced structured Direct/RAG search → natural second-result metadata → exact target/scope confirmation → only selected restore → ordinary following captured turn. Preserve prior call/header/link/request/response reconstruction, saved text/thinking envelopes and exact current-user pins; require terminal revision/verified_equal. Reuse the native/plain/tagged/structured positives and full existing follow-up refusal controls for LibraryBlocked/frozenblocked, PersonaDenied/ask, killed/unreadable switch, globalnativeflagfalse, unsupported key, temporary/remote/legacy, destination change, expired/manual/imported/foreign references and refusal/cancellation cleanup. Retain unknown provider refusal; no transcript-derived authority. The existing full follow-up test file is the maximum targeted selection; no whole-suite or unrelated thinking-settlement rerun.
- [x] Check changed test Ruff/format/scoped whitespace and compare controller baseline diagnostic signatures/changed-block formatting without formatting unrelated code. Verify gateway, bridge, native_tools.py, runtime/factory/service/profile and five strict service guards remain unchanged. Retain raw RED/GREEN commands/exits/source hashes and minimal QA, self-review then commit only owned source/tests/QA. No app/model/network, normal profile, server/model override, other subagent, integration, cleanup or full sweep. Report to Task5 report path with exact commit/tests and material limits.
- [x] Root independently verifies artifacts and reviews only this supplemental range. After approval, separately requalify fresh native Direct/RAG with the corrected tested HTTPX/requests/props guard, preserving actual22consumed/30cap/8remaining. No counter reset or further model request before the fix/review and adequate hard accounting. AC1 stays unchecked until actual native outcomes qualify.


### Task5 scoped ruling: preserve the existing dispatch correlation

The valid no-id fenced completion reaches the admitted restore handler after the controller repair, but the dispatch passes empty call.call_id and the unchanged recovery runtime refuses Native recovery call identity required. The existing parser accepts fences without call_id; adding a synthetic model id to the fixture would hide the actual supported path. Root read parser, per-call correlation construction, guard/review/dispatch and recovery service/coordinator boundary before this ruling.

Ruling: additionally permit only the request_conversation_restore dispatch argument in tldw_chatbook/Agents/agent_runtime.py to use existing current_call_correlation instead of raw call.call_id. This app-owned nonempty correlation is already assigned to every native/fenced call and used by the proposal, guard, review and dispatch trace. Native ids are preserved under the existing uniqueness rule. Session/run ownership and argument digest continue to form the same exact RecoveryCallKey. The wire ToolCall and native/fenced message semantics remain unchanged; no parser, correlation generator, service, archive runtime/coordinator, guard or confirmation change is allowed. This implements the already approved fenced recovery behavior and retains every check; no new authority or design choice is introduced.

Keep the ordinary fenced fixture without call_id. Retain valid pre-controller RED and post-controller identity-refusal evidence, then GREEN only after the one-line dispatch correction. Assert the proposed/dispatch correlation matches the nonempty recovery call identity and exact confirmation, and preserve native controls. Extend baseline static-diagnostic comparison to agent_runtime.py without unrelated reformatting. All other source boundaries and five strict capture guards remain byte-exact. Maximum targeted test selection is unchanged. Worker owns this one additional production call site; root owns metadata. ADR166 clarification and Backlog plan are updated before this additional edit. Root live remains after review, budget22of30/8remaining.


Task5 root native qualification complete2026-10-18 on frozen sourcef12ba0bee7/metadata2391c827bb. Direct23–26 and RAG-only27–30 each complete real search → natural second-result metadata → exact Linden/Atlas workspace confirmation → approved restoration → explicit original-history Open → retained source draft. Eight HTTP200/SSE-DONE/complete calls, exact current-user source pins, terminal revision/verified_equal, predecessor and original history immutability; Atlas archive scope only, source workspace navigation timestamp only. Both nativeQuit/status0, normal fingerprints/source hashes unchanged, health200/idleunmeasured. Global30of30,0remaining, no reset or further inference. QA: Docs/superpowers/qa/console/2026-10-18-archive-fenced-live.md. Existing optionalRAG/Persona and older excluded issues remain separate; no full sweep, integration or cleanup.
