# Console approval provider owner extraction implementation plan

> For agentic workers: execute this bounded plan with superpowers:executing-plans after Root confirms AC12/global-plan readiness. Root owns final review, ratchet pins, CI inventory and commit.

**Goal:** Remove the four PR-added module growths without changing behavior, budgets or authority.
**Architecture:** Mechanical private extraction into the existing tool-presentation owner and two narrow provider/service helpers. Original public methods, decorators, callback seams and owning objects remain.
**Tech Stack:** Python, existing context managers/locks and targeted protected pytest.
**Spec:** Docs/superpowers/specs/2026-10-05-console-approval-ux-and-responsiveness-design.md; TASK34569 AC12/global resync plan (Root readiness required).

ADR required: no. ADR paths: backlog/decisions/195-console-live-tool-call-presentation.md; 235-console-approval-interaction-and-feedback.md; 300-local-agent-tool-permission-boundary.md; 325-workspace-assistant-defaults.md. Reason: private same-layer mechanical movement, no new authority, storage, callback or policy contract.

## Global constraints

- Source before edits:193a2c5cd8; incoming base:a0ae30b7ed9c313fe19cb3a0da5d95ebde182385.
- Ceilings remain5854/12092/6054/8556 for local provider/bridge/MCP service/transcript.
- No compression, padding, cap raises, broad refactors, optional import prewarming, changed grants, worker/thread/currentness/privacy/error contracts.
- No controller/interrupt/runtime, feedback store/UI feedback, RAG, central pins/census/inventory/launcher or Task edits. No App/GUI/full suite/Git mutation.
- Protection: existing finite private_control.py, Main/Core/Tests pins, original-home guard before redirect, genuine startup admission. Finite target admission by Root only.

## Exact files and interfaces

1. Create `tldw_chatbook/Agents/local_approval_state.py`:
   - `local_stamp_scope(provider, run_id)` moves the existing `LocalToolProvider.stamp_scope` body, including observation scope and original lock/save/clear/restore sequence.
   - `persist_local_approval(provider, hub, decision, *, log)` moves `_persist_approval_safe` body, original writer and observation outcomes/error behavior.
   - Modify only the two original provider methods into lazy delegates; keep signatures/contextmanager and provider-owned dictionaries/locks.
2. Create `tldw_chatbook/MCP/session_approval.py`:
   - `apply_session_approval(service, server_key, tool_name, *, profile_id, expected_profile_digest, expected_revision, publish, lifecycle, policy_digest, mutation_error)` moves `approve_for_session` body.
   - Original decorated method/signature remains, forwards original service, arguments and CURRENT module globals for the four dependency seams. Store fence, profile checks, lock/set identity and publication ordering remain exact.
3. Extend existing `tldw_chatbook/Chat/console_tool_activity.py`:
   - `project_approval_feedback(bridge, session_id, run_id, *, replace_presentation)`.
   - `observe_approval_step(bridge, session_id, run_id, step)`.
   - `clear_raw_shell_progress(bridge, run_ids, *, cancelled, log)`.
   - `retire_child_approval_feedback(bridge, run_id, observation_owners)`.
   - `finish_approval_activity(bridge, session_id, tool_activity, tool_run_ids, raw_run_ids, cancelled, providers, *, log)`.
   - `apply_raw_shell_progress(bridge, run_id, call_id, event, *, result_type, stream_type, replace_presentation, step_factory)` moves the entire original feature-touched progress body; it subsumes the preliminary backend-only helper.
   - Preserve original bridge method wrappers and original passed objects. Final projection/raw settle/activity finish/retirement stays before buddy cleanup; terminal callback remains after original usage retirement.
4. Same existing tool-activity owner:
   - `format_tool_activity_detail(activity, presentation) -> str` moves the transcript call-detail formatting block: conditional raw metadata, conditional lazy approval feedback, arguments, output choice.
   - Transcript `_activity_components` calls it only in existing call_id branch. Widget creation, Content, signatures, selection and layout remain there unchanged.
5. Create `Tests/Architecture/test_console_approval_owner_extraction.py` for focused delegate/writer/dependency argument identity, original error identity, exact scope restoration, literal detail output and guarded fresh lazy-import behavior. Full normalized body-AST parity is retained privately as extraction evidence, not a permanent mirrored-body test. Existing affected modules provide actual flow checks.

## Review focus

- Nested or raising stamp scope restores only its exact run and original mapping/lock semantics.
- Stale/missing/imported profile grant refuses before cache mutation; real callback sees completed mutation and original exception object.
- Optional observer failure cannot change writer result, run terminal callback, or other sibling state.
- Late/raw/projection events stay tied to the same session/run/call, with no eager display-module import.
- Detail strings remain byte-equal for ordinary/raw/terminal/missing result and feedback, with the same lazy formatting branches.

## Execution steps (held until Root signal)

- [ ] Root records AC12/global plan and finite-admits exact targets.
- [ ] Capture protected actual module-size RED for only the four parameterized paths, retaining hosted/current counts.
- [ ] Write one small failing delegation/parity test group for the first unit; run protected RED before moving it.
- [ ] Move the exact unit, retaining original source snapshots/body AST hashes; run focused GREEN. Repeat for local scope/persistence, service grant, bridge projection/retirement and transcript detail.
- [ ] Compare normalized original bodies (only owner/dependency names differ), public signatures/decorators and current module-global seam values; prove cold optional-import membership unchanged.
- [ ] Run existing affected observation, local stamp/grant, MCP grant/fence/concurrency, bridge/raw/tool-activity and transcript feedback checks via finite module/-k selections.
- [ ] Run actual four-path growth GREEN; report exact counts to Root for shrink-only ratchet pinning. Do not edit central budgets.
- [ ] Root requests independent final source review and canonical inventory refresh; no commit or App capture by implementer.

## Exact finite admission requests

- `Tests/Architecture/test_module_size_ratchet.py`: `-k "test_module_does_not_grow_past_its_budget and (local_tool_provider or console_agent_bridge or unified_control_plane_service or console_transcript)"`.
- `Tests/Architecture/test_console_approval_owner_extraction.py` (new bounded module).
- `Tests/MCP/test_unified_control_plane_service.py`: `-k "session_approval_listing_survives_concurrent_grants_and_revokes or clearing_one_profiles_session_approvals_keeps_the_same_set_object"`.
- Already admitted existing observation/localprovider/control-plane-permissions/Chatfeedback/toolactivity/rawprogress/UIfeedback/preimport targets will use narrowed relevant `-k` expressions, not broad sweeps.

Status: Root accepted Domain2 symbol plan and recorded AC12/global plan before product edits. The focused test module is a draft pending content review and finite launcher admission; it has not run. All product edits and CPU tests remain held until Root current UI3 replay END. Root links this appendix from the global plan; implementer does not edit that central file.

## Accepted C amendment before its edit

After the initial accepted moves and9behaviorGREEN, actual bridge size remained12102 against12092 (other three owners were below their unchanged limits). Root accepted moving the entire42-line feature-touched `raw_shell_progress_sink` body into `console_tool_activity.apply_raw_shell_progress`, keeping the original method/signature as a lazy delegate with the exact live RawCliResult, RawCliStreamEvent, dataclass_replace and AgentStep objects. The full original lock, preview, clipping, clock, result short-circuit, Unicode refusal, state update and backend observation order move together. This subsumes the preliminary backend-only helper rather than adding another projection layer. Original snapshots and the first11-check parity receipt remain preserved; final proof normalizes only owner/dependency names against the original whole body. No unrelated move or whitespace compaction.


## Follow-up: isolated file-protection test source selection (before edits)

ADR required: no. ADR path: N/A. Reason: test-only correction using the established real config-source selection helper; production security and path policy remain unchanged.

1. Preserve the historical raw_source_selection_changed failures. Trace the two admitted denial nodes and all explicit config consumers before modifying their setup.
2. Select each current per-test TLDW_CONFIG_PATH with Tests.Backup_Recovery.config_test_support.select_config_source. Bind vars(RAG_Search.simplified.config) for its three module-level config accessors; file_operation_tools, local_tool_impls, workspace_file_roots, patch_tool_impls, path_validation, sensitive_paths and canonical rag_state_paths import config lazily and need no namespace rebinding.
3. Assert current HOME/config paths belong to tmp_path before selection; set RAG_PERSIST_DIR to owned tmp_path/chroma before resolving RAG state. Verify every synthetic candidate is within tmp_path before any marker mkdir/write. Retain every existing denial/no-content assertion and the actual tool calls. No bootstrap_profile marker, guard reset, production patch, skip or xfail.
4. Hold pytest during the quiet UI3 diagnostic. After Root releases CPU work, run only the two already admitted exact nodes through protected private_control.py, retaining actual assertion outcomes, stdout/stderr, commands and source hashes. Request independent review of the narrow setup diff before claiming the denial checks verified.


### Windows fixture amendment before edit

The two once-only runs reached original denial assertions: Chroma passed; the cross-family node failed at the SSH oracle. Source tracing shows its marker helper uses per-test HOME while Windows pathlib/ntpath.expanduser uses USERPROFILE retained at the bootstrap test home. Root authorized binding USERPROFILE to the already-asserted-owned HOME in this target only, before select_config_source, then running exactly that node once. Preserve the failing receipt, all existing containment/denial assertions, other environment variables, global fixtures and production guards unchanged.
