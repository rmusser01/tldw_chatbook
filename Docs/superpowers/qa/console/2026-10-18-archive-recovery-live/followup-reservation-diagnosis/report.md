# Native follow-up reservation diagnosis — 2026-10-18

Status: DIAGNOSED / native recovery remains NOT GREEN. Inspected production source is pinned to `20f3f3e24b5c9ba5a58f6793af1cad0ad76034ce`; the four inspected source files match that commit byte-for-byte. No production, actual DB, configuration, fixture or repository edits; no app/model/network calls, test sweep or commit.

The actual production factory on a private SQLite backup raises **`ValueError("completed_tool_turn_revision")`**, through `ConsoleTraceBoundaryFactory.__call__:622` → `prepare_current_surface_delta:2696` → `_validate_completed_tool_turn:3549`. It rolls back before reserving a successor: copy call count remains 2. An observation-only `sys.settrace` records that exact exception stack.

## Durable cause and producer

The prior native search calls `372d724e-0a9e-4cc0-971b-4262cd7048ea` (agent_first/sequence0) and `5e7da41b-75b6-4ad4-a7d1-7808d81190cd` (tool_loop/sequence1) are both COMPLETE/integrity complete. Both response links are nevertheless `artifact/sanitized_artifact`; the terminal link points to artifact `9dc40355-2695-4218-a69f-97f370a534b3`, not saved assistant revision `79aba5f6-f434-4d95-93b1-fca57a610e71`.

`console_trace_service.py::_response_projection_profile:6273` is the exact producer: displayable thinking unconditionally freezes `source_format="start_anchored_think"` (line6304), along with provider llama_cpp, model unsloth/Qwen3.8-27B-GGUF:Q8_0, protocol chat_completions, version1. Both prior headers contain that profile.

Actual saved assistant thinking has round0 `reasoning_content:tool_call` and round1 `reasoning_content`, both complete. The terminal call sequence is1. The real `project_semantic_revision_provider_message(..., response_call=terminal)` selects the terminal block correctly and raises **`ValueError("response_thinking_source_mismatch")`** at `console_semantic_revision.py:194` because the terminal block's concrete source format differs from the frozen profile. Provider/model/protocol/status otherwise agree.

`console_trace_settlement.py::_store_response:625` attempts that canonical projection, catches the failure at line671, and stores the sanitized response artifact instead. This is the implemented safe fallback, not proof of canonical equality. The later completed-turn guard correctly requires `revision/verified_equal` tied to the exact saved assistant, so it refuses at line3549.

## Final versus aggregate semantics and controls

The projector selects only blocks whose round_ordinal equals the exact call_sequence. Removing round0 from a detached envelope still raises the same source mismatch on round1. Aggregate tool thinking is not being mistaken for final thinking in this failure. No saved block or stored artifact was altered for either comparison.

A detached historical nonthinking envelope with the same saved role/content and no thinking sidecar projects successfully to exactly role/content under the existing profile. This is a projector-only control, not a full factory success or native nonthinking send. Existing completed Task3 QA already contains successful plain Direct/RAG native-fixture recovery controls; those were not rerun as an old review wave.

## Attribution boundaries

- This is separate from the three corrected Task3 mismatches: native-locator projection, thinking-owner marker preservation, and exact tool-call reference classification. The actual durable suffix is already `tool_call` + `tool_result`; no legacy `thinking` suffix node is present here.
- The producer, canonical projector, settlement fallback, and completed-turn guard have identical AST hashes at base75b and source20f (baseline-comparison.json). This response-format/equality boundary predates the correction.
- Both prior request headers have streaming=true. Changing the successor to streaming=false cannot retroactively change these persisted profiles or response links. The factory refusal occurs before successor request-header persistence or any transport; `_resolution` contributes only api_key to this prepare comparison. The observed failure does not establish a non-streaming transport defect.
- The factory reconstruction uses ordinary exact canonical history/current-user projections and the actual saved revision IDs. No archive metadata is required to reach the identical guard. The native prepared successor payload and raw exception traceback were not retained, so this is a qualified production-boundary reconstruction, not a byte-for-byte replay of the native archive handoff. It proves an independent durable blocker, rather than claiming every upstream native comparison was observed.
- The actual Qwen response carried reasoning_content in the native fenced-tool path. The successful Task3 thinking fixture qualifies its own supported start-anchored representation; this Qwen response source was not established by those checks. No generic Qwen replay policy change or thinking/history disabling is justified by this diagnosis.
- Native Cancel's unavailable-reservation copy follows the gateway's broad ValueError wrapper: `_reserve_trace_call` catches factory construction errors and records reservation_status="unknown" (gateway3031–3037), so no concrete call boundary exists to reconcile. No successor call was actually established in the retained DB.

## Smallest prospective correction proposal — not implemented

Correct the prospective frozen response-profile/source contract for the already supported typed `reasoning_content` path, preserving the concrete parser provenance and exact terminal-round/provider/model/protocol/status equality. The producer and canonical projector must agree explicitly on that supported format; changing only the producer's current scalar to reasoning_content would hit the projector's profile-shape/source contract, so this needs a narrow, documented paired correction. Keep completed-tool revision verification, equality proofs, suffix checks, privacy policy and existing response bytes unchanged. Do not reinterpret reasoning_content as start-anchored text, accept arbitrary source mismatches, or weaken the completed-turn validator. Existing artifact links/profile rows remain immutable and fail closed; fresh capture is required to qualify corrected prospective behavior. Root owns any scope/ADR decision and live retest.

## Evidence and custody

Command: `PYTHONDONTWRITEBYTECODE=1 /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python /private/tmp/archive-recovery-followup-reservation-diagnosis-20261018/probe.py > /private/tmp/archive-recovery-followup-reservation-diagnosis-20261018/probe.log 2>&1` — final exit0 with explicit assertions for both exact errors, unchanged source inputs, zero network attempts and unchanged call count.

The source SQLite was opened only via URI mode=ro and query_only for inspection, and mode=ro SQLite backup into diagnostic-copy.sqlite. All CharactersRAGDB initialization/factory work runs against that private backup. Subsequent DB audit guards allow only paths within the diagnostic directory; socket/DNS guards deny all network. File writes are confined there, plus /dev/null. `inputs-before.json` and `inputs-after.json` hash all retained JSON/TOML/DB/log/WAL inputs selected at diagnosis start; all entries are identical. Shared SQLite SHM coordination is excluded from content custody.

`rows.json` retains actual durable rows and artifact values; `schema.log`, `rows-summary.log`, `result.json`, `probe.log`, and `baseline-comparison.json` retain raw evidence. Two preliminary probe-only failures are preserved: phase1 audit denied /dev/null during DB initialization subprocess setup; phase2 output serialization rejected an immutable mappingproxy, then was corrected by copying the profile to a plain dict. Neither is a product failure. The earlier inspection-only sqlite URI invocation failed to open because it did not set uri=True; the explicit read-only URI connection succeeded and hashed inputs were unchanged.
