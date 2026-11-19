# Version2 local-thinking response profiles — Task 4

Status: DONE_WITH_CONCERNS (source/fixture qualification only; native live outcome remains root-owned).

Baseline: `38cf68612508d23e9ae70a5deecf76a913fff3e0`, owner-approved spec/ADR097 preflight. Governing design: [approved profile contract](../../specs/2026-10-18-versioned-thinking-response-profile-design.md); ADR090/097/166 remain binding. This is prospective capture behavior, with no migration or reinterpretation of earlier rows.

## Change and exact boundaries

The existing trace service produces version2 for displayable `llama_cpp`, `local_llamacpp`, `vllm`, and `local_vllm` profiles with a nonblank string model. A shared helper in the already imported semantic-revision module returns exactly the ordered JSON formats `start_anchored_think`, `reasoning_content`; case-insensitive Kimi models retain tagged-only support. The reader validates the full version-specific shape and integer versions, exact provider/model-derived format sequence, and literal terminal source evidence. It keeps v1 scalar and legacy visible-only meanings, plus proprietary hosted v1 profiles.

The repository already freezes decoded JSON arrays into tuples. The reader validates that exact frozen tuple; native tests separately assert the persisted JSON array and frozen header tuple. No additional serialized shape or repository change is admitted.

Only two production files change. Gateway/parser, settlement/equality HMAC, native reader, runtime, controller, repository, provider, replay/native-tools policy, permission and schema are unchanged. Five strict service guard ASTs match baseline: `_validate_completed_tool_turn`, `_completed_tool_turn_parent`, `prepare_current_surface_delta`, `persist_request`, `_validate_replacement_projection`. Exact file/guard hashes and Ruff baseline comparisons are in [summary.json](2026-10-18-versioned-thinking-response-profile/summary.json).

The existing actual gateway/controller/native-runtime/SQLite fixture replaces only the documented external adapter seam, returning ordinary structured message `reasoning_content` through the production parser. Fail-closed socket and DNS guards remain enabled, with clean paused/pending-turn cleanup. Plain, tagged, and structured natural recovery work in Direct and RAG-only modes. Tests require terminal `revision/verified_equal` links tied to exact saved assistant/source conversation, including the captured restore successor; exact target UI confirmation precedes the one selected restore. The ordinary following turn still works. Headers, links, request/response reconstructions, saved transcript and thinking envelopes from prior turns remain equal.

The settlement matrix covers v1/v2 raw credential-mask collision refusal, structured/tagged PII and PII mismatch, queued equality proof, source/provider/model/protocol/round/status/type mismatches, missing/extra/scalar/empty/null/reordered/widened profiles, integer-version refusal, Kimi restriction, legacy visible-only behavior, hosted v1, and historical locator retirement. The test-only `_call` helper adds optional provider/model keywords with unchanged defaults; root explicitly approved that extension.

## Raw phases and verification

[commands.json](2026-10-18-versioned-thinking-response-profile/commands.json) records exact command selectors, exits, raw log sizes and SHA256. Gzip artifacts decompress to byte-identical raw stdout/stderr. No old phase was overwritten.

- Native RED: exit1, two structured cases both fail because the parsed terminal response links to an artifact instead of the saved revision. This is valid product RED before production edits.
- Settlement RED: exit1, 32 failed/66 passed. Six are valid product RED: v2 exact, fragmented, PII, queued, structured-format and Kimi-tagged links become artifacts. The other 26 are harness failures: 24 incomplete SimpleNamespace producer fixtures lacked protocol attributes; two duplicate-round envelope fixtures were rejected by existing serializer validation before reaching the projector. They are explicitly excluded from product RED. Corrected tests use the real typed resolution and preserve existing envelope refusal.
- First GREEN attempt: exit1, 36 failed/68 passed. It exposed the repository's frozen-array tuple representation; the above harness issues were also retained. The reader/test representation was corrected without changing wire meaning or repository code.
- Covering run: exit1, 149 passed plus three unrelated settlement failures, listed below and reproduced against baseline.
- Final-control harness phase: exit1, 134 passed plus five ignored-thinking resolutions rejected by the existing typed constructor because the fixture set round-trip version1. The fixture now uses None for ignored disposition; no production validator changed.
- Final GREEN: exit0, **139 passed in 69.46s** across the full thinking-settlement file, six plain/tagged/structured Direct/RAG natural recovery cases, and two exact-authority custody interruption cases.
- Changed evidence-helper callers: exit0, **13 passed in 23.13s** for canonical history/marker owners, transform composition, and unowned/non-text refusal.
- Static: exit0 for comparison script, formatter checks and scoped whitespace. Both authored profile/recovery test files lint clean. Production projector/service retain exactly 4/33 baseline Ruff diagnostic signatures; settlement helper retains exactly 2. All five changed files pass Ruff format. No additional signatures.

## Named concerns and custody

The wider settlement-helper coverage discovered three failures that also fail with the baseline projector, profile producer and original full test module/helper in a read-only in-memory baseline overlay:

1. `test_oversized_response_is_replaced_by_one_bounded_labeled_artifact`: `credential_sanitizer_unavailable` instead of `response_size_limit`.
2. `test_queued_settlement_drops_oversized_response_and_usage_values`: prepared response bytes are None.
3. `test_cold_restart_recovers_open_calls_monotonically_and_idempotently`: recovered call set omits the expected `not_dispatched` call.

Baseline overlay phase2 exits1 with the same three failures and three assert-rewrite warnings. No unrelated source fix was attempted. The baseline bootstrap phase1 imported production before pytest isolation, read normal config metadata, and was denied opening the normal data-root lock by the sandbox (`PrivatePathError`, PermissionError). It made no successful normal-profile write, app or model call. The corrected retained script loads existing Tests.conftest isolation before any production import; all tests then use scratch HOME/XDG/config/data. Root owns normal-profile fingerprint verification. The initial test-edit attempt was also denied by sandbox before any write; approved isolated edits succeeded on escalation. Initial RED logs retain unrelated old pytest temporary-directory cleanup warnings; later commands use explicit unique basetemp paths, with no cleanup of old directories.

Source fixture success does not qualify a live Qwen server/template/native-tool/replay contract, fix the historical source20f rows, or prove the remaining native Direct journey. Root owns live runs and accounting (17 of 20 used, 3 remaining; worker makes zero model requests). AC1 remains unqualified here. Nonempty Persona provenance, historical unsupported_surface_change, aggregate FD growth, provider errors, dictionary second-send and Next Send preview limits remain outside scope.

No worker subagents, live app/model/server/network, full sweep, download, normal-profile edit, push, PR, merge, worktree cleanup, new dependency, history rewriting or authority expansion. Root-owned task/plan/spec/ADR/ledger and transport-guard QA are excluded from this source commit.

## Final source hashes

- `tldw_chatbook/Chat/console_semantic_revision.py`: `5cfe202a5f14262148a7be71ec9b1b3b674d005a0e1815dcc8f99912eebd179f` (SHA256).
- `tldw_chatbook/Chat/console_trace_service.py`: `c768bdf29d1fb003ca5806b0423864b132eccd78c60693903dbe504f9e6afc2f` (SHA256).
