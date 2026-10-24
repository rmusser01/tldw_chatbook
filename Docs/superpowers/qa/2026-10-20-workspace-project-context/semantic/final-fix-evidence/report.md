# Final semantic fix wave

FIX_BASE: `0d618403e19e454a4f3433ec2c10dd2f3c54488b`.
Worktree: `C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui`.
Branch: `codex/workspace-project-semantic-index`.
One implementer, no subagents. Root-owned Task36423 and semantic/final-reviews are excluded. Existing ADR241 and ADR103 were amended before the corresponding interface/workflow changes; no schema, backend or dependency was added.

## Five findings and final behavior

1. **Eligible sources (P1).** `WorkspaceProjectIndexDB._semantic_eligibility` reuses the existing keyword policy/exclusion predicate. Source metadata keyset SQL filters before loading text; service passes fresh authoritative exclusions to source, append and publish. Append validates the next bounded eligible IDs and exact source content; publication counts only eligible sources and refuses a retained forbidden member even if counts coincide. Stale scopes preserve prior refusal behavior. Partial caches can retain forbidden rows without disclosing their text to the provider or blocking eligible completeness. No binding-sized source set or per-page full count was added.
   - `Tests/Workspaces/test_project_semantic_generations.py::test_partial_retained_sources_filter_before_text_and_complete_eligible_set[prefix|suffix|exclusion]` covers retained Partial rows, changed-floor append, equal-count membership/completeness and revoked-scope refusals.
   - Existing selected `::test_model_free_local_batch128_tail_and_native32_commits[normal]` now retains a forbidden synthetic row under current binding exclusion and asserts its sentinel never enters the provider batch. The actual service builds129 eligible chunks as128+1, commits32+32+32+32+1 and retires its native owner. `[max_dimension]`, manifest/checkpoint/zero-member controls passed separately on final source.
2. **Exact destination (P1).** Project wrapper configuration forces the existing OpenAI transport to `allow_redirects=False`; any3xx is a classified request refusal before JSON processing or resend. The flag is excluded from serialized catalog identity, preserving saved fingerprints. Default Library transport retains its prior redirect behavior. Requests retry/cancellation/current-consent and actual Session ownership remain in their existing seams.
   - `Tests/RAG/test_embeddings_lib_logging.py::test_openai_redirect_respects_project_destination_without_changing_library[True-307|True-308|False-307|False-308]` uses two real localhost HTTP servers. Both project cases deliver the synthetic body only to the approved server; the unapproved server receives zero project bodies. Both Library controls follow the redirect. Server threads and Session close positively complete.
   - The issued-carrier peer test also asserts the actual project wrapper's parsed factory config disables redirects. The full transport logging/factory file passed on final source.
3. **Independent retirement (P2).** New semantic work still checks all same-binding pending closes. `ProjectionExecutionScope.close` retains issued/live-parent, exact execution/thread and zero-child checks but asks the existing service guard only about that carrier's own wrapper tokens. A peer's ordinary or failed close cannot prevent an already-admitted carrier from physically closing. Own failed close remains retained and blocks its own carrier until positive same-wrapper retry.
   - `Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement[False|True]` uses real issued carriers and `_AdmittedStoreFactory` on two owning threads. A held real factory-record close overlaps the peer's close; the first close succeeds or subsequently fails. Both owners retire without generic retirement poisoning, and new semantic issuance is denied while pending.
   - Original `::test_failed_project_factory_close_retains_real_parent_until_retry` confirms persistent own-failure, GC rooting, keyword evidence, shutdown refusal and successful same-owner physical retry. Original `::test_issued_scope_uses_live_parent_and_rejects_copy_after_close` confirms native-child and identity/copy gates. These final-source controls passed; no synthetic carrier substitutes for those ownership checks.
4. **Disclosure revocation (P2).** Public search captures its compatible Ready generation when ranking. Candidate and final fences compare current independently consented settings/revision and that exact compatible generation. A mismatch returns unavailable with no semantically selected hits; keyword scope/manifest equality cannot relabel stale semantic selection as ready.
   - `Tests/Workspaces/test_project_semantic_search.py::test_service_search_fuses_synonym_hit_then_live_validates[disable|reconfigure]` runs public search on a worker, holds its live file read after semantic ranking, changes the independent semantic consent while keyword status stays equal, and releases the read. Both reject stale disclosure. `[None]` retains the ordinary synonym hit control. These use real SQLite/ranking/live file text and deterministic query vectors, without a real model claim.
5. **Runtime classification (P2).** `qualified_project_semantic_sqlite` reads actual SQLite version/source/options and positively closes its owned in-memory probe with `contextlib.closing`. Only native-positive semantic allocations explicitly request this fixture across settings/generation/search/app-quality, issued-scope, selected UI and allocation tests. Vector/merge, read-only status, keyword and portable refusal controls stay active. The prior combined lease test was split so wrong-lease refusal remains portable. Production exact-source admission is unchanged and fail-closed.
   - `Tests/Workspaces/test_project_semantic_settings.py::test_native_positive_guard_skips_unqualified_runtime_but_keyword_stays_active[version|source]` deliberately invalidates metadata for refusal only, catches the precise fixture skip, proves keyword operation and actual production semantic allocation refusal/rollback. No metadata is spoofed to obtain a positive.
   - Required Windows bootstrap still asserts real Python3.12.10/SQLite3.49.1/source ID/options before pytest and asserts SentenceTransformers absent. Its final44 cases executed with zero skips. No actual alternative SQLite/Ubuntu runtime was executed here.

## Actual execution and immutable originals

All targeted calls used the existing `run-task4-hosted-fixture-targets.ps1` and main core interpreter `C:/Users/GDesktop-1/Working/Github/tldw_tui/.venv/Scripts/python.exe`. Each used a unique run name and fresh exact `repair-sdd-test-trust.ps1` preflight. Trust output for SDD and native-temp was literally `trusted_directory trusted_existing_directory` in every run. Root was notified before each private group and stopped all filesystem writes through full physical retirement. Source edits occurred only between retired groups. No timeouts, forced terminations or widened deadlines occurred.

The actual launcher form was:

```powershell
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/repair-sdd-test-trust.ps1 -RunName <recorded trust name>
C:/Python312/python.exe .superpowers/sdd/2026-10-20-workspace-project-semantic-index/snapshot-final-fix.py <recorded phase>-pre
& ./.superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-task4-hosted-fixture-targets.ps1 -RunName <run> -Selectors @(<exact selectors below and in pre.json>)
```

For the final lane, the actual selector array was read with `Get-Content -Raw .../final-fix-core16-targets.json | ConvertFrom-Json` and passed as `@($finalSelectors)`. The target JSON was extracted via stdlib AST from the actual `PROJECT_INDEX_TARGETS` tuple, without importing tests. Every `*-pre.json` retains the complete literal Python bootstrap `argv`, exact cwd/selectors and runner hashes; this is the exact child invocation record. Outer PowerShell spelling above is a launcher record, not a substitute for that serialized Python argv.

| Run | Literal terminal result | Interpretation |
| --- | --- | --- |
| `final-fix-red-01` | `9 failed, 3 passed, 1 warning in 110.19s (0:01:50)` | Expected source3/redirect2/disclosure2 RED; peer2 setup failures (Rig factory has no retain), preserved adverse. |
| `final-fix-red-02` | `4 failed, 1 warning in 41.82s` | Corrected real-runtime peer2 RED: project_embedding_close_pending. Runtime guard2 RED: missing fixture. |
| `final-fix-green-01` | `15 passed, 1 warning in 160.30s (0:02:40)` | Initial five-finding GREEN; earlier23-path snapshot, before the later membership/stale-scope/portable-split refinements. |
| `final-fix-compat-01` | `26 passed, 1 warning in 156.19s (0:02:36)` | Final-source compatibility, ownership, transport and exact CI-target contract. |
| `final-fix-core16-01` | `44 passed, 1 warning in 314.19s (0:05:14)` | ONE exact approved16-selector required lane; final source, zero skip/failure/error. |

`final-fix-red-01` and `red-02` are preserved as adverse/RED originals, never relabeled as successful. The first peer fixture error is not evidence of the retirement defect; RED2 reached the actual intended refusal after correcting the fixture to the real runtime. Final-source qualification is compat/core16, not the earlier GREEN source.

Each parent and original private child log/XML pair was copied immediately after that run retired. Final core parent XML records `tests=44 failures=0 errors=0 skipped=0 time=314.190`, with10 private child pairs. Compatibility records26 tests, zero failure/error/skip and3 private child pairs. Child logs have the existing Pydantic warning and pytest-asyncio setup deprecation; no executor-join, unhandled-thread, unretrieved-exception or unfinished-shutdown warning was observed. Retained `stage-timing` JSON from GREEN/compat/core16 includes actual `native_retired=true` and bounded append/batch observations. Passed exact-source assertions establish owner/native-child closure; absent process observations are additional checks, not substitutes for native retirement.

Exact raw snapshots before/after RED1, RED2 and GREEN cover23 paths each; final compatibility and core16 cover30 paths each. Every pair compared equal before further edits. The30-path final manifests include all changed implementation/test/CI/contract/docs paths and relevant unchanged project seams, while excluding root's Task36423. Runtime pre/post hash receipts independently agree. No product/test/workflow bytes changed after final compatibility began.

Origin precision: the original trusted runner does not emit per-import `__file__` traces. Its saved cwd is this worktree; `Tests/private_profile.py` explicitly fixes each child's cwd to the repository containing that helper, and root conftest inserts that exact repository first in `sys.path`. Separately retained `final-fix-current-origins.json` is a post-retirement stdlib PathFinder resolution and current byte/hash check for8 production modules. It performs no product import and is **not** claimed as a historical child import trace.

## Exact selectors per run

`final-fix-red-01`:

```json
[
  "Tests/Workspaces/test_project_semantic_generations.py::test_partial_retained_sources_filter_before_text_and_complete_eligible_set",
  "Tests/Workspaces/test_project_semantic_search.py::test_service_search_fuses_synonym_hit_then_live_validates",
  "Tests/RAG/test_embeddings_lib_logging.py::test_openai_redirect_respects_project_destination_without_changing_library",
  "Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement"
]
```

`final-fix-red-02`:

```json
[
  "Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement",
  "Tests/Workspaces/test_project_semantic_settings.py::test_native_positive_guard_skips_unqualified_runtime_but_keyword_stays_active"
]
```

`final-fix-green-01`:

```json
[
  "Tests/Workspaces/test_project_semantic_generations.py::test_partial_retained_sources_filter_before_text_and_complete_eligible_set",
  "Tests/Workspaces/test_project_semantic_generations.py::test_model_free_local_batch128_tail_and_native32_commits[normal]",
  "Tests/Workspaces/test_project_semantic_search.py::test_service_search_fuses_synonym_hit_then_live_validates",
  "Tests/RAG/test_embeddings_lib_logging.py::test_openai_redirect_respects_project_destination_without_changing_library",
  "Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement",
  "Tests/Workspaces/test_project_semantic_settings.py::test_native_positive_guard_skips_unqualified_runtime_but_keyword_stays_active"
]
```

`final-fix-compat-01`:

```json
[
  "Tests/Workspaces/test_project_semantic_generations.py::test_partial_retained_sources_filter_before_text_and_complete_eligible_set",
  "Tests/Workspaces/test_project_semantic_generations.py::test_generation_pages_checkpoint_and_ready_survives_failed_rebuild",
  "Tests/Workspaces/test_project_semantic_generations.py::test_ready_zero_members_is_distinct_from_unavailable_semantics",
  "Tests/Workspaces/test_project_semantic_generations.py::test_generation_rejects_source_manifest_change_before_checkpoint",
  "Tests/Workspaces/test_project_semantic_generations.py::test_multipage_generation_refuses_changed_manifest_at_page_or_final_cas",
  "Tests/Workspaces/test_project_semantic_generations.py::test_model_free_local_batch128_tail_and_native32_commits[max_dimension]",
  "Tests/RAG/test_embeddings_lib_logging.py",
  "Tests/RAG/test_project_projection_scope.py::test_invalid_project_carrier_refuses_before_source_or_model_effects",
  "Tests/RAG/test_project_projection_scope.py::test_issued_scope_uses_live_parent_and_rejects_copy_after_close",
  "Tests/RAG/test_project_projection_scope.py::test_failed_project_factory_close_retains_real_parent_until_retry",
  "Tests/CI/test_ci_queue_pressure_contract.py::test_project_index_required_lane_has_real_pinned_sqlite_and_bounded_targets"
]
```

`final-fix-core16-01`:

```json
[
  "Tests/DB/test_project_semantic_schema.py",
  "Tests/Workspaces/test_project_semantic_settings.py",
  "Tests/Workspaces/test_project_index_allocation.py::test_group_semantic_peer_qualifies_keyword_target_before_native_open",
  "Tests/Workspaces/test_project_index_allocation.py::test_retained_generation_after_revoke_qualifies_peer_and_closes_probes",
  "Tests/Workspaces/test_project_index_allocation.py::test_pure_keyword_group_uses_keyword_allowance",
  "Tests/Workspaces/test_project_semantic_generations.py::test_unknown_registry_retirement_roots_model_and_native_owners",
  "Tests/Workspaces/test_project_semantic_generations.py::test_model_free_local_batch128_tail_and_native32_commits[normal]",
  "Tests/RAG/test_project_projection_scope.py::test_project_missing_local_package_is_setup_required",
  "Tests/RAG/test_project_projection_scope.py::test_selected_keyless_compatible_endpoint_and_default_key_requirement",
  "Tests/Workspaces/test_project_semantic_search.py::test_incompatible_ready_artifact_skips_query_model",
  "Tests/Agents/test_project_index_tool.py::test_scan_to_public_search_casefolded_prefix",
  "Tests/Workspaces/test_project_semantic_search.py::test_service_search_fuses_synonym_hit_then_live_validates[disable]",
  "Tests/Workspaces/test_project_semantic_search.py::test_service_search_fuses_synonym_hit_then_live_validates[reconfigure]",
  "Tests/RAG/test_project_projection_scope.py::test_peer_pending_close_does_not_poison_admitted_carrier_retirement",
  "Tests/RAG/test_embeddings_lib_logging.py::test_openai_redirect_respects_project_destination_without_changing_library",
  "Tests/UI/test_project_semantic_consent.py::test_stage_without_disclosure_preserves_keyword_and_cancel_makes_no_provider_call"
]
```

## CI and static verification

Root recorded GO before workflow edits. The same approved Windows Project index contracts job now selects16 targets (12 previously), yielding44 cases after two runtime-guard controls and the portable lease split. Four added selectors cover held Disable, held reconfiguration, both peer-close parameters and all four redirect parameters. Existing selected local128 normal covers provider source privacy; the complete settings file covers classification. Job cap1200 seconds, per-test180, Python pin/source assertions, dependencies, UI shard count and required aggregation remain unchanged. Measured initial added-node cost91.3 seconds supported the prospective284-second estimate; the actual complete final lane took314.19 seconds. This is a local exact-lane result, not a hosted final-head CI pass.

Changed15 Python files passed Ruff check and AST parse. Actual formatting was limited to modified regions to preserve unrelated baseline formatting. A final read-only formatter check covered103 modified regions with0 failures; full literal argv/output are retained in `final-fix-format-check.json`. Final scoped lint output: `All checks passed!`; all argv and literal stdout are retained in `final-fix-lint-current.json`. Explicit20-path `git diff --check` returned0. Self-review checked the five actual producer/consumer/cleanup paths and scoped CI diff. No full suite, install, download, new environment, shared-environment update, original-checkout edit or unrelated backup timeout fix was performed.

## Final current source hashes

- `tldw_chatbook/DB/Workspace_Project_Index_DB.py`: `2858160ee0e8b64b1b246b0829aa150a1dcc8a9269f1fb5da619849a03b223ae` (93526 bytes).
- `tldw_chatbook/Embeddings/Embeddings_Lib.py`: `51e4b4ecffe9cf7cbe8985285ee56c084e99d0a0f108fdee807b37c2e6a4d86f` (49193 bytes).
- `tldw_chatbook/RAG_Search/projection_scope.py`: `07fdd471ffdef623624d5507132db23229b9e53aab4493748bcb1c1933f492cb` (9326 bytes).
- `tldw_chatbook/RAG_Search/simplified/embeddings_wrapper.py`: `ce7c94a1a250d48760ff2aa8672da464f2264c100b04fbeb689c8cd96bfd2faf` (59127 bytes).
- `tldw_chatbook/Workspaces/project_index/service.py`: `bca40d4cd5f6e6e94d01529029b94790439ecedeea217c38c2e8c6e3e89ca8fc` (101279 bytes).
- `tldw_chatbook/Workspaces/project_index/semantic.py`: `d0351bb50d2753859330a871c058b4006c5944082f0388d197920a5912afb715` (12275 bytes).
- `tldw_chatbook/Workspaces/project_index/allocation.py`: `b7e5382376a086f056ca85f5be36b11048a87ee647a3ac129067cc88c57af30a` (14796 bytes).
- `tldw_chatbook/Workspaces/project_index/runtime.py`: `33778a422021ee70e8364f5b67718dcbe71fcaac6acda2d22ce2c522dfdedf3c` (13352 bytes).

## Delivery and remaining gates

Separate `semantic/final-fix-evidence` contains an exact-byte manifest and lossless archive of only this wave's original receipts, pre/post source snapshots, current-origin/static/launcher records and this precommit report. Its scoped `-text` attribute prevents Git line-ending conversion. The retention verifier compares every archive member with its private original and manifest, then verifies every actual staged/HEAD evidence blob and member. Existing271 receipt group, foundation/release evidence and root final-reviews are untouched. Root Task36423 is excluded from the explicit commit whitelist.

Commit identity and actual HEAD blob verification are appended to the live private report after commit; that later append is not misrepresented as inside this precommit archive. No push, PR mutation, merge, rebase, automation, goal action or task closure is performed. Hold after scoped commit for the root's exactly one scoped rereview.

Limits: this wave qualifies model-free logic, actual SQLite/source allocation and local synthetic HTTP/native lifecycle behavior on the stated Windows runtime. No new real-model/app/20k run was authorized or performed. Earlier real model/scale receipts remain historical at their documented source, including62-74-second query latency. No universal relevance, performance, cross-platform SQLite support or final hosted CI claim follows. Saved Library policy/item-scope code and embeddings-free keyword operation retain their existing boundaries; current final-lane keyword/Library transport controls pass. Root owns any additional source qualification, final-head CI and release/task decisions.
