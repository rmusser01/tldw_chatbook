### Spec Compliance

- ❌ Issues found: consent controls and AC7 are implemented, but cleanup/status handling has correctness gaps and AC5 remains incomplete.
- ⚠️ AC6 hosted runtime/budget qualification remains unverified. The package contains a local eleven-target, 32-pass receipt, not execution of the required hosted job: Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/README.md:12.
- ⚠️ AC4 final-source qualification remains unverified. The real-app receipt identifies Embeddings_Lib.py hash E611… (task4-real-model-failure-01-pre.json:20); the corrected CPU-FP32 probe identifies 5E2C… (task4-dtype-probe-green-01-pre.json:5). The latter proves loading/embedding precision, not final-source app discovery and lifecycle behavior.
- ⚠️ Existing authority, pagination/publication fences, quota enforcement and live disclosure validation largely belong to unchanged earlier tasks; this scoped diff cannot independently recertify those contracts.

### Strengths

- The disclosure dialog derives settings and human-readable destination from one inspected catalog result and rejects mismatches before approval. Captured epoch/revision still reach the service CAS: tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:388.
- Keyword and semantic selectors are separated, preserving independent input drafts and controls: tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:312.
- The precision correction selects device before model loading and chooses FP16 only for CUDA. Five focused device cases and an actual cached CPU-FP32 receipt support AC7: tldw_chatbook/Embeddings/Embeddings_Lib.py:383; Tests/RAG/test_hf_embedding_device_precision.py:10.
- Required CI aggregation fails closed, prints actual SQLite identity/options, pins Python3.12.10 and keeps optional model qualification separate: .github/workflows/derived-artifacts.yml:818.
- Adverse scale evidence is retained with explicit limits rather than presented as successful qualification: Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/README.md:13.

### Issues

#### Critical (Must Fix)
- None identified in this scoped review.

#### Important (Should Fix)

1. **Cleanup retry republishes an obsolete semantic snapshot.**
tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:511 reuses self.semantic_status and changes only close_pending. A real Disable commits revocation before waiting for native retirement (service.py:994–1007), so a failed Disable can leave the card holding an old Ready/enabled snapshot. Successful cleanup retry then removes “cleanup pending” while continuing to advertise that old state and stale actions. Reload the durable state after retirement when admission permits; otherwise display an explicitly unavailable/stale state. Cover a Disable that actually revokes before raising, rather than the current test double that only raises.

2. **Semantic status failures are silently converted into indefinite “checking”.**
tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:529 catches every exception and substitutes None. A successful keyword read then overwrites any previous semantic snapshot without error feedback. With keyword Ready and semantic None, _poll does not retry (:219), leaving semantic controls unavailable until a manual unrelated status refresh. Preserve the last validated state with visible failure/retry guidance, and distinguish expected availability failures from unexpected errors.

3. **The Library category matrix does not test preservation across changes.**
Tests/Workspaces/test_project_semantic_app_quality.py:286 sets each category combination after semantic staging, saves it and immediately compares it with another save. After disable, :306 resets every combination again before asserting. A regression that erased those choices during stage/disable/chat-provider change would pass. Capture each combination before the transition and compare afterward without rewriting it, or remove this claimed matrix evidence. The separate Console policy test provides useful coverage but does not make these assertions meaningful.

4. **AC5 release qualification is missing.**
Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/README.md:13 records an operator-stopped nested attempt after4,864 chunks. Neither shape has completed semantic qualification; Ready/query latency, whole-run storage/memory peaks and physical cancellation remain unqualified. The read-only diagnostic at:15 cannot fill those gaps. Keep the task/release gate open until actual evidence or an explicit requirement adjustment exists.

5. **Nominally successful mounted receipts still contain executor-retirement uncertainty.**
Both task4-settings-dark-08-test_actual_settings_semantic_0-child.log:19 and task4-settings-light-01-test_actual_settings_semantic_0-child.log:19 report that the executor did not finish joining within three seconds. Those child warnings are absent from the parent’s summarized warning count. They do not support the report’s claim that dark-08 “retired normally.” Retain the warning as unresolved, identify the owner and obtain positive retirement evidence; passing assertions alone do not establish clean teardown.

#### Minor (Nice to Have)

- The requested stale **Disable after Clear/re-enable** case is not represented by the new stale-action test: Tests/UI/test_project_semantic_consent.py:145 exercises a captured **stage** action after Clear without re-enable. Add the exact sequence when repairing the cleanup/status tests.
- Existing pytest-asyncio and Pydantic deprecations remain visible in task4-core-ci-eleven-01-parent.log:3 and:10; mounted children additionally report audioop deprecation. Preserve ownership of these warnings rather than describing evidence as pristine.

### Assessment

**Task quality:** Needs fixes.
**Reasoning:** The consent identity/CAS approach and device correction are sound, but cleanup recovery can display revoked semantics as Ready, and semantic read failures disappear into a stalled UI state. Scale and hosted qualification remain explicit release gates.
**Checks performed:** Read-only review of supplied diff in bounded production/test/doc and receipt passes; no git commands, test reruns, mutations or downloads. Hunk-cut exceptions inspected remaining HuggingFace initialization and card worker tail. Focused outside-diff checks examined destination sanitization/settings identity, semantic revocation/close signaling, UI census behavior, executor-warning fixture contract.
