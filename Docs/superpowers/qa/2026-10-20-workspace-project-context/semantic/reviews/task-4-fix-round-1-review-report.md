Reviewed `6bda3c73..4940a66f` read-only. **Fix round needs changes:** the source defects are addressed, but AC5 remains open and committed evidence has a new integrity defect.

### Original findings

1. **ADDRESSED — obsolete snapshot after cleanup retry.**  
   [project_index_binding_card.py:517](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:517) retains the actual service/key, checks retirement, and rereads durable semantic state. Failed or closed-owner reads display unavailable state and suppress semantic actions.  
   [test_project_semantic_consent.py:415](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/UI/test_project_semantic_consent.py:415) calls the real Disable, verifies durable Off, then raises retirement uncertainty. `task4-card-review-green-02` records both cleanup and stale-Disable cases passing.

2. **ADDRESSED — indefinite “checking” after semantic read failure.**  
   [project_index_binding_card.py:557](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/Widgets/Settings_Widgets/project_index_binding_card.py:557) distinguishes `LifetimeBusy` from unexpected failure. Publication preserves the previous validated snapshot, shows retry guidance, and disables semantic actions; `_poll` retries failures at line 223.  
   [test_project_semantic_consent.py:453](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/UI/test_project_semantic_consent.py:453) covers failure, preserved keyword/draft state, recovery, and Busy. The final paired receipt records its pass.

3. **ADDRESSED — ineffective Library preservation matrix.**  
   [test_project_semantic_app_quality.py:299](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/Workspaces/test_project_semantic_app_quality.py:299) now captures each combination before an actual transition and compares afterward without rewriting it. All sixteen combinations are exercised, alternating eight stages and eight disables, followed by provider changes. `task4-card-library-final-01` records the Library case passing in 142.95 seconds. This does not certify the pending final Console or model-lifecycle checks.

4. **NOT ADDRESSED — AC5 release qualification.**  
   [semantic/README.md:13](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/README.md:13) still records an interrupted nested build and no wide run. Completed retained-shape builds, Ready/query latency, whole-run storage/RSS peaks, and scale cancellation remain absent. The report correctly keeps the task In Progress; diagnostics and the small held-producer control do not close AC5.

5. **ADDRESSED within the stated settled-startup scope — retirement claims corrected.**  
   The original dark/light executor warnings remain retained as adverse evidence. `task4-settings-owner-02-stacks.jsonl:3–5` identifies one physical future still pending at executor shutdown, originating from startup FTS backfill.  
   [test_project_semantic_consent.py:593](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/UI/test_project_semantic_consent.py:593) waits for that real boot worker’s SUCCESS before rechecking the mounted card. The settled-03 child records a pass without an executor warning. It still has inherited Pydantic/audioop warnings. Close during unfinished startup and final light Settings remain unqualified.

The **stale Disable after Clear/re-enable Minor is addressed** by [test_project_semantic_consent.py:169](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/UI/test_project_semantic_consent.py:169), including re-enable and equality of the newer durable semantic state.

### New Breakage in Fix Diff

**[P2 / Important] Committed receipt bytes do not satisfy their manifest.**  
[receipt-manifest.json:16](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Docs/superpowers/qa/2026-10-20-workspace-project-context/semantic/receipt-manifest.json:16) declares `task4-card-library-final-01-parent.log` as **1,873 bytes / BE79984E…**, but its added payload in the guarded diff is **1,856 bytes / BD5C6186…**, matching the controller’s committed-blob check.

I reconstructed all 141 newly added text receipts from the supplied diff: **123 fail their manifest bytes/hash** after line-ending normalization. All 271 working-copy receipts match, so working-copy verification conceals this delivery defect. A fresh checkout cannot verify the claimed original evidence.

Preserve original bytes in committed artifacts—using appropriate `-text` attributes or lossless archives—and verify committed payloads against the manifest. Merely changing hashes to normalized text would abandon the original-byte claim.

**No new runtime source regression identified.** The batching change confines HF provider pages to 128 rows/1 MiB, validates complete vector count and dimensions before append, and retains separately admitted 32-row/1 MiB native slices and checkpoint advancement after commit.

### Out-of-Scope Observations

- AC4 final-source lifecycle, all four final Console cases, final light Settings, twelve-target lane execution, and hosted AC6 remain pending.
- Startup-close uncertainty belongs to the identified existing boot-worker path; this review does not certify it.
- Existing warning ownership remains visible: pytest configuration at `pyproject.toml:647–665`, Pydantic usage at `audio_cpp_guided_config.py:353`, and third-party pydub/audioop.
- The command table slightly overstates `local128-guards-01`: it has one private child log/XML pair; the second selector executes directly and is covered by the parent result.

### Checks Actually Performed

- Verified supplied package SHA-256.
- Read bounded production, test, documentation, and named evidence passes.
- Verified all 271 working-copy receipt hashes/lengths and twelve working-copy source snapshot entries.
- Verified ZIP member bytes and hashes.
- Inspected invocation arrays/selectors, parent results, child logs/XML, owner stacks, and actual local128 retirement receipt.
- Confirmed the direct CI receipt lacks outer CLI argv, as disclosed.
- No git commands, mutations, test reruns, models, installs, or downloads.

**Verdict:** I1, I2, I3, I5 and the stale-Disable Minor are addressed within the qualifications above. **Open: I4/AC5 and the new P2 committed-evidence integrity defect**, plus the explicitly pending integration/release checks.