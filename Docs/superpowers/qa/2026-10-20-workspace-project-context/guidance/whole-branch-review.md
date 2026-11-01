# Final whole-branch review — workspace project guidance

Reviewed base `bbe609a676` through head `7dae24220b` in `C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui`.

## Strengths

- Final capability filtering precedes guidance construction (`tldw_chatbook/Chat/console_agent_bridge.py:5120-5143`). The note uses the existing admitted-alias and scratch helpers, and the changed code adds no grants, global-workspace fallback, scanning, storage, indexing, or dependencies. The actual Goal narrowing and custom operating-prompt tests exercise the production request boundary.
- The strict Library formatter is preserved while the Console formatter shares its source rendering (`tldw_chatbook/Chat/answer_citations.py:168-235`). Runtime optional-marker policy is immutable, Boolean-validated, strict by default, and localized to canonical Console capture. Existing actual-marker validation and claim-preserving repaired selection remain intact.
- Native Windows identity correction is narrow (`tldw_chatbook/Tools/workspace_root_pin.py:258-331`): complete unsigned volume/file IDs replace the truncated legacy handle identity; exact initial and post-chdir comparisons, reparse refusal, retained handle ownership, and fail-closed metadata handling remain. Native tests include actual handle closure, real pin/read/restoration and junction controls. The generated bundle matches the visible helper change; the two source-snapshot additions are documented mechanical drift, not a new authority path.
- Integrated guards use real SQLite policy rows, registered read-only roots, controller/agent dispatch, and production retry acceptance. They hold preparation while both the viewed session and global active workspace change and inspect independent expected aliases/scopes in actual native/fenced payloads (`Tests/Chat/test_console_workspace_guidance.py:185-285`). Four saved Library combinations compare complete durable rows (`Tests/Chat/test_console_library_runtime_policy.py:285-323`).
- Actual-app qualification goes beyond scripted transport: the final receipt records six real native rounds, five successful tools plus truthful Git refusal, matching controller/store ownership, completed runtime/ledger, successful UI sync, and unchanged durable policy. Its real README/manifest results substantiate purpose, marker, version and Python constraint. Interrupted/confounded attempts are preserved rather than promoted to passes.

## Issues

### Critical (must fix)

None found.

### Important (should fix)

1. **Canonical staged evidence loses the instruction to cite Library-supported claims.**
   - Location: `tldw_chatbook/Event_Handlers/Chat_Events/chat_rag_events.py:1804`, in conjunction with `tldw_chatbook/Chat/console_chat_controller.py:10742-10829` and `:22627`.
   - Canonical handoff/manual staging constructs the new optional-marker contract, then passes its exact source-only context directly to the existing `Evidence:` prefix. It bypasses `format_console_evidence_for_cited_answer`, so this path receives none of the new conditional Library-label/file-path instructions. The default agent operating prompt also supplies none (`Internal_Prompts/agents_prompts.py:39-45`). Previously the mandatory missing-marker repair supplied an eventual citation requirement; removing that repair is correct for file-only answers, but this branch now lacks the intended replacement guidance.
   - The recorded explicit mixed journey demonstrates the practical gap: `Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/real-model-receipt.json:1382` reports the Bluefin Library claims without `[S1]`; its first dispatch at `:1396` contains the source body and question only. This is evidence consistent with the code gap, not a claim that a prompt can guarantee every model response or that the omitted separately supplied live system context was captured.
   - Effect: users can receive supported Library claims without their owning citation labels on the canonical staging path, contrary to the binding spec's Grounding and citations section, while automatic retrieval gets the correct instructions. The current mounted handoff regression verifies context, identity and absence of forced repair but does not verify this conditional guidance.
   - Fix: add the same conditional mixed-source wording to canonical provider payloads outside the exact captured evidence block. Preserve the context bytes, snapshots, ordinals and contract identity; keep `require_markers=False`, zero forced repair for file-only answers, and strict validation of actual invalid markers. Extend real mounted/canonical provider-payload coverage for this path, including the distinction between Library labels and file paths. Do not add a classifier or restore blanket mandatory markers.

### Minor (nice to have)

1. **The architecture description reverses the native identity mismatch.**
   - Location: `Docs/Architecture/console-file-authority.md:57-59`.
   - It says capture and revalidation both derive from native handles and refers to a truncated `os.stat` identity. Capture still uses `os.lstat` through `Utils/filesystem_identity.py:49-70`; the old handle query's DWORD volume serial was truncated. The new handle identity is compared to the full Python-captured identity at both pin checks (`Tools/workspace_root_pin.py:198`, `:205`).
   - Effect/fix: future maintainers could misunderstand the security boundary or reproduce the wrong diagnosis. Describe full handle metadata matching the admitted Python identity at initial pin and post-chdir verification.

2. **Older compact answer receipts still contain encoding corruption.**
   - Locations: `Docs/superpowers/qa/2026-10-20-workspace-project-context/guidance/natural-project-initial.json:16`, `all-readers-blocked-receipt.json:39`, and `real-model-receipt.json:39`, `:367`, `:1382`.
   - These `final_answer` strings contain UTF-8-as-cp1252 sequences such as `â€”` and `â‰¥`. A focused comparison against the retained raw scenarios' `visible_copy` establishes that all five nonempty older answers differ, and reversing precisely that encoding error yields exact equality. The final actual-app answer is unaffected. Some older all-reader tool-result text also retains the same visible corruption.
   - Effect/fix: semantic facts remain legible, so this is not a failed product qualification, but these fields are not exact retained final text as described. Rebuild affected text fields from retained raw Unicode observations and validate exact equality. Preserve chronology, genuine model overstatements, missing-observation limits and run ownership; do not edit the response prose or rerun the provider.

## Existing warnings and deferred-minor triage

- **Native listing gap from G1: resolved.** The retained native root-pin log reports 15 passing cases and the real-worker contract log reports five passes. G3's affected selection also includes the successful folder-note paths. The original unchanged-root refusal is no longer an open qualification gap.
- **Exact approved guidance paragraph: resolved.** The production literal read in the supplied diff matches the binding spec paragraph; follow-up/multiple-folder text adds the approved as-needed behavior without authority changes.
- **Initial remaining Windows/platform/profile failures: retained baseline limitations, not claimed passes.** The G1 report names symlink replacement/validation controls, invalid Windows newline-name assumptions, SSH fixture assumptions, and `test_workspace_note_reaches_system_prompt_on_fence_protocol_endpoint`. No changed code relaxes their guards. G3's new native/fenced owning-session dispatch coverage does not erase those older failures or establish other-platform compatibility.
- **G2 cancellation-observer failures: inherited and nonblocking for this diff.** Both `test_citation_repair_stop_while_checking_cancels_before_dispatch[direct/agent]` failures reproduce on pre-G2 source with `citation checking was not observable`; the retained baseline log corroborates them. They remain real test debt and must not be called a clean broad selection. No cancellation logic or observer assertion was weakened here.
- **UI profile-selection fixture: corrected and requalified.** The official private-profile harness and correct capture seam preserve original assertions; retained mounted checks passed. This is not an admission bypass.
- **Lint/warnings: retain explicit non-pristine status.** Existing E721/E402 bridge/root-pin findings, the 74 pre-G2 source findings, pytest-asyncio loop-scope warning, Pydantic `Field(exclude_if=...)` deprecation and inherited unraisable-coroutine warning remain separate cleanup. No introduced source diagnostic was identified. Do not claim all unfiltered lint or all selected tests are green.
- **55 G3 nodes: supported across runs, not one clean sweep.** The retained affected log reports 52 passed/3 new setup assertion failures; the corrected four-policy selection passes. The named-column fix retains the full durable-row comparison. This satisfies final evidence for those nodes without manufacturing regression RED or a pristine-run claim.
- **Prior G3 review findings: chronology/refusal/timing are resolved.** Correct mixed first dispatch carries the explicit mixed question/Bluefin; six retained mixed snapshots and nine retained denied snapshots preserve the declared missing intermediate denied dispatch. Two-folder refusal is accurately described as unloaded-tool refusal. Completed-event time is 214.399 seconds and final-ledger time 216.513 seconds. The remaining older text-encoding defect is Minor 2 above.
- **Model installability inference: not supported and not product evidence.** Its preserved answer infers non-installability from missing build-system metadata. Documentation already calls this unverified; no branch claim or response rewrite should convert it into an observed fact. This is a model-quality limit, distinct from Important 1's absent canonical instruction.
- **UI observation limits: correctly retained.** Host-mounted snapshots show stale/empty UI and cannot prove normal application refresh. Interrupted probes lack final integrity assertions; the 220-second actual-app attempt is incomplete and stopped normally. The later actual-app completion qualifies normal Send and observed refreshed state, not every automatic refresh timing or all models.
- **Native FileIdInfo restriction and remote bundle: accepted fail-closed limit.** Filesystems lacking required metadata refuse; a controlled 128-bit payload is not real ReFS proof. Other Windows filesystems/builds, POSIX and SSH runtime remain unqualified. Mechanical embedded parity does not establish remote execution support.

## Execution-ruling triage

All thirteen published rulings are accounted for. None authorizes weakening the feature's authority boundary.

| Rulings | Disposition |
| --- | --- |
| 1 managed worktree; 2 targeted tests; 3 G-heading brief extraction | Appropriate isolation/scope choices. No full-suite or unrelated-checkout claim. |
| 4 Backlog allocation | Accept for local execution; re-sweep IDs before a later push as already recorded. |
| 5 shared read-only dependency runtime | Accept with stated reproducibility/platform limitation; provenance evidence names this worktree. |
| 6 initial native/platform deferral; 8 native prerequisite; 9 narrowed exception | Coherent supersession: the proven identity blocker was fixed; unrelated inherited failures stay visible. |
| 7 configured provider model substitution | Accept for disposable qualification only. Receipts name deepseek-flash; saved legacy model and other models are unqualified. |
| 10 generated bundle regeneration | Appropriate required consumer update; inert source snapshots and runtime limits are disclosed. |
| 11 task-private ACL restoration | Accept as evidenced setup repair, not a product or user-profile permission change. No new ACL operation was needed in review. |
| 12 actual TldwCli lifecycle; 13 bounded longer observation | Appropriate correction of harness provenance and prematurely ended observation. Earlier adverse attempts are retained and no application budget or permission floor was changed. |

## Review method and concrete outside-diff checks

Read the supplied six-commit diff package in bounded source/test/document passes; retried cut-off outputs without deriving a new Git diff. Parsed all supplied JSON/JSONL and SVG artifacts with stdlib parsers and extracted semantic source/tool/answer/status values, row inventories and viewport text rather than reading repetitive drawing geometry as code. Read all four task reports, all task reviews and G3 re-review, the binding plan/spec sections, ADR-240, ADR-101, tracked QA README/execution notes and ledger. Read retained focused result/baseline log summaries; no tests, provider journeys, Git mutation, or new agents were run.

Concrete named-risk checks outside changed hunks:

1. Final capability note could reach schema construction in stale form: inspected alias helper, root-line admission filtering and final config/schema consumption; no new widening found.
2. Optional repair might change actual-marker validation, claim text or contract identity: inspected the existing scanner, repaired selection, capture normalization and VALID completion branch. Existing eligible-marker semantics remain; code/escaped/non-marker literals are not promoted into citations.
3. Canonical handoff might miss formatter instructions: inspected controller capture branches/prefix and default operating prompt. This establishes Important 1.
4. Windows doc might misstate capture or pin ownership: inspected `capture_directory_chain` and `_pin_windows`; this establishes Minor 1 while confirming exact comparisons and cleanup remain.
5. Compact final answers might no longer be exact retained text: compared only named raw scenario `visible_copy` values with compact `final_answer` fields, and verified the single encoding transformation. This establishes Minor 2.

Review remained read-only on checkout/index/HEAD. This ignored report is the sole authored file.

## Recommendations

Apply one scoped correction for the canonical instruction path and its real payload regression, plus the two small documentation/receipt fixes. Retain all baseline caveats and existing optional-marker tests; no broad suite or new live provider journey is necessary merely to correct documentation or encoding. Re-review the correction diff before closing TASK-35932; its current In Progress status is appropriate.

## Declined to judge

- Optional keyword/semantic indexing, cache storage/migration/refresh/revocation and scale: separate approved deliverables, explicitly excluded here.
- Full-repository regression freedom and clean installation: no user opt-in for a full sweep and no clean-environment qualification; retained targeted evidence is the review basis.
- Untested Windows filesystems/builds, real high-128-bit inode filesystem behavior, POSIX and live SSH execution: native-volume and controlled-layout evidence cannot establish these environments.
- Arbitrary descendant-race defense/general OS sandboxing: outside ADR-101's root-pinning boundary and unchanged by the representation fix.
- Complete live system prompts or HTTP/wire bytes: gateway snapshots explicitly omit separately supplied system context.
- Exact total/late completion timing for the interrupted denied-reader journey: one dispatch snapshot is missing; no evidence can reconstruct it faithfully.
- Product startup/refresh/provider-failure causation from Host or interrupted observations: their lifecycle/cancellation confounds prevent that conclusion.
- Global original-profile content immutability beyond the recorded comparisons: completed launchers assert config SHA256 and data mtime/size, not full byte hashes of all user-data files; interrupted probes lack final comparisons.
- Installability of the synthetic skeleton and reliability across other models: neither was verified by the requested file-summary journey; the retained claim is a model overstatement.

## Assessment

**Ready to merge? With fixes.**

The branch's authority, policy-preservation, native identity and automatic-grounding changes are narrowly implemented and well supported by targeted and real-provider evidence. Add the missing canonical mixed-citation instruction and correct the two small documentation/receipt defects before final approval; no Critical security or data-loss regression was identified.
