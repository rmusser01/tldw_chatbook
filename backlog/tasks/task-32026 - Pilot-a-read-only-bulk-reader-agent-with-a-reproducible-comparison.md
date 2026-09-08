---
id: TASK-32026
title: Pilot a read-only bulk-reader agent with a reproducible comparison
status: Done
assignee:
  - '@codex'
created_date: '2026-09-08 04:45'
updated_date: '2026-09-08 15:21'
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Evaluate whether a cheaper named reader can reduce the total cost of answering repository and long-transcript questions while preserving source evidence and answer quality. Start with an explicit preset and opt-in experiment before considering automatic routing.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Settings can populate and save an editable bulk-reader definition without changing or overwriting existing definitions.
- [x] #2 The preset permits only workspace file discovery and reads and requests quoted findings with explicit gaps; its selected model reaches the Console provider boundary.
- [x] #3 A reproducible comparison exercises the existing reader runtime on nonsensitive repository and transcript fixtures, records both worker and main requests, and leaves missing usage or price unknown.
- [x] #4 Targeted runtime, UI, and comparison checks pass and documentation distinguishes a runnable experiment from measured model-quality or cost results.
- [x] #5 A user-selected supported model pair has a recorded live comparison and manual assessment of evidence, quality, and total cost; failed or unavailable measurements remain explicit.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add an editable read-only bulk-reader preset and verify real named-agent model/tool behavior. 2. Add a scratch-corpus direct/delegated comparison with per-call accounting and manual quality review. 3. Run targeted checks and the chosen live model comparison. ADR required: no. ADR path: N/A. Reason: reuse existing named-agent/provider contracts. Plan: Docs/superpowers/plans/2026-09-07-bulk-reader-pilot.md; spec: Docs/superpowers/specs/2026-09-07-bulk-reader-pilot-design.md. Live continuation: reproduce the normalized-config pricing loss with a real load_settings profile, add a regression through the live entry point, repair only pricing lookup, retain the original live report and separately calculate corrected estimates from recorded provider usage. The first live trace also exposed synthetic fallback text for ZAI reasoning/control deltas. Reproduce through the real ZAI stream and Console normalizer, normalize provider-local empty visible content without changing native tool deltas or terminal metadata, verify targeted regressions, then repeat the comparison with the corrected runtime.
PR #2510 review continuation: rebase onto latest dev; apply shared input/path
validation, strict corpus models, bounded transactional history reads, complete
public docstrings, and PascalCase helper names. Preserve inert CLI refusal and
historical live artifacts. Add boundary regressions before behavioral fixes,
run targeted checks and derived-artifact guards, answer Qodo, and merge after
required PR checks pass. ADR required: no. ADR path: N/A. Reason: reuse existing
validation and database contracts; no schema or runtime boundary changes.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Added an editable Bulk reader preset to the canonical Settings Agents form. Selecting it creates an unsaved form; existing CRUD and duplicate validation remain authoritative. Its four file tools are read-only and its evidence instructions are advisory.

A real-gateway reproducer showed Console discarded named-agent model overrides. The adapter now uses an immutable call-local resolution for request preparation, dispatch and usage, preserving parent continuation ownership.

Added a pinned four-case synthetic repository/transcript comparison through real AgentService, Console adapter, local tools and scratch SQLite. It records each main/worker call, requires named-worker content access, flags token-limited or incomplete results, and keeps missing usage/prices unknown. The live runner supports Moonshot/ZAI through their bounded timeout/retry seam; other preset providers are unchanged. Guide: Docs/Examples/agents/bulk-reader/README.md.

Worktree evidence: 19 focused preset/UI/adapter/regression tests passed, including production CSS at 120x40 and 70x40. After evaluator review fixes, 20 focused evaluator/preset/adapter tests passed. Changed-file lint, formatting, compilation and CLI help/refusal checks passed. One pre-existing requests dependency warning remains. Integration evidence: 37 targeted tests passed in the user checkout, covering the preset, Settings UI, evaluator, worker model override, provider continuation and concurrent subagents. Changed-file Ruff checks and new-file formatting passed. The final review found a test-only hardcoded interpreter path; replacing it with sys.executable passed the affected CLI subprocess test and Ruff/format checks in the integrated checkout. Scoped re-review confirmed all findings addressed with no new Critical/Important breakage. All 16 task files were integrated while preserving staged changes and concurrent edits; the initial patch round trip verified the original bytes for 15 files, and the testing lesson was appended without altering existing entries.

ADR required: no. ADR path: N/A. Existing named-agent/provider contracts are reused; no schema, dependency or automatic routing is introduced. Added an incident-based testing lesson about verifying model selection at the provider boundary.

Live comparison completed on 2026-09-08 after the user delegated model choice and supplied a temporary credential. Used ZAI glm-5.3 (main) / glm-5.3-flash (reader); no key was saved to config or artifacts, and both runs verified the original config remained unchanged. The first 41-call run exposed two defects: normalized load_settings pricing was ignored, and stripped ZAI reasoning/control chunks generated synthetic fallback text. Added real configuration/stream regressions, repaired the live pricing lookup and provider-local empty-content representation, and retained the initial raw report plus a separate $0.089019 pricing correction.

The corrected 35-call repeat delivered core requested facts in three of four direct cases (one minor unsupported closing claim). All four delegated arms failed to run the named reader: one used a direct-reading fallback, two stalled on repeated spawn calls, and one returned malformed tool-call JSON. No worker-model call occurred in the repeat, so its $0.055944 total does not establish reader savings or quality. Estimated total spend across both attempts was $0.144963. All failed arms and spend remain included. The decision is to keep the explicit preset and defer automatic delegation pending reliable named-reader invocation. Full assistant source review, immutable raw reports and limitations: backlog/docs/bulk-reader-zai-live-2026-09-08.md.

Live-fix verification: 72 targeted checks passed; 11 native-tool loopback checks passed after allowing the local HTTP server; 21 evaluator/ZAI regression checks passed in the integrated checkout. Scoped four-file review approved without Critical/Important findings. Existing ZAI lint findings were compared with baseline and are unchanged. One unrelated provider-contract test still omits required native_tools; its test bytes and the adapter constructor signature are unchanged by the fixes. No full suite was run. ADR required: no; these repairs preserve existing provider/configuration boundaries.

Acceptance criterion 5 is satisfied by the recorded comparison and explicit assessment of failed/unavailable measurements. This pilot produced a negative adoption decision, not a measured savings claim.

PR integration against dev (5aeac5ab221958ae612dd84ff47e047b23cd3f5d) applies only the nine pilot commits, excluding unrelated documentation ancestry. The Console fix retains dev's tracing, thinking, redirect and stream-stall behavior; a real gateway preparation regression verifies that capture-on worker requests receive the selected model and matching continuation target without parent private history. Independent scoped review approved the Console integration with no findings.

Adapted the evaluator and tests to the current WorkspaceToolExecutor and in-memory ConsoleChatStore contracts. Synthetic read paths are observed at invocation instead of recovering omitted arguments from run-step metadata. Recording test models have an explicit context capacity; runtime retries are disabled, and the extra budget-summary attempt is recorded as refused without dispatch beyond eight calls. Historical live artifacts are unchanged; this integration made no billable calls.

PR verification: 101 targeted preset, evaluator, Settings, ZAI, Console normalization, redirect and project-trace checks passed. The native-tool loopback file produced 7 passes and 4 failures; the same four failures reproduced on a separate clean checkout of the exact dev base: test_console_runs_two_native_calls_with_private_continuation[moonshot/zai] and test_hosted_tool_error_continues_structurally[moonshot/zai]. These are baseline failures, not introduced regressions. New-file Ruff checks and formatting for all ten touched Python files pass; existing Console/ZAI lint counts are unchanged. No full suite was run. ADR required: no; current runtime interfaces are reused. Added the isolated-helper import-provenance lesson discovered during integration.

PR #2510 Qodo remediation: rebased the ten feature commits without content
conflicts onto dev 1c022378cb66edafc159cc7dcf802b884804aac4. Addressed all seven
rule findings in the evaluator: reused shared model/provider and filesystem
validators; added strict Pydantic corpus/case models; bounded run-history reads
inside the existing held transaction; made overflow explicitly incomplete;
documented public callable contracts; and applied PascalCase helper names.
The corpus models live in Agents/bulk_reader_corpus.py and load only after
consent; CLI help and missing-consent refusal remain application-import free.

Eight boundary regressions first failed against the previous implementation.
The repaired evaluator file passes 28 tests, and the wider targeted run passes
109 checks. All derived-artifact preflight guards, new-file Ruff checks, and
formatting for eleven touched Python files pass. The six historical live JSON
artifacts are byte-identical to the original PR head; no additional model calls
were made. The pre-existing native-tool failures remain documented; the latest
dev changes only added Backlog tasks. ADR required: no; existing validation and
database contracts are reused. No full suite was run. Independent scoped review
found no blockers and separately passed all 28 evaluator tests.
<!-- SECTION:NOTES:END -->
