# Agent orchestration final follow-ups implementation plan

> **For agentic workers:** Use `executing-plans` to implement these existing Backlog tasks with targeted verification and independent review. Steps use checkboxes for tracking.

**Goal:** Close TASK-33648, TASK-34353, TASK-33560 and TASK-33640 without broadening execution/storage authority or overstating evidence.

**Architecture:** Reuse the current hook execution witness and permission owner for the one bounded SessionEnd notification. Reuse process-local admission evidence for the remaining measured storage derivations, with complete per-call stamps and exact native resource custody. Close delivered CI/provider criteria from authenticated evidence rather than repeating completed work.

**Tech stack:** Python >=3.12, Textual 8.x, existing native filesystem/admission primitives and pytest; no new dependency.

**Spec:** The four existing task files and their acceptance criteria under `backlog/tasks/` are authoritative. ADR-163/197 govern hook teardown/consent; ADR-126 governs storage reuse; ADR-103 governs CI aggregation.

## Global constraints

- Work only in the existing agent-recovery-error worktree on `codex/agent-orchestration-final-followups`, initially based on merged dev `6feb84c1d2203bc3c6d0eecd2823f4e2409b4b4e`.
- No full suite, original PR2918 budget replay, changed caps/work/counts, provider key disclosure, unrelated implementation or shared/foreign cleanup.
- Preserve failed/raw/XML evidence and inherited warnings/debt; distinct selections are never summed into a certificate.
- Real tests use private outer HOME/USERPROFILE/TLDW/XDG, repository cwd/PYTHONPATH and default pytest temporary depth.
- Backlog statuses, plans and acceptance checks use the CLI. Only complete evidenced criteria; retain all task history.
- Source changes require a prospective design/ADR assessment and meaningful RED before implementation; independent review before completion.

## Task 1: TASK-33648 — granted SessionEnd at host disposal

**ADR required:** yes, narrow clarification of existing teardown/consent policy.
**ADR paths:** `backlog/decisions/163-expanded-console-hook-runtime.md`, `backlog/decisions/197-console-hook-configuration-review.md`.
**Reason:** retain one existing host-issued, effect-free notification while ordinary admission closes; no new permission owner or grant.

**Files:** `Agents/hooks_v2/{engine,lifecycle}.py`, `Agents/hook_permissions.py`, `Chat/console_runtime.py` under `tldw_chatbook/`; `Tests/Chat/test_hooks_v2_lifecycle.py`; the two governing ADRs.

- [x] Put the task In Progress and record the prospective Backlog plan.
- [x] Reproduce with real saved/granted config and actual controller Send: exact close passes; both host-disposal paths refuse authority and fail the real marker assertion.
- [x] Record the bounded policy prospectively in ADR-163/197.
- [x] Review the design: retain the lifecycle's exact SessionEnd event and existing engine-issued execution state; carry that witness through current worker context. Public teardown calls, copied events and even replay of the same event must not acquire the original state.
- [x] Implement the smallest shared fix. Only the effect-free standalone command may use the private teardown launch guard; canonical config, profile/section, fingerprint and exact grant epoch are re-read under existing launch locks. Ordinary targets remain closed; managed-plugin/MCP/model authority remains unchanged.
- [x] Verify once-only process execution, late revoke/reapproval/definition/missing-store refusal, forged deliveries, cancelled disposal and actual process/ticket/reap settlement under the original three-second notification and cleanup allowance.
- [x] Obtain independent scoped source review, append evidence/limits, then check all criteria through CLI.

## Task 2: TASK-34353 — hosted UI capacity evidence

**ADR required:** no. Existing ADR-103 lane/aggregation contracts remain unchanged.
**Files:** existing TASK-34353 Markdown only.

- [x] Authenticate the existing final PR2918 four-shard raw logs, metadata, checkout tree and actual merge.
- [x] Verify all 152 census files occur once in four ordered 38-file slices, including PR2992's five byte-exact Phase 6 Settings additions.
- [x] Record actual 626/249/911/855-second successful jobs under the unchanged 1200-second cap, retaining historical two-/three-shard estimates and limitations.
- [x] Independent evidence review, CLI AC3 checked and task Done. No completed tests/workflows/budgets rerun.

## Task 3: TASK-33560 — remaining measured admission derivations

**ADR required:** yes, extend the existing allowed-evidence reuse policy.
**ADR path:** `backlog/decisions/126-complete-local-backup-and-recovery.md`.
**Reason:** its current amendment authorizes acquire_storage reuse, whereas additional positive witness, pause-group, raw-pin and companion metadata derivations need explicit complete-dependency qualification.

**Files:** minimal shared owners in `tldw_chatbook/Backup_Recovery/{storage_admission,admission,raw_participants,config_participants}.py` and `tldw_chatbook/Backup_Recovery/generation_witnesses.py`; scoped Backup_Recovery mutation/completeness tests and a task-owned boot/idle probe.

- [x] Put the task In Progress with a prospective plan; preserve the already owner-approved 1.0-second monitor cadence.
- [x] Measure actual native work before source changes: real warm MCP reads and native pause calls, separately from whole-app idle measurement. Existing original audit probe scripts are unavailable; do not substitute old PR budgets.
- [x] Finish the whole positive recovery-witness trace and record the minimum concrete cache design in the governing ADR before source edits.
- [x] Review the design. Prefer existing hold-owned `_Evidence`, complete posture/content stamps, PID/epoch/names and one-second settle. Every gate/lock, pause, active-operation/source-selection and physical-resource check remains per-call. A mismatch runs the exact original derivation and reason codes.
- [x] Add meaningful RED controls for skipped warm handshakes and pause/raw/companion reuse, with full-derivation mutation oracles and dependency-completeness traces. Eight initial cases reproduce seven warm/completeness failures and retain one defensive-copy pass; no errors or skips. History migration/sanitization retains write admission and collision-safe fresh temporary members.
- [x] Implement only proven shared positive derivation reuse. Do not cache a pause verdict, replace a live FD owner, follow unchecked path components or infer authority from absence/pending state.
- [x] Compare paired actual boot/idle workloads under unchanged production cadence and work; require a further >=50% reduction in open calls. Measure actual backup pause latency against the existing owner-approved cadence.
- [x] Run targeted oracle/completeness/cancellation checks and independent source review. Preserve exact platform and resource limits; close all four criteria only when evidenced.

## Task 4: TASK-33640 — captured allowance provenance

**ADR required:** no; existing provider/parser and credential boundaries are unchanged.
**Files:** existing TASK-33640 Markdown and `Tests/fixtures/cloud_live/README.md` only.

- [x] Inventory every current engine preset without printing credentials: 32 total, Together/Fireworks ready, 30 clean missing-key skips.
- [x] Authenticate both existing successful live captures, exact fixture citations, zero unknown keys and clear credential-fragment scans. No new request or allowance was needed.
- [x] Fresh targeted replay through actual engine wrappers and model discovery: 19 passed, zero failed/errors/skips; preserve raw/XML and 27 late foreign-directory rm_rf warnings.
- [x] Independent evidence review; correct its sole README typo; CLI AC5 checked and task Done. Other presets remain provisional until keys arrive.

## Delivery

- [x] Complete both source fixes and review the combined branch for composition regressions.
- [x] Validate only modified functionality, task/ADR hygiene, whitespace and scoped static analysis; distinguish inherited debt from new findings.
- [x] Prepare a reviewable PR with current evidence and practical limits; publish only the clean reviewed commit and verify its actual head/URL. Follow protected head-matched integration only when authorized and all fresh gates qualify; do not claim a merge while only queued/armed.

Preflight record: TASK-33648 design Ready with original execution-state authenticity and ordinary closed-owner refusal retained. TASK-33560 design Ready after adding all consulted historical path chains and fresh fallback for unqualifiable historical/alias dependencies; omitted-history dependency qualification is mandatory. Microbenchmark counts remain separate from the forthcoming actual-app idle measurement.

Measurement checkpoint: the fresh native bound-profile actual app baseline retained its real profile hold (count 8) at both idle edges, drained the boot fleet and reached ChatScreen. At unchanged cadence/work it measured 884 native opens over 10.001132 seconds (88.390/s), 40 credential polls and 10 native probes. The earlier unbound app result is setup evidence only. Exact probe SHA da5ba790b5d59549bab2345f2418a98dc47a24fa6c77ab58e251039c4d94539f is frozen for the before/after comparison; no original PR budget was replayed. Performance production implementation was authorized only after this baseline and meaningful RED.

Independent hook source review found a blocking supported mixed-native-plugin projection path: trusted projection copies a standalone event before its authority check. Before correction, qualify a real installed/activated native plugin plus saved/granted standalone disposal RED. Retain original host event/execution authenticity through a minimal private per-delivery projected-event witness, with exact callback identity, active/non-cancelled membership, fixed deadline, fresh binding/finally reset and brief memory locking. Public same/copied/ambient-context deliveries must still refuse. The earlier frozen package is Changes required, not Ready; preserve it and all evidence. A new frozen review is required after correction. The mounted executor teardown timeout separately reproduced on complete exact6feb base (one body pass plus teardown error); both selections remain non-green.

Final implementation checkpoint: all four task criteria are checked through CLI and Done. Hook and performance independent source reviews, the combined boundary/doc review and portable evidence authenticate the unchanged approved source. Performance final actual idle rate falls67.64561%; the original1Hz cadence produces actual localpause1.069559s. The related mounted timeout and permission ordering/RMW refusals remain separate non-green limitations. Whitespace and Backlog ID/readability guards pass; no original completed PR budget or broad suite is renewed. The PR body is prepared; publication and its actual URL are recorded after this commit, without fabricating a merged/CI-green result. Incident-backed hook lesson: backlog/docs/lessons-hook-teardown.md.


## Authorized PR3037 integration and protected merge

**ADR required:** no new decision.
**ADR paths:** existing ADR126 and ADR163/197; incoming ADR097 unchanged.
**Reason:** preserve both independently delivered implementations and all source/evidence history; no new runtime, storage or permission boundary.

User authorized updating against current dev and merging PR3037. Original reviewed/published head is 8d88ababdf83701174fac753283cbd44153262ad. Exact incoming dev is 518277133cc1281e446387b06d59dc54fea9b727 (eleven commits after6feb). Live protection requires strict current-base status, enforced admins and resolved conversations; linear history is not required. Auto-merge is off.

- [x] Commit this prospective checkpoint after reopening only TASK33648/33560 AC3 verification through CLI.
- [x] Merge exact incoming dev into the feature branch, preserving reviewed commits and exact old source/evidence rather than rewriting them. Review Console failed-attempt/retry and established-store read composition.
- [x] Run only the bounded integration controls, preserve raw/XML and all inherited limitations, authenticate unchanged owned source/evidence, and obtain independent immutable integration reviews. The prior67.64561% paired measurement remains a historical checkpoint, not a new merged-app measurement.
- [x] Close the reopened criteria through CLI with evidence and prepare one clean preserving publication. Actual remote/GitHub identity is recorded after push.
Merge condition: await current-head PR Fast Lane, all four UI shards, latency and Derived artifacts plus clear current review conversations; read live protection/actual refs and perform normal protected head-matched merge.
Completion condition: verify actual GitHub MERGED state, merge parents/tree and current concurrency before reporting completion. Preserve pinned worktree and all prior raw evidence. No automation operation.

Local integration checkpoint: preserving tree authenticated; separate58PASS/181.650XML and17PASS/110.583XML, bothF/E/S0; two independent immutable integration reviews Ready/no actionable findings. All four tasks Done viaCLI, history retained. Portable evidence is in Docs/superpowers/qa/2026-10-06-pr3037-integration. Current hosted CI and actual merge are intentionally pending until publication; no app/test change after source approval.
