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
- [ ] Review the design: retain the lifecycle's exact SessionEnd event and existing engine-issued execution state; carry that witness through current worker context. Public teardown calls, copied events and even replay of the same event must not acquire the original state.
- [ ] Implement the smallest shared fix. Only the effect-free standalone command may use the private teardown launch guard; canonical config, profile/section, fingerprint and exact grant epoch are re-read under existing launch locks. Ordinary targets remain closed; managed-plugin/MCP/model authority remains unchanged.
- [ ] Verify once-only process execution, late revoke/reapproval/definition/missing-store refusal, forged deliveries, cancelled disposal and actual process/ticket/reap settlement under the original three-second notification and cleanup allowance.
- [ ] Obtain independent scoped source review, append evidence/limits, then check all criteria through CLI.

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

**Files:** minimal shared owners in `tldw_chatbook/Backup_Recovery/{storage_admission,admission,raw_participants,config_participants}.py` and MCP recovery witness owner if required; scoped Backup_Recovery mutation/completeness tests and a task-owned boot/idle probe.

- [x] Put the task In Progress with a prospective plan; preserve the already owner-approved 1.0-second monitor cadence.
- [x] Measure actual native work before source changes: real warm MCP reads and native pause calls, separately from whole-app idle measurement. Existing original audit probe scripts are unavailable; do not substitute old PR budgets.
- [ ] Finish the whole positive recovery-witness trace and record the minimum concrete cache design in the governing ADR before source edits.
- [ ] Review the design. Prefer existing hold-owned `_Evidence`, complete posture/content stamps, PID/epoch/names and one-second settle. Every gate/lock, pause, active-operation/source-selection and physical-resource check remains per-call. A mismatch runs the exact original derivation and reason codes.
- [ ] Add meaningful RED controls for skipped warm handshakes and pause/raw/companion reuse, with full-derivation mutation oracles and dependency-completeness traces. History migration/sanitization retains write admission and collision-safe fresh temporary members.
- [ ] Implement only proven shared positive derivation reuse. Do not cache a pause verdict, replace a live FD owner, follow unchecked path components or infer authority from absence/pending state.
- [ ] Compare paired actual boot/idle workloads under unchanged production cadence and work; require a further >=50% reduction in open calls. Measure actual backup pause latency against the existing owner-approved cadence.
- [ ] Run targeted oracle/completeness/cancellation checks and independent source review. Preserve exact platform and resource limits; close all four criteria only when evidenced.

## Task 4: TASK-33640 — captured allowance provenance

**ADR required:** no; existing provider/parser and credential boundaries are unchanged.
**Files:** existing TASK-33640 Markdown and `Tests/fixtures/cloud_live/README.md` only.

- [x] Inventory every current engine preset without printing credentials: 32 total, Together/Fireworks ready, 30 clean missing-key skips.
- [x] Authenticate both existing successful live captures, exact fixture citations, zero unknown keys and clear credential-fragment scans. No new request or allowance was needed.
- [x] Fresh targeted replay through actual engine wrappers and model discovery: 19 passed, zero failed/errors/skips; preserve raw/XML and 27 late foreign-directory rm_rf warnings.
- [x] Independent evidence review; correct its sole README typo; CLI AC5 checked and task Done. Other presets remain provisional until keys arrive.

## Delivery

- [ ] Complete both source fixes and review the combined branch for composition regressions.
- [ ] Validate only modified functionality, task/ADR hygiene, whitespace and scoped static analysis; distinguish inherited debt from new findings.
- [ ] Create a reviewable PR with current evidence and practical limits. Follow protected head-matched integration only when authorized and all fresh gates qualify; do not claim a merge while only queued/armed.
