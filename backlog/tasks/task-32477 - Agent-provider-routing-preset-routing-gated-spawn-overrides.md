---
id: TASK-32477
title: 'Agent provider routing: preset routing + gated spawn overrides'
status: In Progress
assignee: []
created_date: '2026-09-12 00:57'
updated_date: '2026-09-13 19:31'
labels:
  - agents
  - console
  - llm-routing
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Let a Console master/control agent spawn sub-agents onto a specific provider+model (incl. custom-ep: registry endpoints) or a user-configured sub-agent default, instead of always inheriting the parent's selection. Presets (AgentDefinition) carry provider/model/params routing; spawn_subagent gains provider/model ad-hoc args gated by [agents] spawn_override_enabled + spawn_override_allowlist with a final-provider guard; children never inherit parent sampling/API params (six-layer stack: preset > model profile > console.provider_defaults > registry entry params > chat_defaults > api_settings scalars); resolved target snapshot persisted on the run row for stable resume. Spec: Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Master can spawn a child onto a different provider+model via named preset; ad-hoc args honored only when enabled and allowlisted, else loud tool error
- [x] #2 Sub-agent default provider/model setting with inherit-parent fallback
- [x] #3 Children never inherit parent sampling params; preset/endpoint params precedence verified by tests
- [x] #4 Resolved target snapshot persisted; resume reuses snapshot
- [x] #5 Registry entries support optional params table (amends ADR-146)
- [x] #6 No silent cross-provider fallback; RoutingError variants surface as spawn tool errors
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Full plan: Docs/superpowers/plans/2026-09-11-agent-provider-routing.md (executed via subagent-driven development; ledger at .superpowers/sdd/2026-09-11-agent-provider-routing/progress.md). Twelve tasks as executed: T1 sampling_params module; T2 registry entry params (amends ADR-146); T3 AgentDefinition provider/params with legacy-stable fingerprint; T4 AgentRuns schema v16 (12→16 jump — dev's 13–15 land separately; idempotent ALTERs, provably merge-safe); T5 pure resolver + six-layer params; T6 spawn integration (resolver hook before budget, child config, run snapshot); T6B PLAN AMENDMENT — _StreamingModelAdapter honors per-call routing kwargs (added after T6 review confirmed the production adapter swallowed api_endpoint/api_base_url/all sampling kwargs into **_ignored; without it the feature was inert in production); T7 spawn schema gating + roster visibility; T8 continuation reuses persisted snapshot (+ re-freeze onto resumed rows); T9 settings UI (preset routing, defaults, override policy, Test routing dry-run) + endpoint modal params; T10 config template (keys commented out) + user guide; T11 sandboxed live verification (6/6 PASS, evidence below). Follow-ups filed: TASK-32497 (rail resolved-target display), TASK-32498 (custom-ep/built-in conflation), TASK-32499 (polish bundle), TASK-32532 (pre-existing D2 defect), TASK-32508 (fallback_models, pre-existing).
### PR #2651 integration recovery — 2026-09-13

1. Preserve published head `1de64e158b` and the abandoned rebase checkout; merge current `origin/dev` (`8a6ba98c0d`) into an isolated recovery branch.
2. Reconcile routing with landed orchestration ownership, definition caps, Settings editing, and gateway changes. Preserve both sets of behavior and regression coverage; advance routing migration beyond dev schema v20.
3. Regenerate diagnostics, reconcile documentation and task IDs, run targeted routing/bridge/DB/Settings and integration checks, and review the combined diff. Repair review-confirmed production gaps in complete child sampling isolation and snapshot URL precedence through the real gateway, with regression controls for ordinary session behavior.
4. Record exact verification and leave the result ready for PR publication.

ADR required: no (no new architectural decision).
ADR path: backlog/decisions/147-agent-provider-routing.md; existing orchestration ADRs govern landed behavior.
Reason: integration repair implements existing contracts; migration renumbering avoids a collision without changing data ownership.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
ADR: backlog/decisions/147-agent-provider-routing.md (amends ADR-146). Spec: Docs/superpowers/specs/2026-09-11-agent-provider-routing-design.md. Plan: Docs/superpowers/plans/2026-09-11-agent-provider-routing.md

## Live verification (Task 11, sandboxed, 2026-09-12)

Six-step sandboxed live run against the feature build (worktree tldw_chatbook-apr, branch feat/agent-provider-routing). Full report: .superpowers/sdd/2026-09-11-agent-provider-routing/task-11-report.md (main checkout); raw evidence: /tmp/apr-live/evidence/.

Sandbox guarantees: TLDW_CONFIG_PATH pointed at a COPY of the real config (real config never written — sha256 3dca638f…fa536e verified unchanged after the run); [paths] data_dir redirected so every DB (incl. agent_runs.db) is sandbox-local; llama.cpp server on 127.0.0.1:9191 serving Qwen3.8-27B (served id starts with "../", so the allowlist glob had to be custom-ep:qwen-local/*qwen* — the brief's qwen* cannot match); a logging tap on 19191 forwards to 9191 and captures full request JSON — the sandbox endpoint's base_url was deliberately pointed at the tap (documented deviation; llama server untouched). Master: deepseek/deepseek-chat, live-verified (~6 master calls total).

1. Endpoint registration — PASS: [custom_endpoints.qwen-local] written through the app's own mutation path; params { temperature = 0.2 } survive the TOML round-trip.
2. Preset + defaults + policy + Test routing — PASS: preset implementer persisted (provider custom-ep:qwen-local, model = served id, params {"temperature": 0.2}, allowlist ["calculator"]); [agents] subagent_default_provider/model + spawn_override_enabled + allowlist set; the panel's Test routing printed both the preset and the (default) resolutions ready.
3. Console delegation — PASS: real Console (deepseek master) spawned implementer; child answered ROUTING-PROOF 391 (17×23 ✓) and the master relayed it; ad-hoc spawn provider=openai model=gpt-4o rejected with the exact tool error "ERROR: [provider_not_allowlisted] provider 'openai' is not in spawn_override_allowlist"; no run row created for the refusal.
4. Run row + wire evidence — PASS: child run row shows resolved_provider=custom-ep:qwen-local, resolved_model=served id, resolved_base_url=tap, resolved_params_json {"min_p":0.05,"temperature":0.2,"top_k":50,"top_p":0.95}; the tap captured the child's actual request with the served id and temperature 0.2 — preset params overrode the chat_defaults stack (temperature 0.6) on the child only; master traffic never touched the tap.
5. Snapshot semantics — PASS: preset edited to provider=ollama mid-session (AgentRunsDB.update_agent_definition); send_to_agent on the finished child resumed it as a NEW run whose resolved_* stayed the frozen custom-ep:qwen-local snapshot (temperature 0.2; 4-message seeded transcript visible on the wire) and answered SNAP-CONT OK — continuations stay pinned even when the edit points at something invalid; conversely a NEW spawn under the ollama preset errored honestly (ConnectionError :11434), proving new spawns re-resolve the edited preset live. All 7 assertions true (stage-c-checks.json).
6. Evidence recorded here; task file committed on the branch.

Defects found during verification:
- D1 (environment): the real config's anthropic key is invalid (ChatAuthenticationError on a live probe) — master switched to deepseek.
- D2 (product defect, PRE-EXISTING — reproduced identically on master): ProjectInstructionSetupModal → Disable aborts the submit AND strands the native composer send-blocked ("Send blocked — finish provider setup to continue", 26+s, Stop does not recover) while the screen's own readiness says Ready/native — the composer's cached blocked state from the gate moment is never resynced after the abort (console_display_state.py:259 fallback paints the stale reason). Harness workaround: pre-disable project instructions on the session via the same store seam the controller's commit path uses. Evidence: probe3_readiness_delta.py (branch) / probe3_master.py (master).
- D3 (harness, not product): Stage A's first empty Test-routing report was my own button-out-of-viewport geometry.

Task left at To Do / not marked Done per the verification brief — DoD review belongs to the parent flow.

PR #2651 recovery (2026-09-13): integrated published head 1de64e158b with dev 8a6ba98c0d in isolated codex/pr-2651-resume. Preserved landed orchestration ownership/recovery, definition caps, live usage and Settings presets; routing migration is now v21. Fixed review-confirmed child sampling leakage and frozen endpoint URL precedence through the real gateway. Retained both documentation additions and corrected stale allowlist/rail claims. Existing ADR-147 applies (no new ADR); schema and delivered-scope notes updated there.

Verification: combined targeted run 1687 passed / 3 failed; Settings paint synchronization repaired and 38-test rerun passed; two remaining unchanged session-settings failures reproduce identically on clean current dev. Joined adapter/real-gateway regressions 2 passed; catalog/endpoint tests 40 passed; boot census 4 passed at 975/975; all derived checks and critical Ruff/whitespace checks pass. Independent review and scoped re-review have no outstanding findings. Full suite not run; broad lint/formatter cleanliness not claimed. Detailed evidence: Docs/superpowers/reviews/2026-09-13-pr-2651-recovery.md. Task remains In Progress pending PR publication/final closeout rather than declaring every repository DoD gate globally green.
<!-- SECTION:NOTES:END -->
