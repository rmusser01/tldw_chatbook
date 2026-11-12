# Served-Mode Residuals Burn-down Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the residuals recorded after PR #3142 — the TASK-35133 engine stall (investigation + fix) and the three PR-review minors.

**Architecture:** Two tasks on one branch off `dev`. Task 1 is a systematic-debugging investigation (reproduce → diagnose with evidence → minimal fix → regression test); Task 2 is a small batch of recorded one-liner-class minors. No new ADRs (all within ADR-234/236 surfaces).

**Tech Stack:** existing only.
Branch: `fix/served-burndown-residuals` off `origin/dev`. Worktree: /Users/macbook-dev/Documents/GitHub/tldw-auth-phase1a (venv .venv).

## Global Constraints

- Task 1 MUST follow systematic-debugging discipline: reproduce the stall in a bounded test FIRST, collect evidence (stack dumps / await-chain inspection — e.g., `asyncio.wait_for` around the reproduction with a task dump on timeout) BEFORE proposing a cause; the fix must be minimal and evidence-backed. If the evidence points outside the served surface (a pre-existing Console/TUI defect with its own owners), fix only if small and clearly-scoped, else document precisely and STOP that task.
- Standard baselines (name-identical): 11 integration + 4 dep-gate + 209 Agents + gateway/spawn sets; green sets from #3142 must stay green (webauthn 11, web_auth 85, session_store 7, cli_pair 3, approval_constants 2, mobile service 29 / api 30 / page 23, control protocol 64, relay publisher 16, ownership pin 83).
- TASK-494: any new/changed diagnostic statements get the review-then-write pin treatment in the same commit.
- TDD; targeted pytest only; one commit per task; TASK-35133 gets ACs + Implementation Notes + Done at close-out.

---

### Task 1: Engine stall investigation + fix (TASK-35133)

**Files:** TBD by the investigation (suspects: `tldw_chatbook/Web_Server/mobile_service.py` context construction, `Chat/console_chat_controller.py::compose_run_mcp_provider`, `MCP/unified_control_plane_service.py::local_external_catalog`); Test: `Tests/Web_Server/test_mobile_service.py` (regression).

**Symptom (from the T2 report, stash-verified on dev):** with a gated provider gateway parking a provider call mid-run, the engine's second agent step stalls 30 s+ inside `compose_run_mcp_provider` → `UnifiedMCPControlPlaneService.local_external_catalog`, status STREAMING, under the stall watchdog.

**Steps:**
1. Reproduce: a mobile-service agent test with a gateway that parks the FIRST provider call (gate/Event), then a tool-bearing second step; assert the second provider call arrives within a bound (e.g., 10 s) — red today (times out with a task dump on failure: dump all asyncio tasks' await stacks to pinpoint the blocking await).
2. Diagnose from the dump: what does `local_external_catalog` await that the parked turn holds? (Candidate classes: a lock shared with the turn's serialization; a per-service cache being built synchronously behind the parked stream; a blocking (non-async) I/O call on the loop; a watchdog that parks on a condition.) The evidence decides — no guessing into the fix.
3. Fix minimally at the layer the evidence names; prefer not holding a lock across the provider await, or making the catalog call lock-free/cached, per the finding.
4. Regression test stays green; full mobile suites green; baselines name-identical; diagnostic pin if any statement changed.
5. Commit: "fix(web-server): unblock the second agent step while a provider call is parked (TASK-35133)" + close the backlog task (ACs: reproduction test exists and passes; stall no longer reproduces within bound; root cause documented in Implementation Notes).

### Task 2: The three PR-#3142 minors

**Files:** `tldw_chatbook/Web_Server/mobile_api.py`, `mobile_service.py`, `static/mobile/app.js`; tests.

1. **child_id wire passthrough:** `handle_decide` forwards an optional `child_id` (string, from the JSON body) to `decide_approval(child_id=...)` — exact addressing for desktop rounds; absent keeps the deterministic oldest-entry fallback. Validation: non-str/oversized → 400 invalid_child_id? NO — follow the surface's family: ignore-if-absent, `400 invalid_decisions`-style only if present-but-malformed (add `invalid_child_id`). app.js: the approval card's decide POST includes its round's `child_id` when present. Tests: route forwards (fake service records kwargs); collision integration case routes exactly.
2. **Fingerprint-cache projection gate:** `session_messages`' cache adds the projection-equality second line the seam's docstring recommends (compare the cached wire's message count + last visible length against the seam's values; mismatch → rebuild). Test: a fingerprint-invisible mutation (equal-length edit simulated via the seam's documented caveat path — construct directly against the store, e.g., a same-length content edit on the persisted message bypassing append_stream_chunk) forces a rebuild.
3. **Workspace-label comment tightening:** `_persisted_row_workspace_id` documents the CONSOLE_GLOBAL_WORKSPACE_ID-by-assumption invariant (mobile store never switches workspaces; hydrate resolves the true value) — comment-only, no behavior change.

Commit: "chore(web-server): child_id wire passthrough + cache projection gate + workspace note (PR 3142 minors)".

---

## Self-Review

**Coverage:** TASK-35133 (T1) + all three #3142-recorded minors (T2). The advisory publish-gating corner (T6 review, "likely unreachable") stays recorded-only unless the T1 evidence touches it. Dropped earlier and still dropped: everything in the prior plan's dropped list.
**Type consistency:** `decide_approval(child_id=...)` kwarg exists from #3142 (T6); T2 only forwards it. No new cross-task seams.
