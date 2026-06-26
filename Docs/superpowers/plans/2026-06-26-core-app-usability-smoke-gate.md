# Core App Usability Smoke Gate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove latest `dev` still supports the core first-use app path without starting Sync or Persona work.

**Architecture:** Keep this as a focused smoke/evidence slice over existing app behavior. Add automated coverage only where current smoke tests leave a gap, and treat any newly found product defect as either an in-scope fix with a failing test first or a separate backlog task if it belongs to another owner.

**Tech Stack:** Python 3.11, pytest, Textual `run_test`, existing `TldwCli` test harness, Backlog.md.

---

### Task 1: Confirm TASK-88 Readiness Gate

**Files:**
- Read: `backlog/tasks/task-88 - Functionalize-Settings-MCP-defaults-after-Unified-MCP-upgrade.md`
- Read: `tldw_chatbook/tldw_api/mcp_unified_client.py`
- Read: `tldw_chatbook/tldw_api/mcp_unified_schemas.py`
- Read: `tldw_chatbook/MCP/server_unified_service.py`
- Read: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: `backlog/tasks/task-139 - Latest-dev-core-app-usability-smoke-gate.md`

- [x] **Step 1: Check for a server-first persisted MCP defaults contract**

Run targeted `rg` and method-list scans for MCP defaults/settings APIs.

Expected: No `MCPUnifiedClient`, schema, or service method exposes persisted MCP defaults separate from runtime/governance management.

- [x] **Step 2: Decide ADR need**

ADR required: no.
Reason: TASK-139 does not add a new architecture boundary; it records that TASK-88 remains contract-gated and adds a QA smoke gate.

### Task 2: Add Focused Core Smoke Coverage

**Files:**
- Create or modify: `Tests/UI/test_latest_dev_core_app_usability_smoke.py`
- Reuse patterns from: `Tests/UI/test_product_maturity_phase1_navigation_smoke.py`
- Reuse patterns from: `Tests/UI/test_product_maturity_phase1_core_loop.py`
- Reuse patterns from: `Tests/UI/test_product_maturity_phase6_focus_visual_sweep.py`

- [x] **Step 1: Write the smoke test first**

Add a test that starts from a clean first-run environment, reaches Home, navigates only through core non-Sync/non-Persona destinations, and asserts no traceback, empty chrome, raw object repr, or local path leak is rendered.

- [x] **Step 2: Run the smoke test**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_latest_dev_core_app_usability_smoke.py --tb=short
```

Expected: If the behavior already works, the new gate passes. If it exposes a concrete defect, keep the failing test and fix only that defect.

- [x] **Step 3: Add Console recovery and Library-to-Console checks**

Cover the setup-required Console state and deterministic Library/Search-RAG handoff to Console staged context by reusing existing harness patterns instead of changing product behavior.

### Task 3: Verify And Record Evidence

**Files:**
- Create: `Docs/superpowers/qa/core-app-usability-smoke/2026-06-26-latest-dev-core-app-smoke.md`
- Modify: `backlog/tasks/task-139 - Latest-dev-core-app-usability-smoke-gate.md`

- [x] **Step 1: Run focused verification**

Run:

```bash
.venv/bin/python -m pytest -q Tests/UI/test_latest_dev_core_app_usability_smoke.py Tests/UI/test_product_maturity_phase1_navigation_smoke.py Tests/UI/test_product_maturity_phase1_core_loop.py --tb=short
```

- [x] **Step 2: Run diff hygiene**

Run:

```bash
git diff --check
```

- [x] **Step 3: Record evidence and close task**

Record command outcomes, scope exclusions, and any follow-up tasks in the QA evidence file. Then check off TASK-139 acceptance criteria, add implementation notes, and set status to Done.
