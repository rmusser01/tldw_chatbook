# PR 1612 Review Follow-Up Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close both late Qodo findings from PR #1612 with focused, contract-preserving fixes.

**Architecture:** Keep validation and retry policy in their existing owners. Translate the shared transport's internal URL exception at the public boundary, and align only Z.ai's adapter fallback with its established canonical default.

**Tech Stack:** Python 3.12, pytest, Ruff

**ADR required:** no new ADR

**ADR path:** `backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

**Reason:** The correction restores the existing typed-error and provider-default contracts without changing ADR-063's boundary or policy.

---

### Task 1: Reopen And Record The Follow-Up

**Files:**
- Modify: `backlog/tasks/task-15676 - Harden-Moonshot-Kimi-and-Z.ai-GLM-as-first-class-hosted-providers.md`

- [x] **Step 1: Reopen TASK-15676 before implementation**

Use the Backlog CLI to set the task to `In Progress` and record this post-merge
review correction in its implementation plan.

### Task 2: Pin The Late Review Findings

**Files:**
- Modify: `Tests/LLM_Calls/test_hosted_chat.py`
- Modify: `Tests/LLM_Calls/test_zai.py`

- [x] **Step 1: Add the hosted transport regression**

Add a test that supplies a malformed URL containing a canary, asserts the exact
public `ChatProviderError` type and suppressed exception context, and asserts
that neither the URL nor canary appears in the formatted traceback.

- [x] **Step 2: Add the Z.ai fallback regression**

Extend the existing defaults assertion to require `retry_delay == 5.0` while
retaining the configured and explicit-precedence assertions.

- [x] **Step 3: Run both tests to verify RED**

Run:

```bash
.venv/bin/python -m pytest -q \
  Tests/LLM_Calls/test_hosted_chat.py::test_owned_json_post_maps_invalid_base_url_to_redacted_transport_error \
  Tests/LLM_Calls/test_zai.py::test_resolve_zai_request_uses_canonical_precedence_and_current_defaults
```

Expected: two failures matching the reported exception leak and `1.0 != 5.0`.

### Task 3: Apply The Minimal Corrections

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/hosted_chat.py`
- Modify: `tldw_chatbook/LLM_Calls/zai.py`

- [x] **Step 1: Translate the URL validation exception**

Wrap only `normalize_hosted_chat_base_url()` and raise the existing context-free
transport configuration error from `None`.

- [x] **Step 2: Align the Z.ai fallback**

Change the adapter-only `retry_delay` fallback from `1.0` to `5.0`.

- [x] **Step 3: Run the two regressions GREEN**

Run the exact Task 2 command. Expected: `2 passed`.

- [x] **Step 4: Run focused related verification**

Run:

```bash
.venv/bin/python -m pytest -q Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_zai.py
.venv/bin/ruff check tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/zai.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_zai.py
.venv/bin/ruff format --check tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/zai.py Tests/LLM_Calls/test_hosted_chat.py Tests/LLM_Calls/test_zai.py
.venv/bin/python -m compileall -q tldw_chatbook/LLM_Calls/hosted_chat.py tldw_chatbook/LLM_Calls/zai.py
git diff --check
```

- [x] **Step 5: Commit, push, review, and merge**

Record the correction and evidence in TASK-15676's Implementation Notes, check
its acceptance criteria, return it to `Done` through the Backlog CLI, commit the
four source/test files plus task/design/plan documentation, push the follow-up
branch, open a PR against `dev`, resolve both original PR #1612 threads with the
correction commit/PR, wait for required checks, and merge.

### Task 4: Close The Late Compliance Review

**Files:**
- Modify: `tldw_chatbook/LLM_Calls/zai.py`
- Modify: `Tests/LLM_Calls/test_zai.py`
- Modify: `Tests/LLM_Calls/test_hosted_chat.py`

- [x] **Step 1: Name the Z.ai retry fallback**

Introduce one provider-local default constant, use it in resolution, and have
the existing defaults contract reference that owner.

- [x] **Step 2: Document the new regression callable**

Add a concise docstring to the new hosted transport test. No empty
`Args`/`Returns`/`Raises` sections are needed because the test accepts nothing,
returns nothing, and asserts rather than exposing an exception contract.

- [x] **Step 3: Run only the directly related tests and static checks**

Run the two exact regressions, both complete touched test modules, Ruff
lint/format, MyPy for Z.ai, compileall for Z.ai, and `git diff --check`.

- [x] **Step 4: Resolve PR #1614's two threads and merge the final follow-up**

Reply with the exact correction commit/PR, resolve both threads, verify there
are no remaining actionable comments, and merge into `dev`.
