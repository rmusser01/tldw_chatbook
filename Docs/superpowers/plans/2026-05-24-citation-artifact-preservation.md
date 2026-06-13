# Citation Artifact Preservation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve structured citation/snippet metadata when a Console assistant response is saved as a Chatbook artifact.

**Architecture:** Keep PR scope limited to Console-saved artifact metadata and existing Home/Artifacts resume payloads. Reuse the answer-level validation payload and staged evidence bundle already produced by the citation pipeline; do not redesign Chatbook ZIP export/import in this slice.

**Tech Stack:** Python 3.12, Textual event handlers, existing Chatbook registry metadata, pytest.

---

### Task 1: Preserve Citation Payloads In Console-Saved Chatbook Metadata

**Files:**
- Modify: `tldw_chatbook/Event_Handlers/Chat_Events/chat_events.py`
- Test: `Tests/Event_Handlers/Chat_Events/test_chat_events.py`

- [x] **Step 1: Write failing metadata regression**

Add a test proving `_console_chatbook_artifact_metadata()` copies `citation_validation_payload` and `citation_evidence_bundle` from the assistant message widget into Chatbook metadata.

- [x] **Step 2: Run test and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_console_chatbook_artifact_metadata_preserves_citation_payloads --tb=short
```

Expected: FAIL because metadata omits citation payloads.

- [x] **Step 3: Implement minimal preservation**

Add JSON-safe bounded structured metadata copying for `citation_validation` and `evidence_bundle`, preferring widget-level payloads and falling back to app-level state where appropriate.

- [x] **Step 4: Run focused metadata tests**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_console_chatbook_artifact_metadata_preserves_citation_payloads Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_console_chatbook_artifact_metadata_preserves_falsey_simple_values --tb=short
```

Expected: PASS.

### Task 2: Expose Citation Status Through Artifacts/Home Resume Payloads

**Files:**
- Modify: `tldw_chatbook/UI/Screens/artifacts_screen.py`
- Modify: `tldw_chatbook/Home/active_work_adapter.py`
- Test: `Tests/UI/test_console_live_work_handoffs.py`
- Test: `Tests/Home/test_active_work_adapter.py`

- [x] **Step 1: Write failing payload regressions**

Add tests proving Chatbook artifact metadata with `citation_validation` and `evidence_bundle` produces safe summary payload fields such as citation status, cited labels, source count, and snippet count.

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_citation_metadata Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_preserves_console_saved_chatbook_citation_metadata --tb=short
```

Expected: FAIL because payload pass-through does not expose citation summaries.

- [x] **Step 3: Implement minimal payload summarization**

Add shared-shape sanitization in the existing Home and Artifacts metadata adapters without exposing unbounded raw content.

- [x] **Step 4: Run focused UI regressions**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_citation_metadata Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_preserves_console_saved_chatbook_citation_metadata --tb=short
```

Expected: PASS.

### Task 3: Backlog Notes And Verification

**Files:**
- Modify: `backlog/tasks/task-60.4.4 - Post-release-citation-and-snippet-carry-through-tranche.md`

- [x] **Step 1: Update implementation notes**

Record that this slice preserves citation/snippet metadata in Console-saved Chatbook artifact registry metadata and resume payloads, while ZIP export/import remains a follow-up.

- [x] **Step 2: Run final verification**

Run:

```bash
../../.venv/bin/python -m pytest -q Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_console_chatbook_artifact_metadata_preserves_citation_payloads Tests/Event_Handlers/Chat_Events/test_chat_events.py::test_console_chatbook_artifact_metadata_preserves_falsey_simple_values Tests/UI/test_console_live_work_handoffs.py::test_artifacts_destination_reopens_console_saved_chatbook_with_citation_metadata Tests/Home/test_active_work_adapter.py::test_local_notification_adapter_preserves_console_saved_chatbook_citation_metadata Tests/Chat/test_answer_citations.py --tb=short
git diff --check
```

Expected: all tests pass and diff hygiene is clean.
