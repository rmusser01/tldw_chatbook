# Local Citation Key Provisioning Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make canonical local citation writes usable on a fresh enabled
profile without ever replacing a missing key for existing fingerprint-bearing
data.

**Architecture:** Keep key creation in the concrete production keyring adapter
and policy/database eligibility in the existing local composition root. Reuse
`CitationTraceRepository.fingerprint_bearing_rows_exist()` as the replacement
guard; do not change the schema, Console, modal, or Library.

**Tech Stack:** Python 3.11+, SQLite, keyring, pytest

---

### Task 1: Specify secure keyring provisioning

**Files:**

- Modify: `Tests/Chat/test_citation_trace_identity.py`
- Modify: `tldw_chatbook/Chat/citation_trace_identity.py`

- [x] Add a failing test proving the secure concrete adapter creates one
  32-byte secret only when the keyring entry is absent.
- [x] Add failing tests proving existing valid keys are reused and invalid,
  insecure, unreadable, or unwritable backends fail closed without replacement.
- [x] Run only the new identity tests and confirm they fail because provisioning
  is not implemented.
- [x] Add the smallest concrete-adapter method that validates the backend,
  checks before writing, stores `secrets.token_bytes(32)` as base64, and
  validates the read-back value.
- [x] Re-run the identity tests and confirm they pass.

### Task 2: Provision only at the production composition boundary

**Files:**

- Modify: `Tests/Chat/test_citation_service_factory.py`
- Modify: `tldw_chatbook/Chat/citation_service_factory.py`

- [x] Add a failing factory test using a fresh v27 database, enabled policy,
  and secure fake keyring; assert the repository becomes write-ready and the
  same secret is reused on a second composition.
- [x] Add a failing factory test with fingerprint-bearing rows and a missing
  key; assert no key is created and writes remain unavailable.
- [x] Run only the new factory tests and confirm the expected failures.
- [x] Under the database transaction boundary, compose once, call the concrete
  adapter only when writes are enabled, identity exists, the key is missing,
  and the existing row guard is false, then recompose with the stored key.
- [x] Re-run both focused test modules and confirm they pass.

### Task 3: Verify the real user path and close out

**Files:**

- Modify: `Docs/superpowers/qa/2026-07-27-task-553-rag-citation-uat.md`
- Modify: `backlog/tasks/task-553.17 - Provision-local-citation-fingerprint-key-on-first-enabled-use.md`
- Modify: `backlog/tasks/task-553 - Canonical-RAG-citation-provenance-epic.md`

- [x] Run citation identity/factory plus the previously scoped Console and
  Library citation tests; do not run the full suite.
- [x] Repeat the isolated rendered UAT through live generation, reconnect,
  `Sources (1)`, exact chunk display, and exact Library open.
- [x] Record database row evidence, screenshots, and any limitations in the UAT
  report.
- [x] Mark TASK-553.17 and TASK-553 Done only if every acceptance criterion and
  rerun gate passes.
- [x] Run touched-file static checks and `git diff --check`.

ADR required: no

ADR path:
`backlog/decisions/024-rag-citation-provenance-and-source-resolution.md`

Reason: This directly implements the already-approved first-use key rule and
does not change storage, security policy, or module boundaries.
