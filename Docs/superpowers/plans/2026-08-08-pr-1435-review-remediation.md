# PR #1435 Review Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebase PR #1435 onto the latest `origin/dev`, remediate every valid review finding and the failing required check, and leave the PR ready to merge with complete evidence.

**Architecture:** Preserve ADR-030's shared Library contract and both runtime surfaces. Put byte serialization and cursor bounds in the shared contract, keep Console providers as thin adapters, make the RAG result fitter monotonically reduce toward a bounded result, and use each database's transaction context for the new direct reads.

**Tech Stack:** Python 3.11+, pytest, SQLite, Textual 8.2.8+, Backlog.md, GitHub Actions.

## Global Constraints

- ADR required: no.
- ADR path: `backlog/decisions/030-local-library-agent-tool-boundary.md`.
- Reason: these fixes harden and document the existing Console/MCP Library boundary without changing ownership, storage, schemas, or service contracts.
- Every behavioral or security fix follows red-green TDD.
- Public APIs use Google-style `Args`/`Returns`/`Raises` sections.
- Every serialized Console Library result stays within `MAX_RESULT_BYTES` using the same JSON representation that was measured.
- The legacy ten MCP tools and the eighteen descriptor-backed Library tools must all render schema-driven forms.

---

### Task 1: Shared serialization and cursor bounds

**Files:**
- Modify: `tldw_chatbook/Library/library_tool_contract.py`
- Modify: `tldw_chatbook/Agents/library_tool_provider.py`
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py`
- Modify: `Tests/Library/test_library_tool_contract.py`
- Modify: `Tests/Agents/test_library_tool_provider.py`

**Interfaces:**
- Produces: `json_dumps_compact(payload: Any) -> str` and `MAX_CURSOR_CHARS`.
- Consumes: existing `serialized_size`, descriptor schemas, and `ToolResult`.

- [x] **Step 1: Add failing compact-serialization and cursor-bound tests**

  Add tests that use non-ASCII and whitespace-sensitive payloads and assert the provider's returned UTF-8 bytes equal `serialized_size(payload)` and remain within 32 KiB. Add an oversized ASCII cursor test that monkeypatches `base64.b64decode` to fail if called, proving rejection happens before allocation/decoding. Assert every descriptor `cursor` property has `maxLength == MAX_CURSOR_CHARS`.

- [x] **Step 2: Run the new tests and verify the expected failures**

  Run:

  ```bash
  .venv/bin/python -m pytest \
    Tests/Library/test_library_tool_contract.py \
    Tests/Agents/test_library_tool_provider.py -q
  ```

  Expected failures: provider JSON differs from the compact measured form; oversized cursors reach the decoder; schemas lack `maxLength`.

- [x] **Step 3: Implement the minimal shared helpers and provider use**

  Add `json_dumps_compact()` beside `serialized_size()` and make `serialized_size()` measure that helper. Add a conservative `MAX_CURSOR_CHARS = 2048`, reject longer cursors before padding/decoding, and apply the bound to each cursor schema. Replace provider `json.dumps` calls for both success and error payloads with the shared helper.

- [x] **Step 4: Run the focused tests until green**

  Run the Step 2 command and confirm zero failures.

---

### Task 2: Guarantee RAG payload fitting terminates

**Files:**
- Modify: `tldw_chatbook/Agents/library_rag_tool_provider.py`
- Modify: `Tests/Agents/test_library_tool_provider.py`

**Interfaces:**
- Consumes: `json_dumps_compact`, `serialized_size`, and `MAX_RESULT_BYTES`.
- Produces: a bounded `ToolResult` for hostile unbounded row metadata.

- [x] **Step 1: Add a failing hostile-row regression test**

  Return a single real result-shaped row with multi-megabyte `result_id`, `title`, and `runtime_backend` strings plus an empty snippet. Assert `invoke()` returns promptly, its UTF-8 content is at most `MAX_RESULT_BYTES`, and the parsed payload contains only bounded string fields.

- [x] **Step 2: Verify the test fails without hanging the suite**

  Run the single regression with pytest-timeout. Expected result: timeout or oversized/non-terminating fitting on the current implementation.

- [x] **Step 3: Bound projected row strings and make fitting monotonic**

  Bound `result_id`, `title`, and `runtime_backend` during `_project_row`. In `_success_result`, drop trailing rows first; for the final row shrink fields in a fixed order and drop the row if no shrink changes the measured size. Serialize the final result through `json_dumps_compact`.

- [x] **Step 4: Verify green and mutation behavior**

  Run `Tests/Agents/test_library_tool_provider.py`; mentally removing the final no-progress/drop-row branch must make the hostile-row test hang or fail.

---

### Task 3: Transaction and observability compliance

**Files:**
- Modify: `tldw_chatbook/DB/Prompts_DB.py`
- Modify: `tldw_chatbook/DB/ChaChaNotes_DB.py`
- Modify: `tldw_chatbook/DB/Client_Media_DB_v2.py`
- Modify: relevant tests under `Tests/Prompts_DB/`, `Tests/Notes/`, and `Tests/Media/`.

**Interfaces:**
- Consumes: each database class's existing `transaction()` context.
- Produces: the same read payloads with consistent transactional access and safe contextual error logs.

- [x] **Step 1: Add failing transaction-ownership tests**

  Instrument the real database transaction context and assert prompt overview/section, note text, and media text execute their read while the context is active. Preserve existing returned payload assertions.

- [x] **Step 2: Verify the transaction tests fail**

  Run the four new tests and confirm each reports that `execute_query()` bypassed `transaction()`.

- [x] **Step 3: Move direct reads into transaction contexts**

  Use `with self.transaction() as conn:` and `conn.execute(...)` for the four read paths. Keep SQL projections and public return shapes unchanged.

- [x] **Step 4: Add safe operation context to new Library error logs**

  Include safe values such as `limit`, `offset`, query length, UUID/id, section, start, and `max_chars` in each new list/search/get error message across the three database modules. Do not log content, raw queries, filesystem paths, or secrets.

- [x] **Step 5: Run database/service regressions**

  Run:

  ```bash
  .venv/bin/python -m pytest \
    Tests/Prompts_DB/test_prompts_db_pytest.py \
    Tests/Notes/test_notes_library_unit.py \
    Tests/Media/test_local_media_reading_service.py -q
  ```

---

### Task 4: Public API docstrings and architecture documentation

**Files:**
- Modify: `tldw_chatbook/Library/library_collections_service.py`
- Modify: `tldw_chatbook/Notes/Notes_Library.py`
- Modify: `tldw_chatbook/Media/local_media_reading_service.py`
- Modify: `tldw_chatbook/Chat/chat_conversation_service.py`
- Modify: `tldw_chatbook/Prompt_Management/Prompts_Interop.py`
- Modify: `tldw_chatbook/Skills_Interop/local_skills_service.py`
- Modify: `Docs/Design/MCP.md`

- [x] **Step 1: Complete Google-style public docstrings**

  Add explicit `Args`, `Returns`, and applicable `Raises` sections to the newly added public Library list/search/get callables identified by Qodo. Do not change behavior.

- [x] **Step 2: Correct the MCP architecture wording**

  State that the shared service and all eighteen `library_*` tools are callable by Console agents as well as local MCP clients. Narrow the FastMCP-free statement to the descriptor-backed Library dispatch path, preserving the document's legacy FastMCP server description.

- [x] **Step 3: Run static documentation/code checks**

  Run compileall, ruff on touched Python files if configured, and `git diff --check`.

---

### Task 5: Repair the stale MCP inventory test and verify the PR

**Files:**
- Modify: `Tests/UI/test_mcp_tools_mode.py`
- Modify: `backlog/tasks/task-1337 - Add-direct-local-Library-tools-for-Console-agents-and-MCP.md`

- [x] **Step 1: Update the failing test to assert behavior, not the obsolete count**

  Iterate all returned legacy and Library tool schemas and assert `parse_schema()` chooses the form renderer for every tool. Assert the manifest contains the ten named legacy tools plus the eighteen named descriptor tools, so dropping either surface fails without hard-coding only the historical count.

- [x] **Step 2: Re-run the exact failed GitHub Actions command**

  ```bash
  .venv/bin/python -m pytest \
    Tests/CI/test_textual_runtime_contract.py \
    Tests/UI/test_mcp_workbench.py \
    Tests/UI/test_mcp_tools_mode.py \
    --timeout=180 --tb=short
  ```

- [x] **Step 3: Run focused and approved verification**

  Run the Task-1337 focused suite, compileall, ruff for touched files, and `git diff --check`. A repository-wide pytest run was started, then stopped at 6% at the owner's explicit direction; it is not part of the completion evidence.

- [ ] **Step 4: Close Backlog and GitHub review hygiene**

  Record the real rebase base/tip, review fixes, test counts, and any verified baseline failures in TASK-1337. Check all acceptance criteria, set the task to Done only after the DoD is satisfied, reply inside each review thread, and resolve all eight threads.

- [ ] **Step 5: Push and merge**

  Commit the reviewed changes, push the rebased branch with `--force-with-lease`, wait for required checks to finish green, verify zero unresolved threads and mergeability, then merge PR #1435.
