# TASK-492: Metadata-only high-risk diagnostics implementation plan

> **Current-dev reconciliation (2026-07-27):** TASK-545 later retired
> `ToolExecutor`, its result cache, and its history after verifying zero
> production execution callers. The ToolExecutor-specific steps below are
> historical evidence, not live implementation instructions. Do not restore
> System A; the live tool persistence boundary is MCP's metadata-only JSONL
> execution log.

> **Execution:** follow the repository TDD and verification workflows. Do not
> mark the task complete until the checked inventory and sentinel matrix pass
> against the real persistent sinks.

**Goal:** Prevent Chat, provider, summarization, tool, and MCP payloads from
reaching persistent diagnostics while retaining bounded operational metadata.

**Architecture:** ADR-029 already defines the local private-data boundary, so
no new ADR is required. The persistent application file handler will admit
only records emitted through a strict metadata helper for Chatbook-owned code;
UI and terminal handlers remain unchanged. A checked AST inventory assigns
every production diagnostic owner to TASK-492 or TASK-494 and fingerprints
the persistent-sink topology. MCP execution records become bounded metadata
structures instead of redacted payload containers. The historical ToolExecutor
history remediation was superseded by deletion.

**Tech stack:** Python 3.11, stdlib logging/AST/JSON, Loguru bridge, pytest.

## ADR check

- ADR required: no
- ADR path: `backlog/decisions/029-local-private-data-boundary.md`
- Reason: TASK-492 directly implements the already accepted metadata-only
  diagnostic, tool-history, and MCP persistence boundaries.

## Task 1: Establish the persistent metadata admission boundary

**Files:**

- Create: `tldw_chatbook/Utils/persistent_diagnostics.py`
- Modify: `tldw_chatbook/Logging_Config.py`
- Test: `Tests/test_persistent_diagnostic_boundary.py`

1. Write failing tests showing standard-logging and Loguru sentinels currently
   reach a real `PrivateRotatingFileHandler`.
2. Add a file-handler-only filter that rejects Chatbook-owned records unless a
   strict helper created a schema-validated metadata event.
3. Preserve unfiltered UI/terminal handling and forward Loguru record metadata
   into standard logging.
4. Verify normal, debug, and error records plus rotation.

## Task 2: Historical ToolExecutor remediation — superseded by deletion

**Files:**

- Modify: `tldw_chatbook/Tools/tool_executor.py`
- Test: `Tests/Tools/test_tool_executor_privacy.py`

This section records what was implemented before current-dev TASK-545 removed
the zero-caller executor. It must not be replayed.

1. Historical: write failing success, parse-error, timeout, exception, and cache tests with
   private sentinels in argument values, unknown keys, results, and errors.
2. Replace the unbounded mutable history with a 100-entry deque.
3. Store only tool identity, status, duration, registered argument names,
   unknown-argument count, cache status, result type, and result size.
4. Keep immediate return values and the bounded result cache unchanged.
5. Emit only strict persistent metadata events from the execution path.

## Task 3: Make MCP JSONL execution records metadata-only

**Files:**

- Modify: `tldw_chatbook/MCP/execution_log.py`
- Modify: `tldw_chatbook/MCP/unified_control_plane_service.py`
- Modify: MCP and UI audit tests that consume the record schema
- Test: `Tests/MCP/test_execution_log.py`

1. Write failing disk-sentinel tests for arguments, results, HTTP/parser
   failures, timeouts, and execution-log append failures.
2. Replace argument/result/error fields with argument names,
   unknown-argument count, result type/size, status/error category, exception
   type, and optional HTTP status.
3. Ensure the recorder cannot mask the tool result and logs only a sanitized
   recorder-failure category.
4. Update audit rendering to the new public metadata schema.

## Task 4: Check the repository-wide diagnostic inventory

**Files:**

- Create: `scripts/check_persistent_diagnostic_inventory.py`
- Create: `Docs/security/production-diagnostic-inventory.json`
- Test: `Tests/Architecture/test_persistent_diagnostic_inventory.py`

1. Inventory every production standard-logging and Loguru owner.
2. Assign Chat/provider/summarization/tool/MCP owners to TASK-492 and all
   remaining owners to TASK-494; record reviewed exclusions with reasons.
3. Fingerprint call sites and persistent-sink topology so additions and
   changes fail the guard.
4. Run the guard and review the generated owner report before checking it in.

## Task 5: Run the TASK-492 sentinel matrix and reconcile the task

**Files:**

- Modify: `backlog/tasks/task-492 - Remove-private-payloads-from-persistent-diagnostics-and-tool-history.md`

1. Run focused tests for logging, the System A retirement sentinel, MCP
   execution logging, and control-plane integration.
2. Run the parameterized sentinel matrix covering success, HTTP/parsing
   failure, timeout, streaming, and cache paths through standard logging,
   Loguru, MCP JSONL, and the real rotating sink.
3. Run formatting/static checks for changed Python and `git diff --check`.
4. Check every acceptance criterion, add concise implementation notes linking
   ADR-029 and this plan, and set TASK-492 to Done only after all evidence is
   green.
