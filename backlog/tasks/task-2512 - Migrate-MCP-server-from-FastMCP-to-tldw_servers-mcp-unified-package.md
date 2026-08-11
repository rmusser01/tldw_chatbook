---
id: TASK-2512
title: Migrate MCP server from FastMCP to tldw_server's mcp-unified package
status: Done
assignee: []
created_date: '2026-08-06 07:18'
updated_date: '2026-08-11 04:07'
labels:
  - mcp
  - fastmcp
  - migration
dependencies: []
references:
  - >-
    backlog/tasks/task-1337 -
    Add-direct-local-Library-tools-for-Console-agents-and-MCP.md
  - backlog/decisions/030-local-library-agent-tool-boundary.md
  - backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md
  - >-
    Docs/superpowers/specs/2026-08-09-mcp-unified-standalone-server-migration-design.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The MCP server (tldw_chatbook/MCP/server.py) currently uses the official SDK's FastMCP 1.x (optional mcp[cli] extra). It should instead use the standalone mcp-unified package from tldw_server (apps/mcp-unified, on PyPI, GPL-3.0-only — fine as a dependency for this AGPLv3+ project). Decisions (user, 2026-08-05): FULL migration — every implemented built-in tool, resources, prompts, and the phase-4 local agent tools (fs_*/git_*/web_*/fs_patch, permission-gated via LocalToolProvider) move to mcp-unified as modules served via its stdio gateway; dependency via PyPI optional extra. Research pointers: gateway in apps/mcp-unified/src/mcp_unified/gateway/{jsonrpc,fastapi,stdio}.py; modules subclass BaseModule (get_tools/execute_tool) loaded via ModuleRegistry (tldw_Server_API/app/core/MCP_unified/modules/base.py, registry.py); profiles/permission_rules.py + policy_grants/ for the permission layer. KEY UNKNOWN to resolve first: whether mcp-unified supports MCP resources and prompts (not just tools) — chatbook's server exposes both; if unsupported, the spec must decide the fallback. Also: protocol version compat with our hand-rolled client (2025-03-26), and whether serving stdio is programmatic or CLI-only. Refs: ADR-032/033, re-plan spec Docs/superpowers/specs/2026-08-05-local-agent-tools-phases-3-4-replan.md, task-2511 (FastMCP smoke — may become moot if this lands first).

Update (2026-08-08, via task-1337 close-out / PR #1435): (1) LANDMINE — `mcp[cli]>=1.0.0` now resolves to mcp 2.0.0, which REMOVES `mcp.server.fastmcp`; a fresh install therefore breaks the legacy standalone server (`TldwMCPServer`, `MCP/__main__.py`) today, and the current dev venv ships with NO `mcp` installed (the in-app surface is unaffected). This makes the migration time-sensitive rather than cosmetic. (2) Scope reduction — the 18 descriptor-backed `library_*` tools (task-1337) are already FastMCP-free: they ride the capability manifest (`MCP/server.py::_describe_local_library_tools`) plus the direct runtime delegate (`LocalMCPRuntimeDelegate.execute_tool` via `asyncio.to_thread`) with the shared-service factory `build_local_library_tool_service`; they need NO migration. Remaining scope is the implemented legacy built-ins + resources + prompts + the phase-4 local agent tools. (3) Contract to preserve — the in-app direct runtime now refuses raw protocol `tools/call` for every tool with typed `RawToolCallRefusedError` (execution only via the gated, logged Execute Local Tool action); the migrated server must keep an equivalent policy-gated path, and `Tests/MCP/test_library_tools.py` (legacy-name/shape pinning) must stay green throughout.

Integration correction (2026-08-10, approved after rebasing onto TASK-4000): the old `ingest_media` entry returned a fabricated `queued` result without submitting work, so upstream retired it. The standalone catalog therefore contains exactly nine implemented built-ins. `ingest_media` remains absent from discovery and refused by direct dispatch; persistent URL/file ingestion is documented through Library Import. This task does not restore the placeholder or expand into a real ingestion implementation.

Resolved design (2026-08-09): public `mcp-unified==0.2.1` supplies the required programmatic stdio, resources, resource templates, prompts, typed errors, and all required protocol revisions. ADR-053 and the linked specification select a thin `GatewayCoreRuntime` adapter, explicitly exclude the 18 in-app-only Library tools from standalone stdio, define canonical tool/resource/prompt mappings, bound long resources with continuation metadata, preserve the phase-4 permission gate, and retain the existing client command.

## Implementation Plan

ADR required: yes

ADR path: `backlog/decisions/053-mcp-unified-standalone-runtime-boundary.md`

Reason: This migration changes the standalone runtime/dependency, stdio and cross-module adapter contracts, resource continuation, permission-error projection, cancellation semantics, and external local-data privacy boundary.

Detailed plan: `Docs/superpowers/plans/2026-08-09-mcp-unified-standalone-server-migration.md`

1. Pin and verify the released `mcp-unified==0.2.1` public and optional-dependency contract.
2. Implement the strict Chatbook gateway adapter for built-ins and permission-gated local tools.
3. Add bounded canonical resource continuation and prompt argument/result mapping.
4. Compose the real multi-revision stdio server and make the hand-written client cursor-safe.
5. Prove wheel/sdist isolation, update public privacy/install documentation, run final gates, and close the task only after review.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 The public `mcp-unified==0.2.1` programmatic stdio, tool, resource, resource-template, prompt, typed-error, and required protocol-revision surfaces are verified against the real package.
- [x] #2 ADR-053, the reviewed migration specification, and the implementation plan define the standalone runtime, canonical mappings, privacy boundary, resource continuation, and local-tool permission behavior.
- [x] #3 The standalone server exposes exactly nine implemented built-in tools, five resource templates plus the dynamic resource catalog, five prompts, and all explicitly enabled phase-4 local-agent tools through `mcp-unified`; retired `ingest_media` remains absent from discovery and refused by direct dispatch.
- [x] #4 The eighteen in-app-only `library_*` tools remain available through their descriptor-backed direct runtime, remain absent from standalone stdio, and retain raw in-app `tools/call` refusal.
- [x] #5 Tool values, fixed/redacted typed local-tool failures, namespaced resource dictionaries, long resource chunks, non-empty prompt messages, and prompt arguments map to bounded canonical MCP results with regression coverage.
- [x] #6 Both `mcp[cli]` declarations and every live availability/dependency surface are replaced by the exact `mcp-unified==0.2.1` optional dependency, with no production FastMCP or official-SDK import remaining.
- [x] #7 Chatbook's hand-written client negotiates `2025-03-26`, the adapter passes real `2025-11-25` and current-profile flows with batching limited to `2025-03-26`, catalog pagination fails closed at malformed or over-bound cursors, resource `_meta` is preserved exactly, and the existing subprocess configuration command remains valid.
- [x] #8 Wheel and sdist `[mcp]` installs independently pass a site-packages-isolated standalone protocol smoke confined to temporary configuration/data/workspace paths, and the relevant MCP, Library, packaging, documentation, static-analysis, and security gates are green or compared against an identical clean-dev baseline.
- [x] #9 User and developer documentation explain installation, supported protocol behavior, continuation, the standalone-versus-in-app Library boundary, the Library Import path replacing retired `ingest_media`, and the privacy risk of exposing local data to an external MCP client or cloud model.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Migrated the standalone MCP runtime from the removed FastMCP surface to the exact optional dependency mcp-unified==0.2.1. A thin ChatbookGatewayRuntime preserves canonical tool, resource, resource-template, and prompt mapping while the dependency owns stdio JSON-RPC negotiation and projection. The finalized standalone surface contains nine implemented built-ins, five resource templates plus dynamic resources, five prompts, and explicitly enabled permission-gated local tools. The upstream TASK-4000 removal of the fabricated ingest_media placeholder was preserved; ingest_media remains absent/refused and documentation directs persistent ingestion to Library Import.

The eighteen library_* tools remain private to the in-process descriptor-backed runtime and raw calls remain refused. External MCP clients can read authorized private Library data and send it onward; internal diagnostics/refusals remain payload-free. Sync local handlers run through asyncio.to_thread: cancellation stops the await/output path but cannot roll back side effects already started in a worker thread.

Integration renumbered the MCP decision from ADR-052 to ADR-053 to preserve the upstream ADR-052 record without changing the architecture. Review-driven TDD hardened official legacy-name compatibility, bounded catalog/resource behavior, partial and established connection ownership, cancellation/timeout precedence, same-ID pre-spawn reservation, payload-free initialization/discovery logging, and truthful disconnect_all reporting. The same independent reviewer returned Ready with no Critical, Important, or Minor findings on clean commit 0f7200aced210038c2868d132c6ccdf630f43866 over origin/dev ced98b9a42da8fa834e7851b1e7e357bb9a7dfd2.

Modified categories: dependency and optional-feature declarations; MCP adapter/server/client/local-tool/prompt/runtime modules; protocol, lifecycle, Library-boundary, documentation, UI-harness, and artifact-isolation tests; user/developer documentation; ADR/spec/plan/task governance; and the TASK-15104 typed navigation-test stub correction.

Evidence: exact final Tests/MCP passed 1,007 tests with one known dependency warning; the final lifecycle/race subset passed 17; the documentation contract passed 59. Changed-file Ruff format/check passed; mypy over four MCP production modules, Bandit over five MCP security targets, compileall, and working-tree/committed diff checks passed. Earlier Task 8 wheel and sdist site-packages isolation passed. A final normal-network artifact refresh installed both distributions, then both encountered the newly shared upstream omission of chachanotes_v32_to_v33_console_context_memory.sql. The complete final scoped command reported 1,083 passed and four non-MCP failures: clean origin/dev reproduced the frontmatter optional-feature mismatch and the installed-wheel missing-migration failure exactly, while the two branch-only artifact cases reached that same missing migration.

Repository owner instruction: ignore all CI checks. Repository-full and CI were waived and are not green evidence. The earlier full run reached about 83% and stopped at the shared Library navigation hang; the exact node timed out at 300 seconds on both branch and clean dev. TASK-15104 updated the stale stub to NoteFlushOutcome(PERMITTED), after which the exact node passed in 1.08 seconds and its adjacent group passed in 2.18 seconds. No later repository-full or CI run is represented as successful.
<!-- SECTION:NOTES:END -->
