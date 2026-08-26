# TASK-694 Legacy Tool Ownership Reconciliation Design

- **Status:** Approved for specification review
- **Date:** 2026-08-12
- **Task:** TASK-694
- **Existing decisions:** ADR-030 (direct local Library tools), ADR-032 (local
  agent tool permission boundary)

## Goal

Close TASK-694 truthfully without resurrecting System A or duplicating tools
that already have authoritative providers. Record where each legacy capability
lives now, preserve the explicitly tested Python compatibility imports, and
correct the stale follow-up records that still assume the legacy wrappers will
become live agent tools.

## Context

TASK-694 was filed when TASK-545 narrowed its first built-in-tool port. Its
premise was that `rag_search`, `web_search`, `search_notes`, and `code_audit`
still ran through the legacy `ToolExecutor` and needed to survive that system's
retirement. The premise is no longer true:

- TASK-545 P3 deleted System A's execution machinery.
- `web_search` ships through `LocalToolProvider` under ADR-032.
- Console Library retrieval ships through `LibraryToolProvider` when direct
  Library tools are enabled and `LibraryRagToolProvider` otherwise, under
  ADR-030.
- The old audit hook installation path disappeared with System A. At design
  time, TASK-743 owned only the narrower decision to rehome file-write hooks at
  `BuiltinToolProvider.invoke` or delete `file_operation_hooks.py`; it did not
  yet account for the whole `CodeAuditTool` subsystem or local `fs_*` writes.
  This reconciliation expands TASK-743 to own that complete decision.

The remaining legacy classes are compatibility surfaces, not runtime catalog
owners. `Tools.__all__` and its PEP 562 `__getattr__` mapping intentionally keep
`WebSearchTool`, `RAGSearchTool`, and `SearchNotesTool` importable, and tests pin
that behavior. Removing them in TASK-694 would be an unrelated breaking change.

## Findings That Shape the Decision

The legacy implementations are not reference semantics worth porting:

- `RAGSearchTool` accepts `conversations` and `characters`, while
  `create_config_for_collection` recognizes `chat` and `character`. Those
  legacy values fall through to the default collection rather than selecting
  the named data source.
- `SearchNotesTool` hardcodes `user_id="default_user"`; the current Library
  service uses its own explicit service identity (`"local_library"` by
  default). The identifiers differ, so the legacy implementation is not a
  contract oracle for the Library provider even though both are currently
  fixed single-user identities.
- `CodeAuditTool` and `FileOperationMonitor` are unreachable from the live
  runtime. They retain prompts and before/after file content in a process-global
  list, log prompt/error payloads, and hardcode an Anthropic model. Porting them
  is a security and provider-policy redesign, not a registration change.
- `WebSearchTool` is already classified by TASK-1355 as a retired compatibility
  file with no production consumer. The current local `web_search` owns the
  live permission and egress contract.

## Decision

### 1. Do not add built-ins

TASK-694 adds no `_GATEABLE_BUILTINS` entries, `[tools]` flags, risk tags,
`BUILTIN_HIGH_RISK_TAGS`, or `_SHADOWED_BUILTIN_NAMES` entries. The default and
fully gate-enabled built-in inventories stay unchanged.

Adding legacy names would create duplicate catalog capabilities and bypass the
provider boundaries that now own their data and permission semantics.

### 2. Preserve compatibility imports

Keep the legacy wrapper modules and the current `Tools.__all__`/lazy-import
surface unchanged. Describe `WebSearchTool`, `RAGSearchTool`, and
`SearchNotesTool` as compatibility-only and absent from production catalogs.
Do not add runtime deprecation warnings: import-time warnings would add noise
without providing a migration mechanism, and this task does not remove the
imports.

Before TASK-694 closes, expand TASK-743 to own the entire audit subsystem's
implementation-or-deletion decision. Its description and acceptance criteria
must explicitly account for `CodeAuditTool`, `FileAuditSystem`,
`file_operation_hooks.py`, the demo, tests, and current documentation.

If TASK-743 keeps the feature, its acceptance criteria must cover every live
Console file-mutation path: the built-in `write_file` path and local
`fs_write`, `fs_edit`, and `fs_patch`. A built-in-only hook is insufficient.
The retained design must define bounded state ownership, provider/model
selection, payload-free diagnostics, prompt/content privacy, and tests proving
that audit observation cannot bypass or alter the existing permission and
workspace-confinement gates. If TASK-743 deletes the feature, its acceptance
criteria must require removal of the implementation, hook module, demo, live
documentation, and feature-specific tests, plus a stale-reference scan.

TASK-694 may correct current documentation that falsely claims the audit is
wired, but it will not change `code_audit_tool.py` or
`file_operation_hooks.py`; implementation remains with the expanded TASK-743.

### 3. Record authoritative ownership

The current capability map is:

| Legacy name | Current Console owner | Contract |
| --- | --- | --- |
| `web_search` | `LocalToolProvider` | `local:web_search`; the ADR-032 permission gate is unchanged. For each `web_search` invocation, the caller/model selects one allowlisted `search_engine`, and that selection determines the destination. The operator supplies supported per-engine credentials and configurable endpoints where available; fixed-endpoint engines remain implementation-defined. A configured Searx endpoint may be local. `web_search` does not apply public-target validation. |
| `rag_search` | `LibraryRagToolProvider.search_library_rag` when direct Library tools are off | Bounded profile-driven Library retrieval over notes, media, and conversations |
| `search_notes` | `LibraryToolProvider.library_search_notes` when direct Library tools are on | Bounded lexical search through the local Library notes service |
| `code_audit` | None | Never became a live System B capability; TASK-743 owns rehome-or-delete |

This is replacement ownership, not byte-for-byte API equivalence. The modern
providers deliberately use their own names, schemas, bounds, privacy rules,
and error contracts.

### 4. Reconcile stale governance

Update TASK-694 before implementation:

- rename it around ownership reconciliation rather than porting;
- replace its obsolete acceptance criteria with measurable inventory,
  compatibility, documentation, and governance outcomes;
- put it In Progress and add the implementation-plan link before changing
  tests or documentation.

Update TASK-545's historical scope note to record the final ownership outcome
instead of claiming all four tools remain unported.

Correct ADR-032's TASK-1354 addendum heading and TASK-1354's closeout wording
so “public-only” applies only to `web_fetch` target and redirect validation.
Record `web_search` accurately: its permission gate is unchanged. For each
`web_search` invocation, the caller/model selects one allowlisted
`search_engine`, and that selection determines the destination. The operator
supplies supported per-engine credentials and configurable endpoints where
available; fixed-endpoint engines remain implementation-defined. A configured
Searx endpoint may be local. `web_search` does not apply public-target
validation.
This is a truthfulness correction, not new egress hardening.

Narrow TASK-3500 to its genuine MCP parity work. Remove its stale agent-side
`RAGSearchTool` premise and agent ACs as already satisfied/superseded by the
profile-driven Library service consumed by `LibraryRagToolProvider`; retitle
and rewrite the description, links, and remaining criteria around MCP
`perform_rag_search` only. Do not replace the stale class name with a duplicate
agent task: the live agent provider already delegates to
`run_library_rag_search`.

Expand TASK-743's audit ownership exactly as described above, without
implementing its choice in TASK-694. Correct the historical System-A design
and plan statements that assign `code_audit` to TASK-694, and the RAG-P0
design/plan statements that assume TASK-694 or `RAGSearchTool` will own the
agent retrieval path. Preserve their historical observations while appending
the current replacement/ownership outcome.

Add a prominent current-state notice to
`Docs/Development/Agent-Tools/Claude_Code_File_Audit_System.md`: the described
audit system is not wired into the Console agent runtime and must not be relied
on for enforcement or monitoring. Preserve the historical detail for TASK-743
instead of deleting it here.

## Runtime and Data Flow

TASK-694 changes no runtime flow. The existing flow remains:

1. Console composition registers `BuiltinToolProvider`.
2. When local tools are enabled, it registers `LocalToolProvider`, including
   `web_search`.
3. On successful provider composition it registers one Library provider per
   run:
   `LibraryToolProvider` in direct mode or `LibraryRagToolProvider` in fallback
   mode. A factory failure deliberately degrades to no Library provider for
   that run.
4. Provider names join the existing collision filters and are invoked through
   their current permission, validation, result-bounding, and privacy seams.
5. Legacy wrapper imports remain outside every production catalog.

No storage, migration, network, or error-handling behavior changes.

## Testing

Add one focused ownership test module with two isolated concerns.

The inventory test proves, without invoking tools or performing database or
network I/O:

- `gateable_builtin_tools()` contains none of `rag_search`, `web_search`,
  `search_notes`, or `code_audit`;
- the default built-in catalog remains exactly `calculator` and
  `get_current_datetime`;
- `LocalToolProvider` advertises `web_search`;
- `LibraryToolProvider` advertises `library_search_notes`;
- `LibraryRagToolProvider` advertises exactly `search_library_rag`.

The compatibility test starts a fresh Python process with a temporary config
environment, resolves `WebSearchTool`, `RAGSearchTool`, and `SearchNotesTool`
through `tldw_chatbook.Tools`, and asserts their defining modules and names.
Fresh-process isolation is required because `Tools.__getattr__` caches resolved
attributes in module globals; an in-process test can stay green after a lazy
mapping is removed if an earlier test populated that cache. Import/config file
reads are allowed; the test must not invoke a tool, open an application
database, or perform network transport.

Mutation checks must show the tests fail when any replacement name is removed,
a legacy name is inserted into the gateable table, or a compatibility mapping
is removed. Existing provider suites remain the behavioral authority; this
module is an ownership/import ratchet, not a duplicate execution suite.

Run the focused provider and compatibility suites plus documentation/backlog
contract checks. Since production behavior does not change, no live network,
database, or TUI verification is required.

## Documentation and Compatibility

Current docs must distinguish three states precisely:

- **live provider tool** — callable by the Console through its current catalog;
- **compatibility import** — importable Python class with no runtime catalog
  registration;
- **historical/unwired audit design** — not a security control.

Do not claim that legacy schemas are aliases for current provider schemas. Do
not promise a removal release or add a warning mechanism in anticipation of
one.

## ADR Check

- **ADR required:** no
- **ADR path:** N/A; reuse ADR-030 and ADR-032.
- **Reason:** this task introduces no storage, provider, permission, runtime, or
  security boundary. It reconciles backlog and documentation with boundaries
  already accepted and preserves runtime behavior. The expanded TASK-743 remains
  the owner of any future audit-boundary decision and will need to perform its
  own ADR check if it keeps/redesigns the feature.

## Non-Goals

- Porting any legacy wrapper into `BuiltinToolProvider`.
- Deleting or behaviorally changing legacy compatibility imports.
- Reimplementing or deleting the audit subsystem.
- Changing Library direct/fallback selection.
- Aligning MCP retrieval with the active Library RAG profile; TASK-3500 owns
  that separate work after its stale agent-side language is corrected.
- Adding character retrieval to the Library agent provider.
- Changing permission defaults, egress policy, or result schemas.

## Success Criteria

TASK-694 is complete when the replacement ownership is test-pinned, current
docs and related task records no longer promise a four-tool built-in port, the
legacy compatibility imports remain intact, and the runtime diff contains no
provider or behavior change.
