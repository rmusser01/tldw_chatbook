# Token-budgeted agent tool discovery design

Date: 2026-08-30
Status: Proposed for user review
Task: [TASK-15261](../../../backlog/tasks/task-15261%20-%20MCP-tool-reachability-is-unpinned-under-the-shipped-default-catalog.md)
ADR: [ADR-104](../../../backlog/decisions/104-token-budgeted-agent-tool-disclosure.md)

## Purpose

Remove the arbitrary 16-tool disclosure threshold and 24-tool lifetime active
cap. Keep the full allowed catalog reachable while bounding model-visible tool
schemas by their actual token cost relative to the selected model's context.

Success means a large production catalog can reach a late-registered MCP tool
through discovery and permission review, and a run can switch to another tool
set without exhausting permanent activation slots.

## Approved direction

The user selected the dynamic-discovery direction after comparing the current
runtime with Codex and Claude Code: token-budgeted disclosure, bounded search
results, and no sticky cumulative activation limit.

The implementation uses a replaceable working set rather than LRU eviction.
This is the smallest design that removes permanent exhaustion while keeping a
hard, model-relative context bound and deterministic permission behavior.

## Disclosure policy

Two code-owned policy constants replace the count limits:

- Catalog schemas may consume at most 10% of the selected model's context
  window.
- `find_tools` returns at most eight entries.

Core runtime schemas remain governed by their existing feature/authority gates
and sit outside the deferrable catalog allowance. The allowance applies only to
provider catalog schemas.

The service obtains the selected model's context limit through the existing
model-limit resolver. It estimates a schema using the same representation sent
to that provider: OpenAI-style `tools` JSON for native tool calling, or the
rendered fence protocol otherwise. Estimator exceptions, non-positive results,
and invalid model limits fail safely into deferred discovery.

## Initial disclosure

`initial_disclosure` receives the allowed-name set, the derived token allowance,
and an injected schema-cost function. It walks catalog entries in stable order,
skipping disallowed names and duplicate owners already removed by the registry.

It loads and accumulates schemas only while proving that the complete allowed
catalog fits. The probe stops at the first over-budget or invalid schema. A
complete fit returns every allowed schema and omits `find_tools`/`load_tools`;
otherwise it returns no catalog schemas and enables discovery. This preserves
the zero-extra-round-trip behavior for genuinely small schema payloads without
using a tool count as a proxy.

## Search and load flow

`find_tools(query)` filters to allowed entries, ranks exact-name matches first,
then name prefixes, name substrings, and description substrings. Ties sort by
normalized name and id, never provider registration order. The first eight
entries are returned. Empty or unmatched queries retain the existing
`No matching tools` behavior.

`load_tools(ids)` resolves catalog ids and bare names, filters the allow-list,
and deduplicates by model-facing name. It considers every requested schema in
request order and admits a schema when its cost fits the remaining allowance;
an oversized schema is reported as omitted while later smaller schemas are
still considered.

The accepted schemas replace the previous catalog working set atomically. The
runtime's model-visible active list and the service's permission-gate name set
are replaced from the same accepted result. The result text names loaded and
budget-omitted tools. Invalid requests remain errors. A valid request for which
no schema fits returns a specific token-budget result, never `no room`.

## Runtime and interface changes

- Remove `RunBudget.max_active_tools` and its child-budget propagation.
- Remove `DIRECT_DISCLOSE_THRESHOLD`.
- Change the runtime `load_tools` branch from append/slice semantics to working-
  set replacement.
- Keep `LoopDeps.load_schemas(ids) -> list[ToolSchema]`; replacement is the
  runtime meaning of the returned list, so no additional abstraction is needed.
- Change the service closure from monotonically growing `disclosed_names` to
  clearing and replacing that mutable set when a load succeeds.
- Pass model/provider-aware schema budgeting into first-request planning so
  preflight and the live run construct the same disclosure shape.

No database migration or persisted-config migration is required. The removed
field is an internal dataclass default and is not a shipped Settings value.

## Error handling and safety

- Unknown model context, schema-load failure during the initial fit probe, or
  token-estimator failure selects deferred discovery.
- Invalid ids and disallowed names never enter the working set.
- Permission review and execution still occur after disclosure; discovery does
  not grant authority.
- The permission gate blocks calls to tools replaced out of the current working
  set.
- A failed load does not destroy the current working set. Replacement occurs
  only when at least one valid schema is accepted; an explicit valid-but-over-
  budget result leaves the previous set intact and tells the model to refine.

## Testing

Test-first implementation will cover:

1. More than 24 compact schemas direct-disclose when their full definitions fit.
2. Fewer than 16 large schemas defer when definitions exceed the allowance.
3. Invalid model limits and estimator failures defer safely.
4. Search ranking is deterministic, exact-name-first, capped at eight, and
   independent of provider registration order.
5. A second load replaces the first working set, admits newly requested tools,
   removes old permission reachability, and never emits `no room`.
6. Native and fence paths count the representations they actually send.
7. A shipped-size catalog with MCP registered last completes
   find → load → ask/approve → execute.
8. Existing agent runtime/service, Console bridge, MCP permission, skill, and
   local-tool disclosure tests remain green after their obsolete count pins are
   rewritten as token-budget assertions.

## Scope

This task does not add a user setting, BM25 or embedding search, cross-run tool
memory, automatic LRU eviction, or a new compaction subsystem. Those features
would add state and policy without being needed to remove the fixed cap.
