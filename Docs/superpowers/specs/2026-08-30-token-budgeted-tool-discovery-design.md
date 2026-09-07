# Token-budgeted agent tool discovery design

Date: 2026-08-30
Status: Approved
Task: [TASK-15261](../../../backlog/tasks/task-15261%20-%20Replace-the-fixed-active-tool-cap-with-token-budgeted-discovery.md)
ADR: [ADR-104](../../../backlog/decisions/104-token-budgeted-agent-tool-disclosure.md)

## Purpose

Remove the arbitrary 16-tool disclosure threshold and 24-tool lifetime active
cap. Keep the full allowed catalog reachable while bounding model-visible tool
schemas by their actual token cost and the selected model's real request
headroom.

Success means a large production catalog can reach a late-registered MCP tool
through discovery and permission review, and a run can switch to another tool
set without exhausting permanent activation slots.

## Approved direction

The user selected the dynamic-discovery direction after comparing the current
runtime with Codex and Claude Code: token-budgeted disclosure, bounded search
results, and no sticky cumulative activation limit.

The implementation uses a replaceable working set rather than LRU eviction.
This is the smallest design that removes permanent exhaustion while keeping a
fail-safe estimated request-fit bound and deterministic permission behavior.

## Disclosure policy

Two code-owned policy constants replace the count limits:

- The complete allowed catalog is disclosed automatically only when its exact
  provider-visible schema representation consumes at most 10% of the selected
  model's context window *and* the projected complete first request fits after
  the response reserve.
- `find_tools` returns at most eight entries.

Core runtime schemas remain governed by their existing feature/authority gates
and sit outside the 10% automatic-disclosure test. They are still included in
the projected whole-request fit check used for deferred loads.

The service obtains the selected model's context limit through the existing
model-limit resolver. It estimates a candidate schema *set* using the same
representation sent to that provider: OpenAI-style `tools` JSON for native tool
calling, or the rendered fence protocol otherwise. Set-level measurement is
required because list delimiters, protocol wrappers, and tokenizer boundaries
make the cost of a set different from the sum of independently measured
schemas. Estimator exceptions, non-positive results, and invalid model limits
fail safely into deferred discovery.

## Initial disclosure

The shared first-request planner receives the allowed-name set, model/provider
budget inputs, runtime feature gates, and current messages. Its pure catalog
probe walks entries in stable order, skipping disallowed names and duplicate
owners already removed by the registry.

It loads schemas and measures each cumulative candidate set only while proving
that the complete allowed catalog stays at or below 10%. The probe stops at the
first over-threshold or invalid candidate. The planner then builds the projected
first request with the complete candidate and all independently gated runtime
schemas except `find_tools`/`load_tools`. Direct disclosure is selected only if
that request also fits after the response reserve. Otherwise it returns no
catalog schemas and enables discovery. This preserves the zero-extra-round-trip
behavior for genuinely small schema payloads without using a tool count as a
proxy or overflowing a history-heavy first request.

## Search and load flow

`find_tools(query)` filters to allowed entries, ranks exact-name matches first,
then name prefixes, name substrings, and description substrings. Ties sort by
normalized name and id, never provider registration order. The first eight
entries are returned. Empty or unmatched queries retain the existing
`No matching tools` behavior.

`load_tools(ids)` resolves catalog ids and bare names, filters the allow-list,
and deduplicates by model-facing name. It considers every requested schema in
request order and admits a schema when the projected next provider request
still fits the selected model context after its response reserve. That
projected fit check uses the live request builder and includes the current
messages, system content, core runtime schemas, the candidate catalog working
set, and a deterministic load-result message. A candidate that does not fit is
reported as omitted while later smaller schemas are still considered.

The 10% value is an automatic-disclosure threshold, not a hard per-tool
reachability ceiling. A single schema larger than 10% can therefore still be
loaded when the complete projected request fits. Conversely, a load can accept
less than 10% when the current conversation leaves less headroom. This removes
the pathological case where a valid tool is searchable but can never be
loaded.

Schema selection returns a structured `ToolLoadSelection` containing accepted
schemas, budget-omitted names, and invalid inputs. It does not mutate run state.
The runtime commits the accepted schemas as the new model-visible working set
and invokes one non-throwing service callback to replace the permission-gate
name set before any later dispatch. The result text is derived from that same
selection and names loaded and budget-omitted tools. Invalid requests remain
errors. A valid request for which no schema fits returns a specific
request-budget result, never `no room`, and leaves the prior set intact.

`load_tools` is an exclusive control-plane call: if a model emits it alongside
another call in the same native parallel batch, the load is refused with a
retry-alone instruction and the working set is unchanged. Ordinary calls in
that batch continue against the pre-batch set. Multiple `load_tools` calls in
one batch are refused the same way. This prevents call ordering from changing
which already-disclosed ordinary calls remain authorized.

## Runtime and interface changes

- Remove `RunBudget.max_active_tools` and its child-budget propagation.
- Remove `DIRECT_DISCLOSE_THRESHOLD`.
- Change the runtime `load_tools` branch from append/slice semantics to working-
  set replacement.
- Replace `LoopDeps.load_schemas(ids) -> list[ToolSchema]` with a structured,
  side-effect-free selection result and add an explicit permission-name commit
  callback. This keeps invalid, omitted, and accepted outcomes distinguishable.
- Change the service closure from monotonically growing `disclosed_names` to a
  non-throwing replacement callback invoked only when the runtime commits a
  successful selection. The two updates are atomic with respect to model turns
  and later tool dispatch: no dispatch occurs between them.
- Update the `load_tools` schema description to say it replaces the current
  catalog working set, must be called alone, and requires callers to include
  every tool they want to retain.
- Pass model/provider-aware schema budgeting into first-request planning so
  preflight and the live run construct the same disclosure shape.
- Centralize automatic-disclosure and load-fit calculations in shared pure
  helpers so preview/preflight and live execution cannot drift.

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
- A mixed or repeated `load_tools` batch never mutates the working set.

## Testing

Test-first implementation will cover:

1. More than 24 compact schemas direct-disclose when their full definitions fit
   both the 10% threshold and the projected first request.
2. Fewer than 16 large schemas defer when definitions exceed the 10% threshold.
3. A history-heavy first request defers even when catalog schemas alone are
   below 10%; invalid model limits and estimator failures also defer safely.
4. Search ranking is deterministic, exact-name-first, capped at eight, and
   independent of provider registration order.
5. A deferred single schema larger than 10% still loads when the projected next
   request fits, while a conversation with less headroom admits less.
6. A second load replaces the first working set, admits newly requested tools,
   removes old permission reachability, and never emits `no room`.
7. Invalid, budget-omitted, and accepted load outcomes remain distinguishable;
   failed and mixed-batch loads preserve the previous working set.
8. Native and fence paths count the set-level representations they actually
   send, including wrapper/list overhead.
9. A shipped-size catalog with MCP registered last completes
   find → load → ask/approve → execute.
10. Existing agent runtime/service, Console bridge, MCP permission, skill, and
   local-tool disclosure tests remain green after their obsolete count pins are
   rewritten as token-budget assertions.

## Scope

This task does not add a user setting, BM25 or embedding search, cross-run tool
memory, automatic LRU eviction, or a new compaction subsystem. Those features
would add state and policy without being needed to remove the fixed cap.
