# ADR-104: Use token-budgeted replaceable tool working sets

Status: Proposed
Date: 2026-08-30
Related Task: [TASK-15261](../tasks/task-15261%20-%20MCP-tool-reachability-is-unpinned-under-the-shipped-default-catalog.md)
Supersedes: The fixed-count active-set clause in the progressive-tool-disclosure section of the [agent runtime vertical-slice design](../../Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md)

## Decision

Agent catalog disclosure is governed by estimated model-visible schema tokens,
not tool counts.

The runtime derives a catalog-schema allowance equal to 10% of the selected
model's context window. Core runtime tools such as `find_tools`, `load_tools`,
and `spawn_subagent` retain their existing independent gates and do not consume
this deferrable-catalog allowance.

At run start, the registry probes allowed catalog schemas in stable order. If
the complete allowed catalog fits the allowance, every schema is disclosed
directly and discovery tools are omitted. The probe may stop as soon as the
allowance is exceeded. An invalid model limit, failed schema load, or failed
token estimate selects deferred discovery rather than guessing that the
catalog fits.

For deferred catalogs, `find_tools` searches every allowed catalog entry. It
ranks exact-name, name-prefix, name-substring, and description matches in that
order, with deterministic name/id tie-breaking, and returns at most eight
entries. `load_tools` resolves ids and bare names, deduplicates by model-facing
name, estimates the exact native or fence-protocol representation used for the
selected provider, and admits every requested schema that fits the allowance.
One oversized schema cannot prevent a later smaller requested schema from
being considered.

Each successful `load_tools` call atomically replaces the catalog working set.
It does not append to a lifetime set. The permission gate replaces its
disclosed-name set in the same operation, so the gate and model-visible schemas
remain identical. A later load can always replace earlier tools; the runtime
has no permanent `no room` state. Previously replaced tools can be found and
loaded again when needed.

## Context

The original vertical slice used a small-catalog shortcut of eight tools and a
run budget of eight active tools. Those values were raised to 16 and 24 without
a provider limit or benchmark establishing either number. Because loaded tools
were never removed, the active-tool budget was a one-way ratchet. Once full, a
valid later tool could be registered, permitted, and discoverable but still
unusable.

The shipped catalog is already larger than the direct-disclosure count. MCP
providers register after built-in, local, and Library providers, so fixed
prefix/count behavior creates an avoidable risk that registration order affects
reachability. Current Codex and Claude Code designs instead defer large catalogs
and expose bounded relevant subsets; Claude Code's automatic policy uses tool
schema cost relative to the model context rather than an arbitrary catalog
count.

This is a runtime and provider-boundary contract: it changes which schemas enter
model context, which names the permission gate may execute, and how future tool
providers scale. An ADR is therefore required.

## Alternatives considered

| Option | Why rejected |
| --- | --- |
| Keep 24 and raise it again | Moves the failure point without removing registration-order sensitivity or permanent exhaustion. |
| Make the count configurable | Exposes an implementation accident to users and still ignores schema size and model context. |
| Keep a sticky token-bounded set with LRU eviction | Requires usage timestamps, eviction ordering, and compaction semantics the runtime does not otherwise have. Atomic replacement provides deterministic bounded context with less state. |
| Load every catalog schema before deciding | Defeats cheap catalog listing for large or remote providers. The start probe stops once the allowance is exceeded. |
| Remove all bounds | Large schema payloads consume context, increase latency/cost, and reduce tool-selection accuracy. |

## Consequences

Catalogs with many compact schemas may disclose more than 24 tools directly;
catalogs with a few very large schemas may use discovery. Tool reachability no
longer depends on having unused lifetime slots.

A model that replaces its working set must reload an earlier tool before using
it again. Search and load therefore remain model-visible operations, but only
for catalogs whose actual schema cost warrants them.

`RunBudget.max_active_tools` and `DIRECT_DISCLOSE_THRESHOLD` are removed.
Historical plans and reviews keep their original values as provenance;
normative runtime, task, user, and architecture documentation describe the new
contract.

The 10% allowance and eight-result search default are code-owned policy
constants. They are not new Settings controls in this task. Any later user or
administrator configurability requires evidence, bounds, and a separate task.

## Verification

- Unit tests prove count-independent direct/deferred disclosure using injected
  schema token costs.
- Runtime tests prove a second load replaces the first working set and never
  returns `no room`.
- Service tests prove native and fence schema estimates use the provider-visible
  representation and permission names stay in lockstep with replacement.
- A production-shaped integration test registers MCP last and proves
  find → load → ask/approve → execute.
- Existing agent-runtime, Console bridge, MCP permission, and tool-catalog
  shards remain green.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-08-30-token-budgeted-tool-discovery-design.md)
