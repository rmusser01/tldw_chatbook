# ADR-104: Use token-budgeted replaceable tool working sets

Status: Proposed
Date: 2026-08-30
Related Task: [TASK-15261](../tasks/task-15261%20-%20MCP-tool-reachability-is-unpinned-under-the-shipped-default-catalog.md)
Supersedes: The fixed-count active-set clause in the progressive-tool-disclosure section of the [agent runtime vertical-slice design](../../Docs/superpowers/specs/2026-07-12-agent-runtime-vertical-slice-design.md)

## Decision

Agent catalog disclosure is governed by estimated model-visible schema tokens,
not tool counts.

The runtime uses 10% of the selected model's context window as the schema-share
threshold for automatically disclosing the complete allowed catalog. Direct
disclosure additionally requires the projected complete first request to fit
after the response reserve. Core runtime tools such as `find_tools`,
`load_tools`, and `spawn_subagent` retain their existing independent gates and
sit outside the schema-share threshold, but they are included in whole-request
fit checks.

At run start, the registry probes allowed catalog schemas in stable order. If
the complete allowed catalog fits the 10% threshold and its projected first
request fits, every schema is disclosed directly and discovery tools are
omitted. The schema-share probe may stop as soon as the threshold is exceeded.
An invalid model limit, failed schema load, failed token estimate, or
over-budget first request selects deferred discovery rather than guessing that
the catalog fits. Costing measures the complete provider-visible schema set
rather than summing individually measured schemas, so protocol wrappers, list
delimiters, and tokenizer boundaries are included.

For deferred catalogs, `find_tools` searches every allowed catalog entry. It
ranks exact-name, name-prefix, name-substring, and description matches in that
order, with deterministic name/id tie-breaking, and returns at most eight
entries. `load_tools` resolves ids and bare names, deduplicates by model-facing
name, and admits every requested schema whose projected next provider request
fits the model context after the configured response reserve. The projection
uses the live request builder and includes messages, system content, runtime
schemas, the candidate catalog set, and the deterministic load result. One
oversized schema cannot prevent a later smaller requested schema from being
considered.

The 10% value is not a hard ceiling on deferred tools. A schema larger than 10%
can be loaded when its complete next request fits; a long conversation may have
less than 10% available. This keeps every provider-valid schema eligible for
loading while rejecting a candidate whenever the existing estimator cannot
prove its next request fits.

Selection returns a structured, side-effect-free result distinguishing
accepted schemas, budget omissions, and invalid inputs. Each successful
`load_tools` call then replaces the catalog working set and the permission
gate's disclosed-name set as one runtime commit, atomic with respect to later
model turns and tool dispatch. It does not append to a lifetime set. A later
load can always replace earlier tools; the runtime has no permanent `no room`
state. Previously replaced tools can be found and loaded again when needed.

`load_tools` must be the only call in its model-produced batch. A mixed or
repeated load batch refuses the load and preserves the old set, while ordinary
calls continue against the pre-batch set. This makes authorization independent
of provider call ordering.

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
| Load every catalog schema before deciding | Defeats cheap catalog listing for large or remote providers. The start probe stops once the 10% threshold is exceeded. |
| Remove all bounds | Large schema payloads consume context, increase latency/cost, and reduce tool-selection accuracy. |

## Consequences

Catalogs with many compact schemas may disclose more than 24 tools directly;
catalogs with a few very large schemas may use discovery. Tool reachability no
longer depends on having unused lifetime slots, and a schema larger than the
automatic-disclosure threshold is not permanently excluded.

A model that replaces its working set must reload an earlier tool before using
it again. Search and load therefore remain model-visible operations, but only
for catalogs whose actual schema cost warrants them.

`RunBudget.max_active_tools` and `DIRECT_DISCLOSE_THRESHOLD` are removed.
Historical plans and reviews keep their original values as provenance;
normative runtime, task, user, and architecture documentation describe the new
contract.

The 10% automatic-disclosure threshold and eight-result search default are
code-owned policy constants. They are not new Settings controls in this task.
Any later user or administrator configurability requires evidence, bounds, and
a separate task.

## Verification

- Unit tests prove count-independent direct/deferred disclosure using injected
  schema token costs and a history-heavy first-request case.
- Runtime tests prove a second load replaces the first working set, mixed
  batches do not mutate it, and no path returns `no room`.
- Service tests prove native and fence schema estimates use the complete
  provider-visible set representation, large singleton schemas remain
  reachable when the next request fits, and permission names stay in lockstep
  with replacement.
- A production-shaped integration test registers MCP last and proves
  find → load → ask/approve → execute.
- Existing agent-runtime, Console bridge, MCP permission, and tool-catalog
  shards remain green.

## Links

- [Design specification](../../Docs/superpowers/specs/2026-08-30-token-budgeted-tool-discovery-design.md)
