# ADR-032: Local Agent Tool Permission Boundary

Status: Accepted
Date: 2026-08-05
Related Tasks:

- [TASK-2819 - Local agent tools phase 1: plumbing + fs_list pilot](../tasks/task-2819%20-%20Local-agent-tools-phase-1-plumbing-fs_list-pilot.md)
- [TASK-16222 - Expose local Watchlists search tools to Console and MCP](../tasks/task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md)
- [TASK-22859 - Define Watchlists Console tool exposure and approval effects](../tasks/task-22859%20-%20Define-Watchlists-Console-tool-exposure-and-approval-effects.md)
- [TASK-22860 - Migrate durable briefing provenance and atomic Watchlists claims](../tasks/task-22860%20-%20Migrate-durable-briefing-provenance-and-atomic-Watchlists-claims.md)
- [TASK-22861 - Expose bounded Watchlists receipts and briefing query tools](../tasks/task-22861%20-%20Expose-bounded-Watchlists-receipts-and-briefing-query-tools.md)
- [TASK-22862 - Add transactional Console Watchlists authoring commands](../tasks/task-22862%20-%20Add-transactional-Console-Watchlists-authoring-commands.md)
- [TASK-22863 - Coordinate durable Watchlists check and briefing operations](../tasks/task-22863%20-%20Coordinate-durable-Watchlists-check-and-briefing-operations.md)
- [TASK-22864 - Make every-24-hours briefing schedules immediately observable](../tasks/task-22864%20-%20Make-every-24-hours-briefing-schedules-immediately-observable.md)
Supersedes: N/A

## Decision

Workspace-local file, web, and todo tools join the Console agent runtime as a
first-class `ToolProvider` at the `Agents/tool_catalog.py` seam. Local tools
carry `local:<name>` catalog ids and `fs_`/`web_`/`todo_` tool names, following
the ADR-030 `library_*` naming precedent.

Local tools reuse the MCP permission store under the synthetic server key
`local:__local__` — distinct from the existing `builtin:tldw_chatbook`
no-transport precedent — with no store schema change, since server keys are
opaque. Tool-override → server-default → global-default precedence, session
approvals, the kill switch, and rug-pull `definition_hash` checking all apply
unchanged. Write tools carry `mutates` risk tags so the approval card presents
them correctly.

The `__local__` profile segment is reserved exclusively for that synthetic
workspace-tool principal. User-controlled external MCP profiles must never be
saved, imported, loaded, or projected as `__local__` / `local:__local__`.
Pre-existing reserved profile, discovery-snapshot, and runtime-state records
are ignored rather than renamed, so external tool metadata cannot alias a
workspace tool's catalog or permission key. Save-time rejection is backed by
load-time filtering and a catalog-projection guard because persisted JSON and
raw projection records can bypass the normal profile editor.

Approval is fail-closed under the three-mechanism discipline: the approval
callback is cleared first on any refusal path, `invoke()` returns a refusal
without stamping approval when no callback is wired, and `stamp_scope` wraps
sub-agent runs so nested invocations re-check permissions. Refusal strings are
pinned constants from spec §3.3: `LOCAL_DENY_REFUSAL`, `LOCAL_TIMEOUT_REFUSAL`,
and `LOCAL_KILL_SWITCH_REFUSAL`.

**Addendum (TASK-13216, 2026-08-11): session tasks use item-oriented CAS.**
The Console-local `todo_write` full-list replacement is retired. A supplied
Console session store registers `todo_create`, `todo_update`, `todo_get`, and
`todo_list`; create/update remain permission-gated mutations, get/list are
read-only, and no task tool is registered without Console session state. Stable
session-local IDs, exact expected-version checks, and atomic mutation preserve
concurrent parent/fleet changes. State remains process-memory-only; the Console
screen snapshot carries pure task records and the next-ID high-water mark solely
across in-process navigation. Public task-ID numeric values and versions remain
in the portable JSON exact-integer domain `1..2**53-1`; attempting an ID or
version increment beyond that domain fails atomically with a fixed bounded
exhaustion error.

**Addendum (TASK-13216 quality review, 2026-08-11): the synthetic local
principal is reserved.** The external-profile identifier was previously
validated only for delimiters and whitespace. As a result, a profile named
`__local__` survived persistence and its discovered tools projected to the
same `local:__local__::<tool>` identity used by the Console's real workspace
tools. The boundary now rejects that exact reserved profile before
persistence, filters a hand-written reserved profile and its associated
catalog state during load, and drops any raw reserved record during Hub
projection. Other currently valid profile IDs, including case variants, keep
their existing semantics; the reserved token is exact and is never silently
rewritten.

**Addendum (TASK-14807, 2026-08-10): catalog availability defaults on.** The
Console registers the standard local provider (workspace file, read-only Git,
and standard web tools) for fresh profiles and profiles where
`[console] local_tools_enabled` is absent. Registration only makes tool schemas
available to the model; it does not authorize a call. The permission store's
fresh and missing-state default remains `ask`, mutating tools retain their risk
floor, the global kill switch remains authoritative, and every path-taking tool
remains confined to `[console] workspace_root`. An explicit
`local_tools_enabled = false` remains a supported opt-out and removes the
provider on the next Console agent run. This separates discoverability from
authorization: users can see and select the tools by default without granting
silent filesystem access, writes, or network egress.

**Addendum (TASK-1354 closeout, 2026-08-10): both web tools are permission-gated;
`web_fetch` alone enforces public-target egress.** `web_search` and `web_fetch`
are ordinary `local:<name>` tools, not privileged built-ins. Fresh and missing
permission state therefore remains `ask` for both tools; catalog availability
never implies network authorization. For each `web_search` invocation, the
caller/model selects one allowlisted `search_engine`, and that selection
determines the destination. The operator supplies supported per-engine
credentials and configurable endpoints where available; fixed-endpoint engines
remain implementation-defined. A configured Searx endpoint may be local.
`web_search` does not apply public-target validation. `web_fetch` accepts only
HTTP(S) targets whose complete DNS answer is public and repeats that check for
every redirect hop. Private, loopback,
link-local, reserved, multicast, unspecified, cloud-metadata, and unresolvable
fetch targets fail before transport. There is no per-domain approval bypass in
the local-tool contract.
This intentionally supersedes TASK-1354's earlier draft proposal for a
default-Allow search tool and configurable localhost/LAN fetching. Optional
external exposure is governed separately by ADR-053: `[mcp]
expose_local_tools` must be enabled, and an external client cannot satisfy an
`ask` verdict, so it fails closed until the operator records a persistent
tool-level Allow through the Console.

**Addendum (PR-T3 review, Fix Round H, 2026-08-06):** a fourth pinned
constant, `LOCAL_GATE_ERROR_REFUSAL`, distinguishes a permission-resolver
CRASH from a genuine configured deny. `_verdict_for()`'s `resolve_state`
`except` branch used to collapse into the SAME "deny" verdict as an actual
Off, so `invoke()` rendered `LOCAL_DENY_REFUSAL`'s "set to Off" claim to the
calling model even when the tool's real state was never determined. Fails
closed identically (the tool still does not run); only the stated reason
changes, and the new string asserts no configuration state and no
"permanently unavailable" implication (derived from the same
`local_runtime_delegate.PERMISSION_STATE_UNRESOLVED_CLAUSE` the MCP Hub's
Test Tool panel and Advanced runner already share for the identical
condition).

**Addendum (TASK-16222, 2026-08-14): private local-domain reads share the
synthetic local principal.** Read-only `watchlists_*` tools may expose local
feed/source metadata and bounded article evidence through the same
`LocalToolProvider`, `local:__local__` permission principal, kill switch,
definition-hash guard, and optional external MCP exposure established above.
Their names follow ADR-030's domain-prefix precedent. Registration is not
authorization: fresh/missing permission remains `ask`; external MCP cannot
satisfy `ask` and requires an operator-recorded tool-level Allow.

This is a privacy-boundary expansion, not merely another workspace helper.
Watchlists rows can contain private monitoring targets, source names, URLs,
queries, and complete third-party article text. The accepted trade-off is to
reuse one synthetic principal and audit trail instead of adding a second local
permission store/provider. To keep that trade-off visible, UI, config-template,
and user-guide copy must no longer describe the master catalog switch or tool
group only as “Local workspace + web tools”; it must explicitly include local
Watchlists evidence. Each Watchlists tool remains separately configurable in
the permission store, so the shared master switch does not collapse per-tool
consent.

Expected Watchlists domain outcomes (invalid input, disambiguation, not found,
local-only unsupported mode, feature unavailable) travel as bounded structured
tool content rather than new gateway error pass-throughs. Permission failures
retain the pinned fail-closed errors above. Article fields are labeled and
delimited as untrusted evidence, but that label is not claimed to guarantee
model obedience. Output is field-allowlisted; URL userinfo, complete queries,
and fragments are removed without guessing which key names are sensitive; only
absolute HTTP(S) links with a host are emitted; unexpected exception messages
are scrubbed before reaching the model. Preserved URL paths remain disclosed
Watchlists metadata under the same explicit tool permission and are not claimed
to be credential-free.

The in-app Console reuses the application's already-initialized
`SubscriptionsDB`. Standalone external MCP instead opens an existing database
through the registered private-SQLite read-only URI path, with schema
initialization and migration disabled. A read-tagged external call must never
create the subscriptions database file, write its schema or rows, or run
migrations as an incidental side effect; normal profile-path resolution may
still ensure the private parent directory. A missing or old schema returns
feature unavailable until the normal application owns initialization. No
Watchlists domain mutation, server-data access, semantic index, or threat score
is authorized by this addendum.

**Addendum (TASK-22859 through TASK-22864, 2026-08-27): descriptor exposure,
approval effects, and briefing privacy are explicit and fail closed.** Every
`LocalToolSpec` declares one exposure value with no permissive default:
`console_and_external_mcp` or `console_only`. Provider construction rejects a
missing or unknown value. External MCP derives its published inventory from
that descriptor field; it does not use a parallel tool-name allowlist or
denylist, and a persisted permission-store Allow cannot make a Console-only
descriptor externally visible or callable.

Approval presentation is derived from code-owned descriptor effects, separate
from model-supplied arguments and separate from permission-enforced risk tags.
The defined effects are `private_read`, `mutates_local`, `network`, and
`llm_spend`. The existing `mutates` risk tag and permission resolver remain the
authorization controls; effect labels explain an approved call but never grant
it. A read-only project binding or a provider composed with
`allow_write=False` omits every descriptor carrying `mutates_local`, including
Watchlists mutations rather than only filesystem writes.

The shared Console/external-MCP Watchlists reads are
`watchlists_list_sources`, `watchlists_list_collections`,
`watchlists_list_briefings`, `watchlists_get_operations_status`, and
`watchlists_get_operation_status`. They return only bounded, field-allowlisted
metadata and durable operation or briefing receipts. This addendum narrows
TASK-16222's optional external publication: `watchlists_search_items` and
`watchlists_get_item` remain available in Console but are `console_only`, so
external MCP receives neither article snippets nor article bodies. Full
generated briefing Markdown and its ordered selected or cited provenance are
also private Console context and are available only through the Console-only
`watchlists_get_briefing` descriptor.

The Console-only mutation/network inventory is
`watchlists_create_sources`, `watchlists_create_collection`,
`watchlists_update_collection_sources`, `watchlists_check_sources`,
`watchlists_generate_briefing`, and `watchlists_set_briefing_schedule`.
These are domain operations rather than SQL or widget controls. Long-running
checks and briefing generation commit or resolve an exact durable receipt
before returning and then execute under an application-owned coordinator;
model timeout, Console navigation, and screen reconstruction do not own or
silently cancel accepted work. Receipt queries are the cross-surface status
contract.

Short database-local authoring mutations use the complementary ownership
boundary: they execute synchronously on the existing Console tool worker
against the application's shared database/domain owners, and return only after
the transaction commits or rolls back. They are not submitted to the app loop
behind a bounded wait. A timed-out `asyncio.to_thread()` mutation cannot be
cancelled and could otherwise commit after a failure response, making a retry
create an unintended duplicate (notably under `auto_suffix`). The app loop is
reserved here for durable, accepted long-running work with an observable
receipt; short authoring responses are definitive outcomes.

Completed briefings preserve immutable, ordered evidence snapshots rather than
depending on mutable live source/item joins. The focused Subscriptions schema
migration records selected and cited order, sanitized source/item identity,
dates, URLs, featured state, and a provenance format marker before a briefing
is published complete. Nullable live links may supplement that snapshot but
cannot erase it when a source or item changes or is deleted. Legacy junction
rows migrate as `legacy_best_effort` and do not invent order. Partial unique
indexes and owner-level accept/transition methods enforce one active source
check per source and one generating briefing per collection across threads and
processes; uniqueness losers resolve to the winning durable receipt.

External MCP remains a read-only projection over an already initialized local
database. It cannot receive article snippets/bodies, generated briefing bodies,
or ordered provenance; cannot invoke Watchlists commands; cannot cause schema
migration; and cannot gain those capabilities through an Allow decision.
Server Watchlists mode does not fall back to the local database for these
tools. Missing or old read-only storage returns a structured unavailable
outcome so normal application startup remains the sole migration owner.

All path-taking tools confine to a configurable `[console] workspace_root`
(default: the app cwd at startup), coerced and templated following the
`collapse_large_pastes` precedent in `config.py`. Hidden path components are
allowed under the root via a new `allow_hidden` parameter on `validate_path`;
traversal outside the root is always rejected.

## Context

The Console agent runtime currently offers only calculator/datetime plus MCP
and skill tools. The spec
`Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md` adds
workspace-local file/web/todo tools, which gives model-initiated calls local
filesystem read/write and network access for the first time outside the MCP
boundary. That raises the stakes on approval gating: a misconfigured or
bypassable approval path would let a cloud model read or modify arbitrary local
files without user consent.

The MCP permission store already implements the approval semantics these tools
need — per-tool overrides, session grants, persistent "always allow" with
rug-pull hashing, and a global kill switch — but it is keyed by MCP server id.
Introducing a parallel store would duplicate the audit trail and force a second
approval UX. Registering local tools under one synthetic server key reuses all
of it while keeping local tools clearly distinguishable from any real MCP
server.

Path confinement needs a canonical root the user can inspect and change. The
existing `validate_path` helper rejects hidden components outright, which is
too strict for real workspaces (`.git`, dotfile configs), so the boundary is
root confinement with hidden components explicitly allowed under it.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Self-hosted MCP server consumed via the in-process delegate | JSON-RPC plumbing for local file reads, and the runtime would depend on the MCP lifecycle for basic capabilities. |
| Separate local permission store | Duplicates the audit trail and approval UX, and two stores would drift in precedence, session-grant, and kill-switch behavior. |
| Config-flag-only gating | No interactive approval and no per-tool persistence; the weakest safety story for exactly the tools that need it most. |
| Keep catalog registration off by default | Reproduces the discoverability failure where Console agents and the Tools catalog report that capabilities do not exist, despite a separate fail-closed permission layer already governing every invocation. |
| Default every registered tool to Allow | Would turn schema availability into implicit filesystem/network authorization and bypass the consent boundary this ADR establishes. |

## Consequences

### Benefits

- Adding a local tool never touches the runtime loop; registration goes through
  the existing `ToolProvider` seam.
- "Always allow" persists with a `definition_hash` exactly like MCP tools, so
  rug-pull protection covers local tools with no new code.
- The MCP workbench's permission UI can display local tools later without
  migration, since they already live in the same store.
- One approval and audit trail covers MCP and local tools uniformly.

### Accepted trade-offs

- The permission store gains a synthetic server key that is not a real MCP
  server; UI that enumerates servers must special-case or filter it.
- External MCP profiles cannot use the exact id `__local__`; a hand-written
  persisted record with that id and its associated catalog/runtime state are
  deliberately inert rather than migrated or renamed.
- Tool schemas are visible to Console models by default, so prompts may propose
  calls before the user has configured a workspace root; the first call still
  requires the resolved permission verdict and remains confined to the current
  default root.
- `validate_path` grows an `allow_hidden` parameter whose default must remain
  the current strict behavior for existing callers.
- `workspace_root` defaults to the app cwd, so the confinement boundary moves
  with where the app is launched until the user configures it.

## Links

- [TASK-2819](../tasks/task-2819%20-%20Local-agent-tools-phase-1-plumbing-fs_list-pilot.md)
- [TASK-1354](../tasks/task-1354%20-%20Complete-web_search-and-web_fetch-Console-and-MCP-exposure.md)
- [TASK-16222](../tasks/task-16222%20-%20Expose-local-Watchlists-search-tools-to-Console-and-MCP.md)
- [TASK-22859](../tasks/task-22859%20-%20Define-Watchlists-Console-tool-exposure-and-approval-effects.md)
- [TASK-22860](../tasks/task-22860%20-%20Migrate-durable-briefing-provenance-and-atomic-Watchlists-claims.md)
- [TASK-22861](../tasks/task-22861%20-%20Expose-bounded-Watchlists-receipts-and-briefing-query-tools.md)
- [TASK-22862](../tasks/task-22862%20-%20Add-transactional-Console-Watchlists-authoring-commands.md)
- [TASK-22863](../tasks/task-22863%20-%20Coordinate-durable-Watchlists-check-and-briefing-operations.md)
- [TASK-22864](../tasks/task-22864%20-%20Make-every-24-hours-briefing-schedules-immediately-observable.md)
- [Design specification](../../Docs/superpowers/specs/2026-08-04-local-agent-tools-design.md)
- [Watchlists agent search tools design](../../Docs/superpowers/specs/2026-08-14-watchlists-agent-search-tools-design.md)
- [Console-driven Watchlists workflow remediation design](../../Docs/superpowers/specs/2026-08-26-console-driven-watchlists-workflow-uat-remediation-design.md)
- [Implementation plan](../../Docs/superpowers/plans/2026-08-04-local-agent-tools-phase1.md)
- [Watchlists agent boundary and provenance plan](../../Docs/superpowers/plans/2026-08-27-watchlists-agent-boundary-and-provenance.md)
- [Console Watchlists commands and operations plan](../../Docs/superpowers/plans/2026-08-27-console-watchlists-commands-and-operations.md)
- [ADR-030: Direct Local Library Tool Boundary for Console and MCP](030-local-library-agent-tool-boundary.md)
