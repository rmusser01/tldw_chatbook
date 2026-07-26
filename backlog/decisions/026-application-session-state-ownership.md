# ADR-026: Application Session State Ownership

Status: Accepted
Date: 2026-07-26
Related Tasks:
[TASK-643](../tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md),
[TASK-644](../tasks/task-644%20-%20Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md),
[TASK-645](../tasks/task-645%20-%20Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md),
[TASK-646](../tasks/task-646%20-%20Complete-destination-handoff-ownership-and-ACP-target-recovery.md)
Supersedes: the application-state architecture proposed by the historical
documents listed under Links

## Decision

Chatbook will not introduce another root application-state object. State will
remain with the narrow owner whose lifecycle and consistency rules it follows:

- `RuntimePolicyContext` is the sole authority for the active runtime source,
  active server binding, and server capability status.
- A process-memory `ScreenStateStore` owns only cross-visit screen snapshots.
- A process-memory `PendingHandoffStore` owns only destination handoffs.
- Destination screens continue to own their domain and view state, consistent
  with ADR-011.

`TldwCli` will coordinate these owners but will not mirror their values into a
root `AppState`. The exported `AppState`, `ChatState`, `NotesState`,
`NavigationState`, and `UIState` types remain importable compatibility
containers. Their documentation must not call them the application's single
source of truth, and `TldwCli` will not depend on them.

Runtime-policy changes use a monotonic revision. A caller reads a state and
revision snapshot, derives a candidate, and commits only against that revision.
The candidate is persisted before it is published in memory or projected onto
compatibility attributes. A stale asynchronous server-capability result is
discarded instead of overwriting a newer source or server selection.
Application compatibility projections are updated by a non-throwing
publication callback owned by the context boundary; they have no independent
production writers. On `TldwCli`, the three legacy attributes are getter-only
properties backed by one private immutable `(source, active_server_id)`
projection tuple. The callback's sole app publisher replaces that tuple
atomically; direct writes through the public attributes fail.

One `RuntimePolicyContext` identity is installed for the `TldwCli` lifetime.
Configuration changes rebind that context and its server-context provider
rather than replacing the authority object retained by long-lived consumers.
Initial load, synchronization persistence, and direct initial projection
prepare the context before it is attached; failure leaves no half-installed
authority and permits retry.
Context state is read-only; there is no public state setter or standalone
persistence escape hatch, and both the backing store and projection callback
are private. The callback does not read or write `AppState`.

Settings saves the changed URL and token in one checked configuration batch,
loads that configuration as an unpublished candidate, and passes it to one
app-level coordinator. The coordinator derives and durably commits the new
runtime binding through the existing context before changing
`app.app_config`, provider configuration, configured targets/defaults, client
cache, or the active screen. A failed commit leaves all of those observers on
the old binding. After commit, provider configuration and cache invalidation
are installed before screen notification. Legacy-target materialization is
best-effort and non-authoritative: if it fails, the provider's refreshed
configuration remains the fallback for the already-committed binding.
The active-screen callback is a contained post-commit observer; its failure
cannot be represented as if the committed binding rolled back.
Because the Settings TOML and runtime-policy JSON are separate durable files,
a successful Settings save is retained for retry/startup when the subsequent
runtime-policy commit fails; no cross-file rollback is claimed.

Runtime-policy persistence is part of the ADR-022 private-data boundary. Its
default path is resolved from the effective config path when the context is
constructed, including `TLDW_CONFIG_PATH` overrides. Existing files are
descriptor-verified before parsing, and writes use the random-name,
descriptor-verified private atomic writer. Unsafe paths fail closed. The
application neither falls back to nor migrates from the ordinary default
runtime-policy path while a config override is active.

Screen snapshots remain memory-only. The store records a canonical route,
detached outer snapshot mapping, and private runtime identity in an internal
envelope. Runtime metadata is not inserted into a screen's domain dictionary.
A source or active-server mismatch invalidates the snapshot. Navigation always
constructs a fresh screen and applies any explicit navigation context after a
compatible snapshot is restored.

The snapshot key is exactly the route registry's resolved `canonical_tab`.
The outgoing key comes from the app's current tab established by the prior
successful navigation, and the incoming key comes from the new resolution.
Requested aliases, routed screen names, and screen-owned names are not
alternative keys. Aliases sharing a canonical tab intentionally share a
snapshot. Initial screen push is the sole startup writer of the canonical
current tab; deferred post-mount code must not replace it with the unresolved
configured alias.

Destination handoffs remain memory-only and preserve the existing single-slot,
consume-once, last-write-wins behavior. Each typed channel has a monotonic
revision and supports atomic stage, pending-clear, claim, acknowledge, and
release. A pending clear is itself a newer revision, so an older in-flight
claim cannot be resurrected by release after a producer intentionally removes
an optional value. At most one claim may be in flight for a channel, and at
most the latest replacement may wait behind it; this is not a queue. A stale
acknowledge or release cannot clear a newer value. Successful and terminally
rejected handoffs are acknowledged. Transient failures release the claim for
a later existing lifecycle or user-triggered retry; they do not start an
automatic retry loop. Setup-blocked Console prompt insertion is transient
readiness and releases rather than discarding the intent.

Runtime commits, snapshot-store mutation, and handoff-store mutation are
affine to the application thread captured when each owner is created. Foreign
workers marshal mutations to the app event-loop boundary; a different thread
identity rejects. This does not require an event-loop object to exist during
construction. Revisions are process-local coordination tokens, not
cross-process merge guarantees.

A Chat handoff that creates an ephemeral tab must close that exact tab before
releasing after later failure or cancellation. If cleanup itself fails, the
claim is terminally acknowledged with bounded recovery so retries cannot create
duplicate partial tabs. Artifact handoffs use exact canonical
`local:chatbook:<id>` lookup and never substitute a latest record for a missing
target. Artifact workers are guarded by an app-thread refresh generation and
exact active claim: unmount, refresh restart, or a matching terminal
worker-state event releases the active claim, and a stale callback cannot
apply UI or settle a newer lifecycle's claim.

The ACP session-target handoff will be completed without inventing an
arbitrary-session repository. ACP compares the requested target with the
current runtime session by reconstructing the same canonical
`local:acp_session:<session_id>` record ID used by the producer. A match
keeps the current session row selected and scrolls its existing detail pane
into view; a malformed, missing, or mismatched target produces explicit
stale/unsupported recovery and is terminally acknowledged.

## Context

`tldw_chatbook/app.py` is a large coordination surface with many reactive and
ordinary instance attributes. Earlier refactoring documents proposed a
reactive root dataclass or dictionary, mirrored writes, disk persistence, and
cached screen instances. Those approaches conflict with the current
destination-ownership rule and with the application's fresh-screen navigation
behavior.

The live `AppState` integration consists of an import and an otherwise unused
construction in `TldwCli`. Its component models are not the authority for
active screen, chat, Notes, UI, or runtime policy, while its class docstring
claims it is the single source of truth. Keeping that object in the app would
preserve a false architectural contract.

Runtime policy already supplies the effective source, but mutation is not
atomic: the current setter publishes a new state before persistence succeeds.
Capability discovery captures state, awaits network work, then publishes the
derived result unconditionally. A source or server change during that await
can therefore be overwritten by stale capability data. Media Ingest and Study
also write app-level runtime projections directly. The current runtime-policy
store uses ordinary path I/O, predictable temporary names, ambient modes, and
an import-time default path that ignores the effective config override.

Navigation currently keeps a raw `_screen_states` dictionary on `TldwCli` and
adds `runtime_policy_snapshot` to domain-owned dictionaries. Home, Workflows,
and Schedules read the private dictionary to infer recent work.

Seven raw pending fields coordinate Chat, Console, Study, Artifacts, and ACP.
Several consumers await work before clearing the field, allowing an older
consumer to clear a newer replacement. The ACP target is staged but has no
consumer. Chat can retain a failed pending handoff after creating a partial
ephemeral tab, and Artifacts can replace an absent requested target with the
latest record from a limited first page. `pending_notes_workspace_context` is
initialized but never read or written.

These are application-lifetime, runtime-boundary, privacy, and cross-module
interface decisions, so they require one canonical ADR before implementation.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Make `AppState` or a new `AppSessionState` the root authority | It would duplicate active domain owners, require mirrored writes, and create another broad mutable object rather than remove ambiguity. |
| Store complex state in reactive root dictionaries | In-place mutation and notification semantics are easy to misuse, type contracts weaken, and every feature becomes coupled to the root app. |
| Replace handoffs entirely with Textual messages | Messages do not by themselves preserve a value across fresh screen construction, mount timing, or transient destination failure. |
| Persist screen snapshots or handoffs to disk | These values can contain private prompt, context, target, and UI data; restart recovery is not required for this tranche. |
| Cache and remount screen instances | It conflicts with established navigation behavior and risks retaining workers, timers, widget trees, and stale domain services. |
| Queue every handoff | Existing behavior is single-slot and last-write-wins; a queue would be a product behavior change and could replay stale intent. |
| Clear a handoff before doing destination work | It prevents an older consumer from deleting a replacement but loses transient failures and existing retry behavior. |
| Add an ACP arbitrary-session lookup repository | No such runtime authority exists. Fabricating one would broaden the ACP storage and lifecycle contract beyond this repair. |
| Rewrite all application state in one change | The surface is too broad to verify atomically. Runtime policy, snapshots, core handoffs, and remaining handoffs need dependency-ordered tasks. |

## Consequences

### Benefits

- Each mutable value has one truthful owner and an explicit lifecycle.
- Persistence failure cannot publish runtime state that was not durably saved.
- Slow capability probes cannot revert a newer runtime choice.
- Snapshot compatibility metadata is separated from domain-owned screen state.
- Concurrent handoff staging cannot be erased by a stale consumer.
- Navigation context keeps its current higher precedence over restored view
  state.
- Private handoff and snapshot payloads stay out of disk persistence and
  diagnostics.
- Runtime-policy metadata is read and written through the private-file
  boundary at the effective configuration location.
- Failed Chat delivery cannot accumulate duplicate partial handoff tabs, and
  Artifact delivery cannot silently select a different record.
- Static ownership guards can protect small, precise boundaries instead of
  relying on broad repository-wide string searches.

### Accepted Trade-offs

- Screen continuity and pending handoffs are lost when the process exits.
- The compatibility state classes remain importable even though the
  application does not use them; removal, if ever desired, requires a separate
  compatibility decision.
- Runtime-policy tests use revisioned commits or fakes rather than a direct
  mutable-state compatibility seam.
- A transiently failed handoff waits for an existing lifecycle or user action;
  there is no background retry scheduler.
- Revisions coordinate one application process; concurrent processes may
  detect a changed private target and fail, but they do not merge state.
- ACP can recover only the current runtime session until a separately designed
  session repository exists.
- Service construction and shutdown remain outside this tranche, including the
  duplicated writing and chat-conversation wiring calls in `TldwCli.__init__`.

## Links

- [Application session state ownership design](../../Docs/superpowers/specs/2026-07-26-application-session-state-ownership-design.md)
- [ADR-011: Chatbook Workbench UI System](011-chatbook-workbench-ui-system.md)
- [ADR-022: Local Private Data Boundary](022-local-private-data-boundary.md)
- [ADR-024: Bounded Evaluation and Tool Worker Execution](024-bounded-evaluation-and-tool-worker-execution.md)
- [ADR-025: Immutable Installed Distribution Assets](025-immutable-installed-distribution-assets.md)
- [Historical state decomposition analysis](../../Docs/Development/state-decomposition-analysis.md)
- [Historical app refactoring plan](../../Docs/Development/app-refactoring-plan.md)
- [Historical app refactoring plan v2](../../Docs/Development/app-refactoring-plan-v2.md)
- [Historical migration guide](../../Docs/Development/app-refactoring-migration.md)
- [Historical review](../../Docs/Development/refactoring-issues-review.md)
- [Historical review v2](../../Docs/Development/refactoring-issues-review-v2.md)
- [Historical refactoring complete summary](../../Docs/Development/refactoring-complete-summary.md)
- [Historical refactoring fixes summary](../../Docs/Development/refactoring-fixes-summary.md)
