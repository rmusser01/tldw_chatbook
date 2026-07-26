# Application Session State Ownership Design

Date: 2026-07-26
Status: Design approved; TASK-643 corrective plan reviewed and ready, TASK-644 through TASK-646 plans ready
ADR:
[ADR-026](../../../backlog/decisions/026-application-session-state-ownership.md)
Backlog:
[TASK-643](../../../backlog/tasks/task-643%20-%20Make-runtime-policy-the-sole-application-runtime-source-authority.md),
[TASK-644](../../../backlog/tasks/task-644%20-%20Move-cross-visit-screen-snapshots-behind-an-in-memory-owner.md),
[TASK-645](../../../backlog/tasks/task-645%20-%20Move-Chat-and-Console-handoffs-behind-revisioned-single-slot-ownership.md),
[TASK-646](../../../backlog/tasks/task-646%20-%20Complete-destination-handoff-ownership-and-ACP-target-recovery.md)
Plans:
[TASK-643 corrective](../plans/2026-07-26-task-643-structural-ownership-correction.md),
[TASK-643 original (partially superseded)](../plans/2026-07-26-task-643-runtime-policy-authority.md),
[TASK-644](../plans/2026-07-26-task-644-screen-state-store.md),
[TASK-645](../plans/2026-07-26-task-645-chat-console-handoffs.md),
[TASK-646](../plans/2026-07-26-task-646-destination-handoffs.md)

## Summary

This is the first, deliberately narrow application-state decomposition
tranche. It removes a false root-state contract and moves three kinds of
cross-component state behind owners with explicit lifecycle rules:

1. `RuntimePolicyContext` owns runtime-source and server-capability state.
2. `ScreenStateStore` owns cross-visit screen snapshots.
3. `PendingHandoffStore` owns destination handoffs.

The design does not create a new root state object. Domain screens keep owning
their own domain state and services under ADR-011. Cross-visit snapshots and
handoffs remain process-memory only. Existing screen construction, navigation
precedence, single-slot handoff behavior, and compatibility imports are
preserved.

## Verified Baseline and Problems

The findings below were re-checked against the current worktree before this
specification was written.

### Root application state

- `tldw_chatbook/app.py` is 10,613 lines.
- `TldwCli` declares 61 class-level reactive descriptors and assigns roughly
  262 distinct instance attributes across its methods. These measurements
  describe the decomposition pressure; they are not target metrics for this
  tranche.
- The active `AppState` integration is only an import and construction in
  `TldwCli.__init__`. The application does not use its navigation, Chat, Notes,
  UI, or runtime-source members as live authority.
- `AppState` nevertheless claims to be the "single source of truth for all
  application state." That is false and invites future mirrored writes.
- `AppState`, `ChatState`, `NotesState`, `NavigationState`, and `UIState` are
  exported from `tldw_chatbook.state`; compatibility consumers may import and
  serialize them.

### Runtime policy

- `RuntimePolicyContext` exposes a public mutable `state` plus `persist()`.
- `set_authoritative_runtime_source()` assigns `context.state` before calling
  persistence. A storage failure leaves memory and projections capable of
  disagreeing with disk.
- `RuntimeSourceStateStore` reads and writes `runtime_policy.json` with
  ordinary `Path.open()` calls, a predictable `.tmp` sibling, and ambient file
  modes. It does not verify ownership, mode, or symlink safety before reading
  a file that can contain local server identifiers, labels, and status
  metadata.
- `DEFAULT_RUNTIME_POLICY_PATH` is derived from the ordinary default config
  path at import time. It ignores the effective `TLDW_CONFIG_PATH`, so an
  overridden config can silently split config and runtime-policy storage
  across directories.
- `ActiveServerCapabilityService.refresh()` captures state, awaits up to three
  discovery calls, then directly assigns and persists a candidate based on the
  old state. A runtime-source or server switch during the await can be reverted
  by the stale result.
- `MediaIngestScreen` writes both `current_runtime_backend` and
  `runtime_backend` on the app. `StudyScreen` writes
  `current_runtime_backend`. Those compatibility projections therefore have
  independent writers.
- `TldwCli.handle_runtime_backend_changed()` retains a fallback that writes the
  projections when no runtime policy exists, despite normal app startup always
  constructing the policy.

### Screen snapshots

- Navigation stores `save_state()` results in a lazily created app-level
  `_screen_states` dictionary.
- `add_runtime_policy_snapshot()` adds a reserved
  `runtime_policy_snapshot` key to a domain-owned dictionary, and restoration
  removes it again.
- Home, Workflows, and both Schedules surfaces inspect `_screen_states`
  directly to infer recent work.
- Screen routing has three identities that do not always agree: the requested
  route, the resolver's routed `screen_name`, and its `canonical_tab`. For
  example, `ccp` resolves to routed name `ccp` and canonical tab `personas`,
  while the constructed screen reports `screen_name="personas"`. Current
  navigation saves under the screen-owned name and restores under the routed
  name, so aliases can already miss their own snapshots.
- Startup initially writes the resolved canonical tab in
  `_push_initial_screen()`, but `_post_mount_setup()` later overwrites it with
  the raw configured route both directly and through deferred
  `_set_initial_tab()`. An alias-configured startup can therefore leave
  `current_tab` noncanonical before the first outgoing save.
- Navigation flushes outgoing pending work, saves state, constructs a fresh
  screen, restores state, applies explicit navigation context, and switches
  screens. That order carries user-visible semantics.
- Library, Settings, and Watchlists use explicit navigation context that must
  take precedence over restored view state.

### Destination handoffs

The app currently stages these raw fields:

| Channel | Current app field | Current destination behavior |
| --- | --- | --- |
| Chat context | `pending_chat_handoff` | Chat consumes after mount/setup; some failures retain it |
| Console live work | `pending_console_launch` | Chat normalizes it into screen-local launch context |
| Console prompt | `pending_console_prompt_insert` | Chat appends text after the composer is ready |
| Study scope | `pending_study_scope_context` | Study applies then clears it |
| Study section | `pending_study_initial_section` | Study selects then clears it |
| Artifact target | `pending_artifacts_chatbook_target_id` | Artifacts selects if available; current cleanup is eager |
| ACP target | `pending_acp_session_target_id` | Staged by Console action, never consumed |

`pending_notes_workspace_context` is initialized but has no production writer
or reader. It is dead state.

Several consumers read a raw field, await setup or lookup work, and then clear
the field. If another producer stages a replacement during that await, the
older consumer can clear the newer value. Clearing before work would avoid that
race but would lose the established retry behavior on transient failure.

Chat context has an additional partial-application boundary: it creates a new
ephemeral tab before awaited switching and handoff application complete. A
failure or cancellation after tab creation can retain the old pending value
while leaving the partial tab behind, so a later retry can create a duplicate.

Artifacts currently searches only the first 25 local Chatbooks. When an exact
requested target is absent from that page, it silently selects the most recent
record instead. The service already provides exact-ID lookup, so this fallback
can acknowledge the wrong target.

ACP exposes the current runtime process/session state but no repository for
arbitrary historical or concurrent session IDs. The missing ACP consumer must
therefore be repaired against the current runtime session, not by inventing a
lookup source. The current ACP surface already has one selected current-session
row and a `#acp-detail-pane`; successful target recovery therefore needs a
specific focus/visibility behavior rather than a new session selector.

### Adjacent issue intentionally separated

`TldwCli.__init__` calls `_wire_writing_services()` twice and
`_wire_chat_conversation_services()` twice. This is a verified service
lifecycle problem, but combining construction/shutdown ownership with state
ownership would make the first tranche too broad. It remains outside this
state-ownership contract.

## Goals

- Give runtime policy, screen snapshots, and destination handoffs one truthful
  owner each.
- Remove `TldwCli`'s dependency on `AppState` without breaking public imports
  or serialization behavior.
- Make runtime publication persist-before-publish and resistant to stale async
  results.
- Keep runtime-policy persistence inside the ADR-022 private-file boundary and
  colocated with the effective configuration path.
- Preserve fresh-screen construction and current navigation order.
- Keep snapshots and handoffs memory-only.
- Preserve single-slot, consume-once, last-write-wins handoff semantics.
- Prevent a stale consumer from acknowledging or releasing a newer handoff.
- Complete the missing ACP target behavior honestly.
- Prevent failed or cancelled Chat handoffs from leaving duplicate partial
  ephemeral tabs.
- Resolve Artifact handoffs by exact canonical target without substituting the
  latest record.
- Remove the dead Notes handoff field.
- Keep private payloads out of persistent diagnostics.
- Add narrow ownership guards and behavioral regression tests.

## Non-Goals

- Reducing `app.py` to a target line, method, attribute, or reactive count.
- Creating a new `AppState`, `AppSessionState`, Redux-like store, event bus, or
  dependency-injection container.
- Moving domain state or services out of screens.
- Persisting view snapshots or destination handoffs across restart.
- Queueing multiple handoffs or adding automatic retry workers.
- Caching or remounting screen instances.
- Removing the exported legacy state classes.
- Adding arbitrary ACP session storage or lookup.
- Reworking service wiring, shutdown, workers, timers, or screen registry
  architecture.
- Reopening the eval/tool worker contracts in ADR-024 or installed-distribution
  contract in ADR-025.

## Ownership Model

| State | Sole owner | Lifetime | Persistence |
| --- | --- | --- | --- |
| Active source, server binding, capability status | `RuntimePolicyContext` | Application process | Existing runtime-policy store |
| Cross-visit screen view snapshot | `ScreenStateStore` | Application process | None |
| Destination intent awaiting consumption | `PendingHandoffStore` | Application process | None |
| Domain records, editor state, active work | Destination screen/service | Existing domain lifetime | Existing domain policy |
| Legacy state dataclasses | Importing compatibility caller | Caller-defined | Existing `to_dict`/`from_dict` behavior only |

`TldwCli` is a coordinator and composition root. It may expose read-only or
derived compatibility projections where existing screens require them, but it
must not become a second owner.

All three mutable owners capture the creating application thread. Production
mutation occurs on that thread through the app's event loop. A worker or
foreign thread must marshal mutation through `app.call_from_thread()` or the
existing app-loop boundary; a mutation whose thread identity differs from the
captured owner rejects instead of racing. The contract deliberately uses thread
identity rather than an event-loop object because owners may be constructed
before Textual starts its loop. Revisions are process-local coordination tokens
and do not claim cross-process merge or rebase semantics.

## Runtime Policy Contract

### Context interface

`RuntimePolicyContext` owns a private monotonic revision in addition to its
state and store. Its authoritative operations are equivalent to:

```python
def snapshot(self) -> tuple[RuntimeSourceState, int]: ...

def commit_state(
    self,
    candidate: RuntimeSourceState,
    *,
    expected_revision: int,
) -> bool: ...
```

`snapshot()` returns the current immutable state value and revision.
`commit_state()` returns `False` without persistence or publication when the
revision is stale. When the revision matches it:

1. persists `candidate`;
2. only after persistence succeeds, publishes `candidate`;
3. increments the revision;
4. updates app compatibility projections through the existing projection
   callback/boundary;
5. returns `True`.

Storage exceptions propagate. They leave the prior in-memory state, revision,
and app projections unchanged. Runtime state values remain immutable
dataclasses, so a snapshot cannot be mutated in place.

`state` is a read-only property. The context exposes no public setter or
standalone `persist()` escape hatch, and its backing store is private; focused
tests retain their injected stores separately or use revisioned fakes. A
scoped AST guard enforces that production mutation and storage writes go
through `commit_state()` or a higher-level runtime-policy operation.

`snapshot()` may be read from ordinary production callers because it returns
an immutable value. `commit_state()` and all higher-level mutation methods are
owner-thread-affine and reject off-owner calls. The revision protects
in-process asynchronous work; it is not a cross-process compare-and-swap
protocol.

The context accepts an optional non-throwing publication callback when
constructed by `load_runtime_policy_for_app()`. After durable state publication
the callback updates the app's compatibility projections. A projection
failure is contained and logged without private values; it cannot roll back a
durable authoritative commit. Production readers use the context rather than
depending on those projections for authority.

### Persistence boundary

Unless an explicit runtime-policy path is injected,
`load_runtime_policy_for_app()` resolves the path when the context is
constructed as `get_cli_config_path().parent / "runtime_policy.json"`. An
active `TLDW_CONFIG_PATH` therefore colocates the runtime policy with the
effective configuration. The application does not migrate, merge, or fall
back to the ordinary default runtime-policy file while an override is active.

ADR-022 private-path rules apply before parsing:

- On POSIX, existing files are opened through the descriptor-verified private
  reader, rejecting symlinks, wrong ownership, or unsafe file type and
  hardening eligible excessive permissions before any JSON is consumed.
- Writes use the random-name, descriptor-verified private atomic writer rather
  than a predictable sibling temp file.
- The default application config directory is app-owned and hardened to mode
  `0700`. A caller-supplied/custom parent must already exist; runtime-policy
  code does not create or chmod that parent.
- Unsafe targets fail closed. Malformed JSON retains the existing safe-default
  recovery only after the file passes privacy verification.
- Diagnostics report only operation and exception category. They do not log a
  custom path, server identifier, label, endpoint, or serialized state.

On Windows, the same privacy primitives return and surface
`UNVERIFIED_PLATFORM` without claiming that native ACLs were verified, matching
ADR-022.

The private atomic writer can detect a target changed by another process and
fail the commit. There is deliberately no cross-process merge/rebase promise.

### Source changes

`set_authoritative_runtime_source()` reads one snapshot, derives the configured
server binding and source from that state, and commits synchronously against
the captured revision. There is no await between snapshot and commit, so
ordinary event-loop callers cannot interleave a capability update.

The method returns the actually committed state. An invalid source keeps the
current state. A request for server mode with no configured server continues
to resolve to local mode. A persistence failure is reported to the caller and
does not publish a false selection.

`TldwCli.handle_runtime_backend_changed()` contains that failure at the
application boundary: it emits a bounded metadata-only warning and keeps the
prior authoritative source, selected screen, revision, and compatibility
projections.

The exact compatibility projections `current_runtime_backend`,
`runtime_backend`, and `active_server_id` are getter-only properties on
`TldwCli`. The runtime-policy projection boundary's private publisher is the
only writer of their one immutable `(source, active_server_id)` backing tuple.
Media Ingest, Study, and the no-policy branch in
`handle_runtime_backend_changed()` stop writing them independently.

### Capability refresh

`ActiveServerCapabilityService.refresh()`:

1. captures `(state, revision)`;
2. probes using the captured server identity;
3. derives a candidate from that captured state;
4. calls `commit_state(candidate, expected_revision=revision)`;
5. publishes target-status side effects only when the authoritative commit
   succeeds.

If the revision is stale, the capability result is discarded. It must not
change runtime policy, projections, or the selected target's status. The
method returns a snapshot derived from the fresh authoritative state, with no
health/readiness/docs payload from the stale probe and a
`capability_result_superseded` reason code. It must not present stale probe data
as current authority.

The no-server-configured path uses the same commit protocol. Capability error
diagnostics retain reason codes but must not include endpoint credentials,
response bodies, or other private values.

## Screen Snapshot Contract

### Store interface

`ScreenStateStore` provides a narrow API:

```python
def save(
    self,
    route: str,
    snapshot: Mapping[str, Any],
    runtime_identity: RuntimeIdentity,
) -> None: ...

def restore(
    self,
    route: str,
    runtime_identity: RuntimeIdentity,
) -> dict[str, Any] | None: ...

def discard(self, route: str) -> None: ...
def has_snapshots(self, runtime_identity: RuntimeIdentity) -> bool: ...
```

The backing mapping and envelope type are private. An envelope contains a
canonical route, an outer copy of the screen snapshot, and runtime identity.
It does not add policy keys to the domain mapping or expose the backing mapping
to consumers.

The store captures the creating application thread. Off-owner `save()`,
`restore()`, `discard()`, and mutation-capable `has_snapshots()` calls reject.
Screens treat the mapping passed to `restore_state()` as read-only and copy any
nested mutable value they retain or mutate.

For this store, "canonical route" means exactly the `canonical_tab` returned
by `resolve_screen_target()`, not the requested route, the resolver's
`screen_name`, or `BaseAppScreen.screen_name`.

- The outgoing save key is the app's existing `current_tab`, which was set from
  the prior successful resolution's `canonical_tab`.
- The incoming restore key is `current_tab_value`, the `canonical_tab` returned
  while resolving the new request.
- If an outgoing screen predates a populated `current_tab`, navigation may
  resolve its screen-owned name and use the returned `canonical_tab` only when
  that route is registered; otherwise it skips snapshot save rather than
  inventing a key.
- Aliases that resolve to the same `canonical_tab` intentionally share one
  snapshot. Direct routes whose registry entries have different canonical tabs
  remain distinct even when their screen classes report the same screen-owned
  name.

The store accepts an already canonical, non-empty key and does not perform a
second route resolution. This keeps registry policy in the navigation owner
and makes the key contract testable.

`_push_initial_screen()` becomes the sole startup writer of `current_tab`. It
already resolves the configured/first-run route and assigns `resolved_tab`
after the screen is pushed. TASK-644 removes the deferred `_set_initial_tab()`
call/method and the final raw assignment in `_post_mount_setup()` so neither
can overwrite that canonical value. The app always enables screen navigation,
and `watch_current_tab()` returns immediately in that mode, so those later
assignments provide no required watcher behavior. Any startup code that needs
the configured route for a one-time branch may keep it in a local variable,
but it may not publish that unresolved value as `current_tab`.

Runtime identity consists of the active source and, in server mode, the active
server ID. A source mismatch invalidates the saved snapshot. In server mode, a
server-ID mismatch also invalidates it. A restore mismatch or corrupt envelope
discards that route and returns `None`.

`has_snapshots()` lazily discards incompatible envelopes before answering so
recent-work indicators cannot report snapshots that the current runtime is
unable to restore.

The store makes an outer mapping copy at save and restore boundaries. It does
not blindly deep-copy payloads such as large Console histories. Screen owners
remain responsible under ADR-011 for returning detached snapshot values whose
subsequent mutation will not alter live widget/domain state. Focused checks
cover Settings draft containers and a large Console snapshot so shallow
storage does not become either aliasing or accidental deep-copy cost.

### Navigation sequence

Navigation preserves this order:

1. Call and await the outgoing screen's `flush_pending_work()` when present.
2. Abort navigation on an explicit veto or flush exception.
3. Call `save_state()` and offer a valid mapping to `ScreenStateStore`.
4. Construct a fresh destination screen; do not cache/remount an old instance.
5. Restore a compatible snapshot when the destination supports it.
6. Apply explicit navigation context after restore.
7. Switch to the new screen and update the canonical current tab.

A `save_state()` exception or non-mapping result loses only view continuity;
navigation continues after a metadata-only warning. A `restore_state()`
exception discards that snapshot and continues with the fresh screen. An
explicit navigation-context failure produces a bounded warning and continues
with the recovered screen, matching the existing non-veto behavior.

Home, Workflows, and Schedules pass the current runtime identity to
`has_snapshots()` rather than inspecting a private dictionary. Explicit
Library, Settings, and Watchlists context remains higher priority than restored
view state.

## Destination Handoff Contract

### Typed channels

`PendingHandoffStore` defines explicit channel identities and accepted values:

- Chat context: `ChatHandoffPayload`
- Console live work: normalized `ConsoleLiveWorkLaunch`
- Console prompt insert: non-empty `str`
- Study scope: normalized `StudyScopeContext`
- Study initial section: validated section identifier
- Artifact Chatbook target: canonical `local:chatbook:<chatbook_id>` record ID
- ACP session target: canonical `local:acp_session:<session_id>` record ID

Producers stage through channel-specific typed methods or a typed generic
boundary. Values are validated, normalized, and detached before they become
pending. The store remains memory-only and has no serialization path.

"Detached" means later producer mutation cannot change a staged or claimed
value. After channel normalization, the store uses structural copying for
mutable nested containers:

- Chat rebuilds through `ChatHandoffPayload.to_dict()` /
  `ChatHandoffPayload.from_dict()`, which recursively snapshots its JSON-like
  contract mappings and sequences.
- Console rebuilds a normalized `ConsoleLiveWorkLaunch` and deep-copies its
  nested payload containers.
- Study deep-copies the frozen scope dataclass so nested
  `StudySourceItem.locator` mappings are independent.
- Prompt text, section names, and target IDs are normalized strings and need no
  additional copy.

If a value cannot be normalized or structurally copied, staging rejects it
without retaining a partial value and the producer uses its existing bounded
warning path. Handoff payload sizes remain governed by their current domain
limits; this rule does not deep-copy screen snapshots or Console histories.

### Revisioned single-slot protocol

Each channel has an independent monotonic revision and supports:

```python
def stage(channel, value) -> int: ...
def clear_pending(channel) -> int: ...
def claim(channel) -> HandoffClaim[T] | None: ...
def acknowledge(claim: HandoffClaim[T]) -> bool: ...
def release(claim: HandoffClaim[T]) -> bool: ...
```

The externally visible behavior remains a single latest pending value:

- `stage()` replaces any unclaimed pending value and advances the revision.
- `clear_pending()` advances the revision and removes the unclaimed pending
  value. When a claim is in flight, that newer empty revision prevents a later
  release from resurrecting the cleared value.
- `claim()` atomically moves that value into one in-flight claim.
- A second `claim()` while one is in flight returns `None`.
- `stage()` during an in-flight claim retains only the latest replacement.
- `acknowledge()` removes only the matching in-flight revision.
- `release()` returns the matching value to pending only when no newer
  replacement exists; it never overwrites a newer value.
- An acknowledge or release for a stale/non-current claim is a no-op and
  returns `False`.

The one in-flight value plus at most one latest replacement is coordination
state, not a replay queue. Once the in-flight operation settles, the newer
replacement remains the single next value.

Claims never escape into logs or persistence. A claim exposes its typed value
and opaque channel/revision identity only to the destination consumer.

The handoff store captures the creating application thread. Off-owner stage,
claim, acknowledge, and release operations reject. Workers marshal producer
changes to that owner boundary before touching a channel.

### Settlement

Consumers settle claims by outcome:

| Outcome | Settlement |
| --- | --- |
| Destination applied the handoff | Acknowledge |
| Value is invalid, target is definitively missing, or operation is unsupported | Notify bounded recovery and acknowledge |
| Destination is not mounted/ready, setup is incomplete, or a retryable lookup temporarily fails | Release |
| Consumer is cancelled | Release in `finally`, then re-raise cancellation |
| Unexpected exception | Release, emit metadata-only diagnostic, preserve existing user recovery |

The store does not schedule retries. Released values are retried only when an
existing mount, timer, navigation, or explicit user action invokes the
consumer. This preserves current behavior without creating a hidden worker.

### Chat and Console details

Chat context continues to honor the tabs-enabled gate before staging.
Consumption waits for the same Chat mount/setup prerequisites. A handoff is
acknowledged only after its context is actually applied or terminally rejected.

Creating the handoff's ephemeral Chat tab is part of the same settlement
transaction:

- Before an exact new tab ID exists, cancellation or retryable failure releases
  the claim.
- After that tab exists, any later cancellation or failure first closes that
  exact ephemeral tab and then releases the claim.
- If cleanup succeeds, the claim is released and a later retry may create one
  replacement tab. If cleanup itself fails, the claim is terminally
  acknowledged and a bounded warning is shown so the same intent cannot create
  duplicate partial tabs.
- On success, the claim is acknowledged immediately after the context is
  applied, before unrelated awaited UI work.

Console live-work launch keeps its current normalized screen-local context and
inspector behavior. The app-level claim is acknowledged after ownership has
been transferred to that screen-local context.

Console prompt insert continues appending to the user's existing composer
draft, never replacing it. Missing composer/readiness is transient and
releases. First-run setup blocking is an incomplete-readiness outcome and
therefore also releases after showing bounded recovery. An empty or invalid
normalized prompt is terminal and acknowledges.

### Study and Artifacts details

Study scope and section are separate channels so either may be staged alone.
`open_study_screen()` clears the other optional channel when its corresponding
argument is `None`, matching the current raw-field behavior without staging
`None` as a domain value. The clear advances that channel's revision so an
older in-flight failure cannot restore intent that the newer call removed.
Each is claimed and settled independently. Applying explicit incoming values
after restored screen state preserves their higher precedence.

The Artifact channel accepts only
`local:chatbook:<chatbook_id>` with a non-empty suffix. The consumer performs
exact lookup through `LocalChatbookService.get_chatbook(chatbook_id)`:

- Exact lookup and selection success acknowledges.
- A returned record must reconstruct the same canonical target ID as the
  claim; a mismatched or malformed service response cannot be selected or
  acknowledged as success.
- `KeyError` produces explicit missing-target recovery and acknowledges.
- An unavailable service, unready screen, or other lookup failure releases.
- The first-page listing and its "latest record" fallback cannot settle a
  requested target. Ordinary no-handoff latest-record rendering remains
  unchanged.

A newer target staged while exact lookup awaits cannot be cleared by the older
claim.

Artifact worker lifetime is subordinate to the destination screen lifetime.
The app thread tracks the exact active claim and a refresh generation.
Unmounting or restarting an exclusive refresh releases that active claim
before another generation can claim. A matching terminal worker-state event
releases any claim left active when cancellation or error prevents a callback.
A callback may apply or settle only when its generation and exact claim object
still match the live screen; late callbacks from cancelled, restarted, or
unmounted workers are inert and cannot strand or acknowledge a handoff.

### ACP target recovery

ACP claims the requested session target and reads the current
`ACPRuntimeSessionState`. The ACP channel accepts only the canonical record-ID
shape `local:acp_session:<session_id>` with a non-empty suffix. Producer and
consumer use one helper to construct that identifier from a normalized bare
session ID. The consumer reconstructs the current canonical record ID from
`ACPRuntimeSessionState.session_id` and compares the two complete strings; it
does not compare the staged record ID directly with the bare session ID.

- When the reconstructed current record ID exactly matches the claimed target,
  ACP keeps the existing current-session row selected, scrolls
  `#acp-detail-pane` into view after mount, emits a bounded informational
  notification, and acknowledges.
- When no runtime session exists, the target does not match, or the current ACP
  surface cannot focus session details, ACP shows an explicit
  stale/unsupported recovery message and acknowledges.
- ACP does not search fabricated history, change the current runtime session,
  or silently clear the target.

The recovery should tell the user that only the current ACP runtime session is
available and keep the normal ACP surface usable.

## Privacy and Diagnostics

ADR-022 applies to all new diagnostics.

- Screen snapshot and handoff payloads are never persisted or logged.
- Logs may contain route, channel, revision, outcome category, and exception
  type.
- Logs may not contain prompt text, context content, target values, model
  content, serialized snapshots, or object representations that contain them.
- Handoff exception paths do not enable traceback-local diagnosis that could
  serialize the claimed value.
- Regression tests stage unique sentinel secrets, force failures, capture
  logs, and assert the sentinel is absent.

## Compatibility

`AppState`, `ChatState`, `NotesState`, `NavigationState`, and `UIState` remain
importable from their current modules and `tldw_chatbook.state`. Existing
serialization tests remain valid. Their module and class documentation is
corrected to describe compatibility containers, not live application
authority. No import-time deprecation warning is added.

Existing app-level runtime projection attributes may remain temporarily for
read compatibility. Their values are projections from `RuntimePolicyContext`,
and ownership guards prohibit independent production writers. On `TldwCli`,
the projection boundary invokes the sole private publisher behind the three
getter-only properties; it neither reads nor writes a root `AppState`.

The application installs one `RuntimePolicyContext` identity for its lifetime.
Bootstrap prepares any synchronization commit and initial projection before
attaching that identity; failed preparation leaves no half-installed context
and may be retried.
Application integration coverage uses the full `TldwCli`; direct unit coverage
targets app-independent context/preparation/provider functions and classes.
No simplified or partial test application contract is retained in production.
Settings changes persist the server URL and token in one checked configuration
batch, then pass a locally loaded candidate configuration to one app-level
coordinator instead of loading and installing another authority.
The coordinator durably commits the existing context before publishing the
candidate configuration to the app/provider or invalidating provider state.
Commit failure leaves the old binding, configuration, target/default, cache,
projection, and screen unchanged. After commit, provider config/cache
installation precedes screen notification; legacy-target materialization is
best-effort because the refreshed provider config remains a usable fallback.
The screen callback is a contained post-commit observer and cannot turn an
already-committed switch into a reported rollback.
The successful Settings-file write is not rolled back if the separate
runtime-policy commit fails; it remains available for retry/startup.

## Migration Tasks

The implementation is split into four dependency-ordered, independently
reviewable tasks.

### TASK-643: Runtime authority and legacy-state detachment

- Add revision snapshots and persist-before-publish compare-and-swap commits.
- Make runtime-policy persistence follow the effective config path and ADR-022
  private-file boundary.
- Make capability refresh discard stale results.
- Remove `TldwCli`'s `AppState` dependency and correct compatibility docs.
- Remove independent runtime projection writers.
- Add runtime ownership and stale-result regressions.

ADR required: yes

ADR path: `backlog/decisions/026-application-session-state-ownership.md`

Reason: This changes the runtime authority and cross-module mutation contract.

### TASK-644: Screen snapshot owner

- Add `ScreenStateStore`.
- Migrate navigation and recent-work consumers.
- Preserve fresh construction, flush vetoes, restore/context order, and failure
  behavior.
- Remove domain-dictionary policy metadata and `_screen_states`.

ADR required: yes

ADR path: `backlog/decisions/026-application-session-state-ownership.md`

Reason: This defines application-lifetime view-state ownership.

### TASK-645: Chat and Console handoff owner

- Add the typed revisioned single-slot store.
- Migrate Chat context, Console launch, and Console prompt insert.
- Roll back exact partially created Chat tabs before releasing a failed claim.
- Verify replacement, cancellation, transient failure, and mounted-screen
  behavior before expanding to other destinations.

ADR required: yes

ADR path: `backlog/decisions/026-application-session-state-ownership.md`

Reason: This defines a new cross-screen delivery and settlement interface.

### TASK-646: Remaining handoffs and final ownership gate

- Migrate Study, Artifacts, and ACP.
- Resolve Artifact targets exactly and complete visible ACP target recovery.
- Remove the dead Notes slot and every old raw app pending field.
- Add the final AST guard and run integrated release gates once.

ADR required: yes

ADR path: `backlog/decisions/026-application-session-state-ownership.md`

Reason: This completes the same cross-screen ownership decision.

## Verification Strategy

Tests must be deterministic. Async concurrency tests use events, futures, or
barriers rather than timing sleeps.

### TASK-643 focused proof

- Persistence failure leaves state, revision, and projections unchanged.
- Default and overridden runtime-policy paths are colocated with the effective
  config path; no fallback read crosses between them.
- Symlink, ownership, and type violations fail closed before JSON parsing;
  eligible excessive file modes are hardened before parsing; atomic writes use
  private random temporary files and emit no path or state sentinel.
- Successful commit persists before observers can see the new state.
- A capability probe blocked on an event cannot commit after a source/server
  change advances the revision.
- Target-status side effects do not publish for a stale probe.
- Media Ingest and Study use the authoritative runtime operation.
- Compatibility state imports and serialization still work.
- A scoped AST guard rejects production assignments to
  `RuntimePolicyContext.state`, direct store persistence, and the exact app
  runtime projections outside their owner.
- Off-owner runtime mutations reject; a worker-marshalled app-loop mutation
  succeeds.
- Runtime-change persistence failure leaves the prior screen and runtime
  selection active and produces only bounded metadata-only recovery.

### TASK-644 focused proof

- Save and restore make outer copies and never mutate the caller's mapping.
- Source/server identity mismatches and corrupt envelopes are discarded.
- A large nested Console snapshot is not blindly deep-copied by the store.
- Settings copies nested restored draft containers before retaining or
  mutating them.
- `ccp`/`personas`, retired Library aliases, and other aliases sharing a
  canonical tab use the same snapshot, while distinct canonical tabs remain
  isolated even when their screen-owned names match.
- Mounted startup with alias defaults such as `ccp`, `notes`, and `customize`
  leaves `current_tab` at the resolver's canonical tab after post-mount setup,
  then saves/restores through that same key on the next navigation.
- Flush veto/exception prevents navigation; save/restore failure does not.
- Fresh screen construction and explicit Library, Settings, and Watchlists
  context precedence are mounted and verified.
- Recent-work consumers use `has_snapshots()`.
- Failure logs omit a staged sentinel payload.
- Off-owner snapshot mutations reject.
- An AST guard rejects app `_screen_states` ownership and direct consumer
  access.

### TASK-645 focused proof

- Claim is exclusive.
- Staging while claimed retains only the latest replacement.
- Clearing while claimed prevents release from resurrecting the older value.
- Stale acknowledge/release cannot remove the replacement.
- Release restores the claimed value only when no newer replacement exists.
- Mutating a producer's nested Chat or Console mapping after staging cannot
  alter the claimed value.
- Cancellation releases and propagates.
- Failure or cancellation after exact ephemeral Chat-tab creation closes that
  tab before release; injected cleanup failure terminally acknowledges and
  cannot duplicate the tab on retry.
- Chat and Console success, terminal rejection, and transient readiness paths
  settle correctly in mounted flows.
- Failure logs omit unique Chat/Console sentinel content.
- Off-owner handoff mutations reject while marshalled app-loop producers work.

### TASK-646 focused proof

- Study scope and section settle independently.
- Mutating a staged Study source locator after staging cannot alter the claimed
  scope.
- Artifacts resolves only an exact canonical target, never substitutes the
  first-page latest record, and preserves a newer target during awaited lookup.
- Artifact lookup failures omit a unique target sentinel from diagnostics.
- ACP canonical record-ID matching focuses the current session;
  the matching mounted flow exposes `#acp-detail-pane`, while
  malformed/missing/mismatched targets show explicit recovery and do not
  fabricate lookup.
- No old app-level `pending_*` handoff fields remain.
- `pending_notes_workspace_context` is removed.
- A narrow AST guard checks only the forbidden app-owned fields and known
  producer/consumer boundaries; it does not reject legitimate screen-local
  `pending_*` implementation details.

### Integrated release gates

Run focused tests and configured static/format checks for each task.
Planning-time verification found 46 unrelated full-tree Ruff diagnostics and
five files outside the Ruff formatter baseline, so the implementation plans
use explicit tranche-scoped Ruff commands and documented narrow baseline
exceptions rather than falsely claiming the untouched repository is
lint-clean. After TASK-646 integrates the tranche, run these expensive gates
once from the combined worktree:

```bash
pytest -q Tests/Packaging/test_installed_distribution.py
pytest -q Tests/UI/test_product_maturity_phase1_harness.py
pytest -q Tests/RuntimePolicy Tests/UI/test_screen_state_store.py Tests/UI/test_screen_state_full_app.py Tests/UI/test_pending_handoff_store.py Tests/UI/test_pending_handoffs_full_app.py Tests/UI/test_destination_handoffs_full_app.py Tests/UI/test_product_maturity_phase1_harness.py Tests/Packaging/test_installed_distribution.py Tests/test_application_state_ownership.py
```

The authorized integrated suite must remain green and records exact
pass/skip/warning counts and duration. Raw repository-wide pytest is excluded
because it would collect retired test/simplified applications; application
behavior is verified through the normal production `TldwCli` suites above,
while app-independent owners and static contracts are tested directly.

## Documentation Supersession

These older documents are retained only as history and receive explicit
warnings:

- `Docs/Development/state-decomposition-analysis.md`
- `Docs/Development/refactoring-issues-review.md`
- `Docs/Development/refactoring-issues-review-v2.md`
- `Docs/Development/refactoring-complete-summary.md`
- `Docs/Development/refactoring-fixes-summary.md`
- `Docs/Development/app-refactoring-plan.md`
- `Docs/Development/app-refactoring-plan-v2.md`
- `Docs/Development/app-refactoring-migration.md`

They must not be used as implementation instructions. Their root reactive
state, mirrored-write, persisted-screen, and cached-screen proposals are
superseded by ADR-026 and this specification. Their alternate refactored app
entry points were retired under TASK-105.

## Review Gate

Implementation planning may begin only after:

1. this written specification and ADR receive independent specification
   review;
2. all review findings are reconciled and re-reviewed;
3. the user approves the corrected written files.

Each task then moves to In Progress and receives its own detailed
Superpowers implementation plan. No application code changes belong in the
design-document phase.
