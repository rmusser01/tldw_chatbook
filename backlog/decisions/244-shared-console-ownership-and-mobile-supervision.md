# ADR-244: Shared Console ownership and mobile agent supervision

Status: Accepted in conversation (design only; written-spec review pending)
Date: 2026-10-25
Related Task: [TASK-36728](../tasks/task-36728%20-%20Design-shared-Console-ownership-and-mobile-agent-supervision.md)
Spec: [Shared Console ownership and mobile supervision](../../Docs/superpowers/specs/2026-10-25-shared-console-mobile-supervision-design.md)
Supersedes: [ADR-234](234-served-mobile-companion-surface.md), its separate mobile execution/storage contract
Amends: [ADR-236](236-served-desktop-approval-relay.md), its mounted-card, browser-lifetime and Canvas-coupled relay contract
Implementation: Not started; existing production behavior remains until verified cutover

## Context

The operator needs to inspect and correct existing Console agents from a phone
over VPN, including individual subagents, and switch between existing chats.
Work must survive browser closure and network loss while `--serve` remains alive.
Served mode is first; native process attachment follows later.

The existing Console already supports viewless execution. `ConsoleRuntime` owns
the store/controller/bridge and accepted turns above the screen lifetime. The
current mobile service instead constructs a separate engine graph over three
databases under `user_data_dir()/mobile`. That isolates phone-created chats,
workspace bindings and run history from the normal Console. Its partial desktop
approval relay does not make those chats shared. Separately, served terminal
WebSocket teardown stops the owning AppService child.

`AgentRunsDB.reconcile_orphaned_runs` also assumes one app instance per data
directory: its default constructor sweep can mark another process's live runs as
errors. Consolidating paths without qualifying that recovery owner is unsafe.

The storage split is an implementation divergence documented in ADR-234's
2026-10-16 amendment. It must be corrected as part of this work, not retained as
a second execution system behind an improved mobile UI.

## Decision

1. All served interfaces use canonical profile conversation, workspace and run
   stores. Remove mobile-only database selection and the separate mobile engine
   construction. Existing database responsibilities stay separate; phone-created
   chats become ordinary Console chats.
2. Keep execution in the existing Console runtime and domain owners. Retain one
   served child App/runtime per canonical profile/storage identity, containing
   that profile's open chats. A small served registry routes clients and live
   chat identities. Hold a process execution fence before runtime construction
   and run recovery. Native and served execution entry points must honor it;
   non-owning readers cannot sweep another process's runs. No new agent engine,
   job scheduler or mobile controller is introduced.
3. Browser connections are revocable attachments. Retain the owning process and
   output drain across disconnect, and perform a full reset/render handshake on
   terminal reattachment. Permit one full-terminal input/viewport attachment per
   Console at a time plus concurrent independent phone views. Stale attachments
   cannot mutate or tear down their successor. Server shutdown and explicit
   native lifecycle actions retain their cancellation boundaries.
4. Extend the existing authenticated mobile API and bounded process transport
   with runtime-owned snapshots, events and explicit commands. Control transport
   availability is independent of Canvas enablement; Canvas capabilities retain
   their existing session fences and feature revocation. Mobile view selection
   never changes the desktop active tab as an implementation shortcut.
5. Commands identify the owning epoch/runtime, exact chat/session incarnation
   and applicable run/handle/round/revision. Reuse native admission, queues,
   hooks, workspace authority, permissions, decision resolution and subagent
   mailboxes. Add explicit-target seams where native active-tab operations need
   them; keep existing active-tab callers on the same underlying behavior.
6. Assign mutations stable command IDs and body-free receipts. Reconnection
   reconciles authoritative outcomes and a bounded snapshot/event boundary;
   it never blindly replays a mutation. Preserve unknown outcomes through owner
   loss instead of claiming exactly-once external effects. Answer decisions once
   through their native retained owner, including background/viewless rounds.
7. Consolidate legacy mobile data using existing maintenance/capture/recovery
   machinery with explicit coverage of legacy stores and referenced files.
   Journal source/destination identities, conflict mappings, per-DB committed
   batches and final verification. Preserve complete history, references and
   authority restrictions. Fence writes through interruption; cut over only
   after verification. Keep originals as recovery copies; no automatic deletion
   or replay of imported pending agent work is authorized.
8. Build on the existing `/m` page, auth and static assets. Open on Chats and
   provide live transcript/decisions, main-agent controls, and direct subagent
   guidance/Stop with an always-visible target. Keep page-local navigation and
   drafts independent, make connection uncertainty visible, and queue no offline
   commands. Existing browser Library reading remains available.
9. Scope phone control first to the served runtime. Native discovery/attachment
   requires the later phase; exclusive native-or-served execution per canonical
   profile is the first release's coexistence limit. An active native owner or
   another served server prevents competing execution, while ordinary read-only
   access can continue without recovery sweeps. Never treat unregistered native
   rows as remotely controllable or infer owner death from a timeout. Existing
   dispatch recovery and maintenance admission remain separate authorities.

## Alternatives considered

| Alternative | Reason rejected |
| --- | --- |
| Keep mobile databases and add more desktop relay features | Preserves split chat/workspace identities and a second engine that can drift from native hooks, permissions and runtime behavior. |
| Point the mobile engine at the desktop DB paths | Sharing rows does not share live controllers, pending decisions, mailboxes or cancellation; duplicate runtime ownership remains. |
| Move agent execution into a new headless daemon | Viewless execution and its native owners already exist. A new daemon would expand the task and duplicate established runtime behavior. |
| Make the character-grid terminal the primary phone UI | Does not provide adequate touch navigation, reading or explicit per-worker targeting for this use case. |
| Switch desktop active tabs to implement phone commands | Races with desktop input and can apply the wrong draft, settings or command target. |
| Persist and replay every disconnected command | Can repeat uncertain sends, approvals or tool effects. Reconciliation must preserve uncertainty instead. |
| Delete legacy mobile stores or leave them indefinitely active | Deletion loses user history; leaving a normal second write path preserves the rejected design. Verified consolidation and retained recovery copies resolve both. |

## Consequences

The implementation needs served-host lifecycle/attachment handling, a shared
ownership registry, explicit-target native seams, authenticated projections and
command receipts, and a recoverable legacy consolidation. This is more than a
responsive stylesheet change. It does not replace the agent loop or invent a
second runtime state machine.

Durable mapping/receipt changes require the normal schema version increments,
executed SQL migrations, exact schema-catalog and backup/restore coverage.
Legacy mobile backup coverage must be established rather than assumed.
Consolidation cannot claim atomicity across three independent DB owners; the
journal and startup fence govern partial progress.

The operator's app authentication continues to authorize phone control; VPN
access alone grants no app permissions. Existing local-file exclusions, project
instruction consent, capture/privacy policies and tool gates remain in force.
Server/child crashes remain normal durable recovery boundaries. UI snapshots,
timeouts and persisted run rows do not prove dispatch or external effects ended.

The spec records targeted storage/runtime/API tests and real served-browser
qualification. Full-suite execution remains opt-in. Written-spec review precedes
the implementation plan and atomic implementation tasks; this ADR records the
approved design direction without claiming shipped behavior.
