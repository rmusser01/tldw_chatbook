# ADR-162: Managed agent plugins and Git marketplaces

Status: Accepted (2026-09-15) — written specification reviewed; implementation pending.
Date: 2026-09-15
Related Task: [TASK-32645](../tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Supersedes: N/A
Extends: [ADR-009](009-local-skill-trust-boundary.md)
Companion: [ADR-163](163-expanded-console-hook-runtime.md)

## Decision

Distribute agent capabilities as owned immutable plugin packages, acquired from
Git/local sources and explicitly selected Cursor/Codex installations, with
versioned adapters into Chatbook's existing runtime and permission services.
Install once per user-data directory; enable per named workspace or through an
explicit global default. Use a dedicated private SQLite registry and authenticated
trust generations to coordinate visibility, updates, revocation and recovery.

## Context

Skills, MCP servers, instructions and hooks need shared distribution and update
ownership. Flattening imported packages into standalone assets loses provenance,
component dependencies and reliable removal. Vendor packages also contain
execution-affecting catalog definitions and settings that cannot safely be
treated as cosmetic metadata.

[Tool-use Packs](107-portable-tool-use-packs.md) deliberately contain policy,
not executable installation. [Workspace defaults](079-workspace-assistant-defaults.md)
and [project instructions](069-console-project-instruction-local-state-and-preflight.md)
do not grant package or filesystem authority. The new system must preserve
these boundaries and ADR-009's offline-tamper protection.

## Contracts

1. Native authoring uses the portable Agent Plugins core and the versioned
   io.github.rmusser01.chatbook extension. Deterministic adapters record their
   selected dialect and overlays; ambiguous packages require interpretation
   selection. No indiscriminate manifest merging. A malformed recognized extension
   remains an activation blocker where required constraints cannot be recovered,
   even when portable components remain inspectable.
2. Installation IDs are independent of names, publisher claims and catalog
   references. Effective revision identity includes package bytes, catalog
   overlays, interpretation and selected component definitions.
3. Separate immutable package content, persistent shared/workspace data,
   configuration and credential references. Package rollback cannot undo
   arbitrary data or external effects.
4. Activation is intent. Trust, readiness, selection and ordinary runtime
   permission are separate. Required guards/dependencies never vanish through
   partial installation. New support does not auto-enable excluded components.
5. Extend authenticated skill trust through a separate plugin namespace and
   secure generation marker. Authenticate activation defaults/workspace overrides,
   selection/dependencies, hook requirements, execution mappings and credential
   references/authority-binding generations as well as content and revocation.
   Bind credential identity, endpoint/audience and scope through the owning auth
   service. Ordinary verified token renewal preserves that authority; identity,
   scope or revocation changes invalidate captured mappings. Token/storage
   revisions are separate from authority generations. Verify at use time;
   no hash-only authority, foreign grants or permission-default recovery. A live
   hook-disable switch cannot erase required dependencies.
6. One OS-locked plugin execution/mutation owner exists per user-data directory
   in v1. Other instances browse validated state. A real run leases its revision;
   idle connections and archived checkpoints do not block updates forever.
7. Applying updates fences new old-revision admission and drains active work.
   Publication uses staged files, a complete protected authority snapshot and
   authenticated intent. After the registry commits in SQLite, persist a separate
   authenticated commit certificate in the protected trust store; the secure
   marker then binds generation, operation ID and snapshot digest. A crash before
   certificate publication requires reviewed recovery. Prepared intent alone cannot
   advance the marker. Its matching snapshot permits exact authority recovery
   after registry loss, without recreating grants or bypassing surviving-child
   reconciliation. Projections cannot activate themselves.
8. Immediately fence the affected live scope and start host cancellation without
   waiting for persistence or trust unlock. Late approvals/callbacks cannot revive
   it, and affected plugin cleanup hooks are suppressed. Durable disable/uninstall
   success and installation file removal require committed revocation; persistence
   failure retains a session-only block with honest cleanup status. Workspace
   disable invalidates only that workspace's runtime generation; global disable
   and uninstall cover all scopes. Sharing MCP connections requires equivalent
   reviewed execution/configuration/credential authority and compatible session
   state. Cancel/detach scoped requests without killing other authorized users'
   transport. Unclean owner death requires surviving-child reconciliation;
   acquiring its lock does not prove cleanup.
9. Use the existing Git executable, portalocker, private SQLite, HTTP and
   credential seams. Direct generic MCP transport is a separately qualified
   prerequisite; a tldw_server wrapper is not equivalent evidence. SessionStart
   uses provisional normal authority and independently eligible, already-connected
   MCP prerequisites; initialization cannot grant itself readiness. Preserve
   typed MCP result error/structured fields through the service boundary before
   hook interpretation or display projection.
10. Plugins owns package management. Library Skills and MCP retain their component
    surfaces with service-enforced package ownership. Canonical Settings owns
    global preferences only. UI shows workspace versus installation-wide effects.
11. Saved-data deletion separately reviews exact roots, ownership/generations and
    all affected workspaces. Fence new access, drain existing users and confirm
    owned writers stopped before deleting; idle MCP processes may still be writers.
    Persist the root deletion fence through the coordinator and retain it across
    partial cleanup/restart. Surviving or unknown writers leave deletion pending;
    stale review/reattachment cannot redirect deletion. Data stays by default,
    and arbitrary external programs remain outside the containment guarantee.

The detailed state matrix, schemas, resource limits, compatibility behavior and
acceptance criteria are in the linked specification. No runtime feature is
declared implemented by this ADR.

## Alternatives considered

| Alternative | Why rejected |
| --- | --- |
| Flatten packages into standalone skills and MCP profiles | Loses revision, ownership, dependency and uninstall boundaries. |
| Execute each vendor's runtime | Adds parallel permission/session models and still cannot transfer hosted connector access. |
| Arbitrary Python application extensions | Much broader stability and execution boundary than agent capability packages. |
| Require whole-package compatibility | Prevents useful explicit partial installation; required dependencies can be enforced narrowly. |
| Per-workspace versions | Creates concurrent-version mutable-data and runtime conflicts before the ownership foundation exists. |
| Automatic catalog policies or imported grants | Discovery would become authority; foreign approval semantics do not establish Chatbook permission. |
| Multiple concurrent plugin-runtime owners in v1 | Existing stores do not coordinate cross-process runtime/data use sufficiently. |
| Global cross-service transaction framework | A bounded registry/journal authority solves this package lifecycle without rewriting every store. |

## Consequences

- Broad interop is component-specific and versioned, with explicit adaptations.
- The shared hook runtime has its own ADR/spec and receives owned definitions.
- Plugins execute with host privileges after review; process cleanup is not
  sandbox containment.
- Single-owner plugin execution and local-filesystem storage are explicit v1
  limitations. Cross-platform guarantees require platform evidence.
- Canonical ADR-009's standalone scope remains valid; plugin trust is an extension,
  not an automatic migration of existing trusted skills.

## Links

- [Managed plugins design](../../Docs/superpowers/specs/2026-09-15-managed-plugins-design.md)
- [Expanded hook runtime design](../../Docs/superpowers/specs/2026-09-15-expanded-hook-runtime-design.md)
- [Implementation delivery plan](../../Docs/superpowers/plans/2026-09-15-managed-plugins-delivery.md)
- [ADR-074: bounded write-ahead coordination precedent](074-portable-actor-packs-and-local-persona-visual-runtime.md)

### F3 persistence detail (2026-09-15)

Registry schema v2 adds mutable, explicit revision review state separately from
immutable revision content, and independent installation tombstones carrying
revocation generation and operation identity. Upgrade v1 in one owned transaction;
authenticating registry presence does not imply that a revision was reviewed.
Tombstones survive removal of installation rows. Closed authority mappings and
operation results contain references and stable authority generations, not token
material or live process state.


### Native skill admission integration (F5)

Registry schema v3 adds an optional installation alias. Pre-alias authenticated
snapshots omit null aliases from their canonical projection, preserving exact
bytes and digests; an upgrade never grants review, activation, or execution.
The app wiring lives in the current service-wiring and lifecycle modules. Plugin
authority is nested beneath the canonical protected Skills trust directory;
its worker uses the existing connection ownership context for native workspace
lookup caches, and Console work-chain offloads use the existing agent worker owner.

### F7 retry, retention and archived-resume amendment (2026-09-16)

Reviewed updates reserve a review-bound drain ticket, activated only when applying.
The shared lifecycle owner closes fresh admission and child/Stop continuations while
already admitted work retains normal current-authority checks. Waiting, cancelling
the proposal before prepare, and explicitly cancelling its captured work are distinct.
Replacement packages occupy a disjoint owned revision root; historical initial roots
and authenticated snapshot bytes remain immutable. Rollback uses current policy,
selection intersection and fresh review, and reports mutable-data compatibility Unknown.

R29 permits recovery from a contiguous authenticated suffix ending at its exact old
marker tuple. Every present disconnected artifact still refuses. Retire hints before
certificate, intent and orphan snapshot, oldest first; current, unresolved, live review,
leased and current reconstruction material stay protected. Compact package references
only through the same authenticated commit before removing bytes. Reset receipts and
archives remain separately protected.

R32/R37 require host-issued generation/request/result-bound mutation identities.
An immutable 900-second session request handle seals and cancels before worker access;
the existing worker attaches one authenticated durable review ID. Issuance is neither
permission nor commitment. Exact retained lookup never starts a mutation; expired or
changed requests refuse replay. Authenticate all protected evidence before classifying
unmatched SQLite hints: verified old IDs may expire, newer uncertified IDs still require
recovery even when their intent is additionally lost. A fixed authenticated legacy map
(at most 1,001 entries, 8 KiB per entry, 16 MiB total) precedes metadata-v2 cutover and
any pruning. A pending v1 map may change only after exact ancestor/entry requalification;
the durable v2 map is immutable. Salt, historical snapshots and key derivation stay intact.

R38 extends ADR-063's existing continuation field with a closed V2 managed-resume
pin authenticated for the exact namespace, durable owner and entire checkpoint.
The existing producer seals each store-resolved checkpoint before write/dispatch.
Resume verifies revision, actual component definitions/closure, scope generations,
mappings and qualified data coverage before context and again at fresh admission.
V1/pinless recovery has a zero-managed ceiling; foreign/remapped pins refuse exact
managed resume. Historical context and explicit new runs remain ordinary work.
Retained fleet transcripts hold an immutable host pin within their existing memory
budget, with no restart resurrection or separate transcript authority. Caps are
256 KiB per envelope inside 8 MiB checkpoints, 64 installations, 512 components per
installation/1,024 total, and 256 root references. Unknown data ownership refuses exact
resume; F8 and M4 must qualify their actual data and connection owners through this seam.

### F7 retention compaction (R40/R42)

Owned retention uses a closed `retain` operation and the existing commit pipeline.
Its original review and authenticated issuance bind the exact inactive revision set;
its nonce is distinct from a triggering update/request. It changes no current
selection, trust, alias, mapping or grant. Advisory first-observed revision dates
use existing receipt metadata and do not establish ownership. Current, live,
unknown-owner and unresolved recovery references remain protected. Reference
compaction commits before descriptor-qualified removal; file failures are reported
as cleanup pending and can be retried from retained authenticated compaction evidence.
The 2 GiB package/cache/staging quota counts unqualified material. Staging cleanup
requires explicit reconciled producer ownership, terminal status and the original
physical directory identity after 24 hours; mtime or absent leases alone never qualify.

R44: only `retain` results carry `retired_revisions`: 1–1,000 closed entries,
at most 1 MiB canonical UTF-8 aggregate, containing exact digest, owned path and
original physical root/anchor/directory identities frozen before issuance. The
existing authenticated result is restart cleanup custody. Other operation kinds
reject that field; historical absent fields preserve exact bytes. Pending cleanup
keeps its transition evidence protected until exact owned removal/absence is proven.

### R45: interrupted snapshot retirement (2026-09-16)

After successful reconciliation and durable metadata-v2/fixed-cutover validation,
the existing exclusive storage owner may rediscover orphan historical snapshots.
Inventory the entire private snapshot directory before cleanup, bounded at
`2 * MAX_TRANSITIONS + 2` (2,002) regular digest-named files; unexpected names,
links, unqualified ancestry, overflow or invalid authentication refuse cleanup.
Keep the existing 32 MiB per-file bound and decode bodies sequentially. New snapshot
allocation obeys the same count cap; an existing exact snapshot remains retryable
at capacity. Authenticate each candidate's exact header/body/digest and bind those
bytes to its opened file and directory identity before descriptor-relative unlink.
Recheck current marker and authenticated transition references before removal.

Protect bootstrap, current, generation >= current, every present intent/certificate's
new snapshot, unresolved/live-review/recovery material, and separately owned reset
archives/receipts. A retained suffix's old endpoint tuple does not require its old
body under R29. An older unreferenced result qualifies only through its valid pi1
MAC with matching issued/snapshot generation or exact result-digest membership in
the fixed legacy cutover map. Unknown legacy results refuse cleanup; never expand
that map, infer commitment, authorize replay or select package/data/reset bytes.
No live admission/revocation lock is held during this storage work.
