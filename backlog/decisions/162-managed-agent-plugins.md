# ADR-162: Managed agent plugins and Git marketplaces

Status: Accepted (2026-09-15) — written specification reviewed; implementation pending.
Date: 2026-09-15
Related Task: [TASK-32645](../tasks/task-32645%20-%20Design-managed-plugins-and-expanded-hook-runtime.md)
Supersedes: [ADR-111](111-mcp-remote-transport-and-client-dependency.md), only for direct generic Streamable HTTP transport ownership.
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

## Direct MCP transport ownership (R4, 2026-09-17)

This decision partially supersedes [ADR-111](111-mcp-remote-transport-and-client-dependency.md)
for Chatbook's direct generic Streamable HTTP connection. The existing
`MCPClient` and core `httpx` own that connection. Local/unified/provider services
retain permission, readiness, definition-hash and audit ownership; transport
code preserves complete typed/raw results and host-observed uncertain outcomes.
No federation manager, parallel policy plane or new mandatory vendor client is
introduced. The reciprocal ADR-111 amendment records the inspected upstream
revisions and why they do not provide this boundary.

Qualify the named 2026-07-28, 2025-11-25 and 2025-03-26 profiles through actual
transport exchanges: current per-request metadata/discovery and older
initialize/session behavior are distinct. Existing stdio profiles remain
readable. An uncertain invocation is never replayed on reconnect. Streamable
HTTP SSE responses are supported according to the selected profile; this does
not authorize fallback to deprecated HTTP+SSE transport, weakened TLS or
forwarding credentials across origins. Supported explicit credential bindings
are a separate integration; generic OAuth and hosted connector access remain
unclaimed until their actual owners are implemented and qualified.

[TASK-32682](../tasks/task-32682%20-%20Add-qualified-direct-Streamable-HTTP-MCP-transport.md)
and the [MCP delivery plan](../../Docs/superpowers/plans/2026-09-15-plugin-mcp.md)
implement this reconciliation. Recording the owner does not itself establish
wire conformance or runtime qualification.

Version negotiation preserves the selected era (R60). A recognized modern
unsupported-version error cannot trigger legacy initialization. Select only a
different qualified modern revision; otherwise report unsupported version,
including offers containing only legacy/unknown versions or the rejected
version itself. Explicit legacy profiles may negotiate either qualified legacy
revision. A user may select a legacy profile for a separate connection attempt;
no automatic handshake change or uncertain invocation replay follows a failure.
This conservative interpretation of the linked specification may require that
manual selection for some dual-era servers; changing it requires qualification.

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

### F8 exact-root custody and runtime checkpoint (R18/R43, 2026-09-16)

Plugin registry migration 004 adds noncascading exact-root users and a closed
process coverage/expected-grants commitment. Initial owner and all grants commit
together before access; idle, pending, reader and old-generation grants remain
until trusted host terminal evidence. Missing joins or unknown coverage refuse
cleanup. A live pending claim spans storage work under the existing lifecycle
owner, without holding its lock during IO. Original root ownership never changes;
reviewed attachment is separate and advances generation after drain.

Each authenticated root may carry an optional closed physical binding and cleanup
intent. Absent legacy fields preserve historical signed bytes and mean unknown
custody. A cleanup group contains 1–256 exact roots of one original owner, bounded
at 256 KiB. Waiting, deleting and final completion/cancellation each use a distinct
issued review ID/nonce through the existing commit pipeline. Commit deleting before
first unlink; errors or cancellation after that point retain fenced cleanup pending.
Only confirmed removal and directory durability advance generation. No read-only
lookup starts destruction. Every destructive step rechecks exact authenticated
root targets and descriptor-relative no-follow ancestry. Platform proof must be
qualified from actual identity fields; unsupported identity refuses.

One separate, overwriteable checkpoint in the existing marker backend records
version 1, session nonce, namespace identity, exact PluginMarker and clean/dirty
phase (at most 4 KiB). It changes neither PluginMarker nor archived pin identity.
Verified dirty publication precedes every session's first root grant/operation.
Shutdown closes root admission, confirms all owners/handles terminal and settles
joins, then publishes clean against the final authority marker before owner release.
Missing, dirty, malformed or mismatched prior state quarantines every retained root,
including orphans, until explicit reviewed host reconciliation. Empty restored SQL
rows, a released lock or PID liveness are never quiescence proof. Clean publication
failure preserves dirty/recovery state. Fresh bootstrap may initialize clean only
with no prior root custody; reset cannot erase retained unresolved evidence.

The secure backend guards against SQLite rollback relative to that backend; the
explicitly accepted file backend retains reduced rollback protection, including
its separate checkpoint file. Co-restoring those files does not prove drain.
F7 admission/pins compare exact attached membership, original/current owner,
scope, binding, generation and fence; package-only qualified-none is explicit
producer evidence. H2/M4/H6 own actual grants and terminal callbacks through this
owner; I6/I7 keep committed fence, drain, cleanup and recovery outcomes distinct.

R46 qualifies the native APFS binding within one authenticated Darwin boot-session
UUID, read with bounded native sysctl and stable readback. Bind exact native
fstat device/inode/birth seconds and nanoseconds for the owned anchor and leaf;
Python floating birth time and zero st_gen are insufficient. Changed, missing or
malformed boot identity refuses retained-root grant/reuse/deletion with the specific
root_boot_identity_changed reason until explicit reviewed quiescence/rebinding
advances generation. Package-only use remains available. Same-boot application
restart still requires every R43 clean/ownership/identity check. This is a tested
host-owned comparison model, not mathematical non-reuse or cross-boot proof;
after system reboot users must reconcile retained data. No production compiler or
new dependency is used, and simulated boot changes are not reboot experiments.

### F8 retained intent and lifecycle entry

Pending root groups authenticate their action (create/delete/attach) and explicit attachment target alongside membership and phase. Restart resumes that same action; an attachment can never become deletion. Root authority commits require the root lifecycle entry, which owns dirty-checkpoint, proof, drain and phase checks; generic commit cannot bypass it. Cancelling a failed proposal clears only its own live fence after authority recovery proves no deleting phase was committed. A committed deleting phase remains fenced for reviewed recovery. The service installs the exact reviewed live fence before queuing worker storage.

### F8 review corrections: completed absence and final publication

An authenticated `cleaned_absent` root remains a tombstone: reconciliation may confirm its missing leaf only through its exact current no-follow ancestry and original binding. Any unexpected leaf or replaced/missing ancestry refuses; a missing `present` root is never completed absence. Reviewed whole-root proof, generation advancement and boot rebinding rules still apply.

Shutdown seals ordinary service admission before finalization and rejects queued ordinary callbacks on the worker. Already executing calls must finish before clean can be published; a refused close continues to admit exact terminal settlement, not new authority work. The final checkpoint remains the last protected publication. Later root grants preserve all still-owned original epochs, and every cleanup phase retains the original review deadline; expiry leaves pending recovery for a fresh explicit review.

### Literal MCP endpoint queries (R61)

Preserve literal routing query parameters in an MCP endpoint URL, including
duplicates and empty values. They are visible configuration, not credential
references, and changing the full endpoint invalidates its discovery. Do not
expand placeholders/environment values or insert host credentials into URLs.
Host authorization uses its selected-origin credential service. Existing
HTTPS/explicit-loopback, userinfo/fragment and redirect restrictions still apply.
This follows the [Agent Plugins endpoint contract](https://agent-plugins.org/specification)
without adding a blanket query-string restriction.

### M3 credential reference recovery boundary (R62)

M3 captures and validates complete connection mappings through the actual local MCP and credential owners: saved profile target, retained component definition, effective configuration and stable credential binding must all match. Authenticated recovery may reconstruct only those supported current references. M3 recovery fixtures may seed the existing protected snapshot/transaction boundary, but this does not qualify a public mapping-edit or launch workflow. M4 supplies reviewed publication and registration before plugin connections launch. No mapping, successful recovery or credential binding creates tool permission or vendor grants.

### Credential reference identity after record loss (R63)

New MCP credential records receive immutable UUID reference IDs from the host credential service. A supplied missing reference is unready and cannot be recreated at generation one. Normal replacement, renewal and revocation reread the protected record under the existing owner lock; retained tombstones and monotonically advancing signed-64-bit authority generations prevent ordinary reuse, and generation exhaustion refuses. After record loss, the user must create and review a fresh reference before rebinding a plugin mapping. No credential creation restores prior tool permission or proves a prior remote invocation completed.

### Credential I/O and async transport deadlines (R64)

Blocking credential-store and file-lock operations run outside the shared MCP event loop. The existing credential service retains at most one worker operation; other async callers wait within their applicable deadlines before performing a fresh operation, without a queued worker backlog or secret-result cache. Cancellation ends the wait and cannot trigger later HTTP dispatch; it does not terminate an OS keychain call. A stalled backend may retain one daemon worker until completion or process exit and cause authenticated requests to time out, while anonymous connections remain responsive. Capacity releases only after actual completion, and local waiting or cleanup never proves remote invocation completion or permits replay.

M3 implements these decisions with a data-root-scoped MCP-only secure keyring
namespace and schema-3 profile references. The existing credential/configuration
owners remain the only source of current binding validity; protected plugin
snapshots carry metadata, never secret values. Its explicit header wire policy
preserves Latin-1 octets and reserves host protocol headers; it does not narrow
portable inventory parsing or claim generic MCP OAuth support. Operational APIs
and qualification limits are recorded in the MCP subplan's M3 implementation
contract; M4 retains reviewed mapping publication/launch responsibility.

### Shared MCP session qualification and uncertain request custody (R65/R66)

Owned MCP profiles default to separate scoped connections. Sharing requires an explicit host-controlled `request_independent` qualification bound by the same immutable configure review and authenticated mapping as the exact execution, definition, effective configuration and credential authority. Package metadata, a server claim or transport multiplexing cannot supply it. Unknown qualification stays isolated. Every request retains its workspace/parent/permission and current-authority checks; changed qualification or binding requires review. The host attestation can be mistaken and does not prove arbitrary external-server or original-host isolation.

A same-session request with an uncertain outcome retains its published active owner, durable host identity/outcome and complete root joins while exact connection custody remains. Local waiter cancellation does not call recovery settlement merely to mark uncertainty, release a request, or block unrelated authorized B work globally. Existing unresolved and foreign-session records are never promoted; lost custody and restart follow the existing recovery/dirty-checkpoint gates. Request completion cannot settle idle writer-capable server lifetime, and local transport closure cannot prove uncertain remote completion or permit replay. Uncertain requests can continue blocking revision drain and data deletion until positive terminal evidence exists.

### Initial owned MCP setup and discovered definitions (R67)

Saving owned configuration is data-only. Publish its exact connection mapping through the existing immutable configure review/commit before explicit connect or test. That authorized discovery can produce tool definitions; publish their exact reviewed tool mappings through the same configuration owner before plugin advertisement. Unchanged already-reviewed discovery may be reused. First setup can therefore require a connection review followed by a discovered-tool review. Neither step grants ordinary tool permission, starts execution implicitly or creates another approval owner; all per-call checks and no-tools/call probing rules still apply.


### Portable literal HTTP headers and credential separation (R68)

Agent Plugins 1.0.0 section 7.2.1 defines remote headers as visible package data,
with client-generated HTTP/MCP/authorization headers taking precedence by
case-insensitive name. Only the authenticated retained package definition may
supply an owned profile's literal header map; arbitrary save-time raw overrides
and foreign per-install header import remain forbidden. The reviewed exact
configuration digest covers that public declaration. Runtime credentials stay
in the protected credential owner and are resolved fresh for the selected origin;
no resolved secret goes into an authority snapshot, profile, audit or diagnostic.
Header spelling cannot prove a value is non-secret.

Compose one case-insensitive map from package literals, then current credential
headers, then authoritative host HTTP/MCP/routing/session fields. Preserve host
framing, hop-by-hop, proxy and MCP namespace controls, including headers normally
generated by the HTTP client. Never expand placeholders in URL/header names or
values, forward across origins, or follow redirects. Preserve the existing exact
Latin-1 wire policy for empty/interior-HTAB/obs-text values; wider Unicode remains
explicitly unsupported at runtime. Package literals are not a credential mechanism,
and standalone raw secret fields remain refused. Actual controlled-peer observation,
case-collision precedence and no-launch save tests qualify this boundary; they do
not qualify external services, arbitrary Unicode or foreign app behavior.

Source: [Agent Plugins specification, remote MCP configuration](https://agent-plugins.org/specification#streamable-http-and-legacy-httpsse).


### Component readiness and immutable MCP capture ceilings (R69)

A missing, changed or unusable MCP owner mapping makes its own component and
declared dependents unready. It does not discard a valid independent sibling.
Global authenticated authority, retained interpretation and current installation,
scope and root generations remain exact; unknown/missing required dependencies
cannot be treated as independent. Recovery still validates every supported
reference in the complete authenticated snapshot.

The existing MCP capture API may take an explicit component ceiling, checked
against current authenticated selection with the complete prerequisite closure.
Unavailable explicitly requested components refuse instead of silently shrinking
the request. Mappings, dependency records and advertised tools respect that
immutable ceiling. Default capture uses the currently eligible set. An already
admitted snapshot containing a newly failed component still refuses; a fresh
narrower independent capture is required. A B-only snapshot need not fail merely
because unrelated A becomes unready, provided B's captured mappings, complete
requirements and generations still match. No narrowing restores an old approval,
replays an invocation or weakens whole-snapshot recovery.


M4 implementation note: owned configuration is local MCP profile schema 4, while connection/tool authority remains exact mappings published by the existing plugin review/commit owner. `PluginMCPProvider` uses the existing MCP tool and permission runtime; `ConnectionOwnership` adapts that client's actual lifetime into existing `PluginRunOwnership` and root custody. Explicit component ceilings include prerequisites. Current-session unknown requests remain active custody under R66, and recovery still validates every authenticated mapping under R24. API/operation details and qualification limits are in the managed-plugin design's M4 owning-service section.

### Current native owner integration (M4)

Retained MCP launch/request tasks acquire their own recovery admission through the existing worker-isolation seam; inherited task state is never treated as a transferable storage lease. Direct profile admission refusal returns false, and owned launch requires an actual true result plus the exact retained session.

For an exact host-qualified request-independent stdio session with another attached owner, a request deadline/cancel retains its original native producer/source admission until its original validated terminal reply or actual child exit. It never replays or terminates the shared peer to settle one request. Ordinary/separate/last-owner cleanup retains native kill-and-reap custody. Revoked scopes still refuse late results.

Portable MCP expansion recognizes only ${PLUGIN_ROOT} and ${PLUGIN_DATA} in args, env values and cwd, once. Unknown placeholder text remains literal. Stdio configuration requires an explicitly created persistent plugin data-root binding before publication/launch; saving configuration never creates a root or launches a peer.


H6 integration note: native MCP hook effects consume original typed wire evidence through request-local capture inside the ordinary provider, with final object identity binding. No separate raw invoker or hook permission is added. M4 owned request/root records remain the resource owners; bridge cancellation alone cannot release hook lifetime counters. Private prospective Console integration currently composes independently eligible standalone MCP. I1 must publish native managed graph requirements and compose the existing owned provider in the application; unknown declarations remain unavailable, and this handoff does not claim that later integration.


### I1 native capability integration

Selected commands, rules, agent presets, skills, MCP and hooks share the existing
immutable admission. Reviewed host tool/model references reuse authenticated Mapping
records and current catalog/routing owners; they never create permission. Explicit
EMPTY constraints reach actual inline/fork/child catalogs and owned MCP approval and
dispatch. Skill model overrides remain unsupported. Agent models retain the actual
parent provider/endpoint floor. Live-only typed context carries tool restrictions;
historical opaque prose cannot acquire authority. Component-local readiness retains
unavailable metadata without automatically selecting prerequisites.

Native hook command/root custody adapts F2 to the existing H2 process owner. Real
Console MCP composition uses M4's connection owner and ordinary tool discovery.
Exact per-definition and live-material dependency requirements reuse H3 checkpoints;
ordinary unrelated tools do not acquire another component's dependency. Qualification
and limits are recorded in the current-dev integration report for TASK-32686.
