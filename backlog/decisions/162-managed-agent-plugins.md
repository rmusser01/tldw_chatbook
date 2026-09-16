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
