---
title: Foreground Personal Context source inspection authority
status: Accepted
date: 2026-09-26
---

# ADR-191: Foreground Personal Context source inspection authority

## Context

[ADR-201](201-versioned-profile-evidence-and-temporal-claims.md) defines exact
version-bound evidence, but a structural binding is not a source capability.
Personal Context profile reads, Console conversation ownership and citation
source namespaces have independent authority. Current V1 provenance is legacy
metadata; historical projections need not retain exact original bodies.

## Decision

Accept the design direction only, 2026-09-26, after the user endorsed the
contract subject to technical review and the verified findings below were fixed.
No runtime or schema rollout is approved.

Qualify the first native inspection as a foreground Settings inspector action
coordinated by the application, restricted to an independently authorized,
already open persisted local conversation. The
host obtains current profile and source owners; no imported DTO, permission
boolean, legacy reference, caller-supplied DB handle or profile grant mints
conversation access. Source namespace and Personal Context identity stay distinct.
The loaded Console owner may survive behind foreground Settings; do not require
two foreground screens or bootstrap, resume or navigate to obtain source owners.

Authorize before content access. Read current membership, immutable revision
and unmodified content coherently. Verify complete binding identity, codepoint
bounds and both exact digests. Use existing revision metadata and a narrow
parameterized source read, never generic getters or lazy revision creation.
Bound read/decryption/hash/rendering work; a mismatch returns content-free unavailable without historical or approximate
fallback. Quote text is a disposable local observation, not semantic support,
origin attestation, human approval or model guidance.

Ship admission, source read, publication fence, invalidation and the mounted
caller together. Every relevant native writer/revocation must join the gate;
a reader-only lock or optimistic double check is insufficient. Final source and
profile probes require fresh committed snapshots, not borrowed transactions.
Actual worker capacity stays reserved through cancellation cleanup; connection
budgets and interrupts belong only to that inspection. Timed proposal/authority
expiry clears visible results without auto-reading sources. Escape terminal and
bidi controls only in a labelled display projection, preserving hash input.
Cross-process freshness and privacy revocation require explicit deployment qualification.
Display observation time and clear quotes on known lifecycle changes. Do not
retain, sync, export or send the observation to tools/providers/traces.

## Alternatives

- Generic source resolver using caller-supplied IDs/capability DTOs: rejected
  for this release because parsed data does not prove current source authority
  and introduces unattended source/egress paths.
- Complete portable V2 rollout in the same change: rejected as a release unit;
  compatibility, retirement, disclosure and server qualification exceed this
  local inspection interface. They remain prerequisites where applicable.

## Consequences and approval boundary

The constrained flow can inspect exact local spans while keeping unresolved
legacy metadata honest. It deliberately cannot open another conversation,
resolve old versions, attest direct user intent or provide continuous freshness.
The host needs a real source admission factory, bounded storage read and complete
publication/revocation integration; none is installed by this decision.

TASK-25907.15 delivers this accepted design contract
and [specification](../../Docs/superpowers/specs/2026-09-26-personal-context-foreground-source-inspection-design.md)
only. V2 local containing-record admission and applicable retirement/disclosure
controls must be qualified before runtime inspection planning. This includes
retirement and disclosure of binding metadata itself; the existing device-only
model/tool gap cannot be bypassed by assigning that flag. Accepting this
ADR does not authorize a source resolver, new grant, migration, provider call,
key provisioning or portable/server V2 rollout. Recheck ADR numbering at integration.
