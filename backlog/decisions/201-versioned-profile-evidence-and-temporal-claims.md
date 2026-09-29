# ADR-201: Bind Personal Context evidence to source authority and represent temporal claims explicitly

Status: Accepted — design direction approved, 2026-09-25; no runtime or schema rollout approved
Date: 2026-09-25
Task: TASK-25907.5
Extends: [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md), [ADR-182](182-personal-context-memory-evolution.md)
Related: [ADR-024](024-rag-citation-provenance-and-source-resolution.md)
Design: [Versioned evidence and temporal changes](../../Docs/superpowers/specs/2026-09-25-personal-context-versioned-evidence-and-temporal-changes-design.md)

## Context

V1 `ProfileProvenance` has opaque `source_references` and independent span
hashes. They do not bind a source owner, source version, exact span, access
authority, or whether the source supports the accepted wording. Settings now
labels them as legacy, unverified metadata. `ProfileRecord` and
`ProfileProposal` are fixed to schema version 1, and a proposal confidence is
not durable record confidence. A record's creation/update times and
`parent_version_id` are not a history of when its claim was true; deletion can
retire prior bodies. ADR-102 forbids silently downcasting canonical objects or
letting older peers edit unknown newer ones.

ADR-024 already defines governed message-citation snapshots and allowlisted
source resolution. Those message-owned traces are not Personal Context's fact
authority and cannot simply be copied into a profile record. The later memory
roadmap needs exact, inspectable evidence and corrections without inventing a
fourth memory store or making imported IDs executable.

## Decision

Design a **V2 canonical profile manifest, record and proposal contract** in
`tldw-profile-core`, leaving V1 canonical bytes, schemas and fixtures intact.
The new contract is data-only: a bounded record version may contain evidence
bindings, a claim classification, temporal validity and exact relation edges.
The shared package validates and canonically serializes those values; Chatbook
and the server own their repository transactions, authorization, resolver
adapters, encryption and UI. The current source-resolver inventory may be
adapted behind those runtime boundaries, but `tldw-profile-core` does not
import Chatbook's citation models or grant source access.

Each evidence binding identifies the source authority and governance scope,
source kind and object, immutable source version, exact span convention and
digest, and source role (direct user message, quoted material, attachment,
tool result, or import). Bare V1 IDs and hashes stay inert. Imported bindings
also stay inert until a trusted runtime validates and rebinds them to a current
allowlisted source owner. Every inspection or refresh rechecks the caller's
current authority before source decryption; it never searches other profiles,
workspaces, tenants, paths or URLs to make an ID resolve. A matching digest
proves byte identity of the recorded span, not semantic support or user intent.

The record keeps these meanings separate:

- **Source state:** what was bound at capture; current availability and access
  are fresh runtime observations, never immutable proof inside the claim.
- **Semantic support:** not assessed, supports, contradicts or insufficient,
  with assessment origin, exact claim digest and complete binding digest.
  Replacing source/version/span cannot reuse an old assessment even when claim
  wording is unchanged. Hash verification cannot upgrade this state.
- **User approval:** an explicit, version-bound action on the proposed wording,
  validity dates and relation edges;
  it does not certify evidence support. V1 approvals and later Settings edits
  do not acquire a fabricated V2 approval receipt.
- **Confidence:** an optional, attributed inference estimate, never a truth
  score or a substitute for approval.
- **Salience:** an optional, attributed priority hint, never support or
  temporal validity.
- **Temporal validity:** an explicit effective interval and its evidence
  basis; storage timestamps alone never establish it. Missing bounds remain
  unknown, not automatically evergreen.

Corrections point to an exact prior record version and say it was wrong for
an explicit overlap effect. A real change has a required `transition_at` effect
on `change_from`: the new interval begins there and the authorized historical
projection closes the old one there. Old immutable bytes remain unchanged.
The canonical successor carries its reviewed edge; admission atomically checks
all involved heads and the manifest. Stale targets reject admission, while a
later successor does not invalidate an already accepted historical edge.
Supersession names an explicit replacement boundary/effect without asserting
historical falsity. Workspace exceptions target exact global versions; a new
global head holds the affected workspace candidates pending review. Concurrent
or overlapping contradictory claims remain unresolved; newest never wins.

V2 automatic use requires current authority, reviewed known validity, approval
and resolved relations. Inference/imported claims additionally require exact
assessed support still available under current policy; missing observation
withholds them rather than opening sources automatically. Absence of a source
alone does not prove an independently approved direct assertion false. V1
eligibility stays V1 within a compatible profile.

Optional retained excerpts are **encrypted, bounded profile-owned governed
derivatives**, not a second fact authority. The strict intersection of source
and claim policy governs them and inline binding metadata. A later source-policy
revocation fences use/transport, creates a sanitized privacy-only successor and
retires entire metadata-bearing older envelopes, outboxes, managed recovery/Undo
copies and caches. It preserves permitted assertion content, never fabricates
new human approval, and holds automatic use until review. A V2 manifest evidence
retirement epoch and content-free encrypted control receipts fence stale replay,
workers, restore and peer reconnect until acknowledgements. Offline remote
erasure remains unconfirmed until peers receive and apply the receipt. The
separate forgetting implementation must satisfy this required lifecycle before
portable evidence ships. Local-only bindings require device-only records or
explicit reviewed binding-free successors. Agent evidence disclosure stays off
until its separate disclosure contract ships and is authorized.

V1 schemas/bytes remain unchanged and legacy metadata stays unknown. Migration
cannot infer authority, support, approval, dates or relations from IDs/hashes or
time stamps. A distinct V2 manifest declares profile-wide required context
semantics: a V2 exception can affect a still-V1 global record. Cutover requires
all registered active context consumers to acknowledge support or be explicitly
disabled with old profile grants retired. A V1-only active consumer blocks
cutover; unknown manifest vocabulary blocks the whole profile's automatic
context, including V1 records. Opaque retention alone cannot protect related V1
claims. Dormant offline copies must upgrade or be removed from active use; old
grants cannot reconnect, and immediate remote offline stoppage is not promised.
No lossy V2-to-V1 canonical update is permitted. Any opaque transport needs its
own complete qualification and current retirement barriers. Forward-only
migration, V2 models/schemas/vocabulary and shared canonical-byte fixtures must
be published by shared core and pinned in Chatbook and `tldw_server` conformance
tests before rollout.

## Alternatives considered

| Alternative | Decision |
| --- | --- |
| Add unversioned optional fields to V1 | Rejected: changes canonical bytes and invites older peers to drop or misread evidence. |
| Keep all evidence in a profile sidecar | Rejected as the canonical contract: record/proposal and evidence could sync or delete separately and leave a claim falsely cited. Encrypted excerpt payloads may remain separate governed derivatives. |
| Reuse message-owned `CitationTrace` as profile truth | Rejected: answer citation provenance and human profile claims have different owners and retention. A guarded resolver adapter may reuse allowlisted source checks. |
| Let newest statement overwrite older claims | Rejected: a correction, real change, scoped exception and unresolved contradiction have different meanings. |
| Resolve legacy IDs by searching all stores | Rejected: IDs are not portable authority and such a lookup crosses access boundaries. |

## Consequences and delivery gate

The contract will require versioned shared-core models and schemas, encrypted
repository migration, resolver and revocation adapters, source/claim dependency
tracking, Sync capability negotiation, server conformance, and user review
surfaces. TASK-25907.5 creates **only this decision and the design**; it adds
no production model, migration, resolver, provider access or behavior. The
companion server's current implementation has not been audited by this task.
No implementation may claim verified quotations or complete forgetting until
the separate evidence, forgetting and disclosure slices ship and pass their
own acceptance tests. ADR-201's number remains provisional against concurrent
branches until integration-time collision checks.
