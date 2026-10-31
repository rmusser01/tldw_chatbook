# ADR-202: Fence dependency-aware Personal Context forgetting across existing owners

Status: Accepted — design direction approved, 2026-09-25; no deletion, migration or runtime rollout approved
Date: 2026-09-25
Task: TASK-25907.6
Extends: [ADR-102](102-personal-context-profile-authority-sync-and-encryption.md), [ADR-182](182-personal-context-memory-evolution.md), [ADR-201](201-versioned-profile-evidence-and-temporal-claims.md)
Related: [ADR-080](080-trace-v2-exhaustive-event-projection-and-collaboration.md), [ADR-092](092-console-full-semantic-capture-policy.md), [ADR-096](096-console-safe-capture-retention.md), [ADR-097](097-console-reference-backed-semantic-trace-ledger.md), [ADR-119](119-llamacpp-prompt-cache-snapshot-ownership.md), [ADR-052](052-console-conversation-memory-and-compaction-policy.md), [ADR-059](059-notes-folder-import-and-device-local-sync-ownership.md), [ADR-106](106-human-reviewed-agent-lesson-promotion.md)
Design: [Dependency-aware forgetting](../../Docs/superpowers/specs/2026-09-25-personal-context-dependency-aware-forgetting-design.md)

## Context

Current profile deletion retires prior record bodies and pending record outbox
content but creates a 24-hour encrypted Undo image. Proposals/promoted copies,
interviews with separate keys, conversation/tool/compaction content, Notes,
applied lesson promotions, AgentRunsDB steps/results and filesystem run logs,
file-sync recovery, Safe/Full exchange captures and semantic trace artifacts,
app-managed prompt-cache snapshots, copied Sync staging and remote peers have
separate custodians. V1 source IDs and hashes do not establish a
complete dependency graph. A tombstone cannot justify “forgotten everywhere.”

ADR-201 defines exact V2 evidence and metadata retirement with a profile-wide
compatibility gate. It requires a forgetting implementation before portable
evidence or new memory jobs ship. Retained source material must not silently
recreate a deleted claim under a new ID; independently authored user material
must not be destroyed based on a body match.

## Decision

Design **user-initiated, bounded dependency-aware forgetting** through native
owner adapters, an encrypted control/worklist journal and explicit publication
fences. This ledger contains privacy controls, never another fact store.
Archive, deletion with Undo, immediate memory forgetting, source deletion,
local removal and profile delete-everywhere keep distinct semantics.
Immediate means use/admission fenced at durable admission; cross-owner cleanup
and peer completion are separate acknowledged outcomes, with no Undo.

The reviewed plan binds exact target heads, owner permissions, dependency
inventory and scope. Register artifact/input-version edges at admission;
identity similarity or model text interpretation grants no deletion authority.
Retire wholly generated mixed-source artifacts as units without model redaction.
Manual Notes, source transcripts and applied instruction/file promotions remain
independent owners and require separate exact consent. Denied source metadata
is not disclosed or retained to make retries convenient.

Suppress identified memory reuse independently of record IDs. Store encrypted secret-keyed
structured claim-family selectors with the profile (no readable subject text) and native object-wide
no-memory marks with the source owner. Future source versions do not bypass
those marks. Every memory-producing/restore/Sync path checks controls at
admission and final publication; late workers and unsupported owner gaps fail
closed. Restricted cross-owner source locators are retired under ADR-201,
while owner-native marks stay under that owner's own policy. Whole-profile
purge destroys its claim-family ledger, retains only content-free profile
barriers, and leaves source marks with sources the user chose to retain.
Fresh user authorship or source-policy re-enablement requires explicit review.
Fresh authorship admits only the reviewed new record/version and input; it does
not clear family/source controls for replay or later automatic recreation.
This does not promise detection of unknown copies or arbitrary paraphrases.

Extend ADR-201's `evidence_retirement_epoch` for the managed artifact worklist;
use a peer-local retirement revision for device-only operations without a
shared epoch/receipt, and keep ADR-102 `purge_generation` for whole-profile
deletion. A shared-core
versioned content-free retirement receipt uses only random artifact/control
identities, page information and bounded reason enums; no payload, source hash,
locator, path, label or device-only identity is sent for tracking. Canonical
schemas/vocabulary/fixtures and client/server pins must qualify this contract.
Native authority and exact user consent remain runtime-owned.

Cross-database transactions use durable intent and owner acknowledgements,
not a claim of one global transaction. A lease/ticket fences publication at
each owner commit/qualified provider-adapter entry. A common cross-process
retirement admission gate
covers final control check, durable publishing-ticket state, owner transaction
and outcome receipt; forget admission drains it before committing the fence.
Reserved tickets may be cancelled; publishing tickets must finish or durably
acknowledge verified rollback/no retry. A cancellation request or lease expiry
cannot authorize a fence. Ambiguous crashed publishers block admission until
owner-receipt recovery resolves the outcome. App-owned queued payloads remain
managed: scheduling or a prepared checkpoint is not provider delivery. Final
adapter entry consumes current send authority under the gate; its receipt reports
pending/cancelled versus potentially begun send, not confirmed network delivery.
Cancel/retire queued payloads and delayed retry/fallback work, fence later entry,
and retain cleanup responsibility for app-owned copies after in-flight admission.
Phases are Prepared, Fenced, Retiring, Needs attention,
Locally complete and Awaiting peers/complete within declared managed scope.
Retries use immutable operation/page IDs and content-free owner receipts.
Startup/restore apply fences before content becomes usable. Already retired
content is never reconstructed for rollback; uncertain cleanup stays fenced.
Cleanup covers histories, before-images, indexes/FTS, proposals, copied outbox,
staging and managed recovery, through each owner's current guarded boundary.
AgentRunsDB steps/terminal recovery and segmented filesystem run logs can retain
profile tool/model output despite credential/path sanitization. Their native
DB/file owners must fence append/search/recovery/export and retire exact reviewed
generated units under append-only and current file authority constraints; mixed
units or unknown historical coverage cannot silently claim complete cleanup.
Safe capture keeps the first system row and may retain profile text; its blobs,
live caches, late flushes and trace/legacy normalization must be registered and
retired through capture/trace owners. Governed append-only trace storage and
shared retention roots require native cleanup qualification, never profile SQL
that bypasses their invariants. Managed llama.cpp cache binaries/working copies
have separate catalog/launch ownership and currently unknown claim lineage;
that uncertainty blocks a complete managed-copy guarantee until exact native
review, writer quiescence and qualified retirement. Delivered external requests
being outside recall does not exclude app-owned copies.

Remote delivery is not cleanup acknowledgement. The trusted home server and
compatible active clients must fence replay and acknowledge exact epoch/pages,
retirement and suppression; offline copies stay outstanding. Retired grants
prevent reconnect but cannot certify stopped offline decryption. Preserve
ADR-102's `server_trusted_v1` posture; deleting a client key cannot erase a
trusted server's own copy. Do not promise recall of delivered provider requests,
unmanaged exports, backups or forensic/WAL remnants. Local completion describes
managed live-path retirement, not unqualified cryptographic/physical erasure.

The first future implementation release is unlinked-profile local forgetting
with qualified registered-lineage owner coverage for newly captured V2 claims.
Unknown V1 references cannot acquire fabricated graph coverage. Agent audit/log, capture/trace
coverage and managed-cache uncertainty are explicit eligibility gates. Remote/portable forgetting
needs its own server-qualified release. No scope is offered while a promised
owner lacks coverage/fence enforcement. New memory/evidence producers stay
gated on shipped safeguards, not completed design tasks.

## Alternatives considered

| Alternative | Decision |
| --- | --- |
| Label record tombstones as forgetting | Rejected: Undo, staging and source re-extraction can survive. Keep reversible deletion honest. |
| Delete matching bodies across all sources | Rejected: similarity is not ownership/consent and can destroy user-authored material. |
| Regenerate mixed summaries with an LLM | Rejected for cleanup: introduces cost/disclosure and uncertain redaction; retire the generated artifact as a unit. |
| One cross-database atomic deletion | Rejected as a guarantee: independent databases, keys, files and peers need durable recovery and explicit acknowledgements. |
| Restore before-images after a partial failure | Rejected for immediate forgetting: resurrects retired secrets; keep the fence and recover toward completion. |
| Keep plaintext or public hashes of forgotten text | Rejected: the suppression store would retain the secret or permit guessing. Use exact identities/owner controls, not a semantic memory mirror. |

## Consequences and approval boundary

This design needs versioned shared-core control semantics, encrypted journal
and control-key lifecycle, owner dependency coverage/adapters, publication
coordination, source suppression, managed restore and server/peer qualification.
It intentionally limits automatic cleanup of independently authored material
and reports unknown coverage rather than claiming universal deletion.
TASK-25907.6 changes documentation/tracking only; no destructive operation is
performed. The user endorsed the written design subject to another review; the follow-up
review resolved audit/log coverage and queued-dispatch gaps. ADR-202 accepts
design direction only and remains provisionally numbered against concurrent
branches until integration-time allocation checks.
