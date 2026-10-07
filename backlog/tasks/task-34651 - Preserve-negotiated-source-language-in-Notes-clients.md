---
id: TASK-34651
title: Preserve negotiated source language in Notes clients
status: Done
created_date: 2026-10-07 03:07
dependencies:
- TASK-34571
assignee:
- '@codex'
labels:
- notes
- structured-editing
documentation:
- backlog/decisions/194-structured-note-language-and-local-editor-validation.md
updated_date: 2026-10-07 06:50
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Keep language intent and exact source safe when clients and servers support different Notes versions.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Clients require explicit authenticated language capability and isolate or invalidate capability state on server/profile changes.
- [x] #2 Old or unavailable peers leave actionable durable pending intent; a body-only acknowledgement never clears pending language.
- [x] #3 Targeted compatibility tests and scoped static/security evidence cover publication rechecks and account isolation.
- [x] #4 Relevant governing ADR is assessed/linked and repo-local completion notes name actual evidence and remaining limitations.
- [x] #5 Clear Notes round trips and compatible private builder/receiver fixtures preserve exact source, language and lineage; unavailable private server transport retains ciphertext-backed local pending intent.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/194-structured-note-language-and-local-editor-validation.md (C); Docs/ADR/065-portable-note-language-and-negotiated-structured-editing.md (S), existing decisions
Reason: implement approved negotiated client contracts and private confidentiality/pending behavior without a new transport.
1. Read task4 brief, repo-local tasks/instructions/lessons and governing ADRs; trace current clear/private builders, receivers, outbox acknowledgements and authenticated Notes client capability paths.
2. Add failing real SQLite builder/receiver, compatibility/pending/restart and capability/profile controls before edits.
3. Preserve complete source/language using advertised compatible versions, old omission preservation and explicit null; keep private language inside ciphertext and reject clear leakage.
4. Retain note-scoped durable pending language/reason when complete/private support is unavailable; body-only acknowledgements cannot erase intent. Recheck authenticated capability before publication and isolate server/profile state.
5. Run named new and affected C/U selections plus scoped static/security checks; frontend typecheck for shared-client changes. Preserve inherited debt with immutable-base comparison; no full suite.
6. Document evidence/limitations in task notes and ADR pointers, self-review and commit explicit client files separately. Keep In Progress pending independent controller review.
Scope correction: actual publication owners also include Notes/notes_scope_service.py, Sync_Interop/notes_organization_sync_service.py (immutable note_sync_publication_intents and drain), local_first_sync_service.py and sync_state_repository.py. Reuse these existing owners, without a new table/state mechanism. Replace the temporary F2 local active-Sync language veto with durable captured language and publication gating, including restore; local save remains usable. Preserve the exact captured source/language/version, never fill an older intent from a newer row. Existing pending/error state must survive restart, unsupported capability and body-only/unrelated acknowledgements. Add targeted real-store/restart/re-ack controls for these seams.
F4 owner correction continuation: ServerSyncService supplies an authenticated get_v2_capabilities wrapper. Probe and push use one captured operation authority/client and refuse changed current context before mutation/acknowledgement; do not repurpose character-specific authority APIs. Direct local save/restore capture exact immutable note publication intents within the canonical Notes transaction using the existing intent table/drain, so failed post-commit enqueue/crash retains intent. Cover rollback, failed enqueue/restart and same-scope re-ack. No new table or general runtime owner.
Complete Notes acknowledgement uses the existing protocol: verified compatible complete-version publication under the same authority, then matching accepted client_envelope_id, domain/object identity, valid server sequence/cursor and resulting object revision under existing lineage checks, and apply_status=applied. Accepted ID binds the immutable submitted envelope. Do not invent hash/pair echo fields or depend on own excluded-device pull. Insufficient/id-only/body-only/wrong-object/failed-apply/stale-authority acknowledgement retains pending language. Cover actual F3 response shape with real-store positive and refusal controls; other domain ack contracts stay unchanged.
F4 existing clear receiver continuation: LocalFirstSyncService may accept an explicitly scoped existing NotesMirror and forwards dataset_id to SyncEnvelopeApplier; absent/unscoped mirror refuses clear Notes actionably, without a new global mirror storage owner. Real service clear pull to real SQLite and authority isolation must be verified. Existing NotesM1SyncFlow direct push is also an actual clear entry point and must obey negotiated-publication/current-authority guards. Shared dataset/object identifiers cannot reuse another authority mirror/metadata.
F4 local immutable intent scope proof: capture exact server_profile/principal/workspace at the originating transaction in existing local payload metadata and, for new organization receipts where supplied, existing _request metadata. Compare before drain copies an intent into the fully scoped outbox. Explicit null principal/workspace differs from absent proof. Never infer historical scope from current profile/finalizer, synthesize null language for an omitted old field, or refill old snapshot from current row. Old unproved affected intents/receipts remain actionable pending; original _request stores only profile/dataset and cannot prove missing fields. No private clear-envelope disclosure, schema, transport or new state owner. Update existing governing ADR scope notes before code. Test repeated dataset/object with changed principal/workspace, restart and missing-vs-null proof.
<!-- SECTION:PLAN:END -->
## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented complete source/language capture and negotiation through existing Notes clients, immutable intents and scoped receivers. Authenticated provider-owned client, principal and dataset/workspace proof remains current through publication/application/acknowledgement; expected proof-read failures persist bounded pending state. New receipts bind supplied origin in immutable request metadata; historical unproved origins remain pending. Complete backlinks are validated/forwarded and unsupported canonical values refuse before writes. ADR194 and ADR065 govern the contract.
Verification: original compatibility and actual-store coverage plus final affected-owner196 and focused seam3; fix2 final authority/preflight selection156 passed, genuine RED8fail then GREEN14. Scoped Ruff/Bandit introduced none; formatter debt unchanged; git diff --check passed. Fresh task review and both scoped fix reviews complete, latest verdict clean.
Limits: actual server private transport unavailable (ciphertext-backed intent retained); non-null canonical backlinks unsupported; actual Library compound-save caller supplies no remote origin, so no production receipt binding claimed. UI pending-reason consumption and application materializer qualification belong to I2/I3/I5.
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
Exact source and language remain bound to durable scoped intent. Final authority/preflight156 passed and fresh scoped review cleared both remaining defects. Unsupported/unproved paths retain pending state; later UI qualification remains explicit.
<!-- SECTION:FINAL_SUMMARY:END -->
## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
