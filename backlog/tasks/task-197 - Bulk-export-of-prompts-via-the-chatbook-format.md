---
id: TASK-197
title: Bulk export of prompts via the chatbook format
status: Done
assignee: []
created_date: '2026-07-12 13:16'
updated_date: '2026-08-12 17:30'
labels:
  - library
  - prompts
  - enhancement
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Deferred from the 2026-07-12 Library Prompts spec: v1 ships per-prompt .md export (import-parser round-trip). Bulk export should ride the Library chatbook export seam, which requires prompts to be representable in the chatbook format first.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Chatbook prompt records preserve the portable content of every active current Prompt and Recipe: name, author, details, separate System and User lanes, canonical keywords, artifact type, prompt format, schema version, and stored definition.
- [x] #2 Prompt records exclude local database identity and lifecycle state: row IDs, UUIDs, client IDs, optimistic/sync versions, source timestamp values, soft-deleted rows, source retained history, and collection memberships are not exported or restored; required v1 manifest timestamp slots remain null and import creates only ordinary destination-owned lifecycle state.
- [x] #3 The Chatbook importer accepts both the new versioned lossless prompt record and the legacy single-`content` prompt payload; unknown record versions and invalid records fail closed without partially mutating that Prompt.
- [x] #4 Library Export exposes a local-only Prompts scope from the Prompt list and includes Prompts in Everything; counts and selections come from fresh uncapped active-ID database queries rather than the rendered Prompt page.
- [x] #5 A Prompt-scoped export uses the existing Library export canvas, destination, progress, cancellation, Retry, and overwrite behavior; a selected Prompt that disappears or cannot be represented aborts the archive rather than producing a silent partial success.
- [x] #6 A Chatbook exported from a real source database imports into a fresh real database with portable Prompt and Recipe content unchanged, including multiline and Unicode lanes, literal markup-looking text, keywords, structured-v2 definitions, and compatibility-only stored definitions.
- [x] #7 Automated database, codec, Chatbook, Library scope, service, and mounted Textual tests cover legacy compatibility, invalid versions, more than one browse page, deleted exclusion, Everything scope, server-mode refusal, stale/error paths, and narrow-toolbar geometry; user documentation explains the final bulk workflow and exclusions.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
ADR required: yes
ADR path: backlog/decisions/057-portable-chatbook-prompt-records.md
Reason: TASK-197 defines a durable portable Prompt schema, backward-compatible dispatch, privacy/identity exclusions, and an all-or-nothing cross-module export boundary.

Detailed plan: Docs/superpowers/plans/2026-08-12-task-197-bulk-prompt-chatbook-export.md

1. Build the strict portable Prompt-record codec RED-first.
2. Add coherent privacy-safe Prompt export snapshots and uncapped active IDs in real SQLite.
3. Make Chatbook Prompt collection/import lossless, archive-local, backward-compatible, and all-or-nothing.
4. Extend Library export counts, selections, labels, and real round trips with Prompts.
5. Restore Prompt Export… through the existing canvas and verify local-only lifecycle, focus, privacy, and real compositor geometry.
6. Run affected regression/static/security gates, update both user guides, obtain independent review, and complete task hygiene.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented ADR-057 portable Chatbook Prompt records end to end: strict versioned codec with legacy import, coherent privacy-safe SQLite snapshots, archive-local all-or-nothing Chatbook collection/import, local-only Library Prompt scope and Everything integration, and mounted Prompt-list export lifecycle. Updated both Library guides and diagnostic inventory. Verification: integrated matrix 450 passed; proportionate affected gate 688 passed/1 skipped; full Prompt canvas 214 passed; final privacy/database boundary 72 passed; 64x24 and 120x40 compositor harness 2 passed; Ruff/compile/mypy/CSS/diff checks green. Independent review found and verified the final keyword-diagnostic redaction. No dependency, license, schema-migration, or new generalized lesson change.
<!-- SECTION:NOTES:END -->
