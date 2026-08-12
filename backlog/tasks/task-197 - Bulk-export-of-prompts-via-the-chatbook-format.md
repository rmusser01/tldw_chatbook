---
id: TASK-197
title: Bulk export of prompts via the chatbook format
status: In Progress
assignee: []
created_date: '2026-07-12 13:16'
updated_date: '2026-08-12 15:00'
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
- [ ] #1 Chatbook prompt records preserve the portable content of every active current Prompt and Recipe: name, author, details, separate System and User lanes, canonical keywords, artifact type, prompt format, schema version, and stored definition.
- [ ] #2 Prompt records exclude local database identity and lifecycle state: row IDs, UUIDs, client IDs, optimistic/sync versions, timestamps, soft-deleted rows, retained history, and collection memberships are not exported or restored.
- [ ] #3 The Chatbook importer accepts both the new versioned lossless prompt record and the legacy single-`content` prompt payload; unknown record versions and invalid records fail closed without partially mutating that Prompt.
- [ ] #4 Library Export exposes a local-only Prompts scope from the Prompt list and includes Prompts in Everything; counts and selections come from fresh uncapped active-ID database queries rather than the rendered Prompt page.
- [ ] #5 A Prompt-scoped export uses the existing Library export canvas, destination, progress, cancellation, Retry, and overwrite behavior; a selected Prompt that disappears or cannot be represented aborts the archive rather than producing a silent partial success.
- [ ] #6 A Chatbook exported from a real source database imports into a fresh real database with portable Prompt and Recipe content unchanged, including multiline and Unicode lanes, literal markup-looking text, keywords, structured-v2 definitions, and compatibility-only stored definitions.
- [ ] #7 Automated database, codec, Chatbook, Library scope, service, and mounted Textual tests cover legacy compatibility, invalid versions, more than one browse page, deleted exclusion, Everything scope, server-mode refusal, stale/error paths, and narrow-toolbar geometry; user documentation explains the final bulk workflow and exclusions.
<!-- AC:END -->
