---
id: TASK-544
title: >-
  Resolve duplicate task ids 505-512 between two open batches
status: Done
assignee: []
created_date: '2026-07-24 07:15'
updated_date: '2026-07-25 08:47'
labels:
  - backlog-hygiene
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Every id 505-512 exists TWICE on dev (the backlog CLI resolves by id → all eight are ambiguous). Two batches collided: (a) a web-scraping/egress batch created '2026-07-23 12:00' (Confluence sync, scrape_from_sitemap, recursive_scrape, guarded_fetch, Subscriptions validator, redirect credentials — all To Do), and (b) a model-artifact/STT batch created '2026-07-24 01:01-01:03' (artifact leases/descriptors/downloads/browser, GGUF/ONNX import, STT contracts — task-505 of this batch is **In Progress**).

NOT resolved unilaterally because batch (b) appears to belong to a live session (In Progress task); renumbering under an active branch would just re-introduce dupes on its next merge, and per the standing rule the mover should be the not-started side with its owner aware. Whichever session finishes (or a coordinated cleanup) should renumber ONE side (rule: In Progress/older keeps; per-pair — batch (a) is older for all pairs but batch (b) has the In Progress 505) to the next free ids, updating frontmatter `id:` + any cross-references, then re-run the two-namespace dup-check (python os.listdir scanner, not git-ls-tree|uniq).

Note: this session already resolved its own pairs (503 RAG-SP3 kept / MCP-nav → 542; 519 get_user_data_dir kept / console-branching → 543) in the same PR that files this task.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] Each id 505-512 identifies exactly one task on dev (filename prefix AND frontmatter id namespaces).
- [x] All cross-references (dependencies, prose) updated to the surviving/renumbered ids.
- [x] (Scope broadened at execution time — see Implementation Notes) Every other duplicated id found by a full os.listdir scan of backlog/tasks + backlog/drafts + backlog/archive/tasks is resolved the same way, not just 505-512.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Re-scan backlog/tasks + backlog/drafts + backlog/archive/tasks with a python os.listdir+regex scanner (filename prefix AND frontmatter `id:`) to get the current duplicate set — expected to have grown since filing.
2. For each duplicate id, decide keep vs move using the priority policy: (i) status Done/In Progress keeps, (ii) referenced by an open PR head branch, (iii) older created_date; ties broken by existing `dependencies:` references. Both-Done pairs: keep older, move newer. Cross-check with `gh pr list`/`gh pr view` for the colliding ids.
3. Compute the next free contiguous id block from the max id in use across this branch, origin/dev, and every open PR's head branch.
4. Rename each mover's file, update its frontmatter `id:`, and update every cross-reference (dependencies: arrays and prose mentions) that means the mover specifically, repo-wide, without touching references that mean the keeper.
5. Re-run the scanner and confirm zero duplicates. Update this task and commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The pileup had grown well past 505-512 by execution time: a full scan found **28 duplicated ids** (100, 442, 484, 505-519, 523-532), some triples, spanning five independent batches that all grabbed overlapping id ranges around 2026-07-23/24 without checking origin/dev's true max:

- **(a)** web-scraping/egress batch, created 07-23 12:00, ids 505-512, all To Do.
- **(b)** model-artifact/STT batch, created 07-24 01:01-01:04, ids 505-518; task-505 (leases) is **In Progress** and has an open PR (#846) actively working against it.
- **(c)** a large review/lint-fix sweep, created 07-24 16:50-19:55, all Done, ids 505-532 (skipping 520-522).
- **(d)** SSRF/skill-remote-fetch follow-ups, created 07-24 14:10/14:45, ids 524-532, all To Do.
- One standalone MCP-dispatch fix at id 524, created 07-24 00:00, To Do.
- Two pre-existing, unrelated Done-vs-Done collisions at id 100 (two unrelated completed tasks) and id 519 (two unrelated completed tasks; get_user_data_dir was already the established keeper from an earlier pass noted in this task's Description).
- Two **self-duplicates**: id 442 (a stale pre-drop To Do stub in `backlog/tasks/` alongside its own already-archived, already-Done, identical-content twin in `backlog/archive/tasks/` — see [[persona-macro-terminology]]/task-442 history) and id 484 (a byte-identical To Do stub next to its own Done twin, differing only by a missing "RAG" word in the filename). These two were **deleted** rather than renumbered — renumbering would have resurrected a completed task as a new "open" duplicate, which is the opposite of what this cleanup is for. No cross-references pointed at either stale copy.

Verified `In Progress`/PR-referenced status with `gh pr list --state open` and `gh pr view <n> --json body`: PR #846 ("codex/task-505-macos-evidence") is live against batch (b)'s TASK-505 and blocks on batch (b)'s TASK-507, confirming that pair's keeper independent of the created_date rule.

New ids were assigned from a contiguous block starting at 585 — the true max in use anywhere (this branch, `origin/dev`, and the head branch of every open PR, all capped at 584) — through 619.

### Keep / move table

| Old id | Kept (id unchanged) | Moved (new id) |
|---|---|---|
| 100 | Wire-avatar-upload-in-the-ds-native-character-editor (Done, 2026-06-11, older) | Fix-character-card-import-and-lenient-V2-validation (Done, newer) → **619** |
| 442 | archive/tasks Active-persona-concept-with-user-name-substitution-in-chats (Done) | *(deleted)* stale live-tree To Do stub of the same task |
| 484 | Fix-builtin-RAG-profiles-with-invalid-chunking_method-values (Done) | *(deleted)* stale To Do stub "Fix-builtin-profiles-..." (byte-identical minus filename) |
| 505 | Prove-cross-platform-model-artifact-operation-leases (In Progress, batch b, open PR #846) | Confluence-sync-requests-calls-inside-async-methods (batch a) → **585**; Clear-inherited-lint-findings-in-the-citation-verification-scope (batch c) → **606** |
| 506 | Make-character-expression-schema-test-version-agnostic (Done, batch c) | Image-gen-adopts-shared-egress-module (batch a) → **586**; Qualify-Parakeet-v2-and-v3-INT8-artifacts (batch b) → **593** |
| 507 | Align-dead-event-dispatcher-smoke-test-with-ADR-014 (Done, batch c) | Fix-scrape_from_sitemap-async-coroutine-bug (batch a) → **587**; Build-shared-model-artifact-descriptors-and-lifecycle (batch b) → **594** |
| 508 | Restore-Anthropic-native-tool-payload-pass-through (Done, batch c) | Fix-recursive_scrape-browser-tab-leak (batch a) → **588**; Add-verified-managed-model-downloads-and-recovery (batch b) → **595** |
| 509 | Isolate-scheduled-tasks-storage-in-the-shared-test-app-builder (Done, batch c) | Fix-ConfluenceAuth-test_authentication-egress-bypass (batch a) → **589**; Renovate-the-local-model-artifact-browser (batch b) → **596** |
| 510 | Harden-Anthropic-invalid-tool-diagnostics (Done, batch c) | Document-guarded_fetch-early-status-check-residual (batch a) → **590**; Add-bounded-local-GGUF-artifact-import (batch b) → **597** |
| 511 | Make-local-marks-legacy-migration-fixture-pre-v25 (Done, batch c) | Clean-up-dead-Subscriptions-security-validator (batch a) → **591**; Add-descriptor-backed-local-ONNX-bundle-import (batch b) → **598** |
| 512 | Update-chatbook-dependency-test-for-citation-aware-export (Done, batch c) | Strip-client-session-level-credentials-on-cross-origin-redirect-hops (batch a) → **592**; Introduce-provider-neutral-STT-contracts-and-coordinator (batch b) → **599** |
| 513 | Repair-stale-QuestionAnswerRunner-patch-targets (Done, batch c) | Persist-STT-provenance-and-retry-lineage (batch b) → **600** |
| 514 | Remove-retired-embeddings-selectors-from-CSS-QA-guard (Done, batch c) | Add-generation-fenced-local-STT-executor (batch b) → **601** |
| 515 | Complete-audited-watchlist-runtime-policy-expectations (Done, batch c) | Integrate-Parakeet-ONNX-batch-routing (batch b) → **602** |
| 516 | Complete-audited-skill-read-file-policy-expectations (Done, batch c) | Restore-bounded-Parakeet-ONNX-dictation-buffers (batch b) → **603** |
| 517 | Refresh-server-client-migration-audit-semantic-keys (Done, batch c) | Add-curated-optional-transcribe.cpp-STT-provider (batch b) → **604** |
| 518 | Restore-server-parity-connection-builder-reexports (Done, batch c) | Promote-Parakeet-ONNX-defaults-and-remove-legacy-providers (batch b) → **605** |
| 519 | Fix-get_user_data_dir-import-time-home-freeze-breaking-test-HOME-isolation (Done, older, pre-existing keeper) | Exercise-unauthenticated-GitHub-client-creation-in-its-unit-test (Done, batch c, newer) → **607** |
| 523 | Per-intent-Console-gating-in-Roleplay-inspector-Start-Chat-requires-a-ready-provider (Done, older, PR #837) | Remove-stale-compact-model-bar-settings-patch-from-chat-UI-tests (Done, batch c, newer) → **608** |
| 524 | Honor-optional-command-provider-discovery-in-palette-integration-tests (Done, batch c) | Consolidate-SSRF-layers-skill-remote-fetch-vs-Utils-egress (batch d) → **609**; Restore-MCP-character-chat-dead-dispatch-and-persistence (standalone, 00:00) → **618** |
| 525 | Update-dictionary-send-integration-fixture-for-per-turn-skill-context (Done, batch c) | Reject-non-global-addresses-in-remote-fetch-host-check (batch d) → **610** |
| 526 | Refresh-latest-dev-MCP-first-use-smoke-copy (Done, batch c) | Remote-fetch-SSRF-test-hardening-pins (batch d) → **611** |
| 527 | Use-file-backed-SQLite-stores-in-the-shared-UI-app-harness (Done, batch c) | Remote-install-UX-riders-policy-copy-token-degrade-name-guess (batch d) → **612** |
| 528 | Align-Console-live-work-schedule-tests-with-SchedulesWorkbench (Done, batch c) | Library-skills-import-in-flight-cancel-race (batch d) → **613** |
| 529 | Route-Console-Watchlists-actions-to-the-current-Watchlists-destination (Done, batch c) | Bundle-fidelity-hygiene-batch-784-followups (batch d) → **614** |
| 530 | Align-persisted-conversation-rail-test-with-compressed-subtitles (Done, batch c) | Skill-file-runtime-reachability-followups-814 (batch d) → **615** |
| 531 | Seed-valid-user-turns-in-Console-continuation-action-tests (Done, batch c) | Dollar-mention-riders-multimodal-dead-picker (batch d) → **616** |
| 532 | Complete-destination-action-tooltip-coverage-for-Library-and-Schedules (Done, batch c) | Bring-Roleplay-personas-to-parity-with-tldw_server-personas-module (batch d) → **617** |

### Cross-reference updates

Within the moved files: rewrote every `dependencies:` array entry and every self-referencing prose "TASK-N" mention that meant a mover (batch b's own internal dependency chain 507→508→509→510/511, 512→513→514→515→516, etc. was fully remapped to 594/595/596/597/598/599/600/601/602/603/604/605; task-607 and task-608's own "Scope review confirmed... TASK-519"/"combined TASK-523/524/525" self-references updated to their new ids; the keeper task-525 carries the same "combined TASK-523/..." sentence and was updated in lockstep).

Outside `backlog/tasks/`: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md` (`Related Tasks:` line), `backlog/docs/model-artifact-operation-leases.md` (TASK-507→594 prose + link), `Docs/superpowers/plans/2026-07-23-model-artifact-operation-leases.md` (8× TASK-507→594), `Docs/superpowers/plans/2026-07-23-stt-artifact-runtime-delivery-map.md` (full dependency table + file-link rewrite, 506-518→593-605), `Docs/superpowers/specs/2026-07-23-stt-parakeet-onnx-transcribe-cpp-design.md` (`Related tasks:` line), and `Docs/superpowers/specs/2026-07-24-skills-agent-install-design.md` (task-524/525→609/610). Every candidate cross-reference was read in context first to confirm which batch/meaning it belonged to before editing — several looked ambiguous at a glance (e.g. task-510(c) citing "TASK-506", task-517(c) citing "TASK-515/516", task-528(c) depending on "TASK-527") but on inspection all referred to their own batch-c siblings (the keepers), not the movers, and were correctly left untouched.

Verified with a repo-wide `git grep` for every old id pattern, restricted to files outside the ones already edited: no stray references remained.

### Final scanner output

```
Total top-level task files scanned: 573
Duplicate filename ids: 0
Duplicate frontmatter ids: 0
NO DUPLICATES FOUND
Max filename id: 619, Max frontmatter id: 619
```

Files touched: 35 renamed task files (frontmatter `id:` + dependency-array updates), 2 deleted stale-duplicate task files (442, 484), 1 sibling task file edited in place (task-525's cross-batch prose), and 6 non-task docs (1 ADR, 1 backlog/docs file, 4 Docs/superpowers plan/spec files).
<!-- SECTION:NOTES:END -->
