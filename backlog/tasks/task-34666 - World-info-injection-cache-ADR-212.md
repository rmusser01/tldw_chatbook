---
id: TASK-34666
title: World-info injection cache ADR-212
status: Done
assignee: []
created_date: 2026-10-07 02:40
updated_date: 2026-10-07 04:32
labels: []
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 2 / F3+F12: every user message re-fetches all attached world books from SQLite re-parses JSON and reprocesses every entry twice plus recompiles keyword regexes per key per message
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 ADR-221 governs the cache contract; unchanged nonempty books reuse their processor without book queries; keyword patterns compile once per entry; each entry is processed once; recursive value-equal entries retain their prior single-injection behavior.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 3 (T3)
ADR required: yes
ADR path: backlog/decisions/221-prompt-injection-cold-start-caches.md
Reason: the existing ADR defines this long-lived prompt cache and its invalidation contract.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
ADR written first as ADR-221 (backlog/decisions/221-prompt-injection-cold-start-caches.md) — the plan's provisional 212 was already taken by the shared-adaptive-pane-shell ADR, renumbered per lessons-backlog-hygiene's ADR-collision rule; the ADR defines the shared cold-start cache contract that Task 4 (chat dictionaries, TASK-34416) adopts. WorldBookManager gained a monotonic `generation` property backed by a counter cell on the CharactersRAGDB instance (shared across per-call manager constructions), bumped on every successful write: book create/update/delete, entry create/update/delete, conversation associate/disassociate, character snapshot attach/detach (single `_write_character_world_books` seam), and imports via their underlying creates; reads and failed writes never bump. The resolver now caches the built WorldInfoProcessor per (conversation_id, character_id) in a lock-guarded OrderedDict LRU capped at 8 conversations, each entry validated against the current generation plus a db-identity weakref (a re-opened database never inherits another connection's processor); the books= parameter (resolve_world_info_injection + apply_world_info_to_message) accepts pre-collected books and bypasses both fetch and cache — the console wiring is deferred per controller ruling. The processor compiles per-key word-boundary patterns once in _process_entry (compiled_primary_keys/compiled_secondary_keys, case-variant-correct: lowered keys for case-insensitive entries, IGNORECASE baked in for regex keys), with a module-level @lru_cache(4096) _compiled_keyword fallback for ad-hoc strings; regex_search accepts precompiled patterns. _make_candidate reuses the already-processed entry (one _process_entry per raw entry per build — was 2x for keyed entries), and recursion dedup became an id() set (identity is exact because recursion returns references to the same self.entries objects). Golden activation fixtures captured on unmodified code (recursive cycle castle→dragon→keep→castle, selective+secondary, case-sensitive, regex, disabled/keyless, duplicate-entries dedup equivalence) pass byte-identical. Spy evidence, 1000-entry unchanged book: send1 20.2ms / 1 book query / 1000 re.compile; send2+ 0.6ms / 0 queries / 0 compiles; post-edit rebuild 8.3ms / 1 query / 0 compiles (module key cache survives rebuilds). New Tests/Character_Chat/test_world_info_injection_cache.py (26 tests). Files: Character_Chat/world_book_manager.py, world_info_processor.py, world_info_regex.py, world_info_resolver.py. Report: .superpowers/sdd/2026-10-06-nonconsole-efficiency-remediation/task-3-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
<!-- SECTION:NOTES:END -->

PR3045 review qualification (TASK-34665): cache slots now include the native
card version, and unversioned embedded content bypasses reuse. Store generation
publication waits for managed transaction completion; native borrowers suppress
reuse until completion. The cap is eight slots per process. Recursive dedup
retains value equality after a regression showed identity dedup emitted duplicate
lore. ADR-221 records the final contract and linked regression evidence.

## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->
<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->

## Renumbering provenance

Renumbered from TASK-34415 during PR3045 latest-dev integration. The older Console control-refresh task was introduced by 58af133eb1 on 2026-10-06 at 09:12 PDT; this world-info task arrived in 265344ad37 at 19:46 PDT. The older owner keeps TASK-34415 under the TASK-19601 rule. TASK-34666 was checked free across fetched refs and all registered worktrees.
