---
id: TASK-34416
title: Chat-dictionary injection cache
status: Done
created_date: 2026-10-07 02:40
dependencies:
- TASK-34666
updated_date: 2026-10-07 05:47
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Wave 2 / F4: dictionary entries are re-loaded from DB re-instantiated and regex-recompiled on every send and twice per turn
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Store generation counters added,Second send with unchanged dictionaries does no DB loads or regex compiles,Replacement output byte-identical on golden fixture,Console-seam double collection noted for coordination
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
See Docs/superpowers/plans/2026-10-06-nonconsole-efficiency-remediation.md Task 4 (T4)
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:IMPLEMENTATION_NOTES:BEGIN -->
Implemented the ADR-221 dictionary half: generation counters on the three Chat_Dictionary_Lib write choke points (save/update/delete, post-commit rowcount-guarded) plus the attachment and embedded-snapshot writers, mirrored on the server service's 11 remote mutators; bundle cache (lock-guarded OrderedDict, LRU 8, generation read before fetch, weakref db-identity check) returning ready ChatDictionary instances; compile-once via instance-cached _compiled_key behind the key property (is_regex materializes on read, restoring the eager contract for all consumers) + module-level lru_cache whole-word patterns + re.compile hoisted out of the max_replacements loop with identical ReDoS fail-closed semantics; interface-only entries= seam on apply_active_chatdicts_to_text (console seam ruled out of scope). Spy evidence: 3 dictionaries/300 entries — send1 3 loads/300 from_dict/3 JSON parses/375 compiles -> send2 0/0/0/0. Golden replacement fixture byte-identical. Sanctioned semantic change pinned and documented: cached instances persist last_triggered across sends within a generation, making timed-effect cooldowns/delays effective as apply_timed_effects documents (lifecycle test: fires -> suppressed in cooldown -> fires post-cooldown via mocked clock -> resets on generation bump). Files: Chat_Dictionary_Lib.py, local/server_chat_dictionary_service.py, Tests/Character_Chat/test_chat_dictionary_injection_cache.py (32 tests). Report: .superpowers/sdd/task-4-report.md
<!-- SECTION:IMPLEMENTATION_NOTES:END -->
## Final Summary

<!-- SECTION:FINAL_SUMMARY:BEGIN -->

<!-- SECTION:FINAL_SUMMARY:END -->

## Definition of Done
<!-- DOD:BEGIN -->
<!-- DOD:END -->
