---
id: TASK-40310
title: "The restore journal re-reads and re-validates every record on each append and flush, so an acknowledged Tools restore takes 80-140 s"
status: Done
assignee:
- '@claude'
created_date: 2026-11-12 19:36
updated_date: 2026-11-13 08:22
labels:
- follow-up
- backup
- recovery
- performance
priority: medium
references:
- tldw_chatbook/Backup_Recovery/journal.py
- tldw_chatbook/Backup_Recovery/publication.py
- tldw_chatbook/Backup_Recovery/rollback_credentials.py
dependencies:
- TASK-40132
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 13 of the 2026-10-23 burn-down (plugins stream, TASK-40132's end-to-end run, sampled with `faulthandler`). Checked by reading the code at the round-13 stack top 8b633d0101, 2026-11-12; line numbers are that tree's.

**Why.** A Tools restore the user acknowledged takes minutes, and the plugin end-to-end needs a 480 s budget. The cost grows faster than linearly with the record count, because every append re-reads history.

**Evidence.** `Backup_Recovery/journal.py:1605-1620` `_records` lists the directory and re-reads and re-parses every record through `parse_event_records` (`:1494-1540`: pydantic validation, the hash chain, `_validate` against prior records). `_append` (`:1698-1700`) calls `_records` on every append; `_flush_records` (`:1687-1696`) re-reads, compares and fsyncs every record. `_flush_records` runs at `publication.py:687`, `:2012`, `:2337`, `:2526`, `:2736`, `:2761`, `:2857` and `rollback_credentials.py:318`, `:372`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Append and flush validate only new records against history that is cached and checked for identity; a tamper or reorder test still fails
- [x] #2 The acknowledged-restore phase of the plugin end-to-end is measurably faster on the same host, with before and after times recorded
<!-- AC:END -->

## Implementation Plan

1. Measure the acknowledged-restore phase of Tests/Plugins/test_recovery.py's end-to-end at base (same host, load sampled).
2. Journal keeps the members it last read and the events they parsed to; _records still reads and checks every file, but reparses only from the first member that differs (parsing is a pure function of the ordered members).
3. _flush_records skips a record this journal already flushed while its stat identity and parsed record are unchanged.
4. Pin with a test that counts _validate and _flush_record calls (red at base), plus tamper/reorder/truncate tests on a warm journal and a positive control.
5. Re-measure the same phase after the change and record both times with load.

## Implementation Notes

- **Approach.** `Journal` keeps the ordered `(name, value)` members it last read and the events they parsed to. `_records` still opens and checks every record file through `_read` on every call (mode, link count, owner, size), and compares each with the member it read last time; parsing restarts at the first member that differs (`_extend_records`, which `parse_event_records` now calls with an empty history). Parsing is a pure function of the ordered members, so this returns exactly what a full parse would, including the same first error. `_flush_records` skips a record this journal already flushed while both its stat identity (dev, inode, mode, size, mtime, ctime) and its parsed record are unchanged; a new file, even byte-identical, is flushed again.
- **Review r14bk-rev1 fix.** The cache first compared members with dict equality, so a warm journal accepted a rewrite of `"device":1` to `true` or `"sequence":3` to `3.0` (equal in Python, refused by the strict parse) where a fresh one refused it. Members are now compared as their JSON text. `test_cached_journal_history_still_refuses_changed_records[retyped]` was red at 683ce34529 (`DID NOT RAISE`) and is green; that test now also requires a fresh `Journal(...)._records` to refuse each damage case and, undamaged, to equal the warm journal's records. Both pins also run on PRs now, as single nodes in `architecture-guards` (the rest of the file runs in the dev-push job, TASK-40307).
- **Decisions.** Identity is checked by content (JSON text), not by stat alone: a record rewritten within one coarse-timestamp tick keeps its stat identity on Linux, and reading the small files costs far less than the validation it now skips. The double `_Prepared.model_validate` per progress record in `_validate` is left as it is (each record is now validated once or twice, not once per later append).
- **Evidence.** `test_append_and_flush_handle_each_record_once` was red at base 26b2e677dd: 1085 `_validate` calls for 32 records (`assert 1085 <= 64`); green after (each record validated at most twice, each flushed once, and a replaced copy re-flushed). `test_cached_journal_history_still_refuses_changed_records[tampered|reordered|truncated]` refuse on a warm journal (both `_flush_records` and the next append), with an undamaged positive control.
- **Measured** (same host, `Tests/Plugins/test_recovery.py::test_plugin_backup_restores_in_place_and_waits_for_authority_recovery`, `-n 0 -s`, child timeouts raised locally for the before run only; acknowledged restore = PHASE `aborted` to `restored`): before, at load 56-114: first 577 s, again 632 s, whole test 1427 s (the unmodified test times out its `first` child at 360 s). After, at load 60-75: first 91 s, again 92 s, whole test 358 s. A cProfile of the after `first` restore: journal work is about 30 of 122 profiled seconds (`_records` 17 s over 388 calls, mostly reads; `_flush_record` 181 calls, one per record); the rest is outside the journal (`models.py` StorageItem immutability checks 26 s, `publication._rollback_sources` 26 s, `_sidecar_main` 24 s).
- **Covering runs.** `Tests/Backup_Recovery/test_publication_crashes.py` 48 passed; journal callers (`test_large_recovery_records`, `test_detached_evidence`, `test_pending_directory_durability`, `test_rollback_credentials`, `test_replacement_abort`, `test_retirement_only_rollback`, `test_replacement_recovery`, `test_publication_finalization`) 220 passed, 1 failed: `test_large_collection_completes_actual_isolated_publication` times out its 180 s child at load 53-65 on base 26b2e677dd too (same command on a git-archive tree of the base).
- **Files.** `tldw_chatbook/Backup_Recovery/journal.py`, `Tests/Backup_Recovery/test_publication_crashes.py`.
