---
id: TASK-40310
title: "The restore journal re-reads and re-validates every record on each append and flush, so an acknowledged Tools restore takes 80-140 s"
status: To Do
created_date: 2026-11-12 19:36
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
- [ ] #1 Append and flush validate only new records against history that is cached and checked for identity; a tamper or reorder test still fails
- [ ] #2 The acknowledged-restore phase of the plugin end-to-end is measurably faster on the same host, with before and after times recorded
<!-- AC:END -->
