---
id: TASK-39111
title: "The backup producer census cannot see tempfile creation or pydub and ffmpeg writers, so 22 producing symbols have no reviewed row"
status: Done
assignee: ['@claude']
created_date: 2026-11-07 02:03
updated_date: 2026-11-10 21:44
labels:
- follow-up
- backup
- architecture
- tests
priority: low
references:
- Tests/Architecture/test_backup_owner_inventory.py
- backlog/docs/backup-recovery-owner-inventory.md
- tldw_chatbook/TTS/audio_service.py
dependencies:
- TASK-39053
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Found in round 11 of the 2026-10-23 burn-down; checked on dev 2684e5453f, 2026-11-07, with the census from round-11 branch fu/r11-backupinv (TASK-39053), whose doc lists this as a known limit. Line numbers are dev's.

**tempfile.** A probe ran the branch's own census over dev with the tempfile constructors added to its vocabulary:
- That yields 86 tempfile creation rows: NamedTemporaryFile 24, mkstemp 23, mkdtemp 21, TemporaryDirectory 14, TemporaryFile 4.
- 22 of those symbols have no other row, so the census does not know they produce files at all. Examples: `ChatbookImporter._create_extract_dir`, `ConsoleScratchSpaceManager.snapshot`, `ArtifactShareController.start_share`, `SecureTempFileManager.create_temp_dir`, `skill_script_runner.run_script_subprocess`.

**pydub and ffmpeg.**
- `AudioService.create_m4b_with_chapters` has no row. It writes a tempfile (`TTS/audio_service.py:476`), a pydub `combined.export` (`:480`) and an ffmpeg output through `subprocess.run`.
- Matching `export` by name would also catch unrelated Prompts, Evals and OPML exporters, so a name alone over-counts.

A backup owner review that cannot see these writers cannot classify them.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Every tempfile creation call in production has a reviewed census row, and a synthetic `tempfile.mkdtemp()` in a fresh symbol turns the census red (recorded)
- [x] #2 pydub `export` and subprocess output writers either have reviewed rows or are each listed in the doc as owner-accepted known limits; the owner decides which, and the decision is recorded
- [x] #3 The inventory test stays green on its PR lane, with its serial time before and after recorded
<!-- AC:END -->

## Implementation Plan

1. Add the tempfile constructors (NamedTemporaryFile, TemporaryFile, SpooledTemporaryFile, TemporaryDirectory, mkstemp, mkdtemp) to the census vocabulary, with a unit test red first.
2. Recompute the drift with the test's own helpers; give every new tempfile row a reviewed classification (a call in an already-reviewed symbol takes that symbol's disposition; new symbols are read in source: system-temp scratch is process_artifact, profile-directory staging takes its owner's cohort, else unsupported).
3. Record the synthetic `tempfile.mkdtemp()` negative control in a fresh production symbol.
4. pydub export / subprocess outputs: survey the sites; AC#2 needs the owner's choice (reviewed rows or listed limits).
5. Record the inventory test's serial time before and after.

## Implementation Notes

The census in `Tests/Architecture/test_backup_owner_inventory.py` now counts tempfile creation: `NamedTemporaryFile`, `TemporaryFile`, `SpooledTemporaryFile`, `TemporaryDirectory`, `mkstemp` and `mkdtemp`, by name, through `from tempfile import ... as ...` aliases, and passed by reference. At base ae9f9e8d21 that is 86 rows (NamedTemporaryFile 24, mkstemp 23, mkdtemp 21, TemporaryDirectory 14, TemporaryFile 4), 22 of them in symbols with no other row, matching the filing. Recomputed with the test's own helpers, the whole branch adds 99 rows (with TASK-39112, TASK-39115 and the pydub row below) and changes or removes none.

- Classification (AC#1), in the doc's new "Census vocabulary extension" section: the doc's rules in order. 74 rows are new calls in reviewed symbols and take that symbol's disposition (rule 2); the two mixed symbols take their scratch rows' disposition. The 24 lone rows were each read in source: system-temp scratch (no `dir=`) is `process_artifact` / `process`, as the existing `VideoStore._atomic_publish` and `CodeExecutionRunner._execute_code` rows are; child-process log/output capture files are `process-transport`; scratch inside a reviewed module's own directory takes that module's cohort (Chatbooks import extraction, File Notes git proof/hook directories); `Utils/secure_temp_files` is a generic boundary. One row is in a Plugins symbol (`SourceService.commit_import.operation`, `unsupported` / `miscellaneous`, left for TASK-40132).
- Red/green: `test_census_counts_tempfile_creation` failed on the base scanner with `AssertionError: NamedTemporaryFile` (`assert 0 == 1`) and passes now.
- Synthetic control (AC#1): a temporary `def _r13b_negative_control(): return tempfile.mkdtemp()` appended to `tldw_chatbook/Utils/secure_temp_files.py` turned `test_every_persistence_source_symbol_has_reviewed_owner_row` red: "1 added, 0 gone, 0 changed ... added | tldw_chatbook/Utils/secure_temp_files.py | _r13b_negative_control | mkdtemp | 1 |". Reverted.
- Timing (AC#3), serial, whole file, through the scratch wrapper: before 19 passed in 41.17 s (the census node 38.09 s; load average 15-24); after the tempfile/os/renameat2 work 23 passed in 50.62 s (census node 37.54 s; load 21-31); after the pydub change 24 passed in 60.25 s (census node 46.02 s, the Actor_Packs node 11.68 s; load 46-56). Wall time here follows the load, so the code cost was measured as CPU time of `production_census()` in one process, base and head scanners alternating three times: base 27.2-29.0 s / head 29.1-30.1 s before the pydub change, base 30.0-30.6 s / head 30.6-31.8 s after it (at most +6%). A text prefilter keeps the os-helper walk (TASK-39112) to modules that have a `return <os name>` (without it head measured about 58 s wall against base 36-50 s), and the pydub check only walks imports when the source contains "pydub". The lane step's per-test timeout is 180 s; the file is in derived-artifacts.yml's lane step.
- AC#2 (owner decision 2026-11-10, AskUserQuestion): "pydub rows, subprocess as limit (Recommended)". The option read: "Count pydub export in the modules that import pydub (2 file-writing rows to review) and list subprocess outputs as one documented accepted limit. Small, and it catches new pydub writers." Implemented: in a module that imports pydub (`import pydub*` or `from pydub* import`, anywhere in the module), `<x>.export(...)` counts as `pydub.export`, unless its target (first argument or `out_f=`) is a `BytesIO(...)` call or a name bound to one in the same lexical scope; that is how the five in-memory exports (`audio_service` :342/:344, `audio_stitch` :180, `audiobook_generator` :772/:856) stay out. The same-named Prompts, Evals, OPML, meetings and STTS-profile exporters stay out because their modules do not import pydub. Result: one row, `AudioService.create_m4b_with_chapters | pydub.export | 2` (the system-temp `.m4a` at :479 and the chapterless fallback M4B at `output_path`, :521), reviewed as `unsupported` / `tts` because the fallback writes the M4B `AudioBookGenerator._add_metadata` asks for and the generator's reviewed rows are `unsupported` / `tts`. The doc's census vocabulary section lists subprocess outputs as one accepted limit with the survey: 71 `subprocess.run/Popen/call/check_call/check_output` and `asyncio.create_subprocess_exec/_shell` calls in 58 files at ae9f9e8d21; writers found are ffmpeg (the M4B, the parakeet/transcribe.cpp temporary WAVs, `stts_events` :2641) and the stable-diffusion.cpp image; yt-dlp runs as a subprocess only to resolve a stream URL (`Media_Playback/stream_resolve`), and the Local_Ingestion downloads use the yt-dlp library, also uncounted. Red/green: `test_census_counts_pydub_export_only_where_pydub_is_imported` failed on the previous scanner with `assert 0 == 2` and passes now; it also asserts that a BytesIO target and an `export` in a module that does not import pydub are not counted.
- Files: `Tests/Architecture/test_backup_owner_inventory.py`, `backlog/docs/backup-recovery-owner-inventory.md`.

**Review r13b-rev1 fixes (2026-11-10).**
- M2 (pydub): the BytesIO lookup now reads only the innermost scope, and a name counts as a buffer only if every binding of it in that scope is `BytesIO(...)`. A function parameter that shares a module-level buffer's name, or a name rebound from a BytesIO to a path, is a file target again and counted. The added cases in `test_census_counts_pydub_export_only_where_pydub_is_imported` were red (`assert 0 == 1`) and pass. Production rows are unchanged (recompute: 0 added, 0 gone, 0 changed).
- AC#3 timing after the fixes: serial, whole file, 25 passed in 95.94 s with the census node at 70.41 s (load 47-80); the earlier pydub-change run was 24 passed in 60.25 s (load 46-56). Wall time follows the load; paired CPU time is base 29.6-31.6 s and head 31.9-33.4 s, about +5-8%. The lane's per-test timeout is 180 s.

**Review r13b-rev2 fix (2026-11-10).** M2: the pydub buffer rule now sees every binding of a name, not only `=`, `: T =` and `:=`. Each `Name` stored in the scope, other than the target of `x = BytesIO(...)`, and each `except ... as` name makes the name a file target, which covers loop, `with ... as`, unpacking and comprehension targets. The new `looped`, `opened`, `unpacked` and `excepted` cases were red (`AssertionError: looped`, `assert 0 == 1`) and pass. Match-case captures, import aliases and `global`/`nonlocal` rebinding are not seen; the scanner's `ponytail:` note and the doc say so. Production rows unchanged (0 added, 0 gone, 0 changed). Serial run of the inventory file plus the laned detached node: 27 passed in 80.86 s, census node 44.94 s (load 30-38).
