---
id: TASK-39111
title: "The backup producer census cannot see tempfile creation or pydub and ffmpeg writers, so 22 producing symbols have no reviewed row"
status: To Do
created_date: 2026-11-07 02:03
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
- [ ] #1 Every tempfile creation call in production has a reviewed census row, and a synthetic `tempfile.mkdtemp()` in a fresh symbol turns the census red (recorded)
- [ ] #2 pydub `export` and subprocess output writers either have reviewed rows or are each listed in the doc as owner-accepted known limits; the owner decides which, and the decision is recorded
- [ ] #3 The inventory test stays green on its PR lane, with its serial time before and after recorded
<!-- AC:END -->
