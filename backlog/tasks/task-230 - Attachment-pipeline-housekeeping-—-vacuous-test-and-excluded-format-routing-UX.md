---
id: TASK-230
title: Attachment pipeline housekeeping — vacuous test and excluded-format routing UX
status: In Progress
assignee:
  - '@claude'
created_date: '2026-07-14 10:30'
updated_date: '2026-07-16 14:35'
labels:
  - chat
  - tests
  - ux
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Two small riders from TASK-222's reviews, deferred because existing tests were read-only on that branch. (1) `Tests/Chat/test_attachment_core.py::test_process_attachment_bytes_falls_back_to_original_on_processing_failure` now passes vacuously: it monkeypatches `ChatImageHandler._process_image_data`, which `process_attachment_bytes` no longer calls (it calls `prepare_image_payload`); coverage moved to `test_process_attachment_bytes_fallback_probes_mime` but the stale monkeypatch misleads readers — repoint or delete it. (2) UX decision: with a narrowed `[chat.images].supported_formats`, an excluded image extension picked via the pickers' "All Files" row now routes past `ImageFileHandler.can_handle` to `DefaultFileHandler`, which inserts an inline `[File: x.tiff (…) - image/tiff]` placeholder instead of the legacy "Unsupported image format" error toast (TASK-222 final review, finding M1 — rides documented). Decide whether image-looking-but-excluded extensions should get an explicit rejection (e.g. `can_handle` claims effective ∪ DEFAULT_SUPPORTED_IMAGE_FORMATS and the pipeline rejects per the effective list) or keep the generic-file fallthrough, and implement the choice with tests.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 The bytes-fallback test exercises the real fault-injection seam (prepare_image_payload) or is removed in favor of the superseding test
- [ ] #2 The excluded-image-extension pick behavior is an explicit, tested decision (rejection copy or documented generic-file fallthrough), not an accident of registry ordering
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Delete the vacuous bytes-fallback test (superseded by test_process_attachment_bytes_fallback_probes_mime — reviewer-endorsed removal)\n2. Registry-level rejection: file_handler_registry.process_file raises the unsupported-format ValueError (naming the effective list + [chat.images].supported_formats) when a file would fall to DefaultFileHandler but its extension is a known image format excluded by config — restores the legacy error toast instead of a silent [File: …] placeholder; can_handle semantics (routing == effective formats) untouched\n3. RED tests: excluded-known-image ext → ValueError; unknown ext (.xyz) → still DefaultFileHandler placeholder; effective ext → ImageFileHandler (regression)\n4. Sweep; PR
<!-- SECTION:PLAN:END -->
