---
id: TASK-216
title: Clipboard image paste and drag-drop attach in Console composer
status: Done
assignee: ['@claude']
created_date: '2026-07-13 09:30'
labels:
  - console
  - ux
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Console attachments Phase 1 (PR #621) supports the file picker only. Add clipboard image paste (terminal capability permitting) and drag-drop of files onto the composer, both routing through the existing attachment_core.process_attachment_path pipeline and the per-session pending-attachment state.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 An OS-clipboard image is staged as the pending attachment with the composer indicator via the explicit Alt+V grab (terminals deliver no paste event for image clipboards; approved reinterpretation)
- [x] #2 Dropping a supported file onto the terminal (which pastes its path) behaves like picking it in the file dialog — inline vs attachment routing preserved; prose containing paths stays text
- [x] #3 Unsupported/oversized paste/drop shows the same validation toasts as the picker path; empty/unavailable clipboards get honest dedicated toasts
<!-- AC:END -->

## Implementation Plan

1. Pure module Chat/console_paste_attach.py (path extraction, attachability, ImageGrab wrapper). 2. attachment_core.process_attachment_bytes. 3. on_paste interception after existing guards. 4. Alt+V screen binding + off-loop grab worker. 5. Verification + live QA + user gate. (Docs/superpowers/plans/2026-07-13-console-paste-dnd-attach.md)

## Implementation Notes

Terminal reality reshaped the ACs (user-approved): drag-drop = path-paste interception (every line of the paste must decode as a path token; first-of-N attaches on multi-drop); clipboard images = explicit Alt+V (footer-visible, mirrors alt+m; the planned palette entry became the footer hint — the ^p palette is a custom modal) reading PIL.ImageGrab off-loop in the dedicated console-clipboard-grab group, with honest empty/unavailable toasts (Linux unsupported). Clipboard bytes run the identical validate/resize pipeline via attachment_core.process_attachment_bytes (no temp files). Auto-attach gates on existence + home-root + shared picker filters BEFORE any read; prose/relative/out-of-root/unsupported pastes stay text. Final-review hardening: unencodable clipboard modes (CMYK) degrade via RGB conversion instead of crashing; Alt+V honors the setup-modal inert invariant. Known riding items: first-of-N toast precedes processing; file://host collapse; paths-kind runs under the clipboard group (TASK-217 reworks the slot); tiff/svg filter drift inherits TASK-222. Verification: 1061-test gate green, legacy image suites untouched, 6/6 live captures (Docs/superpowers/qa/console-paste-dnd-2026-07/, transport deviations disclosed). Key files: Chat/console_paste_attach.py, Chat/attachment_core.py, UI/Screens/chat_screen.py.
