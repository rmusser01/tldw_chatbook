---
id: TASK-222
title: Drive attachment filters and image caps from chat.images config
status: Done
assignee: ['@claude']
created_date: '2026-07-13 11:15'
labels:
  - chat
  - config
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
PR #621's spec claims the attachment format allowlist and caps read from [chat.images] so the picker and pipeline cannot drift — but ATTACHMENT_FILTER_SPECS (attachment_core.py) and ChatImageHandler constants (SUPPORTED_FORMATS, 10 MB cap, 2048 px resize) are hardcoded, ignoring the existing supported_formats/max_size_mb/resize_max_dimension config keys. Visible drift exists today: the picker's Image Files filter offers .tiff/.tif/.svg which the pipeline rejects with an error toast (behavior inherited from legacy, no regression). Wire both consumers to the config keys so the no-drift-by-construction claim becomes true.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Picker image filter and ChatImageHandler format allowlist derive from [chat.images].supported_formats (no tiff/svg mismatch)
- [x] #2 Image size cap and resize dimension honor max_size_mb / resize_max_dimension
- [x] #3 Legacy regression gate stays green (defaults must reproduce current behavior; zero edits to existing gate tests)
- [x] #4 .tiff/.tif attach end-to-end: decoded and delivered to providers as a payload-safe format (png/jpeg/webp/gif) with mime matching the actual bytes
- [x] #5 .svg attach end-to-end via cairosvg rasterization when available; .svg absent from picker, routing, and allowlist when cairosvg is unavailable (capability gate)
- [x] #6 Payload images always carry provider-safe formats with mime matching bytes (repairs latent bmp passthrough and resized-gif mime mismatch)
<!-- AC:END -->

## Implementation Plan

Spec: `Docs/superpowers/specs/2026-07-14-console-config-caps-design.md`; plan with full TDD code: `Docs/superpowers/plans/2026-07-14-console-config-caps.md`. Four tasks: (1) cairosvg optional extra + `optional_deps.ensure_svg_rendering()` with macOS Homebrew dyld fix; (2) call-time policy functions in `attachment_core` (+ extended template default, Settings fallback, drift-by-construction tests); (3) `ChatImageHandler.prepare_image_payload` — bounded SVG rasterize, payload-safe transcode, truthful mime, pinned-signature `_process_image_data` adapter, `process_attachment_bytes` rewire; (4) call-time routing/picker rewires, retire `ATTACHMENT_FILTER_SPECS`/`SUPPORTED_EXTENSIONS`.

## Implementation Notes

Executed via subagent-driven development (4 tasks, each spec+quality reviewed; final whole-branch review; live QA with user screenshot approval). Single-source policy layer in `Chat/attachment_core.py` (`supported_image_formats`/`max_image_bytes`/`image_resize_max_dimension`/`attachment_filter_specs`/`svg_rendering_available`) consumed call-time by both pickers, routing (`ImageFileHandler.can_handle`), the Console paste gate, and the pipeline. SVG ships as optional extra `svg = ["cairosvg"]` gated through `optional_deps.ensure_svg_rendering()` (in-process `DYLD_FALLBACK_LIBRARY_PATH` fix for Homebrew cairo on macOS); rasterization is aspect-bounded (viewBox/width/height parse via defusedxml; both-dims hard bound fallback) with `unsafe=False` pinned by tests. `prepare_image_payload` returns provider-safe bytes with mime derived from actual bytes, repairing two latent bugs (bmp passthrough mime, resized-gif mime mismatch).

Key deviations/findings: (a) final review found a Critical — `get_cli_setting("chat.images", key)` is a flat lookup that never resolves nested TOML, so all policy reads were silently inert; fixed via `_chat_images_setting` nested read + an UNMOCKED integration test (scratch TOML through the real loader); pre-existing broken dotted-section callers elsewhere → TASK-229. (b) One sanctioned one-line edit to an existing test (`test_attachment_core` bytes-cap fault injection repointed from the module constant to the policy function — the old technique only worked because of the flat-lookup bug). (c) Existing configs pin their `supported_formats` list (present key governs) — enabling tiff/svg on an existing install is a one-line config edit. (d) Vision QA exposed a latent Console streaming defect (worker-group collision) — root-caused and filed as TASK-228; not caused by this branch. Housekeeping riders → TASK-230.

Files: `Chat/attachment_core.py`, `Event_Handlers/Chat_Events/chat_image_events.py`, `Utils/file_handlers.py`, `Utils/optional_deps.py`, `Chat/console_paste_attach.py`, `UI/Screens/chat_screen.py`, `UI/Chat_Modules/chat_attachment_handler.py`, `UI/Tools_Settings_Window.py`, `config.py`, `pyproject.toml`; new tests `Tests/Chat/test_attachment_policy.py`, `test_image_payload.py`, `test_attachment_routing.py`, `Tests/Utils/test_svg_rendering_dep.py`. Sweep 830 passed / 70 skipped / 0 failed; QA evidence `Docs/superpowers/qa/console-config-caps-2026-07/`.
