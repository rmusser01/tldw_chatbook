---
id: TASK-18606
title: Decouple visual transcript renderer identity from the Pillow version
status: Done
assignee: []
created_date: '2026-08-18 23:00'
labels:
  - bug
  - console
  - dependencies
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
ADR-054 made the running Pillow version part of the visual transcript renderer's
identity, and `pyproject.toml` pinned `pillow==11.2.1` to honour it. That is the
wrong place to put the guarantee, and it did not even deliver it.

Pillow is the image parser this application points at untrusted input (media
ingestion, chat attachments, remote image fetch). Holding it a major version back
so that one checked-in evidence file keeps matching is a bad trade.

Worse, the pin did not make the renderer correct outside the pinned version.
`ImageFont.load_default()` is not a stable input — Pillow changed it from the
legacy fixed-cell bitmap font to a proportional TrueType face during the 10.x
line. Measured on Pillow 12.1.1: an 82-character line renders **738px** against
**496px** of usable canvas, so text runs off the right edge and is lost, with no
error raised. The only thing pointing at this was a support-matrix hash mismatch
that read like stale evidence.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 A full-width line renders inside the canvas on any supported Pillow.
- [x] #2 A font whose cell metrics no longer match the layout raises rather than silently clipping.
- [x] #3 Renderer identity is unchanged by a PNG encoder change that leaves pixels identical.
- [x] #4 `renderer_version` does not name the Pillow version, and is not made unstable by it.
- [x] #5 Wire-integrity checking of the exact bytes sent still holds, distinctly from renderer identity.
- [x] #6 The Pillow pin is relaxed and ADR-054 records the amendment and its reason.
- [x] #7 Evidence captured under a superseded renderer cannot authorize a default.
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Three couplings removed, each independently able to invalidate the renderer's
identity on a dependency bump.

**Font.** `renderer_font()` pins `ImageFont.load_default_imagefont()` — Pillow's
frozen legacy fixed-cell font — instead of `load_default()`, and then MEASURES the
advance width and checks it against `CELL_WIDTH`, raising `VisualRendererFontError`
on mismatch. Verified: rightmost ink moves from x=511 (past the x=504 usable edge,
clipping) to x=499. The verification is the durable part — the previous design had
no way to notice its own font had changed under it.

**Encoder.** Page identity now hashes raw pixel data plus mode and size, not
encoded PNG bytes, so compression level or chunk ordering cannot move it. Proven
by constructing two encodings of identical pixels: the old digest differs, the new
one does not.

**Version string.** `renderer_version` no longer embeds `PILLOW_VERSION`. This one
was self-fulfilling — the version string is also DRAWN INTO every page footer, so
it was literally part of the pixels being hashed, and the hash could not survive a
Pillow bump by construction. Both profiles move to a `v3` generation
(`chatbook-visual-transcript-v3` / `-v3-native-512`).

**Kept separate on purpose:** `png_sha256` survives alongside the new
`pixel_sha256`. They answer different questions — "would another host render this
same page" (identity, pixels) versus "are these the exact bytes I hashed" (wire
integrity, checked by `tagged_visual_memory_message`). Conflating them is what put
Pillow's encoder into the identity in the first place. The rename forced every
call site to be classified as one or the other; three were wire paths and three
were evidence paths.

**Evidence guard reframed.** `test_checked_in_evaluator_v3_matrix_is_current_terra_
context_evidence` pinned the evidence to one renderer generation, so any renderer
change — including this bug fix — turned it red with a bare hash mismatch. Rewritten
as `..._never_enables_on_stale_evidence` around the property it was standing in
for: if the evidence matches the current renderer, the full geometry check applies
unchanged; if it describes a superseded renderer, it is historical and
`eligible_models` must be empty with `default_enablement_ready` False, so stale
evidence can never authorize a default. The strict branch reactivates by itself
once evidence is re-captured.

**Not done:** re-capturing the gpt-5.6-terra evaluation under v3. That is a real
model run and a spend decision. The current matrix already says "not ready", so
the safe state is preserved meanwhile.

Files: `Chat/console_visual_transcript.py`, `Chat/console_visual_evaluation.py`,
`Chat/console_chat_controller.py`, `Chat/console_visual_benchmark.py`,
`pyproject.toml`, `backlog/decisions/054-...md`,
`Tests/Chat/test_visual_renderer_decoupling.py` (new, 11),
`Tests/Chat/test_console_visual_evaluation.py`.
<!-- SECTION:NOTES:END -->
