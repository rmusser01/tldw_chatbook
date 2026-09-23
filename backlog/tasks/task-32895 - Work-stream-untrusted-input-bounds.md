---
id: TASK-32895
title: "Work stream: untrusted-input bounds"
status: To Do
assignee: []
created_date: '2026-09-21 23:05'
labels:
  - tier2-review
  - review-security
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Five classes of unbounded read on attacker-influenceable input: three image ingresses with no pixel cap
(the guard already exists in eight other modules), plaintext/HTML/MOBI ingestion with no size cap while
audio and video already have one, a voice download whose digest is computed and never compared, a chat
dictionary that `re.compile`s a user pattern without the validator its sibling uses, and six `subprocess`
calls with no timeout.

Source: tier-2 code review 2026-09-21 -- `qa/tier2-code-review-2026-09-21/report.md` (26 slices, 890,356 lines: the surface tier 1 never reached). Per-slice evidence in `qa/tier2-code-review-2026-09-21/slices/`, reproductions in `phase4-verification.md`, and per-finding re-validation against `origin/dev d0face3ebe` in `validation/`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Every image ingress enforces one shared decoded-pixel cap
- [ ] #2 Text ingestion has a size cap symmetric with the existing audio/video caps
- [ ] #3 The computed digest is compared and a byte ceiling is enforced
- [ ] #4 No `subprocess` call in the children runs without a timeout
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented on `fix/tier2-bounds` as `cb86cea4d9`. Not pushed. All five real.

**Correcting the "eight other modules" figure in this task's description.** It is a *union of two guard
families*, which is why a single grep found only 2:

| family | modules |
|---|---:|
| `warnings.simplefilter("error", DecompressionBombWarning)` | 6 |
| explicit `*_DECODED_PIXELS` comparison | 4 |
| **union** (2 modules use both) | **8** |

That changed the design, for the better. The warning-escalation idiom is the majority (6 of 8) and needs no
shared constant, so `Local_Ingestion` and `Image_Generation` adopted **that** rather than importing a
`Persona_Visual` contract — the coupling the task description asked for would have been a bad dependency
edge for no benefit. Only the persona importer uses the moved constant.

**The persona importer was the serious one.** Its frame loop does `seek(i); load()` on **every** frame
before any cap, while its sibling `assets._decode_selected_frame` has enforced the cap all along. Declared
dimensions bound at 4096 and frames at 240, so the import path permitted 4096² × 240 = **4,026,531,840
pixels — exactly 60× the 67,108,864 cap the load path enforces** (independently re-derived). Cap now applied
before the loop, matching `assets.py:430`.

No overlap with TASK-32806.8 (it owns `chat_image_events.py` + `console_chat_fork.py`; neither touched).

**Kokoro was worse than filed.** No expected digest exists anywhere in the repo or the pinned release URLs —
so there was nothing to compare against. The code hashed every byte, logged the digest, then logged
"Checksum verification skipped". Rather than fabricate a pin, the theatre was removed: `expected_sha256` now
verifies fail-closed *before* `os.replace`, the two pins are explicit `None` with the reason in-line, and one
honest warning says the artifact is unverified. The real bound is the byte ceiling, which covers **all four**
downloads, not the two named.

Text caps matched the existing idiom verbatim (`get_cli_setting("media_processing.max_<kind>_file_size_mb")`,
explicit `None` fallback, the module's own error message), adding `max_text_file_size_mb = 50`. Regex routed
through `validate_regex_pattern`, degrading to literal — the shape the function already used for `re.error`.
Five `subprocess.run` sites got timeouts; `system_audio_tap.py:305` already passed `timeout=5` in the same
file, so `:235` was the outlier. The long-running `Popen` capture at `:342` deliberately keeps none.

## Two process notes worth keeping

1. **One test was NOT born red, and finding out why sharpened the fix.**
   `test_image_format_conversion_refuses_a_pixel_bomb` passed against unpatched code, because Pillow already
   *raises* above **2×** `MAX_IMAGE_PIXELS` and that exception was being converted to
   `ImageGenerationError`. The actual hole is the band between **1× and 2×**, where Pillow only warns and
   decodes anyway. Fixtures now land in that band deliberately (`_WARNING_BAND_CAP`, commented). Had the
   born-red check been skipped, this would have shipped as a test that proved nothing.
2. **Preflight caught a privacy regression inside the fix itself.** Replacing
   `logger.error(f"FFmpeg error: {result.stderr}")` with `{detail}` silently deleted a `legacy_unreviewed`
   path-privacy row from the diagnostic census — the checker could no longer see `result.stderr` statically,
   though ffmpeg stderr still carries filesystem paths. Reshaped the timeout as a
   `subprocess.CompletedProcess` so the original statement survives byte-identical. This is the diagnostic
   inventory doing exactly the job it exists for.

Gates: preflight exit 0; size ratchet exactly 5 failed / 58 passed, unchanged; regression run both ways over
the same selection — **405 failed before -> 388 after, zero new failures**, delta exactly the 17 new tests.

Loose end: `_kokoro_stream_download` still takes `hasher`, now unused in production, kept only because
`test_hasher_sees_every_byte` pins it. Delete both if the observation is not wanted.
<!-- SECTION:NOTES:END -->
