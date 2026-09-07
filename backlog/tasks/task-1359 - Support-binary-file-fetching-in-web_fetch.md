---
id: TASK-1359
title: Support binary file fetching in web_fetch
status: Done
assignee:
  - '@claude'
created_date: '2026-08-05 06:04'
updated_date: '2026-08-08 21:17'
labels:
  - web-tools
dependencies:
  - TASK-1354
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
v1 rejects non-HTML content types. Add safe handling for common binaries (images, audio, archives): bounded temp download, metadata + safe preview/extraction where feasible, never execute downloaded content.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Content-type allowlist + size caps + zero on-disk persistence (no temp files; guarded by the existing no-persistence-import static test — amended 2026-08-07 from "temp-dir hygiene", superseded by the in-memory ruling in Docs/superpowers/specs/2026-08-07-web-fetch-binary-design.md),No execution of downloaded content,Clear result shape; tests
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: generalized _fetch_once's PDF-only sniff (a bool is_pdf) into a
sniffed/declared "kind" string (pdf/image/zip/audio/None), updating all
three positional-destructure call sites (web_fetch, _crawl_fetch_page,
_fetch_robots_parser). Kind resolution mirrors the PDF precedent: an
allowlisted declared content-type resolves immediately (no byte
confirmation); otherwise a magic-byte sniff on the first 12 buffered bytes
(WEBP's two-anchor RIFF/....WEBP check needs all 12, up from PDF's 5)
overrides a wrong/absent declared type. Audio has no reliable magic and is
declared-type-only, matched via a new _top_level_type() helper (kept
distinct from _extract_text's existing main_type splitter per the spec's
Important 7 correction).

Three new in-memory describers, dispatched from the TOP of _extract_text
(above _decode_body, so binary bytes never round-trip through
UTF-8-replace): _describe_image (Pillow open+verify()+defensive re-open,
format/size/mode only, no pixel decode, no EXIF -> "[image] FMT WxH,
size" or [image-error]), _describe_archive (stdlib zipfile listing only,
never extracts/reads members -> "[archive] ZIP, size, N members" + up to
ARCHIVE_LIST_MAX=20 member lines, "... and N more" beyond; hostile member
names screened via _member_display_name -- mirrors
chatbook_importer._validated_archive_parts but flag-and-show via
"[suspicious name] {name!r}", not reject; encrypted members annotated via
flag_bits & 0x1, not refused -- infolist() works fine under encryption,
only a member's own .read() needs the password; only BadZipFile ->
[archive-error]), and _describe_audio (declared content-type + size only,
no new dependency -- mutagen stays out).

All three non-PDF binary kinds share one 10 MB refusal ceiling
(BINARY_MAX_BYTES); a byte-truncated binary body raises [too-large],
never processed partially -- same shape as the existing PDF ceiling.
Entirely in-memory: zero disk writes anywhere in the new code, still
guarded by the pre-existing test_module_never_imports_persistence static
check (AC #1's "temp-dir hygiene" clause was amended 2026-08-07, before
implementation, to "zero on-disk persistence" per the repo's SDD workflow
rule -- see the design doc's ruling 1 for the full provenance).

web_crawl's own marker branch deliberately keeps consulting only
kind == "pdf" (not "any recognized binary kind") and never passes kind
into _extract_text, preserving its pre-existing mojibake-decode behavior
for a page whose body sniffs as image/zip but is declared text/html (or
unlabeled) during a crawl -- an explicit non-goal in the design doc, not
silently implied.

Files: tldw_chatbook/Tools/web_tool_impls.py (core implementation),
tldw_chatbook/Agents/local_tool_provider.py (tool description gains one
clause: images/zip/audio return metadata, not contents),
Tests/Tools/test_web_tool_impls.py (17 new tests: image metadata/corrupt/
sniff-wins/over-ceiling/webp-dribble, archive listing/over-list-max/
hostile-names(+direct unit test for the NUL case)/encrypted/corrupt/
sniff-wins, audio metadata/subtype-variants, short-body-under-sniff-
window fallback, unsupported-binary-type regression pin).

Deviations/findings: (1) CPython's zipfile.ZipInfo constructor silently
truncates a filename at the first NUL byte on BOTH write and read, so a
real zip fixture can never carry a NUL byte through to infolist() -- the
NUL branch of _member_display_name is pinned by a direct unit test
instead of an end-to-end fetch. (2) Setting ZipInfo.flag_bits before
writestr() does not survive the writer (it recomputes the flag itself),
so the "encrypted member" test fixture patches the flag directly in the
built central-directory bytes. (3) Two mutation checks performed and
reverted (Edit-based, never git checkout --): disabling the traversal
screen turned the hostile-name tests red; disabling the ZIP magic sniff
turned the octet-stream-sniff test red -- both restored to green
afterward, confirmed via full suite re-run.

Verification: 152/152 in Tests/Tools/test_web_tool_impls.py +
test_web_crawl.py (135 pre-existing + 17 new, all green -- the is_pdf ->
kind generalization did not disturb PDF/crawl behavior); 568 passed / 3
pre-existing live-search skips across Tests/Tools/ + Tests/Web_Scraping/;
full-repo collect-only unchanged at 32255 tests, no new errors; ruff
clean on all three touched files.

**Fix round 1 (2026-08-08, controller-executed inline after the fix agent hit a usage limit):** review Important 1 — `_member_display_name` now flags ANY non-printable character (`str.isprintable()`), closing listing-row forgery via newline/ESC/RTL-override member names (mutation-verified); Important 2 — the `html_only` early-abort keys on `kind == "pdf"` + declared type again, restoring the documented crawl non-goal (a mislabeled binary reads in full; the regression test's first draft was VACUOUS — single-chunk MockTransport bodies are fully captured before the abort check — caught by the scoped re-review; rewritten with chunked delivery and mutation-verified red against the pre-fix predicate); Minor 3 — the read loop can no longer break below the 12-byte sniff window on a tiny caller max_bytes (partial-prefix `[pdf-error]` eliminated; extracted-text output remains max_bytes-bounded, pre-existing); Minors 4/5/6/7/9/10 — static no-persistence guard now also greps tempfile/mkstemp/mkdtemp, fixed `[image-error]` message for unidentifiable formats (no BytesIO repr/heap address), tool description states the 10 MB binary ceiling and corrected max_bytes wording, comment/doc truth fixes. Minor 8 coverage added (zip-over-ceiling → [too-large] ordering, GIF sniff, absent-(encrypted) pin, size-suffix pin). Minor 11 filed as task-3280 (crawl cache warm-writes leak mojibake for mislabeled binaries into web_fetch's cache).
<!-- SECTION:NOTES:END -->
