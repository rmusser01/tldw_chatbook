---
id: TASK-3305
title: >-
  Ingest copy & labels: human select values, one destination name, honest supported list, exception-free errors
status: Done
assignee:
  - '@claude'
created_date: '2026-08-07 19:30'
labels:
  - library
  - ingest
  - ux
  - copy
priority: medium
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Findings MI-09/11/12/13/14/16/18/19 of the 2026-08-07 Media Ingestion review. (1) Selects render raw internal tokens as user-facing values (`pymupdf4llm`, `filtered`, `parakeet-onnx`, `url_level`, `recursive_scraping`) — canvas builds `[(opt, opt)]`. (2) Three names for one destination on one screen: "Import media" / "Add content…" / "Import / Export". (3) The supported-list copy and the start-gate reason omit web/URLs while the surface accepts them. (4) URL preflight surfaces a raw Python exception repr (`<urlopen error [Errno 8]…>`). (5) Recent ingests shows literal backslashes (escape_markup applied to markup=False Statics). (6) "1 done — in queue" for finished jobs. (7) The audio collapsed title is a ~140-char run-on with a dangling empty value (`Local Parakeet model folder: ,`). (8) Grammar batch: "Applies to all Plain text & HTML in this import.", label==placeholder duplication, failed-empty-row repeating its basename, commit line visible while the option-error gate blocks, breakdown "1 web" noun, URL shown as "1 file · 0 B".
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Every option select shows a human label (with the internal value still persisted); no raw token appears in the rendered panel
- [x] #2 The ingest destination carries one name across header, rail button, and rail section
- [x] #3 Supported-list copy and start-gate reasons name URLs/web pages wherever the surface accepts them
- [x] #4 URL preflight failures render a plain-language message with no exception repr
- [x] #5 Escaped filenames render clean in Recent ingests; queue summary says "done", not "in queue", for finished runs; collapsed titles cap at a few salient pairs with no dangling empty values
- [x] #6 Grammar batch items fixed (scope sentence, placeholder, empty-file row, gate-vs-commit mixed message, breakdown nouns, URL estimate line)
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Add display-label support to the option schema (label map per field option) consumed by the canvas select builder; keep persisted values stable.
2. Naming decision: adopt "Import" family everywhere (matches header + picker frame) — header "Import media", rail button "Import media…", section unchanged if it lists both import and export.
3. Copy fixes at their sources (`ingest_capabilities` hints, `library_ingest_state` breakdown nouns/summaries, `ingest_preflight` error mapping, canvas title builder cap, guardrail/queue captions).
4. Tests: rendered-label assertions, preflight error mapping, collapsed-title cap, breakdown noun table.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
TDD: 19 new tests written first; 18 went RED on the exact defects
(`option_labels` AttributeError; 7 preflight raw-repr failures; counts
suffix / "1 web" / supported copy / URL estimate / basename echo; 5
rendered-canvas failures; the shell repro "commit forecast stayed visible
while the option-error gate blocks Start" — `True is False` on
`.display`), 1 deliberately pinned already-correct behavior (active jobs
keep the "— in queue" suffix). Two mutations run, each RED then restored
via Edit: dropping one `option_labels` entry (utf-16) sent BOTH the
schema meta-test and the rendered-canvas sweep RED; reverting
`_probe_url`'s URLError branch to the old f-string repr sent 5 mapping
tests RED.

**Per-item verdicts (re-verified against the 3300-3304 branch state
first):**
- **AC#1 select labels — fixed here.** Canvas still built `[(opt, opt)]`.
  New `OptionField.option_labels` (value→label pairs, comma-free because
  titles comma-join them) + `select_option_label()` — the ONE resolution
  seam used by the Select builder AND the collapsed-title summariser.
  Labels verified against the backends before writing: pymupdf=plain
  text / pymupdf4llm=Markdown / docling=layout-aware / docext=vision-model
  OCR (`PDF_Processing_Lib`); "default" provider labeled "Auto
  (faster-whisper)" because the Library batch call site never opens the
  Parakeet promotion gate (re-verified `app.py:2465` passes no
  `parakeet_defaults_enabled`, confirming 3301); ebook
  filtered/markdown/basic per `Book_Ingestion_Lib`'s three readers;
  encoding "Auto-detect (UTF-8 first)" per `_decode_ingest_text`'s
  strict-utf8→chardet→replace ladder; scrape methods reworded as scope
  ("This page only" … "Follow links (recursive)"). Persisted values stay
  the internal tokens (the rendered sweep asserts both directions).
- **AC#2 one destination name — ALREADY FIXED by task-2857 (LIB-10,
  merged PR #1410 arc), verdict recorded, no change.** "Add content…" no
  longer exists anywhere; rail row + hub action say "Import…", canvas
  says "Import media"/"Start import", toast "Import finished" — one
  canonical Import verb, pinned by
  `test_library_shell_import_verb_pair_agrees_across_rail_canvas_and_toast`.
  The rail SECTION stays "Import / Export" because it genuinely hosts the
  Export row too (decision recorded). Deliberate deviation from the
  dispatch's "rename to Import media…": that would desync the 2857
  unification (hub action, palette entry) and break its live pin for zero
  information gain — flagged for owner review.
- **AC#3 honest supported list + start gate — fixed here** (3303 had
  updated the document noun but NOT web/URLs): `SUPPORTED_FORMATS_COPY`
  gains "web pages (by URL)", `START_QUIET_LINE_COPY` = "Enter a file
  path or URL to start.", and the intro line gains "web pages" for free
  via the `_TYPE_GROUP_LABELS` web entry (AC#6e).
- **AC#4 exception-free URL errors — fixed here.** `_probe_url` maps
  URLError reasons by kind (gaierror → "the server name could not be
  found."; ConnectionRefusedError; TimeoutError; ssl.SSLError → TLS;
  fallback "the server could not be contacted."), 404/410 → "the server
  says this page does not exist (HTTP 404).", generic probe crash → no
  repr; raw detail goes to `logger.debug` only.
- **AC#5 misc renders — all fixed here.** (a) Recent imports: dropped
  `escape_markup` on the two `markup=False` Statics (double defense
  painted literal backslashes; the queue-row Statics keep their escape —
  they render WITH markup). (b) `_queue_counts_line` appends "— in queue"
  only while a job is queued/parsing/writing; a fully terminal tally
  reads "1 done · 1 failed" under the Queue heading. (c) New shared
  `build_type_group_title()` (canvas compose + the screen's in-place
  receipt): skips empty values (no more "Local Parakeet model folder: ,"),
  changed-from-default pairs first, cap 3 pairs + "…".
- **AC#6 grammar batch — (d) was REAL and live, rest fixed here.**
  (a) scope line now composes from new per-group noun fields
  ("Applies to every plain text & HTML file in this import." /
  "Applies to plain text & HTML files if this import contains any.").
  (b) `OptionField.placeholder` — Parakeet folder shows
  "/path/to/parakeet-model" instead of echoing its label. (c) failed/
  cancelled/skipped rows strip a leading repeat of their own basename
  from the error detail (`_strip_basename_echo`). (d) VERIFIED as a live
  defect, not superseded: the state builder already emptied
  `commit_summary_line` under option errors, but text/number edits take
  the in-place path, which synced the gate and NOT the forecast — the
  commit-summary sync now lives in `_update_library_ingest_gate` (the one
  place every gate change flows through; the duplicate block in
  `_update_library_ingest_dynamic_regions` was removed, one source).
  (e) `_TYPE_GROUP_LABELS["web"] = ("web page", "web pages")` — was
  falling back to "1 web"/"2 webs". (f) new
  `PreflightResult.source_is_url` (set by `analyze_path`) suppresses the
  estimate line for URLs — "1 file · 0 B" was a fabrication; the
  breakdown line already says "1 web page".

Files: `tldw_chatbook/Library/ingest_capabilities.py` (labels, nouns,
placeholder, `select_option_label`), `ingest_types.py` (`source_is_url`),
`ingest_preflight.py` (error mapping), `library_ingest_state.py` (copy,
web noun, counts suffix, estimate skip, basename echo),
`Widgets/Library/library_ingest_canvas.py` (labeled selects, shared
title builder, scope nouns, placeholder, escape fix),
`UI/Screens/library_screen.py` (gate-owned commit sync, shared title
builder), tests (5 files + pins), `Docs/User_Guide/library/
import-and-export.md` (+ task-3305 stamp).

Counts: full arc battery (option wiring, submit, capabilities, state,
preflight, canvas, keyboard, guardrail, structural, integration flow,
url submit) **479 passed / 0 failed**; `test_library_shell.py -k
"ingest or commit or scope or quiet or counts"` 71 passed;
shell-state/screen/config-defaults 60 passed.
<!-- SECTION:NOTES:END -->
