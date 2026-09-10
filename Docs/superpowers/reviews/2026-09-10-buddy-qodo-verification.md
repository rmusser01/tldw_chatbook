# Qodo PR 2572 fix report

Base: `cd66ade719c3d72a21ecc29a0de20ed3fc19152f`

Implementation commit: `84c567f4405dc9999082a5e242b41b9e219df585`

## Verdicts

1. **Confirmed and fixed — Petdex descriptions were dropped.**
   `tldw_chatbook/Petdex/conversion.py:256-264` now constructs the snapshot with
   explicit keywords and carries `source.description`. The regression reads the
   generated native archive through `read_buddy_archive` at
   `Tests/Petdex/test_conversion.py:74-82`.
2. **Confirmed and fixed — corrupt DEFLATE data escaped the local-package
   boundary.** `tldw_chatbook/Petdex/sources.py:652-681` now normalizes
   `zlib.error` to `petdex_source_invalid`; the injected decoder-failure test is
   `Tests/Petdex/test_sources.py:136-149`.
3. **Confirmed and fixed — local package input bypassed shared lexical path
   validation.** `tldw_chatbook/Petdex/sources.py:667` calls
   `validate_path_simple(..., probe_existing=False)` before filesystem access.
   This deliberately preserves the path for the later no-follow admission and
   freshness guard instead of resolving away link evidence. The spy regression
   is `Tests/Petdex/test_sources.py:119-133`.
4. **Contradicted — the retry key already identifies the complete reviewed
   archive.** `BuddyManagementCoordinator.apply_choice` keys on the staged
   snapshot `source_sha256` at
   `tldw_chatbook/UI/Navigation/buddy_management.py:775-800`.
   `read_buddy_archive` assigns that field from the SHA-256 of the complete
   generated archive bytes at `tldw_chatbook/Persona_Visual/snapshot.py:168-180`,
   after reviewed mappings and states have been serialized into the native
   manifest by `tldw_chatbook/Petdex/conversion.py:173-237`. The real
   native/publication regression at
   `Tests/Persona_Buddy/test_buddy_management_import.py:351-407` proves that two
   mappings of one source yield different staged digests, different retry keys,
   and two distinct installed records. No cache implementation change was made.
5. **Confirmed and fixed — transient animated-header failures became stable
   placeholders.** `tldw_chatbook/UI/Console_Modules/character.py:700-708`
   returns before caching and clears the stable tick key so the next ordinary
   refresh retries. The paint guard at lines 497-505 replaces the temporary
   non-animated fallback after recovery. The same-identity retry regression is
   `Tests/UI/test_console_character_avatar.py:1096-1148`.
6. **Confirmed and fixed — initial render failure retained prepared frame
   ownership.** `tldw_chatbook/Widgets/Console/character_expression_avatar.py:131-140`
   closes and clears the prepared expression in the initial preparation error
   path. `Tests/UI/test_character_expression_avatar.py:203-225` verifies both
   frame disposal and shared-budget release while mounted.
7. **Confirmed and fixed — the established one-folder recovery route dropped
   its detected prefix.** `tldw_chatbook/Character_Chat/expression_set_io.py:123-131,265-283`
   carries the detected wrapper into native reading. The snapshot reader and
   importer keep the original pinned source and apply the prefix only inside the
   same strict member/declaration/checksum boundary at
   `tldw_chatbook/Persona_Visual/snapshot.py:75-112` and
   `tldw_chatbook/Persona_Visual/importer.py:307-375,615-628`. Prefixes are one
   safe top-level ASCII directory; dot components, Windows devices, outside
   members, normalized collisions, undeclared data and checksum mismatches fail
   closed. Coverage is in `Tests/Character_Chat/test_expression_set_io.py:580-608`
   and `Tests/Persona_Visual/test_buddy_snapshot.py:49-87`.
8. **Confirmed and fixed — failed replacement loads retained prior source UI.**
   `tldw_chatbook/Widgets/Persona_Widgets/petdex_import_review.py:203-225`
   clears every source-derived logical and display field after replacement is
   committed; lines 248-290 keep file-picker cancellation ahead of that reset.
   The mounted regression is `Tests/UI/test_petdex_import_review.py:205-263`.
9. **Confirmed and fixed — public Petdex APIs lacked contract sections.**
   `source_from_bytes` and `read_local_package` now document all parameters,
   returns and exposed validation failures at
   `tldw_chatbook/Petdex/sources.py:128-155,652-665`.

## Test evidence

All pytest commands used:

```text
PYTHONPATH=/private/tmp/chatbook-buddy-qualification:/private/tmp/chatbook-buddy-qualification/packages/tldw_profile_core/src /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <nodes> -q
```

- RED nodes:
  `Tests/Petdex/test_conversion.py::test_generated_native_archive_preserves_petdex_description`
  `Tests/Petdex/test_sources.py::test_local_package_uses_shared_path_validation_without_resolving_links`
  `Tests/Petdex/test_sources.py::test_corrupt_deflate_error_is_normalized_at_local_package_boundary`
  `Tests/Persona_Buddy/test_buddy_management_import.py::test_staged_retry_cache_distinguishes_complete_reviewed_archives`
  `Tests/UI/test_character_expression_avatar.py::test_initial_renderer_failure_releases_prepared_frames`
  `Tests/Character_Chat/test_expression_set_io.py::test_dispatch_nested_native_archive_keeps_strict_validation`
  `Tests/UI/test_petdex_import_review.py::test_replacement_failure_clears_old_source_but_picker_cancel_preserves_it`
  `Tests/UI/test_console_character_avatar.py::test_transient_animated_header_failure_retries_same_identity`.
  Result before production edits: **7 failed, 1 passed, 1 warning in 8.26s**.
  The passing case was finding 4 and is the evidence that its premise is false.
- The same eight nodes after correction: **8 passed, 1 warning in 6.63s**.
- `Tests/Petdex/test_sources.py Tests/Petdex/test_conversion.py
  Tests/Petdex/test_publication.py
  Tests/Persona_Buddy/test_buddy_management_import.py`: **84 passed, 1 warning
  in 19.39s**.
- `Tests/Character_Chat/test_expression_set_io.py
  Tests/Persona_Visual/test_buddy_snapshot.py
  Tests/Persona_Visual/test_persona_visual_importer.py`: **86 passed, 1 skipped,
  1 warning in 5.30s**. The skip is the opt-in external Buddy collection probe.
- `Tests/UI/test_character_expression_avatar.py
  Tests/Chat/test_character_expression_playback.py
  Tests/UI/test_petdex_import_review.py`: **52 passed, 2 warnings in 16.28s**.
- Focused Console nodes covering refresh/cache, corrupt and transient animation,
  request fences, and mounted playback: **13 passed, 1 failed, 1 warning in
  45.57s**. The failure was
  `test_animated_character_uses_mounted_playback_and_same_asset_mode_change` at
  `Tests/UI/test_console_character_avatar.py:2651`: the expected blue second
  frame `(0, 0, 255, 255)` was still the red first frame `(255, 0, 0, 255)`.
  This assertion runs while the unchanged mounted widget's 30 Hz interval is
  active. The test sets elapsed time and calls `_tick()` without stopping that
  interval; if the scheduled paint already owns `_painting`, the unchanged
  `_paint` guard returns and the manual tick observes the old frame. The failure
  occurs before the controller refresh branch changed by this fix. Its immediate
  isolated rerun was **1 passed, 1 warning in 5.79s**, which supports scheduling
  sensitivity but is not claimed as a baseline comparison. The new
  transient-retry regression passed in both runs. The grouped failure remains
  open for root adjudication.

## Static evidence

- Ruff on the 11 changed files without existing whole-file lint debt:
  `ruff check Tests/Persona_Buddy/test_buddy_management_import.py
  Tests/Persona_Visual/test_buddy_snapshot.py Tests/Petdex/test_conversion.py
  Tests/Petdex/test_sources.py Tests/UI/test_character_expression_avatar.py
  Tests/UI/test_petdex_import_review.py tldw_chatbook/Persona_Visual/snapshot.py
  tldw_chatbook/Petdex/conversion.py tldw_chatbook/Petdex/sources.py
  tldw_chatbook/Widgets/Console/character_expression_avatar.py
  tldw_chatbook/Widgets/Persona_Widgets/petdex_import_review.py`:
  **All checks passed**.
- Ruff on the five changed files with existing lint debt, ignoring only their
  pre-existing `BLE001,C402,I001,PIE807,RUF019,RUF059,RUF100,TRY004` classes:
  **All checks passed**. Owned `F401`, `I001`, and `RUF059` findings surfaced by
  the first unfiltered run were corrected.
- `ruff format --check` on the 13 changed files without existing whole-file
  formatting drift: **13 files already formatted**. The three excluded legacy
  files were already globally unformatted; their owned hunks follow the local
  formatter output without rewriting unrelated lines.
- `git diff --check`: **passed**.
- Statement-level production diff scan for logger, metric and diagnostic calls:
  **no changed diagnostic statements**; no generated artifact was regenerated.

## Limits and handoff

No full suite, network source, live application, or GitHub write was run in this
fix task. Pytest repeatedly emitted the existing `requests` dependency warning
and best-effort temporary-directory cleanup warnings; the graphics widget test
also emitted its upstream Pillow deprecation warning. Root owns the final actual
archive application/helper rerun, current-development integration, Qodo replies,
push/merge, and TASK-32238 closeout. TASK-32238 remains **In Progress**.

## Root qualification and adjudication

Root reviewed the production diff and the publication, wrapper and rendering
regressions. Finding 4 is contradicted by the actual generated-archive identity;
the other eight findings are addressed. Root's additional dot-prefix finding is
also corrected and covered by refusal tests.

The grouped Console failure was a test synchronization defect: its manual elapsed
time competed with the live 30 Hz timer. The assertion now pauses that timer and
drains an in-flight paint before selecting an exact frame. Production timer and
rendering behavior are unchanged. A focused five-test run covering transient
decode retry, corrupt expressions, mounted Dynamic/Static playback, initial render
cleanup and later-render fallback passed in **13.38 seconds**. The earlier failure
is retained above as the evidence for this test-only correction.

On production commit `84c567f4405dc9999082a5e242b41b9e219df585`, the actual
downloaded-source headless application journey passed in **29.98 seconds**,
including six Dynamic frames, fourteen independent character assets, original
credits and offline restart. The installer helper/import checks passed all **12**
cases in **1.097 seconds**. All seven required local preflight gates passed.
Native terminal, actual Windows and physical voice limits remain as recorded in
the [journey verification](2026-09-10-independent-buddy-journey-verification.md).
