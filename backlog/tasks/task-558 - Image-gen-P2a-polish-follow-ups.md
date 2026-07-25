---
id: TASK-558
title: Image-gen P2a polish follow-ups
status: Done
assignee: []
created_date: '2026-07-24 09:05'
updated_date: '2026-07-25 07:51'
labels:
  - image-generation
  - console
  - followup
dependencies: []
priority: medium
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Non-blocking findings from the whole-branch review of the image-gen Console card + variants feature (P2a; spec `Docs/superpowers/specs/2026-07-23-image-gen-console-card-variants-design.md`). None block the P2a PR; group into one cleanup pass. Distinct from [[task-497]] (P1 polish) and [[task-498]] (egress/SSRF), and from the deferred P2b feature slice (TTS, Style-preset picker, prompt-from-context).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 In-memory screen-state restore path (`ChatScreen.restore_state` → `_restore_console_message`) rehydrates `generation_metadata` the same way it already rehydrates attachments, so a tab-switch-restored generation message keeps its card (today only the DB-resume path hydrates).
- [x] #2 Generation records capture the resolved seed/model where the backend reports them (card currently shows the requested seed or "random" and no model; extend `run_generation_batch`/`GenerationVariantMeta` population when `ImageGenResult` grows the fields).
- [x] #3 `/generate-image` clamps the initial batch to `max_variants_per_message` (today only regenerate enforces the cap; a misconfigured `default_batch` can exceed it).
- [x] #4 Stale narrative fixed in `test_console_generation_store.py`'s round-trip test (comments claim `restore_persisted_session` doesn't hydrate — false since the resume fix; drop the redundant manual hydrate calls) and `restore_persisted_session`'s docstring documents the hydration it now performs.
- [x] #5 Draft is restored to the composer when the generation batch RAISES (today only the zero-success return path restores it).
- [x] #6 Test-coverage nits closed: generation-vs-sibling precedence pinned with `sibling_count=3`; exact-limit (80-char) content-marker boundary asserted; empty-negative-prompt card branch covered.
- [x] #7 Cosmetics: `console-generation-card*` CSS classes either get TCSS rules or are removed (currently inert; styling is set in Python); new DB ops' docstrings stop promising `CharactersRAGDBError` for raw `sqlite3.IntegrityError` (or wrap it), matching whichever convention the sibling methods adopt.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. In-memory restore hydration: add ChatScreen._rehydrate_console_message_generation_metadata (batched get_generation_metadata_for_messages across all restored sessions, feeding ConsoleChatStore.hydrate_generation_metadata per session), call it in _restore_native_console_state right after store.restore_state(...). Mirrors the existing _rehydrate_console_message_attachments batching pattern. TDD: red test asserting a tab-switch-restored generation message has empty generation_metadata today, then green after the fix.
2. Resolved seed/model: add optional resolved_seed/resolved_model fields to ImageGenResult (default None, non-breaking). Populate resolved_model in SwarmUIAdapter and StableDiffusionCppAdapter only (both already deterministically compute the model they send/use; verified via docs that neither backend's response reports a resolved seed reliably, so resolved_seed plumbing is added but left unpopulated -- no fabrication). Thread getattr(result, "resolved_seed"/"resolved_model", None) through run_generation_batch into GenerationVariantMeta, falling back to today's variant_seed/None when absent (byte-identical behavior for the 4 untouched adapters and the test fake `_Res`). Stay out of adapter fetch/egress/redirect code (PR #862 conflict zone).
3. Stale test narrative: fix test_console_generation_store.py's reload-round-trip test comments (restore_persisted_session DOES hydrate now) and drop the redundant manual get_generation_metadata_for_messages + hydrate_generation_metadata calls the test was doing on top of the store's own automatic hydration. Add the hydration behavior to restore_persisted_session's docstring.
4. Draft restore on raise: wrap the run_generation_batch call site in _console_command_generate_image with an except clause that restores the saved draft (identical logic to the existing zero-success branch) and reports the error, keeping the in-flight discard in finally. TDD: red test where run_generation_batch raises, asserting draft is lost today, then green.
5. Test nits: (a) sibling_count=3 precedence test in test_console_message_actions.py (today's only such test uses sibling_count=1, which is too weak to prove precedence over the old elif branch), (b) exact-80-char content-marker boundary test in test_console_generate_image.py, (c) empty-negative-prompt card branch test in test_console_generation_card.py.
6. Cosmetics: give real TCSS rules to console-generation-card / console-generation-card-details / console-generation-card-image-placeholder (mirroring sibling .console-transcript-message-* rules in css/components/_agentic_terminal.tcss); remove the redundant console-generation-card-image class (sizing is fully Python-driven, no CSS ever targeted it). Rebuild the CSS bundle + run check_bundle_sync. Fix chat_persistence_service.py's create_message docstring, which claims CharactersRAGDBError for the attachment-table/generation-metadata sidecar writes -- an overreach relative to its own sibling update_message_content (identical set_message_attachments call, no such claim) and unimplemented (those writes run through a raw transaction cursor with no wrap, unlike add_message's explicit sqlite3.IntegrityError->CharactersRAGDBError wrap).
7. Run full targeted test suite + ruff + import smoke check; write task-558-report.md; update task file ACs/notes; commit.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Approach: seven independent, narrowly-scoped fixes, each with its own TDD red/green cycle where behavioral.

1. In-memory restore hydration: added ChatScreen._rehydrate_console_message_generation_metadata (batched get_generation_metadata_for_messages across every restored session, then ConsoleChatStore.hydrate_generation_metadata per session id), called right after store.restore_state(...) in _restore_native_console_state -- mirrors the existing _rehydrate_console_message_attachments pattern exactly (same probe-for-callable, same graceful-degradation-on-failure contract).

2. Resolved seed/model: added ImageGenResult.resolved_seed/resolved_model (both Optional, default None, additive dataclass fields -- non-breaking for every existing keyword-only construction site and for test doubles that predate the fields). Verified via the actual SwarmUI/stable-diffusion.cpp source and docs (WebFetch/WebSearch) that neither backend's response reliably reports a resolved seed without guessing an undocumented format (SwarmUI embeds it, unverified, in an output filename; sd.cpp's CLI prints nothing to stdout about it) -- resolved_seed plumbing exists but is deliberately left unpopulated everywhere, per the AC's own "do NOT fabricate values" instruction. resolved_model IS populated for SwarmUIAdapter and StableDiffusionCppAdapter, since both already deterministically resolve the exact model they use (request override, else configured default/path) before ever contacting the backend -- a real, non-fabricated value, not a guess. Threaded through run_generation_batch via getattr(result, "resolved_seed"/"resolved_model", None) with the pre-existing variant_seed/None fallback preserved. Stayed entirely out of adapter fetch/egress/redirect code per the PR #862 conflict-avoidance instruction.

3. Fixed the stale test_console_generation_store.py narrative: the reload-round-trip test's docstring/comments claimed restore_persisted_session does not hydrate generation_metadata and drove the get_generation_metadata_for_messages + hydrate_generation_metadata seam manually -- false since a prior fix made restore_persisted_session do this internally via _hydrate_generation_metadata_from_persistence. Dropped the redundant manual calls (test still passes, proving they really were redundant) and documented the hydration behavior on restore_persisted_session's own docstring.

4. Draft restore on raise: _console_command_generate_image's try block around run_generation_batch/append/sync gained an except Exception clause that restores the saved draft using the identical composer.clear_draft()+insert_text_as_paste(saved_draft) sequence the zero-success path already uses, logs and reports the error as a system message, while `finally: inflight.discard(...)` continues to guarantee the in-flight guard is always released.

5. Test-coverage nits: added a sibling_count=3 precedence test in test_console_message_actions.py that is actually load-bearing (unlike the existing sibling_count=1 test, which can't fail even if generation gating were removed) -- it constructs a message whose stale sibling fields would produce the OPPOSITE previous/next enabled states from the generation kwargs if sibling gating won; an exact-80-char content-marker boundary test (the strict `>` in generation_content_marker means 80 is the last un-trimmed length); an empty-negative-prompt card-details test (the "Negative" row is conditionally omitted, never previously exercised with a falsy negative_prompt).

6. Cosmetics: gave real TCSS rules (css/components/_agentic_terminal.tcss) to .console-generation-card / .console-generation-card-details / .console-generation-card-image-placeholder (mirroring the sibling .console-transcript-message-*/.console-transcript-jump-pill rules already in that file); removed the one remaining inert class (console-generation-card-image, applied to all three image-widget variants but never targeted by any CSS -- sizing is fully Python-driven per mode). Rebuilt the CSS bundle (./build_css.sh) and verified reproducibility (python -m tldw_chatbook.css.check_bundle_sync). Fixed chat_persistence_service.py's create_message docstring, which claimed CharactersRAGDBError for the attachment-table and generation-metadata sidecar writes -- an overreach both relative to its own sibling update_message_content (identical set_message_attachments call, no such claim in that docstring) and relative to the actual implementation (those two writes run through a raw self.transaction() cursor with no independent sqlite3.Error->CharactersRAGDBError wrap, unlike add_message's explicit one). Left the underlying DB-layer methods' (set_message_generation_metadata etc.) own docstrings untouched, since they already match their pre-existing, file-wide sibling set_message_attachments's identical convention -- rewriting just the new P2a ones would have introduced a NEW inconsistency rather than removed one.

Files touched:
- tldw_chatbook/UI/Screens/chat_screen.py (rehydrate method + call site; except-on-raise draft restore)
- tldw_chatbook/Chat/console_chat_store.py (restore_persisted_session docstring)
- tldw_chatbook/Chat/console_generate_image.py (run_generation_batch resolved-field threading)
- tldw_chatbook/Chat/chat_persistence_service.py (create_message docstring fix)
- tldw_chatbook/Image_Generation/adapters/base.py (ImageGenResult new fields)
- tldw_chatbook/Image_Generation/adapters/swarmui_adapter.py (_resolve_model + resolved_model on both returns)
- tldw_chatbook/Image_Generation/adapters/stable_diffusion_cpp_adapter.py (resolved_model on return)
- tldw_chatbook/Widgets/Console/console_generation_card.py (removed inert class)
- tldw_chatbook/css/components/_agentic_terminal.tcss + tldw_chatbook/css/tldw_cli_modular.tcss (generated, rebuilt)
- Tests/UI/test_console_native_chat_flow.py, Tests/Image_Generation/test_swarmui_adapter.py, Tests/Image_Generation/test_sd_cpp_adapter.py, Tests/Chat/test_console_generate_image.py, Tests/Chat/test_console_generation_actions.py, Tests/Chat/test_console_generation_store.py, Tests/Chat/test_console_message_actions.py, Tests/Chat/test_console_generation_card.py

Verification: every new/modified test run individually (red before green for the behavioral ACs), then the full targeted suite (Tests/Chat/test_console_generation_store.py, Tests/Image_Generation/, Tests/Chat/test_console_generate_image.py, Tests/Chat/test_console_generation_actions.py, Tests/Chat/test_console_generation_card.py, Tests/Chat/test_console_message_actions.py, Tests/Chat/test_console_chat_store.py, and the full 200-test Tests/UI/test_console_native_chat_flow.py) all green; ruff check clean on every touched file; python -c "import tldw_chatbook.app" clean; CSS bundle rebuild reproducible.
<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:NOTES:END -->

<!-- SECTION:PLAN:END -->
