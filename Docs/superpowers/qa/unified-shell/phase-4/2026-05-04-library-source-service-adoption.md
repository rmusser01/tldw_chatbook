# Phase 4.3 Library Source Service Adoption

Task: `TASK-5.3`
Branch: `codex/unified-shell-phase4-library-source-service`

## Goal

Turn the top-level Library destination from static legacy-route links plus generic Console copy into a service-backed source snapshot that tells users whether local notes, media, and conversations are available and stages concrete Library context into Console.

## Implementation Summary

- Loaded local source context through `notes_scope_service.list_notes(scope="local_note")`, `media_reading_scope_service.list_media_items(mode="local")`, and `chat_conversation_scope_service.list_conversations(mode="local")`.
- Rendered loading, available, empty, service-unavailable, and policy-denied recovery states with stable selectors.
- Kept existing legacy navigation buttons for Notes, Media, Conversations, Import/Export, and Search/RAG so current user paths remain reachable.
- Disabled `Use in Console` until concrete local source context exists.
- Built `ChatHandoffPayload` from actual local source counts and sample titles instead of the previous generic Library placeholder.

## Verification

- Baseline focused command before changes: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py -q`
- Baseline focused result: `99 passed, 1 warning in 30.74s` after rerun; the first run had a transient Skills handoff timing failure that passed individually and on full rerun.
- Red command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py -q`
- Red result: `5 failed, 47 passed, 1 warning in 21.22s`.
- First green behavior command after implementation: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py -q`
- First green behavior result: `1 failed, 51 passed, 1 warning in 22.07s`; only the tracking evidence file was still missing.
- Final focused command: `.venv/bin/python -m pytest Tests/UI/test_destination_shells.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_shell_product_model_visibility.py -q`
- Final focused result: `103 passed, 1 warning in 32.21s`.

## QA Walkthrough Notes

- Environment: focused Textual mounted-window tests using the repo virtualenv.
- Entry path: top navigation `Library` destination.
- Visual check: Library keeps the destination title, ownership copy, and existing source navigation buttons while adding a `Local Library snapshot` section.
- Available-state result: local service responses render source counts for Notes, Media, and Conversations plus visible sample titles and enable `Use in Console`.
- Empty-state result: empty local service responses render `No local Library sources are available yet.` and disable Console handoff with add-source recovery copy.
- Service-error result: source-service exceptions render `Library source services unavailable; retry Library later.` and disable Console handoff with retry-oriented recovery copy.
- Functional result: Console handoff stages `library-source-snapshot` with local source counts, sample titles, runtime/source ownership metadata, and no generic placeholder body.

## Residual Risk

- This slice adopts source snapshot and Console staging only; full Library-native note/media/conversation detail views remain future Phase 4 work.
- Import/Export and Search/RAG still route to existing legacy surfaces rather than becoming embedded Library-native pages.
- The walkthrough uses focused mounted-window QA, not a full clean-HOME running app session.
