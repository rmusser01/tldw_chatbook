# Phase 3.9 Library Collections IA Split QA

Date: 2026-05-08
Status: verified

## Scope

Phase 3.9 verifies the product-model split between the top-level Watchlists destination and Library-owned Collections management. The verified scope is local-first Collections management inside Library plus Watchlists continuity through existing compatibility route IDs.

Out of scope for this gate: server sync, collection item membership picker, collection-scoped Search/RAG execution, collection-scoped Study/Flashcards/Quizzes generation, collection-scoped Console execution, collection Import/Export, and citation/snippet carry-through.

## First-Time User Discovery Check

Verified by mounted Textual navigation and Library screen tests:

- Primary navigation and command-palette copy expose `Watchlists`, not `W+C` or `Watchlists+Collections`.
- Collections is discoverable as a Library mode alongside existing Library workflows.
- The Collections empty state explains the purpose: grouping saved Library items for Search/RAG, Study, and Console.
- Collections mode disables deferred Study, Flashcards, Quizzes, and Console actions instead of implying unavailable workflows are ready.

## Power-User Create/Select/Rename/Delete Workflow

Verified with a mounted Library workflow using a fake local Collections service:

1. Open Library.
2. Switch to Collections mode.
3. Create a collection named `Research` with description `Policy sources`.
4. Confirm the created collection is selected and shows `0 items`, `Sync: local-only`, and a stable updated timestamp.
5. Rename the selected collection to `Briefing Queue`.
6. Start delete, verify confirmation is required, then confirm delete.
7. Confirm the empty state returns after delete.

This proves the gate is usable for local management, not only renderable.

## Watchlists Continuity Check

Verified by destination-shell, navigation, Home active-work, Console live-work, and Unified Shell replay tests:

- `watchlists_collections` remains the compatibility route ID.
- User-facing title, navigation label, and active-work copy say `Watchlists`.
- Watchlists no longer renders Library Collections or read-it-later summaries.
- Home and Console active-run follow-through continue to route fixture-backed Watchlists work through compatibility payloads.

## Sync Honesty Check

Verified by display-state, service, and mounted UI tests:

- Local Collections expose `local-only` when persistence is available.
- Service-unavailable states render recoverable copy without exposing raw database errors.
- `sync-unavailable` is displayed as status only.
- No fake sync button or enabled server-sync action appears.

## Verification Commands

Focused commands run for this gate:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_collections_state.py Tests/Library/test_library_collections_service.py Tests/UI/test_product_maturity_phase39_library_collections.py --tb=short
```

Result: `17 passed`.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_library_contract_layout.py --tb=short
```

Result: `19 passed`.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_library_surfaces_study_workflow_entry_points Tests/UI/test_product_maturity_phase3_knowledge_entry.py::test_library_study_entry_buttons_preserve_requested_section --tb=short
```

Result: `2 passed`.

Final closeout verification:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/Library/test_library_collections_state.py Tests/Library/test_library_collections_service.py Tests/UI/test_product_maturity_phase39_library_collections.py Tests/UI/test_destination_shells.py Tests/UI/test_shell_destinations.py Tests/UI/test_master_shell_navigation.py Tests/UI/test_command_palette_providers.py Tests/Home/test_active_work_adapter.py Tests/UI/test_console_live_work_handoffs.py Tests/UI/test_unified_shell_phase6_first_time_replay.py Tests/UI/test_unified_shell_phase6_power_user_replay.py Tests/UI/test_product_maturity_gate16_library_search_rag.py Tests/UI/test_product_maturity_phase3_layout_contracts.py --tb=short
```

Result: `248 passed, 8 warnings in 131.60s`.

```bash
git diff --check
```

Result: pass.

Clean runtime smoke:

```bash
HOME=/private/tmp/tldw-chatbook-phase39-home XDG_CONFIG_HOME=/private/tmp/tldw-chatbook-phase39-config XDG_DATA_HOME=/private/tmp/tldw-chatbook-phase39-data /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m tldw_chatbook.app
```

Result: app created a fresh config and local databases, reached the Console screen, and exited normally through the quit flow. Observed non-blocking optional dependency/network warnings for NLTK data and optional media/transcription packages. Mounted Textual pilot tests remain the primary interaction evidence for the Library Collections create/select/rename/delete workflow because they exercise deterministic selectors and action states.

## Functional Defects

No P0/P1 functional defects remain in the verified Phase 3.9 scope.

Accepted residual functional gaps:

- Collection item membership is represented as a follow-up seam.
- Collection-scoped Search/RAG, Study, Flashcards, Quizzes, Console, and Import/Export are intentionally disabled or deferred.
- Server sync is not implemented because the sync engine does not exist yet.

## UX Defects

No P0/P1 UX defects remain in the verified Phase 3.9 scope.

Accepted residual UX risks:

- Collections cannot yet demonstrate its full reuse value without membership and scoped downstream workflows.
- Power users still need later shortcuts for adding current Library items to a Collection.
- Clean environment launch was smoke-tested, but mounted Textual workflow tests provide the verified interaction evidence for Phase 3.9 controls.

## Visual/UI Defects

No P0/P1 visual/UI defects remain in the verified Phase 3.9 scope.

Visual scope was bounded to mounted Textual layouts at the tested terminal sizes. Broader compact visual regression remains covered by earlier Library layout and shell visual gates.

## Residual Risks

- Workspaces remain a later Phase 3 workflow gate.
- Import/Export depth remains a later Library workflow gate.
- Full server sync for Collections remains blocked until a sync engine exists.
- Deeper Study/Search/RAG flows for collection-scoped execution remain later-stage.
- Citations/snippets and Citation/snippet carry-through into Chat, artifacts, and exported Chatbooks remain later-stage Library/Search/RAG work.

## Result

Pass for Phase 3.9 only if Library Collections management and Watchlists continuity are both usable in the mounted app.

Phase 3.9 passes for the implemented gate scope based on mounted Library management workflow coverage, Watchlists continuity coverage, roadmap tracking, and focused verification. The residual risks above stay open for later PR-sized gates.
