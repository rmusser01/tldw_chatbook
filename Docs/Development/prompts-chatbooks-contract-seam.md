# Prompts / Chatbooks Contract Seam

Date: 2026-04-22

## Scope

This slice adds the first durable local/server contract seam for the `Prompts / Chatbooks` parity row and routes the existing CCP prompt editor through that seam without redesigning the prompt UI.

## Landed

- Typed shared-client support for server prompt pagination, detail, create, update, usage recording, delete, version listing, and version restore.
- Typed shared-client support for server chatbook export/import job listing, status, cancellation, removal, and export archive download.
- A server chatbook service compatibility layer that converts typed client job responses back to plain dictionaries for existing wizard polling and job recording flows, and writes downloaded server export archives into the local chatbooks directory.
- A normalized prompt scope service that routes local/server prompt list, detail, save, delete, and usage actions through explicit backend selection and stable IDs like `local:prompt:<source_id>` and `server:prompt:<source_id>`.
- Source-aware prompt version and restore service methods for the server backend, with explicit local-unavailable behavior because local prompt sync counters are not historical versions.
- App-level prompt scope wiring and CCP prompt handler routing so the existing prompt list/load/create/update/delete flow follows the active runtime source instead of direct local DB access.
- Mounted CCP prompt controls for usage recording, server version listing, and server version restore, with local mode surfacing explicit unavailable status instead of pretending local sync counters are version history.
- Live remote chatbook export/import job browsing in the export management window when a server is configured, merged with the existing locally recorded server-job history.
- Remote server job action controls in the export management window: active server jobs can be cancelled, completed server exports can be downloaded locally, terminal server jobs can be removed, and local-record history rows remain read-only.

## Deferred

- Chatbook cleanup and continuation controls.
- Prompt collection/workflow routes and UI.
- Sync/mirror semantics and cross-backend prompt identity reconciliation.

## Verification

Focused verification command:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/tldw_api/test_prompt_chatbook_client.py Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/Prompt_Management/test_server_prompt_adapter.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Chatbooks/test_server_chatbook_service.py Tests/UI/test_chatbooks_screen_server_actions.py Tests/UI/test_chatbook_management_server_jobs.py Tests/UI/test_ccp_prompt_handler_scope.py Tests/UI/test_ccp_screen.py -q
```

Result: `55 passed in 5.99s`.

Additional live-job UI verification:

```bash
/Users/macbook-dev/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest Tests/UI/test_chatbook_management_server_jobs.py -q
```

Result: `4 passed in 0.58s`.
