# Server Client Provider Migration Audit

Date: 2026-04-28

## Purpose

Track migration from direct legacy `tldw_api` config client construction to `RuntimeServerContextProvider`.

The migration goal is to make active-server selection, credentials, token lifecycle, capability state, and client cache invalidation flow through one runtime-policy authority instead of each server-backed service constructing a `TLDWAPIClient` from `tldw_api` config independently.

## Authoritative Seams

- `RuntimePolicyContext`: authoritative active source/server state.
- `ConfiguredServerTargetStore`: persisted server profile metadata registry. This store must not persist secrets.
- `RuntimeServerContextProvider`: active server context resolution, credential lookup, client construction, and cache invalidation seam for migrated services.
- `ActiveServerCapabilityService`: active-server capability snapshot seam.

## Provider Migration Audit Owner Workflow

`Docs/Development/server-client-provider-migration-audit.md` has one owner for this tranche: the single `Lane C` migration-audit integration owner.

- Domain migration sub-batches must not edit this shared file directly.
- Sub-batches hand off pending audit deltas after their code lands or is ready to land.
- The integration owner reconciles the delta against the latest semantic builder scan, updates this document, and keeps compatibility holdouts explicit.
- Audit enforcement is semantic-first. Line numbers are operator hints only and are allowed to drift as long as the audited builder signatures remain current.

### Pending Audit Delta Handoff Format

Use this exact handoff shape in PR notes, review comments, or the owning integration thread:

```text
Pending audit delta
- Migrated modules:
  - <service/module>: provider-backed seam now used by app/service wiring
- Removed compatibility factories:
  - <path>: <semantic builder signature removed>
- Remaining compatibility factories:
  - <path>: <semantic builder signature still expected>
- Explicit holdouts:
  - <path>: <why this remains outside the current migration batch>
- Notes:
  - constructor/app-wiring follow-ups, merge conflicts, or risks for the audit owner
```

## Migrated Modules

- `ServerRuntimeService`: app wiring now uses `ServerRuntimeService.from_server_context_provider(...)`. Compatibility holdouts remain in `from_config()` and `from_app_config()`.
- `ServerAuthAccountService`: app wiring now uses `ServerAuthAccountService.from_server_context_provider(...)`. Compatibility holdouts remain in `from_config()` and `from_app_config()`.
- `ServerChatConversationService`: app wiring now uses `ServerChatConversationService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerCharacterPersonaService`: app wiring now uses `ServerCharacterPersonaService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerChatDictionaryService`: app wiring now uses `ServerChatDictionaryService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerMediaReadingService`: app wiring now uses `ServerMediaReadingService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerNotesWorkspaceService`: app wiring now uses `ServerNotesWorkspaceService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerPromptService`: app wiring now uses `ServerPromptService.from_server_context_provider(...)`, and `build_prompt_scope_service(...)` now receives the app `RuntimeServerContextProvider`. The `from_config()` factory and legacy prompt-scope fallback remain compatibility holdouts.
- `ServerChatbookService`: app wiring now uses `ServerChatbookService.from_server_context_provider(...)`. Compatibility holdouts remain in `build_tldw_api_client_from_config(...)`, `build_server_chatbook_service_from_config(...)`, and `ServerChatbookService.from_config(...)`.
- `ServerPromptStudioService`: app wiring now uses `ServerPromptStudioService.from_server_context_provider(...)`. The `from_config()` factory remains a compatibility holdout.
- `ServerChatLoopService`: the provider-backed constructor and `from_server_context_provider(...)` seam exist, but there is still no app wiring call site in this tranche. The `from_config()` factory remains a compatibility holdout.

## Direct Builder Audit Command

Generated with:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\(" tldw_chatbook
```

## Indirect Factory Audit Command

The direct builder scan does not catch compatibility wrappers that still construct server clients from `app_config`. The targeted prompt/chatbook indirect factory scan used for this audit is:

```bash
rg -n "build_tldw_api_client_from_config|ServerPromptService\.from_config|build_server_chatbook_service|build_server_chatbook_service_from_config" tldw_chatbook
```

This second scan is intentionally limited to known prompt/chatbook compatibility factories. Broad `app_config` scans are too noisy for this audit, but these wrappers still route through legacy config-based client construction and must remain in the migration backlog.

## Remaining Compatibility Factories

### High Priority

These are core interaction, identity, chat, media, note, prompt, and chatbook surfaces where active-server switching and credential freshness are most user-visible.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py` | 38 | Migrated app wiring; compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client_from_config(self.app_config)`. |
| `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py` | 20 | Migrated app wiring; compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client_from_config(self.app_config)`. |
| `tldw_chatbook/Chat/server_chat_conversation_service.py` | 27 | Migrated app wiring; compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client_from_config(self.app_config)`. |
| `tldw_chatbook/Chat/server_chat_loop_service.py` | 19 | Partially migrated service seam; no app wiring call site in this tranche, and the compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client_from_config(self.app_config)`. |
| `tldw_chatbook/Character_Chat/server_character_persona_service.py` | 39 | Migrated app wiring; compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client_from_config(self.app_config)`. |
| `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py` | 20 | Migrated app wiring; compatibility property builder still calls the legacy config builder on first access. Semantic match: `self._client = build_runtime_api_client(app_config=self.app_config)`. |
| `tldw_chatbook/Chatbooks/server_chatbook_service.py` | 29, 31, 70, 181 | Migrated app wiring; compatibility helpers/factory still call the legacy config builder. Semantic matches: `def build_tldw_api_client_from_config(config: Mapping[str, Any]) -> TLDWAPIClient:`, `return build_runtime_api_client_from_config(config)`, `def build_server_chatbook_service_from_config(`, `build_runtime_api_client_from_config(app_config or {}),`. |
| `tldw_chatbook/Media/server_media_reading_service.py` | 87 | Migrated app wiring; compatibility factory still calls the legacy config builder. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Notes/server_notes_workspace_service.py` | 52 | Migrated app wiring; compatibility factory still calls the legacy config builder. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Prompt_Management/server_prompt_service.py` | 35 | Migrated app wiring; compatibility factory still calls the legacy config builder. Semantic match: `build_runtime_api_client(app_config=app_config),`. |
| `tldw_chatbook/Prompt_Management/prompt_scope_service.py` | 77, 677 | Migrated app wiring passes a provider into `build_prompt_scope_service(...)`; legacy prompt-scope fallback still constructs from `app_config`. Semantic matches: `return cls(client=build_tldw_api_client_from_config(app_config))`, `return ServerPromptService.from_config(app_config or {})`. |
| `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py` | 55 | Migrated app wiring; compatibility factory still calls the legacy config builder. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |

### Medium Priority

These are user-facing feature services that should move to provider-backed construction after the core interaction paths are stable.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Writing_Interop/server_writing_service.py` | 67 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Research_Interop/server_research_service.py` | 37 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Research_Interop/server_research_search_service.py` | 41 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Collections_Interop/server_collections_feeds_service.py` | 37 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Subscriptions/server_watchlists_service.py` | 42 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Sharing/server_sharing_service.py` | 28 | Service migration target. Semantic match: `return cls(client=build_runtime_api_client_from_config(app_config))`. |
| `tldw_chatbook/Sharing_Interop/server_sharing_service.py` | 40 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Outputs/server_outputs_service.py` | 28 | Service migration target. Semantic match: `return cls(client=build_runtime_api_client_from_config(app_config))`. |
| `tldw_chatbook/Outputs_Interop/server_outputs_service.py` | 39 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/WebClipper/server_web_clipper_service.py` | 24 | Service migration target. Semantic match: `return cls(client=build_runtime_api_client_from_config(app_config))`. |
| `tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py` | 36 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Study_Interop/server_study_service.py` | 48 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Study_Interop/server_quiz_service.py` | 31 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Kanban_Interop/server_kanban_service.py` | 178 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Claims_Interop/server_claims_service.py` | 47 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Meetings_Interop/server_meetings_service.py` | 40 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Voice_Assistant_Interop/server_voice_assistant_service.py` | 38 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Companion_Interop/server_companion_service.py` | 40 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Personalization_Interop/server_personalization_service.py` | 40 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Notifications/server_notifications_service.py` | 38 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |

### Low Priority

These are admin, catalog, governance, integration, or out-of-scope-for-this-workstream services. They still need migration, but can follow the higher traffic service paths.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Sync_Interop/server_sync_service.py` | 32 | Service migration target; sync/mirror behavior is outside this workstream. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/RAG_Admin/server_rag_admin_service.py` | 40 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py` | 47 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py` | 49 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Chat_Grammars_Interop/server_chat_grammars_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Tools_Interop/server_tools_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Web_Scraping_Interop/server_web_scraping_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/External_Connectors_Interop/server_connectors_service.py` | 36 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/User_Governance_Interop/server_user_governance_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/MCP_Governance_Interop/server_mcp_governance_service.py` | 52 | Service migration target. No MCP SDK changes are implied by this audit. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Text2SQL_Interop/server_text2sql_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Skills_Interop/server_skills_service.py` | 38 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Feedback_Interop/server_feedback_service.py` | 36 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/Translation_Interop/server_translation_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |
| `tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py` | 32 | Service migration target. Semantic match: `client=build_runtime_api_client_from_config(app_config),`. |

## Explicit Holdouts

### Event And UI Helpers

These are direct helper call sites rather than service classes. They should be reviewed separately from service constructor migration.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/UI/MediaIngestWindowRebuilt.py` | 777 | UI helper call site. Semantic match: `self.api_client = build_runtime_api_client(`. |
| `tldw_chatbook/Event_Handlers/tldw_api_events.py` | 572 | Event helper call site. Semantic match: `api_client = build_runtime_api_client(`. |
| `tldw_chatbook/UI/ChatbookExportManagementWindow.py` | 486, 588 | UI helper indirect chatbook factory consumer through `build_server_chatbook_service_from_config(...)`. Semantic matches: `service, client = build_server_chatbook_service_from_config(`, `service, client = build_server_chatbook_service_from_config(`. |
| `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py` | 706 | UI wizard indirect chatbook factory consumer through `build_server_chatbook_service(...)`. Semantic match: `service = build_server_chatbook_service(app_config=config)`. |
| `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py` | 654 | UI wizard indirect chatbook factory consumer through `build_server_chatbook_service(...)`. Semantic match: `service = build_server_chatbook_service(app_config=config)`. |

### Intentional Current Provider And Bootstrap Usage

These `rg` matches are intentional current seams, not remaining service migration targets.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/app.py` | 2105, 2111 | Unified MCP target-specific client factory. This currently builds a client for the selected MCP target and is separate from server service migration. Semantic matches: `root_client = build_runtime_api_client(`, `root_client = build_runtime_api_client(`. |
| `tldw_chatbook/runtime_policy/server_context.py` | 109 | `RuntimeServerContextProvider.build_client()` provider seam. This is the desired construction point for migrated services. Semantic match: `self._cached_client = build_runtime_api_client(`. |
| `tldw_chatbook/runtime_policy/bootstrap.py` | 34, 75, 76, 79, 88 | Runtime-policy bootstrap and legacy compatibility helpers. These stay as compatibility seams until consumers are fully migrated. Semantic matches: `def build_runtime_api_client(`, `def build_runtime_api_client_from_config(app_config: Mapping[str, Any] | None) -> TLDWAPIClient:`, `return build_runtime_api_client(app_config=app_config)`, `def build_server_chatbook_service(`, `client = build_runtime_api_client(app_config=app_config)`. |

## Follow-Up Guardrails

- New server-backed services should prefer accepting a provider/client-provider seam instead of calling `build_runtime_api_client_from_config()` directly.
- Do not add secret fields to `ConfiguredServerTarget` or target-store JSON.
- Compatibility factories can remain temporarily, but app/service wiring should move to `RuntimeServerContextProvider`.
- Provider cache invalidation and token persistence tests should remain the regression boundary for active-server switching.
