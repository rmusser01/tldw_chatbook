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

## Migrated In This Tranche

- `ServerRuntimeService`: app wiring now uses `ServerRuntimeService.from_server_context_provider(...)`. Its `from_config()` and `from_app_config()` factories remain as compatibility shims.
- `ServerAuthAccountService`: app wiring now uses `ServerAuthAccountService.from_server_context_provider(...)`. Its `from_config()` and `from_app_config()` factories remain as compatibility shims.

## Current Audit Command

Generated with:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\(" tldw_chatbook
```

## Compatibility Mode Remaining

### High Priority

These are core interaction, identity, chat, media, note, prompt, and chatbook surfaces where active-server switching and credential freshness are most user-visible.

| Module | Direct builder lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py` | 7, 52 | Migrated app wiring; compatibility factory still calls the legacy config builder. |
| `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py` | 7, 34 | Migrated app wiring; compatibility factory still calls the legacy config builder. |
| `tldw_chatbook/Chat/server_chat_conversation_service.py` | 7, 39 | Service migration target. |
| `tldw_chatbook/Chat/server_chat_loop_service.py` | 7, 19 | Service migration target. |
| `tldw_chatbook/Character_Chat/server_character_persona_service.py` | 9, 46 | Service migration target. |
| `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py` | 27 | Service migration target. |
| `tldw_chatbook/Chatbooks/server_chatbook_service.py` | 9, 31, 174 | Service/helper migration target. |
| `tldw_chatbook/Media/server_media_reading_service.py` | 77, 80 | Service migration target. |
| `tldw_chatbook/Notes/server_notes_workspace_service.py` | 10, 45 | Service migration target. |
| `tldw_chatbook/Prompt_Management/server_prompt_service.py` | 28 | Service migration target. |
| `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py` | 7, 53 | Service migration target. |

### Medium Priority

These are user-facing feature services that should move to provider-backed construction after the core interaction paths are stable.

| Module | Direct builder lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Writing_Interop/server_writing_service.py` | 8, 67 | Service migration target. |
| `tldw_chatbook/Research_Interop/server_research_service.py` | 7, 37 | Service migration target. |
| `tldw_chatbook/Research_Interop/server_research_search_service.py` | 7, 41 | Service migration target. |
| `tldw_chatbook/Collections_Interop/server_collections_feeds_service.py` | 7, 37 | Service migration target. |
| `tldw_chatbook/Subscriptions/server_watchlists_service.py` | 7, 42 | Service migration target. |
| `tldw_chatbook/Sharing/server_sharing_service.py` | 26, 28 | Service migration target. |
| `tldw_chatbook/Sharing_Interop/server_sharing_service.py` | 7, 40 | Service migration target. |
| `tldw_chatbook/Outputs/server_outputs_service.py` | 26, 28 | Service migration target. |
| `tldw_chatbook/Outputs_Interop/server_outputs_service.py` | 7, 39 | Service migration target. |
| `tldw_chatbook/WebClipper/server_web_clipper_service.py` | 22, 24 | Service migration target. |
| `tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py` | 7, 36 | Service migration target. |
| `tldw_chatbook/Study_Interop/server_study_service.py` | 8, 48 | Service migration target. |
| `tldw_chatbook/Study_Interop/server_quiz_service.py` | 8, 31 | Service migration target. |
| `tldw_chatbook/Kanban_Interop/server_kanban_service.py` | 8, 178 | Service migration target. |
| `tldw_chatbook/Claims_Interop/server_claims_service.py` | 7, 47 | Service migration target. |
| `tldw_chatbook/Meetings_Interop/server_meetings_service.py` | 7, 40 | Service migration target. |
| `tldw_chatbook/Voice_Assistant_Interop/server_voice_assistant_service.py` | 7, 38 | Service migration target. |
| `tldw_chatbook/Companion_Interop/server_companion_service.py` | 7, 40 | Service migration target. |
| `tldw_chatbook/Personalization_Interop/server_personalization_service.py` | 7, 40 | Service migration target. |
| `tldw_chatbook/Notifications/server_notifications_service.py` | 7, 38 | Service migration target. |

### Low Priority

These are admin, catalog, governance, integration, or out-of-scope-for-this-workstream services. They still need migration, but can follow the higher traffic service paths.

| Module | Direct builder lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Sync_Interop/server_sync_service.py` | 7, 32 | Service migration target; sync/mirror behavior is outside this workstream. |
| `tldw_chatbook/RAG_Admin/server_rag_admin_service.py` | 8, 40 | Service migration target. |
| `tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py` | 7, 47 | Service migration target. |
| `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py` | 8, 49 | Service migration target. |
| `tldw_chatbook/Chat_Grammars_Interop/server_chat_grammars_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/Tools_Interop/server_tools_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/Web_Scraping_Interop/server_web_scraping_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/External_Connectors_Interop/server_connectors_service.py` | 7, 36 | Service migration target. |
| `tldw_chatbook/User_Governance_Interop/server_user_governance_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/MCP_Governance_Interop/server_mcp_governance_service.py` | 9, 52 | Service migration target. No MCP SDK changes are implied by this audit. |
| `tldw_chatbook/Text2SQL_Interop/server_text2sql_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/Skills_Interop/server_skills_service.py` | 7, 38 | Service migration target. |
| `tldw_chatbook/Feedback_Interop/server_feedback_service.py` | 7, 36 | Service migration target. |
| `tldw_chatbook/Translation_Interop/server_translation_service.py` | 7, 32 | Service migration target. |
| `tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py` | 7, 32 | Service migration target. |

### Event And UI Helpers

These are direct helper call sites rather than service classes. They should be reviewed separately from service constructor migration.

| Module | Direct builder lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/UI/MediaIngestWindowRebuilt.py` | 777 | UI helper call site. |
| `tldw_chatbook/Event_Handlers/tldw_api_events.py` | 572 | Event helper call site. |

### Intentional Current Provider And Bootstrap Usage

These `rg` matches are intentional current seams, not remaining service migration targets.

| Module | Direct builder lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/app.py` | 2145, 2151 | Unified MCP target-specific client factory. This currently builds a client for the selected MCP target and is separate from server service migration. |
| `tldw_chatbook/runtime_policy/server_context.py` | 104 | `RuntimeServerContextProvider.build_client()` provider seam. This is the desired construction point for migrated services. |
| `tldw_chatbook/runtime_policy/bootstrap.py` | 34, 75, 76, 88 | Runtime-policy bootstrap and legacy compatibility helpers. These stay as compatibility seams until consumers are fully migrated. |

## Follow-Up Guardrails

- New server-backed services should prefer accepting a provider/client-provider seam instead of calling `build_runtime_api_client_from_config()` directly.
- Do not add secret fields to `ConfiguredServerTarget` or target-store JSON.
- Compatibility factories can remain temporarily, but app/service wiring should move to `RuntimeServerContextProvider`.
- Provider cache invalidation and token persistence tests should remain the regression boundary for active-server switching.
