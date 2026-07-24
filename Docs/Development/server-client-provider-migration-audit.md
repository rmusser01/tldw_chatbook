# Server Client Provider Migration Audit

Date: 2026-04-29

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

- High, medium, and low priority server service modules in this audit tranche have been migrated to provider-backed compatibility adapters.
- Public `from_config(...)` compatibility APIs are preserved as provider-backed adapters using the runtime-policy provider seam. They are no longer classified as ordinary migration holdouts.
- `ServerRuntimeService`, `ServerAuthAccountService`, `ServerChatConversationService`, `ServerCharacterPersonaService`, `ServerChatDictionaryService`, `ServerMediaReadingService`, `ServerNotesWorkspaceService`, `ServerPromptService`, `ServerChatbookService`, `ServerPromptStudioService`, and the remaining tranche service adapters now route through provider-backed construction where their public compatibility APIs remain.
- `tldw_chatbook/UI/server_chatbook_service_lease.py` is provider-backed and is not listed as a legacy holdout because the semantic scan has no legacy builder match in that file.

## Direct Builder Audit Command

Generated with:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\(|build_tldw_api_client_from_config|build_server_chatbook_service|build_server_chatbook_service_from_config|Server[A-Za-z]+Service\.from_config" tldw_chatbook
```

## Audit Key Contract

Allowed raw builder entries are keyed by path plus semantic match signature or call pattern, per-file match count, and reason category. Line numbers are informational operator hints only and must not be the sole allowlist key.

## Remaining Semantic Matches

There are no ordinary domain service holdouts for this tranche. Remaining scan matches are classified below as intentional runtime-policy/provider seams, provider-backed compatibility adapter uses, or the explicit event/UI helper holdout.

### Provider-Backed Compatibility Adapter Uses

These `from_config(...)` call sites continue to use public compatibility APIs, but those APIs now construct through provider-backed adapters rather than direct legacy client builders.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/app.py` | 2708 | App wiring compatibility adapter use (lazy RAG-admin builder, task-254). Semantic match: `server_service = ServerRAGAdminService.from_config(`. |
| `tldw_chatbook/app.py` | 1594 | App wiring compatibility adapter use. Semantic match: `self.server_writing_service = ServerWritingService.from_config(`. |
| `tldw_chatbook/app.py` | 1647 | App wiring compatibility adapter use. Semantic match: `self.server_evaluation_service = ServerEvaluationsService.from_config(`. |
| `tldw_chatbook/app.py` | 1689, 1699 | App wiring compatibility adapter uses. Semantic matches: `self.server_study_service = ServerStudyService.from_config(`, `self.server_quiz_service = ServerQuizService.from_config(`. |
| `tldw_chatbook/app.py` | 1734, 1752, 1951, 1969 | App wiring compatibility adapter uses. Semantic matches: `self.server_research_service = ServerResearchService.from_config(`, `self.server_research_search_service = ServerResearchSearchService.from_config(`, `self.server_research_service = ServerResearchService.from_config(`, `self.server_research_search_service = ServerResearchSearchService.from_config(`. |
| `tldw_chatbook/app.py` | 1773 | App wiring compatibility adapter use. Semantic match: `self.server_watchlists_service = ServerWatchlistsService.from_config(`. |
| `tldw_chatbook/app.py` | 1821, 1835, 1857 | App wiring compatibility adapter uses. Semantic matches: `self.server_claims_service = ServerClaimsService.from_config(`, `self.server_meetings_service = ServerMeetingsService.from_config(`, `self.server_kanban_service = ServerKanbanService.from_config(`. |
| `tldw_chatbook/app.py` | 1871, 1885, 1899, 1913 | App wiring compatibility adapter uses. Semantic matches: `self.server_translation_service = ServerTranslationService.from_config(`, `self.server_voice_assistant_service = ServerVoiceAssistantService.from_config(`, `self.server_companion_service = ServerCompanionService.from_config(`, `self.server_personalization_service = ServerPersonalizationService.from_config(`. |
| `tldw_chatbook/app.py` | 1927, 1988, 2007, 2022 | App wiring compatibility adapter uses. Semantic matches: `self.server_outputs_service = ServerOutputsService.from_config(`, `self.server_chat_grammars_service = ServerChatGrammarsService.from_config(`, `self.server_feedback_service = ServerFeedbackService.from_config(`, `self.server_collections_feeds_service = ServerCollectionsFeedsService.from_config(`. |
| `tldw_chatbook/app.py` | 2037, 2051, 2065, 2079 | App wiring compatibility adapter uses. Semantic matches: `self.server_connectors_service = ServerConnectorsService.from_config(`, `self.server_skills_service = ServerSkillsService.from_config(`, `self.server_tools_service = ServerToolsService.from_config(`, `self.server_mcp_governance_service = ServerMCPGovernanceService.from_config(`. |
| `tldw_chatbook/app.py` | 2144, 2176, 2198 | App wiring compatibility adapter uses. Semantic matches: `self.server_sync_service = ServerSyncService.from_config(`, `self.server_llm_provider_catalog_service = ServerLLMProviderCatalogService.from_config(`, `self.server_audio_services_service = ServerAudioServicesService.from_config(`. |
| `tldw_chatbook/app.py` | 2222, 2236, 2250, 2264 | App wiring compatibility adapter uses. Semantic matches: `self.server_user_governance_service = ServerUserGovernanceService.from_config(`, `self.server_sharing_service = ServerSharingService.from_config(`, `self.server_web_clipper_service = ServerWebClipperService.from_config(`, `self.server_web_scraping_service = ServerWebScrapingService.from_config(`. |
| `tldw_chatbook/Prompt_Management/prompt_scope_service.py` | 691 | Lazy prompt scope compatibility adapter use when no app provider is passed. Semantic match: `return ServerPromptService.from_config(app_config or {})`. |
| `tldw_chatbook/UI/Study_Modules/flashcards_handler.py` | 173 | Fallback study handler compatibility adapter use when the app scope service is not already available. Semantic match: `server_service = ServerStudyService.from_config(getattr(self.app_instance, "app_config", {}) or {})`. |
| `tldw_chatbook/UI/Study_Modules/quizzes_handler.py` | 205 | Fallback quiz handler compatibility adapter use when the app scope service is not already available. Semantic match: `server_service = ServerQuizService.from_config(getattr(self.app_instance, "app_config", {}) or {})`. |
| `tldw_chatbook/Chatbooks/server_chatbook_service.py` | 28, 69 | Chatbook public helper compatibility API definitions now delegate through `build_runtime_api_client_provider_from_config(...)`. Semantic matches: `def build_tldw_api_client_from_config(config: Mapping[str, Any]) -> TLDWAPIClient:`, `def build_server_chatbook_service_from_config(`. |

## Explicit Holdouts

### Event And UI Helpers

These are direct helper call sites rather than service classes. They should be reviewed separately from service constructor migration.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/Event_Handlers/tldw_api_events.py` | 572 | Explicit UI/event helper holdout. Direct endpoint/auth form flow is not safely replaceable by the current app provider without a broader event/UI state refactor. Semantic match: `api_client = build_runtime_api_client(`. |

### Intentional Current Provider And Bootstrap Usage

These `rg` matches are intentional current seams, not remaining service migration targets.

| Module | Audit lines | Notes |
| --- | ---: | --- |
| `tldw_chatbook/app.py` | 2105, 2111 | Unified MCP target-specific client factory. This currently builds a client for the selected MCP target and is separate from server service migration. Semantic matches: `root_client = build_runtime_api_client(`, `root_client = build_runtime_api_client(`. |
| `tldw_chatbook/runtime_policy/server_context.py` | 109 | `RuntimeServerContextProvider.build_client()` provider seam. This is the desired construction point for migrated services. Semantic match: `self._cached_client = build_runtime_api_client(`. |
| `tldw_chatbook/runtime_policy/bootstrap.py` | 34, 75, 76, 89, 105, 114 | Runtime-policy bootstrap and provider-backed legacy compatibility helpers. These stay as compatibility seams while public compatibility APIs are preserved. Semantic matches: `def build_runtime_api_client(`, `def build_runtime_api_client_from_config(app_config: Mapping[str, Any] | None) -> TLDWAPIClient:`, `return build_runtime_api_client(app_config=app_config)`, `self._cached_client = build_runtime_api_client_from_config(self.app_config)`, `def build_server_chatbook_service(`, `client = build_runtime_api_client(app_config=app_config)`. |

## Follow-Up Guardrails

- New server-backed services should prefer accepting a provider/client-provider seam instead of calling `build_runtime_api_client_from_config()` directly.
- Do not add secret fields to `ConfiguredServerTarget` or target-store JSON.
- Compatibility factories can remain temporarily, but app/service wiring should move to `RuntimeServerContextProvider`.
- Provider cache invalidation and token persistence tests should remain the regression boundary for active-server switching.
