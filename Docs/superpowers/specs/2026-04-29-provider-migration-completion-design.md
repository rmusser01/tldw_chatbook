# Provider Migration Completion Design

Date: 2026-04-29

Status: Draft for review

Related docs:

- `Docs/superpowers/specs/2026-04-28-remaining-server-parity-roadmap-design.md`
- `Docs/superpowers/specs/2026-04-29-parallel-server-parity-execution-design.md`
- `Docs/superpowers/specs/2026-04-29-connection-auth-foundation-design.md`
- `Docs/superpowers/specs/2026-04-28-next-tranche-parallel-execution-design.md`
- `Docs/Development/server-client-provider-migration-audit.md`

## Purpose

This design defines the next remaining server-parity tranche after the connection/auth and high-priority provider migration work.

The tranche should complete provider-backed server-client construction across the remaining audited services before the roadmap moves to realtime/notifications and domain edge closure. The immediate goal is to remove service-local legacy config-based client construction while preserving public compatibility APIs where callers still depend on them.

## Goals

- Remove remaining service-local calls to legacy `tldw_api` config client builders.
- Preserve public compatibility APIs such as `from_config(...)`, `from_app_config(...)`, and existing helper builders where feasible.
- Route compatibility APIs through `RuntimeServerContextProvider` or a provider-compatible adapter.
- Keep direct injected clients working for tests and narrow internal use.
- Migrate medium and low priority services listed in the migration audit.
- Review explicit UI/event helper holdouts without redesigning current UI surfaces.
- Keep the migration audit semantic, current, and owned by one integration owner.

## Non-Goals

- Do not implement realtime, notification, event observer, or SSE behavior in this tranche.
- Do not implement sync dry-run or write sync behavior in this tranche.
- Do not change domain semantics beyond service construction and client acquisition.
- Do not redesign current UI screens.
- Do not remove public compatibility APIs unless a specific removal is separately approved.
- Do not use the MCP SDK.
- Do not add new direct `build_runtime_api_client_from_config(...)` or `build_runtime_api_client(app_config=...)` consumers.

## Migration Contract

Existing public compatibility APIs remain callable for now. Internally, those APIs should delegate to a provider-backed path instead of directly constructing `TLDWAPIClient` from raw `tldw_api` config inside service modules.

The preferred service shape is:

- Constructor accepts `client: TLDWAPIClient | None`, optional `client_provider`, and optional `policy_enforcer`.
- `from_server_context_provider(provider, ...)` returns an instance with `client=None` and `client_provider=provider`.
- `from_config(app_config, ...)` remains public and delegates through a provider-compatible adapter.
- Direct injected clients remain first priority in `_require_client()`.
- Provider-backed clients are resolved lazily through `client_provider.build_client()`.
- Policy-denied actions should still fail before building a provider client when the service already enforces policy before dispatch.
- Services must not cache auth tokens, base URLs, server profile IDs, or provider-built client instances outside the provider.

If a high-priority service cannot route `from_config(...)` or an equivalent public compatibility API through a provider-compatible adapter, the tranche must stop for explicit follow-up approval before completion. High-priority service-local legacy builders are not allowed to remain as normal audit holdouts.

Medium-priority, low-priority, and UI/event helper paths that cannot safely route through a provider-compatible adapter may remain explicit migration-audit holdouts with a reason and follow-up owner.

## Compatibility Adapter

This tranche should use one shared adapter pattern instead of one-off rewrites in each service.

The preferred adapter lives in `runtime_policy.bootstrap` or a closely related runtime-policy module. It may wrap legacy `app_config` into a provider-compatible object for compatibility-only entry points. That keeps service internals free of direct legacy builder imports while preserving public APIs.

Adapter requirements:

- Expose the same minimal provider behavior services need: `build_client()`.
- Preserve existing public compatibility API signatures and return shapes. This includes positional/keyword arguments, tuple returns from helper builders, async/sync behavior, and the service/client objects existing callers receive.
- Keep secrets out of reprs, logs, and target-store metadata.
- Avoid creating a second active-server authority.
- Reuse `RuntimeServerContextProvider` when runtime context, target store, and credential store are available.
- Fall back to a compatibility provider only for public legacy APIs that lack full runtime-policy dependencies.
- Keep bootstrap/provider construction points explicitly listed in the migration audit as intentional current seams.
- Keep compatibility-provider client caching centralized in the provider or bootstrap adapter. Service instances must not store the first provider-built client in their own `client` field after construction.

## Work Breakdown

### Lane A: High-Priority Holdout Cleanup

Owns the high-priority services that already have provider-backed app wiring but still keep legacy service-local builders.

Services include:

- `Auth_Account_Interop/server_auth_account_service.py`
- `Server_Runtime_Interop/server_runtime_service.py`
- `Chat/server_chat_conversation_service.py`
- `Chat/server_chat_loop_service.py`
- `Character_Chat/server_character_persona_service.py`
- `Character_Chat/server_chat_dictionary_service.py`
- `Media/server_media_reading_service.py`
- `Notes/server_notes_workspace_service.py`
- `Prompt_Management/server_prompt_service.py`
- `Prompt_Management/prompt_scope_service.py`
- `Chatbooks/server_chatbook_service.py`
- `Prompt_Studio_Interop/server_prompt_studio_service.py`

Deliverables:

- Remove service-local direct legacy builder imports.
- Preserve public compatibility factories.
- Establish the shared adapter idiom used by later lanes.
- Keep focused compatibility tests green.

### Lane B: Medium Priority User-Facing Services

Owns medium-priority services from the migration audit.

Services include:

- `Writing_Interop/server_writing_service.py`
- `Research_Interop/server_research_service.py`
- `Research_Interop/server_research_search_service.py`
- `Collections_Interop/server_collections_feeds_service.py`
- `Subscriptions/server_watchlists_service.py`
- `Sharing/server_sharing_service.py`
- `Sharing_Interop/server_sharing_service.py`
- `Outputs/server_outputs_service.py`
- `Outputs_Interop/server_outputs_service.py`
- `WebClipper/server_web_clipper_service.py`
- `Web_Clipper_Interop/server_web_clipper_service.py`
- `Study_Interop/server_study_service.py`
- `Study_Interop/server_quiz_service.py`
- `Kanban_Interop/server_kanban_service.py`
- `Claims_Interop/server_claims_service.py`
- `Meetings_Interop/server_meetings_service.py`
- `Voice_Assistant_Interop/server_voice_assistant_service.py`
- `Companion_Interop/server_companion_service.py`
- `Personalization_Interop/server_personalization_service.py`
- `Notifications/server_notifications_service.py`

`Notifications/server_notifications_service.py` is in scope only for server-client construction migration. Realtime behavior, event observation, SSE/WebSocket transport, notification delivery semantics, and server/local notification authority changes remain deferred to the realtime/notifications tranche.

Deliverables:

- Add `client_provider` support.
- Add `from_server_context_provider(...)`.
- Preserve `from_config(...)` or existing public helpers as provider-backed compatibility adapters.
- Add focused service tests for direct client, provider-backed construction, and compatibility API behavior.

### Lane C: Low Priority Admin, Catalog, And Governance Services

Owns low-priority services from the migration audit.

Services include:

- `Sync_Interop/server_sync_service.py`
- `RAG_Admin/server_rag_admin_service.py`
- `Audio_Services_Interop/server_audio_services_service.py`
- `Evaluations_Interop/server_evaluations_service.py`
- `Chat_Grammars_Interop/server_chat_grammars_service.py`
- `Tools_Interop/server_tools_service.py`
- `Web_Scraping_Interop/server_web_scraping_service.py`
- `External_Connectors_Interop/server_connectors_service.py`
- `User_Governance_Interop/server_user_governance_service.py`
- `MCP_Governance_Interop/server_mcp_governance_service.py`
- `Text2SQL_Interop/server_text2sql_service.py`
- `Skills_Interop/server_skills_service.py`
- `Feedback_Interop/server_feedback_service.py`
- `Translation_Interop/server_translation_service.py`
- `LLM_Provider_Catalog/server_llm_provider_catalog_service.py`

Deliverables:

- Apply the same provider-backed construction pattern as Lane B.
- Keep sync behavior, MCP governance behavior, and catalog behavior unchanged beyond client acquisition.
- Add focused service tests scaled to each service's current test surface.

### Lane D: UI And Event Helper Holdout Review

Owns explicit helper holdouts from the migration audit.

Holdouts include:

- `UI/MediaIngestWindowRebuilt.py`
- `Event_Handlers/tldw_api_events.py`
- `UI/ChatbookExportManagementWindow.py`
- `UI/Wizards/ChatbookImportWizard.py`
- `UI/Wizards/ChatbookCreationWizard.py`

Deliverables:

- Decide whether each helper can use an existing provider from app wiring.
- Move helpers to a tiny provider adapter only when the dependency is already available and the change is low risk.
- Keep helpers listed as explicit holdouts when current UI wiring makes migration too broad for this tranche.
- Do not redesign or broadly refactor current UI code.

### Integration Lane: Provider Migration Audit Owner

Owns shared audit material.

Files:

- `Docs/Development/server-client-provider-migration-audit.md`
- `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

Rules:

- Domain lanes must not edit the shared migration audit directly.
- Domain lanes hand off pending audit deltas to the integration owner.
- The integration owner reconciles the latest direct and indirect scans.
- Audit checks use stable semantic keys: path plus matched builder signature text, or path plus declared builder-class match count.
- Raw line-number-only matching remains forbidden.
- This tranche's Provider Migration Audit Owner supersedes the previous audit document's historical "Lane C Migration-Audit Owner" label. The first audit-owner task must rename that heading in `Docs/Development/server-client-provider-migration-audit.md` before domain migration branches begin.

## Ownership Rules

- Each domain lane owns only its listed service modules and focused tests.
- The shared compatibility adapter is owned by Lane A until it lands, then treated as an integration-owned runtime-policy helper.
- `app.py` changes are out of scope unless a lane identifies an exact small wiring block and gets integration approval.
- UI helper changes are owned only by Lane D.
- Migration audit changes are owned only by the integration lane.
- Any lane that must change a shared runtime-policy helper coordinates through the integration lane before landing.

## Testing Strategy

Required service tests should cover:

- Direct injected client behavior.
- Provider-backed construction through `from_server_context_provider(...)`.
- Public compatibility API behavior through `from_config(...)` or equivalent helper APIs.
- Lazy provider client construction.
- No service-local cache of provider-built clients. Tests should use a fake provider that returns distinct clients or records build calls to prove services keep resolving through the provider rather than copying the built client into service state.
- Policy-denied paths that avoid client construction where existing service behavior supports that ordering.
- Existing action dispatch to the expected API client method and payload shape.
- Public compatibility API shape preservation for existing `from_config(...)`, `from_app_config(...)`, and helper builder call signatures and return shapes.

Required audit tests should cover:

- No new service-local `build_runtime_api_client_from_config(...)` calls.
- No new service-local `build_runtime_api_client(app_config=...)` calls.
- No stale audit entries after legacy builders are removed.
- Explicit bootstrap/provider construction points remain listed separately from remaining migration backlog.
- UI/event helper holdouts remain listed with source and reason when not migrated.

Broad UI tests are not required for this tranche. Service wiring tests are appropriate only where wiring changes.

## Acceptance Criteria

- High-priority service-local legacy builders are removed. Any high-priority exception requires separate explicit approval before the tranche can be marked complete.
- Medium-priority services have provider-backed construction.
- Low-priority services have provider-backed construction.
- Public compatibility APIs keep their existing call signatures and return shapes unless separately approved for removal.
- Service internals no longer import or call direct legacy builders except approved holdouts.
- Service internals do not cache provider-built clients outside the provider or compatibility adapter.
- The migration audit reflects the final semantic scan.
- Focused service and migration-audit tests pass.
- `git diff --check` is clean.
- `python -m compileall tldw_chatbook` exits `0`.

## Follow-On Order

After this tranche lands, the next roadmap work should proceed in this order:

1. Realtime and notifications foundation.
2. Domain edge closure for chat, media/reading, notes/workspaces, writing, and research.
3. Sync dry-run and identity-readiness expansion where domain prerequisites are clear.
