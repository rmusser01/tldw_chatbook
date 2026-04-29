# Provider Migration Completion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove service-local legacy server-client construction across the remaining provider migration audit while preserving public compatibility APIs as provider-backed adapters.

**Architecture:** Add a small provider-compatible adapter in runtime policy/bootstrap code, then migrate services in owned batches so `from_config(...)` and helper builders stay callable without importing direct `tldw_api` config builders inside service modules. Domain lanes do not edit the shared migration audit directly; a final integration task reconciles the semantic audit and guard tests.

**Tech Stack:** Python 3.13, pytest, existing `RuntimeServerContextProvider`, `TLDWAPIClient`, runtime-policy bootstrap helpers, service-level async tests.

---

## Source Documents

- Spec: `Docs/superpowers/specs/2026-04-29-provider-migration-completion-design.md`
- Audit: `Docs/Development/server-client-provider-migration-audit.md`
- Audit guard: `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

## Repo Conventions

- Use the repo venv: `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`.
- Test paths use `Tests/`, not `tests/`.
- Keep UI tests out of this tranche unless a task explicitly changes service wiring.
- Do not use the MCP SDK.
- Do not edit `Docs/Development/server-client-provider-migration-audit.md` from domain migration tasks. Only Task 8 owns that file.

## Shared Provider Pattern

Use this service shape throughout the tranche:

```python
class ExampleServerService:
    def __init__(
        self,
        client: TLDWAPIClient | None,
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ):
        self.client = client
        self.client_provider = client_provider
        self.policy_enforcer = policy_enforcer

    @classmethod
    def from_config(cls, app_config: Mapping[str, Any], **kwargs: Any) -> "ExampleServerService":
        from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config

        return cls(client=None, client_provider=build_runtime_api_client_provider_from_config(app_config), **kwargs)

    @classmethod
    def from_server_context_provider(cls, provider: Any, **kwargs: Any) -> "ExampleServerService":
        return cls(client=None, client_provider=provider, **kwargs)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server operations.")
```

Rules:

- Preserve existing public signatures and return shapes.
- Direct injected clients stay first priority.
- Services must not cache the result of `client_provider.build_client()` in `self.client`.
- `from_config(...)` and public helper builders should use the shared adapter, not direct legacy builders.

## Pending Audit Delta Format For Workers

Every domain task ends with a note in this shape. Do not edit the audit file directly.

```text
Pending audit delta
- Migrated modules:
  - <path>: removed service-local legacy builder, added provider-backed compatibility adapter
- Removed compatibility factories:
  - <path>: <semantic builder signature removed>
- Remaining compatibility factories:
  - none
- Explicit holdouts:
  - <path>: <reason>, only for UI/event helper tasks
- Notes:
  - test command, risks, or exact caller shape preserved
```

---

## Task 1: Shared Compatibility Provider Adapter

**Files:**
- Modify: `tldw_chatbook/runtime_policy/bootstrap.py`
- Test: `Tests/RuntimePolicy/test_runtime_policy_bootstrap.py`

- [ ] **Step 1: Write failing adapter tests**

Add tests proving a compatibility provider exists, builds clients lazily, caches centrally in the adapter, preserves legacy config behavior, and keeps service code from needing direct builders.

```python
def test_config_client_provider_builds_legacy_client_lazily():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_provider_from_config

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )

    first = provider.build_client()
    second = provider.build_client()

    assert first is second
    assert first.base_url == "https://example.test"


def test_config_client_provider_repr_redacts_config_secrets():
    from tldw_chatbook.runtime_policy.bootstrap import build_runtime_api_client_provider_from_config

    provider = build_runtime_api_client_provider_from_config(
        {"tldw_api": {"base_url": "https://example.test", "api_key": "secret"}}
    )

    assert "secret" not in repr(provider)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: FAIL because `build_runtime_api_client_provider_from_config` does not exist.

- [ ] **Step 3: Implement the adapter**

In `tldw_chatbook/runtime_policy/bootstrap.py`, add a small provider-compatible class and factory near the existing client builders.

```python
@dataclass(slots=True)
class LegacyConfigServerClientProvider:
    app_config: Mapping[str, Any] | None
    _cached_client: TLDWAPIClient | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(app_config=<redacted>)"

    def build_client(self) -> TLDWAPIClient:
        if self._cached_client is None:
            self._cached_client = build_runtime_api_client_from_config(self.app_config)
        return self._cached_client

    async def close_cached_client(self) -> None:
        cached_client = self._cached_client
        self._cached_client = None
        if cached_client is not None:
            await cached_client.close()


def build_runtime_api_client_provider_from_config(
    app_config: Mapping[str, Any] | None,
) -> LegacyConfigServerClientProvider:
    return LegacyConfigServerClientProvider(app_config=app_config)
```

If `dataclass` is already imported, reuse it. If not, add it. Keep `build_runtime_api_client_from_config(...)` in bootstrap as an intentional compatibility seam for the adapter.

- [ ] **Step 4: Run tests and verify they pass**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RuntimePolicy/test_runtime_policy_bootstrap.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/runtime_policy/bootstrap.py Tests/RuntimePolicy/test_runtime_policy_bootstrap.py
git commit -m "feat: add provider-compatible legacy config adapter"
```

---

## Task 2: High-Priority Holdout Cleanup

**Files:**
- Modify: `tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py`
- Modify: `tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py`
- Modify: `tldw_chatbook/Chat/server_chat_conversation_service.py`
- Modify: `tldw_chatbook/Chat/server_chat_loop_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_character_persona_service.py`
- Modify: `tldw_chatbook/Character_Chat/server_chat_dictionary_service.py`
- Modify: `tldw_chatbook/Media/server_media_reading_service.py`
- Modify: `tldw_chatbook/Notes/server_notes_workspace_service.py`
- Modify: `tldw_chatbook/Prompt_Management/server_prompt_service.py`
- Modify: `tldw_chatbook/Prompt_Management/prompt_scope_service.py`
- Modify: `tldw_chatbook/Chatbooks/server_chatbook_service.py`
- Modify: `tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py`
- Test: existing focused tests under `Tests/Auth_Account`, `Tests/Server_Runtime`, `Tests/Chat`, `Tests/Character_Chat`, `Tests/Media`, `Tests/Notes`, `Tests/Prompt_Management`, `Tests/Chatbooks`, `Tests/Prompt_Studio`

- [ ] **Step 1: Write failing tests for high-priority compatibility behavior**

For each service family, add or extend tests to prove:

- `from_config(...)` still returns the same service type.
- `from_config(...)` does not call service-local `build_runtime_api_client_from_config(...)`.
- Provider-built clients are lazy.
- A fake provider returning distinct clients is invoked on each action, or existing provider cache is external to the service.

Use this test shape where practical:

```python
class RecordingProvider:
    def __init__(self):
        self.calls = 0

    def build_client(self):
        self.calls += 1
        return FakeClient()


async def test_service_uses_provider_without_service_local_client_cache():
    provider = RecordingProvider()
    service = ServerExampleService.from_server_context_provider(provider)

    await service.some_existing_list_action()
    await service.some_existing_list_action()

    assert provider.calls == 2
    assert getattr(service, "client", None) is None
```

- [ ] **Step 2: Run the focused high-priority tests and verify failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Auth_Account/test_server_auth_account_service.py \
  Tests/Server_Runtime/test_server_runtime_service.py \
  Tests/RuntimePolicy/test_server_runtime_service_compatibility.py \
  Tests/Chat/test_server_chat_conversation_service.py \
  Tests/Chat/test_server_chat_loop_service.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_server_chat_dictionary_service.py \
  Tests/Media/test_server_media_reading_service.py \
  Tests/Notes/test_server_notes_workspace_service.py \
  Tests/Prompt_Management/test_server_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Chatbooks/test_server_chatbook_service.py \
  Tests/Prompt_Studio/test_server_prompt_studio_service.py -q
```

Expected: FAIL on newly added expectations.

- [ ] **Step 3: Replace service-local legacy builders with the shared adapter**

For each listed service:

- Remove imports of `build_runtime_api_client_from_config` or `build_runtime_api_client(app_config=...)` from the service module.
- Preserve public factories and helper names.
- Use `build_runtime_api_client_provider_from_config(app_config)` inside `from_config(...)` or helper builders.
- Ensure `_require_client()` resolves direct client first, then `client_provider.build_client()`, then raises the current service error.
- Do not assign provider-built clients into service state.

For helpers that return `(service, client)`, preserve the tuple shape. If the public helper historically returned both service and client, have the helper build a provider, call `provider.build_client()` once for the returned client, and construct the service with the same provider or direct client only if that exact return shape requires it. Add a note for the audit owner if the helper remains an intentional bootstrap-like compatibility surface.

- [ ] **Step 4: Run the focused high-priority tests**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Confirm no high-priority service-local legacy builders remain**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" \
  tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py \
  tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py \
  tldw_chatbook/Chat/server_chat_conversation_service.py \
  tldw_chatbook/Chat/server_chat_loop_service.py \
  tldw_chatbook/Character_Chat/server_character_persona_service.py \
  tldw_chatbook/Character_Chat/server_chat_dictionary_service.py \
  tldw_chatbook/Media/server_media_reading_service.py \
  tldw_chatbook/Notes/server_notes_workspace_service.py \
  tldw_chatbook/Prompt_Management/server_prompt_service.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Chatbooks/server_chatbook_service.py \
  tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py
```

Expected: no output, except public helper names that do not call legacy builders.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_chatbook/Auth_Account_Interop/server_auth_account_service.py \
  tldw_chatbook/Server_Runtime_Interop/server_runtime_service.py \
  tldw_chatbook/Chat/server_chat_conversation_service.py \
  tldw_chatbook/Chat/server_chat_loop_service.py \
  tldw_chatbook/Character_Chat/server_character_persona_service.py \
  tldw_chatbook/Character_Chat/server_chat_dictionary_service.py \
  tldw_chatbook/Media/server_media_reading_service.py \
  tldw_chatbook/Notes/server_notes_workspace_service.py \
  tldw_chatbook/Prompt_Management/server_prompt_service.py \
  tldw_chatbook/Prompt_Management/prompt_scope_service.py \
  tldw_chatbook/Chatbooks/server_chatbook_service.py \
  tldw_chatbook/Prompt_Studio_Interop/server_prompt_studio_service.py \
  Tests/Auth_Account/test_server_auth_account_service.py \
  Tests/Server_Runtime/test_server_runtime_service.py \
  Tests/RuntimePolicy/test_server_runtime_service_compatibility.py \
  Tests/Chat/test_server_chat_conversation_service.py \
  Tests/Chat/test_server_chat_loop_service.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_server_chat_dictionary_service.py \
  Tests/Media/test_server_media_reading_service.py \
  Tests/Notes/test_server_notes_workspace_service.py \
  Tests/Prompt_Management/test_server_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Chatbooks/test_server_chatbook_service.py \
  Tests/Prompt_Studio/test_server_prompt_studio_service.py
git commit -m "refactor: remove high-priority legacy client builders"
```

---

## Task 3: Medium Batch 1, Writing Research Collections Watchlists

**Files:**
- Modify: `tldw_chatbook/Writing_Interop/server_writing_service.py`
- Modify: `tldw_chatbook/Research_Interop/server_research_service.py`
- Modify: `tldw_chatbook/Research_Interop/server_research_search_service.py`
- Modify: `tldw_chatbook/Collections_Interop/server_collections_feeds_service.py`
- Modify: `tldw_chatbook/Subscriptions/server_watchlists_service.py`
- Test: `Tests/Writing_Interop/test_server_writing_service.py`
- Test: `Tests/Research/test_server_research_service.py`
- Test: `Tests/Research/test_server_research_search_service.py`
- Test: `Tests/Collections/test_server_collections_feeds_service.py`
- Test: `Tests/Subscriptions/test_server_watchlists_service.py`

- [ ] **Step 1: Add provider-construction tests**

For each service, add tests proving direct injected client behavior, `from_server_context_provider(...)`, `from_config(...)` compatibility, and no service-local provider-built client cache.

- [ ] **Step 2: Run focused tests and verify failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Writing_Interop/test_server_writing_service.py \
  Tests/Research/test_server_research_service.py \
  Tests/Research/test_server_research_search_service.py \
  Tests/Collections/test_server_collections_feeds_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py -q
```

Expected: FAIL on missing provider-backed construction.

- [ ] **Step 3: Implement provider-backed construction**

For each service:

- Add `client_provider: Any | None = None` to `__init__`.
- Store `self.client_provider`.
- Add `from_server_context_provider(...)`.
- Change `from_config(...)` to use `build_runtime_api_client_provider_from_config(app_config)`.
- Change `_require_client()` to direct client first, provider second, existing error third.
- Preserve all existing method signatures and payload behavior.

- [ ] **Step 4: Run focused tests**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Confirm legacy builders are gone from this batch**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" \
  tldw_chatbook/Writing_Interop/server_writing_service.py \
  tldw_chatbook/Research_Interop/server_research_service.py \
  tldw_chatbook/Research_Interop/server_research_search_service.py \
  tldw_chatbook/Collections_Interop/server_collections_feeds_service.py \
  tldw_chatbook/Subscriptions/server_watchlists_service.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_chatbook/Writing_Interop/server_writing_service.py \
  tldw_chatbook/Research_Interop/server_research_service.py \
  tldw_chatbook/Research_Interop/server_research_search_service.py \
  tldw_chatbook/Collections_Interop/server_collections_feeds_service.py \
  tldw_chatbook/Subscriptions/server_watchlists_service.py \
  Tests/Writing_Interop/test_server_writing_service.py \
  Tests/Research/test_server_research_service.py \
  Tests/Research/test_server_research_search_service.py \
  Tests/Collections/test_server_collections_feeds_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py
git commit -m "refactor: migrate writing research and collection services"
```

---

## Task 4: Medium Batch 2, Sharing Outputs Web Clipper Study

**Files:**
- Modify: `tldw_chatbook/Sharing/server_sharing_service.py`
- Modify: `tldw_chatbook/Sharing_Interop/server_sharing_service.py`
- Modify: `tldw_chatbook/Outputs/server_outputs_service.py`
- Modify: `tldw_chatbook/Outputs_Interop/server_outputs_service.py`
- Modify: `tldw_chatbook/WebClipper/server_web_clipper_service.py`
- Modify: `tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py`
- Modify: `tldw_chatbook/Study_Interop/server_study_service.py`
- Modify: `tldw_chatbook/Study_Interop/server_quiz_service.py`
- Test: `Tests/Sharing/test_server_sharing_service.py`
- Test: `Tests/Outputs/test_server_outputs_service.py`
- Test: `Tests/WebClipper/test_server_web_clipper_service.py`
- Test: `Tests/Study_Interop/test_server_study_service.py`
- Test: `Tests/Study_Interop/test_server_quiz_service.py`

- [ ] **Step 1: Add provider-construction tests**

Add tests for direct client, provider-backed construction, compatibility API preservation, and no service-local provider-built client cache. For duplicate module pairs such as `Sharing` and `Sharing_Interop`, test both public import paths if both are used by callers.

- [ ] **Step 2: Run focused tests and verify failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Sharing/test_server_sharing_service.py \
  Tests/Outputs/test_server_outputs_service.py \
  Tests/WebClipper/test_server_web_clipper_service.py \
  Tests/Study_Interop/test_server_study_service.py \
  Tests/Study_Interop/test_server_quiz_service.py -q
```

Expected: FAIL on new provider expectations.

- [ ] **Step 3: Implement provider-backed construction**

Apply the shared provider pattern to all listed service modules. Preserve public return shapes for simple `from_config(...)` classmethods and any helper functions.

- [ ] **Step 4: Run focused tests**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Confirm legacy builders are gone from this batch**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" \
  tldw_chatbook/Sharing/server_sharing_service.py \
  tldw_chatbook/Sharing_Interop/server_sharing_service.py \
  tldw_chatbook/Outputs/server_outputs_service.py \
  tldw_chatbook/Outputs_Interop/server_outputs_service.py \
  tldw_chatbook/WebClipper/server_web_clipper_service.py \
  tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py \
  tldw_chatbook/Study_Interop/server_study_service.py \
  tldw_chatbook/Study_Interop/server_quiz_service.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_chatbook/Sharing/server_sharing_service.py \
  tldw_chatbook/Sharing_Interop/server_sharing_service.py \
  tldw_chatbook/Outputs/server_outputs_service.py \
  tldw_chatbook/Outputs_Interop/server_outputs_service.py \
  tldw_chatbook/WebClipper/server_web_clipper_service.py \
  tldw_chatbook/Web_Clipper_Interop/server_web_clipper_service.py \
  tldw_chatbook/Study_Interop/server_study_service.py \
  tldw_chatbook/Study_Interop/server_quiz_service.py \
  Tests/Sharing/test_server_sharing_service.py \
  Tests/Outputs/test_server_outputs_service.py \
  Tests/WebClipper/test_server_web_clipper_service.py \
  Tests/Study_Interop/test_server_study_service.py \
  Tests/Study_Interop/test_server_quiz_service.py
git commit -m "refactor: migrate sharing output web clipper and study services"
```

---

## Task 5: Medium Batch 3, Operational User-Facing Services

**Files:**
- Modify: `tldw_chatbook/Kanban_Interop/server_kanban_service.py`
- Modify: `tldw_chatbook/Claims_Interop/server_claims_service.py`
- Modify: `tldw_chatbook/Meetings_Interop/server_meetings_service.py`
- Modify: `tldw_chatbook/Voice_Assistant_Interop/server_voice_assistant_service.py`
- Modify: `tldw_chatbook/Companion_Interop/server_companion_service.py`
- Modify: `tldw_chatbook/Personalization_Interop/server_personalization_service.py`
- Modify: `tldw_chatbook/Notifications/server_notifications_service.py`
- Test: `Tests/Kanban/test_server_kanban_service.py`
- Test: `Tests/Claims/test_server_claims_service.py`
- Test: `Tests/Meetings/test_server_meetings_service.py`
- Test: `Tests/Voice_Assistant/test_server_voice_assistant_service.py`
- Test: `Tests/Companion/test_server_companion_service.py`
- Test: `Tests/Companion/test_server_personalization_service.py`
- Test: `Tests/Notifications/test_server_notifications_service.py`

- [ ] **Step 1: Add provider-construction tests**

Add construction-only provider tests. For `Notifications/server_notifications_service.py`, do not add realtime, SSE, WebSocket, event delivery, or notification authority tests.

- [ ] **Step 2: Run focused tests and verify failures**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/Kanban/test_server_kanban_service.py \
  Tests/Claims/test_server_claims_service.py \
  Tests/Meetings/test_server_meetings_service.py \
  Tests/Voice_Assistant/test_server_voice_assistant_service.py \
  Tests/Companion/test_server_companion_service.py \
  Tests/Companion/test_server_personalization_service.py \
  Tests/Notifications/test_server_notifications_service.py -q
```

Expected: FAIL on new provider expectations.

- [ ] **Step 3: Implement provider-backed construction**

Apply the shared provider pattern. Keep notification behavior unchanged beyond client acquisition.

- [ ] **Step 4: Run focused tests**

Run the same pytest command from Step 2.

Expected: PASS.

- [ ] **Step 5: Confirm legacy builders are gone from this batch**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" \
  tldw_chatbook/Kanban_Interop/server_kanban_service.py \
  tldw_chatbook/Claims_Interop/server_claims_service.py \
  tldw_chatbook/Meetings_Interop/server_meetings_service.py \
  tldw_chatbook/Voice_Assistant_Interop/server_voice_assistant_service.py \
  tldw_chatbook/Companion_Interop/server_companion_service.py \
  tldw_chatbook/Personalization_Interop/server_personalization_service.py \
  tldw_chatbook/Notifications/server_notifications_service.py
```

Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add \
  tldw_chatbook/Kanban_Interop/server_kanban_service.py \
  tldw_chatbook/Claims_Interop/server_claims_service.py \
  tldw_chatbook/Meetings_Interop/server_meetings_service.py \
  tldw_chatbook/Voice_Assistant_Interop/server_voice_assistant_service.py \
  tldw_chatbook/Companion_Interop/server_companion_service.py \
  tldw_chatbook/Personalization_Interop/server_personalization_service.py \
  tldw_chatbook/Notifications/server_notifications_service.py \
  Tests/Kanban/test_server_kanban_service.py \
  Tests/Claims/test_server_claims_service.py \
  Tests/Meetings/test_server_meetings_service.py \
  Tests/Voice_Assistant/test_server_voice_assistant_service.py \
  Tests/Companion/test_server_companion_service.py \
  Tests/Companion/test_server_personalization_service.py \
  Tests/Notifications/test_server_notifications_service.py
git commit -m "refactor: migrate operational user-facing server services"
```

---

## Task 6: Low Priority Admin Catalog Governance Services

**Files:**
- Modify: `tldw_chatbook/Sync_Interop/server_sync_service.py`
- Modify: `tldw_chatbook/RAG_Admin/server_rag_admin_service.py`
- Modify: `tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py`
- Modify: `tldw_chatbook/Evaluations_Interop/server_evaluations_service.py`
- Modify: `tldw_chatbook/Chat_Grammars_Interop/server_chat_grammars_service.py`
- Modify: `tldw_chatbook/Tools_Interop/server_tools_service.py`
- Modify: `tldw_chatbook/Web_Scraping_Interop/server_web_scraping_service.py`
- Modify: `tldw_chatbook/External_Connectors_Interop/server_connectors_service.py`
- Modify: `tldw_chatbook/User_Governance_Interop/server_user_governance_service.py`
- Modify: `tldw_chatbook/MCP_Governance_Interop/server_mcp_governance_service.py`
- Modify: `tldw_chatbook/Text2SQL_Interop/server_text2sql_service.py`
- Modify: `tldw_chatbook/Skills_Interop/server_skills_service.py`
- Modify: `tldw_chatbook/Feedback_Interop/server_feedback_service.py`
- Modify: `tldw_chatbook/Translation_Interop/server_translation_service.py`
- Modify: `tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py`
- Test: create `Tests/Sync_Interop/test_server_sync_service.py` if no focused sync service test exists
- Test: existing focused service tests under `Tests/RAG_Admin`, `Tests/Audio_Services`, `Tests/Evaluations_Interop`, `Tests/Chat_Grammars`, `Tests/Tools_Interop`, `Tests/Web_Scraping_Interop`, `Tests/External_Connectors`, `Tests/User_Governance`, `Tests/MCP_Governance`, `Tests/Text2SQL_Interop`, `Tests/Skills`, `Tests/Feedback`, `Tests/Translation`, and `Tests/LLM_Provider_Catalog`

- [ ] **Step 1: Locate exact existing tests**

Run:

```bash
rg --files Tests | rg "(server_sync|server_rag_admin|server_audio_services|server_evaluations|server_chat_grammars|server_tools|server_web_scraping|server_connectors|server_user_governance|server_mcp_governance|server_text2sql|server_skills|server_feedback|server_translation|server_llm_provider_catalog)"
```

Expected: lists existing focused service tests. If a specific service has no test file, create one in the matching `Tests/<Domain>/` directory.

- [ ] **Step 2: Add provider-construction tests**

For each service, add tests for direct client, provider construction, compatibility API preservation, and no service-local provider-built client cache.

For `Sync_Interop/server_sync_service.py`, test only constructor/client acquisition. Do not add sync behavior, dry-run, outbox, mirror, or cursor behavior.

For `MCP_Governance_Interop/server_mcp_governance_service.py`, do not use the MCP SDK and do not change governance semantics beyond client acquisition.

- [ ] **Step 3: Run focused low-priority tests and verify failures**

Run each existing focused service test file discovered in Step 1 with:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <test-files> -q
```

Expected: FAIL on new provider expectations.

- [ ] **Step 4: Implement provider-backed construction**

Apply the shared provider pattern to all listed low-priority service modules. Preserve behavior and public signatures.

- [ ] **Step 5: Run focused low-priority tests**

Run the same pytest command from Step 3.

Expected: PASS.

- [ ] **Step 6: Confirm legacy builders are gone from this batch**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" \
  tldw_chatbook/Sync_Interop/server_sync_service.py \
  tldw_chatbook/RAG_Admin/server_rag_admin_service.py \
  tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py \
  tldw_chatbook/Evaluations_Interop/server_evaluations_service.py \
  tldw_chatbook/Chat_Grammars_Interop/server_chat_grammars_service.py \
  tldw_chatbook/Tools_Interop/server_tools_service.py \
  tldw_chatbook/Web_Scraping_Interop/server_web_scraping_service.py \
  tldw_chatbook/External_Connectors_Interop/server_connectors_service.py \
  tldw_chatbook/User_Governance_Interop/server_user_governance_service.py \
  tldw_chatbook/MCP_Governance_Interop/server_mcp_governance_service.py \
  tldw_chatbook/Text2SQL_Interop/server_text2sql_service.py \
  tldw_chatbook/Skills_Interop/server_skills_service.py \
  tldw_chatbook/Feedback_Interop/server_feedback_service.py \
  tldw_chatbook/Translation_Interop/server_translation_service.py \
  tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add \
  tldw_chatbook/Sync_Interop/server_sync_service.py \
  tldw_chatbook/RAG_Admin/server_rag_admin_service.py \
  tldw_chatbook/Audio_Services_Interop/server_audio_services_service.py \
  tldw_chatbook/Evaluations_Interop/server_evaluations_service.py \
  tldw_chatbook/Chat_Grammars_Interop/server_chat_grammars_service.py \
  tldw_chatbook/Tools_Interop/server_tools_service.py \
  tldw_chatbook/Web_Scraping_Interop/server_web_scraping_service.py \
  tldw_chatbook/External_Connectors_Interop/server_connectors_service.py \
  tldw_chatbook/User_Governance_Interop/server_user_governance_service.py \
  tldw_chatbook/MCP_Governance_Interop/server_mcp_governance_service.py \
  tldw_chatbook/Text2SQL_Interop/server_text2sql_service.py \
  tldw_chatbook/Skills_Interop/server_skills_service.py \
  tldw_chatbook/Feedback_Interop/server_feedback_service.py \
  tldw_chatbook/Translation_Interop/server_translation_service.py \
  tldw_chatbook/LLM_Provider_Catalog/server_llm_provider_catalog_service.py \
  Tests
git commit -m "refactor: migrate admin catalog and governance server services"
```

---

## Task 7: UI And Event Helper Holdout Review

**Files:**
- Review: `tldw_chatbook/UI/MediaIngestWindowRebuilt.py`
- Review: `tldw_chatbook/Event_Handlers/tldw_api_events.py`
- Review: `tldw_chatbook/UI/ChatbookExportManagementWindow.py`
- Review: `tldw_chatbook/UI/Wizards/ChatbookImportWizard.py`
- Review: `tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py`
- Test: add or update focused service-wiring tests only if code changes

- [ ] **Step 1: Inspect helper dependency availability**

For each helper holdout, determine whether a `RuntimeServerContextProvider` is already available in the object or app wiring without broad UI refactor.

Run:

```bash
rg -n "server_context_provider|build_runtime_api_client|build_server_chatbook_service|build_server_chatbook_service_from_config" \
  tldw_chatbook/UI/MediaIngestWindowRebuilt.py \
  tldw_chatbook/Event_Handlers/tldw_api_events.py \
  tldw_chatbook/UI/ChatbookExportManagementWindow.py \
  tldw_chatbook/UI/Wizards/ChatbookImportWizard.py \
  tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py
```

- [ ] **Step 2: Classify each helper**

Classify each file as one of:

- `migrate-now`: provider is already available and change is local.
- `holdout`: provider is not available without broad UI/event refactor.
- `no-live-builder`: audit entry is stale after service-helper migration.

- [ ] **Step 3: Implement migrate-now helpers only**

For `migrate-now` helpers, route through the existing provider or provider-backed helper while preserving public behavior. Do not redesign screens, wizard flows, event flow, or UI state.

- [ ] **Step 4: Add focused tests only for changed helpers**

If a helper changed, add or update a focused test proving it consumes a provider-backed service/helper and preserves return shape.

- [ ] **Step 5: Run focused tests or static checks**

Run focused tests for changed helper areas. If no code changed, run:

```bash
rg -n "build_runtime_api_client|build_server_chatbook_service|build_server_chatbook_service_from_config" \
  tldw_chatbook/UI/MediaIngestWindowRebuilt.py \
  tldw_chatbook/Event_Handlers/tldw_api_events.py \
  tldw_chatbook/UI/ChatbookExportManagementWindow.py \
  tldw_chatbook/UI/Wizards/ChatbookImportWizard.py \
  tldw_chatbook/UI/Wizards/ChatbookCreationWizard.py
```

Expected: output is understood and ready for Task 8 audit update.

- [ ] **Step 6: Commit**

If code changed:

```bash
git add <changed-helper-files> <changed-test-files>
git commit -m "refactor: review provider helper holdouts"
```

If no code changed, do not create an empty commit. Hand off explicit holdout classifications to Task 8.

---

## Task 8: Migration Audit Integration And Guard Update

**Files:**
- Modify: `Docs/Development/server-client-provider-migration-audit.md`
- Modify: `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py`

- [ ] **Step 1: Rename the audit-owner heading**

In `Docs/Development/server-client-provider-migration-audit.md`, rename `Lane C Migration-Audit Owner Workflow` to `Provider Migration Audit Owner Workflow`.

- [ ] **Step 2: Run the current semantic scan**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(|build_tldw_api_client_from_config|build_server_chatbook_service|build_server_chatbook_service_from_config|Server[A-Za-z]+Service\\.from_config" tldw_chatbook
```

Classify every remaining match as one of:

- intentional bootstrap/provider adapter seam
- explicit UI/event helper holdout
- stale audit row to remove
- failure that must go back to the owning service lane

- [ ] **Step 3: Update the audit document**

Expected final shape:

- High-priority service rows removed from remaining compatibility factories.
- Medium-priority service rows removed from remaining compatibility factories.
- Low-priority service rows removed from remaining compatibility factories.
- Intentional bootstrap/provider adapter seams listed separately.
- UI/event helper holdouts listed with source and reason if they remain.
- No domain service row remains as an ordinary holdout without explicit human approval.

- [ ] **Step 4: Update the audit guard if needed**

If the guard's indirect matcher no longer reflects remaining holdouts, update `Tests/RuntimePolicy/test_server_client_provider_migration_audit.py` so it enforces the new baseline with semantic matching. Keep line-number-only matching forbidden.

- [ ] **Step 5: Run audit guard**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/RuntimePolicy/test_server_client_provider_migration_audit.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add Docs/Development/server-client-provider-migration-audit.md Tests/RuntimePolicy/test_server_client_provider_migration_audit.py
git commit -m "chore: finalize provider migration audit baseline"
```

---

## Task 9: Final Tranche Verification

**Files:**
- Verify all files changed by Tasks 1-8.

- [ ] **Step 1: Run focused construction and audit suite**

Run:

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest \
  Tests/RuntimePolicy/test_runtime_policy_bootstrap.py \
  Tests/RuntimePolicy/test_server_client_provider_migration_audit.py \
  Tests/Auth_Account/test_server_auth_account_service.py \
  Tests/Server_Runtime/test_server_runtime_service.py \
  Tests/RuntimePolicy/test_server_runtime_service_compatibility.py \
  Tests/Chat/test_server_chat_conversation_service.py \
  Tests/Chat/test_server_chat_loop_service.py \
  Tests/Character_Chat/test_character_persona_scope_service.py \
  Tests/Character_Chat/test_server_chat_dictionary_service.py \
  Tests/Media/test_server_media_reading_service.py \
  Tests/Notes/test_server_notes_workspace_service.py \
  Tests/Prompt_Management/test_server_prompt_service.py \
  Tests/Prompt_Management/test_prompt_scope_service.py \
  Tests/Chatbooks/test_server_chatbook_service.py \
  Tests/Prompt_Studio/test_server_prompt_studio_service.py \
  Tests/Writing_Interop/test_server_writing_service.py \
  Tests/Research/test_server_research_service.py \
  Tests/Research/test_server_research_search_service.py \
  Tests/Collections/test_server_collections_feeds_service.py \
  Tests/Subscriptions/test_server_watchlists_service.py \
  Tests/Sharing/test_server_sharing_service.py \
  Tests/Outputs/test_server_outputs_service.py \
  Tests/WebClipper/test_server_web_clipper_service.py \
  Tests/Study_Interop/test_server_study_service.py \
  Tests/Study_Interop/test_server_quiz_service.py \
  Tests/Kanban/test_server_kanban_service.py \
  Tests/Claims/test_server_claims_service.py \
  Tests/Meetings/test_server_meetings_service.py \
  Tests/Voice_Assistant/test_server_voice_assistant_service.py \
  Tests/Companion/test_server_companion_service.py \
  Tests/Companion/test_server_personalization_service.py \
  Tests/Notifications/test_server_notifications_service.py \
  Tests/RAG_Admin/test_server_rag_admin_service.py \
  Tests/Audio_Services/test_server_audio_services_service.py \
  Tests/Evaluations_Interop/test_server_evaluations_service.py \
  Tests/Chat_Grammars/test_server_chat_grammars_service.py \
  Tests/Tools_Interop/test_server_tools_service.py \
  Tests/Web_Scraping_Interop/test_server_web_scraping_service.py \
  Tests/External_Connectors/test_server_connectors_service.py \
  Tests/User_Governance/test_server_user_governance_service.py \
  Tests/MCP_Governance/test_server_mcp_governance_service.py \
  Tests/Text2SQL_Interop/test_server_text2sql_service.py \
  Tests/Skills/test_server_skills_service.py \
  Tests/Feedback/test_server_feedback_service.py \
  Tests/Translation/test_server_translation_service.py \
  Tests/LLM_Provider_Catalog/test_server_llm_provider_catalog_service.py -q
```

Expected: PASS.

- [ ] **Step 2: Run final legacy builder scan**

Run:

```bash
rg -n "build_runtime_api_client_from_config|build_runtime_api_client\\(app_config" tldw_chatbook
```

Expected: only intentional bootstrap/provider adapter seams and approved UI/event helper holdouts remain.

- [ ] **Step 3: Run hygiene checks**

Run:

```bash
git diff --check
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m compileall tldw_chatbook
```

Expected:

- `git diff --check`: no output.
- `compileall`: exit code `0`. It may be noisy if repo-local `.venv` directories are inside the package tree.

- [ ] **Step 4: Commit final verification notes only if files changed**

If Task 9 requires a small audit/test fix, commit it:

```bash
git add <changed-files>
git commit -m "chore: complete provider migration verification"
```

If no files changed, do not create an empty commit.
