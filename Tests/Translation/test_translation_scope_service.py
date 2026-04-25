import pytest

from tldw_chatbook.Translation_Interop.translation_scope_service import TranslationScopeService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeServerTranslationService:
    def __init__(self):
        self.calls = []

    async def translate_text(self, request_data):
        self.calls.append(("translate_text", request_data))
        return {
            "translated_text": "Bonjour",
            "target_language": request_data["target_language"],
            "model_used": "server-default",
            "backend": "server",
            "record_id": "server:translation:text",
        }


class FakePolicyEnforcer:
    def __init__(self, denied_reason=None):
        self.denied_reason = denied_reason
        self.calls = []

    def require_allowed(self, *, action_id):
        self.calls.append(action_id)
        if self.denied_reason:
            raise PolicyDeniedError(
                action_id=action_id,
                reason_code=self.denied_reason,
                user_message=f"{action_id} denied",
                effective_source="server",
                authority_owner="server",
            )


@pytest.mark.asyncio
async def test_translation_scope_service_routes_server_translation():
    server = FakeServerTranslationService()
    policy = FakePolicyEnforcer()
    scope = TranslationScopeService(server_service=server, policy_enforcer=policy)

    result = await scope.translate_text(
        {"text": "Hello", "target_language": "French"},
        mode="server",
    )

    assert result == {
        "translated_text": "Bonjour",
        "target_language": "French",
        "model_used": "server-default",
        "backend": "server",
        "record_id": "server:translation:text",
    }
    assert server.calls == [("translate_text", {"text": "Hello", "target_language": "French"})]
    assert policy.calls == ["translation.text.launch.server"]


@pytest.mark.asyncio
async def test_translation_scope_service_rejects_local_mode_without_dispatch():
    server = FakeServerTranslationService()
    scope = TranslationScopeService(server_service=server)

    with pytest.raises(ValueError, match="server-only"):
        await scope.translate_text({"text": "Hello", "target_language": "French"}, mode="local")

    assert server.calls == []


@pytest.mark.asyncio
async def test_translation_scope_service_blocks_denied_action_before_dispatch():
    server = FakeServerTranslationService()
    scope = TranslationScopeService(server_service=server, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await scope.translate_text({"text": "Hello", "target_language": "French"}, mode="server")

    assert server.calls == []


def test_translation_scope_service_reports_local_remote_only_gap_and_no_server_gap():
    scope = TranslationScopeService(server_service=None)

    local_report = scope.list_unsupported_capabilities(mode="local")
    server_report = scope.list_unsupported_capabilities(mode="server")

    assert local_report == [
        {
            "operation_id": "translation.remote_only.local",
            "source": "local",
            "supported": False,
            "reason_code": "remote_only_surface",
            "user_message": "Server text translation is unavailable in local/offline mode.",
            "affected_action_ids": ["translation.text.launch.server"],
        }
    ]
    assert server_report == []
