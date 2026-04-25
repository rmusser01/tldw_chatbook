import pytest

from tldw_chatbook.Translation_Interop.server_translation_service import ServerTranslationService
from tldw_chatbook.runtime_policy import PolicyDeniedError


class FakeTranslationClient:
    def __init__(self):
        self.calls = []

    async def translate_text(self, request_data):
        self.calls.append(("translate_text", request_data))
        return {
            "translated_text": "Bonjour",
            "detected_source_language": request_data.source_language,
            "target_language": request_data.target_language,
            "model_used": request_data.model or "server-default",
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
async def test_server_translation_service_launches_text_translation_with_policy_and_normalization():
    client = FakeTranslationClient()
    policy = FakePolicyEnforcer()
    service = ServerTranslationService(client, policy_enforcer=policy)

    result = await service.translate_text(
        {
            "text": "Hello",
            "target_language": "French",
            "source_language": "English",
            "model": "fast-model",
        }
    )

    assert result == {
        "translated_text": "Bonjour",
        "detected_source_language": "English",
        "target_language": "French",
        "model_used": "fast-model",
        "backend": "server",
        "record_id": "server:translation:text",
    }
    assert client.calls[0][0] == "translate_text"
    assert client.calls[0][1].text == "Hello"
    assert client.calls[0][1].target_language == "French"
    assert policy.calls == ["translation.text.launch.server"]


@pytest.mark.asyncio
async def test_server_translation_service_denies_before_dispatch():
    client = FakeTranslationClient()
    service = ServerTranslationService(client, policy_enforcer=FakePolicyEnforcer("authority_denied"))

    with pytest.raises(PolicyDeniedError):
        await service.translate_text({"text": "Hello", "target_language": "French"})

    assert client.calls == []
