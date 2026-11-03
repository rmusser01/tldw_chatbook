# Tests/Chat/test_dispatch_registry_parity.py
"""Dispatch registration parity for engine-driven providers (ADR-179).

Task 8: Databricks joins the dispatch surface through the hosted provider
engine, and the sensitive-audit universe is derived from the provider
registry instead of a hand-maintained literal set.
"""
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)
from tldw_chatbook.provider_registry import AUDITED_ENDPOINT_KEYS


def test_databricks_registered_and_audited():
    assert "databricks" in API_CALL_HANDLERS
    assert callable(API_CALL_HANDLERS["databricks"])
    assert "databricks" in SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS


def test_audited_set_is_registry_derived_and_covers_all_handlers():
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS
    assert set(API_CALL_HANDLERS) <= SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS


# TASK-36946.2: the engine-preset entries are generated from the registry.
# The literal below is the pre-generation hand-written list, so a record that
# is dropped (or an entry the generator skips) fails here.
_BASE_ENGINE_KEYS = ('arcee', 'azure', 'baseten', 'byteplus', 'cerebras', 'cloudflare', 'commandcode', 'custom-hosted', 'databricks', 'deepinfra', 'fireworks', 'gmi', 'kilo', 'meta', 'mimo', 'minimax', 'nebius', 'nous', 'novita', 'nvidia', 'ollama_cloud', 'opencode_zen', 'qianfan', 'sambanova', 'siliconflow', 'stepfun', 'together', 'tokenhub', 'upstage', 'venice', 'vercel', 'wandb', 'zenmux')


def test_engine_handlers_generated_from_registry_match_the_base_list():
    from tldw_chatbook.Chat.Chat_Functions import _LazyHostedChatHandler
    from tldw_chatbook.provider_registry import ENGINE_RECORDS, RECORDS_BY_KEY

    engine = {k for k, v in API_CALL_HANDLERS.items() if isinstance(v, _LazyHostedChatHandler)}
    assert engine == set(_BASE_ENGINE_KEYS)
    assert {r.key for r in ENGINE_RECORDS} == set(_BASE_ENGINE_KEYS)
    for key in _BASE_ENGINE_KEYS:
        assert API_CALL_HANDLERS[key]._record is RECORDS_BY_KEY[key]


def test_engine_param_maps_match_the_base_list():
    from tldw_chatbook.Chat.Chat_Functions import (
        CUSTOM_PROVIDER_PARAM_MAP,
        ENGINE_PROVIDER_PARAM_MAP,
        PROVIDER_PARAM_MAP,
    )

    for key in _BASE_ENGINE_KEYS:
        expected = CUSTOM_PROVIDER_PARAM_MAP if key == "custom-hosted" else ENGINE_PROVIDER_PARAM_MAP
        assert PROVIDER_PARAM_MAP[key] is expected
    # zai is a hand adapter that shares the engine kwargs (explicit one-off).
    assert PROVIDER_PARAM_MAP["zai"] is ENGINE_PROVIDER_PARAM_MAP
    assert API_CALL_HANDLERS["zai"].__name__ == "chat_with_zai"
