"""PIN: the strict-hosted key list derives from the provider registry."""


def test_strict_hosted_keys_derive_from_registry():
    from tldw_chatbook.LLM_Provider_Catalog import (
        local_llm_provider_catalog_service as service,
    )
    from tldw_chatbook.provider_registry import RECORDS_BY_KEY

    expected = {"moonshot", "zai"} | {
        key for key, record in RECORDS_BY_KEY.items() if record.engine_driven
    }
    assert service._STRICT_HOSTED_PROVIDER_KEYS == expected
    assert "databricks" in service._STRICT_HOSTED_PROVIDER_KEYS
    assert "openai" not in service._STRICT_HOSTED_PROVIDER_KEYS
