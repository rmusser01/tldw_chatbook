# Tests/test_provider_registry.py
"""Registry parity tests: registry data must equal today's literal tables."""
from tldw_chatbook.provider_registry import (
    ALL_RECORDS,
    AUDITED_ENDPOINT_KEYS,
    AUTO_REFRESH_KEYS,
    CLOUD_PROVIDER_CONFIG_KEYS,
    DATABRICKS,
    ENGINE_RECORDS,
    NATIVE_TOOLS_KEYS,
    RECORDS_BY_KEY,
)
from tldw_chatbook.Chat.Chat_Functions import (
    API_CALL_HANDLERS,
    SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS,
)


def test_records_unique_and_complete():
    keys = [record.key for record in ALL_RECORDS]
    assert len(keys) == len(set(keys))
    # every dispatch key is a registry key or an alias of one
    assert set(API_CALL_HANDLERS) - set(RECORDS_BY_KEY) <= set()


def test_audited_set_matches_handlers():
    # Since Task 8 the audited set is derived from the registry (ADR-179),
    # so it tracks AUDITED_ENDPOINT_KEYS exactly.
    assert SENSITIVE_AUXILIARY_AUDITED_ENDPOINTS == AUDITED_ENDPOINT_KEYS


def test_cloud_classification_matches_config():
    # Since Task 9, config.py's _cloud_provider_keys is derived from the
    # registry (ADR-179), so full equality holds with no exclusions.
    from tldw_chatbook.config import _cloud_provider_keys
    assert tuple(sorted(_cloud_provider_keys)) == tuple(
        sorted(CLOUD_PROVIDER_CONFIG_KEYS)
    )


def test_auto_refresh_flags_match_catalog_settings():
    # Same style as the cloud-classification parity: the literal list is in
    # [providers]-key form, so compare it against the config keys of the
    # records flagged auto_refresh. Task 12 added "Databricks" to the literal
    # list, so full equality holds with no exclusions.
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        AUTO_REFRESH_PROVIDER_LIST_KEYS,
    )
    auto_refresh_config_keys = {
        RECORDS_BY_KEY[key].config_key for key in AUTO_REFRESH_KEYS
    }
    assert set(AUTO_REFRESH_PROVIDER_LIST_KEYS) == auto_refresh_config_keys


def test_native_tools_flags_match_native_tools_module():
    # NATIVE_TOOLS_PROVIDERS is keyed by dispatch (record) keys. Task 12
    # added "databricks" to the literal set, so full equality holds with no
    # exclusions.
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS
    assert NATIVE_TOOLS_PROVIDERS == NATIVE_TOOLS_KEYS


def test_identity_fields_match_config_tables():
    # api_key_env_var / default api_base_url must be transcribed EXACTLY
    # from the shipped [api_settings.*] tables (the e080a2fb92-class bug
    # guard: a wrong env var or URL must fail here, not at a user's first
    # call). A record's table is keyed by its dispatch key or its config
    # key (lowercased), e.g. mistral -> [api_settings.mistralai],
    # custom-openai-api -> [api_settings.custom]. databricks is excluded:
    # it ships no defaults (workspace host is per-account).
    # custom-hosted is excluded the same way (ADR-179 Phase 2 Task 6): it
    # is an execution key, not an identity -- no table of its own (its
    # fallbacks ride [api_settings.custom] via defaults_settings_section,
    # under the legacy handler's key spellings), no env var (credentials
    # are gateway/entry-resolved; the section's own api_key_env_var flows
    # through the settings read), and no default URL (the per-entry URL is
    # forwarded explicitly).
    import tomllib

    from tldw_chatbook.config import CONFIG_TOML_CONTENT

    tables = tomllib.loads(CONFIG_TOML_CONTENT)["api_settings"]
    checked = 0
    for record in ALL_RECORDS:
        if record.key == "databricks" or record.defaults_settings_section:
            continue
        table_key = next(
            (k for k in (record.key, record.config_key.lower()) if k in tables),
            None,
        )
        assert table_key is not None, f"no [api_settings.*] table for {record.key}"
        table = tables[table_key]
        assert record.api_key_env_var == table.get("api_key_env_var"), (
            f"{record.key}: api_key_env_var {record.api_key_env_var!r} != "
            f"[api_settings.{table_key}] {table.get('api_key_env_var')!r}"
        )
        if "api_base_url" in table:
            assert record.default_base_url == table["api_base_url"], (
                f"{record.key}: default_base_url {record.default_base_url!r} != "
                f"[api_settings.{table_key}] {table['api_base_url']!r}"
            )
        checked += 1
    assert checked > 0


def test_inference_cloud_keys_join_every_parity_surface():
    # Task 5 (ADR-179 Phase 2): together/fireworks/cerebras join the
    # audited set, cloud classification, auto-refresh, native tools,
    # engine records, and dispatch. Every one of those surfaces is now a
    # full-equality comparison with no exclusions (Tasks 8/9/11/12), so
    # this pins each key's membership explicitly.
    from tldw_chatbook.config import _cloud_provider_keys
    from tldw_chatbook.LLM_Provider_Catalog.model_catalog_settings import (
        AUTO_REFRESH_PROVIDER_LIST_KEYS,
    )
    from tldw_chatbook.Agents.native_tools import NATIVE_TOOLS_PROVIDERS

    for key, config_key in (
        ("together", "Together"),
        ("fireworks", "Fireworks"),
        ("cerebras", "Cerebras"),
    ):
        record = RECORDS_BY_KEY[key]
        assert record.engine_driven is True
        assert key in {r.key for r in ENGINE_RECORDS}
        assert config_key in CLOUD_PROVIDER_CONFIG_KEYS
        assert config_key in _cloud_provider_keys
        assert key in AUTO_REFRESH_KEYS
        assert config_key in AUTO_REFRESH_PROVIDER_LIST_KEYS
        assert key in NATIVE_TOOLS_KEYS
        assert key in NATIVE_TOOLS_PROVIDERS
        assert key in AUDITED_ENDPOINT_KEYS
        assert callable(API_CALL_HANDLERS.get(key))


def test_databricks_preset_shape():
    record = DATABRICKS
    assert record.key == "databricks"
    assert record.classification == "cloud"
    assert record.engine_driven is True
    assert record.auth_scheme == "bearer"
    assert record.api_key_env_candidates == ("DATABRICKS_TOKEN",)
    assert record.base_url_suffix == "/openai/v1"
    assert record.default_base_url is None  # workspace host is per-account
    assert record.reasoning_disposition == "ignored"
    assert "databricks" in {r.key for r in ENGINE_RECORDS}


# --- Lookup tables derived from the records (TASK-36946.1) ---


def test_continuation_provider_literal_matches_derived_keys():
    # A typing Literal cannot be computed, so it is pinned to the derived set
    # that the runtime checks (_PAIRINGS) are built from.
    from typing import get_args

    from tldw_chatbook.Provider_Contract.provider_continuation import (
        _PAIRINGS,
        ContinuationProvider,
    )
    from tldw_chatbook.provider_registry import CONTINUATION_PROVIDER_KEYS

    assert set(get_args(ContinuationProvider)) == CONTINUATION_PROVIDER_KEYS
    assert {provider for provider, _ in _PAIRINGS} == CONTINUATION_PROVIDER_KEYS
    assert _PAIRINGS == {
        (key, "chat_completions") for key in CONTINUATION_PROVIDER_KEYS
    } | {("deepseek", "responses")}
    assert {r.key for r in ENGINE_RECORDS if r.continuation_protocol} <= (
        CONTINUATION_PROVIDER_KEYS
    )

# Frozen from the hand lists at dc96c947e8: reclassifying a record between cloud
# and local must update these deliberately.
_BASE_KEYED = frozenset(
    {
        "anthropic",
        "arcee",
        "azure",
        "baseten",
        "byteplus",
        "cerebras",
        "cloudflare",
        "cohere",
        "commandcode",
        "databricks",
        "deepinfra",
        "deepseek",
        "fireworks",
        "gmi",
        "google",
        "groq",
        "huggingface",
        "kilo",
        "meta",
        "mimo",
        "minimax",
        "mistral",
        "mistralai",
        "moonshot",
        "nebius",
        "nous",
        "novita",
        "nvidia",
        "ollama_cloud",
        "openai",
        "opencode_zen",
        "openrouter",
        "qianfan",
        "qwencloud",
        "sambanova",
        "siliconflow",
        "stepfun",
        "together",
        "tokenhub",
        "upstage",
        "venice",
        "vercel",
        "wandb",
        "zai",
        "zenmux",
    }
)
_BASE_KEYLESS = frozenset(
    {
        "aphrodite",
        "custom",
        "custom_2",
        "custom_openai_api",
        "custom_openai_api_2",
        "koboldcpp",
        "llama_cpp",
        "local_llamacpp",
        "local_llamafile",
        "local_llm",
        "local_mlx_lm",
        "local_ollama",
        "local_onnx",
        "local_transformers",
        "local_vllm",
        "mlx_lm",
        "ollama",
        "oobabooga",
        "polystrata",
        "strata",
        "tabbyapi",
        "vllm",
    }
)


def test_readiness_key_sets_are_pinned_to_the_base_values():
    from tldw_chatbook import provider_registry as registry

    assert registry.PROVIDERS_REQUIRING_API_KEY_KEYS == _BASE_KEYED
    assert registry.KEYLESS_PROVIDER_KEYS == _BASE_KEYLESS


def test_readiness_key_sets_are_the_registry_derivations():
    from tldw_chatbook import provider_registry as registry
    from tldw_chatbook.Chat import provider_readiness as readiness

    norm = registry.normalize_provider_config_key
    assert readiness.KEYLESS_PROVIDER_KEYS is registry.KEYLESS_PROVIDER_KEYS
    assert readiness.PROVIDERS_REQUIRING_API_KEY_KEYS is registry.PROVIDERS_REQUIRING_API_KEY_KEYS
    assert readiness.PROVIDERS_REQUIRING_BASE_URL_KEYS is registry.PROVIDERS_REQUIRING_BASE_URL_KEYS
    keyed = registry.PROVIDERS_REQUIRING_API_KEY_KEYS
    keyless = registry.KEYLESS_PROVIDER_KEYS
    assert not keyed & keyless
    for record in ALL_RECORDS:
        if record.key == "custom-hosted":  # execution-only, never a readiness identity
            assert norm(record.key) not in keyed | keyless
        elif record.classification == "cloud":
            assert norm(record.key) in keyed
        else:
            assert norm(record.key) in keyless
    for alias, canonical in registry.ALIASES.items():
        bucket = keyed if RECORDS_BY_KEY[canonical].classification == "cloud" else keyless
        assert norm(alias) in bucket
    # Config tables without a record or handler: the one explicit exception.
    assert keyless - {norm(k) for k in RECORDS_BY_KEY} - {
        norm(r.config_key) for r in ALL_RECORDS
    } == {"local_onnx", "local_transformers"}


def test_base_url_required_matches_records_without_a_default_url():
    from tldw_chatbook.provider_registry import PROVIDERS_REQUIRING_BASE_URL_KEYS

    assert PROVIDERS_REQUIRING_BASE_URL_KEYS == {"azure", "cloudflare", "databricks"}
    for record in ALL_RECORDS:
        if record.base_url_required:
            assert record.engine_driven and record.default_base_url is None


def test_default_api_key_env_var_is_the_record_field():
    from tldw_chatbook.Chat.provider_readiness import default_api_key_env_var
    from tldw_chatbook.provider_registry import (
        KEYLESS_PROVIDER_KEYS,
        PROVIDERS_REQUIRING_API_KEY_KEYS,
    )

    for key in PROVIDERS_REQUIRING_API_KEY_KEYS:
        assert default_api_key_env_var(key) == RECORDS_BY_KEY[key].api_key_env_var
        assert default_api_key_env_var(key)
    assert default_api_key_env_var("mistralai") == "MISTRAL_API_KEY"
    for key in KEYLESS_PROVIDER_KEYS | {"not-a-provider"}:
        assert default_api_key_env_var(key) is None


def test_console_execution_keys_cover_every_record_but_custom_hosted():
    from tldw_chatbook.Chat.console_session_settings import (
        CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS,
    )

    assert CONSOLE_SETTINGS_EXECUTION_PROVIDER_KEYS == set(RECORDS_BY_KEY) - {
        "custom-hosted"
    }


def test_custom_slot_keys_come_from_the_custom_records():
    from tldw_chatbook.Chat.provider_catalog import (
        PROVIDER_CUSTOM_GROUP_KEYS,
        PROVIDER_CUSTOM_SLOT_KEYS,
        PROVIDER_DISPLAY_NAMES,
    )
    from tldw_chatbook.Chat.provider_readiness import KNOWN_PROVIDER_KEYS

    assert PROVIDER_CUSTOM_SLOT_KEYS == {"custom", "custom_2"}
    # Hand-kept UI labels/groups must name real providers (no typo'd keys).
    assert PROVIDER_CUSTOM_GROUP_KEYS <= KNOWN_PROVIDER_KEYS
    assert set(PROVIDER_DISPLAY_NAMES) - KNOWN_PROVIDER_KEYS == {"custom_hosted"}


# Frozen from the hand table at dc96c947e8: a record display_name drift must
# update this deliberately, not silently relabel the UI.
_EXPECTED_DISPLAY_NAMES = {
    'anthropic': 'Anthropic',
    'aphrodite': 'Aphrodite Engine',
    'arcee': 'Arcee AI',
    'azure': 'Azure OpenAI',
    'baseten': 'Baseten',
    'byteplus': 'ByteDance Seed (BytePlus)',
    'cerebras': 'Cerebras',
    'cloudflare': 'Cloudflare Workers AI',
    'cohere': 'Cohere',
    'commandcode': 'Command Code',
    'custom': 'Custom OpenAI-compatible',
    'custom_2': 'Custom OpenAI-compatible #2',
    'custom_hosted': 'Custom Hosted',
    'databricks': 'Databricks',
    'deepinfra': 'DeepInfra',
    'deepseek': 'DeepSeek',
    'fireworks': 'Fireworks',
    'gmi': 'GMI Cloud',
    'google': 'Google Gemini',
    'groq': 'Groq',
    'huggingface': 'Hugging Face',
    'kilo': 'Kilo Gateway',
    'koboldcpp': 'KoboldCpp',
    'llama_cpp': 'llama.cpp',
    'local_llamacpp': 'llama.cpp (legacy alias)',
    'local_llamafile': 'Llamafile',
    'local_llm': 'Local LLM (legacy generic)',
    'local_mlx_lm': 'MLX-LM (Apple silicon)',
    'local_ollama': 'Ollama (legacy alias)',
    'local_onnx': 'ONNX Runtime (local)',
    'local_transformers': 'Transformers (local)',
    'local_vllm': 'vLLM (legacy alias)',
    'meta': 'Meta (Muse Spark)',
    'mimo': 'Xiaomi MiMo',
    'minimax': 'MiniMax',
    'mistral': 'Mistral AI (legacy alias)',
    'mistralai': 'Mistral AI',
    'moonshot': 'Moonshot AI',
    'nebius': 'Nebius Token Factory',
    'nous': 'Nous Research',
    'novita': 'Novita AI',
    'nvidia': 'NVIDIA NIM',
    'ollama': 'Ollama',
    'ollama_cloud': 'Ollama Cloud',
    'oobabooga': 'Text Generation WebUI (Oobabooga)',
    'openai': 'OpenAI',
    'opencode_zen': 'OpenCode Zen',
    'openrouter': 'OpenRouter',
    'polystrata': 'PolyStrata',
    'qianfan': 'Baidu Qianfan',
    'qwencloud': 'QwenCloud',
    'sambanova': 'SambaNova',
    'siliconflow': 'SiliconFlow',
    'stepfun': 'StepFun',
    'strata': 'Strata',
    'tabbyapi': 'TabbyAPI',
    'together': 'Together',
    'tokenhub': 'Tencent TokenHub',
    'upstage': 'Upstage',
    'venice': 'Venice',
    'vercel': 'Vercel AI Gateway',
    'vllm': 'vLLM',
    'wandb': 'W&B Inference (CoreWeave)',
    'zai': 'Z.ai',
    'zenmux': 'ZenMux',
}


def test_display_names_follow_records_and_overrides_are_not_redundant():
    from tldw_chatbook.Chat.provider_catalog import (
        _DISPLAY_NAME_OVERRIDES,
        PROVIDER_DISPLAY_NAMES,
    )
    from tldw_chatbook.provider_registry import normalize_provider_config_key as norm

    assert PROVIDER_DISPLAY_NAMES == _EXPECTED_DISPLAY_NAMES
    by_key = {norm(r.key): r.display_name for r in ALL_RECORDS}
    for key, label in _DISPLAY_NAME_OVERRIDES.items():
        # An override equal to the record value is dead weight.
        assert by_key.get(key) != label, key
