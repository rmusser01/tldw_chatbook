import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py"


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("provider_inventory", INVENTORY_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_importing_inventory_module_does_not_import_chatbook_runtime_modules() -> None:
    script = (
        "import importlib.util, json, sys\n"
        f"spec = importlib.util.spec_from_file_location('provider_inventory_lazy', {str(INVENTORY_PATH)!r})\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "runtime_modules = sorted(\n"
        "    name for name in sys.modules\n"
        "    if name in {'tldw_chatbook.Chat.Chat_Functions', 'tldw_chatbook.config'}\n"
        ")\n"
        "print(json.dumps(runtime_modules))\n"
    )

    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(result.stdout) == []


def test_load_env_values_expands_simple_references_without_shelling_out(tmp_path: Path) -> None:
    inventory = load_inventory_module()
    env_file = tmp_path / ".env"
    sentinel_path = tmp_path / "provider_inventory_shell_executed"
    sentinel_command = f"$(touch {shlex.quote(str(sentinel_path))})"
    env_file.write_text(
        "OPENAI_API_KEY=sk-live\n"
        "CUSTOM_OPENAI_API_KEY=${OPENAI_API_KEY}\n"
        "PLACEHOLDER=<api_key>\n"
        f"SHELL_SENTINEL={sentinel_command}\n",
        encoding="utf-8",
    )

    values = inventory.load_env_values(env_file)

    assert values["OPENAI_API_KEY"] == "sk-live"
    assert values["CUSTOM_OPENAI_API_KEY"] == "sk-live"
    assert values["PLACEHOLDER"] == "<api_key>"
    assert values["SHELL_SENTINEL"] == sentinel_command
    assert not sentinel_path.exists()


def test_should_use_key_value_rejects_empty_and_placeholder_values() -> None:
    inventory = load_inventory_module()

    assert inventory.should_use_key_value("sk-real") is True
    assert inventory.should_use_key_value("") is False
    assert inventory.should_use_key_value("<api_key>") is False
    assert inventory.should_use_key_value("${OPENAI_API_KEY}") is False


def test_mask_secret_never_returns_raw_secret() -> None:
    inventory = load_inventory_module()

    assert inventory.mask_secret("sk-abcdef123456") == "sk-a...3456"
    assert "abcdef" not in inventory.mask_secret("sk-abcdef123456")


def test_classify_external_outcome_is_stable() -> None:
    inventory = load_inventory_module()

    assert inventory.classify_external_outcome("missing_key") == "skip"
    assert inventory.classify_external_outcome("endpoint_unreachable") == "skip"
    assert inventory.classify_external_outcome("explicit_model_missing") == "skip"
    assert inventory.classify_external_outcome("auth") == "fail_external"
    assert inventory.classify_external_outcome("quota_or_rate_limit") == "fail_external"
    assert inventory.classify_external_outcome("request_shape") == "fail_chatbook"


def test_choose_uat_model_records_low_cost_override_source() -> None:
    inventory = load_inventory_module()

    selected = inventory.choose_uat_model(
        provider_key="openai",
        configured_models=["gpt-4.1"],
        provider_config={},
    )

    assert selected.model == "gpt-4o-mini-2024-07-18"
    assert selected.source == "override:openai"


def test_extract_endpoint_config_records_source_key() -> None:
    inventory = load_inventory_module()

    endpoint = inventory.extract_endpoint_config({"api_base": "http://127.0.0.1:8000"})

    assert endpoint.value == "http://127.0.0.1:8000"
    assert endpoint.source == "api_base"


def test_provider_settings_for_key_resolves_legacy_local_config_aliases() -> None:
    inventory = load_inventory_module()
    app_config = {
        "api_settings": {
            "ooba_api": {"api_ip": "http://127.0.0.1:5000/v1/chat/completions"},
            "tabby_api": {"api_url": "http://127.0.0.1:8080/v1/chat/completions"},
            "aphrodite_api": {"api_url": "http://127.0.0.1:2242/v1/chat/completions"},
        }
    }

    assert inventory.provider_settings_for_key(app_config, "oobabooga")["api_ip"] == (
        "http://127.0.0.1:5000/v1/chat/completions"
    )
    assert inventory.provider_settings_for_key(app_config, "tabbyapi")["api_url"] == (
        "http://127.0.0.1:8080/v1/chat/completions"
    )
    assert inventory.provider_settings_for_key(app_config, "aphrodite")["api_url"] == (
        "http://127.0.0.1:2242/v1/chat/completions"
    )


def test_choose_uat_model_can_use_server_default_for_reachable_local_provider() -> None:
    inventory = load_inventory_module()

    selected = inventory.choose_uat_model(
        provider_key="custom-openai-api",
        configured_models=[],
        provider_config={},
        allow_server_default=True,
    )

    assert selected.model == ""
    assert selected.source == "server_default"
    assert selected.requires_explicit_selection is False


def test_inventory_keeps_server_default_model_source_when_endpoint_is_not_reachable() -> None:
    inventory = load_inventory_module()

    rows = inventory.build_provider_inventory(
        app_config={
            "api_settings": {
                "llama_cpp": {
                    "api_url": "http://127.0.0.1:65530/v1/chat/completions",
                    "model": "",
                }
            }
        },
        configured_models_by_provider={"llama_cpp": []},
        environ={},
        probe_endpoints=False,
    )

    row = next(row for row in rows if row["handler_key"] == "llama_cpp")

    assert row["model"] == ""
    assert row["model_source"] == "server_default"
    assert row["requires_explicit_selection"] is False
    assert row["initial_reason"] == "endpoint_unreachable"
    assert row["classification"] == "skip"


def test_markdown_inventory_includes_classification_and_endpoint_probe_fields(tmp_path: Path) -> None:
    inventory = load_inventory_module()
    markdown_path = tmp_path / "inventory.md"

    inventory.write_markdown_inventory(
        markdown_path,
        [
            {
                "provider_name": "Local",
                "display_key": "local",
                "readiness_key": "local",
                "execution_key": "local",
                "model": "",
                "model_source": "server_default",
                "requires_api_key": False,
                "key_source": "not_required",
                "masked_key": "",
                "endpoint": "http://127.0.0.1:8000/v1/chat/completions",
                "endpoint_source": "api_url",
                "endpoint_reachable": False,
                "endpoint_probe_url": "http://127.0.0.1:8000/v1/models",
                "endpoint_probe_status": "unreachable:URLError",
                "initial_status": "skip",
                "initial_reason": "endpoint_unreachable",
                "classification": "skip",
            }
        ],
    )

    markdown = markdown_path.read_text(encoding="utf-8")

    assert "Classification" in markdown
    assert "Endpoint" in markdown
    assert "Probe URL" in markdown
    assert "Probe Status" in markdown
    assert "http://127.0.0.1:8000/v1/chat/completions" in markdown
    assert "http://127.0.0.1:8000/v1/models" in markdown
    assert "unreachable:URLError" in markdown


def test_inventory_redacts_endpoint_credentials_and_query_tokens(tmp_path: Path) -> None:
    inventory = load_inventory_module()
    sensitive_endpoint = "http://user:pass@127.0.0.1:8000/v1?api_key=secret-token&safe=ok"

    rows = inventory.build_provider_inventory(
        app_config={"api_settings": {"custom": {"api_url": sensitive_endpoint, "model": "custom-model"}}},
        configured_models_by_provider={"custom": []},
        environ={},
        probe_endpoints=False,
    )

    row = next(row for row in rows if row["handler_key"] == "custom-openai-api")

    assert "user" not in row["endpoint"]
    assert "pass" not in row["endpoint"]
    assert "secret-token" not in row["endpoint"]
    assert "user" not in row["endpoint_probe_url"]
    assert "pass" not in row["endpoint_probe_url"]
    assert "secret-token" not in row["endpoint_probe_url"]

    json_path = tmp_path / "inventory.json"
    markdown_path = tmp_path / "inventory.md"
    inventory.write_json_inventory(json_path, [row])
    inventory.write_markdown_inventory(markdown_path, [row])

    persisted = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")

    assert "user" not in persisted
    assert "pass" not in persisted
    assert "secret-token" not in persisted
