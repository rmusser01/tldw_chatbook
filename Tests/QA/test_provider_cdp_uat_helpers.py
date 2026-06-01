import importlib.util
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INVENTORY_PATH = ROOT / "Docs/superpowers/qa/provider-cdp-uat/provider_inventory.py"
LAUNCH_PATH = ROOT / "Docs/superpowers/qa/provider-cdp-uat/run_textual_web_with_env.py"


def load_inventory_module():
    spec = importlib.util.spec_from_file_location("provider_inventory", INVENTORY_PATH)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_launch_module():
    spec = importlib.util.spec_from_file_location("run_textual_web_with_env", LAUNCH_PATH)
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


def test_build_launch_environment_sets_isolated_home_and_redacted_env(tmp_path: Path) -> None:
    launch = load_launch_module()
    env_values = {"OPENAI_API_KEY": "sk-real", "PLACEHOLDER": "<api_key>"}
    qa_root = tmp_path / "qa-root"

    launch_env = launch.build_launch_environment(
        worktree=tmp_path,
        qa_root=qa_root,
        env_values=env_values,
        port=8765,
    )

    assert launch_env["HOME"] == str(qa_root / "home")
    assert launch_env["XDG_CONFIG_HOME"] == str(qa_root / "config")
    assert launch_env["XDG_DATA_HOME"] == str(qa_root / "data")
    assert launch_env["TLDW_CONFIG_PATH"] == str(qa_root / "home" / ".config" / "tldw_cli" / "config.toml")
    assert launch_env["PYTHONPATH"] == str(tmp_path)
    assert launch_env["TLDW_TEXTUAL_WEB_PORT"] == "8765"
    assert launch_env["OPENAI_API_KEY"] == "sk-real"
    assert "PLACEHOLDER" not in launch_env


def test_build_launch_environment_scrubs_inherited_provider_keys_and_config(tmp_path: Path) -> None:
    launch = load_launch_module()
    qa_root = tmp_path / "qa-root"

    launch_env = launch.build_launch_environment(
        worktree=tmp_path,
        qa_root=qa_root,
        env_values={
            "OPENAI_API_KEY": "sk-from-env-file",
            "ANTHROPIC_API_KEY": "<api_key>",
        },
        base_environ={
            "PATH": "/usr/bin",
            "OPENAI_API_KEY": "sk-inherited-openai",
            "ANTHROPIC_API_KEY": "sk-inherited-anthropic",
            "TLDW_CONFIG_PATH": "/Users/example/.config/tldw_cli/config.toml",
        },
    )

    assert launch_env["PATH"] == "/usr/bin"
    assert launch_env["OPENAI_API_KEY"] == "sk-from-env-file"
    assert "ANTHROPIC_API_KEY" not in launch_env
    assert "sk-inherited-openai" not in launch_env.values()
    assert "sk-inherited-anthropic" not in launch_env.values()
    assert launch_env["TLDW_CONFIG_PATH"] == str(qa_root / "home" / ".config" / "tldw_cli" / "config.toml")


def test_write_isolated_configs_writes_home_and_xdg_configs(tmp_path: Path) -> None:
    launch = load_launch_module()
    qa_root = tmp_path / "qa-root"

    home_config, xdg_config = launch.write_isolated_configs(qa_root)

    assert home_config == qa_root / "home" / ".config" / "tldw_cli" / "config.toml"
    assert xdg_config == qa_root / "config" / "tldw_cli" / "config.toml"
    assert home_config.read_text(encoding="utf-8") == launch.ISOLATED_CONFIG
    assert xdg_config.read_text(encoding="utf-8") == launch.ISOLATED_CONFIG


def test_print_launch_summary_masks_key_values_and_shell_quotes_command(tmp_path: Path, capsys) -> None:
    launch = load_launch_module()
    raw_key = "sk-secret-value-123456"
    command = [str(tmp_path / "bin with space" / "tldw-serve"), "--port", "8765"]

    launch.print_launch_summary(
        worktree=tmp_path,
        qa_root=tmp_path / "qa-root",
        env_file=tmp_path / ".env",
        config_files=[tmp_path / "qa-root" / "home" / ".config" / "tldw_cli" / "config.toml"],
        command=command,
        port=8765,
        env_values={"OPENAI_API_KEY": raw_key, "PLACEHOLDER": "<api_key>"},
    )

    captured = capsys.readouterr()

    assert raw_key not in captured.out
    assert "OPENAI_API_KEY=sk-s...3456" in captured.out
    assert "PLACEHOLDER" not in captured.out
    assert shlex.join(command) in captured.out


def test_normalize_qa_root_resolves_relative_path(tmp_path: Path, monkeypatch) -> None:
    launch = load_launch_module()
    monkeypatch.chdir(tmp_path)

    qa_root = launch.normalize_qa_root(Path("qa-root"))

    assert qa_root == tmp_path / "qa-root"
    assert qa_root.is_absolute()


def test_validate_launch_inputs_rejects_invalid_worktree_port_and_missing_server(tmp_path: Path) -> None:
    launch = load_launch_module()

    missing_errors = launch.validate_launch_inputs(tmp_path / "missing", 0)
    missing_text = "\n".join(missing_errors)
    assert "worktree does not exist" in missing_text
    assert "port must be between 1 and 65535" in missing_text

    missing_server_errors = launch.validate_launch_inputs(tmp_path, 8765)
    assert any(".venv/bin/tldw-serve" in error for error in missing_server_errors)


def test_main_returns_nonzero_for_invalid_inputs_without_writing_config(tmp_path: Path, capsys) -> None:
    launch = load_launch_module()
    qa_root = tmp_path / "qa-root"

    result = launch.main(
        [
            "--worktree",
            str(tmp_path / "missing-worktree"),
            "--qa-root",
            str(qa_root),
            "--port",
            "70000",
        ]
    )

    captured = capsys.readouterr()

    assert result == 2
    assert "worktree does not exist" in captured.err
    assert "port must be between 1 and 65535" in captured.err
    assert not qa_root.exists()


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
    assert row["classification"] == "endpoint_unreachable"


def test_inventory_surfaces_explicit_model_missing_before_endpoint_reachability() -> None:
    inventory = load_inventory_module()

    rows = inventory.build_provider_inventory(
        app_config={
            "api_settings": {
                "ooba_api": {
                    "api_url": "http://127.0.0.1:65530/v1/chat/completions",
                    "model": "",
                }
            }
        },
        configured_models_by_provider={"oobabooga": []},
        environ={},
        probe_endpoints=False,
    )

    row = next(row for row in rows if row["handler_key"] == "oobabooga")

    assert row["model"] == ""
    assert row["model_source"] == "explicit_model_missing"
    assert row["initial_status"] == "skip"
    assert row["initial_reason"] == "explicit_model_missing"
    assert row["classification"] == "explicit_model_missing"


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
                "classification": "endpoint_unreachable",
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


def test_persisted_inventory_redacts_masked_key_prefixes(tmp_path: Path) -> None:
    inventory = load_inventory_module()
    json_path = tmp_path / "inventory.json"
    markdown_path = tmp_path / "inventory.md"
    dangerous_prefixes = ("sk-proj-", "sk-ant-", "AIza", "gsk_", "hf_", "sk-or-")
    rows = [
        {
            "provider_name": "Google",
            "display_key": "google",
            "readiness_key": "google",
            "execution_key": "google",
            "model": "gemini-2.0-flash-lite",
            "model_source": "override:google",
            "requires_api_key": True,
            "key_source": "env_file:GOOGLE_API_KEY",
            "has_usable_key": True,
            "masked_key": "AIza...FJ44",
            "endpoint": "",
            "endpoint_source": "",
            "endpoint_reachable": None,
            "endpoint_probe_url": "",
            "endpoint_probe_status": "not_applicable",
            "initial_status": "pending_cdp",
            "initial_reason": "ready_for_cdp",
            "classification": "ready_for_cdp",
        }
    ]

    inventory.write_json_inventory(json_path, rows)
    inventory.write_markdown_inventory(markdown_path, rows)

    persisted = json_path.read_text(encoding="utf-8") + markdown_path.read_text(encoding="utf-8")
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_payload["providers"][0]["masked_key"] == "***REDACTED***"
    assert "env_file:GOOGLE_API_KEY ***REDACTED***" in persisted
    for forbidden in dangerous_prefixes:
        assert forbidden not in persisted


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
