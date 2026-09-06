"""Authoritative Canvas configuration policy tests."""

from __future__ import annotations

import tomllib
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook import config as config_module
from tldw_chatbook.Canvas.limits import CanvasLimits

CANVAS_ENVIRONMENT_KEYS = (
    "TLDW_CANVAS_ENABLED",
    "TLDW_CANVAS_AUTO_OPEN_ON_CREATE",
)


@pytest.fixture(autouse=True)
def _isolate_canvas_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep stored/default policy tests independent of the host environment."""

    for key in CANVAS_ENVIRONMENT_KEYS:
        monkeypatch.delenv(key, raising=False)


def test_canvas_defaults_are_enabled_and_shared_by_both_config_templates() -> None:
    policy = config_module.build_canvas_config_policy({})

    assert policy.enabled is True
    assert policy.auto_open_on_create is True
    assert policy.limits == CanvasLimits()
    assert config_module.DEFAULT_CONFIG_FROM_TOML["canvas"] == {
        "enabled": True,
        "auto_open_on_create": True,
    }
    with open("config.toml", "rb") as stream:
        example = tomllib.load(stream)
    assert example["canvas"] == config_module.DEFAULT_CONFIG_FROM_TOML["canvas"]


@pytest.mark.parametrize("invalid", ["true", 1, 0, [], {}])
def test_malformed_canvas_booleans_fail_closed(invalid: object) -> None:
    policy = config_module.build_canvas_config_policy(
        {"canvas": {"enabled": invalid, "auto_open_on_create": invalid}}
    )

    assert policy.enabled is False
    assert policy.auto_open_on_create is False
    assert set(policy.diagnostics) == {
        "canvas.enabled must be a boolean; Canvas is disabled",
        "canvas.auto_open_on_create must be a boolean; auto-open is disabled",
    }


def test_malformed_canvas_table_fails_all_execution_preferences_closed() -> None:
    policy = config_module.build_canvas_config_policy(
        {"canvas": "enabled"},
        environ={
            "TLDW_CANVAS_ENABLED": "true",
            "TLDW_CANVAS_AUTO_OPEN_ON_CREATE": "true",
        },
    )

    assert policy.enabled is False
    assert policy.auto_open_on_create is False
    assert policy.diagnostics == ("canvas must be a table; Canvas is disabled",)


@pytest.mark.parametrize(
    (
        "environment",
        "stored",
        "expected_enabled",
        "expected_auto_open",
        "expected_diagnostics",
    ),
    [
        (
            {
                "TLDW_CANVAS_ENABLED": " TrUe ",
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE": " FALSE ",
            },
            {"enabled": False, "auto_open_on_create": True},
            True,
            False,
            (),
        ),
        (
            {
                "TLDW_CANVAS_ENABLED": "false",
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE": "true",
            },
            {"enabled": True, "auto_open_on_create": False},
            False,
            True,
            (),
        ),
        (
            {
                "TLDW_CANVAS_ENABLED": " ",
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE": "",
            },
            {"enabled": False, "auto_open_on_create": True},
            False,
            True,
            (),
        ),
        (
            {
                "TLDW_CANVAS_ENABLED": "enabled-value-canary",
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE": "1",
            },
            {"enabled": True, "auto_open_on_create": True},
            False,
            False,
            (
                "TLDW_CANVAS_ENABLED must be true or false; Canvas is disabled",
                "TLDW_CANVAS_AUTO_OPEN_ON_CREATE must be true or false; auto-open is disabled",
            ),
        ),
    ],
)
def test_canvas_environment_preferences_strictly_override_stored_values(
    environment: dict[str, str],
    stored: dict[str, bool],
    expected_enabled: bool,
    expected_auto_open: bool,
    expected_diagnostics: tuple[str, ...],
) -> None:
    policy = config_module.build_canvas_config_policy(
        {"canvas": stored}, environ=environment
    )

    assert policy.enabled is expected_enabled
    assert policy.auto_open_on_create is expected_auto_open
    assert policy.diagnostics == expected_diagnostics
    assert "enabled-value-canary" not in repr(policy)


def test_canvas_environment_preferences_are_dynamic_and_match_the_cheap_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stored = {"canvas": {"enabled": False, "auto_open_on_create": True}}
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda: stored,
    )

    monkeypatch.setenv("TLDW_CANVAS_ENABLED", " true ")
    monkeypatch.setenv("TLDW_CANVAS_AUTO_OPEN_ON_CREATE", "false")
    policy = config_module.build_canvas_config_policy(stored)
    assert policy.enabled is config_module.get_canvas_execution_enabled() is True
    assert policy.auto_open_on_create is False

    monkeypatch.setenv("TLDW_CANVAS_ENABLED", "false")
    monkeypatch.setenv("TLDW_CANVAS_AUTO_OPEN_ON_CREATE", "TRUE")
    policy = config_module.build_canvas_config_policy(stored)
    assert policy.enabled is config_module.get_canvas_execution_enabled() is False
    assert policy.auto_open_on_create is True


def test_accepted_process_disable_latch_is_stronger_than_enabling_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda: {"canvas": {"enabled": False}},
    )
    monkeypatch.setenv("TLDW_CANVAS_ENABLED", "true")
    runtime = ConsoleRuntime(
        object(), canvas_enabled_reader=config_module.get_canvas_execution_enabled
    )

    assert runtime.canvas_enabled() is True
    runtime.latch_canvas_disabled()
    assert runtime.canvas_enabled() is False
    assert config_module.get_canvas_execution_enabled() is True


def test_canvas_environment_does_not_create_quota_overrides() -> None:
    policy = config_module.build_canvas_config_policy(
        {},
        environ={
            "TLDW_CANVAS_HTML_BYTES": "1",
            "TLDW_CANVAS_SCRIPT_BYTES": "1",
        },
    )

    assert policy.limits == CanvasLimits()
    assert policy.diagnostics == ()


def test_canvas_quota_overrides_cannot_raise_or_lower_hard_limits() -> None:
    requested = {
        "html_bytes": CanvasLimits().html_bytes + 1,
        "max_script_bytes": 1,
        "max_submit_payload_bytes": "invalid",
    }

    policy = config_module.build_canvas_config_policy({"canvas": requested})

    assert policy.limits == CanvasLimits()
    assert "Canvas quota overrides are unsupported; hard limits remain fixed" in (
        policy.diagnostics
    )
    with pytest.raises(FrozenInstanceError):
        policy.enabled = False  # type: ignore[misc]


@pytest.mark.parametrize(
    ("web", "environ", "expected"),
    [
        ({"host": "127.0.0.1"}, {}, "loopback"),
        (
            {
                "host": "0.0.0.0",
                "public_url": "https://chatbook.example",
                "access_token": "configured-secret",
                "trusted_proxy_addresses": ["127.0.0.1"],
            },
            {},
            "authenticated_tls",
        ),
        (
            {
                "host": "0.0.0.0",
                "public_url": "https://chatbook.example",
                "trusted_proxy_addresses": ["127.0.0.1"],
            },
            {},
            "refused",
        ),
        (
            {
                "host": "0.0.0.0",
                "public_url": "http://chatbook.example",
                "allow_insecure_remote_http": True,
                "access_token": "configured-secret",
            },
            {},
            "insecure_development",
        ),
        (
            {"host": "0.0.0.0", "public_url": "http://chatbook.example"},
            {"TLDW_CHATBOOK_WEB_ACCESS_TOKEN": "environment-secret"},
            "misconfigured",
        ),
    ],
)
def test_canvas_remote_status_is_derived_without_exposing_credentials(
    web: dict[str, object], environ: dict[str, str], expected: str
) -> None:
    policy = config_module.build_canvas_config_policy(
        {"web_server": web}, environ=environ
    )

    assert policy.remote_access_status == expected
    rendered = repr(policy) + policy.remote_access_summary
    assert "configured-secret" not in rendered
    assert "environment-secret" not in rendered


def test_canvas_remote_status_uses_keyring_and_proxy_exposure_on_loopback() -> None:
    policy = config_module.build_canvas_config_policy(
        {
            "web_server": {
                "host": "127.0.0.1",
                "public_url": "https://chatbook.example",
                "trusted_proxy_addresses": ["127.0.0.1"],
            }
        },
        environ={},
        keyring_get=lambda _service, _account: "keyring-secret",
    )

    assert policy.remote_access_status == "authenticated_tls"
    assert "keyring-secret" not in repr(policy)


def test_canvas_remote_status_rejects_an_invalid_https_origin() -> None:
    policy = config_module.build_canvas_config_policy(
        {
            "web_server": {
                "host": "0.0.0.0",
                "public_url": "https://chatbook.example/not-an-origin",
                "trusted_proxy_addresses": ["127.0.0.1"],
                "access_token": "configured-secret",
            }
        },
        environ={},
        keyring_get=lambda _service, _account: None,
    )

    assert policy.remote_access_status == "misconfigured"
    assert "configured-secret" not in policy.remote_access_summary


def test_canvas_remote_status_accepts_validated_effective_server_policy() -> None:
    from tldw_chatbook.Canvas.web_auth import build_web_auth_policy

    web_policy = build_web_auth_policy(
        host="0.0.0.0",
        port=8080,
        access_token="runtime-secret",
        public_url="http://chatbook.example",
        allow_insecure_remote_http=True,
    )

    policy = config_module.build_canvas_config_policy(
        {"web_server": {"host": "127.0.0.1"}},
        environ={},
        web_auth_policy=web_policy,
    )

    assert policy.remote_access_status == "insecure_development"
    assert "runtime-secret" not in repr(policy)


def test_execution_kill_switch_reads_do_not_resolve_remote_credentials(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda: {
            "canvas": {"enabled": True, "auto_open_on_create": False},
            "web_server": {
                "host": "0.0.0.0",
                "public_url": "https://chatbook.example",
            },
        },
    )
    monkeypatch.setattr(
        "tldw_chatbook.Canvas.web_auth.resolve_web_access_token",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("execution gate must not touch credentials")
        ),
    )

    assert config_module.get_canvas_execution_enabled() is True


@pytest.mark.parametrize(
    ("stored", "expected"),
    [
        (None, True),
        ({}, True),
        ({"canvas": None}, True),
        ({"canvas": {}}, True),
        ({"canvas": {"enabled": True}}, True),
        ({"canvas": {"enabled": False}}, False),
        ({"canvas": "enabled"}, False),
        ({"canvas": {"enabled": None}}, False),
        ({"canvas": {"enabled": 1}}, False),
    ],
)
def test_execution_kill_switch_uses_full_policy_strict_boolean_contract(
    monkeypatch: pytest.MonkeyPatch,
    stored: object,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        config_module,
        "load_cli_config_and_ensure_existence",
        lambda: stored,
    )

    assert config_module.get_canvas_execution_enabled() is expected


def test_canvas_settings_persist_and_reload_from_disk(monkeypatch, tmp_path) -> None:
    target = tmp_path / "config.toml"
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(target))
    config_module.load_cli_config_and_ensure_existence(force_reload=True)

    assert config_module.save_settings_to_cli_config(
        {"canvas": {"enabled": False, "auto_open_on_create": False}}
    )

    reloaded = config_module.load_cli_config_and_ensure_existence(force_reload=True)
    policy = config_module.build_canvas_config_policy(reloaded)
    assert policy.enabled is False
    assert policy.auto_open_on_create is False
