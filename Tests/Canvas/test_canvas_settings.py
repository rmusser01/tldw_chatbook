"""Authoritative Canvas configuration policy tests."""

from __future__ import annotations

import tomllib
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Canvas.limits import CanvasLimits
from tldw_chatbook import config as config_module


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
    policy = config_module.build_canvas_config_policy({"canvas": "enabled"})

    assert policy.enabled is False
    assert policy.auto_open_on_create is False
    assert policy.diagnostics == ("canvas must be a table; Canvas is disabled",)


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
            },
            {},
            "authenticated_tls",
        ),
        (
            {"host": "0.0.0.0", "public_url": "https://chatbook.example"},
            {},
            "refused",
        ),
        (
            {
                "host": "0.0.0.0",
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
