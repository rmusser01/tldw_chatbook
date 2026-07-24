from __future__ import annotations

import tomllib

import pytest

from tldw_chatbook import config


def test_rag_citation_canonical_writes_default_off() -> None:
    parsed = tomllib.loads(config.CONFIG_TOML_CONTENT)

    assert parsed["rag_citations"]["canonical_writes_enabled"] is False
    assert config.DEFAULT_CONFIG_FROM_TOML["rag_citations"] == {
        "canonical_writes_enabled": False
    }


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
        (1, True),
        (0, False),
        ("invalid", False),
    ],
)
def test_typed_rag_citation_switch_access(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        config,
        "load_cli_config_and_ensure_existence",
        lambda: {"rag_citations": {"canonical_writes_enabled": value}},
    )

    assert config.get_rag_citation_canonical_writes_enabled() is expected


def test_typed_rag_citation_switch_fails_closed_on_malformed_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        config,
        "load_cli_config_and_ensure_existence",
        lambda: {"rag_citations": "not-a-table"},
    )

    assert config.get_rag_citation_canonical_writes_enabled() is False
