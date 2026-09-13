"""Validated, content-safe configuration for custom trace PII rules."""

from __future__ import annotations

import pytest

from tldw_chatbook.Chat.console_trace_custom_pii import (
    CUSTOM_PII_RULESET_VERSION,
    custom_pii_ruleset_for_revision,
    register_custom_pii_ruleset,
    validate_custom_pii_rules_config,
)
from tldw_chatbook.Chat.console_trace_redaction import (
    BUILTIN_PII_RULESET_REVISION_ID,
)

REVISION_ID = "11111111-1111-4111-8111-111111111111"
SECRET_PATTERN = r"account-(?P<digits>\d{8})"


def _rule(
    rule_id: str,
    *,
    pattern: str = SECRET_PATTERN,
    priority: int = 100,
    enabled: bool = True,
    flags: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": rule_id,
        "label": f"Rule {rule_id}",
        "category": "customer_id",
        "pattern": pattern,
        "flags": [] if flags is None else flags,
        "enabled": enabled,
        "priority": priority,
    }


def _ruleset(rules: list[object]) -> dict[str, object]:
    return {
        "version": CUSTOM_PII_RULESET_VERSION,
        "revision_id": REVISION_ID,
        "rules": rules,
    }


def test_valid_rules_are_versioned_hidden_and_ordered_deterministically() -> None:
    result = validate_custom_pii_rules_config(
        _ruleset(
            [
                _rule("later", priority=20, flags=["ignorecase"]),
                _rule("disabled", enabled=False, priority=0),
                _rule("alpha", priority=10),
                _rule("beta", priority=10),
            ]
        )
    )

    assert result.diagnostics == ()
    assert result.ruleset is not None
    assert result.ruleset.version == CUSTOM_PII_RULESET_VERSION
    assert result.ruleset.revision_id == REVISION_ID
    assert [rule.rule_id for rule in result.ruleset.runnable_rules] == [
        "alpha",
        "beta",
        "later",
    ]
    assert result.ruleset.runnable_rules[-1].flags == ("ignorecase",)
    assert SECRET_PATTERN not in repr(result)
    assert SECRET_PATTERN not in repr(result.ruleset.rules)


def test_invalid_rules_are_non_runnable_with_content_free_diagnostics() -> None:
    malformed = "private-prefix-("
    inline_flags = "(?i)private-prefix-[a-z]+"
    result = validate_custom_pii_rules_config(
        _ruleset(
            [
                _rule("valid"),
                _rule("bad-flag", flags=["verbose"]),
                _rule("bad-inline", pattern=inline_flags),
                _rule("bad-pattern", pattern=malformed),
                _rule("bad-shape") | {"enabled": "true"},
            ]
        )
    )

    assert result.ruleset is not None
    assert [rule.rule_id for rule in result.ruleset.runnable_rules] == ["valid"]
    assert [(item.rule_id, item.code) for item in result.diagnostics] == [
        ("bad-flag", "unsupported_flag"),
        ("bad-inline", "unsupported_construct"),
        ("bad-pattern", "invalid_pattern"),
        ("bad-shape", "invalid_rule"),
    ]
    diagnostics = repr(result.diagnostics)
    assert malformed not in diagnostics
    assert inline_flags not in diagnostics
    assert SECRET_PATTERN not in diagnostics


@pytest.mark.parametrize(
    "pattern",
    (r"\b", r"(?<=a)", r"a*", r"(?:private)?"),
)
def test_zero_width_rules_are_rejected_before_worker_execution(pattern: str) -> None:
    result = validate_custom_pii_rules_config(
        _ruleset([_rule("zero-width", pattern=pattern)])
    )

    assert result.ruleset is not None
    assert result.ruleset.runnable_rules == ()
    assert [(item.rule_id, item.code) for item in result.diagnostics] == [
        ("zero-width", "empty_match")
    ]


def test_duplicate_rule_ids_and_over_limit_rulesets_fail_closed() -> None:
    duplicate = validate_custom_pii_rules_config(
        _ruleset([_rule("same", pattern="first"), _rule("same", pattern="second")])
    )
    over_limit = validate_custom_pii_rules_config(
        _ruleset([_rule(f"rule-{index}") for index in range(65)])
    )

    assert duplicate.ruleset is not None
    assert duplicate.ruleset.runnable_rules == ()
    assert [(item.rule_id, item.code) for item in duplicate.diagnostics] == [
        ("same", "duplicate_rule_id")
    ]
    assert over_limit.ruleset is None
    assert [(item.rule_id, item.code) for item in over_limit.diagnostics] == [
        (None, "rule_count_limit")
    ]


def test_ruleset_envelope_requires_v1_and_canonical_opaque_revision() -> None:
    bad_version = validate_custom_pii_rules_config(
        {"version": 2, "revision_id": REVISION_ID, "rules": []}
    )
    bad_revision = validate_custom_pii_rules_config(
        {"version": 1, "revision_id": "ruleset-from-pattern-hash", "rules": []}
    )
    reserved_revision = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": BUILTIN_PII_RULESET_REVISION_ID,
            "rules": [_rule("custom")],
        }
    )
    absent = validate_custom_pii_rules_config(None)

    assert bad_version.ruleset is None
    assert bad_version.diagnostics[0].code == "unsupported_version"
    assert bad_revision.ruleset is None
    assert bad_revision.diagnostics[0].code == "invalid_revision_id"
    assert reserved_revision.ruleset is None
    assert reserved_revision.diagnostics[0].code == "reserved_revision_id"
    assert absent.ruleset is None
    assert absent.diagnostics == ()


@pytest.mark.parametrize(
    ("value", "expected_code"),
    (
        (_ruleset([]) | {"unexpected": True}, "invalid_ruleset"),
        ({"version": "1", "revision_id": REVISION_ID, "rules": []}, "unsupported_version"),
        ({"version": 1, "revision_id": REVISION_ID, "rules": {}}, "invalid_ruleset"),
    ),
)
def test_ruleset_envelope_is_strict_and_forbids_unknown_fields(
    value: object,
    expected_code: str,
) -> None:
    result = validate_custom_pii_rules_config(value)

    assert result.ruleset is None
    assert [item.code for item in result.diagnostics] == [expected_code]


def test_ephemeral_registry_rejects_revision_reuse_with_different_rules() -> None:
    revision_id = "22222222-2222-4222-8222-222222222222"
    first = validate_custom_pii_rules_config(
        {"version": 1, "revision_id": revision_id, "rules": [_rule("account")]}
    ).ruleset
    conflicting = validate_custom_pii_rules_config(
        {
            "version": 1,
            "revision_id": revision_id,
            "rules": [_rule("account", pattern=r"different-[A-Z]+")],
        }
    ).ruleset
    assert first is not None
    assert conflicting is not None

    assert register_custom_pii_ruleset(first) is True
    assert register_custom_pii_ruleset(conflicting) is False
    assert custom_pii_ruleset_for_revision(revision_id) == first
    assert SECRET_PATTERN not in repr(custom_pii_ruleset_for_revision(revision_id))
