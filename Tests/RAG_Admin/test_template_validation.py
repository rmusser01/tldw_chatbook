"""Local chunking-template validation - spec §7/§7.1 parity (ACs 14-15).

The fixture table (``template_validation_fixtures.json``) is the AC 14
provability artifact: its header records the upstream file + line ranges the
semantics were transcribed from. Expectations pin ``valid``, error counts by
``field``, and warning counts - not message text, because the endpoint's
pydantic-wrapped and hand-rolled passes phrase the same failure differently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from tldw_chatbook.RAG_Admin.rag_admin_scope_service import (
    RAGAdminBackend,
    RAGAdminScopeService,
)
from tldw_chatbook.RAG_Admin.template_validation import (
    FALLBACK_METHODS,
    TemplateValidator,
    validate_template,
)

_FIXTURE_PATH = Path(__file__).parent / "template_validation_fixtures.json"
_TABLE = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
_CASES = _TABLE["cases"]


def _expand(value: Any) -> Any:
    """Expand fixture-only markers into non-JSON-native Python values."""
    if isinstance(value, dict):
        if set(value.keys()) == {"$$set"}:
            return set(value["$$set"])
        return {key: _expand(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand(item) for item in value]
    return value


def _counts_by_field(entries: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        counts[entry["field"]] = counts.get(entry["field"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# AC 14 - the fixture table (transcribed from the pinned endpoint source)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "case", _CASES, ids=[case["id"] for case in _CASES]
)
def test_fixture_table(case: dict[str, Any]) -> None:
    result = validate_template(_expand(case["input"]))
    expected = case["expected"]
    assert result["valid"] is expected["valid"]
    assert _counts_by_field(result["errors"]) == expected["errors_by_field"]
    assert len(result["warnings"]) == expected["warnings_count"]
    for entry in result["errors"] + result["warnings"]:
        assert set(entry.keys()) == {"field", "message"}
        assert isinstance(entry["field"], str)
        assert isinstance(entry["message"], str)


@pytest.mark.parametrize(
    "bad",
    [None, "words", 42, [], {"chunking": "not-a-dict"}, {"chunking": {"method": None}}],
)
def test_never_raises_on_invalid_input(bad: Any) -> None:
    """AC 14: invalid input yields a result, never an exception (the endpoint
    converts validation failures into a 200-with-errors payload, and its outer
    except maps everything else to ``Validation error: ...``)."""
    result = validate_template(bad)
    assert result["valid"] is False
    assert result["errors"]


# ---------------------------------------------------------------------------
# AC 14 - live-registry resolution (and the stale fallback, pinned as parity)
# ---------------------------------------------------------------------------


def test_methods_resolve_against_the_live_engine_registry() -> None:
    """AC 14: the default methods source is the LIVE engine registry
    (``Chunking.engine.chunker.Chunker().get_available_methods()``), not a
    transcribed list - so ``fixed_size`` (absent from the endpoint's stale
    fallback list, §11 item 13) validates here."""
    from tldw_chatbook.Chunking.engine.chunker import Chunker

    live = set(Chunker().get_available_methods())
    # Guard: if the engine ever drops fixed_size, re-pin the fixture row.
    assert "fixed_size" in live
    assert validate_template({"chunking": {"method": "fixed_size"}})["valid"] is True
    unknown = validate_template({"chunking": {"method": "definitely_not_a_method"}})
    assert unknown["valid"] is False
    assert [error["field"] for error in unknown["errors"]] == ["chunking.method"]


def test_stale_fallback_list_is_reproduced_when_registry_source_fails() -> None:
    """PARITY PIN - endpoint :830-832: when the registry call fails, the
    endpoint falls back to a HARDCODED, STALE 11-name list (§11 item 13 /
    UPSTREAM_DEFECTS.md - omits ``fixed_size``, ``code``, ``code_ast``).
    Constructor-injecting that exact list must reproduce the staleness."""
    frozen = TemplateValidator(methods_source=lambda: list(FALLBACK_METHODS))
    assert frozen.validate_template({"chunking": {"method": "words"}})["valid"] is True
    rejected = frozen.validate_template({"chunking": {"method": "fixed_size"}})
    assert rejected["valid"] is False
    assert [error["field"] for error in rejected["errors"]] == ["chunking.method"]


# ---------------------------------------------------------------------------
# AC 15 - the three §7.1 warts, each pinned by a dedicated test
# ---------------------------------------------------------------------------


def test_wart_pin_unknown_operation_name_validates_clean() -> None:
    """PARITY PIN (§7.1 wart 1 / AC 15): the endpoint never checks that an
    operation NAME is registered (:940-948 requires only that the ``operation``
    KEY exists). An unknown op validates clean and is warned-and-skipped at
    runtime. If a later change makes this test fail by REJECTING the unknown
    op, that is a parity break with the server, not a fix - see §11 item 11
    before "correcting" it."""
    result = validate_template(
        {
            "chunking": {"method": "words"},
            "preprocessing": [{"operation": "no_such_operation_anywhere"}],
            "postprocessing": [{"operation": "also_totally_unknown"}],
        }
    )
    assert result == {"valid": True, "errors": [], "warnings": []}


def test_wart_pin_operation_key_required_even_though_runtime_accepts_type() -> None:
    """PARITY PIN (§7.1 wart 2 / AC 15): the vendored runtime accepts the
    ``{type, params}`` op spelling, but validation requires ``operation``
    (:940-948) - so a template that RUNS fails validation. Deliberate
    asymmetry, filed upstream as §11 item 11; do not "fix" one side here."""
    result = validate_template(
        {
            "chunking": {"method": "words"},
            "preprocessing": [{"type": "strip_headers", "params": {}}],
        }
    )
    assert result["valid"] is False
    assert [error["field"] for error in result["errors"]] == ["preprocessing[0]"]


def test_wart_pin_unknown_top_level_keys_are_silently_ignored() -> None:
    """PARITY PIN (§7.1 wart 3 / AC 15): the endpoint's pydantic first pass
    (extra=ignore) silently DROPS unknown top-level keys before the
    hand-rolled checks ever see them, so such templates validate clean.
    Chatbook matches this; §7.1's one carve-out (``name``/``description``/
    ``tags`` never enter the validated body) lives in the CRUD layer, not
    here."""
    result = validate_template(
        {
            "chunking": {"method": "words"},
            "schema_version": 99,
            "some_future_field": {"nested": True},
        }
    )
    assert result == {"valid": True, "errors": [], "warnings": []}


# ---------------------------------------------------------------------------
# Spec §7 wiring - the scope service routes local mode to the validator
# ---------------------------------------------------------------------------


class _MinimalLocalService:
    """The local validation route needs no backing service."""


@pytest.mark.asyncio
async def test_scope_service_local_mode_validates_without_hard_raise() -> None:
    """Spec §7 wiring: local-mode ``validate_template_config`` returns the
    local validator's verdict instead of raising "Server retrieval-admin
    backend is required" (the pre-Task-6 behavior)."""
    scope = RAGAdminScopeService(
        local_service=_MinimalLocalService(), server_service=None
    )
    result = await scope.validate_template_config(
        mode=RAGAdminBackend.LOCAL,
        template_config={"chunking": {"method": "words"}},
    )
    assert result == {"valid": True, "errors": [], "warnings": []}


@pytest.mark.asyncio
async def test_scope_service_local_mode_returns_validator_verdict() -> None:
    scope = RAGAdminScopeService(
        local_service=_MinimalLocalService(), server_service=None
    )
    result = await scope.validate_template_config(
        mode="local", template_config={}
    )
    assert result["valid"] is False
    assert [error["field"] for error in result["errors"]] == ["chunking"]


@pytest.mark.asyncio
async def test_scope_service_default_mode_is_local_for_validation() -> None:
    scope = RAGAdminScopeService(
        local_service=_MinimalLocalService(), server_service=None
    )
    result = await scope.validate_template_config(
        template_config={"chunking": {"method": "words"}}
    )
    assert result["valid"] is True
