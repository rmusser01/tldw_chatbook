"""Behavioral coverage for generic sensitive tool-record projections."""

import json
from dataclasses import FrozenInstanceError

import pytest

from tldw_chatbook.Agents.agent_models import ToolCall, ToolRecordProjection, ToolResult
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry

RAW_ARGUMENT = "<canvas-html-argument>"
RAW_RESULT = "<canvas-html-result>"


class _PlainProvider:
    """Minimal provider whose source exercises catalog ownership variants."""

    def __init__(self, source: str) -> None:
        self.source = source

    def list_catalog(self):
        from tldw_chatbook.Agents.agent_models import ToolCatalogEntry

        return [
            ToolCatalogEntry(
                id=f"{self.source}:canvas_like",
                name="canvas_like",
                one_line_description="test tool",
                source=self.source,
            )
        ]

    def load_schema(self, tool_id):
        from tldw_chatbook.Agents.agent_models import ToolSchema

        return ToolSchema(tool_id, "canvas_like", "test tool", {"type": "object"})

    def invoke(self, tool_id, args):
        return ToolResult(ok=True, content=RAW_RESULT)

class _Provider(_PlainProvider):
    def __init__(self, source: str, *, raises: bool = False) -> None:
        super().__init__(source)
        self.raises = raises

    def project_tool_record(self, audience, call, result):
        if self.raises:
            raise RuntimeError(RAW_RESULT)
        return ToolRecordProjection(
            arguments={"audience": audience},
            content=f"{audience}-content",
            error="",
            ok=result.ok if result is not None else None,
        )


@pytest.mark.parametrize("source", ["builtin", "local", "skill", "mcp"])
@pytest.mark.parametrize("audience", ["display", "log", "cycle", "continuation"])
def test_catalog_default_projection_preserves_existing_provider_values(source, audience):
    """Removing the fallback would change every pre-Canvas provider's records."""
    # Existing providers do not implement the optional hook.
    provider = _PlainProvider(source)
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    call = ToolCall("canvas_like", {"html": RAW_ARGUMENT}, call_id="call-1")
    result = ToolResult(ok=True, content=RAW_RESULT)

    projected = registry.project_tool_record(audience, call, result)

    assert dict(projected.arguments) == {"html": RAW_ARGUMENT}
    assert projected.content == RAW_RESULT
    assert projected.error == ""
    assert projected.ok is True


@pytest.mark.parametrize(
    ("value", "encoded"),
    [
        (float("nan"), "NaN"),
        (float("inf"), "Infinity"),
        (float("-inf"), "-Infinity"),
    ],
)
@pytest.mark.parametrize("audience", ["display", "log", "cycle", "continuation"])
def test_catalog_default_projection_retains_legacy_nonfinite_json(value, encoded, audience):
    """The compatibility fallback must not acquire opt-in strict JSON rules."""
    registry = ToolCatalogRegistry()
    registry.register_provider(_PlainProvider("builtin"))

    projected = registry.project_tool_record(
        audience,
        ToolCall("canvas_like", {"value": value}, call_id="call-1"),
        ToolResult(ok=True, content="ok"),
    )

    assert json.dumps(dict(projected.arguments), sort_keys=True) == f'{{"value": {encoded}}}'


@pytest.mark.parametrize("audience", ["display", "log", "cycle", "continuation"])
def test_catalog_dispatches_each_audience_to_the_owning_provider(audience):
    """A wrong owner lookup would bypass a sensitive provider's redaction."""
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider("canvas"))
    projected = registry.project_tool_record(
        audience,
        ToolCall("canvas_like", {"html": RAW_ARGUMENT}, call_id="call-1"),
        ToolResult(ok=True, content=RAW_RESULT),
    )

    assert dict(projected.arguments) == {"audience": audience}
    assert projected.content == f"{audience}-content"
    assert RAW_ARGUMENT not in str(projected)
    assert RAW_RESULT not in str(projected)


@pytest.mark.parametrize("audience", ["display", "log", "cycle", "continuation"])
def test_failed_projection_is_content_free_and_immutable(audience):
    """A projector exception must not leak raw data through a fallback formatter."""
    registry = ToolCatalogRegistry()
    registry.register_provider(_Provider("canvas", raises=True))
    projected = registry.project_tool_record(
        audience,
        ToolCall("canvas_like", {"html": RAW_ARGUMENT}, call_id="call-1"),
        ToolResult(ok=False, error=RAW_RESULT),
    )

    assert RAW_ARGUMENT not in str(projected)
    assert RAW_RESULT not in str(projected)
    assert dict(projected.arguments) == {
        "tool_name": "canvas_like",
        "call_id": "call-1",
        "success": False,
        "error_category": "RuntimeError",
    }
    with pytest.raises(FrozenInstanceError):
        projected.content = "leak"


@pytest.mark.parametrize("malformed", ["nonfinite", "metadata"])
@pytest.mark.parametrize("audience", ["display", "log", "cycle", "continuation"])
def test_invalid_opt_in_projection_fails_closed_without_payload_or_exception_leak(
    malformed, audience
):
    """Strict validation applies only to projectors that opted into the hook."""

    class InvalidProvider(_PlainProvider):
        def project_tool_record(self, _audience, _call, _result):
            if malformed == "nonfinite":
                return ToolRecordProjection(
                    arguments={"payload": RAW_ARGUMENT, "value": float("nan")},
                    content=RAW_RESULT,
                    error=RAW_RESULT,
                    ok=True,
                )
            return ToolRecordProjection(  # type: ignore[arg-type]
                arguments={"payload": RAW_ARGUMENT},
                content=object(),
                error=RAW_RESULT,
                ok=True,
            )

    registry = ToolCatalogRegistry()
    registry.register_provider(InvalidProvider("canvas"))
    projected = registry.project_tool_record(
        audience,
        ToolCall("canvas_like", {"html": RAW_ARGUMENT}, call_id="call-1"),
        ToolResult(ok=False, error=RAW_RESULT),
    )

    assert dict(projected.arguments) == {
        "tool_name": "canvas_like",
        "call_id": "call-1",
        "success": False,
        "error_category": "ValueError" if malformed == "nonfinite" else "TypeError",
    }
    assert RAW_ARGUMENT not in str(projected)
    assert RAW_RESULT not in str(projected)
