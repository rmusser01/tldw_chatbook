"""Tests for the Tool ABC and risk_tags property."""


def test_tool_risk_tags_defaults_empty_and_is_concrete():
    """Every existing Tool subclass must keep working without declaring tags."""
    from tldw_chatbook.Tools.tool_executor import CalculatorTool, DateTimeTool

    assert CalculatorTool().risk_tags == ()
    assert DateTimeTool().risk_tags == ()


def test_tool_subclass_may_declare_risk_tags():
    from tldw_chatbook.Tools.tool_executor import Tool

    class Mutating(Tool):
        @property
        def name(self) -> str:
            return "mutating"

        @property
        def description(self) -> str:
            return "d"

        @property
        def parameters(self) -> dict:
            return {"type": "object", "properties": {}}

        @property
        def risk_tags(self) -> tuple[str, ...]:
            return ("mutates",)

        async def execute(self, **kwargs):
            return {}

    assert Mutating().risk_tags == ("mutates",)
