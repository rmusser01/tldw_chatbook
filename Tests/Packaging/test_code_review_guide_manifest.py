"""Only the two fixed review recipes belong in runtime distributions."""

import pytest

from Packaging import check_manifest

GUIDES = {
    f"tldw_chatbook/Agents/code_review_guides/{name}.md"
    for name in ("review", "verify")
}


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_review_guides_are_required_in_distribution(kind):
    required = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS")
    assert GUIDES <= required
    for missing in GUIDES:
        assert (
            f"{kind}: missing required path: {missing}"
            in check_manifest._validate_content(
                kind,
                required - {missing},
                required_paths=required,
                required_globs=set(),
            )
        )


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_unrelated_review_markdown_is_forbidden(kind):
    required = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS")
    extra = "tldw_chatbook/Agents/code_review_guides/development.md"
    errors = check_manifest._validate_content(
        kind, required | GUIDES | {extra}, required_paths=required, required_globs=set()
    )
    assert errors == [f"{kind}: forbidden development Markdown: {extra}"]


def test_review_wait_example_uses_runtime_handle_selector():
    import json
    import re
    from importlib.resources import files

    from tldw_chatbook.Agents.tool_catalog import WAIT_AGENTS_SCHEMA

    text = (
        files("tldw_chatbook.Agents")
        .joinpath("code_review_guides/review.md")
        .read_text()
    )
    example = json.loads(re.search(r"wait_agents\((\{[^\n]+\})\)", text).group(1))
    assert example == {"ids": ["HANDLE"]}
    assert set(example) <= WAIT_AGENTS_SCHEMA.parameters["properties"].keys()
