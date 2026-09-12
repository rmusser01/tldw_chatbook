"""Keep runtime Canvas guidance distinct from development documentation."""

import pytest

from Packaging import check_manifest

GUIDE_PATHS = {
    "tldw_chatbook/Canvas/guides/basics.md",
    "tldw_chatbook/Canvas/guides/controls.md",
    "tldw_chatbook/Canvas/guides/repair.md",
    "tldw_chatbook/Canvas/static/mermaid-authoring.txt",
}


def _errors(kind, members):
    required = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS")
    return check_manifest._validate_content(
        kind, members, required_paths=required, required_globs=set()
    )


def test_fixed_runtime_guides_are_accepted():
    members = check_manifest.REQUIRED_WHEEL_PATHS | GUIDE_PATHS
    assert not _errors("wheel", members)


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
@pytest.mark.parametrize("missing", sorted(GUIDE_PATHS))
def test_each_runtime_guide_is_required(kind, missing):
    members = getattr(check_manifest, f"REQUIRED_{kind.upper()}_PATHS") | GUIDE_PATHS
    assert f"{kind}: missing required path: {missing}" in _errors(
        kind, members - {missing}
    )


def test_unrelated_canvas_markdown_remains_forbidden():
    extra = "tldw_chatbook/Canvas/guides/development.md"
    members = check_manifest.REQUIRED_WHEEL_PATHS | GUIDE_PATHS | {extra}
    assert f"wheel: forbidden development Markdown: {extra}" in _errors(
        "wheel", members
    )
