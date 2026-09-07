import pytest

from tldw_chatbook.tldw_api.skills_schemas import (
    validate_supporting_file_path,
    BundleFileInfo,
    MAX_SUPPORTING_FILES_COUNT,
)


@pytest.mark.parametrize("good", [
    "notes.md",
    "scripts/build.sh",
    "references/api/reference.md",
    "assets/img/logo.png",
])
def test_validate_supporting_file_path_accepts_nested(good):
    assert validate_supporting_file_path(good) == good


@pytest.mark.parametrize("bad", [
    "/abs.md",                # absolute
    "../escape.md",           # traversal
    "a/../b.md",              # traversal segment
    "a/./b.md",               # dot segment
    "a//b.md",                # empty segment
    "back\\slash.md",         # backslash
    "SKILL.md",               # reserved body (root)
    "refs/SKILL.md",          # nested shadow (any case)
    "refs/skill.md",          # nested shadow wrong-case
    "-leading-dash.md",       # segment must start alnum
    ".dotfile",                # leading dot
    "a/" * 9 + "deep.md",     # depth > 8
    "x" * 256,                # length > 255
])
def test_validate_supporting_file_path_rejects(bad):
    with pytest.raises(ValueError):
        validate_supporting_file_path(bad)


def test_bundle_file_info_shape():
    info = BundleFileInfo(path="assets/logo.png", size=1234, executable=False, is_text=False)
    assert info.model_dump() == {
        "path": "assets/logo.png", "size": 1234, "executable": False, "is_text": False,
    }


def test_caps_raised():
    assert MAX_SUPPORTING_FILES_COUNT == 500
