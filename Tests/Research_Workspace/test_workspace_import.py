from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from tldw_chatbook.Research_Workspace.workspace_import import read_workspace_bundle


def _bundle(tmp_path, items, manifest_path="items/1-note.md"):
    content = BytesIO()
    with ZipFile(content, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr(
            "manifest.md",
            f"# Research workspace bundle\n\nLocal workspace `source`\n\n- Export content · note `1` · `{manifest_path}`\n",
        )
        for path, body in items:
            archive.writestr(path, body)
    target = tmp_path / "workspace.zip"
    target.write_bytes(content.getvalue())
    return target


def test_import_reads_the_exact_reviewed_markdown_from_a_workspace_bundle(tmp_path):
    bundle = _bundle(
        tmp_path, [("items/1-note.md", "# Field note\n\nCollected samples.\n")]
    )
    imported = read_workspace_bundle(bundle)
    assert len(imported.items) == 1 and len(imported.fingerprint) == 64
    assert imported.items[0].kind == "note" and imported.items[0].title == "Field note"
    assert imported.items[0].body == "Collected samples."


@pytest.mark.parametrize(
    "invalid", ["other.md", "../outside.md", "items/2-artifact.md"]
)
def test_import_rejects_undeclared_or_mismatched_members(tmp_path, invalid):
    bundle = _bundle(
        tmp_path,
        [
            ("items/1-note.md", "# Field note\n\nCollected samples.\n"),
            (invalid, "Ignored"),
        ],
    )
    with pytest.raises(ValueError, match="bundle files"):
        read_workspace_bundle(bundle)
