"""Legacy exports must refuse selected Mix-ins before producing partial output."""

import csv
import io
import zipfile
from pathlib import Path

import pytest

from tldw_chatbook.DB.Prompts_DB import PromptsDatabase, export_prompts_formatted


@pytest.fixture
def export_db():
    db = PromptsDatabase(":memory:", client_id="mixin-export-test")
    ids = {}
    for kind in ("prompt", "recipe"):
        _, ids[kind], _ = db.add_prompt(
            f"Ordinary {kind}",
            "Author",
            "Details",
            system_prompt="System text",
            user_prompt="User text",
            keywords=["ordinary-only", "shared"],
            artifact_type=kind,
        )
    db.add_prompt(
        "PRIVATE MIXIN NAME",
        None,
        None,
        system_prompt="",
        user_prompt="PRIVATE MIXIN BODY",
        keywords=["mixin-only", "shared"],
        artifact_type="mixin",
        prompt_format="structured",
        prompt_schema_version=3,
        prompt_definition={
            "kind": "text_mixin",
            "schema_version": 3,
            "body": "PRIVATE MIXIN BODY",
            "placement": "after",
        },
    )
    try:
        yield db, ids
    finally:
        db.close_connection()


@pytest.mark.parametrize("export_format", ["csv", "markdown"])
@pytest.mark.parametrize("keywords", [None, ["shared"], ["mixin-only"], [" "]])
@pytest.mark.parametrize("include_keywords", [False, True])
def test_selected_mixins_refuse_before_keywords_or_output(
    export_db, monkeypatch, export_format, keywords, include_keywords
):
    db, _ = export_db
    side_effects = []

    def forbidden(*args, **kwargs):
        side_effects.append("keyword or output side effect")
        raise AssertionError("Export must refuse selected Mix-ins first.")

    with monkeypatch.context() as patch:
        patch.setattr(db, "fetch_keywords_for_prompt", forbidden)
        patch.setattr("tempfile.mkstemp", forbidden)
        patch.setattr("tempfile.mkdtemp", forbidden)
        patch.setattr("os.mkdir", forbidden)
        patch.setattr("builtins.open", forbidden)
        patch.setattr("zipfile.ZipFile", forbidden)
        status, output = export_prompts_formatted(
            db,
            export_format,
            filter_keywords=keywords,
            include_associated_keywords=include_keywords,
        )

    assert side_effects == []
    assert output == "None"
    assert "mix-in" in status.lower()
    assert "library" in status.lower() or "chatbook" in status.lower()
    assert "PRIVATE" not in status


@pytest.mark.parametrize(
    ("export_format", "template"),
    [
        ("csv", "Basic Template"),
        ("markdown", "Basic Template"),
        ("markdown", "Detailed Template"),
    ],
)
def test_filter_excluding_mixins_preserves_ordinary_export_bytes(
    export_db, export_format, template
):
    db, ids = export_db
    status, output = export_prompts_formatted(
        db,
        export_format,
        filter_keywords=["  ORDINARY-ONLY  "],
        markdown_template_name=template,
    )
    assert "Successfully exported 2 prompts" in status
    path = Path(output)
    try:
        if export_format == "csv":
            rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
            assert rows == [
                [
                    "Name",
                    "UUID",
                    "Author",
                    "Details",
                    "System Prompt",
                    "User Prompt",
                    "Keywords",
                ],
                *[
                    [
                        f"Ordinary {kind}",
                        ids[kind],
                        "Author",
                        "Details",
                        "System text",
                        "User text",
                        "ordinary-only, shared",
                    ]
                    for kind in ("prompt", "recipe")
                ],
            ]
        else:
            with zipfile.ZipFile(path) as archive:
                bodies = [
                    archive.read(name).decode("utf-8") for name in archive.namelist()
                ]
            expected = []
            for kind in ("prompt", "recipe"):
                if template == "Basic Template":
                    expected.append(
                        f"# Ordinary {kind} ({ids[kind]})\n"
                        "**Author**: Author\n**Details**: Details\n"
                        "**System Prompt**:\n```\nSystem text\n```\n"
                        "**User Prompt**:\n```\nUser text\n```\n"
                        "**Keywords**: ordinary-only, shared"
                    )
                else:
                    expected.append(
                        f"# Ordinary {kind}\n**UUID**: {ids[kind]}\n\n"
                        "## Author\n**Author**: Author\n\n"
                        "## Description\n**Details**: Details\n\n"
                        "## System Prompt\n```\nSystem text\n```\n\n"
                        "## User Prompt\n```\nUser text\n```\n\n"
                        "## Keywords\n**Keywords**: ordinary-only, shared"
                    )
            assert bodies == expected
    finally:
        path.unlink()
