"""Selected foreign roots expose reviewed copies, never grants or connections."""

import json

import pytest


def test_import_preview_reads_allowlisted_catalog_and_excludes_profile_state(tmp_path):
    from tldw_chatbook.Plugins.imports import inspect_app_import

    root = tmp_path / "selected-codex"
    catalog = root / ".agents/plugins/marketplace.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(
        json.dumps(
            {
                "plugins": [
                    {
                        "name": "remote",
                        "source": {
                            "source": "url",
                            "url": "https://example.test/repo.git",
                        },
                    }
                ]
            }
        )
    )
    (root / "auth.json").write_text('{"secret":"never-import"}')
    (root / "config.toml").write_text(
        '[plugins."remote@market"]\nenabled = true\n[projects."secret"]\ntrust_level="trusted"\n'
    )
    preview = inspect_app_import("codex", root)
    assert len(preview.candidates) == 1
    assert "credentials" in preview.exclusions and "enablement" in preview.exclusions
    assert "never-import" not in preview.model_dump_json()
    assert preview.reviewed_digest


def test_import_preview_rejects_links_and_reports_partial_cache(tmp_path):
    from tldw_chatbook.Plugins.imports import inspect_app_import

    root = tmp_path / "selected-cursor"
    root.mkdir()
    (root / "mcp.json").write_text('{"mcpServers": {}}')
    preview = inspect_app_import("cursor", root)
    assert not preview.candidates
    assert "clean_source_required" in preview.exclusions
    linked = tmp_path / "link"
    linked.symlink_to(root)
    with pytest.raises(ValueError):
        inspect_app_import("cursor", linked)


def test_explicit_package_preview_binds_content_changes(native_package):
    from tldw_chatbook.Plugins.imports import inspect_app_import

    root = native_package()
    before = inspect_app_import("codex", root)
    (root / "skills/review/SKILL.md").write_text("changed")
    after = inspect_app_import("codex", root)
    assert before.reviewed_digest != after.reviewed_digest
    assert before.root_identity == after.root_identity
