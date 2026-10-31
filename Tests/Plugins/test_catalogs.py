"""Catalog discovery preserves roots and unsupported siblings without granting authority."""

from pathlib import Path

import pytest


def test_catalog_local_entry_resolves_against_catalog_root(tmp_path):
    from tldw_chatbook.Plugins.catalogs import catalog_local_path

    root = tmp_path / "repo"
    root.mkdir()
    assert catalog_local_path(root, "./plugins/review") == root / "plugins" / "review"


def test_catalog_paths_refuse_escape_and_accept_contained_control(tmp_path):
    from tldw_chatbook.Plugins.catalogs import catalog_local_path

    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "repo"
    root.mkdir()
    (root / "link").symlink_to(outside)
    for path in ("../outside", str(outside), "link/secret", "C:\\secrets"):
        with pytest.raises(ValueError):
            catalog_local_path(root, path)
    assert catalog_local_path(root, "./valid") == root / "valid"


def test_explicit_local_selector_precondition(tmp_path):
    from tldw_chatbook.Plugins.git_source import select_local_source

    assert Path(select_local_source(tmp_path).repository) == tmp_path.resolve()


@pytest.mark.parametrize(
    "catalog_path",
    [
        ".agents/plugins/marketplace.json",
        ".cursor-plugin/marketplace.json",
        ".claude-plugin/marketplace.json",
    ],
)
def test_catalog_shapes_keep_valid_siblings_and_source_qualified_names(catalog_path):
    import json

    from tldw_chatbook.Plugins.catalogs import parse_catalog

    source = (
        {"source": "local", "path": "./plugins/review"}
        if catalog_path.startswith(".agents")
        else "./plugins/review"
    )
    data = json.dumps(
        {
            "name": "market",
            "owner": {"name": "fixture"},
            "plugins": [
                {"name": "review", "source": source, "skills": []},
                {
                    "name": "unsupported",
                    "source": {"source": "npm", "package": "do-not-run"},
                },
                {
                    "name": "broken",
                    "source": {
                        "source": "url",
                        "url": "https://example.test/a/repo",
                        "ref": "main",
                        "sha": "a" * 40,
                    },
                },
                42,
            ],
        }
    ).encode()
    first = parse_catalog(data, source_id="a", catalog_path=catalog_path)
    second = parse_catalog(data, source_id="b", catalog_path=catalog_path)
    assert first.entries[0].local_path == "plugins/review"
    assert first.entries[0].support == "supported"
    assert '"skills":[]' in first.entries[0].overlays[0]
    assert all(entry.support == "unsupported" for entry in first.entries[1:])
    assert first.entries[0].identity != second.entries[0].identity


def test_catalog_explicit_selection_and_metadata_root(tmp_path):
    from tldw_chatbook.Plugins.catalogs import select_catalog

    for path in (".agents/plugins/marketplace.json", ".cursor-plugin/marketplace.json"):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text('{"plugins":[]}')
    with pytest.raises(ValueError, match="selection_required"):
        select_catalog(tmp_path)
    assert (
        select_catalog(tmp_path, ".cursor-plugin/marketplace.json")
        == ".cursor-plugin/marketplace.json"
    )


def test_catalog_bounds_and_duplicate_names():
    import json

    from tldw_chatbook.Plugins.catalogs import MAX_CATALOG_BYTES, parse_catalog

    parse = lambda data: parse_catalog(
        data, source_id="a", catalog_path=".claude-plugin/marketplace.json"
    )
    for data in (
        b" " * (MAX_CATALOG_BYTES + 1),
        b'{"plugins":[],"plugins":[]}',
        json.dumps({"plugins": [None] * 10001}).encode(),
    ):
        with pytest.raises(ValueError):
            parse(data)
    entries = parse(
        json.dumps({"plugins": [{"name": "same", "source": "./a"}] * 2}).encode()
    ).entries
    assert len({entry.identity for entry in entries}) == 2
    assert all(entry.support == "unsupported" for entry in entries)
    assert len(parse(b'{"plugins":[]}').entries) == 0


def test_remote_git_selector_and_catalog_prefix_are_closed():
    import json

    from tldw_chatbook.Plugins.catalogs import parse_catalog
    from tldw_chatbook.Plugins.git_source import SourceLocator

    data = json.dumps(
        {
            "metadata": {"pluginRoot": "./packages"},
            "plugins": [
                {"name": "local", "source": "./one"},
                {
                    "name": "remote",
                    "source": {
                        "source": "git-subdir",
                        "url": "https://example.test/repo.git",
                        "path": "./one",
                        "sha": "a" * 40,
                    },
                },
                {
                    "name": "host",
                    "source": {"source": "local", "path": "/private/secret"},
                },
                {
                    "name": "file",
                    "source": {"source": "url", "url": "file:///private/secret"},
                },
            ],
        }
    ).encode()
    entries = parse_catalog(
        data, source_id="a", catalog_path=".cursor-plugin/marketplace.json"
    ).entries
    assert entries[0].local_path == "packages/one"
    locator = SourceLocator.model_validate_json(entries[1].package_locator)
    assert locator.ref == "a" * 40 and locator.subdirectory == "one"
    assert entries[2].support == entries[3].support == "unsupported"


def test_catalog_deep_nonentry_metadata_is_bounded():
    from tldw_chatbook.Plugins.catalogs import parse_catalog

    data = b'{"plugins":[],"ignored":' + b"[" * 40 + b"0" + b"]" * 40 + b"}"
    with pytest.raises(ValueError, match="depth"):
        parse_catalog(
            data, source_id="a", catalog_path=".claude-plugin/marketplace.json"
        )
