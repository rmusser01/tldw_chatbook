"""Actual native registration geometry; reuses the genuine three-boundary proof."""

import json
from contextlib import contextmanager

from Tests.Backup_Recovery import test_mobile_consolidation_lifecycle as lifecycle
from Tests.Backup_Recovery.test_mobile_safety_source import (
    test_genuine_committed_source_three_prepared_safety_boundaries as three_boundaries,
)
from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.mobile_consolidation import preview_selected_mobile


def test_genuine_covering_unprefixed_multiroot_source(tmp_path, monkeypatch):
    native_fixture = lifecycle._capture_fixture
    namespace = "installed.artwork"
    geometry = {}

    @contextmanager
    def registered(*args, **kwargs):
        with native_fixture(*args, **kwargs) as values:
            pass
        session, journal, config, stores, old_preview, inventory, authority = values
        assert not session._active
        package_roots = {
            item.path
            for item in inventory.items
            if item.owner == "persona.visual_identity_builtin"
            and item.metadata is not None
            and item.metadata.relative_path == ""
            and item.status == "included_directory"
        }
        assert len(package_roots) == 1
        package = next(iter(package_roots))
        broader = package.parent
        spare = tmp_path / "separate-native-source"
        spare.mkdir(mode=0o700)
        assert not namespace.startswith("backup.source.")
        profiles = bootstrap._records(authority.control_root.parent)[1]
        authority.register(namespace, (spare, broader))
        registry = bootstrap._registry(authority.control_root.parent)
        entry = registry[namespace]
        assert entry["roots"] == [str(spare), str(broader)]
        assert entry["pending"] is None and not entry["proposed"]
        assert bootstrap._records(authority.control_root.parent)[1] == profiles
        # New actual review after registration; never edit the older witness.
        preview = preview_selected_mobile(config, config)
        assert (
            not preview.issues
            and preview.source_fingerprint == old_preview.source_fingerprint
        )
        geometry.update(
            namespace=namespace,
            package=str(package),
            covering_root=str(broader),
            other_root=str(spare),
            entry=entry,
            profiles_before=profiles,
            bootstrap_root=str(authority.control_root.parent),
        )
        yield session, journal, config, stores, preview, inventory, authority

    monkeypatch.setattr(lifecycle, "_capture_fixture", registered)
    try:
        three_boundaries(tmp_path, monkeypatch)
        component = json.loads((tmp_path / "mobile-safety-component.json").read_text())
        assert len(component["calls"]) == 3
        assert all(namespace in call["session_names"] for call in component["calls"])
        native = json.loads((tmp_path / "builtin-existing-binding.json").read_text())
        assert native["registry"][namespace] == geometry["entry"]
        assert native["registry_after_public"][namespace] == geometry["entry"]
        assert (
            native["profiles"]
            == geometry["profiles_before"]
            == native["profiles_after_public"]
        )
        geometry["three_native_checks_passed"] = True
    finally:
        (tmp_path / "mobile-safety-geometry.json").write_text(
            json.dumps(geometry, indent=2)
        )
