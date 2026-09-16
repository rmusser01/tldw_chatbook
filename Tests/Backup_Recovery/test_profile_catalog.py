"""A private convenience catalog cannot grant recovery execution authority."""

import json
from pathlib import Path

import pytest

from tldw_chatbook.Backup_Recovery.profile_catalog import ProfileCatalog


def targets(tmp_path):
    config = tmp_path / "new" / "config.toml"
    data = tmp_path / "new" / "data"
    data.mkdir(parents=True)
    config.write_text("[general]\n")
    return config, data


def test_catalog_reopens_explicit_paths_without_changing_target_bytes(tmp_path):
    config, data = targets(tmp_path)
    before = config.read_bytes(), config.stat().st_mtime_ns, data.stat().st_mtime_ns
    ProfileCatalog(tmp_path).register("opaque-id", config, data)
    assert ProfileCatalog(tmp_path).resolve("opaque-id") == (config, data)
    assert (
        config.read_bytes(),
        config.stat().st_mtime_ns,
        data.stat().st_mtime_ns,
    ) == before


def test_original_plan_registers_under_a_new_private_control_directory(tmp_path):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path / "control")
    catalog.register("opaque-id", config, data)
    assert ProfileCatalog(tmp_path / "control").resolve("opaque-id") == (config, data)
    assert catalog.root.parent.stat().st_mode & 0o777 == 0o700


def test_missing_catalog_lookup_creates_nothing(tmp_path):
    control = tmp_path / "absent"
    catalog = ProfileCatalog(control)
    with pytest.raises((FileNotFoundError, ValueError)):
        catalog.resolve("missing")
    assert not control.exists()


def test_exact_registration_retries_but_never_reassigns_an_id(tmp_path):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    catalog.register("opaque-id", config, data)
    catalog.register("opaque-id", config, data)
    other = config.with_name("other.toml")
    other.write_bytes(b"[general]\n")
    with pytest.raises(ValueError, match="catalog_mapping_changed"):
        catalog.register("opaque-id", other, data)
    assert catalog.resolve("opaque-id") == (config, data)


@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "id", "extra", "linked", "public"]
)
def test_catalog_refuses_damaged_local_records(tmp_path, damage):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    catalog.register("opaque-id", config, data)
    record = next(catalog.root.glob("*.json"))
    if damage == "missing":
        record.unlink()
    elif damage == "corrupt":
        record.write_bytes(b"{")
    elif damage in ("id", "extra"):
        doc = json.loads(record.read_bytes())
        doc["profile_id" if damage == "id" else "approved"] = "imported"
        record.write_text(json.dumps(doc))
    elif damage == "linked":
        saved = tmp_path / "saved.json"
        record.rename(saved)
        record.symlink_to(saved)
    else:
        record.chmod(0o644)
    with pytest.raises((OSError, ValueError)):
        catalog.resolve("opaque-id")


@pytest.mark.parametrize("selected", ["config", "data", "parent"])
def test_catalog_refuses_linked_target_components(tmp_path, selected):
    config, data = targets(tmp_path)
    target = {"config": config, "data": data, "parent": config.parent}[selected]
    renamed = target.with_name(target.name + "-original")
    target.rename(renamed)
    target.symlink_to(renamed, target_is_directory=selected != "config")
    catalog = ProfileCatalog(tmp_path)
    with pytest.raises((OSError, ValueError)):
        catalog.register("opaque-id", config, data)
    assert not catalog.root.exists()


def test_catalog_rechecks_target_kind_after_registration(tmp_path):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    catalog.register("opaque-id", config, data)
    data.rmdir()
    data.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="catalog_target_invalid"):
        catalog.resolve("opaque-id")


@pytest.mark.parametrize("identifier", ["", "x" * 257, "bad\x00id", None])
def test_catalog_rejects_invalid_identifiers_without_writes(tmp_path, identifier):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    with pytest.raises(ValueError, match="catalog_id_invalid"):
        catalog.register(identifier, config, data)
    assert not catalog.root.exists()


def test_constructor_rejects_relative_control_root():
    with pytest.raises(ValueError):
        ProfileCatalog(Path("relative"))


def test_catalog_records_and_directory_are_private(tmp_path):
    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    catalog.register("opaque-id", config, data)
    record = next(catalog.root.glob("*.json"))
    assert catalog.root.stat().st_mode & 0o777 == 0o700
    assert record.stat().st_mode & 0o777 == 0o600
    assert set(json.loads(record.read_bytes())) == {
        "version",
        "profile_id",
        "config",
        "data",
    }


def test_catalog_cannot_write_its_records_inside_selected_data(tmp_path):
    config, data = targets(tmp_path)
    before = data.stat().st_mtime_ns
    catalog = ProfileCatalog(data)
    with pytest.raises(ValueError, match="catalog_overlaps_data"):
        catalog.register("opaque-id", config, data)
    assert data.stat().st_mtime_ns == before
    assert not catalog.root.exists()


def test_exact_retry_reflushes_a_completed_record_after_failed_barrier(
    tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import profile_catalog

    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    original = profile_catalog.Admission._write_new_record

    def interrupted(parent, name, value):
        original(parent, name, value)
        raise OSError("simulated final barrier failure")

    monkeypatch.setattr(profile_catalog.Admission, "_write_new_record", interrupted)
    with pytest.raises(OSError):
        catalog.register("opaque-id", config, data)
    monkeypatch.setattr(profile_catalog.Admission, "_write_new_record", original)
    calls = []
    flush = profile_catalog.os.fsync

    def observed(fd):
        calls.append(fd)
        return flush(fd)

    monkeypatch.setattr(profile_catalog.os, "fsync", observed)
    catalog.register("opaque-id", config, data)
    assert calls
    assert ProfileCatalog(tmp_path).resolve("opaque-id") == (config, data)


def test_retry_does_not_skip_failed_catalog_directory_parent_barrier(
    tmp_path, monkeypatch
):
    import os

    from tldw_chatbook.Backup_Recovery import native_files, profile_catalog

    config, data = targets(tmp_path)
    catalog = ProfileCatalog(tmp_path)
    control = tmp_path.stat()
    original = native_files.flush_directory

    def interrupted(fd):
        info = os.fstat(fd)
        if (info.st_dev, info.st_ino) == (control.st_dev, control.st_ino):
            raise OSError("parent barrier failed")
        return original(fd)

    monkeypatch.setattr(native_files, "flush_directory", interrupted)
    monkeypatch.setattr(profile_catalog, "flush_directory", interrupted)
    with pytest.raises(OSError, match="parent barrier failed"):
        catalog.register("opaque-id", config, data)
    assert catalog.root.exists()
    with pytest.raises(OSError, match="parent barrier failed"):
        catalog.register("opaque-id", config, data)
    monkeypatch.setattr(native_files, "flush_directory", original)
    monkeypatch.setattr(profile_catalog, "flush_directory", original)
    catalog.register("opaque-id", config, data)
    assert catalog.resolve("opaque-id") == (config, data)
