"""Redundant absent aliases preserve existing native directory ownership."""

from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

from tldw_chatbook.Backup_Recovery import bootstrap
from tldw_chatbook.Backup_Recovery.control_records import (
    admission_authority,
    bind_profile,
)
from tldw_chatbook.Backup_Recovery.limits import ArchiveLimits
from tldw_chatbook.Backup_Recovery.storage_admission import copy_capture_file
from tldw_chatbook.Utils.platform_files import os as native_os


def _file(path, value=b"source"):
    path.write_bytes(value)
    path.chmod(0o600)
    return path


@pytest.fixture
def enrolled(tmp_path):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config = _file(data / "config.toml", b'name = "profile"\n')
    child = _file(data / "history.jsonl")
    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    authority.register("parent", (data,))
    authority.register("child", (child,))
    bind_profile(root, config, ("parent", "child"), authority.control_root)
    return authority, root, data, config, child


def test_deleted_redundant_alias_keeps_raw_binding_and_native_group(enrolled):
    authority, root, data, config, child = enrolled
    before = (authority.control_root / "registry.json").read_bytes()
    _, profiles = bootstrap._records(root)
    profile = profiles[0]
    child.unlink()
    assert bootstrap._binding(config, [profile], bootstrap._registry(root)) == profile
    with authority.normal(("child",)):
        assert {"parent", "child"} <= set(authority._observed_groups[("child",)])
    with authority.maintenance(("child",), 3) as session:
        assert data in session._roots and child not in session._roots
        assert (data, True) in session._publication_roots
    assert (authority.control_root / "registry.json").read_bytes() == before
    assert bootstrap._records(root)[1] == profiles


def test_every_referencing_profile_must_own_covering_parent(enrolled, tmp_path):
    authority, root, _, _, child = enrolled
    foreign = _file(tmp_path / "foreign.toml")
    authority.register("foreign", (foreign,))
    bind_profile(root, foreign, ("foreign", "child"), authority.control_root)
    child.unlink()
    with pytest.raises((OSError, ValueError)), authority.normal(("parent",)):
        pytest.fail("A supplied own profile cannot hide the foreign reference")


def test_all_referencing_profiles_coown_the_existing_parent(enrolled, tmp_path):
    authority, root, data, _, child = enrolled
    other = _file(tmp_path / "other.toml")
    authority.register("other", (other,))
    bind_profile(root, other, ("other", "parent", "child"), authority.control_root)
    child.unlink()
    with authority.maintenance(("child",), 3) as session:
        assert data in session._roots and child not in session._roots


def test_unreferenced_missing_alias_is_not_inferred_owned(tmp_path):
    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    child = _file(data / "child")
    authority = admission_authority(tmp_path / "bootstrap")
    authority.register("parent", (data,))
    authority.register("child", (child,))
    child.unlink()
    with pytest.raises((OSError, ValueError)), authority.normal(("parent",)):
        pytest.fail("Directory overlap does not replace fixed profile ownership")


@pytest.mark.parametrize("change", ["standalone", "parent_link", "parent_inode"])
def test_missing_alias_cannot_infer_new_parent_authority(enrolled, tmp_path, change):
    authority, root, data, config, child = enrolled
    _, profiles = bootstrap._records(root)
    child.unlink()
    if change == "standalone":
        isolated = _file(tmp_path / "isolated")
        # Keep this independent alias unenrolled in the profile: it must still
        # fail global admission instead of silently disappearing from the graph.
        child.write_bytes(b"temporarily present for registration")
        authority.register("isolated", (isolated,))
        child.unlink()
        isolated.unlink()
    else:
        saved = tmp_path / "saved"
        data.rename(saved)
        if change == "parent_link":
            data.symlink_to(saved, target_is_directory=True)
        else:
            data.mkdir(mode=0o700)
            _file(config, (saved / config.name).read_bytes())
    with pytest.raises((OSError, ValueError)), authority.normal(("parent",)):
        pytest.fail("Missing aliases require the original native parent")
    if change != "standalone":
        with pytest.raises((OSError, ValueError)):
            bootstrap._binding(config, profiles, bootstrap._registry(root))


@pytest.mark.parametrize("replace_after_selection", [False, True])
def test_reappearing_source_needs_its_current_exact_identity(
    enrolled, tmp_path, replace_after_selection
):
    authority, _, data, _, child = enrolled
    child.unlink()
    stage = tmp_path / "private"
    stage.mkdir(mode=0o700)
    limits = ArchiveLimits()
    with authority.maintenance(("child",), 3) as session:
        assert session._roots == (data,)
        _file(child, b"reappeared")
        info = native_os.stat(child)
        with session._capture_bound_sources(
            ((child, info.st_dev, info.st_ino),), stage, limits, limits.expanded_bytes
        ):
            if replace_after_selection:
                _file(data / "replacement").replace(child)
                with pytest.raises((OSError, ValueError, RuntimeError)):
                    copy_capture_file(
                        "config", child, stage / "copy", Event(), max_bytes=1024
                    )
            else:
                copy_capture_file(
                    "config", child, stage / "copy", Event(), max_bytes=1024
                )
                assert (stage / "copy").read_bytes() == b"reappeared"


def test_config_absence_is_not_qualified_by_redundant_root_rule(enrolled):
    _, root, _, config, child = enrolled
    _, profiles = bootstrap._records(root)
    child.unlink()
    config.unlink()
    assert bootstrap._binding(config, profiles, bootstrap._registry(root)) is None


def _physical_roots(registry):
    from tldw_chatbook.Backup_Recovery.effective_roots import effective_roots

    return effective_roots(
        (Path(path) for entry in registry.values() for path in entry["roots"]),
        registry.values(),
    )


def test_native_effective_helper_keeps_parent_and_present_source(enrolled):
    _, root, data, _, child = enrolled
    registry = bootstrap._registry(root)
    assert {data, child} <= set(_physical_roots(registry))
    child.unlink()
    assert data in _physical_roots(registry)
    assert child not in _physical_roots(registry)


def test_native_effective_helper_walks_missing_nested_suffix(enrolled):
    authority, root, data, _, _ = enrolled
    nested = data / "nested"
    nested.mkdir(mode=0o700)
    leaf = _file(nested / "leaf")
    authority.register("nested-leaf", (leaf,))
    registry = bootstrap._registry(root)
    leaf.unlink()
    nested.rmdir()
    assert leaf not in _physical_roots(registry)
    assert data in _physical_roots(registry)


@pytest.mark.parametrize("history", ["other_entry", "multiple_roots", "remapped"])
def test_parent_proof_cannot_borrow_another_enrolled_inode(tmp_path, history):
    first, second = tmp_path / "first", tmp_path / "second"
    for path in (first, second):
        path.mkdir(mode=0o700)
        _file(path / "config.toml", b'name = "unchanged"\n')
    child = _file(first / "history.jsonl")
    root = tmp_path / "bootstrap"
    authority = admission_authority(root)
    authority.register("child", (child,))
    authority.register(
        "first", (first, second) if history == "multiple_roots" else (first,)
    )
    if history != "multiple_roots":
        authority.register("second", (second,))
    names = (
        ("child", "first")
        if history == "multiple_roots"
        else ("child", "first", "second")
    )
    bind_profile(root, first / "config.toml", names, authority.control_root)
    if history == "remapped":
        authority.remap("first", (second,), 3)
        authority.remap("first", (first,), 3)
    before = (authority.control_root / "registry.json").read_bytes()
    child.unlink()
    if history == "other_entry":
        first.rename(tmp_path / "original")
        second.rename(first)
        second.mkdir(mode=0o700)
    assert child in _physical_roots(bootstrap._registry(root))
    assert (authority.control_root / "registry.json").read_bytes() == before


def test_pinned_parent_cannot_be_replaced_during_absence_proof(
    enrolled, tmp_path, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import effective_roots as proof

    _, root, data, config, child = enrolled
    replacement = tmp_path / "replacement"
    replacement.mkdir(mode=0o700)
    _file(replacement / config.name, config.read_bytes())
    child.unlink()
    registry = bootstrap._registry(root)
    native_pin = proof.pinned_directory
    swapped = False

    @contextmanager
    def swap_after_pin(path):
        nonlocal swapped
        with native_pin(path) as descriptor:
            if path == data and not swapped:
                swapped = True
                data.rename(tmp_path / "previous")
                replacement.rename(data)
            yield descriptor

    monkeypatch.setattr(proof, "pinned_directory", swap_after_pin)
    assert child in _physical_roots(registry)
    assert swapped


def test_failed_child_chain_recheck_cannot_keep_stale_absence_proof(
    enrolled, monkeypatch
):
    from tldw_chatbook.Backup_Recovery import effective_roots as proof

    authority, root, data, _, _ = enrolled
    nested = data / "nested"
    nested.mkdir(mode=0o700)
    leaf = _file(nested / "leaf")
    authority.register("nested-leaf", (leaf,))
    leaf.unlink()
    registry = bootstrap._registry(root)
    native_pin = proof.pinned_directory
    checked = 0

    @contextmanager
    def move_child_at_final_parent_check(path):
        nonlocal checked
        with native_pin(path) as descriptor:
            if path == data:
                checked += 1
                if checked == 2:
                    nested.rename(data / "moved")
            yield descriptor

    monkeypatch.setattr(proof, "pinned_directory", move_child_at_final_parent_check)
    assert leaf in _physical_roots(registry)
    assert checked >= 2
