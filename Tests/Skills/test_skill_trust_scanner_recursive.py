import os
from pathlib import Path

from tldw_chatbook.Skills_Interop.skill_trust_scanner import scan_skill_directory


def _write(p: Path, data: bytes, mode: int | None = None):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    if mode is not None:
        os.chmod(p, mode)


def test_recurses_and_classifies(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"---\nname: demo\n---\nbody\n")
    _write(d / "references" / "api.md", b"# api\n")
    _write(d / "scripts" / "run.sh", b"#!/bin/sh\necho hi\n", mode=0o755)
    _write(d / "assets" / "logo.png", b"\x89PNG\x00\x01binary")
    snap = scan_skill_directory("demo", d)
    fps = {f.relative_path: f for f in snap.fingerprints}
    assert set(fps) == {"SKILL.md", "references/api.md", "scripts/run.sh", "assets/logo.png"}
    assert fps["SKILL.md"].file_type == "skill"
    assert fps["assets/logo.png"].file_type == "supporting_binary"
    assert "assets/logo.png" not in snap.text_files       # binary not decoded
    assert "references/api.md" in snap.text_files
    assert fps["scripts/run.sh"].executable is True
    assert snap.unsupported_paths == ()


def test_prunes_junk(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"x")
    _write(d / ".git" / "config", b"junk")
    _write(d / "__pycache__" / "m.pyc", b"\x00pyc")
    _write(d / ".DS_Store", b"\x00")
    _write(d / "keep.md~", b"backup")
    snap = scan_skill_directory("demo", d)
    paths = {f.relative_path for f in snap.fingerprints}
    assert paths == {"SKILL.md"}
    assert snap.unsupported_paths == ()   # junk pruned, NOT recorded


def test_nested_skill_md_is_not_the_body(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    _write(d / "references" / "SKILL.md", b"nested")
    snap = scan_skill_directory("demo", d)
    # A nested SKILL.md is rejected as a shadow (validate_supporting_file_path),
    # so it lands in unsupported_paths, never classified as the body.
    fps = {f.relative_path: f for f in snap.fingerprints}
    assert fps["SKILL.md"].file_type == "skill"
    assert "references/SKILL.md" not in fps
    assert "references/SKILL.md" in snap.unsupported_paths


def test_flat_snapshot_byte_identical_to_legacy_ordering(tmp_path):
    # Guards the backward-compat guarantee: a flat all-text skill's fingerprint
    # entries are in iterdir name-order and carry no executable key.
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    _write(d / "b.md", b"bbb")
    _write(d / "a.md", b"aaa")
    snap = scan_skill_directory("demo", d)
    entries = [f.as_manifest_entry() for f in snap.fingerprints]
    assert [e["relative_path"] for e in entries] == ["SKILL.md", "a.md", "b.md"]
    assert all("executable" not in e for e in entries)


def test_symlink_skipped_not_followed(tmp_path):
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    (d / "link.md").symlink_to(d / "SKILL.md")
    snap = scan_skill_directory("demo", d)
    assert "link.md" not in {f.relative_path for f in snap.fingerprints}
    assert "link.md" in snap.unsupported_paths


def test_symlinked_directory_is_unsupported_not_dropped(tmp_path):
    # A symlinked *directory* must ALWAYS surface in unsupported_paths (the
    # trust gates rely on it): it must never silently vanish from both
    # fingerprints and unsupported_paths.
    d = tmp_path / "demo"
    _write(d / "SKILL.md", b"body")
    outside = tmp_path / "outside"
    _write(outside / "secret.md", b"secret")
    os.symlink(outside, d / "linked_dir")
    snap = scan_skill_directory("demo", d)
    assert "linked_dir" in snap.unsupported_paths
    fps = {f.relative_path for f in snap.fingerprints}
    assert "linked_dir" not in fps
    assert "linked_dir/secret.md" not in fps  # walk must not descend the symlink
