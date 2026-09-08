"""Pinned local Petdex metadata and art admission."""

import io
import json
import zipfile

import pytest
from PIL import Image

from tldw_chatbook.Petdex.sources import read_local_package, source_from_bytes


def sheet(version=1):
    output = io.BytesIO()
    with Image.new("RGBA", (96, 13 * (9 if version == 1 else 11)), "cyan") as image:
        image.save(output, format="PNG")
    return output.getvalue()


def package(root, **metadata):
    root.mkdir(exist_ok=True)
    (root / "pet.json").write_text(
        json.dumps({"name": "Demo", "spriteVersionNumber": 1, **metadata})
    )
    (root / "spritesheet.png").write_bytes(sheet())
    return root


def test_folder_source_preserves_terms_is_immutable_and_catches_replacement(tmp_path):
    root = package(tmp_path / "pet", creator="Actual Artist", license="MIT")
    (root / "LICENSE.txt").write_text("Exact artwork notice.\nSecond line.")
    result = read_local_package(root / "pet.json")
    assert result.title == "Demo"
    assert result.artwork["creator"] == "Actual Artist"
    assert "Exact artwork notice.\nSecond line." in result.artwork["notices"]
    assert result.is_current()
    with pytest.raises(TypeError):
        result.artwork["creator"] = "Other"
    (root / "spritesheet.png").write_bytes(b"changed")
    assert not result.is_current()


def test_zip_wrapper_preserves_notice_but_never_imports_commands(tmp_path):
    path = tmp_path / "pet.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "demo/pet.json", json.dumps({"name": "Demo", "spriteVersionNumber": 1})
        )
        archive.writestr("demo/spritesheet.png", sheet())
        archive.writestr("demo/NOTICE", "Original notice")
        archive.writestr("demo/install.sh", "untrusted text")
    source = read_local_package(path)
    assert source.artwork["license"] is None
    assert source.artwork["notices"] == "demo/NOTICE\nOriginal notice"
    assert "install.sh" not in source.metadata_json
    path.unlink()
    assert not source.is_current()


@pytest.mark.parametrize(
    "member", ["../escape", "/absolute", "demo/../pet.json", "demo\\pet.json"]
)
def test_zip_rejects_unsafe_members_before_admission(tmp_path, member):
    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pet.json", '{"name":"Demo"}')
        archive.writestr("spritesheet.png", sheet())
        archive.writestr(member, b"x")
    with pytest.raises(ValueError):
        read_local_package(path)


def test_ambiguous_sprite_and_symlink_are_rejected(tmp_path):
    root = package(tmp_path / "pet")
    (root / "spritesheet.webp").write_bytes(b"bad")
    with pytest.raises(ValueError):
        read_local_package(root)
    (root / "spritesheet.webp").unlink()
    (root / "spritesheet.png").unlink()
    (root / "spritesheet.png").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        read_local_package(root)


def test_declared_image_cannot_escape_or_silently_conflict(tmp_path):
    root = package(tmp_path / "pet", spritesheet="../escape.png")
    with pytest.raises(ValueError):
        read_local_package(root)
    with pytest.raises(ValueError):
        source_from_bytes(b'{"name":"one","name":"two"}', sheet(), "spritesheet.png")
    with pytest.raises(ValueError):
        source_from_bytes(
            b'{"name":"Demo","spriteVersionNumber":3}', sheet(), "spritesheet.png"
        )


def test_registry_cannot_override_conflicting_version_or_creator():
    with pytest.raises(ValueError):
        source_from_bytes(
            b'{"name":"Demo","spriteVersionNumber":2}',
            sheet(),
            "spritesheet.png",
            registry_entry={"spriteVersionNumber": 1},
        )
    source = source_from_bytes(
        b'{"name":"Demo","creator":"Original"}',
        sheet(),
        "spritesheet.png",
        registry_entry={
            "submittedBy": "Uploader",
            "sourceUrl": "https://petdex.dev/pets/demo",
            "spriteVersionNumber": 1,
        },
    )
    assert source.artwork["creator"] == "Original"
    assert source.artwork["source_url"] == "https://petdex.dev/pets/demo"


def test_upstream_spritesheet_path_declaration_is_enforced(tmp_path):
    root = package(tmp_path / "pet", spritesheetPath="../escape.png")
    with pytest.raises(ValueError):
        read_local_package(root)
    with pytest.raises(ValueError):
        source_from_bytes(
            b'{"name":"Demo","spritesheetPath":"other.png"}', sheet(), "spritesheet.png"
        )
    source = source_from_bytes(
        b'{"name":"Demo","spritesheetPath":"spritesheet.png"}',
        sheet(),
        "spritesheet.png",
    )
    assert source.image_name == "spritesheet.png"


def test_registry_terms_are_carried_without_inventing_license():
    source = source_from_bytes(
        b'{"name":"Demo"}',
        sheet(),
        "spritesheet.png",
        registry_entry={"terms": "Retain this notice."},
    )
    assert source.artwork["license"] is None
    assert source.artwork["notices"] == "Retain this notice."


def test_zip_rejects_names_truncated_by_zipfile(tmp_path):
    path = tmp_path / "nul.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("pet.jsonXevil", '{"name":"Demo"}')
        archive.writestr("spritesheet.png", sheet())
    path.write_bytes(path.read_bytes().replace(b"pet.jsonXevil", b"pet.json\x00evil"))
    with pytest.raises(ValueError):
        read_local_package(path)


def test_nested_folder_notices_and_directory_changes_are_pinned(tmp_path):
    root = package(tmp_path / "pet", spritesheetPath="art/spritesheet.png")
    (root / "art").mkdir()
    (root / "spritesheet.png").rename(root / "art/spritesheet.png")
    (root / "art/LICENSE").write_text("Original nested terms")
    source = read_local_package(root)
    assert "art/LICENSE\nOriginal nested terms" in source.artwork["notices"]
    (root / "art/NOTICE").write_text("Additional terms")
    assert not source.is_current()


def test_supplementary_credits_are_retained_and_validated():
    metadata = b'{"name":"Demo","creator":"Artist","license":"Original terms","source_url":"https://example.org/original"}'
    registry = {
        "creator": "Registry artist",
        "submittedBy": "Uploader",
        "license": "Registry terms",
        "sourceUrl": "https://petdex.dev/pets/demo",
    }
    source = source_from_bytes(
        metadata, sheet(), "spritesheet.png", registry_entry=registry
    )
    assert source.artwork["creator"] == "Artist"
    assert source.artwork["license"] == "Original terms"
    for value in (
        "Registry artist",
        "Uploader",
        "Registry terms",
        "https://example.org/original",
    ):
        assert value in source.artwork["notices"]
    with pytest.raises(ValueError):
        source_from_bytes(
            metadata,
            sheet(),
            "spritesheet.png",
            registry_entry={**registry, "license": {"invalid": True}},
        )


@pytest.mark.parametrize("damage", ["duplicate", "symlink", "expansion"])
def test_zip_rejects_duplicates_special_files_and_expansion(
    tmp_path, monkeypatch, damage
):
    import stat

    from tldw_chatbook.Petdex import sources

    path = tmp_path / "bad.zip"
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("pet.json", '{"name":"Demo"}')
        archive.writestr("spritesheet.png", sheet())
        if damage == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate"):
                archive.writestr("pet.json", '{"name":"Other"}')
        elif damage == "symlink":
            info = zipfile.ZipInfo("LICENSE")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive.writestr(info, "../outside")
        else:
            archive.writestr("unused.txt", b"x" * 8192)
            monkeypatch.setattr(sources, "MAX_PACKAGE_BYTES", 4096)
    with pytest.raises(ValueError):
        read_local_package(path)


def test_folder_inventory_is_bounded_and_rejects_nested_links(tmp_path, monkeypatch):
    from tldw_chatbook.Petdex import sources

    root = package(tmp_path / "pet")
    (root / "art").mkdir()
    (root / "art/LICENSE").symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError):
        read_local_package(root)
    (root / "art/LICENSE").unlink()
    (root / "art/LICENSE").write_text("Terms")
    monkeypatch.setattr(sources, "MAX_MEMBERS", 3)
    with pytest.raises(ValueError):
        read_local_package(root)
