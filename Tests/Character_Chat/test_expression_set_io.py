import io
import zipfile

import pytest
from PIL import Image

from tldw_chatbook.Character_Chat.expression_set_io import (
    resolve_local_expression_set,
    MAX_ZIP_MEMBERS,
)


def _png(color=(10, 20, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, format="PNG")
    return buf.getvalue()


def _jpg() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, format="JPEG")
    return buf.getvalue()


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_zip_maps_by_filename_stem(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"idle.png": _png(), "speaking.jpg": _jpg()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle", "speaking"}
    assert res.images["idle"] == _png()  # bytes preserved verbatim


def test_case_insensitive_stem(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"IDLE.PNG": _png(), "Thinking.png": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle", "thinking"}


def test_non_matching_and_non_image_skipped(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"speaking.png": _png(), "notes.txt": b"hello", "random.png": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"speaking"}
    assert any(name == "notes.txt" for name, _ in res.skipped)  # not an image / no state


def test_bad_image_bytes_skipped_with_reason(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"error.png": b"not-an-image"}))
    res = resolve_local_expression_set([z])
    assert "error" not in res.images
    assert any(name == "error.png" for name, _ in res.skipped)


def test_two_files_one_state_prefers_png(tmp_path):
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"speaking.jpg": _jpg(), "speaking.png": _png()}))
    res = resolve_local_expression_set([z])
    assert res.images["speaking"] == _png()  # .png wins
    assert res.notes  # tie recorded


def test_directory_of_images(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "idle.png").write_bytes(_png())
    (d / "error.png").write_bytes(_png())
    res = resolve_local_expression_set([d])
    assert set(res.images) == {"idle", "error"}


def test_list_of_image_files(tmp_path):
    a = tmp_path / "thinking.png"; a.write_bytes(_png())
    b = tmp_path / "speaking.png"; b.write_bytes(_png())
    res = resolve_local_expression_set([a, b])
    assert set(res.images) == {"thinking", "speaking"}


def test_member_count_cap(tmp_path):
    z = tmp_path / "big.zip"
    z.write_bytes(_zip({f"file{i}.png": _png() for i in range(MAX_ZIP_MEMBERS + 5)}))
    res = resolve_local_expression_set([z])
    # capped: does not read all members; the 4 states may still resolve, but the
    # resolution notes/skips the cap. At minimum it must not raise and must be bounded.
    assert isinstance(res.images, dict)
    assert res.notes  # cap recorded


def test_total_size_cap_rejects(tmp_path, monkeypatch):
    # One member whose declared uncompressed size exceeds the total cap is skipped.
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", 100)
    z = tmp_path / "bomb.zip"
    z.write_bytes(_zip({"idle.png": _png() * 50}))  # > 100 bytes uncompressed
    res = resolve_local_expression_set([z])
    assert "idle" not in res.images
    assert res.notes or res.skipped


def test_directory_oversize_file_skipped_by_size_cap(tmp_path, monkeypatch):
    # Qodo fix 5: the directory branch must cap per-file size BEFORE
    # read_bytes(), same as the .zip branch already did.
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    monkeypatch.setattr(mod, "MAX_MEMBER_BYTES", 1000)
    d = tmp_path / "imgs"
    d.mkdir()
    (d / "idle.png").write_bytes(_png())          # small, valid -> resolves
    (d / "speaking.png").write_bytes(b"X" * 5000)  # over the (patched) cap
    res = resolve_local_expression_set([d])
    assert "idle" in res.images
    assert "speaking" not in res.images
    assert ("speaking.png", "file too large") in res.skipped


def test_standalone_file_oversize_skipped_by_size_cap(tmp_path, monkeypatch):
    # Same cap, exercised via the standalone-file branch of _candidate_pairs.
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    monkeypatch.setattr(mod, "MAX_MEMBER_BYTES", 1000)
    big = tmp_path / "error.png"
    big.write_bytes(b"X" * 5000)
    res = resolve_local_expression_set([big])
    assert "error" not in res.images
    assert ("error.png", "file too large") in res.skipped


def test_zip_member_backslash_path_normalized_to_basename(tmp_path):
    # Qodo fix 6: a zip member written with Windows separators
    # ("dir\\idle.png") must still map to the "idle" state.
    z = tmp_path / "set.zip"
    z.write_bytes(_zip({"dir\\idle.png": _png()}))
    res = resolve_local_expression_set([z])
    assert "idle" in res.images
    assert res.images["idle"] == _png()


def test_not_a_zip_fails_cleanly(tmp_path):
    bad = tmp_path / "broken.zip"
    bad.write_bytes(b"this is not a zip")
    res = resolve_local_expression_set([bad])  # must not raise
    assert res.images == {}
    assert res.skipped or res.notes


def test_build_zip_round_trips_through_resolver(tmp_path):
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    images = {"idle": _png(), "speaking": _jpg()}
    blob = build_expression_set_zip("Ada Lovelace", images)
    out = tmp_path / "ada.zip"
    out.write_bytes(blob)
    res = resolve_local_expression_set([out])
    assert set(res.images) == {"idle", "speaking"}


def test_build_zip_uses_detected_extension(tmp_path):
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    blob = build_expression_set_zip("Ada", {"speaking": _jpg()})
    names = zipfile.ZipFile(io.BytesIO(blob)).namelist()
    assert "speaking.jpg" in names          # JPEG bytes -> .jpg, not .png
    assert "expression_set.json" in names    # provenance marker present


def test_build_zip_empty_set_is_valid_zip():
    from tldw_chatbook.Character_Chat.expression_set_io import build_expression_set_zip
    blob = build_expression_set_zip("Ada", {})
    zf = zipfile.ZipFile(io.BytesIO(blob))
    assert zf.namelist() == ["expression_set.json"]


@pytest.fixture
def db(tmp_path):
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    return CharactersRAGDB(tmp_path / "expr.db", "test-client")   # file-backed, not :memory:


def test_apply_images_to_db_writes_only_non_idle(db):
    from tldw_chatbook.Character_Chat.expression_set_io import apply_expression_images_to_db
    cid = db.add_character_card({"name": "Ada"})
    applied, skipped = apply_expression_images_to_db(
        db, cid, {"idle": _png(), "speaking": _png(), "thinking": _png()}
    )
    assert set(applied) == {"speaking", "thinking"}     # idle NOT written to the table
    assert db.get_character_expression_image(cid, "speaking") is not None
    assert db.get_character_expression_image(cid, "idle") is None


def test_apply_images_to_db_best_effort(db, monkeypatch):
    from tldw_chatbook.Character_Chat.expression_set_io import apply_expression_images_to_db
    cid = db.add_character_card({"name": "Ada"})
    orig = db.set_character_expression_image
    def boom(c, s, i, m=None):
        if s == "error":
            raise RuntimeError("disk full")
        return orig(c, s, i, m)
    monkeypatch.setattr(db, "set_character_expression_image", boom)
    applied, skipped = apply_expression_images_to_db(db, cid, {"speaking": _png(), "error": _png()})
    assert applied == ["speaking"]
    assert any(s == "error" for s, _ in skipped)
