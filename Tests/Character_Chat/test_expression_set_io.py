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


# ---------- P3d-3: vpack builder + extractor ----------

def make_vpack_bytes(
    *,
    manifest: dict | None,
    assets_entries: list[dict],
    asset_files: dict[str, bytes],
    prefix: str = "",
    manifest_override: bytes | None = None,
    assets_json_override: bytes | None = None,
    extra_members: dict[str, bytes] | None = None,
    manifest_last: bool = False,
) -> bytes:
    """Craft a .tldw-persona-vpack archive in memory.

    Args:
        manifest: The pack manifest dict (``states``/``animations``/etc.);
            ignored when ``manifest_override`` is given.
        assets_entries: The ``assets`` list for ``metadata/assets.json``;
            ignored when ``assets_json_override`` is given.
        asset_files: Archive member path -> raw asset bytes, appended after
            the core members (and after ``extra_members``).
        prefix: Member-path prefix applied to every written member, to
            simulate a pack nested under a single shared top-level directory.
        manifest_override: Raw bytes to write for ``manifest.json`` instead
            of JSON-encoding ``manifest`` (e.g. to craft invalid JSON).
        assets_json_override: Raw bytes to write for ``metadata/assets.json``
            instead of JSON-encoding ``{"assets": assets_entries}``.
        extra_members: Additional archive member path -> bytes pairs, written
            FIRST (before the core members) unless ``manifest_last`` is True.
        manifest_last: When True, write ``extra_members`` before the core
            members instead of after -- used to test targeted (non-enumerate)
            member reads against a manifest buried past the member-count cap.

    Returns:
        The in-memory zip archive as bytes.
    """
    import json as _json
    members: list[tuple[str, bytes]] = []
    manifest_bytes = (
        manifest_override
        if manifest_override is not None
        else _json.dumps(manifest or {}).encode()
    )
    assets_bytes = (
        assets_json_override
        if assets_json_override is not None
        else _json.dumps({"assets": assets_entries}).encode()
    )
    core = [
        (f"{prefix}manifest.json", manifest_bytes),
        (f"{prefix}metadata/pack.json", b"{}"),
        (f"{prefix}metadata/assets.json", assets_bytes),
        (f"{prefix}checksums/sha256.json", b"{}"),
    ]
    for path, data in (extra_members or {}).items():
        members.append((f"{prefix}{path}", data))
    if manifest_last:
        members = members + core
    else:
        members = core + members
    for path, data in asset_files.items():
        members.append((f"{prefix}{path}", data))
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in members:
            zf.writestr(name, data)
    return buf.getvalue()


def simple_vpack(images: dict[str, bytes], prefix: str = "") -> bytes:
    """A standard pack: one standalone asset + one 1-frame animation per state.

    Args:
        images: {state: bytes} for each expression state to include; each
            gets its own standalone asset and single-frame animation.
        prefix: Member-path prefix applied to every written member, to
            simulate a pack nested under a single shared top-level directory.

    Returns:
        The in-memory .tldw-persona-vpack archive as bytes.
    """
    states, animations, entries, files = {}, {}, [], {}
    for state, data in images.items():
        aid = f"asset-{state}"
        states[state] = f"anim-{state}"
        animations[f"anim-{state}"] = {"frame_rate": 1, "frames": [{"asset_id": aid}]}
        path = f"assets/persona_visuals/{aid}.png"
        entries.append({"source_asset_id": aid, "asset_path": path,
                        "asset_bytes_status": "present"})
        files[path] = data
    return make_vpack_bytes(
        manifest={"manifest_version": 1, "renderer_type": "sprite_frames",
                  "states": states, "animations": animations},
        assets_entries=entries, asset_files=files, prefix=prefix,
    )


def _open_vpack(data: bytes) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BytesIO(data))


def _sheet_2x1(left=(255, 0, 0), right=(0, 255, 0)) -> bytes:
    """A 16x8 sheet: left 8x8 solid `left`, right 8x8 solid `right`."""
    img = Image.new("RGB", (16, 8), left)
    for x in range(8, 16):
        for y in range(8):
            img.putpixel((x, y), right)
    buf = io.BytesIO(); img.save(buf, format="PNG"); return buf.getvalue()


def _extract(data: bytes, prefix: str = ""):
    from tldw_chatbook.Character_Chat.expression_set_io import _resolve_vpack_expression_set
    with _open_vpack(data) as zf:
        res, _total = _resolve_vpack_expression_set(zf, prefix=prefix)
    return res


def test_vpack_simple_pack_extracts_all_four_states():
    data = simple_vpack({s: _png() for s in ("idle", "thinking", "speaking", "error")})
    res = _extract(data)
    assert set(res.images) == {"idle", "thinking", "speaking", "error"}
    for b in res.images.values():
        assert b == _png()   # whole-asset bytes pass through verbatim


def test_vpack_sprite_sheet_region_crop_pixel_checked():
    sheet = _sheet_2x1()
    manifest = {
        "states": {"speaking": "talk"},
        "animations": {"talk": {"frame_rate": 2, "frames": [
            {"asset_id": "sheet", "region": {"x": 8, "y": 0, "width": 8, "height": 8}},
        ]}},
    }
    data = make_vpack_bytes(
        manifest=manifest,
        assets_entries=[{"source_asset_id": "sheet",
                         "asset_path": "assets/persona_visuals/sheet.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/sheet.png": sheet},
    )
    res = _extract(data)
    out = Image.open(io.BytesIO(res.images["speaking"]))
    assert out.size == (8, 8)
    assert out.convert("RGB").getpixel((0, 0)) == (0, 255, 0)   # the RIGHT half


def test_vpack_preview_frame_honored_and_invalid_falls_back():
    frames = [
        {"asset_id": "a0"},
        {"asset_id": "a1"},
    ]
    def pack(preview):
        anim = {"frame_rate": 1, "frames": frames}
        if preview is not None:
            anim["preview_frame"] = preview
        return make_vpack_bytes(
            manifest={"states": {"idle": "a"}, "animations": {"a": anim}},
            assets_entries=[
                {"source_asset_id": "a0", "asset_path": "assets/persona_visuals/a0.png",
                 "asset_bytes_status": "present"},
                {"source_asset_id": "a1", "asset_path": "assets/persona_visuals/a1.png",
                 "asset_bytes_status": "present"},
            ],
            asset_files={
                "assets/persona_visuals/a0.png": _png((1, 1, 1)),
                "assets/persona_visuals/a1.png": _png((2, 2, 2)),
            },
        )
    assert _extract(pack(1)).images["idle"] == _png((2, 2, 2))       # honored
    assert _extract(pack(99)).images["idle"] == _png((1, 1, 1))      # out of range -> frames[0]
    assert _extract(pack("x")).images["idle"] == _png((1, 1, 1))     # non-int -> frames[0]
    assert _extract(pack(None)).images["idle"] == _png((1, 1, 1))    # absent -> frames[0]
    # Qodo fix: bool is a subclass of int -- True must NOT be treated as a
    # valid frame index (it would otherwise select frames[1]).
    assert _extract(pack(True)).images["idle"] == _png((1, 1, 1))    # bool -> frames[0]


def test_vpack_region_with_bool_value_skipped():
    # Qodo fix: bool is a subclass of int -- a region value of True/False
    # must NOT pass the int-type check (it would smuggle through as 1/0 and
    # yield a garbage crop instead of being rejected as an invalid region).
    data = make_vpack_bytes(
        manifest={"states": {"idle": "a"},
                  "animations": {"a": {"frames": [
                      {"asset_id": "x", "region": {"x": True, "y": 0, "width": 8, "height": 8}},
                  ]}}},
        assets_entries=[{"source_asset_id": "x",
                         "asset_path": "assets/persona_visuals/x.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/x.png": _sheet_2x1()},
    )
    res = _extract(data)
    assert "idle" not in res.images
    assert res.skipped


def test_vpack_asset_ids_shorthand():
    data = make_vpack_bytes(
        manifest={"states": {"thinking": "t"},
                  "animations": {"t": {"frame_rate": 1, "asset_ids": ["b"]}}},
        assets_entries=[{"source_asset_id": "b",
                         "asset_path": "assets/persona_visuals/b.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/b.png": _png()},
    )
    assert "thinking" in _extract(data).images


def test_vpack_per_state_skip_reasons():
    # idle: absent from states; speaking: unknown animation; thinking: asset
    # missing from archive; error: explicit asset_bytes_status missing.
    data = make_vpack_bytes(
        manifest={"states": {"speaking": "nope", "thinking": "t", "error": "e"},
                  "animations": {
                      "t": {"frames": [{"asset_id": "gone"}]},
                      "e": {"frames": [{"asset_id": "m"}]},
                  }},
        assets_entries=[
            {"source_asset_id": "gone", "asset_path": "assets/persona_visuals/gone.png",
             "asset_bytes_status": "present"},   # entry exists; member does NOT
            {"source_asset_id": "m", "asset_path": "assets/persona_visuals/m.png",
             "asset_bytes_status": "missing"},
        ],
        asset_files={},
    )
    res = _extract(data)
    assert res.images == {}
    reasons = dict((s, r) for s, r in res.skipped)
    assert "idle" in reasons and "speaking" in reasons
    assert "thinking" in reasons and "error" in reasons


def test_vpack_absent_status_key_tolerated():
    data = make_vpack_bytes(
        manifest={"states": {"idle": "a"},
                  "animations": {"a": {"frames": [{"asset_id": "x"}]}}},
        assets_entries=[{"source_asset_id": "x",
                         "asset_path": "assets/persona_visuals/x.png"}],  # no status key
        asset_files={"assets/persona_visuals/x.png": _png()},
    )
    assert "idle" in _extract(data).images


def test_vpack_traversal_asset_path_fails_lookup_safely():
    data = make_vpack_bytes(
        manifest={"states": {"idle": "a"},
                  "animations": {"a": {"frames": [{"asset_id": "x"}]}}},
        assets_entries=[{"source_asset_id": "x",
                         "asset_path": "../../etc/passwd",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/x.png": _png()},
    )
    res = _extract(data)   # member lookup fails -> skip; never touches the FS
    assert res.images == {}
    assert res.skipped


def test_vpack_invalid_region_skipped():
    data = make_vpack_bytes(
        manifest={"states": {"idle": "a"},
                  "animations": {"a": {"frames": [
                      {"asset_id": "x", "region": {"x": 0, "y": 0, "width": 999, "height": 8}},
                  ]}}},
        assets_entries=[{"source_asset_id": "x",
                         "asset_path": "assets/persona_visuals/x.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/x.png": _sheet_2x1()},
    )
    res = _extract(data)   # region exceeds the 16x8 image
    assert "idle" not in res.images
    assert res.skipped


def test_vpack_shared_sheet_read_once_within_budget(monkeypatch):
    """RED against a non-caching implementation: 4 states off ONE sheet must
    charge the budget once, not four times."""
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    sheet = _sheet_2x1()
    regions = {"idle": 0, "thinking": 0, "speaking": 8, "error": 8}
    manifest = {"states": {}, "animations": {}}
    for state, x in regions.items():
        manifest["states"][state] = f"anim-{state}"
        manifest["animations"][f"anim-{state}"] = {"frames": [
            {"asset_id": "sheet", "region": {"x": x, "y": 0, "width": 8, "height": 8}},
        ]}
    data = make_vpack_bytes(
        manifest=manifest,
        assets_entries=[{"source_asset_id": "sheet",
                         "asset_path": "assets/persona_visuals/sheet.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/sheet.png": sheet},
    )
    # Budget covers manifest + assets.json + exactly ONE sheet charge, plus a
    # small slack -- NOT four sheet charges. A non-caching implementation
    # would need 3 more sheet charges (each len(sheet) bytes) to satisfy all
    # four states, which is far more than the 16-byte slack allows.
    with _open_vpack(data) as zf:
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    cap = (sizes["manifest.json"] + sizes["metadata/assets.json"]
           + sizes["assets/persona_visuals/sheet.png"] + 16)
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", cap)
    res = _extract(data)
    assert set(res.images) == {"idle", "thinking", "speaking", "error"}


def test_vpack_corrupt_manifest_member_never_raises(tmp_path):
    # A vpack whose manifest.json member has a corrupted compressed payload
    # (bad CRC) must be handled by the guarded zf.read() in
    # _read_member_capped -- not raise BadZipFile straight through
    # _resolve_vpack_expression_set (the manifest/assets.json reads happen
    # BEFORE the per-state try/except loop, so nothing else guards this).
    data = simple_vpack({s: _png() for s in ("idle", "thinking", "speaking", "error")})
    with _open_vpack(data) as zf:
        info = zf.getinfo("manifest.json")
        header_offset = info.header_offset

    raw = bytearray(data)
    # Local file header: 30 fixed bytes, then filename, then extra field.
    name_len = int.from_bytes(raw[header_offset + 26:header_offset + 28], "little")
    extra_len = int.from_bytes(raw[header_offset + 28:header_offset + 30], "little")
    data_start = header_offset + 30 + name_len + extra_len
    # Flip a couple of bytes inside the compressed payload -- sizes/CRC in
    # the header are left untouched, so this breaks decompression/CRC
    # checking without breaking the zip's central directory.
    raw[data_start] ^= 0xFF
    raw[data_start + 1] ^= 0xFF
    corrupted = bytes(raw)

    bad = tmp_path / "corrupt.vpack"
    bad.write_bytes(corrupted)

    with _open_vpack(bad.read_bytes()) as zf:
        from tldw_chatbook.Character_Chat.expression_set_io import _resolve_vpack_expression_set
        res, _total = _resolve_vpack_expression_set(zf)   # must not raise
    assert res.images == {}


def test_vpack_broken_manifest_never_raises():
    data = make_vpack_bytes(manifest=None, manifest_override=b"{not json",
                            assets_entries=[], asset_files={})
    res = _extract(data)   # must not raise
    assert res.images == {}
    assert res.notes or res.skipped


# ---------- P3d-3 Task 2: content-based detection + dispatch ----------


def test_dispatch_vpack_renamed_zip_autodetects(tmp_path):
    z = tmp_path / "pack.zip"   # wrong extension on purpose
    z.write_bytes(simple_vpack({"idle": _png(), "speaking": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle", "speaking"}


def test_dispatch_plain_zip_named_vpack_falls_back_to_stems(tmp_path):
    z = tmp_path / "set.tldw-persona-vpack"   # plain stem zip, wrong extension
    z.write_bytes(_zip({"idle.png": _png()}))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle"}   # stem mapping still works


def test_dispatch_nested_single_root_extracts(tmp_path):
    z = tmp_path / "pack.tldw-persona-vpack"
    z.write_bytes(simple_vpack({"thinking": _png()}, prefix="MyPack/"))
    res = resolve_local_expression_set([z])
    assert "thinking" in res.images


def test_dispatch_two_roots_falls_through(tmp_path):
    # Two top-level roots -> not detected as a pack -> stem mapping (no matches).
    buf = io.BytesIO(simple_vpack({"idle": _png()}, prefix="A/"))
    with zipfile.ZipFile(buf, "a") as zf:
        zf.writestr("B/stray.txt", b"x")   # a second root breaks single-prefix detection
    z = tmp_path / "two-roots.tldw-persona-vpack"
    z.write_bytes(buf.getvalue())
    res = resolve_local_expression_set([z])
    assert res.images == {}   # fell through to stem mapping, nothing matched


def test_dispatch_zip_with_manifest_but_no_pack_json_falls_back_to_stems(tmp_path):
    # Qodo fix: a zip carrying manifest.json + metadata/assets.json but NOT
    # metadata/pack.json is not a real vpack (the server's REQUIRED_MEMBERS
    # always include pack.json) -- it must fall through to stem mapping
    # instead of being routed exclusively to the vpack extractor (which
    # would silently yield nothing here, even though idle.png is present).
    z = tmp_path / "not-a-vpack.zip"
    z.write_bytes(_zip({
        "manifest.json": b"{}",
        "metadata/assets.json": b"{}",
        "idle.png": _png(),
    }))
    res = resolve_local_expression_set([z])
    assert set(res.images) == {"idle"}


def test_dispatch_manifest_beyond_member_64(tmp_path):
    """RED against enumerate-and-slice dispatch: manifest.json written LAST,
    after 70 filler members, must still be found (targeted reads only)."""
    filler = {f"assets/persona_visuals/filler{i}.bin": b"x" for i in range(70)}
    data = make_vpack_bytes(
        manifest={"states": {"idle": "a"},
                  "animations": {"a": {"frames": [{"asset_id": "x"}]}}},
        assets_entries=[{"source_asset_id": "x",
                         "asset_path": "assets/persona_visuals/x.png",
                         "asset_bytes_status": "present"}],
        asset_files={"assets/persona_visuals/x.png": _png()},
        extra_members=filler,
        manifest_last=True,
    )
    z = tmp_path / "big.tldw-persona-vpack"
    z.write_bytes(data)
    res = resolve_local_expression_set([z])
    assert "idle" in res.images


def test_dispatch_mixed_inputs_share_budget(tmp_path, monkeypatch):
    import tldw_chatbook.Character_Chat.expression_set_io as mod
    vp = simple_vpack({"idle": _png()})
    z = tmp_path / "pack.tldw-persona-vpack"
    z.write_bytes(vp)
    loose = tmp_path / "speaking.png"
    loose.write_bytes(_png())
    # Budget covers exactly the vpack's targeted reads (manifest.json +
    # metadata/assets.json + the one asset it extracts) plus a small slack
    # -- NOT the loose file on top. A generous margin here (e.g. the asset
    # size plus a few KB of slack) leaves so much headroom that the loose
    # file fits regardless of whether the budget is actually shared, so the
    # cap is derived from MEASURED member sizes to keep the test load-bearing.
    with zipfile.ZipFile(io.BytesIO(vp)) as zf:
        sizes = {i.filename: i.file_size for i in zf.infolist()}
    cap = (sizes["manifest.json"] + sizes["metadata/assets.json"]
           + sizes["assets/persona_visuals/asset-idle.png"] + 16)
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", cap)
    res = resolve_local_expression_set([z, loose])
    assert "idle" in res.images          # vpack consumed the budget
    assert "speaking" not in res.images  # loose file hit the shared cap
    assert res.notes or res.skipped
