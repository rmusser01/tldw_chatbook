# Roleplay P3d-3 — `.tldw-persona-vpack` Extraction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import a `.tldw-persona-vpack` archive through the existing "Import set…" button, extracting one static image per state (idle/thinking/speaking/error) into the same `_apply_expression_set` flow.

**Architecture:** A new pure extractor in `Character_Chat/expression_set_io.py` resolves each state via `manifest.states → animations → frame → assets-index → capped cached member read → optional region crop`. `resolve_local_expression_set` gains content-based detection (namelist sniff, root or single-shared-prefix) and dispatches vpacks to the extractor with **targeted reads only** (never the generic 64-member enumeration), sharing one size budget with the generic path via a `start_total` thread through `_candidate_pairs`. The UI change is one picker-filter line.

**Tech Stack:** Python ≥3.11, `zipfile`, `json`, PIL, Textual (UI filter line only).

## Global Constraints

- **NO migration** (dev schema is v24 via unrelated Console-branching work; P3d-3 touches no schema).
- ALL new logic in the PURE `tldw_chatbook/Character_Chat/expression_set_io.py` (zipfile + PIL + json only; NO Textual, NO DB-module imports).
- The vpack path **NEVER enumerate-reads**: targeted members only (`manifest.json`, `metadata/assets.json`, ≤4 resolved assets), each size-capped vs `ZipInfo.file_size` BEFORE read, one shared `MAX_TOTAL_BYTES` budget per resolver call (threaded via `_candidate_pairs(start_total=...)` — single caller, verified safe), and a per-call cache of distinct member bytes + opened PIL images (one read/decode per distinct sprite-sheet, cropped up to 4×).
- Frame choice: `preview_frame` if a valid in-range int, else `frames[0]`. Region crops re-encode as **PNG**; whole assets pass through **verbatim**; EVERYTHING passes `_valid_image` before entering `images` (satisfies the `_apply_expression_set` callers-pass-validated-bytes contract by construction).
- Lenient best-effort per state (every skip carries a reason); **checksums NOT verified**; `fallbacks`/`frame_rate`/`duration_ms`/`alignment` inert; the extractor **NEVER raises**.
- Detection by CONTENT: `manifest.json` + `metadata/assets.json` in the namelist, at the root OR under a single shared top-level directory (the re-zip papercut — strip the prefix for all lookups). A vpack renamed `.zip` auto-detects; a plain zip named `.vpack` falls back to stem mapping; deeper nesting / multiple roots fall through.
- Safe-by-construction (preserve as code comments): `asset_path` is only ever a **zip-member key** (traversal-shaped values fail the lookup → skip; nothing touches the filesystem); region-crop decodes an untrusted image → PIL's default `MAX_IMAGE_PIXELS` bomb guard + per-state try/except→skip; manifest/assets.json parsed with plain `json.loads` UNDER the member-size cap (parse failures incl. RecursionError caught by the never-raise wrapper).
- UI change is ONLY the picker filter — `validate_path_simple` boundary, `_io_dialog_active` gate, orchestrator, summary all untouched.
- Tests: crafted **in-test** vpack builder (no on-disk fixtures); pixel-checked crops; the RED-provable pins (shared-sheet budget; manifest-written-LAST beyond member #64); nested-root; absent-status-key; mixed-input budget; broken-JSON never-raise; P3d-2 stem-mapping regression. DB tests file-backed `CharactersRAGDB(tmp_path/...)` NOT `:memory:`; Tests/UI asyncio rules (don't mix dirs OR explicit `@pytest.mark.asyncio`).
- CONCURRENT-SESSION HAZARD: `personas_screen.py` heavily edited elsewhere — localize; expect a rebase. Implementers PREPEND `cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && ` to EVERY Bash call; stage ONLY task files (never `git add -A`, never `.superpowers/`); NO background/broad sweeps; NEVER pkill.
- **Test env prefix:** `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest ... -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`

---

## File Structure

- `tldw_chatbook/Character_Chat/expression_set_io.py` — **modify**: `_read_member_capped` + `_read_vpack_json` + `_resolve_vpack_expression_set` (Task 1); `_detect_vpack` + dispatch in `resolve_local_expression_set` + `_candidate_pairs(start_total)` (Task 2).
- `tldw_chatbook/UI/Screens/personas_screen.py` — **modify**: the import picker filter (Task 3).
- `Tests/Character_Chat/test_expression_set_io.py` — **extend**: vpack builder + extractor/dispatch tests (Tasks 1-2).
- `Tests/UI/test_personas_expression_slots.py` — **extend**: vpack end-to-end (Task 3).

---

## Task 1: Pure `_resolve_vpack_expression_set` extractor + in-test vpack builder

**Files:**
- Modify: `tldw_chatbook/Character_Chat/expression_set_io.py`
- Test: `Tests/Character_Chat/test_expression_set_io.py` (extend)

**Interfaces:**
- Consumes: `MAX_MEMBER_BYTES`/`MAX_TOTAL_BYTES`, `_valid_image`, `ExpressionSetResolution`, `EXPRESSION_STATES` (all existing).
- Produces (for Task 2):
  - `_resolve_vpack_expression_set(zf: zipfile.ZipFile, *, prefix: str = "", start_total: int = 0) -> tuple[ExpressionSetResolution, int]` — (resolution, new running total).
  - Test module exports (importable by name, no `test_` prefix): `make_vpack_bytes(...)` + `simple_vpack(images)`.

- [ ] **Step 1: Write the vpack builder helpers + failing tests**

Append to `Tests/Character_Chat/test_expression_set_io.py` (reuses the existing `_png`/`_zip` helpers):

```python
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

    asset_files maps archive member path -> raw bytes. extra_members are
    written FIRST (before manifest) when manifest_last is True.
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
    """A standard pack: one standalone asset + one 1-frame animation per state."""
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
    # Budget covers manifest + assets.json + ONE sheet read -- not four.
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", len(sheet) + 4096)
    res = _extract(data)
    assert set(res.images) == {"idle", "thinking", "speaking", "error"}


def test_vpack_broken_manifest_never_raises():
    data = make_vpack_bytes(manifest=None, manifest_override=b"{not json",
                            assets_entries=[], asset_files={})
    res = _extract(data)   # must not raise
    assert res.images == {}
    assert res.notes or res.skipped
```

- [ ] **Step 2: Run to verify RED**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_expression_set_io.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Expected: the 10 new tests FAIL with `ImportError: cannot import name '_resolve_vpack_expression_set'`; the 18 existing tests still pass.

- [ ] **Step 3: Implement the extractor**

Append to `tldw_chatbook/Character_Chat/expression_set_io.py`:

```python
# ---------- P3d-3: .tldw-persona-vpack extraction ----------


def _read_member_capped(
    zf: zipfile.ZipFile,
    member: str,
    bytes_cache: dict[str, bytes],
    total: int,
    skipped: list[tuple[str, str]],
) -> tuple[bytes | None, int]:
    """Read one zip member through the size caps, with a per-call cache.

    ``member`` is only ever used as a zip-member KEY -- a traversal-shaped
    value simply fails the ``getinfo`` lookup and is skipped; nothing here
    touches the filesystem. A cache hit does not re-charge the budget (the
    common pack shape is ONE sprite sheet referenced by every state).
    """
    if member in bytes_cache:
        return bytes_cache[member], total
    try:
        info = zf.getinfo(member)
    except KeyError:
        skipped.append((member, "not found in archive"))
        return None, total
    if info.file_size > MAX_MEMBER_BYTES:
        skipped.append((member, "file too large"))
        return None, total
    if total + info.file_size > MAX_TOTAL_BYTES:
        skipped.append((member, "total size cap exceeded"))
        return None, total
    data = zf.read(info)
    bytes_cache[member] = data
    return data, total + info.file_size


def _read_vpack_json(
    zf: zipfile.ZipFile,
    member: str,
    bytes_cache: dict[str, bytes],
    total: int,
    skipped: list[tuple[str, str]],
):
    """Parse a JSON member under the size caps. Returns (obj|None, total).
    Parse failures (bad JSON, RecursionError on pathological nesting) are
    caught -- the vpack path never raises."""
    data, total = _read_member_capped(zf, member, bytes_cache, total, skipped)
    if data is None:
        return None, total
    try:
        obj = json.loads(data)
    except Exception:
        skipped.append((member, "invalid JSON"))
        return None, total
    return obj if isinstance(obj, dict) else None, total


def _resolve_vpack_expression_set(
    zf: zipfile.ZipFile, *, prefix: str = "", start_total: int = 0
) -> tuple[ExpressionSetResolution, int]:
    """Extract one static image per expression state from a persona visual pack.

    Per state: ``manifest.states[state]`` -> ``animations[id]`` (honoring the
    ``asset_ids`` shorthand) -> ``frames[preview_frame]`` if that is a valid
    in-range int else ``frames[0]`` -> assets-index (``source_asset_id`` ->
    ``asset_path``; skip only on explicit ``asset_bytes_status == "missing"``)
    -> capped cached member read -> optional ``region`` crop (bounds-checked,
    re-encoded as PNG) -> ``_valid_image`` -> ``images[state]``.

    Lenient/best-effort: every failure skips that state with a reason; a
    broken manifest degrades to an empty resolution. NEVER raises.
    ``frame_rate``/``duration_ms``/``fallbacks``/``alignment``/checksums are
    inert (all four of our states are in the server's REQUIRED_VISUAL_STATES,
    so any server-valid pack defines them directly).

    Args:
        zf: The open archive.
        prefix: Member-path prefix ("" for a root-level pack; "Pack/" when
            the pack sits under a single shared top-level directory).
        start_total: The resolver call's running size total so far.

    Returns:
        (resolution, new_running_total).
    """
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = start_total
    bytes_cache: dict[str, bytes] = {}
    image_cache: dict[str, Image.Image] = {}

    manifest, total = _read_vpack_json(zf, f"{prefix}manifest.json", bytes_cache, total, skipped)
    assets_doc, total = _read_vpack_json(zf, f"{prefix}metadata/assets.json", bytes_cache, total, skipped)
    if manifest is None or assets_doc is None:
        notes.append("Archive looks like a persona visual pack, but its manifest could not be read.")
        return ExpressionSetResolution(images={}, skipped=skipped, notes=notes), total

    states = manifest.get("states")
    animations = manifest.get("animations")
    states = states if isinstance(states, dict) else {}
    animations = animations if isinstance(animations, dict) else {}
    assets_index: dict[str, dict] = {}
    entries = assets_doc.get("assets")
    for entry in entries if isinstance(entries, list) else []:
        if isinstance(entry, dict) and entry.get("source_asset_id"):
            assets_index[str(entry["source_asset_id"])] = entry

    images: dict[str, bytes] = {}
    for state in EXPRESSION_STATES:
        try:
            animation_id = states.get(state)
            if not animation_id:
                skipped.append((state, "state not in pack"))
                continue
            animation = animations.get(animation_id) if isinstance(animation_id, str) else None
            if not isinstance(animation, dict):
                skipped.append((state, "unknown animation"))
                continue
            frames = animation.get("frames")
            if not (isinstance(frames, list) and frames):
                asset_ids = animation.get("asset_ids")
                frames = (
                    [{"asset_id": a} for a in asset_ids]
                    if isinstance(asset_ids, list) and asset_ids
                    else None
                )
            if not frames:
                skipped.append((state, "animation has no frames"))
                continue
            idx = animation.get("preview_frame")
            frame = frames[idx] if isinstance(idx, int) and 0 <= idx < len(frames) else frames[0]
            if not isinstance(frame, dict):
                skipped.append((state, "invalid frame"))
                continue
            asset_id = frame.get("asset_id")
            entry = assets_index.get(str(asset_id)) if asset_id else None
            if entry is None:
                skipped.append((state, "unknown asset"))
                continue
            # Skip only an EXPLICIT "missing"; an absent status key is
            # tolerated (mirrors the server's own defensive .get reads).
            if entry.get("asset_bytes_status") == "missing":
                skipped.append((state, "asset bytes missing from archive"))
                continue
            asset_path = str(entry.get("asset_path") or "")
            if not asset_path:
                skipped.append((state, "asset has no archive path"))
                continue
            member = f"{prefix}{asset_path}"
            data, total = _read_member_capped(zf, member, bytes_cache, total, skipped)
            if data is None:
                skipped.append((state, "asset could not be read"))
                continue
            region = frame.get("region")
            if region is not None:
                img = image_cache.get(member)
                if img is None:
                    # Decoding an untrusted image: PIL's default
                    # MAX_IMAGE_PIXELS decompression-bomb guard applies, and
                    # any decode failure lands in this state's try/except.
                    img = Image.open(io.BytesIO(data))
                    img.load()
                    image_cache[member] = img
                if not (
                    isinstance(region, dict)
                    and all(isinstance(region.get(k), int) for k in ("x", "y", "width", "height"))
                    and region["x"] >= 0 and region["y"] >= 0
                    and region["width"] > 0 and region["height"] > 0
                    and region["x"] + region["width"] <= img.width
                    and region["y"] + region["height"] <= img.height
                ):
                    skipped.append((state, "invalid region"))
                    continue
                crop = img.crop((
                    region["x"], region["y"],
                    region["x"] + region["width"], region["y"] + region["height"],
                ))
                buf = io.BytesIO()
                crop.save(buf, format="PNG")   # a crop must be standalone bytes
                out = buf.getvalue()
            else:
                out = data   # whole-asset bytes pass through verbatim
            if not _valid_image(out):
                skipped.append((state, "not a valid image"))
                continue
            images[state] = out
        except Exception as exc:
            skipped.append((state, f"extraction failed: {exc}"))
    return ExpressionSetResolution(images=images, skipped=skipped, notes=notes), total
```

- [ ] **Step 4: Run to verify GREEN**

Same command as Step 2. Expected: 28 passed (18 existing + 10 new). Then `... python -c "import tldw_chatbook.app"`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/expression_set_io.py Tests/Character_Chat/test_expression_set_io.py
git commit -m "feat(personas): P3d-3 Task 1 — pure vpack extractor (capped cached reads, region crops)"
```

---

## Task 2: Detection + dispatch + budget threading

**Files:**
- Modify: `tldw_chatbook/Character_Chat/expression_set_io.py` (`_detect_vpack`; `_candidate_pairs` signature; `resolve_local_expression_set` dispatch)
- Test: `Tests/Character_Chat/test_expression_set_io.py` (extend)

**Interfaces:**
- Consumes: `_resolve_vpack_expression_set` (Task 1), `simple_vpack`/`make_vpack_bytes` (Task 1 test helpers).
- Produces: `_detect_vpack(zf: zipfile.ZipFile) -> str | None` (the member prefix, or None if not a vpack); `_candidate_pairs(paths, start_total: int = 0) -> tuple[pairs, skipped, notes, int]` (now returns the consumed total).

- [ ] **Step 1: Write the failing tests**

Append to `Tests/Character_Chat/test_expression_set_io.py`:

```python
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
    # Budget covers the vpack's reads but NOT the loose file afterwards.
    idle_png = _png()
    monkeypatch.setattr(mod, "MAX_TOTAL_BYTES", len(idle_png) + 4096)
    res = resolve_local_expression_set([z, loose])
    assert "idle" in res.images          # vpack consumed the budget
    assert "speaking" not in res.images  # loose file hit the shared cap
    assert res.notes or res.skipped
```

- [ ] **Step 2: Run to verify RED**

Same test command. Expected: the 6 new tests fail (`_detect_vpack` undefined via the dispatch not existing — the renamed-zip test resolves 0 states, etc.); all prior tests pass.

- [ ] **Step 3: Implement detection + dispatch**

In `expression_set_io.py`:

1. Add `_detect_vpack`:
```python
def _detect_vpack(zf: zipfile.ZipFile) -> str | None:
    """Detect a persona visual pack by CONTENT (never by file extension).

    Returns the member prefix to use for lookups: "" for a root-level pack,
    "Root/" when the pack sits under a single shared top-level directory
    (the extract-and-re-zip papercut), or None when the archive is not a
    pack (deeper nesting and multiple roots fall through to stem mapping).
    The sniff reads the namelist only -- zero member reads.
    """
    names = set(zf.namelist())
    if "manifest.json" in names and "metadata/assets.json" in names:
        return ""
    roots = {n.split("/", 1)[0] for n in names if n and not n.startswith("/")}
    if len(roots) == 1:
        root = next(iter(roots))
        if f"{root}/manifest.json" in names and f"{root}/metadata/assets.json" in names:
            return f"{root}/"
    return None
```

2. Change `_candidate_pairs(paths)` → `_candidate_pairs(paths, start_total: int = 0)`: initialize `total = start_total`, and change the return to the 4-tuple `(pairs, skipped, notes, total)` (update the docstring's budget sentence to mention the threaded start value).

3. Rewrite `resolve_local_expression_set`'s body to dispatch (keep the docstring, adding one line about vpack detection):
```python
    images: dict[str, bytes] = {}
    skipped: list[tuple[str, str]] = []
    notes: list[str] = []
    total = 0
    generic: list[Path] = []
    for path in paths:
        handled = False
        try:
            if path.is_file() and zipfile.is_zipfile(path):
                with zipfile.ZipFile(path) as zf:
                    vprefix = _detect_vpack(zf)
                    if vprefix is not None:
                        res, total = _resolve_vpack_expression_set(
                            zf, prefix=vprefix, start_total=total
                        )
                        for state, data in res.images.items():
                            images.setdefault(state, data)   # first-writer-wins
                        skipped.extend(res.skipped)
                        notes.extend(res.notes)
                        handled = True
        except Exception as exc:
            skipped.append((str(path), f"could not read: {exc}"))
            handled = True
        if not handled:
            generic.append(path)
    if generic:
        pairs, g_skipped, g_notes, total = _candidate_pairs(generic, start_total=total)
        skipped.extend(g_skipped)
        notes.extend(g_notes)
        chosen: dict[str, tuple[str, bytes]] = {}
        for name, data in pairs:
            state = Path(name).stem.lower()
            if state not in _STATE_SET:
                skipped.append((name, "filename is not a known state"))
                continue
            if not _valid_image(data):
                skipped.append((name, "not a valid image"))
                continue
            if state in chosen:
                keep = chosen[state][0]
                if _prefer(keep, name):
                    notes.append(f"Multiple files for {state}; used {name}.")
                    chosen[state] = (name, data)
                else:
                    notes.append(f"Multiple files for {state}; used {keep}.")
                continue
            chosen[state] = (name, data)
        for state, (_, data) in chosen.items():
            images.setdefault(state, data)   # vpack states win over generic
    return ExpressionSetResolution(images=images, skipped=skipped, notes=notes)
```
(The stem-mapping loop is the EXISTING loop moved verbatim under the `generic` batch — do not change its tie logic.)

- [ ] **Step 4: Run to verify GREEN**

Same command. Expected: 34 passed (all existing P3d-2 tests — the stem/tie/caps behavior is unchanged — plus the 16 P3d-3 tests). Then `... python -c "import tldw_chatbook.app"`.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/Character_Chat/expression_set_io.py Tests/Character_Chat/test_expression_set_io.py
git commit -m "feat(personas): P3d-3 Task 2 — content-based vpack detection + targeted dispatch + shared budget"
```

---

## Task 3: UI filter + end-to-end

**Files:**
- Modify: `tldw_chatbook/UI/Screens/personas_screen.py` (READ FRESH — the `_expression_set_import_dialog_worker` picker filter)
- Test: `Tests/UI/test_personas_expression_slots.py` (extend)

**Interfaces:**
- Consumes: everything from Tasks 1-2 (via `resolve_local_expression_set`); the Task-1 test helper `simple_vpack` (importable: `from Tests.Character_Chat.test_expression_set_io import simple_vpack` — cross-test-module imports are established practice, e.g. `test_console_character_avatar.py` imports from `test_destination_shells`).

- [ ] **Step 1: Write the failing test**

Append to `Tests/UI/test_personas_expression_slots.py` (mirror `test_import_expression_set_from_zip_path`):

```python
@pytest.mark.asyncio
async def test_import_vpack_from_path(personas_editor_with_saved_character, tmp_path):
    app, screen, db, char_id = personas_editor_with_saved_character
    from Tests.Character_Chat.test_expression_set_io import simple_vpack, _png
    z = tmp_path / "pack.tldw-persona-vpack"
    z.write_bytes(simple_vpack({"idle": _png(), "speaking": _png(), "thinking": _png()}))

    await screen._import_expression_set_from_path(char_id, str(z))

    assert db.get_character_expression_image(char_id, "speaking") is not None
    assert db.get_character_expression_image(char_id, "thinking") is not None
    from tldw_chatbook.Widgets.Persona_Widgets.personas_character_editor_widget import (
        PersonasCharacterEditorWidget,
    )
    assert screen.query_one(PersonasCharacterEditorWidget).current_avatar_bytes() is not None  # idle staged
```

- [ ] **Step 2: Run to verify current behavior**

Run:
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py::test_import_vpack_from_path -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
NOTE: with Tasks 1-2 in place this may already PASS (the path method calls the resolver, which now auto-detects). If it passes immediately, confirm it is asserting real behavior by temporarily breaking `_detect_vpack` (return None) and seeing it fail, then restore — record that in the report. The test is the end-to-end regression lock either way.

- [ ] **Step 3: Widen the picker filter**

In `_expression_set_import_dialog_worker` (READ FRESH), change the picker construction:
```python
            picker = EnhancedFileOpen(
                title="Import Expression Set (.zip / .tldw-persona-vpack)",
                filters=Filters(
                    ("Archives", lambda p: p.suffix.lower() in (".zip", ".tldw-persona-vpack")),
                ),
                context="character_expression_set_import",
            )
```
Nothing else in the worker, the path method, the orchestrator, or the summary changes.

- [ ] **Step 4: Run the focused suites**

```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/UI/test_personas_expression_slots.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
then (separate invocation — different rootdir):
```
cd /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.claude/worktrees/personas-redesign && HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Character_Chat/test_expression_set_io.py -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread
```
Then `... python -c "import tldw_chatbook.app"`. Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add tldw_chatbook/UI/Screens/personas_screen.py Tests/UI/test_personas_expression_slots.py
git commit -m "feat(personas): P3d-3 Task 3 — accept .tldw-persona-vpack in the Import set picker + end-to-end"
```

---

## Self-Review (author)

- **Spec coverage:** Unit 2 extractor incl. preview_frame/asset_ids/region/status-leniency/cache/never-raise (Task 1) ✓; Unit 1 detection (root + single-shared-prefix), targeted dispatch, budget threading, first-writer-wins merge (Task 2) ✓; Unit 3 filter + end-to-end (Task 3) ✓; security posture is code-commented in Tasks 1-2 ✓; out-of-scope items appear in no task ✓; the two RED-provable pins (shared-sheet budget → Task 1; manifest-beyond-#64 → Task 2) ✓.
- **Type consistency:** `_resolve_vpack_expression_set(zf, *, prefix, start_total) -> (ExpressionSetResolution, int)` defined T1, consumed T2 with matching kwargs. `_detect_vpack(zf) -> str | None` defined and consumed in T2. `_candidate_pairs(paths, start_total=0) -> (pairs, skipped, notes, total)` — the 4-tuple is consumed by the T2 dispatch code shown. Test helpers `make_vpack_bytes`/`simple_vpack`/`_sheet_2x1`/`_extract` defined T1, reused T2/T3.
- **Placeholder scan:** every code step shows complete code; the two READ-FRESH notes point at a named method (`_expression_set_import_dialog_worker`) and the moved-verbatim stem loop.
- **Deliberate note:** Task 3's e2e may be GREEN-on-arrival (Tasks 1-2 make the path method work end-to-end); the step tells the implementer to break-and-restore `_detect_vpack` to confirm it is load-bearing — the P3d-2 Task-5 precedent.
