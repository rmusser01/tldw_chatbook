# S15+S16 validation — Models/inference; Persona/visual/pets

Reviewed against worktree `tldw-t2-base` at `d0face3ebe` (origin/dev). `git diff --stat 3722a85748..HEAD` over every
package this slice touches (`LLM_Provider_Catalog`, `Local_Inference`, `Models`, `Widgets/Tamagotchi`,
`Persona_Visual`, `Persona_Buddy`, `Model_Artifacts`, `Actor_Packs`, `Petdex`, `TTS`) is **empty** — none of these
files changed in the 25 commits since the review. All line numbers below matched the review's cited numbers exactly
unless noted.

## 1. P1 [D1] — `use_models_dev` documented as working, catalog never fetched in production
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/LLM_Provider_Catalog/models_dev_catalog.py:150` (unchanged)
- Proof: `grep -rn "fetch_models_dev" --include='*.py' tldw_chatbook/ Tests/` → only the definition (`:150`) and its
  own docstring reference (`:232`) in production; every other hit is in `Tests/Chat/test_models_dev_catalog.py`.
  `model_capabilities.py:878` and `pricing_catalog.py:425` import only `models_dev_entry`, never `fetch_models_dev`.
  `Docs/User_Guide/console/context-and-rag.md:112-114` still reads "the catalog is fetched in the background with a
  conditional ETag request and disk-cached" — false as shipped.

## 2. P2 [D1] — `Widgets/Tamagotchi/tamagotchi_storage.py` reports six storage failures via `print()`
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py:227,241,495,507,519,579` (unchanged)
- Proof: `grep -n "print(" tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py` → exactly those six lines;
  `:495` is `except Exception as e: print(f"Error saving to SQLite: {e}"); return False`. The same file has
  `logger.warning/info/error` calls at `:123,127,131,136,269,281,296`, confirming the logger exists and is used
  elsewhere but not on these six paths.

## 3. P2 [D3] — `tldw_chatbook/Models/evaluation_state.py` has zero importers, package has no `__init__.py`
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Models/evaluation_state.py` (whole file, unchanged)
- Proof: `grep -rln "evaluation_state" --include='*.py' .` → only the file itself. `ls tldw_chatbook/Models/` →
  `evaluation_state.py` alone, no `__init__.py`. `RunStatus` is defined both at `evaluation_state.py:27` (plain
  `Enum`) and `Scheduling/models.py:63` (`str, Enum`) — confirmed name collision.

## 4. P2 [D4] — `Local_Inference/mlx_lm_inference_local.py` is a dead second MLX spawn/stop implementation, already drifted from the live path
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Local_Inference/mlx_lm_inference_local.py:22,110` (unchanged)
- Proof: `grep -rn "start_mlx_lm_server\|stop_mlx_lm_server" --include='*.py' .` → only the module's own definitions
  outside `Tests/LLM_Management/test_mlx_lm.py`. Live path confirmed:
  `llm_management_events_mlx_lm.py` registers `"mlx-start-server-button": handle_start_mlx_server_button_pressed`.
  Drift confirmed: dead copy uses `additional_args.split()` (`:58`) vs live copy's `shlex.split(additional_args)`
  (`llm_management_events_mlx_lm.py:100`); dead copy's `Popen(..., stdout=PIPE, stderr=PIPE, text=True, bufsize=1)`
  (`:65-72`) has no reader in this module. Both spawn the literal string `"python"` (`:47` and
  `llm_management_events_mlx_lm.py:89`), not `sys.executable`.

## 5. P2 [D1] — Both `LLM_Provider_Catalog` disk writes do `os.replace` with no `fsync` at all
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/LLM_Provider_Catalog/models_dev_catalog.py:132-144`, `model_discovery_disk_cache.py:374-387` (unchanged)
- Proof: `grep -rn "fsync" tldw_chatbook/LLM_Provider_Catalog/` → zero matches (exit code 1). Read both write paths:
  `_write_cache_file` (`models_dev_catalog.py`) does `mkstemp` → `os.fdopen(...).write(payload)` → `os.replace` with
  no `os.fsync` call anywhere in the function; `ModelDiscoveryDiskCache.save` does `tmp_path.write_bytes(encoded)` →
  `os.replace(tmp_path, self.path)`, also no fsync.

## 6. P2 [D1/D4] — Persona Visual **import** path is the only one of three decode sites missing the aggregate pixel bound
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Persona_Visual/importer.py:762` `_inspect_image` (was `:762-793`, matches exactly)
- Proof: `grep -n "MAX_ASSET_DECODED_PIXELS" tldw_chatbook/Persona_Visual/{contracts,assets,authoring_workspace,importer}.py`
  → defined in `assets.py:49`, enforced at `assets.py:430` and `authoring_workspace.py:395`, **absent from
  `importer.py`** entirely (0 hits) even though `importer.py:46-48` imports `MAX_ASSET_DIMENSION`,
  `MAX_ASSET_TOTAL_BYTES`, `MAX_FRAMES_PER_ANIMATION` from the same `contracts` module in the same import block.
  `contracts.py` defines `MAX_ASSET_DIMENSION` but not the pixel-bound constant, confirming the review's claim that
  it doesn't live there to be imported.

## 7. P2 [D1] — `Persona_Buddy/controller.py` swallows eight exceptions to `None`/`False` with zero logging in the 1,444-line module
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Persona_Buddy/controller.py:304,753,759,1104,1146,1175,1190,1218` (unchanged)
- Proof: `grep -n "except Exception" tldw_chatbook/Persona_Buddy/controller.py` → exactly those 8 lines.
  `grep -c "logger\."` → 0. `wc -l` → 1444 lines. Only `:1146` carries `# noqa: BLE001`; the other seven do not.

## 8. P2 [D3] — `Widgets/Tamagotchi/` widget half unreachable while storage half is wired into backup/recovery
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Widgets/Tamagotchi/{base_tamagotchi,tamagotchi_behaviors,validators,tamagotchi_messages,tamagotchi_sprites}.py` + `examples/simple_tamagotchi.py`
- Proof: `grep -rn "base_tamagotchi\|tamagotchi_behaviors\|TamagotchiWidget\|BaseTamagotchi" --include='*.py' tldw_chatbook/`
  excluding `Widgets/Tamagotchi/` itself → zero hits. `tamagotchi_storage.py:344-348`'s own docstring states "No
  import site outside this module was found for `SQLiteStorage`... this class appears dormant/unmounted."
  `DB/private_sqlite.py:634` lists `"tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage"` in its owner-policy
  registry, confirming the storage half (not the widget half) is live infrastructure.

## 9. P3 [D4] — Seven `json.loads` sites in `Model_Artifacts/` pass `object_pairs_hook` but omit `parse_constant`
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Model_Artifacts/recovery.py:106`; `service.py:2597,3687,3737,4289,4370,4809` (unchanged)
- Proof: `grep -n "object_pairs_hook" tldw_chatbook/Model_Artifacts/{recovery,service}.py` → exactly those 7 lines,
  none carrying `parse_constant=`. Cross-check: `grep -rn "parse_constant" tldw_chatbook/{Persona_Visual,Actor_Packs/export.py,Petdex}`
  → 8 hits across those packages, **0 in `Model_Artifacts/`** — confirms the asymmetry the finding claims.

## 10. P3 [D4] — `Petdex/review.py::write_native_export` omits the parent-directory fsync its near-twin performs
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Petdex/review.py:170-191` vs `Actor_Packs/publication.py:141,259` `_fsync_parent` (unchanged)
- Proof: read both. `write_native_export` does `output.flush(); os.fsync(output.fileno())` on the temp file only,
  then `os.replace(temporary, target)` — no directory-fd fsync anywhere in the function or its callers.
  `Actor_Packs/publication.py:141` calls `_fsync_parent(parent_fd)` (defined `:259`, handles `EINVAL`/`ENOTSUP`/
  `EOPNOTSUPP`) and returns a `durability` string; `Petdex/review.py` has no equivalent call or return value.

## 11. P3 [D4] — Two archive-member validators (`Actor_Packs/importer.py` vs `Persona_Visual/importer.py`) have drifted
- Verdict: CONFIRMED
- Site now: `Actor_Packs/importer.py:925-970`, `Persona_Visual/importer.py:342-389` (both `_validated_members`, line numbers match)
- Proof: read both bodies. `Actor_Packs` has `info.file_size > MAX_MEMBER_BYTES` (per-member cap) — **absent** in
  `Persona_Visual`. `Actor_Packs` has `info.create_system not in {0, ZIP_CREATE_SYSTEM}` (allowlist) — **absent** in
  `Persona_Visual`. `Actor_Packs` has `2 <= len(infos) <= MAX_FILES + 1` (floor+ceiling); `Persona_Visual` has only
  `len(infos) > _MAX_MEMBER_COUNT` (ceiling only, no floor). Ratio-bomb guard, running-total cap, encrypted-flag
  check, `external_attr` mode check, NFC-casefold collision check, and nested-archive-suffix check are present in
  both, matching the review's table exactly.

## 12. P3 [D4] — `_valid_url_hostname`/`_valid_url_authority` byte-identical across `Model_Artifacts/` and `TTS/`
- Verdict: CONFIRMED
- Site now: `Model_Artifacts/service.py:493,514` ≡ `TTS/audio_cpp_artifact_catalog.py:161,182` (unchanged); third pair `service.py:1455 take_cleanup_owner` ≡ `TTS/audio_cpp_guided_launch.py:98`
- Proof: `diff <(sed -n '493,532p' service.py) <(sed -n '161,200p' audio_cpp_artifact_catalog.py)` → no output
  (byte-identical 40-line span covering both functions). `grep -n "take_cleanup_owner"` confirms the third pair
  exists at both cited sites.

## 13. P3 [D3] — `tamagotchi_storage.py` guards import of its own sibling `validators.py` as if it were optional
- Verdict: CONFIRMED
- Site now: `tldw_chatbook/Widgets/Tamagotchi/tamagotchi_storage.py:19-23` (was `:20-24`, off by one — same code)
- Proof: read the import block — `try: from .validators import StateValidator / except ImportError: StateValidator = None`.
  `validators.py:229` defines `class StateValidator` unconditionally (no guard, no optional-dep gate), confirming
  it is a genuine sibling module, not an optional dependency — a real `ImportError` inside `validators.py` (e.g. a
  syntax error) would silently disable state validation rather than fail loudly.

## 14. P3 [D4] — Two near-verbatim helper clusters: exact-length fd read (×4) and write-all-to-fd (×2)
- Verdict: CONFIRMED
- Site now: `Persona_Visual/{importer.py:1048,assets.py:382,publication.py:1120}`, `Character_Chat/visual_identity.py:1552`; `Actor_Packs/importer.py:1551`, `Persona_Visual/importer.py:1018` (fd-read sites shifted +1-2 lines; write-all sites unchanged)
- Proof: the three `Persona_Visual` fd-read functions (`_read_fd`, `_read_fd_bounded`, `_read_bounded`) are
  identical in shape: `remaining = expected + 1; while remaining: chunk = os.read(fd, min(CHUNK, remaining)); ...`.
  `_write_all` in `Actor_Packs/importer.py:1551` and `Persona_Visual/importer.py:1018` is byte-identical
  (`memoryview` + `while view: written = os.write(...)`). `Utils/fd_protection.py` exists, confirming the proposed
  home is real.
- Note: the fourth "exact-length fd read" member is not literally the same inline loop. `Character_Chat/visual_identity.py:1542-1550`
  defines two thin wrappers (`_read_fd_bounded`, `_read_stream_bounded`) that delegate to a local `_read_bounded`
  (`:1552`) whose body additionally clamps against `MAX_EXPRESSION_ASSET_BYTES` (`limit = min(expected_bytes, MAX_EXPRESSION_ASSET_BYTES) + 1`)
  — same idiom, one extra guard, not a byte-identical fourth copy. Doesn't change the verdict (the review's own
  phrasing — "plus Character_Chat/visual_identity.py:1542" — already treats it as the same-idiom fourth member,
  not a fifth byte-identical one).

TOTALS: confirmed=14 fixed=0 wrong=0 demoted=0 promoted=0
