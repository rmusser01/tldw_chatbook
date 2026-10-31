# S15 + S16 — Models/inference; Persona/visual/pets

**Coverage S15** (39 files / 20,758 lines): read in full **3** · sampled **10** · mechanical only **26**.
**Coverage S16** (50 files / 23,660 lines): read in full **0** · sampled **21** · mechanical only **29**.
Mechanical = scanned for 12 pattern families (subprocess, locks, `get_cli_setting`, fsync, egress, `json.loads`
hooks, `print`, logging, `re.compile`, `run_worker`, timers, importers), **not read for logic**.
**No file in either slice exceeds 5,000 lines** (largest: `Model_Artifacts/service.py` 4,853) — no god-module finding.

## Findings

### P1 [D1] — `[model_catalog] use_models_dev` is documented to users as a working feature but nothing in production ever fetches the catalog, so the lookup layer is permanently empty
- Where: `LLM_Provider_Catalog/models_dev_catalog.py:150 fetch_models_dev`; consumers
  `model_capabilities.py:877-881`, `LLM_Calls/pricing_catalog.py:424-428`; user-facing claim
  `Docs/User_Guide/console/context-and-rag.md:106-115`.
- Evidence: `grep -rn "fetch_models_dev\|models_dev_catalog.json" --include='*.py' .` → the only non-test hits for
  `fetch_models_dev` are **its own definition**; the two production importers import `models_dev_entry` only.
  `reset_memory_cache`'s own docstring (`:229-234`) says *"the **future** fetch-wiring MUST call this"*.
  `backlog/tasks/task-26023*.md` records the deferral.
- Why it matters: the **shipped user guide** states "the catalog is fetched in the background with a conditional
  ETag request and disk-cached". A user who turns the setting on gets **zero behaviour change** — no context
  windows, no vision flags, no prices — and no diagnostic says why.
- Recommended correction: wire the refresh (the task names the pattern) **or** amend the user guide. Do not leave
  the doc asserting a code path that does not exist. · Size: M · Confidence: verified
- Pinning test: `Tests/Chat/test_models_dev_catalog.py` exercises `fetch_models_dev` **with an injected `http_get`**
  — it pins the function, not the wiring, so **it stays green forever.**

### P2 [D1] — `Widgets/Tamagotchi/tamagotchi_storage.py` reports six storage failures with `print()`, including a failed pet-state **save**
- `:227, 241, 495, 507, 519, 579` — exactly these six in either slice, **and the same file already has a
  `logger` at `:26` and uses it at seven other lines.** `:495` is
  `except Exception: print(f"Error saving to SQLite: {e}"); return False` **on a data-write path**. Under a Textual
  app stdout is not a log sink, so a persistent write failure produces no log line anywhere. · Size: S · verified

### P2 [D3] — `tldw_chatbook/Models/evaluation_state.py` (616 lines) has zero importers anywhere, including tests, and the package has no `__init__.py`
- `grep -rn "evaluation_state" --include='*.py' .` → **only the file itself.** Per-symbol sweep for
  `EvaluationState`, `EvaluationRun`, `EvaluationConfig`, `ViewType` → **zero hits outside the file**.
  `ls tldw_chatbook/Models/` → **no `__init__.py`.** It still occupies a row in the shipped production-diagnostic
  inventory. `RunStatus` collides with the live `Scheduling/models.py:63`, a different enum.
- Size: S · Confidence: verified
- Already covered: **`Models/` appears in none of TASK-32807's six sub-tasks.**

### P2 [D4] — `Local_Inference/mlx_lm_inference_local.py` is a dead second implementation of the **live** MLX spawn/stop path, and the two have already drifted
- Dead: `:22 start_mlx_lm_server` / `:110 stop_mlx_lm_server`. Live:
  `Event_Handlers/LLM_Management_Events/llm_management_events_mlx_lm.py:88-119` + `server_lifecycle.py:25` +
  `app.py:7818`, reached from `mlx-start-server-button`. **Every non-self hit is in
  `Tests/LLM_Management/test_mlx_lm.py`.**
- **Drift already present:** argument parsing — dead copy `additional_args.split()`, live copy `shlex.split(...)`;
  the dead copy passes `stdout=PIPE, stderr=PIPE, bufsize=1` **with no reader** (a server writing >64 KB to stdout
  would block forever) while the live copy drains through a worker; **both** spawn bare `"python"` rather than
  `sys.executable`, so on this repo's uv-managed venv the spawned interpreter is whatever PATH resolves — **the
  defect is duplicated, not isolated.**
- Why it matters: **TASK-32806.5 is In Progress on the *live* stop path.** A second `stop_mlx_lm_server` with its own
  terminate/kill ladder and a green 30-test suite is exactly the thing a fixer patches by mistake.
- Size: S (delete) · Confidence: verified
- Pinning test: `Tests/LLM_Management/test_mlx_lm.py` — **30 tests asserting the dead code's current behaviour, which
  is why it has survived.** A test suite for an unreachable module, not a requirement.

### P2 [D1] — Both `LLM_Provider_Catalog` disk writes do `os.replace` with **no `fsync` at all**
- `models_dev_catalog.py:133-147` (`mkstemp` → `fdopen`/`write` → `os.replace`, **no flush, no fsync**) and
  `model_discovery_disk_cache.py:375-387` (`tmp_path.write_bytes` → `os.replace`).
  `grep -rn "fsync" tldw_chatbook/LLM_Provider_Catalog/` → **no matches.** For contrast, within these slices
  `Model_Artifacts/fetch.py:277`, `service.py:4003`, `LLM_Management/snapshot_store.py:130,162`,
  `Actor_Packs/publication.py:110`, `Persona_Visual/importer.py:733,1013` all fsync — and
  `Actor_Packs/publication.py:259 _fsync_parent` is **the only site in either slice that fsyncs the parent
  directory and reports a `durability` string.**
- Why it matters: a power loss between write and writeback publishes a zero-length or partial file under the real
  name. `model_discovery_disk_cache.load` treats a corrupt entry as "rejected" and logs a warning, **so the user
  silently loses their discovered-model catalog rather than crashing — which is why nobody has noticed.**
- Size: S · Confidence: verified
- Already covered: **partially, and the "Done" is now insufficient** — TASK-32808.5 is marked Done but these two
  sites were not converted and have **no fsync of any kind**.

### P2 [D1/D4] — The Persona Visual **import** path (the untrusted one) is the only one of three decode sites missing the aggregate decoded-pixel bound
- Missing at `Persona_Visual/importer.py:762-793 _inspect_image`; present at `assets.py:430` and
  `authoring_workspace.py:395`.
- Evidence: `MAX_ASSET_DECODED_PIXELS` is defined at `assets.py:49` (= `MAX_ASSET_DIMENSION**2 * 4` = 67,108,864),
  enforced at two sites, and **absent from `importer.py`**, whose `from .contracts import (...)` block imports
  `MAX_ASSET_DIMENSION`, `MAX_ASSET_TOTAL_BYTES`, `MAX_FRAMES_PER_ANIMATION` but **not the pixel bound — the
  constant does not live in `contracts.py` at all.**
- Why it matters: `_inspect_image` bounds each frame and the frame count (≤240), then decodes **every** frame in a
  loop. Ceiling: **4096 × 4096 × 240 = 4.03 × 10⁹ pixels per asset, 60× the bound its two siblings enforce**, from a
  compressed APNG/animated-WebP that fits easily inside the 100 MB `MAX_ASSET_TOTAL_BYTES`. Peak memory stays
  bounded (frames decode sequentially) so this is **CPU/wall-clock amplification on an import, not a heap
  blow-up — which is why it reads as safe.**
- Recommended correction: move `MAX_ASSET_DECODED_PIXELS` into `Persona_Visual/contracts.py` (which already owns the
  three siblings and is the module all three importers already import from) and add the
  `width * height * frame_count` check before the decode loop. · Size: S · Confidence: verified (ceiling is
  arithmetic from the constants, not measured)
- Already covered: **no.** TASK-32806.8's named guards (`Actor_Packs/contracts.py:713-739`, `Petdex/sources.py:195`)
  are **both present and correct** — see triage. This third site is the one without the check.

### P2 [D1] — `Persona_Buddy/controller.py` swallows eight exceptions to `None`/`False` with **zero logging in the entire 1,444-line module**
- `:304, 753, 759, 1104, 1146, 1175, 1190, 1218`. `grep -c "except Exception"` → **8**; `grep -c "logger\."` → **0**;
  no loguru or stdlib import at all. Only one of the eight carries a justifying `# noqa: BLE001`.
- Why it matters: `_read_local_persona`, `_read_graph` and `_resolve_runtime` each catch bare `Exception` and return
  `None`, so a repository bug, a schema mismatch, or a corrupted visual pack renders as **"this persona has no
  avatar"** with no log line, no notification, and no diagnostic field. · Size: S · Confidence: verified

### P2 [D3] — The `Widgets/Tamagotchi/` **widget** half is unreachable while its **storage** half is wired into backup/recovery and the private-SQLite allowlist
- Dead ≈ **2,181 lines**: `base_tamagotchi.py` (580), `tamagotchi_behaviors.py` (408), `validators.py` (420),
  `tamagotchi_messages.py` (317), `tamagotchi_sprites.py` (252), `examples/simple_tamagotchi.py` (204).
  Live: `tamagotchi_storage.py` (608) via `DB/private_sqlite.py:634`, three `Backup_Recovery` participants.
- Evidence: per-symbol sweep → every hit is inside `Widgets/Tamagotchi/` itself, `Tests/`, or `css/`. No screen, no
  registry route, no `compose()` mounts any of them. **The code says so itself:** `tamagotchi_storage.py:344-348` —
  *"No import site outside this module was found for `SQLiteStorage` … this class appears dormant/unmounted."*
- Why it matters: **the dead half is not inert.** It keeps CSS rows alive (checked by two tests), a timer-inventory
  row, and a production-diagnostic-inventory entry — all maintenance cost for a pet no UI can create.
- Recommended correction: **decide the subsystem's status first — it is a product question, not a cleanup.**
  If retired, the delete is a multi-file commit (modules + CSS block + timer rows + inventory rows), and the
  storage/backup half is a separate decision for existing users' data. · Size: M · Confidence: verified
- Already covered: **no** — TASK-32807.1 covers "21 unreferenced **top-level** widget modules"; this is a sub-package.

### P3 [D4] — Seven `json.loads` sites in `Model_Artifacts/` pass `object_pairs_hook` but omit `parse_constant`
- `recovery.py:105-107`; `service.py:2597, 3687, 3737, 4289, 4370, 4809`. **Every site in `Persona_Visual/` (4),
  `Actor_Packs/export.py` (2) and `Petdex/` (2) pairs the hook with `parse_constant=`; all seven `Model_Artifacts/`
  sites pass the hook alone.** `recovery.py:105` parses a restore descriptor read out of a **backup archive** — an
  import boundary; a `NaN` there passes the duplicate-key gate and downstream equality checks against it are
  silently always-false. · Size: S · Confidence: verified
- Already covered: **task-32855 owns the consolidation, and the drift runs the other way from its brief** — the
  three `Persona_Visual/` copies it names are all *correct*; the defective sites are in `Model_Artifacts/`.

### P3 [D4] — `Petdex/review.py::write_native_export` omits the parent-directory fsync its near-twin performs and reports
- `Petdex/review.py:170-191` vs `Actor_Packs/publication.py:95-146` + `:259 _fsync_parent`. Both are user-facing
  "export this pack" publications with an identity-pinned destination. **One tells the caller whether the rename is
  durable; the other cannot.** `_fsync_parent` is 11 lines and already handles `EINVAL`/`ENOTSUP`/`EOPNOTSUPP`;
  `Utils/atomic_file_ops.py` is the obvious destination, **and lifting it there would close that helper's own
  parent-dir gap at the same time.** · Size: S

### P3 [D4] — Two archive-member validators on the same security boundary have drifted
- `Actor_Packs/importer.py:925-970` vs `Persona_Visual/importer.py:341-389`, both named `_validated_members`:

| check | `Actor_Packs` | `Persona_Visual` |
|---|---|---|
| per-member cap | `> MAX_MEMBER_BYTES` (50 MiB) | **absent** |
| `info.create_system` allowlist | `{0, ZIP_CREATE_SYSTEM}` | **absent** |
| member-count floor | `2 <= len <= MAX_FILES + 1` | upper bound only |
| ratio bomb guard, running-total cap, encrypted-flag, external_attr mode, NFC-casefold collision, nested-archive suffixes | present | present |

  **The drift is mitigated today** — `Persona_Visual` re-checks each member against its manifest's declared
  `byte_count` and caps JSON members — so no hole is open. But two hand-maintained copies of a zip validator on an
  untrusted-import boundary will keep diverging. Neither package may import the other; needs a neutral home.
  · Size: M · Confidence: verified

### P3 [D4] — `_valid_url_hostname` / `_valid_url_authority` are byte-identical across two artifact-catalog trust boundaries
- `Model_Artifacts/service.py:493, 514` ≡ `TTS/audio_cpp_artifact_catalog.py:161, 182`. Both validate a declared
  artifact **source URL** before anything is fetched. `Utils/egress.py` already owns URL parsing and policy but has
  **no hostname-*syntax* validator — which is why two catalogs rolled their own.**
  `Model_Artifacts/service.py:1455 take_cleanup_owner` ≡ `TTS/audio_cpp_guided_launch.py:98` is a **third verbatim
  pair between the same two packages**, suggesting one was cloned from the other wholesale. · Size: S

### P3 [D3] — `tamagotchi_storage.py` guards an import of its own sibling module as if it were an optional dependency
- `:20-24` `try: from .validators import StateValidator / except ImportError: StateValidator = None`, while
  `validators.py:229` always defines it. **A genuine `ImportError` (a syntax error inside `validators.py`) silently
  disables state validation instead of failing loudly** — `load_with_recovery` then accepts unvalidated state. · Size: S

### P3 [D4] — Two near-verbatim helper clusters inside S16 with zero behavioural drift
- exact-length fd read ×3 in one package (`Persona_Visual/{importer:1046, assets:380, publication:1118}`) plus
  `Character_Chat/visual_identity.py:1542`; write-all-to-fd ×2 byte-identical
  (`Actor_Packs/importer.py:1551`, `Persona_Visual/importer.py:1018`). Home: `Utils/fd_protection.py` (already
  exists and already owns fd-level concerns) — **not a new module.** · Size: S

## Candidate triage
**RETIRED — the brief's own hypotheses, each with evidence:**
- **Model download with no checksum or size bound: retired.** `Model_Artifacts/fetch.py::stream_fetch` enforces a
  hard `max_bytes` **mid-stream** (`:268`), per-hop `check_url_or_raise_async` (`:173`), cross-origin credential
  stripping at both header and client-`auth` level (`:184-201`), `follow_redirects=False` with a manual hop cap,
  HTTPS-downgrade rejection, strong-validator-only resume with `Content-Range` start verification, and `os.fsync`
  before return. `acquisition.py:1762-1822` then SHA-256-verifies every staged file **off the loop** and refetches on
  mismatch up to `MAX_FILE_REFETCHES`. **At least as strong as the `Audio/diarizer_engine_onnx.py` reference shape.**
- **12-of-26 `subprocess.run` without `timeout=` — check `LLM_Management/`: retired for these slices.**
  `grep -rn "subprocess.run"` → **zero matches.** `LLM_Management/` here is `snapshot_*` only and spawns nothing.
  **The brief's premise that `LLM_Management/` supervises inference servers does not hold** — that lives in
  `Event_Handlers/LLM_Management_Events/`. The only `Popen` sites are `machine_memory_probe.py:826,844` (deadline
  loop, bounded output buffer, trusted-executable verification, `_terminate_and_reap` ladder) and the dead MLX module.
- **A fourth "is this process alive?" copy: retired.** Two hits, both in the dead module.
- **`persona_visual_participants` vs `visual_identity_participants` locking divergence: RETIRED — both lock.**
  `persona_visual_participants.py:35` holds a per-**source** `RLock` (acquired `:410`, released `:508` in a
  `finally`); `visual_identity_participants.py:294` has `"_lock"` in `__slots__` and uses
  `with request(), candidate._lock:` at `:981` — per-**candidate**. The granularities differ because the structures
  differ, and the comment at `:276` says so. *(Lead confirmed and recorded this as a cross-slice conflict resolution
  — S25's duplication finding survives, its lock-drift claim does not. See `phase4-verification.md`.)*
- **`Actor_Packs/contracts.py:713-739` and `Petdex/sources.py:195` image guards: confirmed present and correct**, and
  stronger than described (magic-byte gate, format-vs-extension match, dimension cap, pixel cap, byte cap,
  `image.verify()`, `DecompressionBombError` in the catch tuple, `n_frames == 1`). Note:
  `grep -rn "MAX_IMAGE_PIXELS" <S16>` → **zero matches anywhere in the slice** — these are dimension caps, not a
  `MAX_IMAGE_PIXELS` assignment.
- **"every image ingress has the guard": confirmed, one gap.** 11 `Image.open` sites; ten bound dimensions before
  decode; the exception is `Persona_Visual/importer.py:771` (P2 above).
- **Archive traversal / symlink / bomb: confirmed handled** (with the drift filed as P3). Neither importer uses
  `extractall`; `Actor_Packs/importer.py:828-909` pre-validates EOCD/central-directory geometry **before `ZipFile`
  allocates**.
- **`Widgets/Tamagotchi/` Textual hazards: all three retired.** `query_one` in timers → `grep` → **zero**;
  `run_worker`/`@work` → **zero matches in either slice**; mutable class attributes → all are `reactive(...)`
  descriptors or immutable, every mutable assigned in `__init__`.
- `get_cli_setting` dotted → all four sites use the two-arg section form. `re.compile` in bodies → one hit, a class
  attribute compiled once at import. loguru+stdlib in one file → **none** (three files use stdlib alone; one also
  `print()`s → P2). `fetchall` ×3 → point lookups / one persona's intents / a store with no UI to create rows.
  `Model_Artifacts/recovery.py:215,334` inline `is_relative_to` → operates on paths already enumerated by the backup
  inventory, and a containment failure marks the item `malformed` rather than admitting it.
  `Persona_Buddy/library.py` `lock_and_execute` → all three `execute_query` calls run inside
  `with self.db.transaction(immediate=True):`.
  `openai_compatible_model_discovery.py` / `remote_huggingface.py` bypass egress → **retired with reason**: the
  former targets a **user-configured, often private-IP** endpoint where egress's private-IP policy would break the
  intended use, and compensates with `follow_redirects=False`, `Accept-Encoding: identity`, and a bounded read;
  the latter is fixed-origin and bounded. **Neither leaks credentials across a redirect because neither follows one.**
- **`_maybe_await` doing blocking I/O on the loop: not found in these slices.** `Persona_Buddy/` has no
  `_maybe_await` at all.
**CONFIRMED:** both `LLM_Provider_Catalog` `os.replace` sites (**reframed** — no fsync of any kind, not a missing
helper adoption); `Models/evaluation_state.py`; the dead MLX module; the seven `parse_constant` omissions.
**NOT FILED:** `Model_Artifacts/service.py` at 4,853 lines — below the 5k threshold and below every ratchet row
(smallest 6,760). **TASK-32809.2's hand-picked scope is correct as drawn**; noted as the largest ungoverned module here.

## D4 observations for repo-wide Phase 3
1. **Exact-length fd read — 4 members, no drift, plus 1 drifted outlier.** `Petdex/sources.py:287 _read_fd(fd, cap)`
   takes a *cap* rather than an exact expectation **and additionally `fstat`s before and after to reject a file
   swapped mid-read** (`raise ValueError("petdex_source_stale")`). **Do not fold that one in — it is a stricter
   contract.** Home for the other four: `Utils/fd_protection.py`.
2. **`_write_all(fd, bytes)` — 2 byte-identical in-slice**, plus same-idiom-different-handle at
   `Terminal/posix_backend.py:179`, `Notes/sync_paths.py:924`, `Backup_Recovery/crypto.py:159`. One fd helper covers
   three of five.
3. **Strict-JSON parse — the drift is `parse_constant`, not `object_pairs_hook`.** When task-32855 lands its single
   parser, **the migration must *add* `parse_constant` at the seven `Model_Artifacts/` sites, not merely rename the
   hook.**
4. **`_unique_object`/`_unique`/`_reject_*` — 8 in-slice copies with four naming conventions.** Bodies identical;
   only the exception type and message differ. **A shared parser needs a caller-supplied error factory, or these
   call sites lose their typed errors.**
5. **Three verbatim pairs between `Model_Artifacts/` and `TTS/`** (`_valid_url_hostname`, `_valid_url_authority`,
   `take_cleanup_owner`) — suggests one package was cloned from the other wholesale. `Utils/egress.py` owns URL
   *policy* but has **no syntax validator**; that absence is why both rolled their own.
6. **Zip-member validation — 2 hand-maintained copies on the same untrusted boundary**, drift table above. Neither
   package may import the other; needs a neutral home.
7. **`_non_empty_model_id` ≡ `_valid_model_id`** — byte-identical, 6 lines, **same package**, different names. The
   cheapest possible fold.
8. **Atomic-publish shape — 3 in-slice variants at three durability levels:** `Actor_Packs/publication.py` (fd-based,
   `O_NOFOLLOW|O_EXCL`, temp-identity re-validation, file fsync, **parent fsync**, durability reported) >
   `Petdex/review.py:170` (file fsync, no parent fsync) > `LLM_Provider_Catalog/*` (**no fsync at all**).
   **`Utils/atomic_file_ops.py` sits between tiers 2 and 3.** If Phase 3 consolidates here, tier 1's `_fsync_parent`
   is the piece worth lifting into the helper.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| `Local_Inference/ollama_model_mgmt.py`'s `requests.Session()` (`:110`) sets a timeout on every request and does not follow a redirect with credentials attached to a user-supplied Ollama base URL | read `:100-260` only; did not trace `_ollama_request`'s kwargs through all nine callers. `Utils/egress.py:1012 create_default_session` exists precisely for this and is not used | `sed -n '80,200p' …/ollama_model_mgmt.py && grep -n "timeout\|allow_redirects\|headers" …/ollama_model_mgmt.py` |
| The Persona_Visual pixel amplification is reachable from the Personas import UI (vs only the library API), and whether `_inspect_image` runs on the loop or in a thread — which decides background stall vs app freeze | not traced | `grep -rn "_inspect_image\|import_persona_visual\|read_persona_visual_pack" --include='*.py' tldw_chatbook/ \| grep -v Persona_Visual/importer.py` |
| `Model_Artifacts/recovery.py:215,334`'s `is_relative_to` cannot be defeated by a symlink inside the recovery staging tree | `is_relative_to` is **lexical**; read the call sites but not `Backup_Recovery/file_inventory`'s path construction | `grep -rn "resolve()\|realpath\|O_NOFOLLOW" tldw_chatbook/Backup_Recovery/file_inventory.py \| head -30` |
| The Tamagotchi widget half has no dynamic/string-based mount a symbol grep cannot see | `Widgets/Tamagotchi/__init__.py:56` has a PEP-562 `__getattr__`; swept exact module paths and every exported symbol but did not run `--collect-only` | `pytest --collect-only -q 2>&1 \| tail -5 && grep -rn "Tamagotchi" --include='*.py' tldw_chatbook/UI/ tldw_chatbook/app.py` |
| Deleting `Models/evaluation_state.py` and the dead MLX module does not break a derived artifact | both appear in `Docs/security/production-diagnostic-inventory.json`, which preflight checks | `grep -n "evaluation_state\|mlx_lm_inference_local\|base_tamagotchi" Docs/security/production-diagnostic-inventory.json` |
| The models.dev layer is genuinely inert at runtime (not fed by some other writer of the cache file) | verified by grep over `*.py`/`*.toml`/`*.md`; app must not be run | `pytest Tests/Chat/test_models_dev_catalog.py -q && ls -la ~/.local/share/tldw_cli/models_dev_catalog.json` |
