# S03 + S04 — `TTS/` (backends/adapters/audio.cpp; profiles/bundles/remainder)

**Coverage S03** (29 files / 26,521 lines): read in full **3** · sampled (targeted multi-region reads + per-file AST
map + pattern greps) **17** · mechanical only **9**.
**Coverage S04** (62 files / 39,395 lines): read in full **5** · sampled **18** · mechanical only **39**.
Repo-wide sweeps across both: AST scan for `re.compile` in function bodies (**0**), dotted `get_cli_setting` (**0**),
`run_worker` (**0**), `fetchall` (1, bounded), all 34 `except Exception: pass` sites read **with context**, module/
class/method size census, every `os.replace`/`fsync` site, every `subprocess` site.

## Findings

### P0 [D1] — `SimpleAudioPlayer.stop()` SIGKILLs the whole tldw_chatbook process group when a stuck player survives `kill()`
- Where: `TTS/audio_player.py:556-559` (`os.killpg(os.getpgid(self._current.process.pid), signal.SIGKILL)`),
  reachable from `:541-549`; the children are spawned at `:348`, `:353`, `:360` with **no `start_new_session`**.
- Evidence: **lead-verified** — see `phase4-verification.md` "S03-P0". A `Popen` child with no `start_new_session`
  inherits the parent's process group (`os.getpgid(child) == os.getpgrp()` → **True**, executed), so `killpg`
  targets the app itself and every other process in the terminal's foreground group. Guard chain:
  `terminate()` → `wait(2)` → `kill()` → `wait(0.5)` → on `TimeoutExpired` **and** Darwin → `killpg(SIGKILL)`.
- Why it matters: the user presses Stop on TTS playback and **the whole TUI is SIGKILLed** — no shutdown, no
  `close_tts_resources()`, no DB quiesce.
- Recommended correction: pass `start_new_session=(os.name == "posix")` on all three `Popen` calls; then `killpg` is
  correct. **The repo already ships this exact pairing with explanatory comments six times** —
  `Agents/run_hooks.py:342,365`, `Skills_Interop/skill_script_runner.py:299,425-427`,
  `Notes/git_process_containment.py:200`, `Agents/agent_worktree_git.py:90`, `Tools/git_tool_impls.py:254`,
  `Web_Server/artifact_share.py:169`. **TTS is the only site that calls `killpg` without it.**
- Size: S · Confidence: verified (pgid identity proven by execution; the `TimeoutExpired` precondition inferred)
- Pinning test: none (`rg "killpg|getpgid|start_new_session" Tests/` → zero under `Tests/TTS/`; the canonical
  positive is `Tests/STT/test_executor_process_tree.py:265,694`).
- Already covered: **no.** TASK-32806.5 names only `server_lifecycle.py`. Same defect class, and **worse here**:
  that task's site merely *leaks* a child; this one *kills the parent*.

### P1 [D1] — A transient read error on the Chatterbox voice-profile file permanently destroys every Chatterbox voice profile; there is no backup
- Where: `TTS/backends/chatterbox_voice_manager.py:48-59` (`except Exception: … self._profiles_cache = {}`) then
  `:61-72 save_profiles` — which, unlike the Higgs sibling, does **not** call `_create_backup()`.
- Evidence: `load_profiles` swallows any exception (incl. `json.JSONDecodeError`, `OSError`, and
  `bootstrap.RecoveryRequired` — verified a `RuntimeError` subclass, so `except Exception` catches it) and caches
  `{}`. `create_profile`/`update_profile`/`delete_profile` all do `profiles = self.load_profiles()` → `{}` →
  `save_profiles({one_profile})` → `atomic_private_write_text` **replaces the file**. The Higgs sibling at
  `higgs_voice_manager.py:100-114` takes a backup first; Chatterbox has no `_create_backup` at all.
- Why it matters: **one unreadable read plus one subsequent profile edit erases the user's entire Chatterbox voice
  library, irrecoverably.**
- Recommended correction: distinguish "absent" from "unreadable" — raise on a read failure rather than caching `{}`,
  and mirror Higgs's pre-write backup. Prior art for the correct shape: `TTS/profile_repository.py`, which treats an
  unreadable store as `corrupt_data` and refuses. · Size: S · Confidence: verified (path traced end to end)
- Already covered: **no.** TASK-32806.3 is the identical defect class but scoped to `MCP/permission_store.py`.
  **This is a second instance, on user-created content rather than config, with no backup layer.**

### P1 [D1] — The Kokoro ONNX model is downloaded, its SHA-256 is computed, and the digest is explicitly thrown away; the artifact is loaded by onnxruntime unverified, with no size cap and no durability barrier
- Where: `TTS/backends/kokoro.py:482-488` (`actual_checksum = hasher.hexdigest()` … `logger.warning("Checksum
  verification skipped - no known checksum available")`) and the identical block at `:545-547` for
  `voices-v1.0.bin`; downloader at `:164-224`.
- Evidence: the digest is fed (`:192-193`) and printed (`:483`, `:546`) but **never compared**.
  `_kokoro_stream_download` tracks `written` only for the progress log — **no ceiling**, so a redirected host can
  write unbounded bytes into `~/.config`. `os.replace(partial, destination)` at `:210` with **no `os.fsync` and no
  parent-directory fsync** (`rg "fsync" kokoro.py` → **0 hits**). The result is handed to `kokoro_onnx.Kokoro(...)`.
- Why it matters: (a) an ~310 MB binary blob whose integrity is never checked is deserialized by onnxruntime;
  (b) an unbounded download; (c) after a crash between `os.replace` and writeback, a torn file exists at
  `self.model_path`, `if not os.path.exists(...)` is then False **forever**, and every subsequent run loads a
  truncated model — the "recovery that cannot recover" shape.
- Recommended correction: pin the two release digests and fail on mismatch (**the code already computes them**); add
  a `MAX_*_BYTES` ceiling inside the chunk loop; `os.fsync(handle.fileno())` before `os.replace` plus a parent-dir
  fsync after. **Note on the helper question:** `Utils/atomic_file_ops.py` fsyncs the file but never the parent and
  uses plain `os.fsync`, so "adopt the helper" is not the right instruction — this site has *no* barrier at all and
  needs the parent fsync the helper also lacks. The correct in-repo model is `TTS/profile_repository.py:3157-3181`.
- Size: M · Confidence: verified
- Pinning test: `Tests/TTS/test_kokoro_download_hardening.py` has 5 tests — **`test_hasher_sees_every_byte` pins that
  the hasher is *fed*, not that the digest is *checked***, and the code comment calls the skip a TODO, so the repo is
  not asserting it as a decision. The atomicity test pins `os.replace` visibility, not durability.

### P1 [D1+D2] — Audiobook generation runs the entire decode/concat/normalize/ffmpeg pipeline on the event loop, and its `ffmpeg` call has no `timeout=` and inherits the TUI's stdin
- Where: `TTS/audio_service.py:412` (`subprocess.run(cmd, capture_output=True, text=True)` — no `timeout=`, no
  `stdin=DEVNULL`) inside `async def create_m4b_with_chapters` at `:321`; the same coroutine does
  `AudioSegment.from_file` per chapter (`:352`) and `combined.export` (`:387`, `:415`). Chain:
  `Event_Handlers/STTS_Events/stts_events.py:2683` → `:2803` → `TTS/audiobook_generator.py:742`.
- Evidence: `rg -n "to_thread|run_in_executor"` over both files → **zero hits**. Every blocking op in both
  (`open(...).write` at `audiobook_generator.py:653,699-706`, `AudioSegment.from_file` at `:835,885`, `.export` at
  `:772,856`) runs directly under `await` on the loop.
- Why it matters: the whole TUI is frozen for the duration of an audiobook build (**minutes to hours**). Worse,
  `subprocess.run` with no `timeout=` and **inherited stdin** — ffmpeg reads stdin by default and the parent's stdin
  is the terminal in raw mode — means a stuck ffmpeg wedges the application permanently **and steals the user's
  keystrokes**, with no cancellation path.
- Recommended correction: `await asyncio.to_thread(...)` the whole body and the pydub work; give the
  `subprocess.run` both `timeout=` and `stdin=subprocess.DEVNULL` (plus `-nostdin`). · Size: M · Confidence: verified

### P1 [D1] — The two remote TTS backends read credentials env-first (contradicting the settled ADR-012 amendment) and gate them on truthiness, so `<API_KEY_HERE>` and whitespace-padded keys reach the provider
- Where: `TTS/backends/openai.py:78-120` (env var read **first** at `:79`; `api_settings.openai.api_key` only fourth
  at `:96`); `TTS/backends/elevenlabs.py:28-34`; the gate is `TTS/base_backends.py:129-133 _validate_api_key`
  (`if not self.api_key`); the value goes on the wire at `elevenlabs.py:134` and `openai.py:172`.
- Evidence: with a scratch profile:
  ```
  API/elevenlabs_api_key          -> '<API_KEY_HERE>'   | resolve_provider_api_key -> None
  app_tts/ELEVENLABS_..._fallback -> '  padded-key  '   | resolve_provider_api_key -> 'padded-key'
  what elevenlabs.py __init__ ends with -> '<API_KEY_HERE>' | truthiness gate passes: True
  ```
  `config.resolve_provider_api_key` has 12 importers — **none in `TTS/`**.
  `backlog/decisions/012-provider-credential-settings-boundary.md:100-137` ("Amendment 2026-09-19 … TASK-32806.2")
  rules that *"the stored `api_key` outranks the environment variable it names"*, with the rationale that under
  environment-first *"a stale `OPENAI_API_KEY` exported in a shell profile silently outranked what the UI showed"*.
- Why it matters: (a) a user who sets an OpenAI key in Settings **still spends against a stale shell env var when
  using TTS, while Chat uses the configured one** — exactly the split the amendment closed; (b) a placeholder is
  sent as a credential and the user sees a provider 401 instead of "not configured".
- Size: S · **ADR: yes — non-adoption of ADR-012's 2026-09-19 amendment**, not a new decision · Confidence: verified
- Already covered: **no.** TASK-32806.1 is scoped to `config.get_api_key()` and its five named call sites; these
  backends do not call it. TASK-32806.2's implementation covers "the ~9 bridged chat providers" — **the TTS spend
  path is a separate accessor and was missed.**

### P1 [D1] — Higgs voice-profile backups are named with naive local time, so a DST fall-back overwrites a backup, and `restore_from_backup` then restores the *older* file
- Where: `TTS/backends/higgs_voice_manager.py:566` (`datetime.now().strftime("%Y%m%d_%H%M%S")`), `:573-576`
  (retention by **lexical** `sorted(glob(...))`), `:591-594` (`backups[-1]` = "most recent"). Sibling naive-local
  sites: `:426`, `chatterbox_voice_manager.py:373`.
- Evidence: `rg -n "datetime\.now\(\)|utcnow|strftime" tldw_chatbook/TTS/` → 7 hits, all naive-local **except**
  `profile_repository.py:3771`, which correctly does `.astimezone(UTC)`. Mechanism: during a DST fall-back hour,
  `20261101_013000` (DST) and `20261101_013000` (standard, one hour later in real time) **collide** — the copy
  overwrites — and non-colliding names in that hour sort in **reverse real-time order**, so `backups[-1]` is the
  older snapshot.
- Why it matters: the one recovery artifact the Higgs voice store has is silently destroyed **and** mis-selected, in
  the exact hour it is most likely to be needed after a bad edit. · Size: S · Confidence: verified (mechanism traced)
- Already covered: TASK-32803.5 is **Done**; these 6 production sites are non-adopters. *(Sixth slice to find the
  same gap.)*

### P1 [D1] — audio.cpp child-process diagnostics are Rich-escaped into a `markup=False` RichLog — and TASK-32802.1's sweep would make it *worse* here, not better
- Where: escape at `TTS/audio_cpp_supervisor.py:18` and `:250` (`_sanitize`); the escaped text is projected at
  `UI/Speech/audio_cpp_runtime_card.py` and written at `:683` into `RichLog(..., markup=False)` declared at `:603-608`.
- Evidence:
  ```
  s = 'audio.cpp: loaded [WARN] model [ok] path=[1,2]'
  rich.markup.escape(s)                    -> 'audio.cpp: loaded [WARN] model \[ok] path=[1,2]'
  Utils.input_validation.escape_markup(s)  -> 'audio.cpp: loaded \[WARN] model \[ok] path=\[1,2]'
  ```
  The sink is markup-**OFF**, so both escapes render their backslashes **literally**.
- Recommended correction: **delete the escape** at `:250` (the sink is markup-off; `_sanitize` already strips ANSI
  and control characters at `:244-245`). **Do NOT swap it for `escape_markup`:** that escaper is strictly more
  aggressive (3 backslashes vs 1 above) and would **triple** the visible corruption. · Size: S · Confidence: verified
- Already covered: **partially, and dangerously.** This is the shape of TASK-32802.**4** (markup-off
  double-escaping), which names only two `Chat/` sites. Meanwhile TASK-32802.**1** AC#2 says *"the correct escaper is
  used at every site that currently calls `rich.markup.escape` for a Textual surface"* — **applying that AC
  mechanically here produces a regression.** Flag to the .1/.4 owner that this file belongs to .4's list.

### P2 [D3] — The whole `TTS/utils/` package (512 lines) is dead in production, and its download helper can never succeed
- `TTS/utils/performance.py` (240) — `rg "get_performance_tracker|PerformanceTracker"` → **zero hits anywhere**
  including Tests. `TTS/utils/download_models.py` (272) — the only live references are
  `Tests/Architecture/test_recovery_entrypoints.py:27` (an ADR-126 fencing route) and
  `test_task2062_3_legacy_downloader_retirement.py:106`, which asserts production must **not** reference it.
  Separately, `download_models.py:112-117` compares `bytes_downloaded != expected_size` against **rounded** constants
  (`311_000_000` for "~311MB"), so the check **can never pass** and the file is always deleted — the module is dead
  *and* broken. · Size: S · Confidence: verified
- Already covered: **`TTS/utils/` is in none of TASK-32807's six sub-tasks.**

### P2 [D4] — Three sibling reference-audio validators enforce three different bound sets, and the repo's own bounded validator lives two files away in the same package, unused by both
- Bounded helper: `TTS/sample_audio_validation.py:21` (`MAX_PLAYABLE_AUDIO_BYTES = 8 MiB`,
  `_read_bounded_regular_file`, `validate_path`), consumed by `TTS/profile_service.py:70` and
  `UI/Screens/settings_speech_tts.py:46`. Re-rolls: `TTS/backends/voice_manager_base.py:171-201` (extension
  allowlist, **no size cap, no duration cap**) and `higgs_voice_manager.py:472-511` (300 s duration cap on the
  soundfile branch; **the 100 MB size cap and the extension allowlist sit only on the soundfile-unavailable `else`
  branch**, i.e. they do not apply in the common installed case).
- Why it matters: `ChatterboxVoiceManager` inherits the base, so it has **no size or duration bound at all**, and a
  user-picked reference file is then copied whole via a full in-memory read (`loose_voice_lifetime.py:314-316`).
- **Delta against task-32863 AC#2:** "make `HiggsVoiceProfileManager` adopt `VoiceManagerBase`" as written would
  **remove Higgs's 300 s duration cap** (the base has none) unless the base gains bounds first. The task file does
  not mention that; it should name the bounded validator as the adoption target. · Size: M · Confidence: verified

### P2 [D3] — task-32863's AC#1 is already satisfied on `origin/dev`; the task is stale as filed
- AC#1 is *"`chatterbox_isolated.py` deleted with evidence of zero references"*. `ls tldw_chatbook/TTS/backends/` at
  `3722a857480b` → **the file does not exist.** Tick it with this evidence and re-scope to AC#2 plus the
  bound-validator decision above. · Size: S · Confidence: verified

### P2 [D3] — The two largest TTS modules have no size-ratchet row, though both are larger than rows that are governed
- `TTS/profile_repository.py` — 6,304 lines, `class TTSProfileRepository` **4,880 lines / 118 methods**;
  `TTS/TTS_Generation.py` — 4,046, `class TTSService` **2,825 / 93**. Next tier: `profile_service.py` 3,678,
  `profile_schema.py` 2,481, `voice_bundle_service.py` 2,334, `adapters/audio_cpp.py` 2,096, `backends/kokoro.py` 2,043.
- Evidence: `_BUDGETS`' smallest row is `mcp_workbench.py: 6760` — **smaller in class terms and only 456 lines larger
  in file terms than `profile_repository.py`, which has no row.** **Baseline for the ratchet at this SHA:
  `5 failed, 9 passed`** (pre-existing reds: `console_chat_controller` 29529>29367, `mcp_workbench` 6774>6760,
  `personas_screen` 16449>16436, `console_transcript` 8399>8353, `app.py` slack 152>50) — i.e. **32809.1's re-pin is
  still outstanding.**
- Recommended correction: add both rows in the same commit that re-pins the 5 reds. **Do not open a split** — the
  profile-store cluster is the highest-invariant code in the slice. · Size: S (rows) / L (split) · Confidence: verified

### P2 [D4] — Five intra-TTS duplication clusters with no shared home; two sit on the migration/recovery path
1. `_sidecars_absent` — `profile_migration_recovery.py:172` ≡ `profile_migration_publication.py:179`
   (**byte-identical**), and `DB/private_sqlite.py:2773` is the same function over `_SIDECAR_SUFFIXES`. All three
   suffix tuples are `("-wal","-shm","-journal")` today. **Canonical home already exists** and
   `profile_migration_native.py:176,409` already imports from it. **Risk: a future 4th suffix landing in one place
   leaves an orphan sidecar the next open will trust.**
2. `_safe_failure` — `profile_migration_recovery.py:104` ≡ `profile_migration_publication.py:157`, byte-identical.
3. `open_directory`, `_maintenance_close_admission`, `_maintenance_resume` —
   `profile_reference_materialization.py:186/575/589` ≡ `voice_bundle_service.py:547/1578/1590`, all byte-identical.
4. `_checked_clone_native`/`_bundle_native` — same 4-line body, only the type and error class differ (a legitimate
   specialization; **documented, not a defect**).
5. The freeze family — `Chat/console_turn_context.py:155`, `TTS/playground_types.py:35`, `TTS/preferences.py:21`,
   `TTS/adapter_registry.py:92`, `TTS/effective_settings.py:182`. All five **semantically identical**; they differ
   only in name, docstring and line wrapping. · Size: S · Confidence: verified

### P2 [D1] — Neither TTS subprocess spawn uses `start_new_session`, so a grandchild survives the stop path
- `TTS/audio_cpp_supervisor.py:480-489` and `TTS/backends/chatterbox.py:284-291`; teardown reaches only the direct
  child. `chatterbox_process.py` runs `sys.executable` under torch, which forks DataLoader/multiprocessing workers.
- **Retired half:** the app *does* stop the supervisor at exit (`app.py:407,15062 await close_tts_resources()` →
  `TTS_Generation.py:4035` → `adapters/audio_cpp.py:1266 await supervisor.stop(...)`), so 32806.5's "nothing in
  unmount stops these" half does **not** apply here; only the process-group half does — and it is the prerequisite
  for the P0 fix. · Size: S · Confidence: verified for the flag; inferred for whether audio.cpp forks

### P2 [D4] — `_tts_provider_for_model` / `_tts_internal_model_id` are a 4th and 5th copy of a mapping whose canonical home runs an explicit copy pact they are absent from
- Canonical home: `TTS/legacy_request_builder.py:104-119`, whose module docstring (`:13-19`) runs a
  "TASK-1393 pact convention, greppable both ways" naming the one deliberately-different copy. The unlisted copies
  are `Media/local_media_reading_service.py:5210,5219` and `Audio_Services_Interop/local_audio_services_service.py:184,193`.
- **Answering the lead's question: there is no third copy inside `TTS/`.** Drift: higgs →
  `"local_higgs_default"` vs `"local_higgs_v2"` (**benign** — `legacy_bridge.py:89-90` accepts each); but **neither
  copy handles `"audio_cpp"`, which IS a shipped provider** (`TTS/provider_ids.py:7`), so an `audio_cpp` selection
  there falls through to provider `"openai"` with internal model id `"audio_cpp"` and **routes nowhere**.
- Size: S · Confidence: verified

### P3 [D1] — Audiobook metadata is interpolated into FFMETADATA without escaping
- `TTS/audio_service.py:360-362`, fed from `stts_events.py:731-739` with the user's title/artist/album/genre/
  description. FFMETADATA1 requires `=`, `;`, `#`, `\` and newline to be backslash-escaped; **none are.** A
  description containing `\n[CHAPTER]\nTIMEBASE=1/1000\n…` injects chapter records into the user's own M4B.
  Self-inflicted, local file, no privilege crossing. · Size: S · Confidence: inferred

### P3 [D4] — Four strict-JSON decoders in `TTS/` alone, with measurably different guarantees
| site | dup keys | `parse_constant` | number bounds | depth cap | canonical round-trip |
|---|---|---|---|---|---|
| `audio_cpp_contract.py:87-143` | ✔ | ✔ | ✔ | ✗ | ✗ |
| `audio_cpp_managed_config.py:213-223` | ✔ | ✔ | ✗ | ✗ | ✗ |
| `voice_bundle_codec.py:303-327` | ✔ | ✔ | ✗ | ✔ (≤4) | ✔ |
| `audio_cpp_artifact_catalog.py:358-368` | ✔ | **✗ not set** | ✗ | ✗ | ✗ |
**The drifted one is the artifact catalog, not `audio_cpp_contract.py`** — and it is **not reachable**: its only
input is a shipped in-package manifest, and every numeric field goes through `_positive_integer` requiring
`type(value) is int`. Defense-in-depth gap only. `voice_bundle_codec._strict_json` is the strongest of the four and
the natural donor for task-32855. · Size: S · Confidence: verified

### P3 [D3] — A `print()`-based demo CLI ships inside a production voice-manager module
- `higgs_voice_manager.py:620-678`. Also registered as an ADR-126 fencing route at
  `Tests/Architecture/test_recovery_entrypoints.py:28`, so deletion must update that list. · Size: S

## Candidate triage
**CONFIRMED:** `audio_service.py:412` no `timeout=` (**plus** no `stdin=DEVNULL`, **plus** on the event loop);
`kokoro.py:210` (**reframed** — `rg "fsync" kokoro.py` → **0 hits**, the site has *no* barrier of any kind, so
adopting the shared helper would still leave the parent unsynced); `audio_cpp_supervisor.py` markup escape
(**with a correction to the owner task**); the 4 strict-JSON decoders; `audio_cpp_artifact_catalog.py:136`
`_positive_limit` (duplication, no drift — it is the **strictest** form in its cluster).
**RETIRED:** `profile_repository.py:3162` `os.replace` — `operation.fsync_file()` at `:3157` precedes it and
`os.fsync(operation.descriptors["parent"])` at `:3181` follows, with `published`/`uncertain` tracking for the
"rename completed but wrapper reported an error" case. **Correct as written.**
Downloaded-artifact verification — **split**: audio.cpp artifacts are **retired**
(`audio_cpp_artifact_catalog.py:39-46` bounds manifest bytes/packages/files/paths, `:240-247` requires a 64-hex
sha256 and positive `size_bytes`, and `audio_cpp_guided_launch.py:295-309` requires absolute + `S_ISREG` + `X_OK`;
nothing ever chmods a downloaded file to +x); the Kokoro model is **confirmed** (P1).
`loguru_and_logging` — **zero real hits in either slice**: `backends/alltalk.py:5` uses `import logging` only to
`addFilter`/`removeFilter` a privacy filter on **httpx's** stdlib logger; same for `adapters/audio_cpp.py` and
`audio_cpp_supervisor.py:475` (asyncio's logger).
`except_exception_pass` ×34 — **retired as a class**, every S03 site read with context: 8 are the
`task.exception()` "observe-so-the-loop-does-not-warn" idiom, 4 are cleanup/seal paths that deliberately swallow
(`adapter_registry.py:862` marks the slot `unavailable=True` immediately after), 1 is optional JSON error-detail
extraction. **None on a data-write path.** Residual note, not filed: `TTS_Generation.py:3401 except BaseException:
pass` around `await asyncio.shield(ticket.completion)` absorbs a `CancelledError` delivered to the awaiting task.
`inline_path_check_no_pv` ×7 — **retired, these are *stronger* than `Utils/path_validation`**:
`loose_voice_lifetime.py:107-125` does `lexical_path` + `is_relative_to(root)` + per-component symlink rejection,
and every mutation goes through `_open_verified_parent` with `dir_fd`-relative ops.
`mutable_class_attr` ×2, `sys_path_mutation` ×1, `fetchall` ×1 — retired.
`re.compile` in bodies / dotted `get_cli_setting` / `run_worker(exclusive=True)` without `group=` — **zero hits.**
sqlite on the loop — **retired**: `profile_repository.py` routes every statement through a dedicated
`ThreadPoolExecutor(max_workers=1)` with `asyncio.to_thread` at the seam.
`voice_bundle_codec.py` zip handling — **retired, verified well-hardened** (40 MiB archive cap, 33 MiB uncompressed,
expansion ratio 100, per-member limits, member-name allowlist, `allowZip64=False`, **no path extraction at all**).
Shipped placeholder credentials reaching the wire — **retired with evidence**: `DEFAULT_APP_TTS_CONFIG` is merged
under the key `APP_TTS_CONFIG` while the backends read `get_cli_setting("app_tts", …)`; probed against a fresh
default profile, all three lookups → `None`. **A user-written `[app_tts]` section IS honoured** (verified) — which
is the P1 above, not this.
A 2026-07 plan doc's claim that `openai.py:96,105,114` "all three always return None" — **retired, already fixed.**
`strftime` — 1 confirmed defect (P1), 2 low, 1 retired. `legacy_markers` ×90 — retired as noise (`RETIRING` is a
lifecycle state name; the `legacy_*` modules are the intentional ADR-023 staged bridge).
`maintenance_drain` 5-copy cluster — **confirmed, already owned** by TASK-32808.11: AST-diffed,
`adapter_registry.py:194` ≡ `TTS_Generation.py:1127` byte-identical, the other three differ only in docstring and
error string. **Zero behavioural drift.** All five busy-poll at 50 Hz.
**`_maybe_await` — no definition or call site anywhere in `TTS/`.**

## D4 observations for repo-wide Phase 3
1. **`_sidecar_suffixes` ×3 — helper exists, ignored.** `DB/private_sqlite.py:675` is canonical and
   `profile_migration_native.py` already imports from it, so the import is free.
2. **The freeze family — no helper, 5 copies, zero drift.** The census put this in `dup_shape`, not `dup_verbatim` —
   **it belongs in `dup_verbatim` modulo the name.** Home: `Utils/`.
3. **`TTS/legacy_request_builder.py` is an existing canonical home with a documented copy pact, and two copies are
   missing from it** — with real drift (`audio_cpp` unhandled).
4. **Reference-audio validation ×3 — helper exists in the same package, ignored**, and the drift **is** a D1 because
   the outputs gate a whole-file read into memory.
5. **`maintenance_drain` — 20+ definitions repo-wide** (`rg` list in the slice transcript), not the 5 the census row
   suggests. Already owned by TASK-32808.11; flagging the arity.
6. **The two "materialization" services share three byte-identical methods** — this looks like **one un-extracted
   native-operation base class**, not five independent re-rolls; treat as a single extraction.

## Left UNVERIFIED
| Claim | Why | Command |
|---|---|---|
| `os.killpg` in `audio_player.py:556` actually fires in practice (needs a player surviving SIGKILL >0.5 s on Darwin) | cannot induce an uninterruptible-sleep player read-only; **the pgid identity itself is proven** | monkeypatch `self._current.process.wait` to raise `TimeoutExpired` twice, patch `os.killpg`, assert the recorded pgid `!= os.getpgrp()` |
| Whether the audio.cpp server binary forks children (decides whether the missing `start_new_session` there is exploitable) | requires launching the binary | `pgrep -P $(pgrep -f 'audio.cpp.*--config' \| head -1)` after a managed start |
| Whether `Tests/TTS/test_higgs_validation.py` / `test_chatterbox_validation.py` pin the *current unbounded* validation as a requirement (which would make the bound adoption a decision, not a fix) | mechanical-only files | run both, then read for size/duration assertions |
| That the escaped audio.cpp line visibly renders `\[` in the running TUI (escaper output and markup-off sink verified; not the composited pixels) | app must not be run | start a managed audio.cpp whose stderr contains `[ok]`, open the runtime card, read `#audio-cpp-diagnostics-lines` |
| Whether TASK-32806.5's PR #2723 already swept `start_new_session` into `TTS/`, or TASK-32802.1's #2717 touched `audio_cpp_supervisor.py:250` | branches absent from this worktree | `gh pr diff 2723 -- tldw_chatbook/TTS/` and `gh pr diff 2717 -- tldw_chatbook/TTS/audio_cpp_supervisor.py` |
| 7 mechanically-scanned files (`profile_service.py` 3,678, `profile_schema.py` 2,481, `request_admission.py`, `profile_migration_namespace.py`, `audio_cpp_recipes.py`, `audio_cpp_package_scanner.py`, `backends/higgs.py`) | budget. **The profile-store cluster that WAS read deeply is the highest-invariant code in either slice** (fd-relative `O_NOFOLLOW\|O_DIRECTORY` opens, file+parent fsync at every publication point, a 12-stage fault-injection checkpoint enum, `_OpaqueAuthority` refusing pickle/copy, an artifact byte ceiling, `published`/`uncertain` split after `os.replace`) — so **none** of the three data-loss shapes the brief asked about was found: no non-idempotent migration, no half-applying publication, no unrecoverable recovery. **That is a coverage statement about what was read, not a clean bill for these 7.** | `pytest Tests/TTS/ -q` as a baseline, then read `profile_service.py` and `profile_schema.py` against the journal state machine at `profile_migration_journal.py:33-48` |
