# Meetings — self voiceprint enrollment (remember my voice across meetings)

**Status:** design, awaiting user review
**Date:** 2026-09-06
**Backlog:** TASK-31826
**Builds on:** `Docs/superpowers/specs/2026-09-05-meeting-diarization-design.md` (phase 2, merged as #2456) and its follow-ups (#2471)

## 1. Goal

Let the local user enroll their own voice once so that every later meeting tags their speech with their display name automatically — including room mode and hybrid rooms, where the mic channel is not a clean "You" — with the voiceprint stored encrypted on the device, exportable on demand, and deletable in one action.

**In scope:** the self voiceprint store (encryption, export/import, merge, delete), worker-side matching, session/owner wiring (stable first match, override, Stop-pass match, learning offer), explicit enrollment, Settings/Meetings UI, config, degradation, tests.

**Out of scope (own designs):** enrolling other people's voices (third-party biometrics and consent); automatic sync of the voiceprint across devices; cross-device portability of per-meeting speaker-name maps.

## 2. Decisions locked during brainstorming

| Question | Decision |
|---|---|
| Whose voice | **Only the local user.** Third-party enrollment deferred. |
| How enrolled | **Both:** an explicit "Enroll my voice" step, and learning from accepted meetings. |
| Where it lives | **Local file, encrypted at rest, export/import on demand.** No automatic sync. |
| On match | **Auto-name with a visible marker and easy override.** |
| Architecture | **Approach A:** matching inside the diarizer worker — the voiceprint goes in, verdicts come out; embeddings otherwise stay in the worker. |

## 3. Architecture

### 3.1 `Audio/voiceprint.py` — the store (new, pure Python, no torch)

Owns the single self voiceprint.

- **Record** (`format_version: 1`): `model_id` (the embedding model name plus its weights revision when the loader exposes one, else `@unpinned` — unpinned weights can drift silently, which the best-similarity diagnostic would reveal), `centroid` (unit-normalised floats), `sample_count` (effective samples), `meetings_contributed`, `created_at`, `updated_at`, `threshold_used` (the match threshold in force when last written, for diagnostics).
- **File:** `<user data dir>/voiceprint.json` — the user data dir ROOT, never the configurable `recordings_dir` (which may be a synced folder). It is a new profile-owned path, so it must be registered with the profile-owned path census that `./scripts/preflight.sh` checks. An envelope: `format_version`, `mode` (`keyring` | `keyfile` | `passphrase`), encrypted payload. Writes are atomic (temp file + rename).
- **Keys:** reuses `Utils/config_encryption.ConfigEncryption`'s value primitives with the store's OWN random key — deliberately not the config-encryption password (its availability depends on the user unlocking an encrypted config). Mode order: keyring (the `keyring` dependency), else an owner-only key file in the user data dir (a key file with broader permissions is refused, as ssh does). The mode in use is reported in Settings. **Keychain prompt timing:** on macOS the first keyring access by a terminal app can raise a system Keychain prompt, so the key is CREATED during the user-initiated enrollment flow (where a prompt is expected) and READ at meeting Start on a worker thread with a short timeout; a blocked or failed read never delays Start — matching is off for that meeting with the reason "keyring locked".
- **Operations:** `load() -> Voiceprint | None` (reports `needs_reenrollment` on model-id mismatch and `cannot_decrypt` on key/corruption problems without raising into callers), `save(record)`, `merge_sample(centroid, weight)` (unit-normalise, count-weighted running mean, per-meeting weight capped), `delete()` (file only; the key stays and holds nothing; a meeting already running keeps its in-memory copy until it ends, documented), `export(path, passphrase)` (re-encrypts under a typed passphrase via the existing PBKDF2 derivation with a fresh salt; empty passphrase refused), `import_(path, passphrase, replace: bool)` (reads only `passphrase` mode; merge allowed only when `model_id` matches, otherwise replace after confirmation).
- The store is injectable (key provider, clock, paths) so tests never touch the real keyring.

### 3.2 Worker (`Audio/diarizer_worker.py`) — two ops, two reply fields

- `enroll {"vector": [...], "threshold": t, "min_seconds": s}`: hold the voiceprint in memory for this process. The worker never persists anything.
- `assign` reply gains `"self": true` when the assigned cluster's centroid is within `threshold` cosine distance of the voiceprint AND the cluster has accumulated at least `min_seconds` of embedded audio (a single short window cannot match).
- `diarize` reply gains `"self": "<live id>"` for the batch cluster nearest the voiceprint within the threshold (or absent), so a meeting whose worker never warmed up live still gets a match at Stop.
- `export_centroid {"id": "<live id>"}` → the requested cluster's BATCH centroid after `diarize` (falls back to the live centroid before it), unit-normalised, with its accumulated seconds.
- `enroll_from_pcm` (length-prefixed PCM) → a unit-normalised centroid and the seconds embedded (explicit enrollment). The PCM is never written by the worker.

The match threshold is distinct from and stricter than the clustering threshold.

### 3.3 Backend (`Audio/diarizer_local.py`)

`SpeechBrainDiarizer` accepts an optional voiceprint at construction and sends `enroll` itself immediately after READY (the session never observes readiness), and re-sends it after its single crash restart. `assign` keeps its protocol return (`str | None`); the backend records the FIRST cluster id the worker flags as `self` in a read-only attribute `self_cluster_id: str | None` (later flags only increment `self_candidates_seen`), which the session reads after each `assign` — so stable-first-match is enforced once, in the backend. It forwards `export_centroid` / `enroll_from_pcm` and never logs the vector.

### 3.4 Session and owner

- **When matching runs.** Room mode always; call mode only when `diarize_mic_channel` is on. In plain call mode the mic channel already is the user, so remote clusters are never compared against the voiceprint (this removes the largest false-positive path).
- **Stable first match.** The backend fixes the first cluster the worker flags as `self` (`self_cluster_id`, §3.3); the session reads it after each `assign`, and the first time it sees a value it names that cluster with the user's display name, records `matched_self: "<cluster id>"` in `meeting.json`, and re-emits it like any speaker update. Later flags are counted by the backend, never applied. Stop reconciliation treats it as a named cluster.
- **Override.** Renaming the matched cluster sets `matched_self_overridden: true` and clears the marker; a rename to a name equal to the display name is not an override. Pin-forwarding applies as to any named cluster.
- **Stop-pass match.** The `diarize` reply's `self` id is applied only when no live match was recorded.
- **Learning offer** (owner, app-owned, non-blocking, never delays finalise or ingest). A qualifying clean sample is: the matched cluster when not overridden; or, in plain call mode, the mic channel — offered with the explicit question "Was it only you on the mic?". At most one offer per meeting; unanswered offers lapse; a "don't ask again" flips `voice_learn_offer` off. On accept: for a matched cluster the owner fetches its batch centroid via `export_centroid`; in plain call mode the mic channel was never diarized (no cluster exists), so the owner instead sends the recorded `you.wav` through `enroll_from_pcm` (the same op explicit enrollment uses) and takes the returned centroid. Either way the result goes to `merge_sample` with the per-meeting cap. On decline nothing is kept.
- **Explicit enrollment** (owner): records ~30 s from the mic through the existing capture path, refused while a meeting is active or any other capture is running (Console guard copy), with a visible countdown and cancel; spawns the diarizer worker if needed (warm-up indicator); sends `enroll_from_pcm`; stores the centroid stamped with the model id. The sample lives in memory only.

### 3.5 Screens

- **Meetings:** the matched row and legend entry carry a small marker; rail status line "Voice match: on" / "off (no voiceprint)" / "off (needs re-enrollment)" / "off (plain call mode)"; the post-Stop offer; an "Enroll my voice" action.
- **Settings › Meetings:** Delete, Export…, Import…, the `voice_match` toggle, the learning-offer toggle, the encryption mode in use, and the privacy-neutral diagnostic "best similarity seen for you in the last meeting" (a number stored in the encrypted record as `last_best_similarity`, no vectors).
- All UI is best-effort and `is_mounted`-guarded; new TCSS rules are class-keyed (Perf Guard CSS ratchet).

## 4. Data flow

1. **Start.** Owner → store `load()`. Absent / off / plain call mode → nothing. Else the vector is handed to the backend, which sends `enroll` after READY.
2. **Live.** `assign` replies carry `self`; the session applies stable-first-match and persists `matched_self`; renames override.
3. **Stop.** Batch pass as today; `self` from `diarize` applied if no live match; then the learning offer (rules above); on accept `export_centroid` → `merge_sample`.
4. **Enroll.** Mic sample → `enroll_from_pcm` → `save`.
5. **Settings.** Delete / Export (passphrase) / Import (passphrase; merge only on matching model id).

## 5. Config (under `[meetings]`)

| Key | Default | Meaning |
|---|---|---|
| `voice_match` | `true` | Use an enrolled voiceprint to tag the user; no effect until one exists. |
| `voice_match_threshold` | `0.2` | Cosine-distance ceiling for a `self` match. A documented STARTING value; unvalidated on this host (no real embeddings) — calibrate with the gated real test before advertising. |
| `voice_match_min_seconds` | `4` | Embedded audio a cluster needs before it can be declared `self`. |
| `voice_learn_offer` | `true` | Offer to remember the voice after a qualifying meeting (at most once per meeting). |

## 6. Error handling and degradation (all best-effort)

- Store unreadable (bad key, corrupt file, model-id mismatch): matching off for the meeting; the rail and Settings show the static reason; the meeting proceeds.
- Keyring unavailable: fall through to the key file; the mode is reported; Start never fails over it.
- Worker never ready / crashed: no live match; the Stop-pass match still runs if the worker recovers; otherwise none.
- Enrollment capture refused while another capture is active; a failed embed discards the sample and reports the exception type only.
- Learning merge failure: the offer reports failure; the previous voiceprint stays intact (atomic write).
- Privacy: the vector never appears in logs, `meeting.json`, transcripts, or the exported meeting folder; logs carry mode names, counts, and exception types only. The vector crosses only the local pipe to the worker, on-device.

## 7. Testing

- **Store, pure unit:** envelope round-trip under both key modes with an injected key provider; atomic-write crash survival; model-id mismatch → `needs_reenrollment`; merge math (normalisation, weighting, per-meeting cap); delete; export/import with passphrase, empty passphrase refused, merge-vs-replace and the model-id gate; key-file permission check.
- **Worker, torch-free via `serve()`:** `enroll` then `assign` with a fake embed — `self` only after `min_seconds` and only within the threshold; `export_centroid`; `enroll_from_pcm` normalisation; the `diarize` reply's `self`.
- **Backend:** with the fake subprocess, `enroll` is sent after READY and re-sent after the single restart; the vector never appears in any log line.
- **Session, fake diarizer:** stable-first-match (second flag counted, not applied); marker + `matched_self` persisted; override semantics incl. the display-name exception; Stop-pass `self` only without a live match; plain call mode never matches; learning offer at most once and only for a clean sample.
- **Owner and screens:** rail status per state; app-owned non-blocking offer that lapses; "don't ask again"; enrollment refuses during another capture, shows the countdown, keeps the sample in memory only; Settings actions call the store.
- **Invariants:** boot imports no torch/diarizer module; UI-ready census unchanged; no vector/name/path in logs; class-keyed TCSS; diagnostic inventory re-pinned after review.
- **Gated real test** (`diarization` extra): ECAPA embeddings of pure tones do not behave like speech, so the fixture is TTS speech — two distinct voices generated with the platform's TTS (`say -v` on macOS; skipped where no TTS is available). Enroll from one voice, run a two-voice meeting, assert the user's cluster matches and the other does not, and record the best similarity for threshold calibration.

## 8. File map (for planning)

- Create `tldw_chatbook/Audio/voiceprint.py` (store) and tests `Tests/Audio/test_voiceprint_store.py`.
- Modify `tldw_chatbook/Audio/diarizer_worker.py` (ops + reply fields), `diarizer_local.py` (enroll after READY / restart, side channel, forwarding), `diarizer_cluster.py` (per-cluster accumulated seconds if not already tracked).
- Modify `tldw_chatbook/Audio/meeting_session.py` (self application, override, `matched_self` fields, Stop-pass self), `meeting_owner.py` (load at Start, learning offer, explicit enrollment, settings fields).
- Modify `tldw_chatbook/UI/Screens/meetings_screen.py` (marker, rail line, offer, Enroll action) and the Settings › Meetings surface (Delete/Export/Import/toggles/diagnostic).
- Modify `tldw_chatbook/config.py` (four keys), `Docs/User_Guide/meetings.md` (enrollment, matching, export/import, privacy).

## 9. Follow-ups (not in this design)

1. Third-party voice enrollment with per-person consent records and deletion.
2. Automatic cross-device sync of the voiceprint (needs its own security review).
3. Cross-device portability of per-meeting speaker-name maps (store the map in the synced DB).
4. Threshold auto-calibration from accepted meetings.
