# Meetings: torch-free ONNX diarizer backend — design

**Date:** 2026-09-07. **Backlog:** TASK-31827 (first sub-project; the server streaming
backend and the MOSS post-meeting re-transcription are separate follow-on specs).
**Research:** `Docs/Design/2026-09-07-self-hosted-diarization-options.md`.
**Builds on:** the phase-2 diarization design (`2026-09-05-meeting-diarization-design.md`)
and the voiceprint design (`2026-09-06-meeting-voiceprint-design.md`).

## 1. Goal and decisions

Give the **base install** live speaker labels and a Stop pass without the 2 GB torch
extra, by adding a second engine to the existing diarizer worker built on sherpa-onnx
(Apache-2.0, ONNX runtime, wheels for Linux/macOS/Windows incl. arm64). The `Diarizer`
seam, the online clusterer, reconciliation, the voiceprint feature and the worker protocol
are reused unchanged.

Decisions taken in brainstorming:

| Question | Decision |
|---|---|
| Which TASK-31827 piece first | The torch-free local backend. |
| Packaging and default | sherpa-onnx (and numpy) become base dependencies; ONNX is the default engine **if the bake-off passes**; the SpeechBrain path stays selectable. |
| Bake-off | A measured go/no-go gates the default flip (§7). |
| Stop pass engine | sherpa-onnx offline diarization (pyannote segmentation 3.0 + the same embedder), reconciled onto live ids. |
| Voiceprints across engines | One record; a different engine reads as "needs re-enrollment"; Enroll replaces. |

**Out of scope:** the server backend and MOSS (own specs); Library ingest-time diarization
(still torch; follow-up); CUDA providers for sherpa-onnx; a per-meeting engine picker.

## 2. Engine layout and the worker flag

- `Audio/diarizer_worker.py` keeps `serve()`, the framing checks and the wire protocol
  unchanged. `main()` gains `--engine {speechbrain,onnx}` (default `speechbrain`, so an old
  spawn command still works). Each engine's loader returns the triple the loop already
  consumes: `embed(pcm, sr) -> vector`, `batch(...) -> (segments, centroids_by_live_id)`,
  and `MODEL_ID`. `serve()` never learns which engine runs; `sr` is now passed to `embed`
  so the 16 kHz assumption is explicit.
- Engines live in their own modules: `Audio/diarizer_engine_speechbrain.py` (today's
  `_load_encoder`, `_embed`, `_batch`, moved verbatim) and `Audio/diarizer_engine_onnx.py`
  (new). Both modules import their packages **only inside the loader**, so they import
  cleanly without torch or sherpa-onnx (tests inject fakes through `sys.modules`).
- Reconciliation splits in two: `_cluster_windows(...)` (SpeechBrain-only: 1.5 s windows +
  sklearn agglomerative) and a shared `_map_final_clusters(final_clusters, live_centroids,
  threshold, start_id, out_centroids) -> mapping` holding today's threshold-gated matching,
  minting past `start_id`, and seconds-weighted centroid folds. The ONNX engine calls only
  the shared half and never imports sklearn.
- ONNX live path: `embed` feeds the window (float32 mono at `sr`) to a
  `SpeakerEmbeddingExtractor` stream and returns the unit-normalised vector. Provider `cpu`;
  2 threads for the live extractor, 4 for the Stop pass (constants, no config key).
- App side: `SpeechBrainDiarizer` becomes `LocalDiarizer(engine=..., ...)` with a
  `SpeechBrainDiarizer` alias kept for existing imports and tests. It adds the engine flag
  to the spawn argv. A module-level `model_id_for(engine)` in `diarizer_local.py` gives the
  owner the model id **before** any worker exists (it needs it at Start for the voiceprint
  check); the owner stops reading `diarizer_worker.MODEL_ID`.

## 3. Model acquisition and placement

- **Files.** Two ONNX assets from the sherpa-onnx GitHub release tags: the pyannote
  segmentation 3.0 model (5.7 MB float, 1.5 MB int8) for the Stop pass, and one speaker
  embedder for both paths. Bake-off candidates: NeMo TitaNet-small (English, fastest
  published RTF), 3D-Speaker ERes2Net-base, WeSpeaker ResNet34 (VoxCeleb), each 25–30 MB.
  After the bake-off one is the default; the others stay selectable via
  `[meetings] onnx_embedder`, whose accepted values are exactly the manifest's embedder
  keys (validated like `diarizer_backend`). Total download 30–40 MB; the rail copy uses the
  manifest's real total.
- **Manifest.** One constant in the ONNX engine module lists per asset: release URL,
  SHA-256, size, licence. It is the only place a model is named; changing the embedder is a
  one-line change that also changes the model id (§6).
- **Placement.** `get_user_data_dir()/models/diarization/onnx/<file>`. (The SpeechBrain
  path's cwd-relative `pretrained_models/` folder is left as is.) Air-gapped installs set
  `[meetings] onnx_models_dir` to pre-placed files with the same names; the user guide lists
  names, URLs and hashes.
- **Downloader.** httpx (streaming, `trust_env` proxies). Initial URL on `github.com`;
  redirects followed only to `*.githubusercontent.com` or back to a host already in the
  initial allowlist (`github.com` — GitHub's release URLs do bounce between the two);
  anything else refused. Temp name
  then atomic rename; SHA-256 verified when a download completes and again when the worker
  loads a file. No network at import, in `prepare()`, or on screen open.
- **When.** The fetch is the first step of the backend's own asynchronous warm-up
  (fetch → spawn → watch READY). Recording starts the instant the user presses Start;
  labels are coarse until READY, exactly as today. The download has its own 10 min budget,
  separate from the 120 s READY timeout; on failure the backend marks itself coarse with
  reason `models_unavailable`. Explicit enrollment reuses the same path through its
  existing progress callback. The rail's per-second tick reads a status string from the
  owner ("downloading 12 / 35 MB", "warming up", "ready").
- Concurrent fetches (a retried enrollment during a meeting's warm-up) are harmless: unique
  temp names, atomic rename, identical content, hash-checked on load.

## 4. Backend selection and availability

- **Config.** `[meetings] diarizer_backend ∈ {auto, onnx, speechbrain, server}`; the legacy
  `local` maps to `auto`; `server` stays reserved ("not available yet"). A pydantic
  validator rejects anything else with the accepted values in the error. New default: `auto`.
- **`AUTO_ORDER`** (owner module constant) is `("onnx", "speechbrain")` if the bake-off
  passes, else `("speechbrain", "onnx")`.
- **Resolution.** `resolve_engine(settings) -> str | None` walks the order and returns the
  first engine whose packages are importable, by `find_spec` only: `onnx` needs
  `sherpa_onnx` and `numpy`; `speechbrain` needs torch, torchaudio, speechbrain,
  scikit-learn. An explicit choice skips the walk and reports its missing packages. Model
  files are not part of availability (fetched at Start). The owner never imports an engine.
- **No runtime cross-engine fallback.** A package that exists but fails to import inside
  the worker takes the existing `ERROR load` → coarse path (`backend unavailable`); the user
  fixes the install or pins the config.
- **`PrepareResult`** gains `diarizer_engine: str | None`, `diarizer_missing: tuple[str, ...]`,
  `diarizer_models_ready: bool` (presence + byte size only; hashes are checked on download
  and on load). `live_diarization_active` now means "live labels on and an engine resolved".
- **Rail copy:** "Live speaker labels: on (ONNX)", "on (ONNX, ~N MB of models fetched at
  Start)" with N from the manifest total, "on (SpeechBrain)", "off (install the diarization
  extra)", "off (missing: sherpa-onnx)".
- **Stamping.** The owner records `MeetingMeta.diarizer_engine` and the engine's model id in
  `meeting.json` (back-filled as `None` for older folders).
- **Readouts.** `Utils/optional_deps.py` gains a `diarization_onnx` feature (sherpa-onnx,
  numpy) beside the torch one so the first-run wizard and readiness lines stay truthful.

## 5. The ONNX Stop pass and reconciliation

- **Input.** `batch` reads the requested span of the meeting's own 16 kHz mono PCM16 WAV
  with `wave` + numpy (no torchaudio), converts to float32, and runs
  `OfflineSpeakerDiarization` (segmentation model + the live embedder) with the manifest's
  per-embedder clustering threshold, auto speaker count, `min_duration_on = 0.3 s`,
  `min_duration_off = 0.5 s`. More clusters than `max_speakers` → the smallest fold into
  their nearest centroid (mirrors the live cap).
- **Centroids.** The pipeline does not expose its internal embeddings, so for each cluster
  the engine embeds its longest segments separately until 30 s is covered and averages the
  unit vectors (the same distribution the live centroids have), keeping the cluster's total
  seconds. Cost is bounded by speakers × 30 s.
- **Reconciliation.** `(cluster, centroid, seconds)` triples go through
  `_map_final_clusters`; segments come back relabelled with live ids, sorted by start,
  overlaps preserved — the same shape the SpeechBrain path returns.
- **Overlap rule (session, both engines).** The Stop overlay picks the batch segment with the
  largest time overlap with the transcript segment; midpoint containment is the tiebreak.
  This replaces "first segment containing the midpoint".
- **Budget.** The Stop pass gets ~1 s per second of audio, clamped to 60–600 s, so a
  60-minute meeting needs RTF ≤ 0.17. sherpa-onnx publishes 0.11 (TitaNet-small) and
  0.24–0.30 (ERes2Net); the bake-off gate is RTF ≤ 0.15 (§7), with the int8 segmentation
  model as the first lever (its DER cost is measured too). Over budget → skipped with the
  existing "kept near-live labels" footer.
- **Memory.** A 60-minute span is ~230 MB as float32 (90 min ≈ 350 MB) inside the worker
  subprocess — same order as the torchaudio path; it is never read into the app process.
- **First run.** If Stop arrives while the first download is still running, the Stop pass
  waits at most the READY timeout and then skips with the footer. Expected, documented.
- **Voiceprint at Stop.** The worker's Stop-pass `self` verdict (distance **and**
  `min_seconds`), `export_centroid` and the learning offer read the folded centroids and
  seconds as today; nothing in the owner changes.

## 6. Voiceprint and model ids

- `model_id_for("speechbrain")` = `speechbrain/spkrec-ecapa-voxceleb@unpinned` (unchanged;
  existing voiceprints stay valid). `model_id_for("onnx")` =
  `sherpa-onnx/<embedder file name>@<first 12 hex of its manifest SHA-256>`; the
  segmentation model is not part of the id.
- One record, re-enroll on switch: the owner passes the resolved engine's id as
  `expected_model_id` at Start; a mismatch reads `needs_reenrollment`; Enroll replaces.
  Switching back re-enrolls again. No store format change.
- **`auto` is not sticky.** Installing or removing the torch extra can change the resolved
  engine under `auto` and therefore trigger re-enrollment; the copy explains it, and pinning
  `diarizer_backend` avoids it. (A sticky rule was rejected: the stored id is inside the
  encrypted payload and reading it in `prepare()` would need a key access on screen open.)
- Enrollment and learning always use the resolved engine, so persisted centroids are in the
  active vector space; `merge_sample` refuses a cross-engine merge by comparing model ids
  (the store's existing check if it already covers merges, added there if today it only
  guards `import_`).
- **Learning-offer gate.** An offer is made only when the Start-time load (cached, no extra
  key read) reported the stored id equal to the active engine's id.
- Import/export unchanged; a foreign-engine file offers Replace only.
- **Copy.** Re-enrollment reason gains one static sentence when both ids are known engines:
  "Voiceprint was recorded with SpeechBrain; the active engine is ONNX." Names map from the
  id prefix (`speechbrain/` → SpeechBrain, `sherpa-onnx/` → ONNX, else "a different
  engine"); never the raw id.

## 7. The bake-off and its go/no-go

- **Measured per candidate** (TitaNet-small, ERes2Net-base, WeSpeaker ResNet34; float and
  int8 segmentation), with ECAPA through the same harness as baseline: Stop-pass DER;
  live-path purity/coverage of the online clusterer fed 3 s windows; Stop-pass RTF and peak
  worker memory; per-window embed latency; self-match separation (enrolled voice vs itself
  vs others), which also yields the recommended `voice_match_threshold` per embedder.
- **Corpus.** A fixed, listed subset of the VoxConverse dev set (CC-BY-4.0, RTTM labels,
  ~20 files) plus 3–5 AMI meetings (CC-BY-4.0, headset mix) for meeting acoustics; the
  `say`-synthesised voices only for the reproducible pass/fail the gated test already uses.
- **Machines.** Primary: an Apple M-series laptop. Secondary (x86 proxy): a GitHub Ubuntu
  runner via a manual `workflow_dispatch` job that uploads the report as an artifact.
- **Harness.** `Tests/Audio/bakeoff/` (never collected by CI): drives the real worker through
  both engines; scores with a small in-repo DER implementation (0.25 s collar) that has its
  own unit test against hand-computed values; writes the report to
  `Docs/STT_Evaluation/task-31827/` with corpus list, machine specs, sherpa-onnx version,
  model hashes and commit SHA.
- **Go/no-go for ONNX as default** — all must hold for the chosen embedder: DER within
  2 points absolute of ECAPA; live purity within 3 points; RTF ≤ 0.15 on both machines;
  per-window embed latency ≤ 150 ms (M-series) / ≤ 300 ms (runner); self-match separation
  within 0.05 of ECAPA's, measured on VoxConverse speakers. Pass: `AUTO_ORDER` ONNX-first,
  manifest pins embedder, threshold and licence. Fail: ONNX ships as the base-install
  fallback, `AUTO_ORDER` stays SpeechBrain-first, and the report names the failed gate.
- **Order.** Engine + downloader land first behind the flag (not default); the bake-off runs
  on that code; the default flip and rail copy follow. The outcome is recorded in §10.

## 8. Degradation and privacy

| Failure | Behaviour | Reason |
|---|---|---|
| sherpa-onnx not importable | `auto` walks on (SpeechBrain, then energy labels) | explicit choice only: "off (missing: sherpa-onnx)" |
| Download fails / over 10 min | backend coarse; recording unaffected; retried next Start | `models_unavailable` |
| Hash mismatch (downloaded) | discarded, re-fetched once, then coarse | `models_unavailable` (log: "hash", type only) |
| Hash mismatch (pre-placed, air-gapped) | refused, no retry | `models_unavailable` |
| Worker import / model load error | existing `ERROR load` → coarse | `backend unavailable` |
| Stop pass over budget | skipped, near-live labels kept | existing footer |
| Wrong-dimension voiceprint | worker guard → matching off | existing "needs re-enrollment" |

No new exception reaches the UI; the owner stays Textual-free and never imports an engine.

Privacy: all audio stays on the device; the only network traffic is the model fetch from
GitHub (no user data; refused off the allowlist; the user guide says the first Start
fetches models). Logs carry engine names, reason strings and exception types — never paths
(the models directory is logged only through the existing redaction), vectors or audio.
The manifest's licence field feeds an attribution line in the user guide (CC-BY-4.0
requires it for TitaNet).

Packaging: sherpa-onnx in the base install means the frozen app must bundle its native
library — a step for TASK-31747, not settled by the wheel alone.

## 9. Tests

- Worker loop: existing torch-free `serve()` tests untouched.
- ONNX engine (fake `sherpa_onnx` via `sys.modules`): WAV span read, per-cluster centroid
  averaging, cap fold, `_map_final_clusters`, overlap preserved; loader with the fake never
  imports torch (subprocess probe, `--engine onnx`).
- Downloader (local HTTP stub): temp-and-rename, hash verify/refuse, redirect allowlist,
  proxy passthrough, air-gapped directory, budget timeout.
- Owner: `resolve_engine` for every config value × installed-package combination (fake
  `find_spec`); `PrepareResult` fields; `meeting.json` stamping and back-fill; the
  learning-offer model-id gate; switch copy; `optional_deps` entry.
- Session: the largest-overlap rule, incl. a case where midpoint and largest overlap disagree.
- Backend: warm-up that downloads then spawns with the status sequence; download failure →
  coarse; READY budget unaffected by the fetch.
- Screen: rail copy per engine state; download progress line; census and CSS ratchet flat.
- Opt-in real tests: the gated worker and voiceprint tests gain an `--engine onnx` run; the
  bake-off harness is separate.
- Boot invariants: sherpa-onnx and numpy absent from the owner/session/screen import graphs.

## 10. Recorded deviations and outcomes

### Bake-off outcome — FAIL; ONNX ships as the fallback, not the default

Measured on an Apple M5 Max (18 cores, macOS 26.5.2, Python 3.12.11, sherpa-onnx 1.13.7)
over 24 files / 188 minutes (20 VoxConverse dev + 4 AMI dev), 0.25 s collar, 3 s live
window, `max_speakers` 8. Full report and raw `results.json`:
`Docs/STT_Evaluation/task-31827/report.md`.

Baseline (SpeechBrain/ECAPA, same harness): DER 0.371, live purity 0.941, RTF 0.015, embed
latency 27.6 ms, self-match separation 0.734. The §7 gates therefore are: DER ≤ 0.391,
purity ≥ 0.911, RTF ≤ 0.15, latency ≤ 150 ms (this machine), separation ≥ 0.684.

| embedder | DER | live purity | RTF | embed latency | separation | verdict |
|---|---|---|---|---|---|---|
| titanet_small | 0.143 ✓ | 1.000 ✓ | 0.038 ✓ | 11.8 ms ✓ | **0.640 ✗** | FAIL (separation) |
| eres2net_en | 0.125 ✓ | 1.000 ✓ | 0.082 ✓ | 36.7 ms ✓ | **0.631 ✗** | FAIL (separation) |
| wespeaker_resnet34 | 0.205 ✓ | **0.888 ✗** | 0.085 ✓ | 36.9 ms ✓ | **0.241 ✗** | FAIL (purity, separation) |
| campplus_en | 0.371 ✓ | **0.847 ✗** | 0.043 ✓ | 12.9 ms ✓ | **0.515 ✗** | FAIL (purity, separation) |

**The failed gate is self-match separation, for every candidate** — the best, titanet_small
at cluster 0.95, is 0.094 short against a 0.05 allowance. Live purity additionally fails for
`wespeaker_resnet34` and `campplus_en`. DER, RTF and latency pass everywhere, three of them
by a wide margin (ONNX's best DER is 3× better than ECAPA's).

Result, per §7's fail branch: **`AUTO_ORDER` stays `("speechbrain", "onnx")`** — ONNX is the
base-install engine when the torch extra is absent, and the alternative when it is present.
`DEFAULT_EMBEDDER` stays `titanet_small` (best overall of the four). The measured thresholds
are pinned anyway, since an explicit `diarizer_backend = "onnx"` should run at them:
`CLUSTER_THRESHOLD` = titanet_small 0.95 / eres2net_en 0.90 / wespeaker_resnet34 0.60 /
campplus_en 0.80; `LIVE_THRESHOLD` = 0.45 / 0.45 / 0.10 / 0.15. The shipped 0.5 / 0.25 were
wrong for all four (0.25 live on titanet_small minted 8 clusters inside 30 s; mean live
cluster-count error +4.67 at 0.25 against +0.54 at 0.45).

One qualification, which does not change the verdict as the gate is written: separation
compares an *absolute* cosine margin across two different embedding spaces. titanet_small's
self-similarity is higher than ECAPA's (0.933 vs 0.847); its margin is narrower only because
its other-speaker similarities also sit higher (0.303 vs 0.107), which is why the harness
reports a per-embedder recommended `voice_match_threshold` (0.377 for titanet_small, 0.526
for ECAPA) instead of one constant. A reviewer who decides an absolute cross-space margin is
the wrong test can flip the default with a **one-line change** to `AUTO_ORDER` in
`meeting_owner.py` plus the corresponding passage in `Docs/User_Guide/meetings.md`; nothing
else in the code branches on it.

Not measured: the x86 runner half. The `workflow_dispatch` job is landed but **was not
dispatched for this PR**, so "RTF ≤ 0.15 on both machines" and the 300 ms runner latency
ceiling are unverified. Both pass on the M-series with 2–4× of headroom, and the two gates
that actually fail are machine-independent. int8 segmentation was measured separately and is
worse-or-equal on DER and slower on all four embedders, so it stays out of `model_paths`.

- Library ingest diarization remains torch-based (follow-up to file).
- CUDA providers for sherpa-onnx are not wired (CPU only in this design).

## 11. Files

- Modify `tldw_chatbook/Audio/diarizer_worker.py` (flag, `sr` to `embed`, shared map/mint).
- Create `tldw_chatbook/Audio/diarizer_engine_speechbrain.py`, `diarizer_engine_onnx.py`
  (manifest, downloader, embed, batch).
- Modify `tldw_chatbook/Audio/diarizer_local.py` (`LocalDiarizer`, `model_id_for`, warm-up
  fetch, status string), `meeting_owner.py` (config values, `resolve_engine`, `AUTO_ORDER`,
  `PrepareResult` fields, stamping, offer gate, copy), `meeting_session.py` (overlap rule,
  `diarizer_engine`), `UI/Screens/meetings_screen.py` (rail copy, progress),
  `Utils/optional_deps.py`, `config.py` (keys + template), `pyproject.toml` (base deps).
- Create `Tests/Audio/bakeoff/` + `Docs/STT_Evaluation/task-31827/`; extend the gated tests,
  the import-safety tests and `Docs/User_Guide/meetings.md`.
