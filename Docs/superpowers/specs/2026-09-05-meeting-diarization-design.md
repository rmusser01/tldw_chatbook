# Meetings phase 2 — near-live per-speaker diarization with a local option and speaker renaming

**Status:** design, awaiting user review
**Date:** 2026-09-05
**Backlog:** TASK-31589 (phase 2 live speaker labels); voiceprint enrollment to be filed as a separate exploration
**Builds on:** `Docs/superpowers/specs/2026-09-04-meeting-transcription-design.md` (phase 1, merged as PR #2415)

## 1. Goal

Give a meeting real per-speaker labels instead of only "You / Others", produced
near-live during the meeting and finalized on Stop, with a cross-platform
**local** diarizer as the default and a server/MOSS backend reserved behind the
same seam. Let a user rename speakers to real people, both during the meeting
and afterward on the Library item, with names that survive relabeling.

**In scope:** the local near-live diarizer, the online clusterer, the Stop
reconciliation, the speaker-identity/name model, live and after-the-fact
renaming, config, degradation, and tests.

**Out of scope (own designs / backlog):** the MOSS/server backend
implementation; cross-meeting voiceprint enrollment ("remember this voice as
Alice"); Windows loopback verification (inherited from phase 1).

## 2. Decisions locked during brainstorming

| Question | Decision |
|---|---|
| How live must per-speaker labels be? | **Near-live.** Fast You/Others stays; distinct speakers fill in a few seconds behind; Stop is authoritative. |
| Local engine | **Reuse the SpeechBrain ECAPA embedding** already shipped for the offline pass; add a **new** online clusterer. Cross-platform (CPU everywhere, CUDA on Linux/Windows, MPS on Mac). |
| MOSS / server | A second `Diarizer` backend behind the existing seam, selectable by config; its own design later. |
| Identity persistence | **Per-meeting names now**, with the identity model shaped so cross-meeting voice enrollment can be added later. Enrollment filed as a backlog exploration. |
| Renaming | Live in the meeting screen and afterward on the Library item; names key on the cluster id, not the segment. |

## 3. Architecture

Everything sits behind the `Diarizer` protocol already declared in
`tldw_chatbook/Audio/meeting_session.py`. The session learns no engine details.

### 3.1 The `Diarizer` seam (refined)

Phase 1 defined `diarize(wav_path, start_s, end_s) -> list[SpeakerSegment]`. That
file-path call is correct only for the **Stop** pass. Near-live must not read the
growing on-disk WAV, whose placeholder header is patched only at close. The seam
gains a PCM-in entry point for the rolling pass:

```python
class Diarizer(Protocol):
    def assign(self, pcm: bytes, sample_rate: int, seq: int) -> str | None: ...   # near-live: returns a stable cluster id
    def diarize(self, wav_path: Path, start_s: float, end_s: float) -> list[SpeakerSegment]: ...  # Stop: authoritative
    def close(self) -> None: ...
```

A backend instance is created **once per meeting** and holds its own clustering
state, so a cluster id is stable across successive `assign` calls. `assign`
returns `None` when the audio is too short or too overlapped to attribute; the
segment then keeps its coarse label.

### 3.2 Backends

- **`SpeechBrainDiarizer` (local, default).** A thin adapter over the ECAPA
  embedding in `Local_Ingestion/diarization_service.py`. Near-live: embed the
  segment PCM and feed the online clusterer (§3.3). Stop: run the existing batch
  clustering over the full file. One model, two modes.
- **`ServerDiarizer` / MOSS (reserved).** Same interface, selected by config;
  full design deferred. Reserving the seam is all this spec does for it.
- **Backend factory** in the session owner picks `local` or `server` from config
  and injects the backend where the owner already builds the session. When
  `live_diarization` is off, no backend is built and phase 1's energy labeller is
  used unchanged; `live_diarization` is the single on/off switch, so
  `diarizer_backend` has no `off` value.

### 3.3 The online clusterer (new)

The shipped offline clustering (scikit-learn Spectral / Agglomerative) is batch:
it needs every embedding and a target count up front, so it cannot drive the
near-live path. The near-live path uses a small incremental clusterer:

- Keep one running centroid per speaker (plus a count), never the full embedding
  history, so memory is bounded by speaker count, not meeting length.
- Assign each new embedding to the nearest centroid within a cosine threshold;
  otherwise start a new cluster, up to `max_speakers`. Past the cap, fold into
  the nearest existing cluster (documented behavior, not an error).
- A **user-named cluster is sticky:** new segments may still join it, but it is
  never auto-merged away, so a live rename cannot vanish mid-meeting.

The clusterer is a **standalone pure-numpy module with no torch dependency**. It
runs inside the worker next to the embedder (so the worker returns cluster ids,
not embeddings), but because it is import-light it is unit-tested directly with
synthetic embedding vectors, without spawning the subprocess.

### 3.4 Isolation: a diarizer subprocess

The local backend runs in a **subprocess**, for three reasons: GIL isolation, so
embedding never stalls the UI thread or speech-to-text; crash isolation, so a
native torch fault cannot take down the app; and clean memory reclamation at
meeting end. (The UI-ready import census is *not* a reason: the diarizer is
imported at meeting start, after boot, so a thread would score identically. And
torch is already resident in-process via the transcriber, so this is additive
isolation, not the first heavy model.)

The subprocess receives audio PCM and returns cluster ids only; **names and
transcript text never leave the app process**, and the audio stays on-device over
a local pipe. Packaging cost is real and must be resolved in the plan: spawning a
Python+torch subprocess from the frozen macOS `.app` and Windows build needs
multiprocessing spawn with `freeze_support`, or a bundled worker entry point the
frozen app can exec, verified on an actual packaged build. If that proves too
costly, the documented fallback is a worker thread that accepts GIL contention.

## 4. Near-live data flow

**Which channel is diarized.** Diarization refines only the multi-person side.

- **Call mode:** the mic ("You") is the local user and keeps its label; the
  system channel ("Others") holds the remote participants and is diarized into
  Speaker A/B/C.
- **Room mode:** everyone shares the one mic (phase-1 segments carry no You/Others
  label here), so that channel is diarized into all the speakers. Because no
  channel identifies the local user, room mode has **no pre-named "You"**; every
  speaker starts generic until renamed. "You" pre-naming applies only to the mic
  channel in call mode.
- **Hybrid limitation:** if the mic also captures someone sitting next to you,
  the whole mic channel is still "You" in v1. Diarizing the mic channel too is a
  later config option.

**Rolling loop.** Partials keep the instant energy label so the live line never
stalls. When a segment finalizes, a processing worker sends its in-memory PCM to
the diarizer's `assign`; the returned cluster id upgrades the segment's label
from "Others" to a stable speaker a few seconds behind. Results reach the UI
through the existing sink calls, serialized by the session lock; embedding
happens **outside** the lock, over the subprocess pipe (phase 1's C2
lock-across-blocking-work lesson). `assign` is bounded: if the worker does not
answer within a small budget, the segment stays coarse (this is the backpressure
path in §6.3), and the Stop pass labels it authoritatively.

**Overlap / short bursts.** "Overlap" is the existing phase-1 `both` label (mic
and system both active in call mode); this design adds no new overlap detector,
and room mode has no overlap signal. Segments that are `both`, or too short to
embed, keep the coarse "You + Others" label and are not attributed to a person.

**Reconciliation on Stop.** The batch pass is authoritative. Live cluster ids are
matched to final ones by embedding similarity, so a relabeled segment keeps the
name the user assigned. When the final pass merges two live clusters the user
named differently, **both names are kept on the merged speaker and flagged for
the user to resolve** — never silently dropped. Turning on `live_diarization`
implies this Stop pass runs even when `post_diarize` is off, since reconciliation
requires it.

## 5. Speaker identity and renaming

### 5.1 Model and authority

Each meeting owns a name map from cluster id to display name, held in
`meeting.json`. Segments in `transcript.jsonl` carry the channel (`you`/`others`)
plus a cluster id; names live only in the map, so relabeling never loses a name.
"You" is a reserved id, pre-filled from the existing `user_display_name` config.

The meeting folder (`transcript.jsonl` + `meeting.json`, both surviving
raw-track cleanup) is authoritative **until ingest**. At ingest the rendered
transcript text is stored on the Library media item, and authority for
after-the-fact renaming transfers to the folder-backed map reached via the media
item. The media item links to the folder for free: `Media.url` is the
`mixed.wav` path, so the folder is its parent directory. The name map is **not**
stored in `transcription_provenance_json` — that column is a schema-validated
provenance document with its own producer and must not be repurposed.

### 5.2 Renaming live

The meeting screen gains a **speaker legend** in the right pane listing the
speakers seen so far, each with an inline rename. An edit updates the map and
re-renders the visible transcript lines in place. It is a small addition, not a
new screen. Empty rename reverts to the generated label; duplicate names are
allowed without auto-merge.

### 5.3 Renaming afterward

The Library transcript view shows the same legend and edit. An edit updates the
stored name map, re-renders the transcript text, and, in one DB transaction,
rewrites the transcript field Library displays and searches, reindexes
`media_fts`, and writes a new versioned `Transcripts` row for sync. The exact
field (`Media.content` vs the `Transcripts` row) is pinned during planning by
reading the media render path. Because labels are baked into the searchable text
(so a person can be found by name), each rename is a full rewrite plus reindex;
renames are rare, so this cost is acceptable. If the meeting folder was deleted,
the rename control is disabled with an explanation rather than maintaining a
second authority.

**Multi-device limitation.** The name map lives in the meeting folder, which is
local and not synced; only the rendered transcript text syncs with the media
item. So rename-after works on the device that holds the folder and is disabled
elsewhere. Making names portable across devices means storing the map in the
synced DB (not in `transcription_provenance_json`), which is deferred alongside
voiceprint enrollment rather than solved here.

## 6. Config, dependencies, performance

### 6.1 Config (new keys under `[meetings]`)

| Key | Default | Meaning |
|---|---|---|
| `live_diarization` | `false` | Opt-in near-live switch. Off = today's You/Others. |
| `diarizer_backend` | `local` | `local` or `server`; which engine to use when live is on. |
| `max_speakers` | `8` | Online-clusterer cap. |
| `post_diarize` | `true` (existing) | Drives the authoritative Stop pass. Independent of `live_diarization`, but `live_diarization` forces a Stop pass regardless. |

Device is auto-detected (CUDA on Linux/Windows, MPS on Mac, CPU otherwise) with
an override.

### 6.2 Dependencies

**No new dependency.** The local engine reuses the existing `diarization` extra
(torch, torchaudio, speechbrain, scikit-learn). The online clusterer uses only
what is already installed. `diarization_requirements` already reports missing
modules; its readout is extended to distinguish live from offline availability.

### 6.3 Performance and resource

- Subprocess isolation (§3.4): GIL, crash, and memory isolation; killable.
- Model **warmed at meeting Start**, not on the first segment, reusing the
  phase-1 warm-up follow-up (TASK-31636), so the first label is not delayed by a
  cold load.
- **Backpressure:** under CPU pressure the worker skips embedding and leaves the
  coarse label; the Stop pass fills the gaps. Live labels are best-effort.
- Memory bounded over a long meeting: the clusterer keeps centroids, not every
  embedding.
- **Opt-in first ship** (`live_diarization = false`): the already-shipped offline
  pass remains the default, so no existing meeting gets slower.

## 7. Error handling and degradation

Governing rule: diarization is best-effort and never blocks the recording, the
transcript, or Library ingest. Every failure falls back to the coarse
You/Others labels.

- **Deps missing:** stay on energy labels; the status line names what to install.
- **Subprocess start failure:** whole meeting on coarse labels; footer notes
  "speaker labels unavailable" with the reason. Mirrors the tap helper's
  mic-only fallback.
- **Subprocess crash mid-meeting:** the clusterer's centroids live in the worker,
  so a restart cannot continue the same speaker ids. To avoid showing the same
  person under a new id (or two people under one), a crash sends the **rest of the
  meeting to coarse labels** and the footer says so; the Stop batch pass then
  labels the whole recording authoritatively. (One restart is still attempted so
  a transient failure does not permanently disable the Stop pass.) Recording is
  never interrupted.
- **Backpressure:** skip embedding, keep coarse; the segment is still
  transcribed.
- **Stop pass failure:** the meeting saves with whatever labels it has and
  ingest proceeds; footer notes the failure. Diarization never gates ingest.
- **Crash recovery:** a recovered meeting gets speakers from the standard Library
  ingest diarization when `post_diarize` is on, not from a meeting batch pass;
  renames made before a crash are not recovered.
- **Privacy:** failure logs redact paths and never carry transcript text or
  speaker names. The server backend, which sends audio off-device, must address
  that in its own design.

## 8. Testing

- **Online clusterer, pure unit** (synthetic numpy embeddings, no torch): stable
  ids, threshold, `max_speakers` fold, sticky user-named clusters.
- **Reconciliation, pure unit:** live→final mapping by similarity, many-to-one
  merge keeps both names and flags, name-follows-cluster, rename edge cases.
- **Name map and format, unit:** rename by cluster id re-renders; versioned
  `meeting.json` / `transcript.jsonl` round-trip; older recordings without
  speaker fields still load.
- **Session integration with a fake diarizer** (phase-1 fake-capture pattern):
  segments upgrade coarse→speaker, partials stay coarse, no sink call under the
  session lock.
- **Degradation, unit:** backend off; deps missing; subprocess start failure;
  crash-restart-once; Stop-pass failure never blocking ingest; the
  live-implies-Stop-pass rule.
- **Rename-after, integration** (real in-memory media DB): edit rewrites the
  transcript text, bumps the version, reindexes FTS; disabled when the folder is
  gone.
- **Invariants + one gated real test:** extend the import-safety test so boot
  pulls in no torch; assert no transcript text or names reach logs; an opt-in,
  env-gated test spawns the real worker with the `diarization` extra and embeds
  real audio, skipped by default like the tap helper test.

## 9. New / changed files (map for planning)

- `tldw_chatbook/Audio/meeting_session.py` — extend the `Diarizer` protocol
  (`assign`, `close`); segment model gains a cluster id; swap the energy labeller
  for the diarizer when configured.
- `tldw_chatbook/Audio/diarizer_local.py` (new) — `SpeechBrainDiarizer` adapter +
  the online clusterer + subprocess host.
- `tldw_chatbook/Audio/diarizer_worker.py` (new) — the subprocess entry point
  (audio in, cluster ids out).
- `tldw_chatbook/Audio/meeting_owner.py` — backend factory + config; extend the
  diarization-availability readout.
- `tldw_chatbook/UI/Screens/meetings_screen.py` — the live speaker legend +
  inline rename.
- Library transcript view — the after-the-fact legend + rename, transcript
  rewrite + FTS reindex + versioned write (exact file pinned in planning).
- `tldw_chatbook/config.py` — the new `[meetings]` keys.
- `Docs/User_Guide/meetings.md` — document live speaker labels and renaming.
- Tests under `Tests/Audio/` and `Tests/UI/` per §8.

## 10. Follow-ups (not in this phase)

1. **MOSS / server backend** — `Diarizer` over MOSS-Transcribe-Diarize
   (CUDA/server), including the off-device-audio privacy design.
2. **Cross-meeting voiceprint enrollment** — remember a voice as a named person
   across meetings; needs a voiceprint store, match thresholds, and a
   consent/privacy surface. File as a backlog exploration.
3. **Diarize the mic channel** for hybrid rooms (config option).
4. **Windows loopback verification** (inherited from phase 1).
