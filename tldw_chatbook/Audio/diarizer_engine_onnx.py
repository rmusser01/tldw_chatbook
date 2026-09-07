"""sherpa-onnx diarizer engine: manifest, live `embed`, offline Stop `batch`.

torch-free counterpart to `diarizer_engine_speechbrain.py` (task 2: 31827) --
sherpa-onnx/numpy live ONLY inside `load()` and the functions it builds, so
`import`ing this module alone (as `diarizer_worker.ENGINES["onnx"]` does by
name, unconditionally) never pulls either in. Nothing here logs a path,
vector, or audio sample (spec §8) -- this module logs nothing at all.

Engine module contract (spec §2): `load(live, max_speakers) -> LoadedEngine`
plus a `MODEL_ID` constant, same as the SpeechBrain engine. `main()` in
`diarizer_worker.py` imports this module by name and calls `load()`;
`serve()` never learns which engine is running.

Model acquisition (spec §3) is Task 3's job; this module only defines the
manifest (`SEGMENTATION`, `EMBEDDERS`) and the placement helpers
(`models_dir`, `model_paths`, `models_ready`) Task 3's downloader targets --
clean seams, no network code here.
"""
from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelAsset:
    key: str
    kind: str  # "segmentation" | "embedder"
    file_name: str
    url: str
    sha256: str
    size: int
    licence: str


#: The pyannote segmentation 3.0 model, used by every embedder's Stop pass
#: (spec §3). `url` is the release TARBALL -- Task 3's downloader extracts
#: `model.onnx` from it under this asset's `file_name`; `sha256`/`size` are
#: of the extracted file, not the tarball.
SEGMENTATION = ModelAsset(
    "segmentation", "segmentation", "pyannote-segmentation-3-0.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    size=5_992_913,
    licence="MIT (pyannote/segmentation-3.0)",
)

#: The int8 variant of the same tarball's `model.int8.onnx`, kept for the
#: bake-off (spec §7) -- not wired into `model_paths`/`load()` yet.
SEGMENTATION_INT8 = ModelAsset(
    "segmentation_int8", "segmentation", "pyannote-segmentation-3-0-int8.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d",
    size=1_540_506,
    licence="MIT (pyannote/segmentation-3.0)",
)

#: Bake-off candidate embedders (spec §3/§7). Each is used for BOTH the live
#: path and the Stop pass -- the segmentation model never changes.
EMBEDDERS: dict[str, ModelAsset] = {
    "titanet_small": ModelAsset(
        "titanet_small", "embedder", "nemo_en_titanet_small.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/nemo_en_titanet_small.onnx",
        sha256="ad4a1802485d8b34c722d2a9d04249662f2ece5d28a7a039063ca22f515a789e",
        size=40_257_283,
        licence="CC-BY-4.0 (NVIDIA)",
    ),
    "wespeaker_resnet34": ModelAsset(
        "wespeaker_resnet34", "embedder", "wespeaker_en_voxceleb_resnet34.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/wespeaker_en_voxceleb_resnet34.onnx",
        sha256="5ef208a9da1453335308a6b6f4e6dfbd7e183a38b604de0a57664f45d257fe94",
        size=26_534_365,
        licence="Apache-2.0 (WeSpeaker)",
    ),
    "eres2net_en": ModelAsset(
        "eres2net_en", "embedder", "3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_eres2net_sv_en_voxceleb_16k.onnx",
        sha256="c59158379255ad66e161679cca6af8d52d51e389e3224ab7d7a7baae295c2db5",
        size=26_485_263,
        licence="Apache-2.0 (3D-Speaker; verify ModelScope terms)",
    ),
    "campplus_en": ModelAsset(
        "campplus_en", "embedder", "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx",
        sha256="357a834f702b80161e5b981182c038e18553c1f2ca752ed6cec2052365d4129b",
        size=29_596_978,
        licence="Apache-2.0 (3D-Speaker; verify ModelScope terms)",
    ),
}

DEFAULT_EMBEDDER = "titanet_small"  # Task 9 (bake-off) may change it

#: Cosine-distance clustering threshold per embedder (spec §5). Starting
#: point for every candidate; the bake-off (spec §7) tunes these.
CLUSTER_THRESHOLD: dict[str, float] = {key: 0.5 for key in EMBEDDERS}

MIN_DURATION_ON, MIN_DURATION_OFF = 0.3, 0.5
LIVE_THREADS, BATCH_THREADS = 2, 4
#: Cap on the per-cluster centroid-building cost (spec §5): stop embedding a
#: cluster's segments once this many seconds of it are covered.
CENTROID_SECONDS = 30.0


def model_id_for_embedder(key: str) -> str:
    """The voiceprint model id for embedder `key` (spec §6): stable across
    a run, changes if the manifest's file or hash for `key` changes."""
    asset = EMBEDDERS[key]
    return f"sherpa-onnx/{asset.file_name}@{asset.sha256[:12]}"


MODEL_ID = model_id_for_embedder(DEFAULT_EMBEDDER)


def models_dir(override: Path | None = None) -> Path:
    """Where ONNX model files live: `override`, or the user data dir's
    standard placement (spec §3). No directory is created here -- Task 3's
    downloader does that on write."""
    if override is not None:
        return Path(override)
    from tldw_chatbook.config import get_user_data_dir

    return get_user_data_dir() / "models" / "diarization" / "onnx"


def model_paths(embedder: str, override: Path | None = None) -> tuple[Path, Path]:
    """`(segmentation_path, embedder_path)` for `embedder` under `models_dir(override)`."""
    d = models_dir(override)
    return d / SEGMENTATION.file_name, d / EMBEDDERS[embedder].file_name


def models_ready(embedder: str, override: Path | None = None) -> bool:
    """Presence + byte size only (spec §4) -- hashes are checked on download
    and again on `load()`, never here (this runs on screen-open paths)."""
    asset = EMBEDDERS.get(embedder)
    if asset is None:
        return False
    seg_path, emb_path = model_paths(embedder, override)
    return (
        seg_path.is_file() and seg_path.stat().st_size == SEGMENTATION.size
        and emb_path.is_file() and emb_path.stat().st_size == asset.size
    )


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_wav_span(path: str, start_s: float, end_s: float):
    """Read `[start_s, end_s)` of a mono 16-bit PCM WAV as float32 samples in
    `[-1, 1]` (`wave` + numpy only, per spec §5 -- no torchaudio).

    Returns:
        `(samples, sr)`.

    Raises:
        ValueError: anything but mono 16-bit PCM ("unsupported wav").
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("unsupported wav")
        sr = wf.getframerate()
        total = wf.getnframes()
        a = max(0, int(round(start_s * sr)))
        b = min(total, int(round(end_s * sr))) if end_s else total
        b = max(a, b)
        wf.setpos(a)
        raw = wf.readframes(b - a)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sr


def _embed_vector(extractor, np, samples, sr: int):
    """Unit-normalised embedding (np.ndarray) for `samples` at `sr` via the
    sherpa-onnx stream API. A zero-magnitude embedding is returned as-is --
    the worker's `_unit()` refuses it downstream (task 1)."""
    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=sr, waveform=samples)
    stream.input_finished()
    vec = np.asarray(extractor.compute(stream), dtype=np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0.0 else vec


def _embed(extractor, np, pcm: bytes, sr: int) -> list:
    """PCM16 bytes -> a unit-normalised embedding (the live path)."""
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    return _embed_vector(extractor, np, samples, sr).tolist()


def _cluster_centroids(extractor, np, samples, sr, groups):
    """Per spec §5: for each Stop-pass cluster, embed its longest segments
    SEPARATELY until `CENTROID_SECONDS` of that cluster is covered, average
    the unit vectors, then unit-normalise. `seconds` is the cluster's FULL
    total duration, uncapped -- the pipeline exposes no internal embeddings,
    so this is the only way to get a centroid in the same vector space the
    live centroids live in.

    Args:
        groups: `{speaker_label: [(start_s, end_s), ...]}`, times relative to
            `samples` (already offset by the batch's own start).

    Returns:
        `{"F<label>": (centroid: np.ndarray, seconds: float)}`.
    """
    clusters = {}
    for label, segs in groups.items():
        total_secs = sum(e - s for s, e in segs)
        vecs = []
        covered = 0.0
        for s, e in sorted(segs, key=lambda se: se[1] - se[0], reverse=True):
            a, b = int(round(s * sr)), int(round(e * sr))
            chunk = samples[a:b]
            if chunk.size == 0:
                continue
            vecs.append(_embed_vector(extractor, np, chunk, sr))
            covered += (e - s)
            if covered >= CENTROID_SECONDS:
                break
        if vecs:
            mean = np.mean(vecs, axis=0)
            norm = float(np.linalg.norm(mean))
            centroid = mean / norm if norm > 0.0 else mean
        else:  # degenerate: every segment for this label was empty
            centroid = np.zeros(extractor.dim, dtype=np.float32)
        clusters[f"F{label}"] = (centroid, total_secs)
    return clusters


def _fold_to_cap(np, clusters, cap: int):
    """Fold the smallest (by seconds) cluster into its nearest remaining
    centroid (cosine) until at most `cap` remain (spec §5 -- mirrors the
    live clusterer's own cap fold), re-normalising and adding the seconds.

    Returns:
        `(folded_clusters, redirect)` -- `redirect` maps every ORIGINAL
        label to the (possibly transitively) folded-into survivor's label,
        so a segment whose cluster got folded away can still be routed to
        the surviving centroid's reconciled live id.
    """
    active = dict(clusters)
    redirect = {label: label for label in clusters}
    while len(active) > cap and len(active) > 1:
        smallest = min(active, key=lambda k: active[k][1])
        remaining = [k for k in active if k != smallest]
        cen_s, secs_s = active[smallest]
        target = max(remaining, key=lambda k: float(np.dot(cen_s, active[k][0])))
        cen_t, secs_t = active[target]
        combined = cen_t * secs_t + cen_s * secs_s
        norm = float(np.linalg.norm(combined))
        active[target] = (combined / norm if norm > 0.0 else combined, secs_t + secs_s)
        del active[smallest]
        for label, dest in redirect.items():
            if dest == smallest:
                redirect[label] = target
    return active, redirect


def _batch(sherpa_onnx, np, extractor, seg_path, emb_path, threshold, live, max_speakers, wav, start_s, end_s):
    """Run the offline diarization pipeline over `[start_s, end_s)` of `wav`,
    fold past `max_speakers`, and reconcile onto `live`'s cluster ids (spec §5).

    Returns:
        `(segments, final_centroids_by_live_id)` -- same shape the
        SpeechBrain engine's `_batch` returns.
    """
    from tldw_chatbook.Audio.diarizer_worker import _map_final_clusters

    samples, sr = _read_wav_span(wav, start_s, end_s)

    diar_cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(model=str(seg_path)),
            num_threads=BATCH_THREADS,
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(emb_path), num_threads=BATCH_THREADS, provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(num_clusters=-1, threshold=threshold),
        min_duration_on=MIN_DURATION_ON,
        min_duration_off=MIN_DURATION_OFF,
    )
    result = sherpa_onnx.OfflineSpeakerDiarization(diar_cfg).process(samples).sort_by_start_time()

    groups: dict = {}
    for r in result:
        groups.setdefault(r.speaker, []).append((r.start, r.end))

    clusters = _cluster_centroids(extractor, np, samples, sr, groups)
    clusters, redirect = _fold_to_cap(np, clusters, max_speakers)

    final_clusters = [(label, cen, secs) for label, (cen, secs) in clusters.items()]
    out_centroids: dict = {}
    mapping = _map_final_clusters(
        final_clusters, live.centroids(), threshold=live.threshold, start_id=live.max_id,
        out_centroids=out_centroids,
    )

    segs = [
        {
            "start_s": start_s + r.start,
            "end_s": start_s + r.end,
            "speaker": mapping.get(redirect.get(f"F{r.speaker}", f"F{r.speaker}"), f"F{r.speaker}"),
        }
        for r in result
    ]
    return segs, out_centroids


def load(
    live,
    max_speakers: int,
    *,
    embedder: str | None = None,
    models_dir_override: Path | None = None,
    verify_hashes: bool = True,
):
    """Load the sherpa-onnx speaker embedding extractor and return the
    worker's `LoadedEngine` triple (spec §2).

    Args:
        live: The `OnlineClusterer` held for the whole meeting -- `batch`
            reads its centroids/threshold/max_id for reconciliation.
        max_speakers: The Stop pass's speaker-count ceiling.
        embedder: One of `EMBEDDERS`' keys; `DEFAULT_EMBEDDER` if omitted.
        models_dir_override: Test/air-gapped seam for `models_dir()`.
        verify_hashes: When true (the default), a SHA-256 mismatch against
            the manifest raises before any model is constructed.

    Returns:
        A `diarizer_worker.LoadedEngine` bound to a freshly loaded extractor.

    Raises:
        ValueError: unknown `embedder`, or a model hash mismatch
            ("model hash mismatch") -- `diarizer_worker.main()` frames this
            as `ERROR load ValueError`.
    """
    import numpy as np
    import sherpa_onnx

    from tldw_chatbook.Audio.diarizer_worker import LoadedEngine

    key = embedder or DEFAULT_EMBEDDER
    if key not in EMBEDDERS:
        raise ValueError(f"unknown embedder: {key}")
    seg_path, emb_path = model_paths(key, models_dir_override)

    if verify_hashes:
        if _sha256_of(seg_path) != SEGMENTATION.sha256 or _sha256_of(emb_path) != EMBEDDERS[key].sha256:
            raise ValueError("model hash mismatch")

    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(
        sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_path), num_threads=LIVE_THREADS, provider="cpu")
    )
    threshold = CLUSTER_THRESHOLD[key]

    return LoadedEngine(
        lambda pcm, sr: _embed(extractor, np, pcm, sr),
        lambda wav, s, e: _batch(
            sherpa_onnx, np, extractor, seg_path, emb_path, threshold, live, max_speakers, wav, s, e
        ),
        model_id_for_embedder(key),
    )
