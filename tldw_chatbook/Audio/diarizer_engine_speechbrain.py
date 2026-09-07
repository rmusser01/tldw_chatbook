"""SpeechBrain (ECAPA-TDNN) diarizer engine. torch/SpeechBrain live ONLY here.

Split out of `diarizer_worker.py` (task 1: 31827) so the worker's command
loop (`serve()`) and its `--engine` selection are reachable without torch --
this module, like `diarizer_worker.py` itself, must import torch/torchaudio/
speechbrain/numpy ONLY inside `load()` and the functions it calls at runtime
(never at module scope), so `import`ing it alone never pulls torch in.

Engine module contract (spec §2): `load(live, max_speakers) -> LoadedEngine`
plus a `MODEL_ID` constant. `main()` in `diarizer_worker.py` imports this
module by name (via `ENGINES["speechbrain"]`) and calls `load()`; `serve()`
never learns which engine is running.
"""
from __future__ import annotations

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# No loader-exposed revision today; a real pin is TODO once one exists.
MODEL_ID = f"{MODEL}@unpinned"
WINDOW_S = 1.5  # batch clustering window over the recording


def _load_encoder():
    from pathlib import Path

    from tldw_chatbook.Local_Ingestion.diarization_service import (
        DiarizationService,
        _lazy_import_speechbrain,
    )

    EncoderClassifier = _lazy_import_speechbrain()
    if EncoderClassifier is None:
        raise RuntimeError("SpeechBrain EncoderClassifier unavailable")
    # Qodo Q13: the Stop pass loads and embeds the WHOLE recording under a
    # parent timeout, so an accelerator is worth having. Reuse the project's
    # own selection (`[diarization] embedding_device`, "auto" -> CUDA when
    # present) rather than hard-coding CPU here. Constructing the service is
    # cheap: it loads config only, never a model.
    try:
        device = DiarizationService()._get_device()
    except Exception:  # noqa: BLE001 - a config problem must not lose the pass
        device = "cpu"
    savedir = Path("pretrained_models") / "spkrec-ecapa-voxceleb"
    return EncoderClassifier.from_hparams(source=MODEL, savedir=str(savedir), run_opts={"device": device})


def _embed(encoder, torch, np, pcm: bytes):
    """PCM16 bytes -> a 1-D float32 embedding via the ECAPA encoder."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        emb = encoder.encode_batch(wav)
    return np.asarray(emb.squeeze().detach().cpu().numpy(), dtype=np.float32)


def _cluster_windows(spans, embeddings, cluster_fn):
    """Cluster window embeddings into final label groups.

    The window-clustering half of the pre-split `_reconcile_windows` (task 1:
    31827) -- SpeechBrain-only (1.5 s windows + the project's agglomerative
    pass); the ONNX engine builds its `final_clusters` a different way and
    never calls this. ``cluster_fn(embeddings, num_speakers)`` is the
    project's batch agglomerative pass -- its labels can differ from (and so
    correct) the greedy live labels, which is the whole point of
    reconciliation.

    The batch speaker count is decided by ``cluster_fn`` ITSELF (called with
    ``None``), which runs the service's single-speaker check and silhouette
    estimate bounded by its configured ``min_speakers``/``max_speakers``.

    Args:
        spans: One ``(start_s, end_s)`` per window, in file order.
        embeddings: One embedding per window (same order/length as ``spans``).
        cluster_fn: ``(np.ndarray[n,d], int | None) -> labels[n]``; skipped
            when there are fewer than two windows to cluster.

    Returns:
        ``(fids, final_clusters)`` -- ``fids[i]`` is window ``i``'s final
        label (``"F<n>"``), aligned with ``spans``/``embeddings``;
        ``final_clusters`` is ``[(fid, mean_centroid, seconds), ...]`` for
        `diarizer_worker._map_final_clusters`.
    """
    import numpy as np

    if not embeddings:
        return [], []
    if len(embeddings) < 2:
        labels = [0] * len(embeddings)
    else:
        labels = [int(x) for x in cluster_fn(np.asarray(embeddings, dtype=np.float32), None)]

    fids = [f"F{label}" for label in labels]
    grouped: dict[str, list] = {}
    seconds_by_fid: dict[str, float] = {}
    for (s0, s1), fid, emb in zip(spans, fids, embeddings):
        grouped.setdefault(fid, []).append(emb)
        seconds_by_fid[fid] = seconds_by_fid.get(fid, 0.0) + (s1 - s0)
    final_clusters = [(fid, np.mean(vecs, axis=0), seconds_by_fid[fid]) for fid, vecs in grouped.items()]
    return fids, final_clusters


def _batch(encoder, torch, np, live, wav_path: str, start_s: float, end_s: float, max_speakers: int):
    """Embed the whole file (torch), then cluster + reconcile to live ids.

    Returns:
        ``(segments, final_centroids_by_live_id)`` -- the reconciled segment
        dicts, plus every reconciled final cluster's centroid keyed by the
        live id it maps to (voiceprint `self` matching and `export_centroid`
        read this after a `diarize`; see `diarizer_worker.serve()`).
    """
    from tldw_chatbook.Audio import diarizer_worker
    from tldw_chatbook.Local_Ingestion.diarization_service import (
        ClusteringMethod,
        DiarizationService,
        _lazy_import_torchaudio,
    )

    # Qodo Q4: torchaudio is part of the optional `diarization` extra; go
    # through the project's centralized loader, not a bare import.
    torchaudio = _lazy_import_torchaudio()
    if torchaudio is None:
        raise RuntimeError("torchaudio unavailable")

    wav, sr = torchaudio.load(wav_path)  # (channels, samples)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    total = wav.shape[1]
    a = max(0, int(start_s * sr))
    b = min(total, int(end_s * sr)) if end_s else total
    win = int(WINDOW_S * sr)
    floor = int(0.4 * sr)  # skip a too-short tail window

    spans: list[tuple[float, float]] = []
    embeddings: list = []
    for pos in range(a, b, win):
        chunk = wav[0, pos:pos + win]
        if chunk.shape[0] < floor:
            continue
        with torch.no_grad():
            emb = encoder.encode_batch(chunk.unsqueeze(0)).squeeze().detach().cpu().numpy()
        spans.append((pos / sr, min(pos + win, b) / sr))
        embeddings.append(np.asarray(emb, dtype=np.float32))

    # Cheap: __init__ loads no models; only _cluster_speakers (sklearn) runs.
    # These bounds ARE the [1, max_speakers] bound on the batch speaker count
    # `_cluster_windows` relies on (Q11).
    svc = DiarizationService(config={
        "max_speakers": max_speakers,
        "min_speakers": 1,
        "clustering_method": ClusteringMethod.AGGLOMERATIVE.value,
    })
    fids, final_clusters = _cluster_windows(spans, embeddings, svc._cluster_speakers)
    out_centroids: dict = {}
    mapping = diarizer_worker._map_final_clusters(
        final_clusters, live.centroids(), live.threshold, live.max_id, out_centroids=out_centroids,
    )
    segments = [
        {"start_s": s0, "end_s": s1, "speaker": mapping.get(fid, fid)}
        for (s0, s1), fid in zip(spans, fids)
    ]
    return segments, out_centroids


def load(live, max_speakers: int):
    """Load the ECAPA encoder and return the worker's `LoadedEngine` triple.

    Args:
        live: The `OnlineClusterer` held for the whole meeting -- `_batch`
            reads its centroids/threshold/max_id for reconciliation.
        max_speakers: The Stop pass's speaker-count ceiling.

    Returns:
        A `diarizer_worker.LoadedEngine` bound to a freshly loaded encoder.
    """
    from tldw_chatbook.Audio.diarizer_worker import LoadedEngine
    from tldw_chatbook.Local_Ingestion.diarization_service import _lazy_import_torch

    import numpy as np

    torch = _lazy_import_torch()
    if torch is None:
        raise RuntimeError("torch unavailable")
    encoder = _load_encoder()

    return LoadedEngine(
        lambda pcm, sr: _embed(encoder, torch, np, pcm),
        lambda wav, s, e: _batch(encoder, torch, np, live, wav, s, e, max_speakers),
        MODEL_ID,
    )
