"""Subprocess worker: PCM -> speaker id. torch/SpeechBrain live ONLY here.

Never import this module in the app process -- it pulls in torch. `main()` is
run as ``python -m tldw_chatbook.Audio.diarizer_worker`` by
`diarizer_local.SpeechBrainDiarizer`, which owns the wire protocol:

    stdin :  one JSON control line per command; an "assign" line is followed
             by exactly ``n`` bytes of raw PCM16 (16 kHz mono).
    stdout:  one ``{"id": ..., "seq": ...}`` line per assign (the ``seq`` is
             echoed so the app can discard a reply whose window already gave
             up); ``{"segments": [...]}`` for a diarize.
    stderr:  ``READY`` once the ECAPA model is warm; ``ERROR <type>`` on a
             per-command failure. Never PCM, text, names, or paths.

The worker holds the live `OnlineClusterer` for the whole meeting. On a
``diarize`` command it clusters the whole file (batch), computes final
centroids, reconciles them against the live centroids, and returns segments
whose ``speaker`` is already the live cluster id -- so reconciliation lives
here, never in the session (spec §3.3).
"""
from __future__ import annotations

import json
import os
import sys

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
WINDOW_S = 1.5  # batch clustering window over the recording


def _read_exactly(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes (PCM is length-prefixed by the control line)."""
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            break
        buf.extend(chunk)
    return bytes(buf)


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


def _reconcile_windows(spans, embeddings, live_centroids, cluster_fn):
    """Pure (no torch): cluster window embeddings, reconcile to live ids.

    The authoritative Stop pass. ``cluster_fn(embeddings, num_speakers)`` is the
    project's batch agglomerative pass -- its labels can differ from (and so
    correct) the greedy live labels, which is the whole point of reconciliation.

    The batch speaker count is decided by ``cluster_fn`` ITSELF (called with
    ``None``), which runs the service's single-speaker check and silhouette
    estimate bounded by its configured ``min_speakers``/``max_speakers``.
    Deriving it from ``len(live_centroids)`` instead (Qodo Q11) capped the
    authoritative pass at the best-effort live count, so a backpressured live
    pass that found one speaker forced the whole recording into one cluster --
    the exact gap this pass exists to fill (spec §6.3).

    Args:
        spans: One ``(start_s, end_s)`` per window, in file order.
        embeddings: One embedding per window (same order/length as ``spans``).
        live_centroids: The live cluster centroids held during the meeting;
            used ONLY to map final clusters back to live ids, never to bound
            the count.
        cluster_fn: ``(np.ndarray[n,d], int | None) -> labels[n]``; skipped
            when there are fewer than two windows to cluster.

    Returns:
        Segment dicts (``start_s``/``end_s``/``speaker``), speaker = reconciled
        live id (falls back to the final label when nothing lives to match).
    """
    import numpy as np

    from tldw_chatbook.Audio.diarizer_cluster import reconcile

    if not embeddings:
        return []
    if len(embeddings) < 2:
        labels = [0] * len(embeddings)
    else:
        labels = [int(x) for x in cluster_fn(np.asarray(embeddings, dtype=np.float32), None)]

    grouped: dict[str, list] = {}
    for label, emb in zip(labels, embeddings):
        grouped.setdefault(f"F{label}", []).append(emb)
    final_centroids = [(key, np.mean(vecs, axis=0)) for key, vecs in grouped.items()]
    mapping = reconcile(live_centroids, final_centroids)  # final label -> live id
    # An unmatched final cluster (no live centroid to match -- e.g. near-live
    # labelling was backpressured the whole meeting, so live_centroids is
    # empty) must NOT surface as "Speaker F0" (final whole-branch review I2):
    # mint it a fresh live-style id continuing past the highest live number.
    next_n = max((int(k[1:]) for k in live_centroids if k[1:].isdigit()), default=0) + 1
    for fid, _cen in final_centroids:
        if fid not in mapping:
            mapping[fid] = f"S{next_n}"
            next_n += 1
    return [
        {"start_s": s0, "end_s": s1, "speaker": mapping.get(f"F{label}", f"F{label}")}
        for (s0, s1), label in zip(spans, labels)
    ]


def _batch(encoder, torch, np, live, wav_path: str, start_s: float, end_s: float, max_speakers: int):
    """Embed the whole file (torch), then cluster + reconcile to live ids."""
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
    # `_reconcile_windows` relies on (Q11).
    svc = DiarizationService(config={
        "max_speakers": max_speakers,
        "min_speakers": 1,
        "clustering_method": ClusteringMethod.AGGLOMERATIVE.value,
    })
    return _reconcile_windows(spans, embeddings, live.centroids(), svc._cluster_speakers)


def _write(stdout, obj) -> None:
    stdout.write((json.dumps(obj) + "\n").encode())
    stdout.flush()


def main() -> int:
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    max_speakers = int(os.environ.get("TLDW_DIARIZER_MAX_SPEAKERS", "8"))

    try:
        import numpy as np

        from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer
        from tldw_chatbook.Local_Ingestion.diarization_service import _lazy_import_torch

        torch = _lazy_import_torch()
        if torch is None:
            raise RuntimeError("torch unavailable")
        encoder = _load_encoder()
    except Exception as exc:  # noqa: BLE001 - type only, never the message content
        sys.stderr.write(f"ERROR load {type(exc).__name__}\n")
        sys.stderr.flush()
        return 1

    live = OnlineClusterer(max_speakers=max_speakers)
    sys.stderr.write("READY\n")
    sys.stderr.flush()

    while True:
        line = stdin.readline()
        if not line:
            break
        try:
            cmd = json.loads(line)
        except Exception:  # noqa: BLE001 - ignore a garbled control line
            continue
        op = cmd.get("cmd")
        if op == "assign":
            pcm = _read_exactly(stdin, int(cmd.get("n", 0)))
            try:
                sid = live.assign(_embed(encoder, torch, np, pcm))
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"ERROR assign {type(exc).__name__}\n")
                sys.stderr.flush()
                sid = None
            _write(stdout, {"id": sid, "seq": cmd.get("seq")})
        elif op == "diarize":
            try:
                segs = _batch(
                    encoder, torch, np, live,
                    cmd["wav"], float(cmd.get("start", 0.0)), float(cmd.get("end", 0.0)), max_speakers,
                )
            except Exception as exc:  # noqa: BLE001
                sys.stderr.write(f"ERROR diarize {type(exc).__name__}\n")
                sys.stderr.flush()
                segs = []
            _write(stdout, {"segments": segs})
        elif op == "close":
            break
    return 0


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # ponytail: harmless dev no-op; a real frozen-app spawn seam is TODO in diarizer_local._command
    sys.exit(main())
