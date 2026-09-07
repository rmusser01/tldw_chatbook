"""Subprocess worker: PCM -> speaker id. torch/SpeechBrain live ONLY here.

Never import this module in the app process -- it pulls in torch. `main()` is
run as ``python -m tldw_chatbook.Audio.diarizer_worker`` by
`diarizer_local.SpeechBrainDiarizer`, which owns the wire protocol:

    argv  :  ``--start-id N`` (optional) -- start cluster numbering past ``N``.
             Set only on a RESTART, so the replacement worker cannot re-mint
             an id the dead one already handed out (31749).
    stdin :  one JSON control line per command; an "assign" or "enroll_from_pcm"
             line is followed by exactly ``n`` bytes of raw PCM16 (16 kHz mono).
    stdout:  one ``{"id": ..., "seq": ..., "self": ...}`` line per assign (the
             ``seq`` is echoed so the app can discard a reply whose window
             already gave up); ``{"segments": [...], "self": ...}`` for a
             diarize; ``{"centroid": [...] | null, "seconds": ...}`` for
             ``export_centroid`` / ``enroll_from_pcm``. ``enroll`` and ``pin``
             send no reply.
    stderr:  ``READY`` once the ECAPA model is warm; ``ERROR <op> <type>`` on a
             per-command failure. Never PCM, text, names, or paths -- and
             never the voiceprint vector (spec §3.2/§6).

The worker holds the live `OnlineClusterer` for the whole meeting. On a
``diarize`` command it clusters the whole file (batch), computes final
centroids, reconciles them against the live centroids, and returns segments
whose ``speaker`` is already the live cluster id -- so reconciliation lives
here, never in the session (spec §3.3).

Self-voiceprint matching (spec §3.2, 31826 task 2): ``enroll`` holds a
voiceprint vector plus its own (stricter) match threshold and a minimum
accumulated-seconds gate in memory for this process only -- never persisted,
never logged. ``assign`` and ``diarize`` replies then carry a ``self`` field
matched against that voiceprint; ``export_centroid`` and ``enroll_from_pcm``
hand the app process a centroid to persist itself (`Audio/voiceprint.py`).
"""
from __future__ import annotations

import json
import math
import os
import sys

MODEL = "speechbrain/spkrec-ecapa-voxceleb"
# No loader-exposed revision today; a real pin is TODO once one exists.
MODEL_ID = f"{MODEL}@unpinned"
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


def _unit(v) -> list[float] | None:
    """Unit-normalise a float-iterable (list or numpy array); ``None`` for a
    zero-magnitude vector (review round 1, Minor 9 -- never hand the app a
    dead voiceprint it would happily persist).

    Pure Python (``math`` only): `serve()` must not import
    `Audio/voiceprint.py` (drags the encryption stack into the diarizer
    subprocess) or use numpy at its own scope (review round 1, Minor 6/7).
    """
    values = [float(x) for x in v]
    magnitude = math.sqrt(sum(x * x for x in values))
    if magnitude == 0.0:
        return None
    return [x / magnitude for x in values]


def _cos_dist(a, b) -> float:
    """1 - cosine similarity between two float-iterables; ``1.0`` (max
    distance) if either is a zero vector. Pure Python, on lists -- mirrors
    `diarizer_cluster._cos`'s zero-safety without importing a private name
    across modules (review round 1, Minor 5/10)."""
    a = [float(x) for x in a]; b = [float(x) for x in b]
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    return 1.0 - sum(x * y for x, y in zip(a, b)) / (na * nb)


def _seconds_by_speaker(segs) -> dict[str, float]:
    """Sum each segment's duration into its speaker id (review round 1,
    Important 3: `export_centroid`'s batch-sourced ``seconds`` is computed
    from `diarize`'s own segments, not the live clusterer, which never saw a
    Stop-pass-only id and would otherwise report a silent ``0.0``)."""
    out: dict[str, float] = {}
    for s in segs:
        out[s["speaker"]] = out.get(s["speaker"], 0.0) + (s["end_s"] - s["start_s"])
    return out


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


def _parse_start_id(argv) -> int:
    """``--start-id N`` -> N (0 when absent or garbled).

    Written by `diarizer_local.SpeechBrainDiarizer._command` when it restarts a
    dead worker, so this one continues the first worker's numbering (31749).
    """
    try:
        return max(0, int(argv[list(argv).index("--start-id") + 1]))
    except (ValueError, IndexError, TypeError):
        return 0


def _reconcile_windows(spans, embeddings, live_centroids, cluster_fn, threshold=0.25, start_id=0,
                        out_centroids=None):
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
        threshold: The live clusterer's own cosine-distance threshold, passed
            to `reconcile` so a surplus final cluster that is plainly the same
            voice keeps that speaker's live id (and name) instead of being
            minted a new one.
        start_id: The live clusterer's `max_id` -- the highest cluster number
            in use, INCLUDING ids a pre-crash worker minted and this one only
            inherited (31749). A minted id always continues past it, so a Stop
            pass run on a restarted (centroid-less) worker cannot hand out an
            id the user already named.
        out_centroids: Optional dict the caller supplies to receive
            ``{live_id: final_centroid}`` (unit-normalised, as a plain
            ``list[float]`` -- never a numpy array, so `serve()` can treat
            every batch centroid as pure-Python data) for every reconciled
            final cluster (mint included) -- an out-param rather than a
            return-value change so the four `_reconcile_windows`-level tests
            above, which only look at the returned segment list, stay
            untouched. When two or more final clusters reconcile to the SAME
            live id -- `reconcile`'s surplus rule, the designed common case
            for a batch pass that over-split one person -- the stored
            centroid is the seconds-weighted mean of all of them (weight =
            each final cluster's total segment duration), not last-write-wins
            (review round 1, Important 2).

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
    seconds_by_fid: dict[str, float] = {}
    for (s0, s1), label, emb in zip(spans, labels, embeddings):
        fid = f"F{label}"
        grouped.setdefault(fid, []).append(emb)
        seconds_by_fid[fid] = seconds_by_fid.get(fid, 0.0) + (s1 - s0)
    final_centroids = [(key, np.mean(vecs, axis=0)) for key, vecs in grouped.items()]
    mapping = reconcile(live_centroids, final_centroids, threshold)  # final label -> live id
    # An unmatched final cluster (no live centroid to match -- e.g. near-live
    # labelling was backpressured the whole meeting, so live_centroids is
    # empty) must NOT surface as "Speaker F0" (final whole-branch review I2):
    # mint it a fresh live-style id continuing past the highest live number.
    # ... past the highest live id AND past `start_id`: after a crash the
    # restarted worker has no live centroids at all, so counting from them
    # alone would mint "S1" straight onto a pre-crash speaker's name (31749).
    next_n = max((int(k[1:]) for k in live_centroids if k[1:].isdigit()), default=0)
    next_n = max(next_n, int(start_id)) + 1
    for fid, _cen in final_centroids:
        if fid not in mapping:
            mapping[fid] = f"S{next_n}"
            next_n += 1
    if out_centroids is not None:
        cen_by_fid = dict(final_centroids)
        fids_by_live: dict[str, list[str]] = {}
        for fid, _cen in final_centroids:
            fids_by_live.setdefault(mapping[fid], []).append(fid)
        for lid, fids in fids_by_live.items():
            total_w = sum(seconds_by_fid[f] for f in fids)
            if total_w > 0:
                mean = sum(cen_by_fid[f] * seconds_by_fid[f] for f in fids) / total_w
            else:  # every fid folding here had 0 s of windows (degenerate) -- plain mean
                mean = np.mean([cen_by_fid[f] for f in fids], axis=0)
            norm = float(np.linalg.norm(mean))
            out_centroids[lid] = (mean / norm if norm > 0.0 else mean).tolist()
    return [
        {"start_s": s0, "end_s": s1, "speaker": mapping.get(f"F{label}", f"F{label}")}
        for (s0, s1), label in zip(spans, labels)
    ]


def _batch(encoder, torch, np, live, wav_path: str, start_s: float, end_s: float, max_speakers: int):
    """Embed the whole file (torch), then cluster + reconcile to live ids.

    Returns:
        ``(segments, final_centroids_by_live_id)`` -- the reconciled segment
        dicts, plus every reconciled final cluster's centroid keyed by the
        live id it maps to (voiceprint `self` matching and `export_centroid`
        read this after a `diarize`; see `serve()`).
    """
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
    out_centroids: dict = {}
    segments = _reconcile_windows(
        spans, embeddings, live.centroids(), svc._cluster_speakers, live.threshold, live.max_id,
        out_centroids=out_centroids,
    )
    return segments, out_centroids


def _write(stdout, obj) -> None:
    stdout.write((json.dumps(obj) + "\n").encode())
    stdout.flush()


def serve(stdin, stdout, live, embed, batch) -> int:
    """Run the wire protocol until stdin closes or a ``close`` arrives.

    Split out of `main()` (Qodo Q2) so the command loop is reachable without
    torch: `main()` is the only place the encoder is loaded, and the loop
    used to sit behind that load, so nothing but a hand-written fake process
    on the app side could exercise command dispatch. A serialization or
    dispatch regression -- the ``pin`` forwarding in particular, which sends
    no reply and so shows up nowhere else -- could not fail a test. `main()`
    passes the real encoder-bound callables; a test passes its own.

    Args:
        stdin: Binary input stream: one JSON control line per command, an
            "assign"/"enroll_from_pcm" line followed by exactly its ``n``
            bytes of PCM.
        stdout: Binary output stream for the one-line JSON replies.
        live: The `OnlineClusterer` held for the whole meeting.
        embed: ``(pcm: bytes) -> embedding`` for an assign or enroll_from_pcm.
        batch: ``(wav, start_s, end_s) -> (segment dicts, {live_id: centroid})``
            for a diarize.

    Returns:
        0 -- the process exit code, so `main()` can return it directly.
    """
    enrolled = None  # (voiceprint_vector: list[float], threshold, min_seconds), or None
    last_batch_centroids: dict = {}  # live_id -> centroid (list[float]), from the last diarize
    last_batch_seconds: dict = {}    # live_id -> seconds, summed from that diarize's own segments

    while True:
        line = stdin.readline()
        if not line:
            break
        try:
            cmd = json.loads(line)
        except Exception:  # noqa: BLE001 - ignore a garbled control line
            continue
        op = cmd.get("cmd")
        # Review round 1, Important 4: one framed error path for every op --
        # a malformed `enroll`/`export_centroid` used to raise straight out of
        # `serve()` (unhandled, a traceback on stderr, the worker dead for the
        # rest of the meeting). `assign`/`diarize`/`enroll_from_pcm` still read
        # their PCM unconditionally first, so a bad `n`/`sr` doesn't desync
        # the pipe for whatever comes after it consumes.
        try:
            if op == "assign":
                n = int(cmd.get("n", 0))
                sr = int(cmd.get("sr", 16000)) or 16000
                pcm = _read_exactly(stdin, n)
                seconds = len(pcm) / (2 * sr)  # Minor 2: bytes actually read, not the declared n
                sid = live.assign(embed(pcm), seconds=seconds)
                is_self = False
                if enrolled is not None and sid is not None:
                    evec, ethresh, emin_s = enrolled
                    dist = live.distance_to(sid, evec)
                    if dist is not None:
                        is_self = dist <= ethresh and live.seconds(sid) >= emin_s
                _write(stdout, {"id": sid, "seq": cmd.get("seq"), "self": is_self})
            elif op == "diarize":
                segs, final_centroids = batch(cmd["wav"], float(cmd.get("start", 0.0)), float(cmd.get("end", 0.0)))
                # M3 (ruling): a successful batch -- even an empty one -- always
                # replaces the stored state; only a raised exception (below)
                # leaves it untouched, since a failed diarize produced nothing.
                last_batch_centroids = final_centroids
                last_batch_seconds = _seconds_by_speaker(segs)
                self_id = None
                # M4: judged against THIS batch's own centroids, not whatever
                # a previous successful diarize left behind.
                if enrolled is not None and final_centroids:
                    evec, ethresh = enrolled[0], enrolled[1]
                    dists = [(cid, _cos_dist(cen, evec)) for cid, cen in final_centroids.items()]
                    best_id, best_dist = min(dists, key=lambda t: t[1])
                    if best_dist <= ethresh:
                        self_id = best_id
                _write(stdout, {"segments": segs, "self": self_id})
            elif op == "pin":
                live.pin(str(cmd.get("id", "")))
            elif op == "enroll":
                # No reply: held in memory only, for this process's lifetime.
                enrolled = (
                    [float(x) for x in cmd.get("vector", [])],
                    float(cmd.get("threshold", 0.2)),
                    float(cmd.get("min_seconds", 4.0)),  # M1: the config default, not 0
                )
            elif op == "export_centroid":
                cid = str(cmd.get("id", ""))
                if cid in last_batch_centroids:
                    # Important 3: the BATCH centroid pairs with BATCH seconds
                    # (summed from its own segments) -- live.seconds() reads
                    # 0.0 for a Stop-pass-only id and silently zeroes out the
                    # §3.4 learning merge's weight.
                    cen, secs = last_batch_centroids[cid], last_batch_seconds.get(cid, 0.0)
                else:
                    cen, secs = live.centroids().get(cid), live.seconds(cid)
                unit = None if cen is None else _unit(cen)
                _write(stdout, {"centroid": None} if unit is None else {"centroid": unit, "seconds": secs})
            elif op == "enroll_from_pcm":
                n = int(cmd.get("n", 0))
                sr = int(cmd.get("sr", 16000)) or 16000
                pcm = _read_exactly(stdin, n)
                unit = _unit(embed(pcm))
                seconds = len(pcm) / (2 * sr)  # Minor 2
                _write(stdout, {"centroid": None} if unit is None else {"centroid": unit, "seconds": seconds})
            elif op == "close":
                break
        except Exception as exc:  # noqa: BLE001 - framed, never a traceback (paths) on stderr
            sys.stderr.write(f"ERROR {op} {type(exc).__name__}\n")
            sys.stderr.flush()
            if op == "assign":
                _write(stdout, {"id": None, "seq": cmd.get("seq"), "self": False})
            elif op == "diarize":
                _write(stdout, {"segments": [], "self": None})
            elif op in ("export_centroid", "enroll_from_pcm"):
                _write(stdout, {"centroid": None})
            # enroll/pin: no reply either way; `enrolled`/`live` are untouched
            # by a raise partway through, so the previous state stands.
    return 0


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

    live = OnlineClusterer(max_speakers=max_speakers, start_id=_parse_start_id(sys.argv[1:]))
    sys.stderr.write("READY\n")
    sys.stderr.flush()

    return serve(
        stdin, stdout, live,
        lambda pcm: _embed(encoder, torch, np, pcm),
        lambda wav, start_s, end_s: _batch(
            encoder, torch, np, live, wav, start_s, end_s, max_speakers
        ),
    )


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()  # ponytail: harmless dev no-op; a real frozen-app spawn seam is TODO in diarizer_local._command
    sys.exit(main())
