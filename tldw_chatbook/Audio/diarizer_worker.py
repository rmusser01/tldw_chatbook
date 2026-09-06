"""Subprocess worker: PCM -> speaker id. torch/SpeechBrain live ONLY here.

Never import this module in the app process -- it pulls in torch. `main()` is
run as ``python -m tldw_chatbook.Audio.diarizer_worker`` by
`diarizer_local.SpeechBrainDiarizer`, which owns the wire protocol:

    stdin :  one JSON control line per command; an "assign" line is followed
             by exactly ``n`` bytes of raw PCM16 (16 kHz mono).
    stdout:  one ``{"id": ...}`` line per assign; ``{"segments": [...]}`` for
             a diarize.
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

    from tldw_chatbook.Local_Ingestion.diarization_service import _lazy_import_speechbrain

    EncoderClassifier = _lazy_import_speechbrain()
    if EncoderClassifier is None:
        raise RuntimeError("SpeechBrain EncoderClassifier unavailable")
    savedir = Path("pretrained_models") / "spkrec-ecapa-voxceleb"
    return EncoderClassifier.from_hparams(source=MODEL, savedir=str(savedir), run_opts={"device": "cpu"})


def _embed(encoder, torch, np, pcm: bytes):
    """PCM16 bytes -> a 1-D float32 embedding via the ECAPA encoder."""
    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    wav = torch.from_numpy(audio).unsqueeze(0)  # (1, samples)
    with torch.no_grad():
        emb = encoder.encode_batch(wav)
    return np.asarray(emb.squeeze().detach().cpu().numpy(), dtype=np.float32)


def _batch(encoder, torch, np, live, wav_path: str, start_s: float, end_s: float, max_speakers: int):
    """Cluster the whole file, reconcile to live ids, return segment dicts."""
    import torchaudio

    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer, reconcile

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

    batch = OnlineClusterer(max_speakers=max_speakers)
    placed: list[tuple[float, float, str]] = []
    for pos in range(a, b, win):
        chunk = wav[0, pos:pos + win]
        if chunk.shape[0] < floor:
            continue
        with torch.no_grad():
            emb = encoder.encode_batch(chunk.unsqueeze(0)).squeeze().detach().cpu().numpy()
        fid = batch.assign(np.asarray(emb, dtype=np.float32))
        placed.append((pos / sr, min(pos + win, b) / sr, fid))

    mapping = reconcile(live.centroids(), list(batch.centroids().items()))  # final id -> live id
    return [{"start_s": s0, "end_s": s1, "speaker": mapping.get(fid, fid)} for s0, s1, fid in placed]


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
            _write(stdout, {"id": sid})
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
