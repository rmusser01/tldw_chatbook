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

Model acquisition (spec §3, task 3: 31827): `ensure_models` places the
manifest's assets under `models_dir()`, streaming through httpx (imported
only inside `ensure_models`/`_stream_to_file`, never at module scope) so
this module still never pulls httpx in just by being imported.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ModelAsset:
    key: str
    kind: str  # "segmentation" | "embedder"
    file_name: str
    url: str
    sha256: str
    size: int
    licence: str
    #: Bytes of the thing actually DOWNLOADED when it differs from `size`
    #: (the segmentation asset is fetched as a tarball but `size`/`sha256`
    #: describe the extracted member). The streamed-byte cap uses this so a
    #: 6.9 MB archive is not judged against its 6.0 MB member (re-review).
    download_size: int | None = None


#: The pyannote segmentation 3.0 model, used by every embedder's Stop pass
#: (spec §3). `url` is the release TARBALL -- Task 3's downloader extracts
#: `model.onnx` from it under this asset's `file_name`; `sha256`/`size` are
#: of the extracted file, not the tarball; `download_size` is the tarball's.
SEGMENTATION = ModelAsset(
    "segmentation", "segmentation", "pyannote-segmentation-3-0.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="220ad67ca923bef2fa91f2390c786097bf305bceb5e261d4af67b38e938e1079",
    size=5_992_913,
    licence="MIT (pyannote/segmentation-3.0)",
    download_size=6_958_444,
)

#: The int8 variant of the same tarball's `model.int8.onnx`, kept for the
#: bake-off (spec §7) -- not wired into `model_paths`/`load()` yet.
SEGMENTATION_INT8 = ModelAsset(
    "segmentation_int8", "segmentation", "pyannote-segmentation-3-0-int8.onnx",
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
    sha256="d582f4b4c6b48205de7e0643c57df0df5615a3c176189be3fc461e9d18827b5d",
    size=1_540_506,
    licence="MIT (pyannote/segmentation-3.0)",
    download_size=6_958_444,  # the same tarball as SEGMENTATION
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
    """`(segmentation_path, embedder_path)` for `embedder` under `models_dir(override)`.

    Raises:
        ValueError: `embedder` is not a manifest key -- `models_ready` takes
            the same input and returns False instead of raising (Task 3
            calls both together; only one of them needs to raise).
    """
    if embedder not in EMBEDDERS:
        raise ValueError(f"unknown embedder: {embedder}")
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


class ModelsUnavailable(RuntimeError):
    """`ensure_models` failed (spec §3/§8). The message is always one of the
    four static strings below -- never a URL, host, path or file name."""


#: The manifest's release host and the only redirect target suffix a hop may
#: land on (spec §3: "redirects followed only to *.githubusercontent.com").
ALLOWED_HOSTS = ("github.com",)
ALLOWED_REDIRECT_SUFFIX = ".githubusercontent.com"
DOWNLOAD_BUDGET_S = 600.0

#: The only hosts allowed to use plain `http` -- the test stub's own loopback
#: server. Every other hop must be `https` (review C1): a plaintext hop lets
#: an on-path attacker choose the bytes `_extract_tar_member` consumes, which
#: is not itself hash-pinned (the manifest sha256 is of the extracted file).
_LOOPBACK_HOSTS = ("127.0.0.1", "localhost")

_MAX_REDIRECTS = 5
_MB = 1 << 20
#: Slack over a landed file's declared manifest size before a download is
#: refused outright (review C2) -- catches a hostile/misbehaving server
#: filling the disk (or memory, for the tarball) well before any hash check
#: could run.
_DOWNLOAD_SLACK_BYTES = 1 << 20


def _url_for(asset: ModelAsset) -> str:
    """Seam tests monkeypatch by replacing the manifest asset itself
    (`dataclasses.replace(asset, url=...)`) rather than this function."""
    return asset.url


def _temp_path_for(final: Path) -> Path:
    return final.with_name(f"{final.name}.tmp-{os.getpid()}-{secrets.token_hex(4)}")


def _scheme_allowed(url) -> bool:
    """`https` everywhere, except the test stub's own loopback host."""
    if url.scheme == "https":
        return True
    return url.scheme == "http" and (url.host or "") in _LOOPBACK_HOSTS


def _new_http_client():
    """Seam so a test can assert the exact client kwargs (`trust_env` for
    proxy passthrough, `follow_redirects=False` so every hop is checked
    here) without patching httpx itself."""
    import httpx

    return httpx.Client(follow_redirects=False, trust_env=True, timeout=30.0)


def _extract_tar_member(tar_path: Path, wanted_basename: str, dest: Path, expected_size: int) -> None:
    """Extract `wanted_basename` (`model.onnx` or `model.int8.onnx`) from the
    segmentation release tarball, rejecting any candidate that is not an
    exact `expected_size` (the manifest's size for the file it produces --
    review C2: a header lying about size is refused before any byte is
    read, never after), a link, or whose name escapes the archive (spec
    §3's safe-member filter). Copies with `shutil.copyfileobj`, never a
    single `.read()` of the whole member (review C2)."""
    import shutil
    import tarfile

    with tarfile.open(tar_path, mode="r:bz2") as tf:
        member = None
        for candidate in tf.getmembers():
            name = candidate.name
            if os.path.basename(name) != wanted_basename:
                continue
            if candidate.size != expected_size:
                continue
            if name.startswith("/") or ".." in Path(name).parts:
                continue
            if not candidate.isfile() or candidate.issym() or candidate.islnk():
                continue
            member = candidate
            break
        if member is None:
            raise ModelsUnavailable("download failed")
        extracted = tf.extractfile(member)
        if extracted is None:
            raise ModelsUnavailable("download failed")
        with open(dest, "wb") as out:
            shutil.copyfileobj(extracted, out, 1 << 20)


def _stream_to_file(
    http_client, url: str, dest: Path, deadline: float, on_bytes: Callable[[int], None], max_bytes: int,
) -> None:
    """GET `url`, following at most `_MAX_REDIRECTS` 3xx hops whose target
    host is allow-listed and whose scheme never changes (review C1),
    streaming the final 2xx body to `dest` -- refusing past `max_bytes`
    (review C2) before it is ever written. Raises
    `ModelsUnavailable("download failed")` or `("budget exceeded")`; never
    mentions the URL/host in the exception."""
    import httpx

    current = url
    start = httpx.URL(current)
    if start.host not in ALLOWED_HOSTS or not _scheme_allowed(start):
        raise ModelsUnavailable("download failed")

    for _ in range(_MAX_REDIRECTS + 1):
        if time.monotonic() >= deadline:
            raise ModelsUnavailable("budget exceeded")
        try:
            with http_client.stream("GET", current) as resp:
                if resp.status_code in (301, 302, 303, 307, 308):
                    location = resp.headers.get("location")
                    if not location:
                        raise ModelsUnavailable("download failed")
                    cur_url = httpx.URL(current)
                    next_url = cur_url.join(location)
                    if next_url.scheme != cur_url.scheme or not _scheme_allowed(next_url):
                        raise ModelsUnavailable("download failed")
                    host = next_url.host or ""
                    if host not in ALLOWED_HOSTS and not host.endswith(ALLOWED_REDIRECT_SUFFIX):
                        raise ModelsUnavailable("download failed")
                    current = str(next_url)
                    continue
                if not (200 <= resp.status_code < 300):
                    raise ModelsUnavailable("download failed")
                written = 0
                with open(dest, "wb") as fh:
                    # No `chunk_size`: httpx yields pieces as they arrive off
                    # the wire instead of buffering up to a fixed size first,
                    # which is what lets the budget check below actually cut
                    # a slow transfer short between chunks (ruling 6) rather
                    # than only after the whole body has landed.
                    for piece in resp.iter_bytes():
                        if time.monotonic() >= deadline:
                            raise ModelsUnavailable("budget exceeded")
                        written += len(piece)
                        if written > max_bytes:
                            raise ModelsUnavailable("download failed")
                        fh.write(piece)
                        on_bytes(len(piece))
                return
        except httpx.HTTPError:
            raise ModelsUnavailable("download failed") from None
    raise ModelsUnavailable("download failed")


def _fetch_asset(http_client, asset: ModelAsset, path: Path, deadline: float, on_bytes: Callable[[int], None]) -> None:
    """Download `asset` (retrying once on a verification failure) and
    atomically place it at `path` (spec §3/§8 -- rulings 3/4/9)."""
    # `ensure_models` only ever fetches `SEGMENTATION` (not the int8 variant,
    # which isn't in `model_paths`' plan -- spec §7 bake-off leaves it as a
    # future candidate), so the tarball member wanted is always the same.
    is_tarball = asset.kind == "segmentation"
    wanted_member = "model.onnx"
    # Cap the STREAM by what is downloaded (the archive for a tarball asset),
    # never by the extracted member's size -- the two differ by ~1 MB here.
    max_bytes = (asset.download_size or asset.size) + _DOWNLOAD_SLACK_BYTES

    for attempt in range(2):
        final_tmp = _temp_path_for(path)
        try:
            if is_tarball:
                tar_tmp = _temp_path_for(path.with_suffix(".tar.bz2"))
                try:
                    _stream_to_file(http_client, _url_for(asset), tar_tmp, deadline, on_bytes, max_bytes)
                    _extract_tar_member(tar_tmp, wanted_member, final_tmp, asset.size)
                finally:
                    tar_tmp.unlink(missing_ok=True)
            else:
                _stream_to_file(http_client, _url_for(asset), final_tmp, deadline, on_bytes, max_bytes)

            if final_tmp.stat().st_size == asset.size and _sha256_of(final_tmp) == asset.sha256:
                os.replace(final_tmp, path)
                return
        finally:
            final_tmp.unlink(missing_ok=True)

        if attempt == 1:
            raise ModelsUnavailable("hash mismatch")


def ensure_models(
    embedder: str,
    *,
    models_dir_override: Path | None = None,
    progress: Callable[[str], None] | None = None,
    budget_s: float = DOWNLOAD_BUDGET_S,
    client: Any | None = None,
) -> tuple[Path, Path]:
    """Place the segmentation + `embedder` ONNX models under
    `models_dir(models_dir_override)`, downloading whatever is missing
    (spec §3).

    A failing asset only removes its own temp file -- a sibling asset that
    already landed and verified in this same call is left in place (review
    I5: spec §3's "concurrent fetches ... are harmless" and §8's "retried
    next Start" both require that a valid, hash-verified file is never
    rolled back just because a later asset in the same call failed).

    Air-gapped installs (`models_dir_override` pointing at a pre-placed
    directory): a file that already exists there is fully hash-verified and
    never touched over the network, matching or not -- a mismatch is fatal
    (`"air-gapped file invalid"`), never retried. A file that is simply
    absent is still fetched normally, so an override directory can also be
    used as a plain download destination (this is also how the test suite
    redirects writes away from the real user data dir).

    The default (no override) placement never hashes a pre-existing file
    (presence + size only, like `models_ready`) -- hashing 40 MB on every
    call would defeat the point of the no-op path; `load()` hashes on open.

    Args:
        embedder: One of `EMBEDDERS`' keys.
        models_dir_override: Test/air-gapped seam for `models_dir()`.
        progress: Called with static strings like `"downloading 12 / 35 MB"`
            (integers only, no file names or paths), at most once per MB of
            combined progress across whatever still needs fetching.
        budget_s: Wall-clock ceiling for the whole call.
        client: An httpx-client-shaped object (`.stream`); a fresh
            `httpx.Client` is created (and closed) when omitted.

    Returns:
        `(segmentation_path, embedder_path)`.

    Raises:
        ValueError: `embedder` is not a manifest key (from `model_paths`).
        ModelsUnavailable: `"download failed"`, `"hash mismatch"`,
            `"budget exceeded"`, or `"air-gapped file invalid"`.
    """
    seg_path, emb_path = model_paths(embedder, models_dir_override)
    plan = ((SEGMENTATION, seg_path), (EMBEDDERS[embedder], emb_path))

    to_fetch: list[tuple[ModelAsset, Path]] = []
    for asset, path in plan:
        if models_dir_override is not None and path.is_file():
            if path.stat().st_size == asset.size and _sha256_of(path) == asset.sha256:
                continue
            raise ModelsUnavailable("air-gapped file invalid")
        # Mirrors `models_ready`'s presence+size contract (inline, since
        # this loop also needs the per-asset `(asset, path)` pair for
        # `to_fetch`) -- do not "fix" this into a hash check; ruling 8 is
        # explicit that a fresh call must not re-hash an already-placed file.
        if path.is_file() and path.stat().st_size == asset.size:
            continue
        to_fetch.append((asset, path))

    if not to_fetch:
        return seg_path, emb_path

    if budget_s <= 0:
        raise ModelsUnavailable("budget exceeded")
    deadline = time.monotonic() + budget_s

    total_mb = sum(asset.size for asset, _ in to_fetch) // _MB
    state = {"done": 0, "reported_mb": 0}

    def _on_bytes(n: int) -> None:
        state["done"] += n
        mb = min(state["done"] // _MB, total_mb)
        if progress is not None and mb > state["reported_mb"]:
            state["reported_mb"] = mb
            progress(f"downloading {mb} / {total_mb} MB")

    seg_path.parent.mkdir(parents=True, exist_ok=True)
    owns_client = client is None
    http_client = client
    try:
        for asset, path in to_fetch:
            if owns_client and http_client is None:
                http_client = _new_http_client()
            _fetch_asset(http_client, asset, path, deadline, _on_bytes)
    finally:
        if owns_client and http_client is not None:
            http_client.close()

    return seg_path, emb_path


def _read_wav_span(path: str, start_s: float, end_s: float):
    """Read `[start_s, end_s)` of a mono 16 kHz 16-bit PCM WAV as float32
    samples in `[-1, 1]` (`wave` + numpy only, per spec §5 -- no torchaudio;
    the meeting always writes 16 kHz mono, so this never resamples).

    Returns:
        `(samples, sr)`.

    Raises:
        ValueError: anything but mono 16 kHz 16-bit PCM, or a `start_s` at or
            past end-of-file (both "unsupported wav").
    """
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or sr != 16000:
            raise ValueError("unsupported wav")
        total = wf.getnframes()
        a = max(0, int(round(start_s * sr)))
        if a >= total:
            raise ValueError("unsupported wav")
        b = min(total, int(round(end_s * sr))) if end_s else total
        b = max(a, b)
        wf.setpos(a)
        raw = wf.readframes(b - a)
    samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    return samples, sr


def _embed_vector(extractor, np, samples, sr: int):
    """Unit-normalised embedding (np.ndarray) for `samples` at `sr` via the
    sherpa-onnx stream API. A zero-magnitude embedding is returned as-is --
    the enroll/export paths' `_unit()` refuses it downstream; `assign` does
    not (task 1's own surface, unaffected by this engine)."""
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
        groups: `{speaker_label: [(start_s, end_s), ...]}` -- times relative
            to the START of `samples` (i.e. span-relative, NOT yet offset);
            the batch's own `start_s` is added back later, at segment
            assembly in `_batch`.

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
            provider="cpu",
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

    def _make_extractor(num_threads: int):
        return sherpa_onnx.SpeakerEmbeddingExtractor(
            sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(emb_path), num_threads=num_threads, provider="cpu")
        )

    extractor = _make_extractor(LIVE_THREADS)
    threshold = CLUSTER_THRESHOLD[key]

    return LoadedEngine(
        lambda pcm, sr: _embed(extractor, np, pcm, sr),
        # A fresh BATCH_THREADS extractor per Stop pass (spec §2: "4 for the
        # Stop pass") -- built lazily here rather than eagerly alongside
        # `extractor` above, since the Stop pass runs at most once per
        # meeting and a live-only meeting should not pay for a second
        # loaded model it never uses.
        lambda wav, s, e: _batch(
            sherpa_onnx, np, _make_extractor(BATCH_THREADS), seg_path, emb_path, threshold, live, max_speakers, wav, s, e
        ),
        model_id_for_embedder(key),
    )
