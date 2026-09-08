#!/usr/bin/env python3
"""The diarizer bake-off runner (spec §7, TASK-31827). NOT collected by pytest.

    python Tests/Audio/bakeoff/run_bakeoff.py --help

Measures, per (engine, embedder, segmentation) and per threshold:

* **Stop-pass DER** -- `LocalDiarizer.diarize` over the whole file, scored by
  `der.der` with a 0.25 s collar against the corpus RTTM.
* **Live purity / coverage** -- 3 s windows that lie entirely inside one
  reference speaker's turn, fed to `assign` in time order. Purity is the
  fraction of windows whose cluster's majority reference speaker is the
  window's own; coverage the fraction of reference speakers that have at
  least one such cluster.
* **RTF** -- `diarize` wall time / audio duration, after `wait_ready`.
* **Peak worker RSS** -- `resource.getrusage(RUSAGE_CHILDREN).ru_maxrss`
  after `close()`. Every cell runs in its own child process, so the
  high-water mark belongs to that cell's workers and nothing else.
* **Per-window embed latency** -- median and p95 of `assign`, first call dropped.
* **Self-match separation** -- `enroll_from_pcm` over the first 30 s of one
  speaker's turns, then `cos(that speaker's cluster centroid, enrolled)`
  minus the best of the other clusters, with the midpoint distance as the
  recommended `voice_match_threshold`.

Two independent knobs, so the sweep is 5 + 5 and not 5 x 5: the Stop pass
runs on a worker that never saw an `assign`, so it has no live centroids to
reconcile against and its DER depends only on the clustering threshold; the
live path never runs the Stop pass. The report says so explicitly.

int8 segmentation is measured through the engine in-process
(`load(verify_hashes=False)`) rather than through the worker: the int8 file
has to sit under the float file's name for `model_paths` to find it, and the
worker's own load hash-checks it against the manifest, as it should. Float is
measured BOTH ways so the int8-vs-float delta is apples-to-apples.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from Tests.Audio.bakeoff import corpus
from Tests.Audio.bakeoff.der import der, mapping

WINDOW_S = 3.0
MAX_LIVE_WINDOWS = 60          # per file; keeps a live cell to ~20 s of assigns
ENROLL_S = 30.0
COLLAR_S = 0.25
MAX_SPEAKERS = 8
READY_S = 300.0
CELL_TIMEOUT_S = 5400.0
ECAPA = "ecapa"                # the SpeechBrain engine's single "embedder"


# --------------------------------------------------------------------------
# one cell, in its own process
# --------------------------------------------------------------------------
def _diarizer(spec: dict, models_dir: Path | None):
    from tldw_chatbook.Audio.diarizer_local import LocalDiarizer

    kwargs: dict = {"max_speakers": MAX_SPEAKERS}
    if spec["engine"] == "onnx":
        kwargs["embedder"] = spec["embedder"]
        kwargs["models_dir_override"] = models_dir
        # The models are pre-placed by the driver; skip the fetch/verify pass
        # so an int8 segmentation under the float name is not refused here.
        kwargs["ensure_models"] = lambda *a, **kw: None
    return LocalDiarizer(spec["engine"], **kwargs)


def _live_windows(reference: list[tuple[float, float, str]]) -> list[tuple[float, str]]:
    """`(start_s, speaker)` for 3 s windows inside exactly one reference turn."""
    windows: list[tuple[float, str]] = []
    for start, end, speaker in reference:
        t = start
        while t + WINDOW_S <= end:
            clean = not any(
                other_spk != speaker and other_s < t + WINDOW_S and other_e > t
                for other_s, other_e, other_spk in reference
            )
            if clean:
                windows.append((t, speaker))
            t += WINDOW_S
    windows.sort()
    if len(windows) > MAX_LIVE_WINDOWS:
        step = len(windows) / MAX_LIVE_WINDOWS
        windows = [windows[int(i * step)] for i in range(MAX_LIVE_WINDOWS)]
    return windows


def _live_file(diarizer, entry: dict, reference: list) -> dict:
    windows = _live_windows(reference)
    wav = Path(entry["wav"])
    assigned: list[tuple[str | None, str]] = []
    latencies: list[float] = []
    for seq, (start, speaker) in enumerate(windows):
        pcm = corpus.read_pcm(wav, start, start + WINDOW_S)
        t0 = time.perf_counter()
        cluster = diarizer.assign(pcm, 16000, seq)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        assigned.append((cluster, speaker))

    majority: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    for cluster, speaker in assigned:
        if cluster is None:
            continue
        counts.setdefault(cluster, {})
        counts[cluster][speaker] = counts[cluster].get(speaker, 0) + 1
    for cluster, by_speaker in counts.items():
        majority[cluster] = max(by_speaker, key=lambda s: by_speaker[s])

    scored = [(c, s) for c, s in assigned if c is not None]
    pure = sum(1 for c, s in scored if majority.get(c) == s)
    speakers = {s for _, s in assigned}
    covered = {s for s in speakers if any(m == s for m in majority.values())}
    return {
        "windows": len(assigned),
        "labelled": len(scored),
        "purity": (pure / len(scored)) if scored else 0.0,
        "coverage": (len(covered) / len(speakers)) if speakers else 0.0,
        "clusters": len(counts),
        "ref_speakers": len(speakers),
        "latency_ms": latencies[1:],
    }


def _separation(diarizer, entry: dict, reference: list, hypothesis: list) -> dict | None:
    """Enrol the busiest reference speaker, then compare cluster centroids."""
    import math

    by_speaker: dict[str, float] = {}
    for start, end, speaker in reference:
        by_speaker[speaker] = by_speaker.get(speaker, 0.0) + (end - start)
    if len(by_speaker) < 2 or not hypothesis:
        return None
    target = max(by_speaker, key=lambda s: by_speaker[s])

    wav = Path(entry["wav"])
    pcm = b""
    for start, end, speaker in sorted(reference):
        if speaker != target:
            continue
        pcm += corpus.read_pcm(wav, start, min(end, start + ENROLL_S))
        if len(pcm) >= int(ENROLL_S * 16000) * 2:
            break
    enrolled = diarizer.enroll_from_pcm(pcm[: int(ENROLL_S * 16000) * 2], 16000)
    if enrolled is None:
        return None
    vector = enrolled[0]

    hyp_to_ref = mapping(reference, hypothesis)
    own = [cluster for cluster, ref_speaker in hyp_to_ref.items() if ref_speaker == target]
    if not own:
        return None

    def _cos(a, b) -> float | None:
        if a is None or b is None or len(a) != len(b):
            return None
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(x * x for x in b))
        if na == 0.0 or nb == 0.0:
            return None
        return sum(x * y for x, y in zip(a, b)) / (na * nb)

    similarity: dict[str, float] = {}
    for cluster in {c for _, _, c in hypothesis}:
        exported = diarizer.export_centroid(cluster)
        value = _cos(exported[0] if exported else None, vector)
        if value is not None:
            similarity[cluster] = value
    mine = [similarity[c] for c in own if c in similarity]
    others = [v for c, v in similarity.items() if c not in own]
    if not mine:
        return None
    best_self = max(mine)
    best_other = max(others) if others else None
    return {
        "self_similarity": best_self,
        "other_similarity": best_other,
        "separation": None if best_other is None else best_self - best_other,
        # As a DISTANCE, which is what `voice_match_threshold` is compared against.
        "recommended_threshold": (
            None if best_other is None else ((1.0 - best_self) + (1.0 - best_other)) / 2.0
        ),
    }


def _stop_file_inprocess(spec: dict, models_dir: Path, entry: dict, duration: float) -> dict:
    """The engine's `batch` without the worker -- how int8 segmentation is measured."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as engine
    from tldw_chatbook.Audio.diarizer_cluster import OnlineClusterer

    loaded = engine.load(
        OnlineClusterer(max_speakers=MAX_SPEAKERS), MAX_SPEAKERS,
        embedder=spec["embedder"], models_dir_override=models_dir, verify_hashes=False,
    )
    t0 = time.perf_counter()
    segments, _centroids = loaded.batch(entry["wav"], 0.0, duration)
    elapsed = time.perf_counter() - t0
    hypothesis = [(s["start_s"], s["end_s"], s["speaker"]) for s in segments]
    return {"hypothesis": hypothesis, "elapsed": elapsed}


def run_cell(spec: dict) -> dict:
    """Measure one cell over its files; returns the per-file rows."""
    models_dir = Path(spec["models_dir"]) if spec.get("models_dir") else None
    if spec["engine"] == "onnx":
        for name, value in (
            ("TLDW_DIARIZER_LIVE_THRESHOLD", spec.get("live_threshold")),
            ("TLDW_DIARIZER_CLUSTER_THRESHOLD", spec.get("cluster_threshold")),
        ):
            if value is not None:
                os.environ[name] = str(float(value))

    rows: list[dict] = []
    for entry in spec["files"]:
        reference = corpus.read_rttm(Path(entry["rttm"]))
        duration = corpus.check_wav(Path(entry["wav"]))
        row: dict = {"id": entry["id"], "source": entry["source"], "duration_s": duration}
        try:
            if spec["kind"] == "stop" and spec.get("via") == "inprocess":
                measured = _stop_file_inprocess(spec, models_dir, entry, duration)
                row["der"] = der(reference, measured["hypothesis"], collar=COLLAR_S)
                row["rtf"] = measured["elapsed"] / duration
                row["hyp_segments"] = len(measured["hypothesis"])
                row["hyp_speakers"] = len({s for _, _, s in measured["hypothesis"]})
                rows.append(row)
                continue

            diarizer = _diarizer(spec, models_dir)
            try:
                if not diarizer.wait_ready(READY_S):
                    row["error"] = f"not ready ({diarizer.warmup_status})"
                    rows.append(row)
                    continue
                if spec["kind"] == "live":
                    row.update(_live_file(diarizer, entry, reference))
                else:
                    t0 = time.perf_counter()
                    segments = diarizer.diarize(Path(entry["wav"]), 0.0, duration)
                    row["rtf"] = (time.perf_counter() - t0) / duration
                    hypothesis = [(s.start_s, s.end_s, s.speaker) for s in segments]
                    row["der"] = der(reference, hypothesis, collar=COLLAR_S)
                    row["hyp_segments"] = len(hypothesis)
                    row["hyp_speakers"] = len({s for _, _, s in hypothesis})
                    row["ref_speakers"] = len({s for _, _, s in reference})
                    if entry["source"] == "voxconverse" and hypothesis:
                        row["separation"] = _separation(diarizer, entry, reference, hypothesis)
            finally:
                diarizer.close()
        except Exception as exc:  # noqa: BLE001 - one bad file must not lose the cell
            row["error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)

    # The worker is a CHILD of this cell process; an in-process cell loads the
    # models here instead, so that arm reports its own high-water mark.
    scope = resource.RUSAGE_SELF if spec.get("via") == "inprocess" else resource.RUSAGE_CHILDREN
    usage = resource.getrusage(scope).ru_maxrss
    # macOS reports bytes, Linux kilobytes.
    peak_mb = usage / (1 << 20) if sys.platform == "darwin" else usage / 1024.0
    return {
        "spec": {k: v for k, v in spec.items() if k != "files"},
        "rows": rows,
        "peak_rss_mb": peak_mb,
        "rss_scope": "self" if scope == resource.RUSAGE_SELF else "children",
    }


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def _probe(python: str, modules: list[str]) -> list[str]:
    """Modules `python` cannot import (so a cell is skipped, not crashed)."""
    code = (
        "import importlib.util,sys;"
        f"print(','.join(m for m in {modules!r} if importlib.util.find_spec(m) is None))"
    )
    try:
        out = subprocess.run([python, "-c", code], capture_output=True, text=True, timeout=180, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"<{type(exc).__name__}>"]
    if out.returncode != 0:
        return [f"<probe failed: {out.stderr.strip().splitlines()[-1:] or ''}>"]
    return [m for m in out.stdout.strip().split(",") if m]


def _stage_models(source: Path, work: Path, embedder: str, segmentation: str) -> Path:
    """A per-cell models dir: the embedder plus the chosen segmentation file
    under the float file's name (the only name `model_paths` looks for)."""
    from tldw_chatbook.Audio import diarizer_engine_onnx as engine

    dest = work / f"{embedder}-{segmentation}"
    dest.mkdir(parents=True, exist_ok=True)
    seg = source / (engine.SEGMENTATION.file_name if segmentation == "float" else engine.SEGMENTATION_INT8.file_name)
    for src, name in ((seg, engine.SEGMENTATION.file_name), (source / engine.EMBEDDERS[embedder].file_name, None)):
        target = dest / (name or src.name)
        if not target.exists():
            target.symlink_to(src)
    return dest


def ensure_int8_segmentation(models_dir: Path) -> Path:
    """Extract `model.int8.onnx` from the segmentation release tarball.

    `ensure_models` only ever fetches the float member (it is the only one
    `model_paths` plans for), so the bake-off's int8 arm fetches the same
    tarball and extracts the other member through the engine's own,
    size-checked extractor.
    """
    from tldw_chatbook.Audio import diarizer_engine_onnx as engine

    dest = models_dir / engine.SEGMENTATION_INT8.file_name
    if dest.is_file() and dest.stat().st_size == engine.SEGMENTATION_INT8.size:
        return dest
    tar = models_dir / "segmentation.tar.bz2"
    if not tar.is_file():
        corpus._download(engine.SEGMENTATION_INT8.url, tar)
    engine._extract_tar_member(tar, "model.int8.onnx", dest, engine.SEGMENTATION_INT8.size)
    return dest


def _machine() -> dict:
    import multiprocessing

    info = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": multiprocessing.cpu_count(),
        "python": platform.python_version(),
    }
    try:
        info["commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30, check=False,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        info["commit"] = "unknown"
    if sys.platform == "darwin":
        try:
            info["cpu_brand"] = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=30, check=False,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return info


def _sherpa_version(python: str) -> str:
    try:
        out = subprocess.run(
            [python, "-c", "import sherpa_onnx;print(sherpa_onnx.__version__)"],
            capture_output=True, text=True, timeout=180, check=False,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _aggregate(rows: list[dict], field: str) -> float | None:
    values = [r[field] for r in rows if isinstance(r.get(field), (int, float))]
    return statistics.mean(values) if values else None


def _fmt(value, digits=3) -> str:
    if value is None:
        return "--"
    return f"{value:.{digits}f}"


#: spec §7's go/no-go, as (name, comparison) pairs. All must hold for the
#: chosen embedder before `AUTO_ORDER` may put ONNX first.
LATENCY_CEILING_MS = 150.0      # M-series; the runner's own ceiling is 300 ms
RTF_CEILING = 0.15
DER_SLACK = 0.02
PURITY_SLACK = 0.03
SEPARATION_SLACK = 0.05


def _cell_summary(cell: dict) -> dict:
    """The per-cell aggregates the gate table compares."""
    rows = [r for r in cell["rows"] if "error" not in r]
    latencies = [v for r in rows for v in r.get("latency_ms", [])]
    separations = [
        r["separation"]["separation"] for r in rows
        if isinstance(r.get("separation"), dict) and isinstance(r["separation"].get("separation"), (int, float))
    ]
    cluster_errors = [
        abs(r["clusters"] - r["ref_speakers"]) for r in rows
        if isinstance(r.get("clusters"), int) and isinstance(r.get("ref_speakers"), int)
    ]
    return {
        "der": _aggregate(rows, "der"),
        # How far the live pass's cluster count sits from the reference's
        # speaker count: purity saturates at 1.0 well before the clusterer
        # stops over-splitting, so it is the tie-break, not decoration.
        "cluster_error": statistics.mean(cluster_errors) if cluster_errors else None,
        "rtf": _aggregate(rows, "rtf"),
        "purity": _aggregate(rows, "purity"),
        "coverage": _aggregate(rows, "coverage"),
        "latency_median_ms": statistics.median(latencies) if latencies else None,
        "latency_p95_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None,
        "separation": statistics.mean(separations) if separations else None,
        "peak_rss_mb": cell.get("peak_rss_mb"),
        "cluster_threshold": cell["spec"].get("cluster_threshold"),
        "live_threshold": cell["spec"].get("live_threshold"),
    }


def _best_cells(results: dict) -> dict:
    """`{(engine, embedder): {"stop": summary, "live": summary}}`.

    "Best" is the lowest mean DER over the cluster-threshold sweep and the
    highest mean live purity over the live-threshold sweep -- the two knobs
    the harness sweeps independently.
    """
    best: dict[tuple[str, str], dict] = {}
    for cell in results["cells"]:
        spec = cell["spec"]
        if spec.get("segmentation", "float") != "float" or spec.get("via", "worker") != "worker":
            continue
        key = (spec["engine"], spec["embedder"])
        summary = _cell_summary(cell)
        slot = best.setdefault(key, {})
        if spec["kind"] == "stop" and summary["der"] is not None and (
            slot.get("stop") is None or summary["der"] < slot["stop"]["der"]
        ):
            slot["stop"] = summary
        if spec["kind"] == "live" and summary["purity"] is not None:
            current = slot.get("live")
            better = current is None or summary["purity"] > current["purity"] or (
                summary["purity"] == current["purity"]
                and (summary["cluster_error"] or 0.0) < (current["cluster_error"] or 0.0)
            )
            if better:
                slot["live"] = summary
    return best


def _gate_rows(results: dict) -> list[list[str]]:
    """One row per ONNX embedder: each spec §7 gate, PASS/FAIL against ECAPA."""
    best = _best_cells(results)
    baseline = best.get(("speechbrain", ECAPA), {})
    base_stop, base_live = baseline.get("stop"), baseline.get("live")
    rows: list[list[str]] = []
    for (engine, embedder), slot in sorted(best.items()):
        if engine != "onnx":
            continue
        stop, live = slot.get("stop"), slot.get("live")

        def verdict(value, limit, better_is_lower=True):
            if value is None or limit is None:
                return "n/a"
            return "PASS" if (value <= limit if better_is_lower else value >= limit) else "FAIL"

        der_limit = None if base_stop is None or base_stop["der"] is None else base_stop["der"] + DER_SLACK
        purity_floor = None if base_live is None or base_live["purity"] is None else base_live["purity"] - PURITY_SLACK
        sep_floor = (
            None if base_stop is None or base_stop["separation"] is None
            else base_stop["separation"] - SEPARATION_SLACK
        )
        rows.append([
            embedder,
            f"{_fmt(stop and stop['der'])} ({verdict(stop and stop['der'], der_limit)})",
            f"{_fmt(live and live['purity'])} ({verdict(live and live['purity'], purity_floor, False)})",
            f"{_fmt(stop and stop['rtf'])} ({verdict(stop and stop['rtf'], RTF_CEILING)})",
            (f"{_fmt(live and live['latency_median_ms'], 1)} ms "
             f"({verdict(live and live['latency_median_ms'], LATENCY_CEILING_MS)})"),
            f"{_fmt(stop and stop['separation'])} ({verdict(stop and stop['separation'], sep_floor, False)})",
            f"cluster {_fmt(stop and stop['cluster_threshold'], 2)} / live {_fmt(live and live['live_threshold'], 2)}",
        ])
    return rows


def _report(results: dict) -> str:
    lines: list[str] = []
    add = lines.append
    machine = results["machine"]
    add("# TASK-31827 diarizer bake-off: ONNX vs SpeechBrain (spec §7)")
    add("")
    add(f"- Commit: `{machine['commit']}`")
    add(f"- Machine: {machine.get('cpu_brand', machine['processor'])}, "
        f"{machine['cpu_count']} cores, {machine['platform']}, Python {machine['python']}")
    add(f"- sherpa-onnx: {results['sherpa_onnx']}")
    add(f"- Corpus: {results['corpus']['files']} files "
        f"({results['corpus']['voxconverse']} VoxConverse dev + {results['corpus']['ami']} AMI dev), "
        f"{results['corpus']['seconds'] / 60.0:.0f} minutes of audio")
    add(f"- Collar: {COLLAR_S} s. Live window: {WINDOW_S} s. max_speakers: {MAX_SPEAKERS}.")
    add("")
    add("## Model hashes")
    add("")
    add("| asset | sha256 (first 12) | bytes |")
    add("| --- | --- | --- |")
    for name, digest, size in results["models"]:
        add(f"| {name} | `{digest}` | {size} |")
    add("")

    gate_rows = _gate_rows(results)
    if gate_rows:
        best = _best_cells(results)
        base = best.get(("speechbrain", ECAPA), {})
        add("## Go/no-go (spec §7) -- best cell per embedder vs the ECAPA baseline")
        add("")
        add(f"Baseline (SpeechBrain/ECAPA): DER {_fmt(base.get('stop', {}).get('der'))}, "
            f"purity {_fmt(base.get('live', {}).get('purity'))}, "
            f"RTF {_fmt(base.get('stop', {}).get('rtf'))}, "
            f"latency {_fmt(base.get('live', {}).get('latency_median_ms'), 1)} ms, "
            f"separation {_fmt(base.get('stop', {}).get('separation'))}.")
        add("")
        add(f"Gates: DER within {DER_SLACK} absolute, purity within {PURITY_SLACK}, "
            f"RTF <= {RTF_CEILING}, embed latency <= {LATENCY_CEILING_MS:.0f} ms (M-series) / 300 ms (runner), "
            f"separation within {SEPARATION_SLACK}.")
        add("")
        add("| embedder | DER | live purity | RTF | embed latency | separation | best thresholds |")
        add("| --- | --- | --- | --- | --- | --- | --- |")
        for row in gate_rows:
            add("| " + " | ".join(row) + " |")
        add("")

    add("## Stop-pass DER, RTF and peak worker RSS")
    add("")
    add("| engine | embedder | segmentation | via | cluster threshold | DER | RTF | peak RSS (MB) | files |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["cells"]:
        spec = cell["spec"]
        if spec["kind"] != "stop":
            continue
        rows = [r for r in cell["rows"] if "error" not in r]
        add(f"| {spec['engine']} | {spec['embedder']} | {spec.get('segmentation', '--')} | "
            f"{spec.get('via', 'worker')} | {_fmt(spec.get('cluster_threshold'), 2)} | "
            f"{_fmt(_aggregate(rows, 'der'))} | {_fmt(_aggregate(rows, 'rtf'))} | "
            f"{_fmt(cell['peak_rss_mb'], 1)} | {len(rows)}/{len(cell['rows'])} |")
    add("")

    add("## Live purity / coverage and per-window embed latency")
    add("")
    add("| engine | embedder | live threshold | purity | coverage | clusters vs speakers | "
        "latency median (ms) | p95 (ms) | windows |")
    add("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["cells"]:
        spec = cell["spec"]
        if spec["kind"] != "live":
            continue
        rows = [r for r in cell["rows"] if "error" not in r]
        latencies = [v for r in rows for v in r.get("latency_ms", [])]
        median = statistics.median(latencies) if latencies else None
        p95 = statistics.quantiles(latencies, n=20)[18] if len(latencies) >= 20 else None
        add(f"| {spec['engine']} | {spec['embedder']} | {_fmt(spec.get('live_threshold'), 2)} | "
            f"{_fmt(_aggregate(rows, 'purity'))} | {_fmt(_aggregate(rows, 'coverage'))} | "
            f"+{_fmt(_cell_summary(cell)['cluster_error'], 2)} | "
            f"{_fmt(median, 1)} | {_fmt(p95, 1)} | {len(latencies)} |")
    add("")

    add("## Self-match separation (VoxConverse speakers)")
    add("")
    add("| engine | embedder | cluster threshold | self cos | best other cos | separation | recommended voice_match_threshold |")
    add("| --- | --- | --- | --- | --- | --- | --- |")
    for cell in results["cells"]:
        spec = cell["spec"]
        if spec["kind"] != "stop":
            continue
        found = [r["separation"] for r in cell["rows"] if isinstance(r.get("separation"), dict)]
        if not found:
            continue
        def _mean(key, found=found):
            values = [f[key] for f in found if isinstance(f.get(key), (int, float))]
            return statistics.mean(values) if values else None
        add(f"| {spec['engine']} | {spec['embedder']} | {_fmt(spec.get('cluster_threshold'), 2)} | "
            f"{_fmt(_mean('self_similarity'))} | {_fmt(_mean('other_similarity'))} | "
            f"{_fmt(_mean('separation'))} | {_fmt(_mean('recommended_threshold'))} |")
    add("")

    if results["skipped"]:
        add("## Not run")
        add("")
        for line in results["skipped"]:
            add(f"- {line}")
        add("")
    return "\n".join(lines) + "\n"


def _cells_for(args, files: list[dict], models_root: Path, work: Path) -> list[dict]:
    cells: list[dict] = []
    for engine in args.engines:
        if engine == "onnx":
            for embedder in args.embedders:
                for segmentation in args.segmentation:
                    via = "worker" if segmentation == "float" else "inprocess"
                    for threshold in args.cluster_thresholds:
                        cells.append({
                            "kind": "stop", "engine": engine, "embedder": embedder,
                            "segmentation": segmentation, "via": via,
                            "cluster_threshold": threshold,
                            "models_dir": str(_stage_models(models_root, work, embedder, segmentation)),
                            "files": files,
                        })
                    if args.inprocess_float_control and segmentation == "float":
                        cells.append({
                            "kind": "stop", "engine": engine, "embedder": embedder,
                            "segmentation": "float", "via": "inprocess",
                            "cluster_threshold": args.cluster_thresholds[0],
                            "models_dir": str(_stage_models(models_root, work, embedder, "float")),
                            "files": files,
                        })
                for threshold in args.live_thresholds:
                    cells.append({
                        "kind": "live", "engine": engine, "embedder": embedder,
                        "live_threshold": threshold,
                        "models_dir": str(_stage_models(models_root, work, embedder, "float")),
                        "files": files,
                    })
        else:
            # The SpeechBrain baseline has no threshold knobs: the Stop pass
            # picks its own speaker count (silhouette) and the live path runs
            # at the 0.25 it has always used.
            cells.append({"kind": "stop", "engine": engine, "embedder": ECAPA, "files": files})
            cells.append({"kind": "live", "engine": engine, "embedder": ECAPA, "files": files})
    return cells


def _merge(paths: list[str], out_path: Path) -> int:
    """Render several runs' `results.json` as one report.

    Each embedder turns out to have its own cosine scale (measured: the live
    optimum is 0.45 for titanet_small and 0.10 for wespeaker_resnet34), so the
    final numbers come from one narrow run per embedder rather than one grid
    swept over all of them. This stitches those runs back together; the corpus
    and machine come from the first, and a mismatch in either is recorded in
    "Not run" rather than silently averaged away.
    """
    merged: dict | None = None
    for raw in paths:
        loaded = json.loads(Path(raw.strip()).read_text())
        if merged is None:
            merged = loaded
            continue
        if loaded["corpus"]["ids"] != merged["corpus"]["ids"]:
            merged["skipped"].append(
                f"merge: {Path(raw).name} ran a different corpus ({len(loaded['corpus']['ids'])} files) "
                "-- its cells are included but are NOT comparable"
            )
        merged["cells"].extend(loaded["cells"])
        merged["skipped"].extend(loaded["skipped"])
        for asset in loaded["models"]:
            if asset not in merged["models"]:
                merged["models"].append(asset)
    if merged is None:
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.with_name("results.json").write_text(json.dumps(merged, indent=2))
    out_path.write_text(_report(merged))
    print(f"wrote {out_path} from {len(paths)} runs")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cell", help="internal: run the cell in this JSON file and print the result")
    parser.add_argument("--cell-out", help="internal: where --cell writes its JSON")
    parser.add_argument("--engines", default="onnx,speechbrain")
    parser.add_argument("--embedders", default="titanet_small,wespeaker_resnet34,eres2net_en,campplus_en")
    parser.add_argument("--segmentation", default="float,int8")
    parser.add_argument("--live-thresholds", default="0.25,0.35,0.45,0.55,0.65")
    parser.add_argument("--cluster-thresholds", default="0.4,0.5,0.6,0.7,0.8")
    parser.add_argument("--files", default="all", help="'all' or a comma-separated list of corpus ids")
    parser.add_argument("--models-dir", default=None, help="where the ONNX models are (fetched if missing)")
    parser.add_argument("--corpus-dir", default=None, help="a prepared corpus to link from")
    parser.add_argument("--speechbrain-python", default=sys.executable,
                        help="interpreter with torch/speechbrain (never install torch into the app venv)")
    parser.add_argument("--inprocess-float-control", action="store_true",
                        help="also measure float segmentation in-process, so the int8 delta is apples-to-apples")
    parser.add_argument("--merge", default=None,
                        help="comma-separated results.json paths to render as ONE report and results.json "
                             "(each embedder has its own threshold scale, so the final numbers come from "
                             "one run per embedder rather than one grid over all of them)")
    parser.add_argument("--out", default="Docs/STT_Evaluation/task-31827/report.md")
    args = parser.parse_args(argv)

    if args.merge:
        return _merge(args.merge.split(","), Path(args.out))

    if args.cell:
        result = run_cell(json.loads(Path(args.cell).read_text()))
        Path(args.cell_out).write_text(json.dumps(result))
        return 0

    args.engines = [e for e in args.engines.split(",") if e]
    args.embedders = [e for e in args.embedders.split(",") if e]
    args.segmentation = [s for s in args.segmentation.split(",") if s]
    args.live_thresholds = [float(v) for v in args.live_thresholds.split(",") if v]
    args.cluster_thresholds = [float(v) for v in args.cluster_thresholds.split(",") if v]

    from tldw_chatbook.Audio import diarizer_engine_onnx as engine

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = corpus.DATA_DIR / "run"
    work.mkdir(parents=True, exist_ok=True)
    models_root = Path(args.models_dir) if args.models_dir else corpus.DATA_DIR / "models"
    models_root.mkdir(parents=True, exist_ok=True)

    subset = None if args.files == "all" else [f for f in args.files.split(",") if f]
    entries = corpus.materialise(scratch=Path(args.corpus_dir) if args.corpus_dir else None, subset=subset)
    files = [{"id": e.id, "source": e.source, "wav": str(e.wav), "rttm": str(e.rttm)} for e in entries]
    seconds = sum(corpus.check_wav(e.wav) for e in entries)

    skipped: list[str] = []
    missing = _probe(sys.executable, ["sherpa_onnx", "numpy"])
    if "onnx" in args.engines and missing:
        skipped.append(f"engine onnx: not run (missing: {', '.join(missing)})")
        args.engines = [e for e in args.engines if e != "onnx"]
    sb_missing = _probe(args.speechbrain_python, ["torch", "torchaudio", "speechbrain", "sklearn"])
    if "speechbrain" in args.engines and sb_missing:
        skipped.append(f"engine speechbrain: not run (missing: {', '.join(sb_missing)})")
        args.engines = [e for e in args.engines if e != "speechbrain"]

    if "onnx" in args.engines:
        for embedder in args.embedders:
            engine.ensure_models(embedder, models_dir_override=models_root)
        if "int8" in args.segmentation:
            ensure_int8_segmentation(models_root)

    cells = _cells_for(args, files, models_root, work)
    results = {
        "machine": _machine(),
        "sherpa_onnx": _sherpa_version(sys.executable),
        "corpus": {
            "files": len(entries),
            "voxconverse": sum(1 for e in entries if e.source == "voxconverse"),
            "ami": sum(1 for e in entries if e.source == "ami"),
            "seconds": seconds,
            "ids": [e.key for e in entries],
        },
        "models": [
            (asset.file_name, asset.sha256[:12], asset.size)
            for asset in [engine.SEGMENTATION, engine.SEGMENTATION_INT8]
            + [engine.EMBEDDERS[k] for k in args.embedders if k in engine.EMBEDDERS]
        ],
        "cells": [],
        "skipped": skipped,
    }

    for index, cell in enumerate(cells, 1):
        python = args.speechbrain_python if cell["engine"] == "speechbrain" else sys.executable
        label = (f"{cell['kind']} {cell['engine']}/{cell['embedder']}"
                 f"/{cell.get('segmentation', '-')}/{cell.get('via', 'worker')}"
                 f" live={cell.get('live_threshold')} cluster={cell.get('cluster_threshold')}")
        spec_path = work / f"cell-{index}.json"
        result_path = work / f"cell-{index}-out.json"
        spec_path.write_text(json.dumps(cell))
        started = time.time()
        print(f"[{index}/{len(cells)}] {label} ...", flush=True)
        try:
            done = subprocess.run(
                [python, str(Path(__file__).resolve()), "--cell", str(spec_path), "--cell-out", str(result_path)],
                capture_output=True, text=True, timeout=CELL_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired:
            skipped.append(f"{label}: cell timed out after {CELL_TIMEOUT_S:.0f} s")
            continue
        if done.returncode != 0 or not result_path.is_file():
            tail = (done.stderr or "").strip().splitlines()[-3:]
            skipped.append(f"{label}: cell failed ({' | '.join(tail)})")
            continue
        cell_result = json.loads(result_path.read_text())
        results["cells"].append(cell_result)
        print(f"    done in {time.time() - started:.0f} s", flush=True)
        out_path.with_name("results.json").write_text(json.dumps(results, indent=2))
        out_path.write_text(_report(results))

    out_path.with_name("results.json").write_text(json.dumps(results, indent=2))
    out_path.write_text(_report(results))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
