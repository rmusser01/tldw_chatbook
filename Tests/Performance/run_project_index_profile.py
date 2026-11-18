"""Generated-only bounded index profile; no application or Library startup.

Run with --files 20000 --directories 20000 --wide-files 20000. The harness uses
an on-disk frontier and discards each page synchronously, like the durable owner.
"""

from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import asdict
import gc
import json
import os
from pathlib import Path
import platform
import sqlite3
import tempfile
import threading
import time

import psutil

from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexAuthority,
    ProjectIndexKey,
    ProjectIndexPolicy,
    ProjectIndexScope,
    locator_fingerprint,
    root_identity_fingerprint,
)
from tldw_chatbook.Workspaces.project_index.scanner import ProjectScanner


def make_authority(root):
    chain = capture_directory_chain(root)
    key = ProjectIndexKey(
        "profile",
        "generated",
        "binding",
        locator_fingerprint(root),
        root_identity_fingerprint(chain),
    )
    return ProjectIndexAuthority(key, root, chain, (), 1)


def measure(root, frontier_file):
    scanner = ProjectScanner()
    auth = make_authority(root)
    process = psutil.Process()
    peaks = {
        "parent_rss": process.memory_info().rss,
        "worker_rss": 0,
        "combined_rss": 0,
    }
    finished = threading.Event()
    page_count = max_entries = captured_bytes = 0
    observed_worker_pids = set()
    handles_before = process.num_handles() if os.name == "nt" else process.num_fds()

    def sample():
        while not finished.wait(0.01):
            parent_rss = process.memory_info().rss
            child_rss = 0
            try:
                if scanner.worker_pid:
                    worker = psutil.Process(scanner.worker_pid)
                    workers = [worker, *worker.children(recursive=True)]
                    for child in workers:
                        observed_worker_pids.add(child.pid)
                        child_rss += child.memory_info().rss
            except psutil.Error:
                pass
            peaks["parent_rss"] = max(peaks["parent_rss"], parent_rss)
            peaks["worker_rss"] = max(peaks["worker_rss"], child_rss)
            peaks["combined_rss"] = max(peaks["combined_rss"], parent_rss + child_rss)

    sampler = threading.Thread(target=sample)
    sampler.start()
    with closing(sqlite3.connect(frontier_file)) as pending:
        pending.execute("CREATE TABLE pending(path TEXT PRIMARY KEY)")
        pending.execute("INSERT INTO pending VALUES('')")

        def frontier():
            while row := pending.execute("SELECT path FROM pending LIMIT 1").fetchone():
                pending.execute("DELETE FROM pending WHERE path=?", row)
                yield row[0]

        def emit(page):
            nonlocal page_count, max_entries, captured_bytes
            page_count += 1
            max_entries = max(max_entries, len(page.entries))
            captured_bytes += sum(c.size_bytes for c in page.captures)
            pending.executemany(
                "INSERT OR IGNORE INTO pending VALUES(?)",
                ((e.relative_path,) for e in page.entries if e.kind == "directory"),
            )
            if page_count % 1000 == 0:
                pending.commit()
            return True

        start = time.monotonic()
        try:
            result = scanner.scan(
                auth,
                ProjectIndexScope(auth.key, 1, 1),
                "profile",
                frontier(),
                emit,
                threading.Event(),
                admit_unit=lambda _: auth,
                policy=ProjectIndexPolicy(("",), (".txt",)),
            )
        finally:
            elapsed = time.monotonic() - start
            finished.set()
            sampler.join()
    gc.collect()
    handles_after = process.num_handles() if os.name == "nt" else process.num_fds()
    report = {
        **asdict(result),
        **peaks,
        "seconds": elapsed,
        "entries_per_second": result.discovered / elapsed,
        "pages": page_count,
        "max_page_entries": max_entries,
        "captured_bytes": captured_bytes,
        "max_queued_text_bytes": scanner.max_queued_text_bytes,
        "max_active_reads": scanner.max_active_reads,
        "max_frame_bytes": scanner.max_frame_bytes,
        "worker_pid": scanner.worker_pid,
        "worker_gone": not psutil.pid_exists(scanner.worker_pid),
        "observed_worker_pids": sorted(observed_worker_pids),
        "entire_observed_tree_gone": all(
            not psutil.pid_exists(pid) for pid in observed_worker_pids
        ),
        "retired": scanner.retired,
        "pipes_closed": scanner.pipes_closed,
        "handles_before": handles_before,
        "handles_after": handles_after,
        "frontier": "SQLite on-disk rows; one fetched path; never a tree-sized Python list",
    }
    assert result.status == "complete", report
    assert (
        max_entries <= 256
        and scanner.max_active_reads <= 2
        and scanner.max_queued_text_bytes <= 4 * 1024**2
    )
    assert scanner.retired and report["worker_gone"] and scanner.pipes_closed
    return report


def cancellation(root):
    auth = make_authority(root)
    scanner = ProjectScanner()
    stop = threading.Event()
    result = []

    def run():
        result.append(
            scanner.scan(
                auth,
                ProjectIndexScope(auth.key, 1, 1),
                "cancel",
                iter([""]),
                lambda _: True,
                stop,
                admit_unit=lambda _: auth,
                policy=ProjectIndexPolicy(("",), (".txt",)),
            )
        )

    thread = threading.Thread(target=run)
    thread.start()
    deadline = time.monotonic() + 10
    while not scanner.worker_pid and time.monotonic() < deadline:
        time.sleep(0.001)
    start = time.monotonic()
    stop.set()
    thread.join(10)
    elapsed = time.monotonic() - start
    report = {
        "seconds": elapsed,
        "status": result[0].status if result else "no_result",
        "retired": scanner.retired,
        "worker_gone": not psutil.pid_exists(scanner.worker_pid),
        "pipes_closed": scanner.pipes_closed,
    }
    assert not thread.is_alive() and scanner.retired and report["worker_gone"], report
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("files", "directories", "wide-files"):
        parser.add_argument("--" + name, type=int, default=20000)
    args = parser.parse_args()
    metadata = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "logical_cpus": os.cpu_count(),
        "physical_ram": psutil.virtual_memory().total,
        "corpora": vars(args),
        "rss_sampling_seconds": 0.01,
        "limits": "Sampled RSS (not exact native peak); synthetic local NTFS/POSIX only; no remote qualification.",
    }
    print(json.dumps({"metadata": metadata}), flush=True)
    with tempfile.TemporaryDirectory(prefix="project-index-profile-") as temporary:
        base = Path(temporary)
        for name, count in [
            ("files", args.files),
            ("directories", args.directories),
            ("wide", args.wide_files),
        ]:
            root = base / name
            root.mkdir()
            if name == "files":
                for group in range(100):
                    (root / f"g{group:03}").mkdir()
                for i in range(count):
                    (root / f"g{i % 100:03}" / f"{i:06}.txt").write_text(
                        f"generated project keyword {i}\n", encoding="utf-8"
                    )
            elif name == "directories":
                for i in range(count):
                    (root / f"d{i:06}").mkdir()
            else:
                for i in range(count):
                    (root / f"{i:06}.txt").write_text(
                        f"wide marker {i}\n", encoding="utf-8"
                    )
            report = measure(root, base / f"{name}-frontier.sqlite")
            report["cancellation"] = cancellation(root)
            print(json.dumps({"corpus": name, **report}), flush=True)


if __name__ == "__main__":
    main()


def measure_pipeline(
    app,
    artifact_root,
    shape,
    *,
    count=20000,
    poll_status=True,
    timing_load="Unspecified observer load; qualify with the run source manifest",
):
    """Qualify generated corpora through runtime, scanner, store and bounded tool.

    Unlike the historical scanner-only receipts above, this retains one stable
    corpus and the application's admitted cache until explicitly cleared.
    """
    from types import SimpleNamespace
    from Tests.Agents.test_project_index_tool import provider, invoke
    from tldw_chatbook.Workspaces.project_index.runtime import (
        get_project_index_service,
        project_index_key,
    )
    from tldw_chatbook.DB.Workspace_Project_Index_DB import index_store_path

    artifact_root = Path(artifact_root)
    root = artifact_root / shape
    root.mkdir(parents=True, exist_ok=True)
    for number in range(count):
        directory = (
            root if shape == "wide" else root / f"g{number // 100:03}" / f"d{number:05}"
        )
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"f{number:05}.txt"
        if not target.exists():
            target.write_text(
                f"generated file {number}\n"
                + (
                    "qualificationdeepmarker\n"
                    if number == count - 1
                    else "ordinary text\n"
                ),
                encoding="utf-8",
            )
    registry = app.workspace_registry_service
    workspace = f"qualification-{shape}"
    registry.create_workspace(workspace_id=workspace, name=f"Generated {shape}")
    binding = registry.add_folder_binding(workspace, root)
    service = get_project_index_service(app)
    key = project_index_key(app, workspace, binding.binding_id, root)
    initial = service.status(key)
    enabled = service.enable(
        key, ProjectIndexPolicy(("",), (".txt",)), expected_scope=initial.scope
    )
    process = psutil.Process()
    baseline = process.memory_info().rss
    peaks = {"parent_rss": baseline, "combined_rss": baseline, "whole_binding_bytes": 0}
    observed = set()
    stop = threading.Event()
    cache = index_store_path(service._factory.data_root, key).parent.parent
    progress_path = artifact_root / f"{shape}-progress.json"

    def sample():
        while not stop.wait(1):
            parent = process.memory_info().rss
            child_bytes = 0
            for child in process.children(recursive=True):
                try:
                    observed.add(child.pid)
                    child_bytes += child.memory_info().rss
                except psutil.Error:
                    pass
            peaks["parent_rss"] = max(peaks["parent_rss"], parent)
            peaks["combined_rss"] = max(peaks["combined_rss"], parent + child_bytes)
            from tldw_chatbook.Workspaces.project_index.service import binding_bytes

            peaks["whole_binding_bytes"] = max(
                peaks["whole_binding_bytes"], binding_bytes(cache)
            )
            progress_path.write_text(
                json.dumps(
                    {
                        "elapsed": time.monotonic() - start,
                        "observer": "passive RSS/disk sampler; no registry status polling",
                        "peaks_sampled_1s": peaks,
                        "child_pids": sorted(observed),
                    }
                ),
                encoding="utf-8",
            )

    sampler = threading.Thread(target=sample)
    start = time.monotonic()
    sampler.start()
    (artifact_root / f"{shape}-launch.json").write_text(
        json.dumps(
            {
                "key": asdict(key),
                "enabled": asdict(enabled),
                "cache": str(cache),
                "corpus": str(root),
                "pid": os.getpid(),
                "timing_load": timing_load,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    job = service.refresh(
        key, expected_scope=enabled.scope, expected_job_revision=enabled.job_revision
    )
    try:
        if not poll_status:
            service._runs[job].supervisor.join()
        while True:
            status = service.status(key)
            progress_path.write_text(
                json.dumps(
                    {
                        "elapsed": time.monotonic() - start,
                        "status": asdict(status),
                        "peaks_sampled_1s": peaks,
                    }
                ),
                encoding="utf-8",
            )
            if status.state != "Building":
                break
            time.sleep(2)
        elapsed = time.monotonic() - start
    finally:
        stop.set()
        sampler.join()
    run = service._runs[job]
    run.supervisor.join()
    assert status.state == "Ready" and status.indexed == count, status
    assert run.native_retired and run.db_closed and run.pin.closed
    rig = SimpleNamespace(service=service, root=root, key=key)
    tool = provider(rig)
    result = invoke(tool, {"query": "qualificationdeepmarker"})
    payload = json.loads(result.content)
    assert (
        result.ok
        and len(payload["hits"]) == 1
        and "qualificationdeepmarker" in result.content
    ), result
    files = sum(1 for path in root.rglob("*.txt"))
    directories = sum(1 for path in root.rglob("*") if path.is_dir())
    assert files >= count and (shape == "wide" or directories >= count)
    report = {
        "pipeline": "production runtime/service/scanner/store/LocalToolProvider/Capture",
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "shape": shape,
        "text_files": files,
        "folders": directories,
        "seconds": elapsed,
        "files_per_second": files / elapsed,
        "baseline_parent_rss": baseline,
        "peaks_sampled_1s": peaks,
        "peak_extra_combined_rss_sampled": peaks["combined_rss"] - baseline,
        "final_status": asdict(status),
        "tool_payload": payload,
        "worker_retired": run.native_retired,
        "writer_db_closed": run.db_closed,
        "storage_pin_closed": run.pin.closed,
        "scanner_pipes_closed": service._scanner.pipes_closed,
        "observed_child_pids": sorted(observed),
        "observed_children_gone": all(not psutil.pid_exists(pid) for pid in observed),
        "whole_binding_files": {
            str(p.relative_to(cache)): p.stat().st_size
            for p in cache.rglob("*")
            if p.is_file()
        },
        "timing_load": timing_load,
        "tool_authority_fixture": "LocalToolProvider roots/guard are test-seeded; runtime admission, registry fences and native Capture are real",
        "scanner_max_frame_bytes": service._scanner.max_frame_bytes,
        "scanner_max_queued_text_bytes": service._scanner.max_queued_text_bytes,
        "scanner_max_active_reads": service._scanner.max_active_reads,
        "writer_registry_closed": run.registry_closed,
    }
    (artifact_root / f"{shape}-pipeline.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return rig, report
