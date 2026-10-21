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
