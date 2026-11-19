"""Real process tests for bounded project scanning."""

from dataclasses import replace
import importlib
import threading

import pytest

from Tests.private_profile import private_profile_test

from tldw_chatbook.Utils.filesystem_identity import capture_directory_chain
from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion
from tldw_chatbook.Workspaces.project_index.models import (
    ProjectIndexAuthority,
    ProjectIndexKey,
    ProjectIndexScope,
    ProjectIndexPolicy,
    locator_fingerprint,
    root_identity_fingerprint,
)


def authority(root):
    chain = capture_directory_chain(root)
    key = ProjectIndexKey(
        "test",
        "workspace",
        "binding",
        locator_fingerprint(root),
        root_identity_fingerprint(chain),
    )
    return ProjectIndexAuthority(key, root, chain, (), 1)


def scanner_class():
    return importlib.import_module(
        "tldw_chatbook.Workspaces.project_index.scanner"
    ).ProjectScanner


def scan(root, *, frontier=None, admit=None, emit=None, policy=None):
    auth = authority(root)
    scanner = scanner_class()()
    pages = []
    result = scanner.scan(
        auth,
        ProjectIndexScope(auth.key, 1, 1),
        "job",
        iter(frontier or [""]),
        emit or (lambda page: pages.append(page) is None),
        threading.Event(),
        admit_unit=admit or (lambda _: auth),
        policy=policy or ProjectIndexPolicy(("",), (".txt", ".py")),
    )
    return scanner, result, pages


def test_wide_directory_is_not_listing_truncated(tmp_path):
    tmp_path = tmp_path / "project"
    tmp_path.mkdir()
    for i in range(10003):
        (tmp_path / f"{i:05}.txt").write_text(f"marker-{i}", encoding="utf-8")
    scanner, result, pages = scan(tmp_path)
    assert result.status == "complete"
    assert result.discovered == 10003
    assert {c.text for p in pages for c in p.captures} >= {"marker-201", "marker-10002"}
    assert all(len(p.entries) <= 256 for p in pages)
    assert scanner.max_queued_text_bytes <= 4 * 1024 * 1024
    assert scanner.max_active_reads <= 2
    assert scanner.retired and scanner.worker_pid > 0


def test_filters_policy_and_live_capture_use_same_descriptor(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "ok.py").write_text('print("hello")', encoding="utf-8")
    for name, data in [
        ("bad.txt", b"\xff"),
        ("binary.txt", b"x\x00y"),
        ("large.txt", b"x" * (1024**2 + 1)),
    ]:
        (tmp_path / name).write_bytes(data)
    for dirname in ["node_modules", ".git", "build", "secret"]:
        (tmp_path / dirname).mkdir()
        (tmp_path / dirname / "hidden.txt").write_text("secret")
    auth = replace(
        authority(tmp_path), exclusions=(SensitiveExclusion("subtree", "secret"),)
    )
    scanner, result, pages = scan(tmp_path, frontier=["", "src"], admit=lambda _: auth)
    assert result.status == "complete"
    assert [c.relative_path for p in pages for c in p.captures] == ["src/ok.py"]
    live = scanner.capture_file(auth, "src/ok.py", stop=threading.Event())
    assert live == next(c for p in pages for c in p.captures)
    assert (
        scanner.capture_file(auth, "secret/hidden.txt", stop=threading.Event()) is None
    )
    _, _, pages = scan(
        tmp_path, frontier=["", "src"], policy=ProjectIndexPolicy(("src",), (".py",))
    )
    assert {e.relative_path for p in pages for e in p.entries} == {"src", "src/ok.py"}


def test_escaped_megabyte_round_trips_without_frame_overflow(tmp_path):
    text = "\x01" * (1024**2)
    (tmp_path / "escaped.txt").write_bytes(text.encode())
    scanner, result, pages = scan(tmp_path)
    assert result.status == "complete"
    assert [c.text for p in pages for c in p.captures] == [text]
    assert scanner.max_frame_bytes <= 2 * 1024**2


def test_exclusion_revision_narrows_continuation_and_never_widens(tmp_path):
    for i in range(600):
        (tmp_path / f"{i:04}.txt").write_text("body")
    auth = authority(tmp_path)
    calls = []

    def admit(directory):
        calls.append(directory)
        return (
            auth
            if len(calls) == 1
            else replace(
                auth,
                authority_revision=2,
                exclusions=(SensitiveExclusion("subtree", ""),),
            )
        )

    scanner, result, pages = scan(tmp_path, admit=admit)
    assert result.status == "partial"
    assert len(calls) == 2
    assert len(pages[0].entries) == 256
    assert all(not p.entries for p in pages[1:])
    assert not any(p.directory_complete for p in pages)
    assert scanner.retired


def test_emit_rejection_and_stop_retire_actual_process(tmp_path):
    (tmp_path / "a.txt").write_text("a")
    scanner, result, _ = scan(tmp_path, emit=lambda _: False)
    assert result.status == "paused"
    assert scanner.retired
    assert scanner.process_returncode is not None
    # A retained directory/pin or inherited pipe must not survive retirement.
    moved = tmp_path.with_name(tmp_path.name + "-moved")
    tmp_path.rename(moved)
    moved.rename(tmp_path)


def test_changed_root_identity_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    auth = authority(root)
    root.rename(tmp_path / "old")
    root.mkdir()
    (root / "leak.txt").write_text("wrong incarnation")
    scanner = scanner_class()()
    assert scanner.capture_file(auth, "leak.txt", stop=threading.Event()) is None
    assert scanner.retired


@pytest.mark.skipif(
    __import__("os").name == "nt", reason="Windows disallows newline filenames"
)
def test_unicode_newline_filename_remains_one_field(tmp_path):
    name = "hello\n\u96ea.txt"
    (tmp_path / name).write_text("one")
    _, result, pages = scan(tmp_path)
    assert result.status == "complete"
    assert [c.relative_path for p in pages for c in p.captures] == [name]


def test_page_handoff_does_not_retain_previous_directory_text(tmp_path):
    # The emit callback sees immutable pages; the scanner must release a handoff
    # before the next directory admission (not just continuation admission).
    for name in ("one", "two"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.txt").write_bytes(b"a" * 1024**2)
    auth = authority(tmp_path)
    seen = []

    def emit(page):
        # slots intentionally disallow weakrefs: inspect reference count instead.
        if page.captures:
            seen.append(page.captures[0].text)
        else:
            assert page.directory_complete
        return True

    def admit(directory):
        if seen:
            import sys

            # one list reference + getrefcount argument only after handoff.
            references = sys.getrefcount(seen[-1])
            assert references == 2
        return auth

    scanner, result, _ = scan(tmp_path, frontier=["one", "two"], admit=admit, emit=emit)
    assert len(seen) == 2
    assert result.status == "complete"
    assert scanner.retired


def test_directory_mutation_never_certifies_completion(tmp_path):
    tmp_path = tmp_path / "project"
    tmp_path.mkdir()
    for i in range(300):
        (tmp_path / f"{i:04}.txt").write_text("before")
    pages = []

    def emit(page):
        pages.append(page)
        if len(pages) == 1:
            (tmp_path / "added.txt").write_text("mutation")
        return True

    scanner, result, _ = scan(tmp_path, emit=emit)
    assert result.status == "partial"
    assert "directory_changed" in result.issues
    assert not any(p.directory_complete for p in pages)
    assert scanner.retired
    # Restart from the directory beginning: never trust the old iterator offset.
    _, result, pages = scan(tmp_path)
    assert result.status == "complete" and result.discovered == 301
    assert any(c.relative_path == "added.txt" for p in pages for c in p.captures)


def test_capture_rejects_mutation_during_descriptor_read(tmp_path, monkeypatch):
    import os
    from tldw_chatbook.Tools import project_index_scan_worker as worker

    file = tmp_path / "race.txt"
    file.write_text("old")
    original = os.read
    changed = False

    def racing_read(fd, count):
        nonlocal changed
        result = original(fd, count)
        if not changed:
            changed = True
            file.write_text("replacement body")
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker.os, "read", racing_read)
    with pytest.raises(worker.UnsafeCapture):
        worker.capture_candidate("race.txt", ())


def test_symlink_or_junction_escape_is_never_captured(tmp_path):
    import os
    import subprocess

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "secret.txt").write_text("outside")
    link = root / "escape"
    if os.name == "nt":
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True
        )
        if completed.returncode:
            pytest.skip("OS denied private junction creation")
    else:
        link.symlink_to(outside, target_is_directory=True)
    try:
        scanner, result, pages = scan(root, frontier=["", "escape"])
        assert not any(p.captures for p in pages)
        assert result.status == "partial"
        assert (
            scanner.capture_file(
                authority(root), "escape/secret.txt", stop=threading.Event()
            )
            is None
        )
        assert (outside / "secret.txt").read_text() == "outside"
    finally:
        os.rmdir(link) if os.name == "nt" else link.unlink()


def test_current_witness_in_another_process_narrows_and_removal_does_not_widen(
    tmp_path,
):
    import json
    import subprocess
    import sys

    root = tmp_path / "root"
    root.mkdir()
    for name in ("first", "secret", "last"):
        (root / name).mkdir()
        (root / name / "content.txt").write_text(name)
    witness = tmp_path / "witness.json"
    witness.write_text("[1, []]")
    auth = authority(root)
    pages = []

    def admit(directory):
        revision, excludes = json.loads(witness.read_text())
        return replace(
            auth,
            authority_revision=revision,
            exclusions=tuple(SensitiveExclusion(*e) for e in excludes),
        )

    def update(revision, excludes):
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-c",
                "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text(sys.argv[2])",
                str(witness),
                json.dumps([revision, excludes]),
            ],
            check=True,
        )

    def emit(page):
        pages.append(page)
        if page.directory == "first":
            update(2, [["subtree", "secret"]])
        elif page.directory == "secret":
            update(3, [])
        return True

    scanner, result, _ = scan(
        root, frontier=["first", "secret", "last", "secret"], admit=admit, emit=emit
    )
    assert result.status == "partial"
    assert [c.text for p in pages for c in p.captures] == ["first", "last"]
    assert [p.authority_revision for p in pages] == [1, 2, 3, 3]
    assert scanner.retired


def test_stop_during_pipe_io_retires_process_and_handles(tmp_path):
    import time
    import psutil

    root = tmp_path / "root"
    root.mkdir()
    for i in range(20):
        (root / f"{i}.txt").write_bytes(b"x" * 1024**2)
    auth = authority(root)
    scanner = scanner_class()()
    stop = threading.Event()

    def emit(page):
        stop.set()
        return True

    started = time.monotonic()
    result = scanner.scan(
        auth,
        ProjectIndexScope(auth.key, 1, 1),
        "cancel",
        iter([""]),
        emit,
        stop,
        admit_unit=lambda _: auth,
        policy=ProjectIndexPolicy(("",), (".txt",)),
    )
    assert result.status == "interrupted"
    assert scanner.retired and scanner.pipes_closed
    assert not psutil.pid_exists(scanner.worker_pid)
    assert time.monotonic() - started < 10


@pytest.mark.skipif(
    __import__("os").name == "nt",
    reason="POSIX permission-bit fixture; Windows ACL-denial scenario not qualified",
)
def test_unreadable_directory_has_incomplete_receipt(tmp_path):
    import os

    if os.geteuid() == 0:
        pytest.skip("root bypasses POSIX mode denial")
    denied = tmp_path / "denied"
    denied.mkdir()
    denied.chmod(0)
    try:
        _, result, pages = scan(tmp_path, frontier=["denied"])
        assert result.status == "partial"
        assert pages and not pages[0].directory_complete
    finally:
        denied.chmod(0o700)


def test_deep_directory_capture_has_no_python_recursion(tmp_path):
    current = tmp_path
    depth = 1050
    try:
        for _ in range(depth):
            current = current / "d"
            current.mkdir()
        (current / "leaf.txt").write_text("deep")
        relative = "/".join(["d"] * depth)
        scanner, result, pages = scan(tmp_path, frontier=[relative])
        assert result.status == "complete"
        assert [c.text for p in pages for c in p.captures] == ["deep"]
        assert scanner.retired
    finally:
        # Iterative fixture cleanup must not be the source of RecursionError.
        if (current / "leaf.txt").exists():
            (current / "leaf.txt").unlink()
        while current != tmp_path:
            parent = current.parent
            if current.exists():
                current.rmdir()
            current = parent


def test_megabyte_captures_apply_backpressure_before_next_read(tmp_path):
    for i in range(7):
        (tmp_path / f"{i}.txt").write_bytes(b"x" * 1024**2)
    scanner, result, pages = scan(tmp_path)
    assert result.status == "complete"
    assert sum(len(p.captures) for p in pages) == 7
    assert all(sum(c.size_bytes for c in p.captures) <= 3 * 1024**2 for p in pages)
    assert scanner.max_queued_text_bytes <= 4 * 1024**2
    assert scanner.max_active_reads == 1


def test_missing_frontier_directory_emits_incomplete_io_receipt(tmp_path):
    scanner, result, pages = scan(tmp_path, frontier=["vanished"])
    assert result.status == "partial"
    assert result.directories_completed == 0
    assert "directory_io" in result.issues
    assert len(pages) == 1 and not pages[0].directory_complete
    assert scanner.retired


@private_profile_test
def test_registry_witness_changes_between_directory_acknowledgements(tmp_path, request):
    import json
    import subprocess
    import sys
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService
    from tldw_chatbook.Workspaces.models import (
        WorkspaceRuntimeBinding,
        RuntimeBindingKind,
        RuntimeBindingStatus,
    )

    root = tmp_path / "project"
    root.mkdir()
    for name in ("first", "secret", "last"):
        (root / name).mkdir()
        (root / name / "body.txt").write_text(name)
    db = WorkspaceDB(tmp_path / "registry.sqlite")
    try:
        registry = LocalWorkspaceRegistryService(db)
        registry.create_workspace(workspace_id="workspace", name="Test")
        registry.save_runtime_binding(
            WorkspaceRuntimeBinding(
                workspace_id="workspace",
                binding_id="binding",
                binding_kind=RuntimeBindingKind.LOCAL_FILESYSTEM,
                label="Project",
                locator=str(root),
                status=RuntimeBindingStatus.READY,
            )
        )
        key = authority(root).key
        reader, _ = registry.project_index_authority_adapters(lambda: "test")
        initial = reader(key)
        assert initial is not None
        pages = []

        def mutate(excluded):
            metadata = {
                "exclusions": [
                    {
                        "path": "secret",
                        "kind": "directory",
                        "added_at": "2026-10-21T00:00:00Z",
                    }
                ]
                if excluded
                else []
            }
            code = 'import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute("UPDATE workspace_runtime_bindings SET metadata_json=? WHERE binding_id=?", (sys.argv[2], "binding")); c.commit(); c.close()'
            subprocess.run(
                [
                    sys.executable,
                    "-I",
                    "-c",
                    code,
                    str(db.db_path),
                    json.dumps(metadata),
                ],
                check=True,
            )

        def emit(page):
            pages.append(page)
            if page.directory == "first":
                mutate(True)
            elif page.directory == "secret":
                mutate(False)
            return True

        scanner = scanner_class()()
        result = scanner.scan(
            initial,
            ProjectIndexScope(key, 1, 1),
            "registry-job",
            iter(["first", "secret", "last", "secret"]),
            emit,
            threading.Event(),
            admit_unit=lambda _: reader(key),
            policy=ProjectIndexPolicy(("",), (".txt",)),
        )
        assert result.status == "partial"
        assert [c.text for p in pages for c in p.captures] == ["first", "last"]
        assert pages[1].authority_revision > pages[0].authority_revision
        assert pages[2].authority_revision > pages[1].authority_revision
        assert scanner.retired

    finally:
        db.close()


@pytest.mark.skipif(
    __import__("os").name != "posix",
    reason="Windows retained no-delete-share ancestor handles prevent this rename",
)
def test_capture_rejects_ancestor_replacement_during_read(tmp_path, monkeypatch):
    import os
    from tldw_chatbook.Tools import project_index_scan_worker as worker

    directory = tmp_path / "sub"
    directory.mkdir()
    (directory / "a.txt").write_text("body")
    original = os.read
    changed = False

    def racing_read(fd, count):
        nonlocal changed
        value = original(fd, count)
        if not changed:
            changed = True
            directory.rename(tmp_path / "moved")
            directory.mkdir()
        return value

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(worker.os, "read", racing_read)
    with pytest.raises(worker.UnsafeCapture):
        worker.capture_candidate("sub/a.txt", ())


def test_containment_pid_is_actual_interpreter_and_handles_retire(tmp_path):
    import gc
    import os
    import psutil
    import sys
    from pathlib import Path

    scanner = scanner_class()()
    auth = authority(tmp_path)
    descendants = []
    with scanner._worker(
        auth, ProjectIndexPolicy(("",), (".txt",)), threading.Event()
    ) as process:
        native = psutil.Process(process.pid)
        descendants = native.children(recursive=True)
        if os.name == "nt":
            assert Path(native.exe()) == Path(sys._base_executable)
        assert not descendants
        assert str(tmp_path) not in " ".join(native.cmdline())
    assert scanner.retired and scanner.pipes_closed
    assert not psutil.pid_exists(scanner.worker_pid)
    assert all(not psutil.pid_exists(child.pid) for child in descendants)
    owner = psutil.Process()
    gc.collect()
    before = owner.num_handles() if os.name == "nt" else owner.num_fds()
    for _ in range(3):
        scanner.capture_file(auth, "missing.txt", stop=threading.Event())
    gc.collect()
    after = owner.num_handles() if os.name == "nt" else owner.num_fds()
    assert after <= before + 1


def test_live_capture_applies_fixed_mandatory_names_without_scan_policy(tmp_path):
    (tmp_path / "credentials").write_text("never disclose")
    (tmp_path / "ordinary").write_text("allowed")
    scanner = scanner_class()()
    auth = authority(tmp_path)
    assert scanner.capture_file(auth, "credentials", stop=threading.Event()) is None
    capture = scanner.capture_file(auth, "ordinary", stop=threading.Event())
    assert capture is not None and capture.text == "allowed"
    assert scanner.retired


@pytest.mark.skipif(__import__("os").name != "nt", reason="Windows native ACL fixture")
def test_windows_unreadable_directory_emits_incomplete_receipt(tmp_path):
    import getpass
    import subprocess

    root = tmp_path / "project"
    root.mkdir()
    denied = root / "denied"
    denied.mkdir()
    (denied / "body.txt").write_text("cannot list")
    user = getpass.getuser()
    changed = subprocess.run(
        ["icacls", str(denied), "/deny", f"{user}:(RD)"], capture_output=True
    )
    if changed.returncode:
        pytest.skip("OS refused private ACL-denial fixture")
    try:
        scanner, result, pages = scan(root, frontier=["denied"])
        assert result.status == "partial"
        assert result.directories_completed == 0
        assert result.issues == ("directory_io",)
        assert len(pages) == 1 and not pages[0].directory_complete
        assert not pages[0].entries and scanner.retired
    finally:
        subprocess.run(
            ["icacls", str(denied), "/remove:d", user], check=True, capture_output=True
        )


@pytest.mark.parametrize("character", ["x", "\U0001f642"])
@pytest.mark.parametrize("first_size", [1024**2, 1024**2 - 65536])
def test_live_source_buffers_include_sender_slice_at_parent_join(
    tmp_path, monkeypatch, character, first_size
):
    """Catch missing sender headroom with actual buffers held across real pipes."""
    import inspect
    import json
    from pathlib import Path
    import subprocess
    import sys

    module = importlib.import_module("tldw_chatbook.Workspaces.project_index.scanner")
    scanner = module.ProjectScanner()
    corpus = tmp_path / "overlap"
    corpus.mkdir()
    for name, size in [("a.txt", first_size), ("b.txt", 1024**2)]:
        (corpus / name).write_bytes(
            character.encode("utf-8") * (size // len(character.encode("utf-8")))
        )

    # Only scheduling/instrumentation is substituted: real capture, encoding,
    # pipe parsing, page admission, process containment and retirement still run.
    child = """
import json, sys
from tldw_chatbook.Tools import project_index_scan_worker as worker
original_write = worker.write_frame
def hold_final(stream, value):
    result = original_write(stream, value)
    if value['type'] == 'File' and value['final']:
        sender = sys._getframe(1).f_locals
        buffers = {id(v): v for v in (sender['capture'].text, sender['part'])}
        print(json.dumps({'source': sum(len(v.encode('utf-8')) for v in buffers.values()), 'slice': len(sender['part'].encode('utf-8'))}), file=sys.stderr, flush=True)
        if sys.stdin.buffer.read(1) != b'!':
            raise RuntimeError('overlap_release_missing')
    return result
worker.write_frame = hold_final
raise SystemExit(worker.run_project_scan_worker(sys.stdin.buffer, sys.stdout.buffer))
"""
    real_popen = subprocess.Popen
    processes = []

    def launch(argv, **kwargs):
        argv = list(argv)
        argv[-1] = (
            "import sys;sys.path.insert(0,"
            + repr(str(Path(module.__file__).resolve().parents[3]))
            + ");exec("
            + repr(child)
            + ")"
        )
        kwargs["stderr"] = subprocess.PIPE
        process = real_popen(argv, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(module.subprocess, "Popen", launch)
    lines, start = inspect.getsourcelines(module.ProjectScanner._unit)
    joined_line = start + next(
        i for i, line in enumerate(lines) if "parts.clear()" in line
    )
    observations = []

    def trace(frame, event, arg):
        if (
            frame.f_code is module.ProjectScanner._unit.__code__
            and event == "line"
            and frame.f_lineno == joined_line
        ):
            local = frame.f_locals
            buffers = {
                id(v): v
                for v in [
                    *(c.text for c in local["captures"]),
                    *local["parts"],
                    local["text"],
                    local["value"]["text"],
                ]
            }
            worker = json.loads(processes[-1].stderr.readline())
            observations.append(
                {
                    "parent": sum(len(v.encode("utf-8")) for v in buffers.values()),
                    **worker,
                }
            )
            processes[-1].stdin.write(b"!")
            processes[-1].stdin.flush()
        return trace

    previous_trace = sys.gettrace()
    auth = authority(corpus)
    try:
        sys.settrace(trace)
        result = scanner.scan(
            auth,
            ProjectIndexScope(auth.key, 1, 1),
            "overlap",
            iter([""]),
            lambda _: True,
            threading.Event(),
            admit_unit=lambda _: auth,
            policy=ProjectIndexPolicy(("",), (".txt",)),
        )
    finally:
        sys.settrace(previous_trace)
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            process.stderr.close()
    assert result.status == "complete"
    assert scanner.retired and scanner.pipes_closed
    assert len(observations) == 2
    assert max(v["slice"] for v in observations) == (
        16384 if character == "x" else 65536
    )
    actual_maximum = max(v["parent"] + v["source"] for v in observations)
    print(
        {
            "first_size": first_size,
            "utf8_width": len(character.encode("utf-8")),
            "observations": observations,
            "actual_maximum": actual_maximum,
            "accounted": scanner.max_queued_text_bytes,
        }
    )
    assert actual_maximum <= 4194304, observations
    assert actual_maximum <= scanner.max_queued_text_bytes <= 4194304


@pytest.mark.parametrize("shape,count", [("nested", 12), ("empty", 300), ("wide", 600)])
def test_multi_directory_units_have_aggregate_budgets_and_dynamic_frontier(
    tmp_path, shape, count
):
    from tldw_chatbook.Workspaces.project_index.models import ScanUnit

    tmp_path = tmp_path / "corpus"
    tmp_path.mkdir()
    if shape == "nested":
        for index in range(count):
            leaf = tmp_path / f"d{index:03}"
            leaf.mkdir()
            (leaf / "marker.txt").write_text("unitmarker")
    elif shape == "empty":
        for index in range(count):
            (tmp_path / f"d{index:03}").mkdir()
    else:
        for index in range(count):
            (tmp_path / f"{index:03}.txt").write_text("unitmarker")
    auth = authority(tmp_path)
    scanner = scanner_class()()
    pending = {""}
    units, admissions = [], []

    def frontier():
        return tuple(sorted(pending)[:256])

    def publish(unit):
        assert isinstance(unit, ScanUnit)
        assert len(unit.entries) <= 256
        assert len(unit.directories) <= 256
        assert unit.inspections <= 256
        for entry in unit.entries:
            if entry.kind == "directory":
                pending.add(entry.relative_path)
        for receipt in unit.directories:
            if not receipt.more:
                pending.discard(receipt.directory)
        units.append(unit)
        return True

    def admit(seed):
        admissions.append(seed)
        return auth

    result = scanner.scan(
        auth,
        ProjectIndexScope(auth.key, 1, 1),
        "unit-job",
        frontier,
        publish,
        threading.Event(),
        admit_unit=admit,
        policy=ProjectIndexPolicy(("",), (".txt",)),
    )
    assert result.status == "complete" and not pending
    assert len(admissions) == len(units)
    if shape == "nested":
        assert len(units) == 1
        assert len(units[0].directories) == count + 1
    assert sum(len(unit.captures) for unit in units) == (
        0 if shape == "empty" else count
    )
    assert scanner.max_queued_text_bytes <= 4 * 1024**2
    assert scanner.retired and scanner.pipes_closed


@pytest.mark.parametrize(
    "policy,expected",
    [
        (
            ProjectIndexPolicy((), ()),
            {"README", "config.toml", "src/code.py", "src/package.json"},
        ),
        (ProjectIndexPolicy(("src",), ()), {"src/code.py", "src/package.json"}),
        (ProjectIndexPolicy((), (".py",)), {"src/code.py"}),
    ],
)
def test_empty_processing_choices_keep_mandatory_exclusions(tmp_path, policy, expected):
    root = tmp_path / "project"
    root.mkdir()
    for name in (
        "README",
        "config.toml",
        "src/code.py",
        "src/package.json",
        "node_modules/hidden.py",
        ".git/hidden.py",
        "secret/hidden.py",
    ):
        target = root / name
        target.parent.mkdir(exist_ok=True)
        target.write_text("marker", encoding="utf-8")
    (root / "binary.py").write_bytes(b"a\x00b")
    (root / "invalid.py").write_bytes(b"\xff")
    (root / "large.py").write_bytes(b"a" * (1024**2 + 1))
    auth = replace(
        authority(root), exclusions=(SensitiveExclusion("subtree", "secret"),)
    )
    scanner, result, pages = scan(
        root, frontier=["", "src"], admit=lambda _: auth, policy=policy
    )
    assert result.status == "complete"
    assert {c.relative_path for page in pages for c in page.captures} == expected
    assert scanner.retired and scanner.pipes_closed
