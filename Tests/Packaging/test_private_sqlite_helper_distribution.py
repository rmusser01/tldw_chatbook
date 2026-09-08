"""Installed-wheel qualification for the fixed private SQLite helper."""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import NamedTuple

import pytest

from Tests.Packaging.test_installed_distribution import (
    _copy_build_inputs,
    _sanitized_build_env,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
MAX_BODY_BYTES = 65_536
PRIVATE_HELPER_CLOSURE = frozenset(
    {
        "tldw_chatbook/DB/private_sqlite_files.py",
        "tldw_chatbook/DB/private_sqlite_helper.py",
        "tldw_chatbook/DB/private_sqlite_helper_entry.py",
        "tldw_chatbook/DB/private_sqlite_protocol.py",
        "tldw_chatbook/DB/sql_identifier_core.py",
        "tldw_chatbook/TTS/migrations/v0_to_v1.py",
        "tldw_chatbook/TTS/migrations/v2_to_v3.py",
        "tldw_chatbook/TTS/migrations/v3_to_v4.py",
        "tldw_chatbook/TTS/profile_errors.py",
        "tldw_chatbook/TTS/profile_migration_journal.py",
        "tldw_chatbook/TTS/profile_reference_types.py",
        "tldw_chatbook/TTS/profile_sqlite_proof.py",
        "tldw_chatbook/TTS/profile_types.py",
        "tldw_chatbook/TTS/profile_validation.py",
        "tldw_chatbook/Utils/private_paths.py",
    }
)
APPROVED_STDLIB_TOP_LEVELS = frozenset({*sys.stdlib_module_names, "__main__"})
APPROVED_PRODUCT_MODULES = frozenset(
    {
        "tldw_chatbook",
        "tldw_chatbook.DB",
        "tldw_chatbook.TTS",
        "tldw_chatbook.TTS.migrations",
        "tldw_chatbook.Utils",
        *(
            relative.removesuffix(".py").replace("/", ".")
            for relative in PRIVATE_HELPER_CLOSURE
        ),
    }
)
FORBIDDEN_IMPORT_CONTROLS = (
    "keyring",
    "loguru",
    "textual",
    "openai",
    "anthropic",
    "tldw_chatbook.app",
    "tldw_chatbook.config",
    "tldw_chatbook.LLM_Calls.fake_provider",
    "tldw_chatbook.LLM_Provider_Catalog.fake_provider",
    "tldw_chatbook.Chat.console_provider_gateway",
    "tldw_chatbook.Agents.mcp_tool_provider",
    "tldw_chatbook.TTS.provider_ids",
)


class InstalledHelper(NamedTuple):
    """One wheel and its no-dependency installation under an owned target."""

    wheel: Path
    target: Path
    entry: Path


def _frame(payload: dict[str, object]) -> bytes:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    assert len(body) <= MAX_BODY_BYTES
    return struct.pack(">I", len(body)) + body


def _decode_frames(data: bytes) -> list[dict[str, object]]:
    replies: list[dict[str, object]] = []
    offset = 0
    while offset < len(data):
        assert len(data) - offset >= 4
        size = struct.unpack(">I", data[offset : offset + 4])[0]
        assert 0 < size <= MAX_BODY_BYTES
        offset += 4
        body = data[offset : offset + size]
        assert len(body) == size
        decoded = json.loads(body)
        assert type(decoded) is dict
        replies.append(decoded)
        offset += size
    return replies


def _installed_env(
    state_root: Path, pythonpath: Path, user_state: Path
) -> dict[str, str]:
    env = _sanitized_build_env(state_root)
    env.update(
        {
            "PYTHONPATH": str(pythonpath),
            "XDG_CONFIG_HOME": str(user_state / "xdg-config"),
            "XDG_DATA_HOME": str(user_state / "xdg-data"),
            "_TLDW_PRIVATE_SQLITE_PARENT_PID": str(os.getpid()),
        }
    )
    return env


def _poison_import_root(root: Path, marker: Path) -> None:
    module = root / "tldw_chatbook" / "TTS" / "profile_validation.py"
    module.parent.mkdir(parents=True)
    module.write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('imported', encoding='utf-8')\n"
        "raise RuntimeError('hostile import root used')\n",
        encoding="utf-8",
    )


def _hostile_launch_context(tmp_path: Path) -> tuple[Path, Path, Path, list[Path]]:
    hostile_cwd = tmp_path / "hostile-cwd"
    hostile_pythonpath = tmp_path / "hostile-pythonpath"
    user_state = tmp_path / "hostile-user-state"
    markers = [tmp_path / "cwd-imported", tmp_path / "pythonpath-imported"]
    _poison_import_root(hostile_cwd, markers[0])
    _poison_import_root(hostile_pythonpath, markers[1])
    for relative in (
        Path("config.toml"),
        Path(".config/tldw_cli/config.toml"),
        Path("xdg-config/tldw_cli/config.toml"),
        Path("xdg-data/tldw_cli/model_catalog_cache.json"),
    ):
        selected = user_state / relative
        selected.parent.mkdir(parents=True, exist_ok=True)
        selected.write_text("raise-if-consumed", encoding="utf-8")
        selected.chmod(0)
    return hostile_cwd, hostile_pythonpath, user_state, markers


def _run_helper(
    entry: Path,
    requests: list[dict[str, object]],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float = 5.0,
    verbose_imports: bool = False,
) -> subprocess.CompletedProcess[bytes]:
    command = [sys.executable, "-I", "-S"]
    if verbose_imports:
        command.append("-v")
    command.append(str(entry))
    return subprocess.run(
        command,
        input=b"".join(_frame(request) for request in requests),
        env=env,
        capture_output=True,
        cwd=cwd,
        timeout=timeout,
        check=False,
    )


def _prepare_request(path: Path) -> dict[str, object]:
    return {
        "version": 1,
        "operation": "prepare",
        "path": str(path),
        "writable": True,
        "create_if_missing": True,
        "preserve_source_mode": False,
    }


def _tts_request(path: Path) -> dict[str, object]:
    return {
        "version": 1,
        "operation": "tts_exact_current",
        "path": str(path),
        "writable": False,
        "create_if_missing": False,
        "preserve_source_mode": False,
    }


def _close_request() -> dict[str, object]:
    return {"version": 1, "operation": "close"}


@pytest.fixture(scope="module")
def installed_helper(tmp_path_factory: pytest.TempPathFactory) -> InstalledHelper:
    source_root = tmp_path_factory.mktemp("private-sqlite-distribution-source")
    _copy_build_inputs(source_root)
    dist_dir = source_root / "dist"
    command = [
        sys.executable,
        "-m",
        "build",
        "--wheel",
        "--no-isolation",
        "--outdir",
        str(dist_dir),
    ]
    completed = subprocess.run(
        command,
        cwd=source_root,
        env=_sanitized_build_env(source_root / "build-state"),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    wheels = sorted(dist_dir.glob("*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as archive:
        entries = [
            member
            for member in archive.namelist()
            if member == "tldw_chatbook/DB/private_sqlite_helper_entry.py"
        ]
    assert len(entries) == 1

    target = tmp_path_factory.mktemp("private-sqlite-installed-target")
    install_command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--disable-pip-version-check",
        "--target",
        str(target),
        str(wheels[0]),
    ]
    installed = subprocess.run(
        install_command,
        cwd=target.parent,
        env=_sanitized_build_env(source_root / "install-state"),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert installed.returncode == 0, installed.stdout + installed.stderr
    entry = target / entries[0]
    assert entry.is_file()
    return InstalledHelper(wheels[0], target, entry)


def test_private_sqlite_helper_leaf_closure_ships_byte_exact(
    installed_helper: InstalledHelper,
) -> None:
    """Deleting or omitting any fixed helper leaf would break installed use."""

    with zipfile.ZipFile(installed_helper.wheel) as archive:
        members = set(archive.namelist())
        assert PRIVATE_HELPER_CLOSURE <= members
        for relative in PRIVATE_HELPER_CLOSURE:
            assert archive.read(relative) == (REPO_ROOT / relative).read_bytes()
            assert (installed_helper.target / relative).read_bytes() == archive.read(
                relative
            )


def test_installed_entry_runs_close_prepare_and_fixed_tts_from_hostile_cwd(
    installed_helper: InstalledHelper,
    tmp_path: Path,
) -> None:
    """The absolute installed entry performs every fixed operation in isolation."""

    hostile_cwd, hostile_pythonpath, user_state, markers = _hostile_launch_context(
        tmp_path
    )
    env = _installed_env(tmp_path / "process-state", hostile_pythonpath, user_state)
    env["HOME"] = str(user_state)

    close = _run_helper(
        installed_helper.entry,
        [_close_request()],
        cwd=hostile_cwd,
        env=env,
    )
    assert close.returncode == 0
    assert _decode_frames(close.stdout) == [
        {"version": 1, "operation": "close", "status": "ok"}
    ]
    assert close.stderr == b""

    prepared_path = tmp_path / "prepared.sqlite3"
    prepared = _run_helper(
        installed_helper.entry,
        [_prepare_request(prepared_path), _close_request()],
        cwd=hostile_cwd,
        env=env,
    )
    assert prepared.returncode == 0
    prepared_replies = _decode_frames(prepared.stdout)
    assert [reply["operation"] for reply in prepared_replies] == ["prepare", "close"]
    assert prepared_replies[0]["status"] == "ok"
    assert prepared_replies[0]["result"]["artifacts"] == [
        "created_private",
        "absent",
        "absent",
        "absent",
    ]
    assert prepared_replies[1] == {
        "version": 1,
        "operation": "close",
        "status": "ok",
    }
    assert prepared.stderr == b""

    from tldw_chatbook.TTS.profile_schema import open_profile_store

    profile_path = tmp_path / "profiles.sqlite3"
    profile = open_profile_store(profile_path)
    profile.close()
    fixed_tts = _run_helper(
        installed_helper.entry,
        [_tts_request(profile_path), _close_request()],
        cwd=hostile_cwd,
        env=env,
        timeout=35.0,
    )
    assert fixed_tts.returncode == 0
    tts_replies = _decode_frames(fixed_tts.stdout)
    assert [reply["operation"] for reply in tts_replies] == [
        "tts_exact_current",
        "close",
    ]
    assert tts_replies[0]["status"] == "ok"
    assert set(tts_replies[0]["identity"]) == {"parent", "main", "wal", "shm"}
    assert tts_replies[1]["status"] == "ok"
    assert fixed_tts.stderr == b""
    assert not any(marker.exists() for marker in markers)


@pytest.mark.parametrize(
    ("extra_args", "parent_value"),
    [(["unexpected"], None), ([], None), ([], "1")],
)
def test_installed_entry_preserves_argv_and_original_parent_checks(
    installed_helper: InstalledHelper,
    tmp_path: Path,
    extra_args: list[str],
    parent_value: str | None,
) -> None:
    """Isolation cannot bypass the launch-only argv or parent identity checks."""

    cwd = tmp_path / "cwd"
    cwd.mkdir()
    env = _sanitized_build_env(tmp_path / "state")
    if parent_value is not None:
        env["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = parent_value
    completed = subprocess.run(
        [sys.executable, "-I", "-S", str(installed_helper.entry), *extra_args],
        input=_frame(_close_request()),
        cwd=cwd,
        env=env,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert completed.returncode == 1
    assert completed.stdout == completed.stderr == b""


def _instrument_installed_entry(
    entry: Path,
    trace_path: Path,
    *,
    controlled_module_name: str | None = None,
) -> None:
    source = entry.read_text(encoding="utf-8")
    insertion_point = "from types import ModuleType\n"
    controlled_module = (
        ""
        if controlled_module_name is None
        else (
            f"\nsys.modules[{controlled_module_name!r}] = "
            f"ModuleType({controlled_module_name!r})\n"
        )
    )
    instrumentation = f"""{insertion_point}
import atexit as _trace_atexit
import json as _trace_json

_trace_fd = os.open({str(trace_path)!r}, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
_trace_opened = []

def _trace_audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        _trace_opened.append(os.fsdecode(args[0]))

sys.addaudithook(_trace_audit)

def _write_trace():
    product_modules = {{
        name: getattr(module, "__file__", None)
        for name, module in sys.modules.items()
        if name == "tldw_chatbook" or name.startswith("tldw_chatbook.")
    }}
    payload = {{
        "module_names": sorted(sys.modules),
        "product_modules": product_modules,
        "opened": _trace_opened,
    }}
    os.write(_trace_fd, _trace_json.dumps(payload).encode("utf-8"))
    os.close(_trace_fd)

_trace_atexit.register(_write_trace)
{controlled_module}
"""
    assert source.count(insertion_point) == 1
    entry.write_text(
        source.replace(insertion_point, instrumentation),
        encoding="utf-8",
    )


def _unexpected_imports(module_names: set[str]) -> list[str]:
    return sorted(
        name
        for name in module_names
        if (name == "tldw_chatbook" or name.startswith("tldw_chatbook."))
        and name not in APPROVED_PRODUCT_MODULES
        or (
            name != "tldw_chatbook"
            and not name.startswith("tldw_chatbook.")
            and name.partition(".")[0] not in APPROVED_STDLIB_TOP_LEVELS
        )
    )


def test_owned_import_and_file_audit_stays_inside_installed_leaf_closure(
    installed_helper: InstalledHelper,
    tmp_path: Path,
) -> None:
    """A test-owned audit wrapper proves imports and user-file access stay isolated."""

    audited_target = tmp_path / "audited-install"
    shutil.copytree(installed_helper.target, audited_target)
    entry = audited_target / "tldw_chatbook" / "DB" / "private_sqlite_helper_entry.py"
    trace_path = tmp_path / "child-audit.json"
    _instrument_installed_entry(entry, trace_path)
    hostile_cwd, hostile_pythonpath, user_state, markers = _hostile_launch_context(
        tmp_path / "hostile"
    )
    env = _installed_env(tmp_path / "process-state", hostile_pythonpath, user_state)
    env["HOME"] = str(user_state)

    from tldw_chatbook.TTS.profile_schema import open_profile_store

    profile_path = tmp_path / "audited-profiles.sqlite3"
    profile = open_profile_store(profile_path)
    profile.close()
    completed = _run_helper(
        entry,
        [_tts_request(profile_path), _close_request()],
        cwd=hostile_cwd,
        env=env,
        timeout=35.0,
        verbose_imports=True,
    )
    assert completed.returncode == 0
    assert len(_decode_frames(completed.stdout)) == 2
    # The plain production launch above owns the clean-stderr assertion. This
    # supplemental run intentionally uses CPython's import trace on stderr.
    assert b"private_sqlite_helper" in completed.stderr
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    module_names = set(trace["module_names"])
    product_modules = trace["product_modules"]
    assert set(product_modules) <= module_names
    assert _unexpected_imports(module_names) == []
    file_modules = {
        name: Path(origin).resolve()
        for name, origin in product_modules.items()
        if origin is not None
    }
    assert file_modules
    assert all(
        origin.is_relative_to(audited_target.resolve())
        for origin in file_modules.values()
    )
    forbidden_roots = (
        REPO_ROOT.resolve(),
        hostile_cwd.resolve(),
        hostile_pythonpath.resolve(),
        user_state.resolve(),
    )
    # Descriptor-relative parent walks legitimately emit component-only audit
    # arguments. They cannot be resolved without the event's dir_fd, so only
    # absolute names can establish access to one of these absolute trap roots.
    opened = [
        Path(item).resolve() for item in trace["opened"] if Path(item).is_absolute()
    ]
    assert not any(
        selected.is_relative_to(root) for selected in opened for root in forbidden_roots
    )
    assert not any(marker.exists() for marker in markers)


@pytest.mark.parametrize("controlled_module_name", FORBIDDEN_IMPORT_CONTROLS)
def test_owned_import_audit_rejects_controlled_forbidden_names(
    installed_helper: InstalledHelper,
    tmp_path: Path,
    controlled_module_name: str,
) -> None:
    """Every claimed forbidden boundary must be visible to the audit gate."""

    audited_target = tmp_path / "audited-install"
    shutil.copytree(installed_helper.target, audited_target)
    entry = audited_target / "tldw_chatbook" / "DB" / "private_sqlite_helper_entry.py"
    trace_path = tmp_path / "child-audit.json"
    _instrument_installed_entry(
        entry,
        trace_path,
        controlled_module_name=controlled_module_name,
    )
    env = _sanitized_build_env(tmp_path / "state")
    env["_TLDW_PRIVATE_SQLITE_PARENT_PID"] = str(os.getpid())

    completed = _run_helper(
        entry,
        [_close_request()],
        cwd=tmp_path,
        env=env,
    )
    assert completed.returncode == 0
    assert completed.stderr == b""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    module_names = set(trace["module_names"])
    assert controlled_module_name in module_names
    assert _unexpected_imports(module_names) == [controlled_module_name]


def test_missing_installed_leaf_cannot_fall_back_to_hostile_pythonpath(
    installed_helper: InstalledHelper,
    tmp_path: Path,
) -> None:
    """The hostile roots cannot rescue a wheel with a required leaf removed."""

    broken_target = tmp_path / "broken-install"
    shutil.copytree(installed_helper.target, broken_target)
    entry = broken_target / "tldw_chatbook" / "DB" / "private_sqlite_helper_entry.py"
    (broken_target / "tldw_chatbook" / "TTS" / "profile_validation.py").unlink()
    hostile_cwd, hostile_pythonpath, user_state, markers = _hostile_launch_context(
        tmp_path / "hostile"
    )
    env = _installed_env(tmp_path / "process-state", hostile_pythonpath, user_state)
    env["HOME"] = str(user_state)

    from tldw_chatbook.TTS.profile_schema import open_profile_store

    profile_path = tmp_path / "profiles.sqlite3"
    profile = open_profile_store(profile_path)
    profile.close()
    completed = _run_helper(
        entry,
        [_tts_request(profile_path), _close_request()],
        cwd=hostile_cwd,
        env=env,
        timeout=35.0,
    )
    assert completed.returncode == 0
    assert _decode_frames(completed.stdout) == [
        {
            "version": 1,
            "operation": "tts_exact_current",
            "status": "helper_unavailable",
        }
    ]
    assert completed.stderr == b""
    assert not any(marker.exists() for marker in markers)
