"""Isolated wheel and sdist smoke tests for the standalone MCP server."""

from __future__ import annotations

from email.parser import Parser
import io
import json
import os
from pathlib import Path
from queue import Empty, Queue
import shutil
import subprocess
import sys
import tarfile
import threading
from typing import Any, BinaryIO
import zipfile

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
import pytest

from Tests.Packaging.test_installed_distribution import (
    BuiltDistributions,
    built_distributions as _built_distributions,
)


built_distributions = _built_distributions
pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILTIN_TOOL_NAMES = {
    "chat_with_llm",
    "chat_with_character",
    "search_rag",
    "search_conversations",
    "create_note",
    "search_notes",
    "list_characters",
    "get_conversation_history",
    "export_conversation",
    "ingest_media",
}
LOCAL_TOOL_NAMES = {
    "fs_list",
    "fs_read",
    "fs_write",
    "fs_edit",
    "fs_patch",
    "fs_glob",
    "fs_grep",
    "git_status",
    "git_diff",
    "git_log",
    "git_blame",
    "git_branches",
    "web_fetch",
    "web_search",
    "web_crawl",
}
RESOURCE_TEMPLATES = {
    "conversation://{conversation_id}",
    "note://{note_id}",
    "character://{character_id}",
    "media://{media_id}",
    "rag-chunk://{chunk_uuid}",
}
PROMPT_NAMES = {
    "summarize_conversation",
    "generate_document",
    "analyze_media",
    "search_and_synthesize",
    "character_writing",
}
SAFE_HOST_ENV_KEYS = (
    "COMSPEC",
    "LANG",
    "LC_ALL",
    "LC_CTYPE",
    "PATH",
    "PATHEXT",
    "SSL_CERT_DIR",
    "SSL_CERT_FILE",
    "SYSTEMROOT",
    "TZ",
    "WINDIR",
)
PROCESS_EOF = object()
PATH_PROBE = r"""
from pathlib import Path
import json
import os
import sys
import sysconfig


def is_under(path, root):
    return path == root or path.is_relative_to(root)


def sensitive_name(name):
    upper = name.upper()
    return "PROXY" in upper or any(
        marker in upper
        for marker in (
            "API_KEY", "APIKEY", "AUTH", "CREDENTIAL", "DATABASE_URL",
            "PASSWORD", "PRIVATE_KEY", "SECRET", "TOKEN",
        )
    )


assert not any(sensitive_name(name) for name in os.environ)
assert "PYTHONPATH" not in os.environ

root = Path(os.environ["ARTIFACT_TEST_ROOT"]).resolve(strict=True)
checkout = Path(os.environ["CHECKOUT_ROOT"]).resolve(strict=True)
build_root = Path(os.environ["BUILD_SOURCE_ROOT"]).resolve(strict=True)

import mcp_unified
import tldw_chatbook

purelib = Path(sysconfig.get_paths()["purelib"]).resolve(strict=True)
import_paths = {
    "tldw_chatbook": Path(tldw_chatbook.__file__).resolve(strict=True),
    "mcp_unified": Path(mcp_unified.__file__).resolve(strict=True),
}
for path in import_paths.values():
    assert is_under(path, purelib), (path, purelib)
    assert not is_under(path, checkout), (path, checkout)
    assert not is_under(path, build_root), (path, build_root)

from tldw_chatbook.config import (
    get_chachanotes_db_path,
    get_cli_config_path,
    get_media_db_path,
    get_user_data_dir,
)
from tldw_chatbook.MCP.local_server_tools import resolve_server_workspace_root

data_dir = get_user_data_dir()
resolved_paths = {
    "config": get_cli_config_path(),
    "data": data_dir,
    "chachanotes_db": get_chachanotes_db_path(),
    "media_db": get_media_db_path(),
    "permission_store": data_dir / "mcp_permissions.json",
    "workspace": resolve_server_workspace_root(),
    "cwd": Path.cwd(),
    "home": Path(os.environ["HOME"]),
    "xdg_config": Path(os.environ["XDG_CONFIG_HOME"]),
    "xdg_data": Path(os.environ["XDG_DATA_HOME"]),
    "tmp": Path(os.environ["TMPDIR"]),
    "venv": Path(sys.prefix),
}
for name, path in resolved_paths.items():
    assert is_under(path.resolve(), root), (name, path, root)

print(json.dumps({
    "imports": {name: str(path) for name, path in import_paths.items()},
    "purelib": str(purelib),
    "paths": {name: str(path) for name, path in resolved_paths.items()},
}, sort_keys=True))
"""
SEED_DATA = r"""
from tldw_chatbook.config import CLI_APP_CLIENT_ID, get_chachanotes_db_path
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

database = CharactersRAGDB(get_chachanotes_db_path(), client_id=CLI_APP_CLIENT_ID)
conversation_id = database.add_conversation({"id": "7", "title": "Artifact conversation"})
assert conversation_id == "7"
assert database.add_message({
    "conversation_id": conversation_id,
    "sender": "user",
    "role": "user",
    "content": "é" * 300_000,
})
assert database.add_note(
    title="Artifact note",
    content="Deterministic packaged note",
    note_id="artifact-note",
) == "artifact-note"
assert database.add_character_card({
    "name": "Artifact Ada",
    "description": "Deterministic packaged character",
})
database.close()
"""


class _ProcessLineReader:
    """Bound reads from a pipe without assuming it is selectable."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._items: Queue[object] = Queue()
        self._eof_seen = False
        self._thread = threading.Thread(
            target=self._read_lines,
            name="packaged-mcp-stdout",
            daemon=True,
        )
        self._thread.start()

    def _read_lines(self) -> None:
        try:
            while line := self._stream.readline():
                self._items.put(line)
        except BaseException as error:
            self._items.put(error)
        finally:
            self._items.put(PROCESS_EOF)

    def _next(self, timeout: float) -> object:
        try:
            return self._items.get(timeout=timeout)
        except Empty:
            raise TimeoutError("packaged MCP server produced no response") from None

    def readline(self, *, timeout: float = 30) -> str:
        item = self._next(timeout)
        if item is PROCESS_EOF:
            self._eof_seen = True
            return ""
        if isinstance(item, BaseException):
            raise item
        assert isinstance(item, str)
        return item

    def finish(self, *, timeout: float = 5) -> list[str]:
        lines: list[str] = []
        while not self._eof_seen:
            item = self._next(timeout)
            if item is PROCESS_EOF:
                self._eof_seen = True
            elif isinstance(item, BaseException):
                raise item
            else:
                assert isinstance(item, str)
                lines.append(item)
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "packaged MCP stdout reader leaked"
        return lines

    def close(self, *, timeout: float = 5) -> None:
        self._stream.close()
        self._thread.join(timeout)
        assert not self._thread.is_alive(), "packaged MCP stdout reader leaked"


def _artifact_path(built: BuiltDistributions, artifact_kind: str) -> Path:
    return built.wheel if artifact_kind == "wheel" else built.sdist


def _metadata_and_license(artifact: Path, artifact_kind: str) -> tuple[Any, bool]:
    if artifact_kind == "wheel":
        with zipfile.ZipFile(artifact) as archive:
            names = archive.namelist()
            metadata_name = next(
                name for name in names if name.endswith(".dist-info/METADATA")
            )
            metadata = Parser().parsestr(archive.read(metadata_name).decode("utf-8"))
            dist_info = metadata_name.removesuffix("METADATA")
            has_license = f"{dist_info}licenses/LICENSE" in names
        return metadata, has_license

    with tarfile.open(artifact, "r:gz") as archive:
        members = archive.getmembers()
        pkg_info = next(
            member
            for member in members
            if member.isfile() and member.name.endswith("/PKG-INFO")
        )
        stream: BinaryIO | None = archive.extractfile(pkg_info)
        assert stream is not None
        metadata = Parser().parsestr(stream.read().decode("utf-8"))
        sdist_root = pkg_info.name.rsplit("/", maxsplit=1)[0]
        has_license = any(
            member.isfile() and member.name == f"{sdist_root}/LICENSE"
            for member in members
        )
    return metadata, has_license


def _assert_metadata_contract(metadata: Any, *, has_license: bool) -> None:
    assert metadata["License-Expression"] == "AGPL-3.0-or-later"
    assert "LICENSE" in (metadata.get_all("License-File") or [])
    assert has_license
    requirements = [
        Requirement(value) for value in metadata.get_all("Requires-Dist") or []
    ]
    mcp_unified = [
        requirement
        for requirement in requirements
        if canonicalize_name(requirement.name) == "mcp-unified"
    ]
    assert len(mcp_unified) == 2
    assert all(requirement.url is None for requirement in mcp_unified)
    assert all(not requirement.extras for requirement in mcp_unified)
    assert {
        (str(requirement.specifier), str(requirement.marker))
        for requirement in mcp_unified
    } == {
        ("==0.2.1", 'extra == "mcp"'),
        ("==0.2.1", 'extra == "all-tools"'),
    }


def _assert_tool_inventory(tool_names: set[str]) -> None:
    assert tool_names == BUILTIN_TOOL_NAMES | LOCAL_TOOL_NAMES
    assert not any(name.startswith("library_") for name in tool_names)


def _assert_private_stderr(stderr: str, *, secret: str, path: str) -> None:
    assert secret not in stderr
    assert path not in stderr
    assert '"jsonrpc"' not in stderr
    assert "Traceback" not in stderr


def _test_metadata(*requirements: str) -> Any:
    headers = [
        "License-Expression: AGPL-3.0-or-later",
        "License-File: LICENSE",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
    ]
    return Parser().parsestr("\n".join(headers) + "\n\n")


def test_sdist_license_must_be_at_the_project_root(tmp_path: Path) -> None:
    artifact = tmp_path / "nested-license.tar.gz"
    with tarfile.open(artifact, "w:gz") as archive:
        for name, content in (
            ("project-1.0/PKG-INFO", b"Metadata-Version: 2.4\n"),
            ("project-1.0/vendor/LICENSE", b"vendored license\n"),
        ):
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))

    _, has_license = _metadata_and_license(artifact, "sdist")
    assert not has_license


def test_wheel_license_must_match_the_metadata_dist_info_directory(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "mismatched-license.whl"
    with zipfile.ZipFile(artifact, "w") as archive:
        archive.writestr("project-1.0.dist-info/METADATA", "Metadata-Version: 2.4\n")
        archive.writestr("other-1.0.dist-info/licenses/LICENSE", "wrong project\n")

    _, has_license = _metadata_and_license(artifact, "wheel")
    assert not has_license


@pytest.mark.parametrize(
    "unexpected_requirement",
    [
        "mcp-unified==0.2.1",
        'mcp-unified==0.2.1; extra == "dev"',
        'mcp_unified>=0.2.1; extra == "mcp"',
        'mcp-unified==0.2.1; extra == "mcp"',
        'mcp-unified==0.2.1; extra == "all-tools"',
    ],
)
def test_metadata_contract_rejects_every_additional_mcp_unified_requirement(
    unexpected_requirement: str,
) -> None:
    expected = (
        'mcp-unified==0.2.1; extra == "mcp"',
        'mcp-unified==0.2.1; extra == "all-tools"',
    )
    _assert_metadata_contract(_test_metadata(*expected), has_license=True)
    metadata = _test_metadata(*expected, unexpected_requirement)
    with pytest.raises(AssertionError):
        _assert_metadata_contract(metadata, has_license=True)


def test_metadata_contract_accepts_only_the_exact_normalized_requirement() -> None:
    metadata = _test_metadata(
        "mcp_unified == 0.2.1 ; extra == 'all-tools'",
        "mcp-unified == 0.2.1 ; extra == 'mcp'",
    )
    _assert_metadata_contract(metadata, has_license=True)


def _python_in(venv_root: Path) -> Path:
    directory = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_root / directory / executable


def _write_config(root: Path) -> Path:
    config = root / "xdg-config" / "tldw_cli" / "config.toml"
    data = root / "xdg-data"
    databases = root / "databases"
    workspace = root / "workspace"
    for directory in (config.parent, data, databases, workspace, root / "tmp"):
        directory.mkdir(parents=True, mode=0o700)
    (workspace / "private.txt").write_text(
        "packaged workspace sentinel\n", encoding="utf-8"
    )
    config.write_text(
        "\n".join(
            (
                "[general]",
                'users_name = "artifact_user"',
                "",
                "[paths]",
                f"data_dir = {json.dumps(str(data))}",
                "",
                "[database]",
                f"chachanotes_db_path = {json.dumps(str(databases / 'chachanotes.sqlite'))}",
                f"media_db_path = {json.dumps(str(databases / 'media.sqlite'))}",
                "",
                "[console]",
                f"workspace_root = {json.dumps(str(workspace))}",
                "",
                "[mcp]",
                "expose_local_tools = true",
                "",
                "[model_catalog]",
                "auto_refresh_enabled = false",
            )
        ),
        encoding="utf-8",
    )
    config.chmod(0o600)
    return config


def _isolated_env(root: Path, config: Path, build_root: Path) -> dict[str, str]:
    env = {key: os.environ[key] for key in SAFE_HOST_ENV_KEYS if key in os.environ}
    env.update(
        {
            "APPDATA": str(root / "xdg-config"),
            "ARTIFACT_TEST_ROOT": str(root),
            "BUILD_SOURCE_ROOT": str(build_root),
            "CHECKOUT_ROOT": str(REPO_ROOT),
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_OFFLINE": "1",
            "HOME": str(root / "home"),
            "LOCALAPPDATA": str(root / "xdg-data"),
            "LOGURU_LEVEL": "ERROR",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHON_KEYRING_BACKEND": "keyring.backends.null.Keyring",
            "PYTHONNOUSERSITE": "1",
            "PYTHONUTF8": "1",
            "TLDW_CONFIG_PATH": str(config),
            "TLDW_TEST_MODE": "1",
            "TEMP": str(root / "tmp"),
            "TMP": str(root / "tmp"),
            "TMPDIR": str(root / "tmp"),
            "TRANSFORMERS_OFFLINE": "1",
            "USERPROFILE": str(root / "home"),
            "UV_CACHE_DIR": str(root / "uv-cache"),
            "UV_NO_PROGRESS": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "XDG_CONFIG_HOME": str(root / "xdg-config"),
            "XDG_DATA_HOME": str(root / "xdg-data"),
        }
    )
    return env


def _run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=timeout,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command: {command}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    )
    return completed


def _request(
    process: subprocess.Popen[str], reader: _ProcessLineReader, payload: dict[str, Any]
) -> dict[str, Any]:
    assert process.stdin is not None
    process.stdin.write(json.dumps(payload, separators=(",", ":")) + "\n")
    process.stdin.flush()
    line = reader.readline()
    assert line
    response = json.loads(line)
    assert isinstance(response, dict) and response.get("jsonrpc") == "2.0"
    return response


def _legacy_request(
    request_id: str, method: str, params: dict[str, Any] | None = None
) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": params or {},
    }


def _exercise_server(python: Path, run_root: Path, env: dict[str, str]) -> None:
    secret = "TASK2512_ARTIFACT_SECRET_SENTINEL"
    checkout_sentinel = str(REPO_ROOT / secret / "private-source-sentinel")
    process = subprocess.Popen(
        [str(python), "-I", "-m", "tldw_chatbook.MCP"],
        cwd=run_root,
        env=env,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert process.stdout is not None
    reader = _ProcessLineReader(process.stdout)
    responses: list[dict[str, Any]] = []
    try:
        initialized = _request(
            process,
            reader,
            _legacy_request(
                "initialize",
                "initialize",
                {
                    "protocolVersion": "2025-03-26",
                    "capabilities": {},
                    "clientInfo": {"name": "artifact-test", "version": "1.0"},
                },
            ),
        )
        assert initialized["result"]["protocolVersion"] == "2025-03-26"
        assert process.stdin is not None
        process.stdin.write(
            '{"jsonrpc":"2.0","method":"notifications/initialized","params":{}}\n'
        )
        process.stdin.flush()

        tools = _request(process, reader, _legacy_request("tools", "tools/list"))
        tool_names = {item["name"] for item in tools["result"]["tools"]}
        _assert_tool_inventory(tool_names)
        responses.append(tools)

        templates = _request(
            process,
            reader,
            _legacy_request("templates", "resources/templates/list"),
        )
        assert {
            item["uriTemplate"] for item in templates["result"]["resourceTemplates"]
        } == RESOURCE_TEMPLATES
        responses.append(templates)

        resources = _request(
            process, reader, _legacy_request("resources", "resources/list")
        )
        assert {item["uri"] for item in resources["result"]["resources"]} == {
            "conversation://7",
            "note://artifact-note",
        }
        responses.append(resources)

        prompts = _request(process, reader, _legacy_request("prompts", "prompts/list"))
        assert {item["name"] for item in prompts["result"]["prompts"]} == PROMPT_NAMES
        responses.append(prompts)

        characters = _request(
            process,
            reader,
            _legacy_request(
                "characters",
                "tools/call",
                {"name": "list_characters", "arguments": {}},
            ),
        )
        character_rows = json.loads(characters["result"]["content"][0]["text"])
        assert {row["name"] for row in character_rows} == {
            "Artifact Ada",
            "Default Assistant",
        }
        responses.append(characters)

        refusal = _request(
            process,
            reader,
            _legacy_request(
                "refusal",
                "tools/call",
                {"name": "fs_read", "arguments": {"path": checkout_sentinel}},
            ),
        )
        assert refusal["result"]["isError"] is True
        assert refusal["result"]["content"] == [
            {
                "type": "text",
                "text": "Operator approval is required for this local tool.",
            }
        ]
        assert (
            refusal["result"]["_meta"]["io.github.rmusser01.mcp-unified/error"][
                "reasonCode"
            ]
            == "operator_approval_required"
        )
        responses.append(refusal)

        resource_uri: str | None = "conversation://7"
        resource_text: list[str] = []
        reads = 0
        while resource_uri is not None and reads < 10:
            read = _request(
                process,
                reader,
                _legacy_request(
                    f"read-{reads}", "resources/read", {"uri": resource_uri}
                ),
            )
            result = read["result"]
            assert set(result["_meta"]) == {
                "tldw.chatbook/continuation",
                "tldw.chatbook/resource",
            }
            continuation = result["_meta"]["tldw.chatbook/continuation"]
            assert continuation["returnedBytes"] <= 256 * 1024
            resource_text.append(result["contents"][0]["text"])
            resource_uri = continuation["nextUri"]
            reads += 1
            responses.append(read)
        assert resource_uri is None and reads > 1
        assert "é" * 300_000 in "".join(resource_text)

        prompt = _request(
            process,
            reader,
            _legacy_request(
                "prompt",
                "prompts/get",
                {
                    "name": "summarize_conversation",
                    "arguments": {"conversation_id": "7"},
                },
            ),
        )
        assert prompt["result"]["messages"][0]["role"] == "user"
        assert (
            "Artifact conversation"
            in prompt["result"]["messages"][0]["content"]["text"]
        )
        responses.append(prompt)

        assert process.stdin is not None
        process.stdin.close()
        process.wait(timeout=30)
        trailing = reader.finish(timeout=5)
        assert process.stderr is not None
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        reader.close(timeout=5)

    assert process.returncode == 0
    assert trailing == []
    assert responses
    _assert_private_stderr(stderr, secret=secret, path=checkout_sentinel)


@pytest.mark.parametrize("artifact_kind", ["wheel", "sdist"])
def test_mcp_extra_installs_and_runs_from_each_isolated_artifact(
    artifact_kind: str,
    built_distributions: BuiltDistributions,
    tmp_path: Path,
) -> None:
    artifact = _artifact_path(built_distributions, artifact_kind)
    metadata, has_license = _metadata_and_license(artifact, artifact_kind)
    _assert_metadata_contract(metadata, has_license=has_license)

    root = tmp_path / f"{artifact_kind}-root"
    root.mkdir(mode=0o700)
    try:
        config = _write_config(root)
        (root / "home").mkdir(mode=0o700)
        run_root = root / "consumer"
        run_root.mkdir(mode=0o700)
        venv_root = root / "venv"
        env = _isolated_env(root, config, built_distributions.source_root)
        _run(
            [
                "uv",
                "venv",
                "--no-project",
                "--python",
                sys._base_executable,
                str(venv_root),
            ],
            cwd=run_root,
            env=env,
            timeout=60,
        )
        python = _python_in(venv_root)

        _run(
            [
                "uv",
                "pip",
                "install",
                "--link-mode",
                "copy",
                "--python",
                str(python),
                f"{artifact}[mcp]",
            ],
            cwd=run_root,
            env=env,
            timeout=600,
        )
        probe = _run(
            [str(python), "-I", "-c", PATH_PROBE],
            cwd=run_root,
            env=env,
            timeout=60,
        )
        assert set(json.loads(probe.stdout)) == {"imports", "paths", "purelib"}
        _run(
            [str(python), "-I", "-c", SEED_DATA],
            cwd=run_root,
            env=env,
            timeout=120,
        )
        _exercise_server(python, run_root, env)
    finally:
        shutil.rmtree(root, ignore_errors=True)

    assert not root.exists()
