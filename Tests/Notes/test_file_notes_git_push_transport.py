from __future__ import annotations

import asyncio
import base64
import getpass
import http.server
import json
import os
import selectors
import shlex
import shutil
import socket
import ssl
import subprocess
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path

import pytest
from loguru import logger

import tldw_chatbook.Notes.file_notes_git_network as git_network
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitArg,
    GitCommandResult,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitPublication,
    FileNotesSessionOwner,
    FileSystemIdentity,
    HeadIdentity,
    IndexBaseline,
    IndexEntry,
    PushIncludedNote,
    RepositoryIdentity,
    SessionBinding,
    SessionChange,
    StagingOwnership,
)


BRANCH_REF = "refs/heads/main"
PROVIDER_CANARY = "TRANSPORT_PROVIDER_SECRET_CANARY"
HELPER_OUTPUT_CANARY = "TRANSPORT_HELPER_OUTPUT_CANARY"
CREDENTIAL_CANARY = "TRANSPORT_CREDENTIAL_CANARY"
SERVER_OUTPUT_CANARY = "TRANSPORT_SERVER_OUTPUT_CANARY"
RAW_STDOUT_CANARY = "TRANSPORT_RAW_STDOUT_CANARY"
RAW_STDERR_CANARY = "TRANSPORT_RAW_STDERR_CANARY"


def _structured_state_contains(value: object, canary: str) -> bool:
    """Inspect bounded retained data structures without following live tasks."""
    seen: set[int] = set()

    def visit(item: object) -> bool:
        if isinstance(item, str):
            return canary in item
        if isinstance(item, bytes):
            return canary.encode() in item
        identity = id(item)
        if identity in seen:
            return False
        seen.add(identity)
        if isinstance(item, Mapping):
            return any(visit(key) or visit(entry) for key, entry in item.items())
        if isinstance(item, (tuple, list, set, frozenset)):
            return any(visit(entry) for entry in item)
        if is_dataclass(item) and not isinstance(item, type):
            return any(visit(getattr(item, field.name)) for field in fields(item))
        return False

    return visit(value)


def _git(
    repository: Path,
    *arguments: str,
    input_bytes: bytes | None = None,
) -> bytes:
    environment = _fixture_git_environment(repository.parent)
    operation = _bounded_git_operation(arguments)
    try:
        result = subprocess.run(
            (
                _git_executable(),
                "-c",
                f"core.hooksPath={os.devnull}",
                "-c",
                "commit.gpgSign=false",
                "-C",
                str(repository),
                *arguments,
            ),
            input=input_bytes,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(f"fixture Git {operation} timed out") from None
    except OSError:
        raise AssertionError(f"fixture Git {operation} could not run") from None
    if result.returncode != 0:
        raise AssertionError(
            f"fixture Git {operation} exited {result.returncode}; "
            f"stdout={len(result.stdout)} bytes; "
            f"stderr={len(result.stderr)} bytes"
        )
    return result.stdout


def _git_dir(repository: Path, *arguments: str) -> bytes:
    environment = _fixture_git_environment(repository.parent)
    operation = _bounded_git_operation(arguments)
    try:
        result = subprocess.run(
            (
                _git_executable(),
                "-c",
                f"core.hooksPath={os.devnull}",
                f"--git-dir={repository}",
                *arguments,
            ),
            stdin=subprocess.DEVNULL,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(f"fixture Git {operation} timed out") from None
    except OSError:
        raise AssertionError(f"fixture Git {operation} could not run") from None
    if result.returncode != 0:
        raise AssertionError(
            f"fixture Git {operation} exited {result.returncode}; "
            f"stdout={len(result.stdout)} bytes; "
            f"stderr={len(result.stderr)} bytes"
        )
    return result.stdout


def _bounded_git_operation(arguments: Sequence[str]) -> str:
    if not arguments:
        return "unknown"
    operation = arguments[0]
    if operation in {
        "add",
        "commit",
        "config",
        "init",
        "push",
        "remote",
        "rev-parse",
    }:
        return operation
    return "unknown"


def _fixture_git_environment(anchor: Path) -> dict[str, str]:
    """Return bootstrap Git state isolated from all live user configuration."""
    state = anchor / ".transport-test-git-state"
    home = state / "home"
    config_home = state / "config"
    templates = state / "templates"
    temporary = state / "tmp"
    for directory in (state, home, config_home, templates, temporary):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        directory.chmod(0o700)
    return {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_PAGER": "",
        "GIT_TEMPLATE_DIR": str(templates),
        "GIT_TERMINAL_PROMPT": "0",
        "HOME": str(home),
        "LANG": "C",
        "LC_ALL": "C",
        "PAGER": "",
        "PATH": os.defpath,
        "TEMP": str(temporary),
        "TMP": str(temporary),
        "TMPDIR": str(temporary),
        "XDG_CONFIG_HOME": str(config_home),
    }


def _git_executable() -> str:
    executable = shutil.which("git", path=os.defpath)
    assert executable is not None
    return str(Path(executable).resolve())


def _git_exec_path() -> Path:
    result = subprocess.run(
        (_git_executable(), "--exec-path"),
        env={"LC_ALL": "C", "PATH": os.defpath},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0
    path = Path(result.stdout.decode().strip()).resolve(strict=True)
    assert path.is_dir()
    return path


def _identity(path: Path) -> FileSystemIdentity:
    metadata = path.stat()
    return FileSystemIdentity(metadata.st_dev, metadata.st_ino)


@dataclass(frozen=True)
class _CandidateRepository:
    parent_oid: str
    candidate_oid: str
    owner: FileNotesSessionOwner
    binding: SessionBinding


def _candidate_repository(
    root: Path,
    *,
    remote: Path,
    endpoint: str,
) -> _CandidateRepository:
    source = root / "notes"
    _git(root, "init", "--initial-branch=main", str(source))
    _git(source, "config", "user.name", "Chatbook Transport Test")
    _git(source, "config", "user.email", "transport@example.test")
    note = source / "note.md"
    note.write_text("parent\n", encoding="utf-8")
    _git(source, "add", "note.md")
    _git(source, "commit", "-m", "Parent")
    parent_oid = _git(source, "rev-parse", "HEAD").decode().strip()
    _git(root, "init", "--bare", str(remote))
    _git(source, "remote", "add", "origin", str(remote))
    _git(source, "push", "--set-upstream", "origin", "main")
    note.write_text("candidate\n", encoding="utf-8")
    _git(source, "add", "note.md")
    _git(source, "commit", "-m", "Candidate")
    candidate_oid = _git(source, "rev-parse", "HEAD").decode().strip()
    _git(source, "remote", "set-url", "--push", "origin", endpoint)
    owner, binding = _owner_for_candidate(
        source,
        parent_oid,
        candidate_oid,
    )
    return _CandidateRepository(
        parent_oid,
        candidate_oid,
        owner,
        binding,
    )


def _owner_for_candidate(
    source: Path,
    parent_oid: str,
    candidate_oid: str,
) -> tuple[FileNotesSessionOwner, SessionBinding]:
    git_dir = source / ".git"
    repository = RepositoryIdentity(
        worktree_root=str(source.resolve()),
        git_dir=str(git_dir.resolve()),
        git_common_dir=str(git_dir.resolve()),
        worktree_identity=_identity(source),
        git_dir_identity=_identity(git_dir),
        git_common_dir_identity=_identity(git_dir),
    )
    owner = FileNotesSessionOwner()
    binding = owner.select_root(source)
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
    assert owner.publish_trust(binding, repository)
    parent_blob = _git(source, "rev-parse", f"{parent_oid}:note.md").decode().strip()
    candidate_blob = _git(
        source,
        "rev-parse",
        f"{candidate_oid}:note.md",
    ).decode().strip()
    old_head = HeadIdentity.attached(BRANCH_REF, parent_oid)
    ownership = StagingOwnership(
        repository=repository,
        head=old_head,
        approved_endpoint_topology=("note.md",),
        approved_move_edges=(),
        approved_current_path="note.md",
        original_baselines={
            "note.md": IndexBaseline(
                IndexEntry("note.md", "100644", parent_blob)
            )
        },
        post_stage_entries={
            "note.md": IndexEntry("note.md", "100644", candidate_blob)
        },
    )
    assert owner.publish_ownership(binding, {1: ownership})
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    reviewed = owner._capture_commit_authority_after_review(
        lease,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=old_head,
        group_sequence_ids={1: (1,)},
        subject="Guarded candidate",
        included_notes=(PushIncludedNote(1, "note.md"),),
        change_types=("Modified",),
    )
    assert reviewed is not None
    capture = owner._recapture_commit_authority(
        lease,
        prior_capture=reviewed,
    )
    assert capture is not None
    publication = owner.publish_commit_outcome(
        lease,
        capture,
        CommitPublication(
            "succeeded",
            new_head=HeadIdentity.attached(BRANCH_REF, candidate_oid),
            retired_sequence_ids=(1,),
            candidate_seed=capture._candidate_seed,
        ),
    )
    assert publication.published
    lease.release()
    return owner, binding


class _TransportRunner(AsyncGitProcessRunner):
    """Run real Git while retaining only bounded, secret-free evidence."""

    def __init__(
        self,
        *,
        ca_certificate: Path | None = None,
        git_exec_path: Path | None = None,
        decorate_push_output: bool = False,
    ) -> None:
        super().__init__()
        self._ca_certificate = ca_certificate
        self._git_exec_path = git_exec_path
        self._decorate_push_output = decorate_push_output
        self.network_commands: list[str] = []
        self.network_environment_safe: list[bool] = []
        self.command_evidence: list[tuple[str, int | None, int, int]] = []
        self.raw_canary_hits: set[str] = set()

    async def run(
        self,
        argv: Sequence[GitArg],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
        owned_process_tree: bool = False,
    ) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        is_network = "ls-remote" in command or "push" in command
        child_environment = environment
        if is_network:
            self.network_commands.append(
                "push" if "push" in command else "ls-remote"
            )
            self.network_environment_safe.append(
                all(PROVIDER_CANARY not in value for value in environment.values())
            )
            if self._ca_certificate is not None:
                injected = dict(environment)
                injected["GIT_SSL_CAINFO"] = str(self._ca_certificate)
                child_environment = injected
        result = await super().run(
            argv,
            cwd=cwd,
            environment=child_environment,
            stdin=stdin,
            timeout=timeout,
            stdout_limit=stdout_limit,
            stderr_limit=stderr_limit,
            on_spawn=on_spawn,
            cancel_before_spawn=cancel_before_spawn,
            owned_process_tree=owned_process_tree,
        )
        if (
            self._git_exec_path is not None
            and "--exec-path" in command
            and result.returncode == 0
        ):
            result = replace(
                result,
                stdout=os.fsencode(self._git_exec_path) + b"\n",
            )
        if is_network:
            for canary in (
                HELPER_OUTPUT_CANARY,
                CREDENTIAL_CANARY,
                SERVER_OUTPUT_CANARY,
                RAW_STDOUT_CANARY,
                RAW_STDERR_CANARY,
            ):
                if canary.encode() in result.stdout + result.stderr:
                    self.raw_canary_hits.add(canary)
            if self._decorate_push_output and "push" in command:
                result = replace(
                    result,
                    stdout=(
                        result.stdout
                        + b"!\t"
                        + RAW_STDOUT_CANARY.encode()
                        + b"\tmalformed\n"
                    ),
                    stderr=result.stderr + RAW_STDERR_CANARY.encode() + b"\n",
                )
                self.raw_canary_hits.update(
                    {RAW_STDOUT_CANARY, RAW_STDERR_CANARY}
                )
        interesting = next(
            (
                name
                for name in (
                    "--exec-path",
                    "symbolic-ref",
                    "rev-parse",
                    "cat-file",
                    "config",
                    "diff-tree",
                    "read-tree",
                    "check-attr",
                    "ls-remote",
                    "push",
                )
                if name in command
            ),
            command[-1],
        )
        self.command_evidence.append(
            (interesting, result.returncode, len(result.stdout), len(result.stderr))
        )
        return result


def _service(
    candidate: _CandidateRepository,
    root: Path,
    *,
    runner: _TransportRunner,
    environment: Mapping[str, str],
    ssh_executable: Path | None = None,
    git_exec_path: Path | None = None,
) -> FileNotesGitService:
    contexts = root / "network-contexts"
    contexts.mkdir(mode=0o700)
    return FileNotesGitService(
        candidate.owner,
        runner=runner,
        git_executable=_git_executable(),
        environment=environment,
        network_context_factory=git_network.NetworkContextFactory(
            environment=environment,
            temporary_parent=contexts,
            git_executable=_git_executable(),
            git_exec_path=(
                _git_exec_path() if git_exec_path is None else git_exec_path
            ),
            ssh_executable=(
                None if ssh_executable is None else str(ssh_executable)
            ),
        ),
        push_query_timeout=5,
        push_timeout=5,
    )


def _run_checked(argv: Sequence[str], *, timeout: float = 10) -> None:
    operation = Path(argv[0]).name if argv else "fixture tool"
    if operation not in {"openssl", "ssh-keygen"}:
        operation = "fixture tool"
    try:
        result = subprocess.run(
            tuple(argv),
            env={"LC_ALL": "C", "PATH": os.defpath},
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"{operation} timed out") from None
    except OSError:
        raise RuntimeError(f"{operation} could not run") from None
    if result.returncode != 0:
        raise RuntimeError(
            f"{operation} exited {result.returncode}; "
            f"stdout={len(result.stdout)} bytes; "
            f"stderr={len(result.stderr)} bytes"
        )


def _append_only_script(path: Path, source: str) -> Path:
    path.write_text(
        f"#!{Path(sys.executable).resolve()} -I\n{source}",
        encoding="utf-8",
    )
    path.chmod(0o700)
    return path


@dataclass(repr=False)
class _OpenSSHServer:
    root: Path
    port: int
    host_public_key: str
    client_key: Path
    wrong_client_key: Path
    connections_log: Path
    process: subprocess.Popen[bytes]
    stderr_lines: list[str]
    stderr_condition: threading.Condition
    connection_baseline: int = 0

    def endpoint(self, remote: Path) -> str:
        return (
            f"ssh://{getpass.getuser()}@127.0.0.1:{self.port}"
            f"{remote.resolve()}"
        )

    def reset_counts(self) -> None:
        self.connections_log.write_bytes(b"")
        self.connection_baseline = self._total_connections()

    def _total_connections(self) -> int:
        with self.stderr_condition:
            return sum(
                line.startswith("Connection from 127.0.0.1 port ")
                for line in self.stderr_lines
            )

    def wait_for_connections(self, expected: int) -> bool:
        with self.stderr_condition:
            return self.stderr_condition.wait_for(
                lambda: self._total_connections()
                >= self.connection_baseline + expected,
                timeout=2,
            )

    def counts(self) -> tuple[int, int, int]:
        kinds = self.connections_log.read_text(encoding="utf-8").splitlines()
        return (
            self._total_connections() - self.connection_baseline,
            kinds.count("upload"),
            kinds.count("receive"),
        )

    def client_wrapper(
        self,
        root: Path,
        *,
        known_host: str | None = None,
        identity: Path | None = None,
    ) -> tuple[Path, Path, Path]:
        known_hosts = root / "known_hosts"
        host_key = self.host_public_key if known_host is None else known_host
        known_hosts.write_text(
            f"[127.0.0.1]:{self.port} {host_key}\n",
            encoding="ascii",
        )
        known_hosts.chmod(0o600)
        prompt_marker = root / "ssh-prompt-ran"
        prompt = _append_only_script(
            root / "ssh-askpass",
            (
                "from pathlib import Path\n"
                f"Path({str(prompt_marker)!r}).write_text('prompted')\n"
                "raise SystemExit(72)\n"
            ),
        )
        environment_marker = root / "provider-canary-reached-ssh"
        selected_identity = self.client_key if identity is None else identity
        wrapper = _append_only_script(
            root / "fixture-ssh",
            (
                "import os\n"
                "from pathlib import Path\n"
                "import sys\n"
                f"if any({PROVIDER_CANARY!r} in value "
                "for value in os.environ.values()):\n"
                f"    Path({str(environment_marker)!r}).write_text('leaked')\n"
                "args = sys.argv[1:]\n"
                "try:\n"
                "    split = args.index('--')\n"
                "except ValueError:\n"
                "    raise SystemExit(126) from None\n"
                "fixture = [\n"
                "    '-o', "
                f"'UserKnownHostsFile={known_hosts}',\n"
                "    '-o', 'GlobalKnownHostsFile=none',\n"
                "    '-o', 'IdentitiesOnly=yes',\n"
                "    '-o', 'IdentityFile=none',\n"
                f"    '-i', {str(selected_identity)!r},\n"
                "]\n"
                "child = {\n"
                "    'DISPLAY': 'fixture:0',\n"
                "    'LC_ALL': 'C',\n"
                f"    'SSH_ASKPASS': {str(prompt)!r},\n"
                "    'SSH_ASKPASS_REQUIRE': 'force',\n"
                "}\n"
                f"os.execve('/usr/bin/ssh', "
                "('/usr/bin/ssh', *args[:split], *fixture, *args[split:]), child)\n"
            ),
        )
        return wrapper, prompt_marker, environment_marker

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)


def _drain_stream(
    stream,
    lines: list[str],
    condition: threading.Condition,
) -> None:
    for raw_line in iter(stream.readline, b""):
        with condition:
            if len(lines) < 200:
                lines.append(raw_line.decode(errors="replace").rstrip())
            condition.notify_all()


@pytest.fixture(scope="module")
def openssh_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> _OpenSSHServer:
    if os.name != "posix":
        pytest.skip("guarded push is POSIX-only")
    tool_path = os.pathsep.join((os.defpath, "/usr/sbin", "/usr/local/sbin"))
    required = {
        name: shutil.which(name, path=tool_path)
        for name in ("sshd", "ssh-keygen")
    }
    if any(value is None for value in required.values()):
        pytest.skip("OpenSSH server/key tools are unavailable")
    root = tmp_path_factory.mktemp("guarded-push-openssh")
    root.chmod(0o700)
    host_key = root / "ssh_host_ed25519_key"
    client_key = root / "client_ed25519"
    wrong_client_key = root / "wrong_client_ed25519"
    for key in (host_key, client_key, wrong_client_key):
        _run_checked(
            (
                required["ssh-keygen"] or "ssh-keygen",
                "-q",
                "-t",
                "ed25519",
                "-N",
                "",
                "-f",
                str(key),
            )
        )
    listener = socket.socket()
    try:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    except PermissionError as error:
        pytest.skip(f"ephemeral OpenSSH fixture unavailable: {error}")
    finally:
        listener.close()
    authorized_keys = root / "authorized_keys"
    authorized_keys.write_bytes(client_key.with_suffix(".pub").read_bytes())
    authorized_keys.chmod(0o600)
    connections_log = root / "connections.log"
    connections_log.write_bytes(b"")
    server_wrapper = _append_only_script(
        root / "serve-git",
        (
            "import os\n"
            "from pathlib import Path\n"
            "import shlex\n"
            "import shutil\n"
            "command = os.environ.get('SSH_ORIGINAL_COMMAND', '')\n"
            "try:\n"
            "    parts = shlex.split(command, posix=True)\n"
            "except ValueError:\n"
            "    raise SystemExit(126) from None\n"
            "if len(parts) != 2 or parts[0] not in "
            "{'git-upload-pack', 'git-receive-pack'}:\n"
            "    raise SystemExit(126)\n"
            "repository = Path(parts[1]).resolve(strict=True)\n"
            f"repository.relative_to(Path({str(root)!r}).resolve())\n"
            "kind = 'upload' if parts[0] == 'git-upload-pack' else 'receive'\n"
            f"with Path({str(connections_log)!r}).open('a', encoding='utf-8') "
            "as stream:\n"
            "    stream.write(kind + '\\n')\n"
            "executable = shutil.which(parts[0], path=os.defpath)\n"
            "if executable is None:\n"
            "    raise SystemExit(127)\n"
            "environment = {'LC_ALL': 'C', 'PATH': os.defpath}\n"
            "protocol = os.environ.get('GIT_PROTOCOL')\n"
            "if protocol is not None:\n"
            "    environment['GIT_PROTOCOL'] = protocol\n"
            "os.execve(executable, (executable, str(repository)), environment)\n"
        ),
    )
    config = root / "sshd_config"
    config.write_text(
        "\n".join(
            (
                f"Port {port}",
                "ListenAddress 127.0.0.1",
                f"HostKey {host_key}",
                f"PidFile {root / 'sshd.pid'}",
                f"AuthorizedKeysFile {authorized_keys}",
                f"ForceCommand {server_wrapper}",
                f"AllowUsers {getpass.getuser()}",
                "StrictModes no",
                "PubkeyAuthentication yes",
                "PasswordAuthentication no",
                "KbdInteractiveAuthentication no",
                "ChallengeResponseAuthentication no",
                "UsePAM no",
                "PermitTTY no",
                "AllowTcpForwarding no",
                "X11Forwarding no",
                "PermitTunnel no",
                "AcceptEnv GIT_PROTOCOL",
                "LogLevel VERBOSE",
                "",
            )
        ),
        encoding="utf-8",
    )
    process = subprocess.Popen(
        (
            str(Path(required["sshd"] or "sshd").resolve()),
            "-D",
            "-e",
            "-f",
            str(config),
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        env={"LC_ALL": "C", "PATH": os.defpath},
    )
    assert process.stderr is not None
    startup_lines: list[str] = []
    selector = selectors.DefaultSelector()
    selector.register(process.stderr, selectors.EVENT_READ)
    listening = False
    try:
        while selector.select(timeout=5):
            line = process.stderr.readline()
            if not line:
                break
            decoded = line.decode(errors="replace").rstrip()
            startup_lines.append(decoded)
            if "Server listening" in decoded:
                listening = True
                break
    finally:
        selector.close()
    if not listening:
        process.terminate()
        process.wait(timeout=5)
        raise RuntimeError(
            "ephemeral OpenSSH daemon unavailable; "
            f"diagnostic_lines={len(startup_lines)}"
        )
    stderr_condition = threading.Condition(threading.RLock())
    drain = threading.Thread(
        target=_drain_stream,
        args=(process.stderr, startup_lines, stderr_condition),
        daemon=True,
    )
    drain.start()
    host_public_key = " ".join(
        host_key.with_suffix(".pub").read_text(encoding="ascii").split()[:2]
    )
    fixture = _OpenSSHServer(
        root,
        port,
        host_public_key,
        client_key,
        wrong_client_key,
        connections_log,
        process,
        startup_lines,
        stderr_condition,
    )
    try:
        yield fixture
    finally:
        fixture.close()
        drain.join(timeout=5)
        if drain.is_alive():
            raise RuntimeError("OpenSSH diagnostic drain did not stop")


@pytest.mark.asyncio
async def test_openssh_authorization_and_confirm_bound_the_only_push(
    tmp_path: Path,
    openssh_server: _OpenSSHServer,
) -> None:
    """Removing authorization, fixed routing, or the exact CAS must fail."""
    case = tmp_path / "ssh-success"
    case.mkdir(mode=0o700)
    remote = openssh_server.root / f"success-{id(case):x}.git"
    endpoint = openssh_server.endpoint(remote)
    candidate = _candidate_repository(case, remote=remote, endpoint=endpoint)
    hostile_home = case / "hostile-home"
    hostile_ssh = hostile_home / ".ssh"
    hostile_ssh.mkdir(parents=True, mode=0o700)
    proxy_marker = case / "hostile-proxy-ran"
    helper_marker = case / "hostile-credential-helper-ran"
    helper = _append_only_script(
        case / "hostile-credential-helper",
        (
            "from pathlib import Path\n"
            f"Path({str(helper_marker)!r}).write_text('invoked')\n"
            "raise SystemExit(74)\n"
        ),
    )
    (hostile_home / ".gitconfig").write_text(
        (
            "[credential]\n"
            f"\thelper = !{shlex.quote(str(helper))}\n"
        ),
        encoding="utf-8",
    )
    (hostile_ssh / "config").write_text(
        (
            "Host 127.0.0.1\n"
            "    HostName 192.0.2.99\n"
            f"    ProxyCommand touch {proxy_marker}\n"
            f"    IdentityFile {openssh_server.wrong_client_key}\n"
        ),
        encoding="utf-8",
    )
    wrapper, prompt_marker, environment_marker = openssh_server.client_wrapper(
        case
    )
    environment = {
        "HOME": str(hostile_home),
        "PATH": os.defpath,
        "OPENAI_API_KEY": PROVIDER_CANARY,
    }
    openssh_server.reset_counts()
    runner = _TransportRunner()
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        ssh_executable=wrapper,
    )
    try:
        local = await service.start_push_review(candidate.binding)

        assert local.state == "ready", runner.command_evidence
        assert openssh_server.counts() == (0, 0, 0)
        assert not helper_marker.exists()
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None

        reviewed = await service.authorize_and_check_push(
            candidate.binding,
            operation,
        )

        assert reviewed.state == "review"
        assert reviewed.handle is not None
        assert await asyncio.to_thread(openssh_server.wait_for_connections, 1)
        assert openssh_server.counts() == (1, 1, 0)
        assert not helper_marker.exists()
        result = await service.start_push(candidate.binding, reviewed.handle)

        assert result.state == "succeeded"
        assert await asyncio.to_thread(openssh_server.wait_for_connections, 4)
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.candidate_oid
        )
        assert runner.network_commands == [
            "ls-remote",
            "ls-remote",
            "push",
            "ls-remote",
        ]
        assert all(runner.network_environment_safe)
        assert openssh_server.counts() == (4, 3, 1)
        assert not proxy_marker.exists()
        assert not helper_marker.exists()
        assert not prompt_marker.exists()
        assert not environment_marker.exists()
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    ["unknown_host", "wrong_host_key", "bad_authentication"],
)
async def test_openssh_trust_and_authentication_fail_without_prompting(
    tmp_path: Path,
    openssh_server: _OpenSSHServer,
    failure: str,
) -> None:
    """Weakening strict host checks, batch mode, or identity pinning must fail."""
    case = tmp_path / failure
    case.mkdir(mode=0o700)
    remote = openssh_server.root / f"{failure}-{id(case):x}.git"
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=openssh_server.endpoint(remote),
    )
    known_host = (
        " ".join(
            openssh_server.wrong_client_key.with_suffix(".pub")
            .read_text(encoding="ascii")
            .split()[:2]
        )
        if failure == "wrong_host_key"
        else None
    )
    identity = (
        openssh_server.wrong_client_key
        if failure == "bad_authentication"
        else None
    )
    wrapper, prompt_marker, environment_marker = openssh_server.client_wrapper(
        case,
        known_host=known_host,
        identity=identity,
    )
    if failure == "unknown_host":
        (case / "known_hosts").write_bytes(b"")
    environment = {
        "HOME": str(case / "isolated-home"),
        "PATH": os.defpath,
        "OPENAI_API_KEY": PROVIDER_CANARY,
    }
    openssh_server.reset_counts()
    runner = _TransportRunner()
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        ssh_executable=wrapper,
    )
    try:
        local = await service.start_push_review(candidate.binding)
        assert local.state == "ready"
        assert openssh_server.counts() == (0, 0, 0)
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None

        result = await asyncio.wait_for(
            service.authorize_and_check_push(candidate.binding, operation),
            timeout=6,
        )

        assert result.state == "blocked"
        assert runner.network_commands == ["ls-remote"]
        assert all(runner.network_environment_safe)
        assert await asyncio.to_thread(openssh_server.wait_for_connections, 1)
        assert openssh_server.counts() == (1, 0, 0)
        assert not prompt_marker.exists()
        assert not environment_marker.exists()
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.parent_oid
        )
    finally:
        await service.shutdown()


@dataclass(frozen=True)
class _HTTPSCounts:
    requests: int = 0
    auth_challenges: int = 0
    upload_requests: int = 0
    receive_attempts: int = 0
    receive_backend_runs: int = 0


class _GitHTTPHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler contract
        self.server.transport_fixture.handle(self)  # type: ignore[attr-defined]

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler contract
        self.server.transport_fixture.handle(self)  # type: ignore[attr-defined]

    def log_message(self, _format: str, *_arguments: object) -> None:
        return None


class _GitHTTPServer(http.server.ThreadingHTTPServer):
    def handle_error(
        self,
        _request: object,
        _client_address: object,
    ) -> None:
        """Keep raw request/handler diagnostics out of captured test output."""
        return None


@dataclass(repr=False)
class _HTTPSFixture:
    root: Path
    ca_certificate: Path
    git_exec_path: Path
    helper_log: Path
    helper_password_path: Path
    username: str
    password: str
    server: _GitHTTPServer
    thread: threading.Thread
    lock: threading.RLock
    modes: dict[str, str]
    counters: dict[str, _HTTPSCounts]

    @property
    def port(self) -> int:
        return int(self.server.server_address[1])

    def remote(self, name: str) -> Path:
        assert name and "/" not in name and "\\" not in name
        return self.root / f"{name}.git"

    def endpoint(self, name: str, *, hostname: str = "localhost") -> str:
        return f"https://{hostname}:{self.port}/{name}.git"

    def configure_remote(
        self,
        remote: Path,
        *,
        mode: str = "normal",
        hostile_hook: bool = False,
        helper_password: str | None = None,
    ) -> None:
        assert remote.parent == self.root and remote.name.endswith(".git")
        _git_dir(remote, "config", "http.receivepack", "true")
        if hostile_hook:
            hook = remote / "hooks" / "post-receive"
            hook.parent.mkdir(mode=0o700, exist_ok=True)
            _append_only_script(
                hook,
                (
                    "import sys\n"
                    f"print({SERVER_OUTPUT_CANARY!r}, file=sys.stderr)\n"
                ),
            )
        with self.lock:
            self.modes[remote.name] = mode
            self.counters[remote.name] = _HTTPSCounts()
        self.helper_log.write_bytes(b"")
        self.helper_password_path.write_text(
            self.password if helper_password is None else helper_password,
            encoding="utf-8",
        )
        self.helper_password_path.chmod(0o600)

    def counts(self, remote: Path) -> _HTTPSCounts:
        with self.lock:
            return self.counters.get(remote.name, _HTTPSCounts())

    def helper_records(self) -> tuple[dict[str, object], ...]:
        if not self.helper_log.exists():
            return ()
        return tuple(
            json.loads(line)
            for line in self.helper_log.read_text(encoding="utf-8").splitlines()
            if line
        )

    def client_environment(self, root: Path) -> tuple[dict[str, str], Path]:
        home = root / "https-home"
        home.mkdir(mode=0o700)
        (home / ".gitconfig").write_text(
            (
                "[credential]\n"
                "\thelper =\n"
                "\thelper = osxkeychain\n"
                "\tuseHttpPath = true\n"
            ),
            encoding="utf-8",
        )
        prompt_marker = root / "https-prompt-ran"
        prompt = _append_only_script(
            root / "https-askpass",
            (
                "from pathlib import Path\n"
                f"Path({str(prompt_marker)!r}).write_text('prompted')\n"
                "raise SystemExit(72)\n"
            ),
        )
        return (
            {
                "GIT_ASKPASS": str(prompt),
                "HOME": str(home),
                "OPENAI_API_KEY": PROVIDER_CANARY,
                "PATH": os.defpath,
                "SSH_ASKPASS": str(prompt),
            },
            prompt_marker,
        )

    def handle(self, handler: _GitHTTPHandler) -> None:
        path, _, query = handler.path.partition("?")
        components = path.lstrip("/").split("/", 1)
        repository_name = components[0] if components else ""
        repository = self.root / repository_name
        if (
            not repository_name.endswith(".git")
            or repository.parent != self.root
            or not repository.is_dir()
        ):
            self._plain_response(handler, 404, b"")
            return
        with self.lock:
            counts = self.counters.get(repository_name, _HTTPSCounts())
            self.counters[repository_name] = replace(
                counts,
                requests=counts.requests + 1,
            )
        expected = "Basic " + base64.b64encode(
            f"{self.username}:{self.password}".encode()
        ).decode("ascii")
        if handler.headers.get("Authorization") != expected:
            with self.lock:
                counts = self.counters[repository_name]
                self.counters[repository_name] = replace(
                    counts,
                    auth_challenges=counts.auth_challenges + 1,
                )
            handler.send_response(401)
            handler.send_header("WWW-Authenticate", 'Basic realm="fixture"')
            handler.send_header("Content-Length", "0")
            handler.end_headers()
            return

        is_receive = (
            handler.command == "POST" and path.endswith("/git-receive-pack")
        )
        is_upload = (
            "service=git-upload-pack" in query
            or path.endswith("/git-upload-pack")
        )
        with self.lock:
            counts = self.counters[repository_name]
            self.counters[repository_name] = replace(
                counts,
                upload_requests=counts.upload_requests + int(is_upload),
                receive_attempts=counts.receive_attempts + int(is_receive),
            )
            mode = self.modes.get(repository_name, "normal")
        body = self._read_body(handler)
        if is_receive and mode == "drop_before_accept":
            self._drop_connection(handler)
            return
        backend = self._run_backend(
            handler,
            path=path,
            query=query,
            body=body,
        )
        if is_receive:
            with self.lock:
                counts = self.counters[repository_name]
                self.counters[repository_name] = replace(
                    counts,
                    receive_backend_runs=counts.receive_backend_runs + 1,
                )
            if mode == "drop_after_accept":
                self._drop_connection(handler)
                return
        self._backend_response(handler, backend)

    @staticmethod
    def _read_body(handler: _GitHTTPHandler) -> bytes:
        length_text = handler.headers.get("Content-Length", "0")
        try:
            length = int(length_text)
        except ValueError:
            length = -1
        if length < 0 or length > 16 * 1024 * 1024:
            raise RuntimeError("invalid fixture request length")
        return handler.rfile.read(length) if length else b""

    def _run_backend(
        self,
        handler: _GitHTTPHandler,
        *,
        path: str,
        query: str,
        body: bytes,
    ) -> subprocess.CompletedProcess[bytes]:
        environment = {
            "AUTH_TYPE": "Basic",
            "CONTENT_LENGTH": str(len(body)),
            "CONTENT_TYPE": handler.headers.get("Content-Type", ""),
            "GIT_HTTP_EXPORT_ALL": "1",
            "GIT_PROJECT_ROOT": str(self.root),
            "LC_ALL": "C",
            "PATH": os.defpath,
            "PATH_INFO": path,
            "QUERY_STRING": query,
            "REMOTE_ADDR": "127.0.0.1",
            "REMOTE_USER": self.username,
            "REQUEST_METHOD": handler.command,
            "SCRIPT_NAME": "",
            "SERVER_NAME": "localhost",
            "SERVER_PORT": str(self.port),
            "SERVER_PROTOCOL": handler.request_version,
        }
        protocol = handler.headers.get("Git-Protocol")
        if protocol is not None:
            environment["HTTP_GIT_PROTOCOL"] = protocol
        return subprocess.run(
            (_git_executable(), "http-backend"),
            cwd=self.root,
            env=environment,
            input=body,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
            check=False,
        )

    @staticmethod
    def _backend_response(
        handler: _GitHTTPHandler,
        result: subprocess.CompletedProcess[bytes],
    ) -> None:
        if result.returncode != 0:
            _HTTPSFixture._plain_response(handler, 500, b"")
            return
        if b"\r\n\r\n" in result.stdout:
            raw_headers, body = result.stdout.split(b"\r\n\r\n", 1)
            header_lines = raw_headers.split(b"\r\n")
        elif b"\n\n" in result.stdout:
            raw_headers, body = result.stdout.split(b"\n\n", 1)
            header_lines = raw_headers.split(b"\n")
        else:
            _HTTPSFixture._plain_response(handler, 500, b"")
            return
        status = 200
        headers: list[tuple[str, str]] = []
        for line in header_lines:
            name, separator, value = line.partition(b":")
            if not separator:
                continue
            decoded_name = name.decode("ascii")
            decoded_value = value.strip().decode("latin-1")
            if decoded_name.lower() == "status":
                status = int(decoded_value.split(" ", 1)[0])
            else:
                headers.append((decoded_name, decoded_value))
        handler.send_response(status)
        for name, value in headers:
            handler.send_header(name, value)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        handler.wfile.write(body)

    @staticmethod
    def _plain_response(
        handler: _GitHTTPHandler,
        status: int,
        body: bytes,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", str(len(body)))
        handler.end_headers()
        if body:
            handler.wfile.write(body)

    @staticmethod
    def _drop_connection(handler: _GitHTTPHandler) -> None:
        handler.send_response(200)
        handler.send_header(
            "Content-Type",
            "application/x-git-receive-pack-result",
        )
        handler.send_header("Content-Length", "128")
        handler.send_header("Connection", "close")
        handler.end_headers()
        try:
            handler.wfile.write(b"0008NAK\n")
            handler.wfile.flush()
        except OSError:
            pass
        handler.close_connection = True
        try:
            handler.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        handler.connection.close()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError("HTTPS fixture thread did not stop")


def _generate_tls_material(root: Path, openssl: str) -> tuple[Path, Path]:
    ca_key = root / "ca.key"
    ca_certificate = root / "ca.pem"
    leaf_key = root / "leaf.key"
    leaf_request = root / "leaf.csr"
    leaf_certificate = root / "leaf.pem"
    extensions = root / "leaf-extensions.cnf"
    extensions.write_text(
        (
            "basicConstraints=CA:FALSE\n"
            "keyUsage=digitalSignature,keyEncipherment\n"
            "extendedKeyUsage=serverAuth\n"
            "subjectAltName=DNS:localhost\n"
        ),
        encoding="ascii",
    )
    _run_checked(
        (
            openssl,
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-days",
            "1",
            "-subj",
            "/CN=Chatbook Transport Test CA",
            "-keyout",
            str(ca_key),
            "-out",
            str(ca_certificate),
        )
    )
    _run_checked(
        (
            openssl,
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-sha256",
            "-subj",
            "/CN=localhost",
            "-keyout",
            str(leaf_key),
            "-out",
            str(leaf_request),
        )
    )
    _run_checked(
        (
            openssl,
            "x509",
            "-req",
            "-in",
            str(leaf_request),
            "-CA",
            str(ca_certificate),
            "-CAkey",
            str(ca_key),
            "-CAcreateserial",
            "-days",
            "1",
            "-sha256",
            "-extfile",
            str(extensions),
            "-out",
            str(leaf_certificate),
        )
    )
    return leaf_certificate, leaf_key


@pytest.fixture(scope="module")
def https_server(
    tmp_path_factory: pytest.TempPathFactory,
) -> _HTTPSFixture:
    if sys.platform != "darwin":
        pytest.skip("fake approved credential helper requires macOS policy")
    openssl = shutil.which("openssl", path=os.defpath)
    if openssl is None:
        pytest.skip("OpenSSL certificate tooling is unavailable")
    root = tmp_path_factory.mktemp("guarded-push-https")
    root.chmod(0o700)
    leaf_certificate, leaf_key = _generate_tls_material(root, openssl)
    ca_certificate = root / "ca.pem"
    git_exec_path = root / "git-exec"
    git_exec_path.mkdir(mode=0o700)
    remote_https = (_git_exec_path() / "git-remote-https").resolve(strict=True)
    (git_exec_path / "git-remote-https").symlink_to(remote_https)
    helper_log = root / "credential-helper.jsonl"
    helper_log.write_bytes(b"")
    helper_password_path = root / "credential-helper-password"
    username = "fixture-user"
    password = CREDENTIAL_CANARY
    helper_password_path.write_text(password, encoding="utf-8")
    helper_password_path.chmod(0o600)
    _append_only_script(
        git_exec_path / "git-credential-osxkeychain",
        (
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "import sys\n"
            "operation = sys.argv[1] if len(sys.argv) == 2 else 'invalid'\n"
            "sys.stdin.read()\n"
            "record = {\n"
            "    'operation': operation,\n"
            f"    'provider_safe': all({PROVIDER_CANARY!r} not in value "
            "for value in os.environ.values()),\n"
            "    'terminal_prompt': os.environ.get('GIT_TERMINAL_PROMPT'),\n"
            "    'git_askpass': 'GIT_ASKPASS' in os.environ,\n"
            "    'ssh_askpass': 'SSH_ASKPASS' in os.environ,\n"
            "}\n"
            f"with Path({str(helper_log)!r}).open('a', encoding='utf-8') "
            "as stream:\n"
            "    stream.write(json.dumps(record, sort_keys=True) + '\\n')\n"
            f"print({HELPER_OUTPUT_CANARY!r}, file=sys.stderr)\n"
            "if operation == 'get':\n"
            f"    print('username={username}')\n"
            f"    password = Path({str(helper_password_path)!r}).read_text("
            "encoding='utf-8')\n"
            "    print('password=' + password)\n"
        ),
    )
    lock = threading.RLock()
    modes: dict[str, str] = {}
    counters: dict[str, _HTTPSCounts] = {}
    try:
        server = _GitHTTPServer(
            ("127.0.0.1", 0),
            _GitHTTPHandler,
        )
    except PermissionError as error:
        pytest.skip(f"loopback HTTPS fixture unavailable: {error}")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(leaf_certificate, leaf_key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    fixture = _HTTPSFixture(
        root,
        ca_certificate,
        git_exec_path,
        helper_log,
        helper_password_path,
        username,
        password,
        server,
        thread,
        lock,
        modes,
        counters,
    )
    server.transport_fixture = fixture  # type: ignore[attr-defined]
    thread.start()
    try:
        yield fixture
    finally:
        fixture.close()


@pytest.mark.asyncio
async def test_https_verified_tls_helper_and_exact_push_are_noninteractive(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """TLS trust, copied helper policy, and prompt suppression are mandatory."""
    case = tmp_path / "https-success"
    case.mkdir(mode=0o700)
    name = f"success-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name),
    )
    https_server.configure_remote(remote, hostile_hook=True)
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        local = await service.start_push_review(candidate.binding)

        assert local.state == "ready", runner.command_evidence
        assert https_server.counts(remote) == _HTTPSCounts()
        assert https_server.helper_records() == ()
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None

        reviewed = await service.authorize_and_check_push(
            candidate.binding,
            operation,
        )

        assert reviewed.state == "review", runner.command_evidence
        assert reviewed.handle is not None
        assert https_server.counts(remote).upload_requests >= 1
        assert https_server.counts(remote).receive_attempts == 0
        helper_records = https_server.helper_records()
        assert any(record["operation"] == "get" for record in helper_records)

        result = await service.start_push(candidate.binding, reviewed.handle)

        assert result.state == "succeeded", runner.command_evidence
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.candidate_oid
        )
        counts = https_server.counts(remote)
        assert counts.receive_attempts == 1
        assert counts.receive_backend_runs == 1
        assert runner.network_commands == [
            "ls-remote",
            "ls-remote",
            "push",
            "ls-remote",
        ]
        assert all(runner.network_environment_safe)
        helper_records = https_server.helper_records()
        assert helper_records
        assert all(record["provider_safe"] is True for record in helper_records)
        assert all(record["terminal_prompt"] == "0" for record in helper_records)
        assert all(record["git_askpass"] is False for record in helper_records)
        assert all(record["ssh_askpass"] is False for record in helper_records)
        assert not prompt_marker.exists()
        credential_was_not_echoed = CREDENTIAL_CANARY not in (
            runner.raw_canary_hits
        )
        assert credential_was_not_echoed
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_https_rejected_helper_credentials_block_without_prompting(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """Rejected helper credentials must not fall back to interactive input."""
    case = tmp_path / "https-rejected-credentials"
    case.mkdir(mode=0o700)
    name = f"rejected-credentials-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name),
    )
    https_server.configure_remote(
        remote,
        helper_password="wrong-fixture-password",
    )
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        local = await service.start_push_review(candidate.binding)
        assert local.state == "ready", runner.command_evidence
        assert https_server.counts(remote) == _HTTPSCounts()
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None

        result = await asyncio.wait_for(
            service.authorize_and_check_push(candidate.binding, operation),
            timeout=6,
        )

        assert result.state == "blocked", runner.command_evidence
        assert runner.network_commands == ["ls-remote"]
        assert all(runner.network_environment_safe)
        assert https_server.counts(remote) == _HTTPSCounts(
            requests=2,
            auth_challenges=2,
        )
        assert [
            record["operation"] for record in https_server.helper_records()
        ] == ["get", "erase"]
        assert not prompt_marker.exists()
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.parent_oid
        )
    finally:
        await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failure", "hostname", "trust_fixture_ca"),
    (
        ("untrusted_ca", "localhost", False),
        ("wrong_hostname", "127.0.0.1", True),
    ),
)
async def test_https_certificate_failures_block_without_prompting(
    tmp_path: Path,
    https_server: _HTTPSFixture,
    failure: str,
    hostname: str,
    trust_fixture_ca: bool,
) -> None:
    """Disabling certificate-chain or hostname verification must fail."""
    case = tmp_path / failure
    case.mkdir(mode=0o700)
    name = f"{failure}-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name, hostname=hostname),
    )
    https_server.configure_remote(remote)
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=(
            https_server.ca_certificate if trust_fixture_ca else None
        ),
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        local = await service.start_push_review(candidate.binding)
        assert local.state == "ready", runner.command_evidence
        assert https_server.counts(remote) == _HTTPSCounts()
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None

        result = await asyncio.wait_for(
            service.authorize_and_check_push(candidate.binding, operation),
            timeout=6,
        )

        assert result.state == "blocked", runner.command_evidence
        assert runner.network_commands == ["ls-remote"]
        assert all(runner.network_environment_safe)
        assert https_server.counts(remote).receive_attempts == 0
        assert https_server.counts(remote).receive_backend_runs == 0
        assert https_server.helper_records() == ()
        assert not prompt_marker.exists()
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.parent_oid
        )
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_https_credential_bearing_endpoint_is_rejected_before_contact(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """Endpoint credentials and queries must never reach transport setup."""
    case = tmp_path / "credential-bearing-endpoint"
    case.mkdir(mode=0o700)
    name = f"credential-bearing-{id(case):x}"
    remote = https_server.remote(name)
    endpoint = (
        f"https://fixture-user:{CREDENTIAL_CANARY}@localhost:"
        f"{https_server.port}/{name}.git?token={CREDENTIAL_CANARY}"
    )
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=endpoint,
    )
    https_server.configure_remote(remote)
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        result = await service.start_push_review(candidate.binding)

        assert result.state == "blocked"
        assert runner.network_commands == []
        assert https_server.counts(remote) == _HTTPSCounts()
        assert https_server.helper_records() == ()
        assert not prompt_marker.exists()
        endpoint_was_redacted = CREDENTIAL_CANARY not in "\n".join(
            (
                repr(result),
                repr(candidate.owner.snapshot(candidate.binding)),
                repr(service),
            )
        )
        assert endpoint_was_redacted
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_https_transport_redacts_all_raw_boundary_canaries(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """Raw environment, helper, Git, and server text must stay transient."""
    case = tmp_path / "https-redaction"
    case.mkdir(mode=0o700)
    name = f"redaction-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name),
    )
    https_server.configure_remote(remote, hostile_hook=True)
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
        decorate_push_output=True,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    assert PROVIDER_CANARY in service._environment.values()
    factory = service._network_context_factory
    assert factory is not None
    factory_base_is_redacted = not _structured_state_contains(
        factory._base_environment,
        PROVIDER_CANARY,
    )
    assert factory_base_is_redacted
    log_messages: list[str] = []
    sink = logger.add(
        lambda message: log_messages.append(str(message)),
        format="{message}",
    )
    shutdown_complete = False
    try:
        local = await service.start_push_review(candidate.binding)
        assert local.state == "ready"
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None
        reviewed = await service.authorize_and_check_push(
            candidate.binding,
            operation,
        )
        assert reviewed.state == "review"
        assert reviewed.handle is not None

        result = await service.start_push(candidate.binding, reviewed.handle)

        assert result.state == "uncertain", runner.command_evidence
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.candidate_oid
        )
        expected_inputs_were_observed = {
            HELPER_OUTPUT_CANARY,
            SERVER_OUTPUT_CANARY,
            RAW_STDOUT_CANARY,
            RAW_STDERR_CANARY,
        } <= runner.raw_canary_hits
        credential_was_not_echoed = CREDENTIAL_CANARY not in (
            runner.raw_canary_hits
        )
        assert expected_inputs_were_observed
        assert credential_was_not_echoed
        assert all(runner.network_environment_safe)
        helper_records = https_server.helper_records()
        assert helper_records
        assert all(record["provider_safe"] is True for record in helper_records)
        assert not prompt_marker.exists()

        uncertain = service._uncertain_push
        retained = service.retained_push_operation(candidate.binding)
        owner_snapshot = candidate.owner.snapshot(candidate.binding)
        assert uncertain is not None
        assert retained is not None
        push_specific_state = (
            service._push_destination_policy,
            service._push_authorization,
            tuple(service._push_review_snapshots.values()),
            service._unsettled_push_preflight,
            service._retained_push_operation,
            service._uncertain_push,
            local,
            operation,
            reviewed,
            result,
            owner_snapshot,
        )
        structured_state_is_redacted = all(
            not _structured_state_contains(push_specific_state, canary)
            for canary in (
                PROVIDER_CANARY,
                HELPER_OUTPUT_CANARY,
                CREDENTIAL_CANARY,
                SERVER_OUTPUT_CANARY,
                RAW_STDOUT_CANARY,
                RAW_STDERR_CANARY,
            )
        )
        assert structured_state_is_redacted
        await service.shutdown()
        shutdown_complete = True
        final_owner_snapshot = candidate.owner.snapshot(candidate.binding)
        final_retained = service.retained_push_operation(candidate.binding)
        final_push_state = (
            service._push_destination_policy,
            service._push_authorization,
            tuple(service._push_review_snapshots.values()),
            service._unsettled_push_preflight,
            service._retained_push_operation,
            service._uncertain_push,
            final_retained,
            final_owner_snapshot,
        )
        final_structured_state_is_redacted = all(
            not _structured_state_contains(final_push_state, canary)
            for canary in (
                PROVIDER_CANARY,
                HELPER_OUTPUT_CANARY,
                CREDENTIAL_CANARY,
                SERVER_OUTPUT_CANARY,
                RAW_STDOUT_CANARY,
                RAW_STDERR_CANARY,
            )
        )
        assert final_structured_state_is_redacted
        retained_text = "\n".join(
            (
                repr(local),
                repr(operation),
                repr(reviewed),
                repr(result),
                repr(service),
                repr(uncertain),
                repr(retained),
                repr(owner_snapshot),
                repr(service._uncertain_push),
                repr(final_retained),
                repr(final_owner_snapshot),
                *log_messages,
            )
        )
        retained_surfaces_are_redacted = all(
            canary not in retained_text
            for canary in (
                PROVIDER_CANARY,
                HELPER_OUTPUT_CANARY,
                CREDENTIAL_CANARY,
                SERVER_OUTPUT_CANARY,
                RAW_STDOUT_CANARY,
                RAW_STDERR_CANARY,
            )
        )
        assert retained_surfaces_are_redacted
    finally:
        try:
            if not shutdown_complete:
                await service.shutdown()
        finally:
            logger.remove(sink)


@pytest.mark.asyncio
async def test_https_drop_after_acceptance_recovers_by_query_without_retry(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """A lost receive result remains uncertain until a query sees candidate."""
    case = tmp_path / "https-drop-after-accept"
    case.mkdir(mode=0o700)
    name = f"drop-after-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name),
    )
    https_server.configure_remote(remote, mode="drop_after_accept")
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        assert (await service.start_push_review(candidate.binding)).state == (
            "ready"
        )
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None
        reviewed = await service.authorize_and_check_push(
            candidate.binding,
            operation,
        )
        assert reviewed.state == "review"
        assert reviewed.handle is not None

        result = await service.start_push(candidate.binding, reviewed.handle)

        assert result.state == "uncertain", runner.command_evidence
        assert https_server.counts(remote).receive_attempts == 1
        assert https_server.counts(remote).receive_backend_runs == 1
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.candidate_oid
        )
        uncertain = service.retained_push_operation(candidate.binding)
        assert uncertain is not None

        recovered = await service.check_push_again(
            candidate.binding,
            uncertain,
        )

        assert recovered.state == "succeeded"
        assert recovered.query_only
        assert sum(command == "push" for command in runner.network_commands) == 1
        assert runner.network_commands[-1] == "ls-remote"
        assert https_server.counts(remote).receive_attempts == 1
        assert https_server.counts(remote).receive_backend_runs == 1
        assert not prompt_marker.exists()
    finally:
        await service.shutdown()


@pytest.mark.asyncio
async def test_https_drop_before_acceptance_stays_uncertain_across_queries(
    tmp_path: Path,
    https_server: _HTTPSFixture,
) -> None:
    """Repeated parent observations never infer that a dropped push failed."""
    case = tmp_path / "https-drop-before-accept"
    case.mkdir(mode=0o700)
    name = f"drop-before-{id(case):x}"
    remote = https_server.remote(name)
    candidate = _candidate_repository(
        case,
        remote=remote,
        endpoint=https_server.endpoint(name),
    )
    https_server.configure_remote(remote, mode="drop_before_accept")
    environment, prompt_marker = https_server.client_environment(case)
    runner = _TransportRunner(
        ca_certificate=https_server.ca_certificate,
        git_exec_path=https_server.git_exec_path,
    )
    service = _service(
        candidate,
        case,
        runner=runner,
        environment=environment,
        git_exec_path=https_server.git_exec_path,
    )
    try:
        assert (await service.start_push_review(candidate.binding)).state == (
            "ready"
        )
        operation = service.retained_push_operation(candidate.binding)
        assert operation is not None
        reviewed = await service.authorize_and_check_push(
            candidate.binding,
            operation,
        )
        assert reviewed.state == "review"
        assert reviewed.handle is not None

        result = await service.start_push(candidate.binding, reviewed.handle)

        assert result.state == "uncertain", runner.command_evidence
        assert https_server.counts(remote).receive_attempts == 1
        assert https_server.counts(remote).receive_backend_runs == 0
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.parent_oid
        )

        recovery_states: list[str] = []
        for _attempt in range(2):
            uncertain = service.retained_push_operation(candidate.binding)
            assert uncertain is not None
            recovery = await service.check_push_again(
                candidate.binding,
                uncertain,
            )
            recovery_states.append(recovery.state)
            assert recovery.query_only
            assert recovery.can_check_again

        assert recovery_states == ["uncertain", "uncertain"]
        assert sum(command == "push" for command in runner.network_commands) == 1
        assert runner.network_commands[-2:] == ["ls-remote", "ls-remote"]
        assert https_server.counts(remote).receive_attempts == 1
        assert https_server.counts(remote).receive_backend_runs == 0
        assert _git_dir(remote, "rev-parse", BRANCH_REF).decode().strip() == (
            candidate.parent_oid
        )
        assert not prompt_marker.exists()
    finally:
        await service.shutdown()
