from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import socket
import stat
import subprocess
import sys
import zlib
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, replace
from functools import lru_cache
from pathlib import Path
from typing import get_args

import pytest

import tldw_chatbook.Notes.file_notes_git_network as git_network
import tldw_chatbook.Notes.file_notes_git_push as push_contracts
import tldw_chatbook.Notes.file_notes_git_service as git_service
from tldw_chatbook.Notes.file_notes_git_push import PushContractError
from tldw_chatbook.Notes.file_notes_git_service import (
    AsyncGitProcessRunner,
    FileNotesGitService,
    GitCommandResult,
    build_local_push_proof_environment,
)
from tldw_chatbook.Notes.file_notes_session_owner import (
    CommitPublication,
    FileNotesSessionOwner,
    FileSystemIdentity,
    HeadIdentity,
    GitMutationAdmissionReason as OwnerGitMutationAdmissionReason,
    IndexBaseline,
    IndexEntry,
    PushIncludedNote,
    RepositoryIdentity,
    SessionBinding,
    SessionChange,
    SessionGitStatus,
    StagingOwnership,
)


BRANCH_REF = "refs/heads/main"


def test_git_mutation_admission_reasons_cover_push_recovery_contract() -> None:
    required_reasons = {"authorization_required", "recovery_not_ready"}

    assert required_reasons <= set(
        get_args(git_service.GitMutationAdmissionReason)
    )
    assert required_reasons <= set(get_args(OwnerGitMutationAdmissionReason))


def _fact(
    key: str,
    value: str,
    *,
    scope: str = "local",
    origin: str = "a" * 64,
):
    return push_contracts._GitConfigFact(scope, origin, key, value)


def _base_facts(*, push_url: bool = True):
    facts = [
        _fact("branch.main.remote", "origin"),
        _fact("branch.main.merge", BRANCH_REF),
        _fact("remote.origin.url", "https://fetch.example.test/team/notes.git"),
    ]
    if push_url:
        facts.append(
            _fact(
                "remote.origin.pushurl",
                "https://push.example.test/team/notes.git",
            )
        )
    return tuple(facts)


def _resolve(
    facts,
    *,
    admission=None,
):
    return push_contracts._resolve_push_configuration(
        tuple(facts),
        BRANCH_REF,
        push_contracts.TransportAdmission()
        if admission is None
        else admission,
    )


def test_destination_configuration_selects_one_push_url_or_one_fetch_fallback() -> None:
    explicit = _resolve(_base_facts())
    fallback = _resolve(_base_facts(push_url=False))

    assert explicit.transport.destination.host == "push.example.test"
    assert fallback.transport.destination.host == "fetch.example.test"
    assert explicit.merge_ref == BRANCH_REF
    assert explicit.tracking_remote == "origin"
    assert explicit.configuration_fingerprint != fallback.configuration_fingerprint


@pytest.mark.parametrize("remote_label", ["", "   "])
def test_destination_configuration_rejects_empty_tracking_remote_label(
    remote_label: str,
) -> None:
    facts = (
        _fact("branch.main.remote", remote_label),
        _fact("branch.main.merge", BRANCH_REF),
        _fact(
            f"remote.{remote_label}.url",
            "https://fetch.example.test/team/notes.git",
        ),
    )

    with pytest.raises(PushContractError) as error:
        _resolve(facts)

    assert error.value.code == "invalid_configuration"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda facts: [fact for fact in facts if fact.key != "branch.main.remote"],
        lambda facts: [*facts, _fact("branch.main.remote", "backup")],
        lambda facts: [fact for fact in facts if fact.key != "branch.main.merge"],
        lambda facts: [*facts, _fact("branch.main.merge", "refs/heads/other")],
        lambda facts: [
            *facts,
            _fact(
                "remote.origin.pushurl",
                "https://other.example.test/team/notes.git",
            ),
        ],
        lambda facts: [*facts, _fact("remote.origin.mirror", "true")],
        lambda facts: [
            *facts,
            _fact("remote.origin.push", "refs/heads/*:refs/heads/*"),
        ],
        lambda facts: [*facts, _fact("push.pushoption", "ci.skip")],
        lambda facts: [*facts, _fact("remote.origin.receivepack", "custom-rp")],
        lambda facts: [*facts, _fact("remote.origin.vcs", "danger")],
        lambda facts: [
            fact
            for fact in facts
            if fact.key not in {"remote.origin.pushurl", "remote.origin.url"}
        ],
        lambda facts: [
            *[
                fact
                for fact in facts
                if fact.key != "remote.origin.pushurl"
            ],
            _fact(
                "remote.origin.url",
                "https://second-fetch.example.test/team/notes.git",
            ),
        ],
        lambda facts: [
            replace(
                fact,
                value=".",
            )
            if fact.key == "branch.main.remote"
            else fact
            for fact in facts
        ],
    ],
    ids=[
        "missing-tracking",
        "plural-tracking",
        "missing-merge",
        "plural-merge",
        "multiple-push-urls",
        "mirror",
        "push-refspec",
        "push-option",
        "custom-receive-pack",
        "custom-remote-helper",
        "missing-url",
        "multiple-fetch-fallbacks",
        "dot-remote",
    ],
)
def test_destination_configuration_blocks_ambiguous_or_broadening_policy(
    mutation,
) -> None:
    with pytest.raises(PushContractError) as error:
        _resolve(mutation(list(_base_facts())))

    assert error.value.code == "invalid_configuration"


@pytest.mark.parametrize(
    ("selector_key", "selector_value", "allowed"),
    [
        ("branch.main.pushRemote", "origin", True),
        ("remote.pushDefault", "origin", True),
        ("branch.main.pushRemote", "backup", False),
        ("remote.pushDefault", "backup", False),
    ],
)
def test_destination_selector_can_only_resolve_to_the_tracking_remote(
    selector_key: str,
    selector_value: str,
    allowed: bool,
) -> None:
    facts = (*_base_facts(), _fact(selector_key, selector_value))

    if allowed:
        resolved = _resolve(facts)
        assert resolved.tracking_remote == "origin"
        assert resolved.transport.destination.host == "push.example.test"
    else:
        with pytest.raises(PushContractError):
            _resolve(facts)


def test_branch_push_remote_precedence_allows_same_tracking_remote() -> None:
    resolved = _resolve(
        (
            *_base_facts(),
            _fact("branch.main.pushRemote", "origin"),
            _fact("remote.pushDefault", "backup"),
        )
    )

    assert resolved.transport.destination.host == "push.example.test"


def test_explicit_push_url_ignores_push_rewrite_and_uses_ordinary_rewrite() -> None:
    facts = (
        *_base_facts(),
        _fact(
            "url.ssh://git@ignored.example:22/team/.pushInsteadOf",
            "https://push.example.test/team/",
        ),
        _fact(
            "url.https://ordinary.example/team/.insteadOf",
            "https://push.example.test/team/",
        ),
    )

    resolved = _resolve(facts)

    assert resolved.transport.destination.scheme == "https"
    assert resolved.transport.destination.host == "ordinary.example"


def test_fetch_url_fallback_uses_longest_push_rewrite_and_blocks_ties() -> None:
    facts = (
        *_base_facts(push_url=False),
        _fact("url.https://short.example/.pushInsteadOf", "https://fetch."),
        _fact(
            "url.ssh://git@literal.example:22/team/.pushInsteadOf",
            "https://fetch.example.test/team/",
        ),
        _fact(
            "url.https://ordinary.example/team/.insteadOf",
            "https://fetch.example.test/team/",
        ),
    )

    resolved = _resolve(facts)

    assert resolved.transport.destination.scheme == "ssh"
    assert resolved.transport.destination.host == "literal.example"
    with pytest.raises(PushContractError):
        _resolve(
            (
                *facts,
                _fact(
                    "url.https://tie.example/team/.pushInsteadOf",
                    "https://fetch.example.test/team/",
                ),
            )
        )


def test_explicit_push_url_blocks_ambiguous_ordinary_rewrite() -> None:
    with pytest.raises(PushContractError):
        _resolve(
            (
                *_base_facts(),
                _fact(
                    "url.https://first.example/team/.insteadOf",
                    "https://push.example.test/team/",
                ),
                _fact(
                    "url.https://second.example/team/.insteadOf",
                    "https://push.example.test/team/",
                ),
            )
        )


@pytest.mark.parametrize(
    ("key", "value", "scope"),
    [
        ("http.sslVerify", "false", "global"),
        ("http.https://push.example.test.sslCAInfo", "/tmp/ca", "local"),
        ("http.https://push.example.test.sslKey", "/tmp/key", "worktree"),
        ("http.https://push.example.test.extraHeader", "Secret: x", "local"),
        ("http.https://push.example.test.proxy", "exec://proxy", "local"),
        ("credential.helper", "!danger", "local"),
        ("core.sshCommand", "/tmp/ssh", "worktree"),
    ],
)
def test_destination_security_and_repository_executable_configuration_blocks(
    key: str,
    value: str,
    scope: str,
) -> None:
    with pytest.raises(PushContractError):
        _resolve((*_base_facts(), _fact(key, value, scope=scope)))


def test_global_credential_helper_is_fingerprinted_but_not_invoked_or_exposed() -> None:
    first = _resolve(
        (*_base_facts(), _fact("credential.helper", "store-one", scope="global"))
    )
    second = _resolve(
        (*_base_facts(), _fact("credential.helper", "store-two", scope="global"))
    )

    assert first.transport.destination == second.transport.destination
    assert first.configuration_fingerprint != second.configuration_fingerprint
    assert "store-one" not in repr(first)
    assert "store-two" not in repr(second)


def test_configuration_fingerprint_includes_origin_identity_and_value() -> None:
    initial = _resolve(_base_facts())
    changed_origin = _resolve(
        tuple(replace(fact, origin_identity="b" * 64) for fact in _base_facts())
    )
    changed_value = _resolve(
        tuple(
            replace(
                fact,
                value="https://push2.example.test/team/notes.git",
            )
            if fact.key == "remote.origin.pushurl"
            else fact
            for fact in _base_facts()
        )
    )

    assert len(
        {
            initial.configuration_fingerprint,
            changed_origin.configuration_fingerprint,
            changed_value.configuration_fingerprint,
        }
    ) == 3


def test_local_destination_proof_environment_strips_all_redirects_and_helpers() -> None:
    removed = {
        "GIT_DIR": "repo",
        "GIT_WORK_TREE": "worktree",
        "GIT_INDEX_FILE": "index",
        "GIT_OBJECT_DIRECTORY": "objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "alternates",
        "GIT_CONFIG": "config",
        "GIT_CONFIG_SYSTEM": "system",
        "GIT_CONFIG_GLOBAL": "global",
        "GIT_NAMESPACE": "namespace",
        "GIT_REPLACE_REF_BASE": "replace",
        "GIT_SSH": "ssh",
        "GIT_SSH_COMMAND": "ssh-command",
        "GIT_ASKPASS": "askpass",
        "SSH_ASKPASS": "ssh-askpass",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_ATTR_SOURCE": "refs/heads/other",
        "HTTPS_PROXY": "proxy",
        "http_proxy": "proxy",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.sshCommand",
        "GIT_CONFIG_VALUE_0": "helper",
    }

    environment = build_local_push_proof_environment(
        {**removed, "PATH": "/usr/bin"},
        index_file="/tmp/service-owned-index",
    )

    assert all(
        key not in environment
        for key in removed
        if key != "GIT_INDEX_FILE"
    )
    assert environment["GIT_INDEX_FILE"] == "/tmp/service-owned-index"
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["LC_ALL"] == "C"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"sha1\n" + b"a" * 40 + b"\n", ("sha1", "a" * 40)),
        (b"sha256\n" + b"b" * 64 + b"\n", ("sha256", "b" * 64)),
        (b"sha1\n" + b"a" * 64 + b"\n", None),
        (b"sha256\n" + b"b" * 40 + b"\n", None),
        (b"sha512\n" + b"c" * 64 + b"\n", None),
        (b"sha1\n" + b"a" * 40, None),
        (b"sha1\n" + b"a" * 40 + b"\nextra\n", None),
    ],
)
def test_git_object_format_and_oid_parser_is_exact(
    payload: bytes,
    expected: tuple[str, str] | None,
) -> None:
    assert git_service._git_object_format_and_oid(payload) == expected


def _filesystem_identity(path: Path) -> FileSystemIdentity:
    metadata = path.stat()
    return FileSystemIdentity(metadata.st_dev, metadata.st_ino)


def _network_repository(
    tmp_path: Path,
    *,
    object_format: str = "sha1",
) -> RepositoryIdentity:
    root = tmp_path / "source-notes"
    git_dir = root / ".git"
    objects = git_dir / "objects"
    (git_dir / "refs" / "heads").mkdir(parents=True)
    objects.mkdir()
    (git_dir / "config").write_text(
        (
            "[core]\n"
            f"\trepositoryFormatVersion = {1 if object_format == 'sha256' else 0}\n"
            "\tbare = false\n"
            + (
                "[extensions]\n\tobjectFormat = sha256\n"
                if object_format == "sha256"
                else ""
            )
        ),
        encoding="utf-8",
    )
    (git_dir / "index").write_bytes(b"source-index")
    (git_dir / "refs" / "heads" / "main").write_text(
        "b" * (64 if object_format == "sha256" else 40) + "\n",
        encoding="ascii",
    )
    (root / "note.md").write_text(
        "PRIVATE NOTE BODY MUST NOT ENTER NETWORK CONTEXT\n",
        encoding="utf-8",
    )
    return RepositoryIdentity(
        worktree_root=str(root.resolve()),
        git_dir=str(git_dir.resolve()),
        git_common_dir=str(git_dir.resolve()),
        worktree_identity=_filesystem_identity(root),
        git_dir_identity=_filesystem_identity(git_dir),
        git_common_dir_identity=_filesystem_identity(git_dir),
    )


def _network_destination(
    endpoint: str = "https://push.example.test/team/notes.git",
):
    return _network_endpoint(endpoint).projection


def _network_endpoint(
    endpoint: str = "https://push.example.test/team/notes.git",
):
    return push_contracts._freeze_push_endpoint(endpoint, BRANCH_REF)


def _network_authorizations(
    repository: RepositoryIdentity,
    destination,
    *,
    facts=(),
    object_format: str = "sha1",
    environment: Mapping[str, str] | None = None,
):
    source_objects = Path(repository.git_common_dir) / "objects"
    source_authorization = git_network._authorize_source_object_directory(
        source_objects,
        _filesystem_identity(source_objects),
        object_format,
    )
    config_authorization = (
        git_network._authorize_network_config_snapshot(
            tuple(facts),
            configuration_fingerprint="f" * 64,
            destination=destination,
            environment=environment,
            repository=repository,
        )
        if destination.scheme == "ssh"
        else git_network._authorize_network_config_facts(
            tuple(facts),
            configuration_fingerprint="f" * 64,
            destination=destination,
        )
    )
    return source_authorization, config_authorization


@lru_cache(maxsize=1)
def _test_git_installation() -> tuple[Path, Path]:
    developer_git = Path(
        "/Library/Developer/CommandLineTools/usr/bin/git"
    )
    if developer_git.is_file():
        git_executable = developer_git
    else:
        selected = shutil.which("git", path=os.defpath)
        assert selected is not None
        git_executable = Path(selected)
    result = subprocess.run(
        (str(git_executable), "--exec-path"),
        env={"LC_ALL": "C", "PATH": os.defpath},
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    assert result.returncode == 0
    git_exec_path = Path(result.stdout.strip())
    assert git_exec_path.is_absolute() and git_exec_path.is_dir()
    return git_executable.resolve(), git_exec_path.resolve()


def _network_factory(
    tmp_path: Path,
    *,
    environment: Mapping[str, str] | None = None,
    ssh_executable: Path | None = None,
    git_exec_path: Path | None = None,
    allow_ssh_agent: bool = False,
) -> git_network.NetworkContextFactory:
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700, exist_ok=True)
    git_executable, installed_exec_path = _test_git_installation()
    return git_network.NetworkContextFactory(
        environment={} if environment is None else environment,
        temporary_parent=parent,
        git_executable=str(git_executable),
        git_exec_path=(
            installed_exec_path
            if git_exec_path is None
            else git_exec_path
        ),
        ssh_executable=(
            None if ssh_executable is None else str(ssh_executable)
        ),
        allow_ssh_agent=allow_ssh_agent,
    )


def _create_network_context(
    tmp_path: Path,
    *,
    endpoint: str = "https://push.example.test/team/notes.git",
    facts=(),
    environment: Mapping[str, str] | None = None,
    ssh_executable: Path | None = None,
    git_exec_path: Path | None = None,
    allow_ssh_agent: bool = False,
):
    repository = _network_repository(tmp_path)
    frozen_endpoint = _network_endpoint(endpoint)
    destination = frozen_endpoint.projection
    source_authorization, config_authorization = _network_authorizations(
        repository,
        destination,
        facts=facts,
        environment=environment,
    )
    factory = _network_factory(
        tmp_path,
        environment=environment,
        ssh_executable=ssh_executable,
        git_exec_path=git_exec_path,
        allow_ssh_agent=allow_ssh_agent,
    )
    context = factory.create(
        repository=repository,
        source_objects=source_authorization,
        configuration=config_authorization,
        destination=destination,
        endpoint=frozen_endpoint,
    )
    return repository, destination, context


def test_network_context_factory_rejects_explicit_empty_git_executable(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)

    with pytest.raises(git_network.NetworkContextError) as error:
        git_network.NetworkContextFactory(
            environment={"PATH": os.defpath},
            temporary_parent=parent,
            git_executable="",
            git_exec_path=_test_git_installation()[1],
        )

    assert error.value.code == "invalid_executable"


def test_network_context_factory_rejects_explicit_empty_ssh_executable(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint("git@push.example.test:team/notes.git")
    destination = endpoint.projection
    source, configuration = _network_authorizations(
        repository,
        destination,
        environment=isolated_ssh_environment,
    )
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)
    git_executable, git_exec_path = _test_git_installation()
    factory = git_network.NetworkContextFactory(
        environment=isolated_ssh_environment,
        temporary_parent=parent,
        git_executable=str(git_executable),
        git_exec_path=git_exec_path,
        ssh_executable="",
        allow_ssh_agent=True,
    )

    with pytest.raises(git_network.NetworkContextError) as error:
        factory.create(
            repository=repository,
            source_objects=source,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )

    assert error.value.code == "invalid_executable"


def test_network_context_builds_private_bare_layout_without_source_mutation(
    tmp_path: Path,
) -> None:
    repository = _network_repository(tmp_path)
    source_files = {
        relative: (Path(repository.git_dir) / relative).read_bytes()
        for relative in ("config", "index", "refs/heads/main")
    }
    destination = _network_destination()
    source_authorization, config_authorization = _network_authorizations(
        repository,
        destination,
    )
    context = _network_factory(tmp_path).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=config_authorization,
        destination=destination,
        endpoint=_network_endpoint(),
    )

    settings = context.command_settings()
    environment = settings.environment
    git_dir = Path(environment["GIT_DIR"])
    root = git_dir.parent

    assert settings.cwd == str(git_dir)
    assert settings.stdin is None
    assert settings.stdin_closed is True
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE(git_dir.stat().st_mode) == 0o700
    assert set(path.name for path in root.iterdir()) == {
        "global.gitconfig",
        "home",
        "repository.git",
        "tmp",
        "xdg-config",
    }
    assert set(path.name for path in git_dir.iterdir()) == {
        "HEAD",
        "config",
        "objects",
        "refs",
    }
    assert set(path.name for path in (git_dir / "objects").iterdir()) == {
        "info",
        "pack",
    }
    assert not any(
        (git_dir / name).exists()
        for name in ("hooks", "index", "remotes", "worktrees")
    )
    assert not any((git_dir / "refs").iterdir())
    assert (git_dir / "HEAD").read_text(encoding="ascii") == (
        "ref: refs/heads/chatbook-isolated\n"
    )
    private_config = (git_dir / "config").read_text(encoding="utf-8")
    assert "bare = true" in private_config
    assert "remote" not in private_config.lower()
    assert "push.example.test" not in private_config
    assert Path(environment["GIT_CONFIG_GLOBAL"]).read_bytes() == b""
    assert environment["GIT_OBJECT_DIRECTORY"] == str(git_dir / "objects")
    assert environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] == str(
        Path(repository.git_common_dir) / "objects"
    )
    assert Path(environment["HOME"]).parent == root
    assert Path(environment["XDG_CONFIG_HOME"]).parent == root
    assert Path(environment["TMPDIR"]).parent == root
    for path in root.rglob("*"):
        mode = stat.S_IMODE(path.lstat().st_mode)
        expected_mode = (
            0o500
            if path in {
                Path(environment["HOME"]),
                Path(environment["XDG_CONFIG_HOME"]),
                Path(environment["TMPDIR"]),
            }
            else (0o700 if path.is_dir() else 0o600)
        )
        assert mode == expected_mode
        if path.is_file():
            assert b"PRIVATE NOTE BODY" not in path.read_bytes()
    assert {
        relative: (Path(repository.git_dir) / relative).read_bytes()
        for relative in source_files
    } == source_files

    assert context.close() is True
    assert not root.exists()


def test_network_context_child_scratch_directories_reject_writes(
    tmp_path: Path,
) -> None:
    _repository, _destination, context = _create_network_context(tmp_path)
    settings = context.command_settings()

    scratch_names = (
        "HOME",
        "XDG_CONFIG_HOME",
        "TMP",
        "TEMP",
        "TMPDIR",
    )
    for name in scratch_names:
        directory = Path(settings.environment[name])
        assert stat.S_IMODE(directory.stat().st_mode) == 0o500
        result = subprocess.run(
            (
                str(Path(sys.executable).resolve()),
                "-I",
                "-c",
                (
                    "import os, sys; from pathlib import Path; "
                    "(Path(os.environ[sys.argv[1]]) / "
                    "('forbidden-' + sys.argv[1])).write_bytes(b'x')"
                ),
                name,
            ),
            cwd=settings.cwd,
            env=dict(settings.environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=5,
            check=False,
        )
        assert result.returncode != 0
        assert not (directory / f"forbidden-{name}").exists()

    git_result = subprocess.run(
        (
            str(_test_git_installation()[0]),
            f"--git-dir={settings.cwd}",
            "rev-parse",
            "--git-dir",
        ),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert git_result.returncode == 0
    assert context.close() is True


def test_network_environment_is_allowlist_with_chatbook_controls(
    tmp_path: Path,
) -> None:
    ambient = {
        "PATH": "/trusted/bin",
        "TMPDIR": "/ambient/tmp",
        "SSH_AUTH_SOCK": "/private/agent.sock",
        "GIT_DIR": "CANARY_GIT_DIR",
        "GIT_WORK_TREE": "CANARY_WORKTREE",
        "GIT_COMMON_DIR": "CANARY_COMMON",
        "GIT_INDEX_FILE": "CANARY_INDEX",
        "GIT_OBJECT_DIRECTORY": "CANARY_OBJECTS",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "CANARY_ALTERNATE",
        "GIT_CONFIG": "CANARY_CONFIG",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.sshCommand",
        "GIT_CONFIG_VALUE_0": "CANARY_COMMAND",
        "GIT_EXEC_PATH": "/ambient/git-exec",
        "GIT_NAMESPACE": "CANARY_NAMESPACE",
        "GIT_REPLACE_REF_BASE": "CANARY_REPLACE",
        "GIT_AUTHOR_NAME": "CANARY_AUTHOR",
        "GIT_AUTHOR_EMAIL": "author@example.test",
        "GIT_AUTHOR_DATE": "yesterday",
        "GIT_COMMITTER_NAME": "CANARY_COMMITTER",
        "GIT_COMMITTER_DATE": "tomorrow",
        "GIT_ASKPASS": "CANARY_ASKPASS",
        "SSH_ASKPASS": "CANARY_SSH_ASKPASS",
        "GIT_EDITOR": "CANARY_EDITOR",
        "GIT_PAGER": "CANARY_PAGER",
        "PAGER": "CANARY_PAGER",
        "GIT_TERMINAL_PROMPT": "1",
        "GIT_SSH": "CANARY_SSH",
        "GIT_SSH_COMMAND": "CANARY_SSH_COMMAND",
        "GIT_PROXY_COMMAND": "CANARY_PROXY_COMMAND",
        "HTTPS_PROXY": "https://proxy.example.test",
        "http_proxy": "http://proxy.example.test",
        "OPENAI_API_KEY": "CANARY_PROVIDER_TOKEN",
        "AWS_SECRET_ACCESS_KEY": "CANARY_CLOUD_TOKEN",
        "CHATBOOK_UNRELATED_STATE": "CANARY_APP_STATE",
        "HOME": "/ambient/home",
        "XDG_CONFIG_HOME": "/ambient/xdg",
    }
    _repository, _destination, context = _create_network_context(
        tmp_path,
        environment=ambient,
    )

    settings = context.command_settings()
    environment = settings.environment

    assert "/trusted/bin" not in environment["PATH"].split(os.pathsep)
    assert all(
        Path(component).is_absolute()
        for component in environment["PATH"].split(os.pathsep)
    )
    assert "SSH_AUTH_SOCK" not in environment
    assert environment["GIT_TERMINAL_PROMPT"] == "0"
    assert environment["GCM_INTERACTIVE"] == "Never"
    assert environment["GCM_GUI_PROMPT"] == "0"
    assert environment["SSH_ASKPASS_REQUIRE"] == "never"
    assert environment["GIT_PAGER"] == ""
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_NO_LAZY_FETCH"] == "1"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_CONFIG_NOSYSTEM"] == "1"
    assert environment["GIT_EXEC_PATH"] != ambient["GIT_EXEC_PATH"]
    assert Path(environment["GIT_EXEC_PATH"]) == _test_git_installation()[1]
    assert environment["LC_ALL"] == "C"
    assert environment["TMPDIR"] != ambient["TMPDIR"]
    assert environment["HOME"] != ambient["HOME"]
    assert environment["XDG_CONFIG_HOME"] != ambient["XDG_CONFIG_HOME"]
    rejected = {
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_CONFIG",
        "GIT_CONFIG_COUNT",
        "GIT_CONFIG_KEY_0",
        "GIT_CONFIG_VALUE_0",
        "GIT_NAMESPACE",
        "GIT_REPLACE_REF_BASE",
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_AUTHOR_DATE",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_DATE",
        "GIT_ASKPASS",
        "SSH_ASKPASS",
        "GIT_EDITOR",
        "PAGER",
        "GIT_SSH",
        "GIT_SSH_COMMAND",
        "GIT_PROXY_COMMAND",
        "HTTPS_PROXY",
        "http_proxy",
        "OPENAI_API_KEY",
        "AWS_SECRET_ACCESS_KEY",
        "CHATBOOK_UNRELATED_STATE",
    }
    assert rejected.isdisjoint(environment)
    assert all("CANARY" not in value for value in environment.values())
    assert settings.environment_fingerprint
    with pytest.raises(TypeError):
        environment["INJECTED"] = "value"  # type: ignore[index]
    assert context.close() is True


def test_windows_network_context_refuses_before_private_or_external_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    forbidden_calls: list[str] = []
    before = set(tmp_path.iterdir())

    def forbidden(name: str):
        def invoke(*_args, **_kwargs):
            forbidden_calls.append(name)
            raise AssertionError(f"Windows refusal reached {name}")

        return invoke

    monkeypatch.setattr(git_network.os, "name", "nt")
    monkeypatch.setattr(git_network.tempfile, "mkdtemp", forbidden("mkdtemp"))
    monkeypatch.setattr(git_network.shutil, "which", forbidden("which"))
    monkeypatch.setattr(
        git_network,
        "_pin_git_dispatch_executable",
        forbidden("git-dispatch"),
    )

    with pytest.raises(git_network.NetworkContextError) as error:
        git_network.NetworkContextFactory(
            environment={},
            temporary_parent=tmp_path,
            git_exec_path=tmp_path,
        )

    assert error.value.code == "unsupported_platform"
    assert forbidden_calls == []
    assert set(tmp_path.iterdir()) == before


def _short_agent_socket_path(tmp_path: Path) -> Path:
    suffix = f".s{os.getpid():x}{id(tmp_path) & 0xFFFF:x}"
    return Path.cwd() / suffix


def _bound_agent_socket(path: Path) -> socket.socket:
    agent = socket.socket(socket.AF_UNIX)
    try:
        agent.bind(str(path))
    except OSError as error:
        agent.close()
        pytest.skip(f"AF_UNIX fixture unavailable: {error.errno}")
    return agent


def test_network_environment_preserves_ssh_agent_only_when_authorized(
    tmp_path: Path,
) -> None:
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    try:
        ambient = {
            "PATH": "/trusted/bin",
            "SSH_AUTH_SOCK": str(agent_path),
        }
        _repository, _destination, context = _create_network_context(
            tmp_path,
            environment=ambient,
        )

        environment = context.command_settings().environment

        assert "SSH_AUTH_SOCK" not in environment
        assert context.close() is True
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


def test_network_environment_pins_authorized_owner_agent_socket(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "agent-pin-home"
    home.mkdir(mode=0o700)
    agent_path = _short_agent_socket_path(tmp_path)
    original = _bound_agent_socket(agent_path)
    try:
        _repository, _destination, context = _create_network_context(
            tmp_path,
            endpoint="ssh://git@push.example.test:22/team/notes.git",
            environment={
                "HOME": str(home.resolve()),
                "PATH": os.defpath,
                "SSH_AUTH_SOCK": str(agent_path.resolve()),
            },
            ssh_executable=_fake_ssh_executable(tmp_path),
            allow_ssh_agent=True,
        )
        assert context.command_settings().environment[
            "SSH_AUTH_SOCK"
        ] == str(agent_path.resolve())

        original.close()
        agent_path.unlink()
        replacement = _bound_agent_socket(agent_path)
        try:
            with pytest.raises(git_network.NetworkContextError):
                context.command_settings()
        finally:
            replacement.close()

        assert context.close() is True
    finally:
        original.close()
        agent_path.unlink(missing_ok=True)


def test_network_environment_rejects_non_socket_ssh_agent(
    tmp_path: Path,
) -> None:
    home = tmp_path / "invalid-agent-home"
    home.mkdir(mode=0o700)
    not_a_socket = tmp_path / "agent.sock"
    not_a_socket.write_text("not a socket\n", encoding="utf-8")

    with pytest.raises(git_network.NetworkContextError):
        _create_network_context(
            tmp_path,
            endpoint="ssh://git@push.example.test:22/team/notes.git",
            environment={
                "HOME": str(home.resolve()),
                "PATH": os.defpath,
                "SSH_AUTH_SOCK": str(not_a_socket),
            },
            ssh_executable=_fake_ssh_executable(tmp_path),
            allow_ssh_agent=True,
        )


def test_https_network_context_ignores_invalid_ssh_agent(
    tmp_path: Path,
) -> None:
    missing_agent = tmp_path / "missing-agent.sock"

    _repository, _destination, context = _create_network_context(
        tmp_path,
        environment={"SSH_AUTH_SOCK": str(missing_agent)},
        allow_ssh_agent=True,
    )

    assert "SSH_AUTH_SOCK" not in context.command_settings().environment
    assert context.close() is True


@pytest.mark.parametrize(
    "socket_path",
    [
        "/private/agent-%h.sock",
        "/private/agent-${LC_ALL}.sock",
        "/private/agent socket",
    ],
)
def test_network_environment_rejects_openssh_agent_token_expansion(
    socket_path: str,
) -> None:
    assert not git_network._safe_environment_value(
        "SSH_AUTH_SOCK",
        socket_path,
    )


def test_network_environment_rejects_agent_identity_substitution_at_command_seam(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "virtual-agent-home"
    home.mkdir(mode=0o700)
    socket_path = Path.cwd() / ".chatbook-virtual-agent"
    identity = 1001
    original_stat = Path.stat
    original_resolve = Path.resolve

    def virtual_stat(path: Path, *args, **kwargs):
        if path == socket_path:
            return os.stat_result(
                (
                    stat.S_IFSOCK | 0o600,
                    identity,
                    16777231,
                    1,
                    os.geteuid(),
                    os.getegid(),
                    0,
                    0,
                    0,
                    0,
                )
            )
        return original_stat(path, *args, **kwargs)

    def virtual_resolve(path: Path, *, strict: bool = False) -> Path:
        if path == socket_path:
            return socket_path
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "stat", virtual_stat)
    monkeypatch.setattr(Path, "resolve", virtual_resolve)
    _repository, _destination, context = _create_network_context(
        tmp_path,
        endpoint="ssh://git@push.example.test:22/team/notes.git",
        environment={
            "HOME": str(home.resolve()),
            "PATH": os.defpath,
            "SSH_AUTH_SOCK": str(socket_path),
        },
        ssh_executable=_fake_ssh_executable(tmp_path),
        allow_ssh_agent=True,
    )
    assert context.command_settings().environment["SSH_AUTH_SOCK"] == str(
        socket_path
    )

    identity += 1

    with pytest.raises(git_network.NetworkContextError):
        context.command_settings()
    assert context.close() is True


def test_https_network_context_drops_authorized_ssh_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    socket_path = Path.cwd() / ".chatbook-virtual-https-agent"
    original_stat = Path.stat
    original_resolve = Path.resolve

    def virtual_stat(path: Path, *args, **kwargs):
        if path == socket_path:
            return os.stat_result(
                (
                    stat.S_IFSOCK | 0o600,
                    2001,
                    16777231,
                    1,
                    os.geteuid(),
                    os.getegid(),
                    0,
                    0,
                    0,
                    0,
                )
            )
        return original_stat(path, *args, **kwargs)

    def virtual_resolve(path: Path, *, strict: bool = False) -> Path:
        if path == socket_path:
            return socket_path
        return original_resolve(path, strict=strict)

    monkeypatch.setattr(Path, "stat", virtual_stat)
    monkeypatch.setattr(Path, "resolve", virtual_resolve)
    _repository, _destination, context = _create_network_context(
        tmp_path,
        environment={"SSH_AUTH_SOCK": str(socket_path)},
        allow_ssh_agent=True,
    )

    assert "SSH_AUTH_SOCK" not in context.command_settings().environment
    assert context.close() is True


@pytest.mark.parametrize("search_path", [".:/usr/bin", ":/usr/bin", "bin"])
def test_network_environment_rejects_relative_or_empty_search_path(
    tmp_path: Path,
    search_path: str,
) -> None:
    git_executable = shutil.which("git")
    assert git_executable is not None

    with pytest.raises(git_network.NetworkContextError):
        git_network.NetworkContextFactory(
            environment={"PATH": search_path},
            temporary_parent=tmp_path,
            git_executable=git_executable,
            git_exec_path=_test_git_installation()[1],
        )


def test_network_context_config_copy_preserves_order_and_origin_binding(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    facts = (
        _fact("credential.helper", "", scope="system", origin="1" * 64),
        _fact(
            "credential.helper",
            helper_name,
            scope="global",
            origin="2" * 64,
        ),
        _fact(
            "credential.useHttpPath",
            "true",
            scope="global",
            origin="2" * 64,
        ),
    )
    repository = _network_repository(tmp_path)
    destination = _network_destination()
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        facts=facts,
    )
    changed_origin = git_network._authorize_network_config_facts(
        tuple(
            replace(fact, origin_identity="3" * 64)
            if fact.value == helper_name
            else fact
            for fact in facts
        ),
        configuration_fingerprint="f" * 64,
        destination=destination,
    )
    helper_bin = tmp_path / "helper-bin"
    helper_bin.mkdir(mode=0o700)
    helper = helper_bin / f"git-credential-{helper_name}"
    helper.write_text("not executed\n", encoding="utf-8")
    helper.chmod(0o700)
    context = _network_factory(
        tmp_path,
        environment={"PATH": str(helper_bin)},
    ).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=configuration,
        destination=destination,
        endpoint=_network_endpoint(),
    )

    settings = context.command_settings()
    private_config = Path(settings.environment["GIT_DIR"], "config").read_text(
        encoding="utf-8"
    )

    assert private_config.index("\thelper =\n") < private_config.index(
        f"\thelper = {helper_name}\n"
    )
    assert "\tuseHttpPath = true\n" in private_config
    assert configuration.copy_fingerprint != changed_origin.copy_fingerprint
    assert context.config_copy_fingerprint == configuration.copy_fingerprint
    assert helper_name not in repr(configuration)
    assert repository.git_dir not in repr(context)
    assert repository.git_dir not in repr(settings)
    assert context.close() is True


def test_network_context_rejects_unregistered_config_capability(
    tmp_path: Path,
) -> None:
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint()
    destination = endpoint.projection
    source_authorization, _configuration = _network_authorizations(
        repository,
        destination,
    )
    forged = object.__new__(git_network.NetworkConfigAuthorization)

    with pytest.raises(git_network.NetworkContextError):
        _network_factory(tmp_path).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=forged,
            destination=destination,
            endpoint=endpoint,
        )


def test_network_context_rejects_unregistered_source_object_capability(
    tmp_path: Path,
) -> None:
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint()
    destination = endpoint.projection
    forged = object.__new__(git_network.SourceObjectDirectoryAuthorization)
    _source_authorization, configuration = _network_authorizations(
        repository,
        destination,
    )

    with pytest.raises(git_network.NetworkContextError):
        _network_factory(tmp_path).create(
            repository=repository,
            source_objects=forged,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )


@pytest.mark.parametrize(
    ("key", "value", "scope"),
    [
        ("url.https://safe.example/.insteadOf", "https://old/", "global"),
        ("remote.origin.url", "https://safe.example/repo.git", "global"),
        ("remote.origin.push", "refs/heads/*:refs/heads/*", "global"),
        ("remote.origin.mirror", "true", "global"),
        ("remote.origin.receivePack", "custom", "system"),
        ("push.pushOption", "ci.skip", "global"),
        ("http.extraHeader", "Authorization: secret", "global"),
        ("http.sslVerify", "false", "system"),
        ("http.proxy", "https://proxy.example.test", "global"),
        ("core.sshCommand", "/tmp/custom-ssh", "system"),
        ("ssh.variant", "plink", "global"),
        ("credential.username", "private-user", "global"),
        ("credential.helper", "osxkeychain", "local"),
        ("credential.helper", "manager", "worktree"),
        ("credential.helper", "!touch SHOULD_NOT_RUN", "global"),
        ("credential.helper", "manager --file secret", "global"),
        ("credential.helper", "/tmp/custom-helper", "system"),
        (
            "credential.https://user:secret@safe.example.helper",
            "manager",
            "global",
        ),
        ("credential.useHttpPath", "sometimes", "global"),
    ],
    ids=[
        "url-rewrite",
        "remote-url",
        "remote-refspec",
        "mirror",
        "receive-pack",
        "push-option",
        "extra-header",
        "tls-exception",
        "proxy",
        "ssh-command",
        "ssh-variant",
        "credential-value",
        "local-helper",
        "worktree-helper",
        "shell-helper",
        "helper-arguments",
        "helper-path",
        "credential-bearing-scope",
        "unvalidated-boolean",
    ],
)
def test_network_context_config_copy_rejects_unapproved_facts(
    key: str,
    value: str,
    scope: str,
) -> None:
    destination = _network_destination()

    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_network_config_facts(
            (_fact(key, value, scope=scope),),
            configuration_fingerprint="f" * 64,
            destination=destination,
        )


def test_network_context_config_copy_is_https_only() -> None:
    destination = _network_destination(
        "ssh://git@push.example.test:22/team/notes.git"
    )

    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_network_config_facts(
            (_fact("credential.helper", "manager", scope="global"),),
            configuration_fingerprint="f" * 64,
            destination=destination,
        )


def test_ssh_config_snapshot_omits_transport_irrelevant_credential_facts(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    repository = _network_repository(tmp_path)
    destination = _network_destination(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    fingerprint = "f" * 64
    snapshot = git_network._authorize_network_config_snapshot(
        (
            _fact(
                "credential.helper",
                "osxkeychain",
                scope="system",
            ),
            _fact(
                "credential.useHttpPath",
                "true",
                scope="global",
            ),
        ),
        configuration_fingerprint=fingerprint,
        destination=destination,
        environment=isolated_ssh_environment,
        repository=repository,
    )
    empty_copy = git_network._authorize_network_config_snapshot(
        (),
        configuration_fingerprint=fingerprint,
        destination=destination,
        environment=isolated_ssh_environment,
        repository=repository,
    )

    assert snapshot.configuration_fingerprint == fingerprint
    assert snapshot.copy_fingerprint == empty_copy.copy_fingerprint


def test_network_context_config_snapshot_rejects_scoped_use_http_path() -> None:
    destination = _network_destination()

    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_network_config_snapshot(
            (
                _fact(
                    "credential.https://push.example.test/team/notes.git."
                    "useHttpPath",
                    "true",
                    scope="global",
                ),
            ),
            configuration_fingerprint="f" * 64,
            destination=destination,
        )


def test_network_context_config_rejects_unapproved_named_helper() -> None:
    destination = _network_destination()

    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_network_config_facts(
            (
                _fact(
                    "credential.helper",
                    "chatbook-unapproved-helper",
                    scope="global",
                ),
            ),
            configuration_fingerprint="f" * 64,
            destination=destination,
        )


def test_network_context_rejects_source_alternate_path_separator(
    tmp_path: Path,
) -> None:
    source_objects = tmp_path / "source:alternate" / "objects"
    source_objects.mkdir(parents=True)

    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_source_object_directory(
            source_objects,
            _filesystem_identity(source_objects),
            "sha1",
        )


def test_source_object_authorization_fingerprint_binds_object_format(
    tmp_path: Path,
) -> None:
    source_objects = tmp_path / "source" / ".git" / "objects"
    source_objects.mkdir(parents=True)
    identity = _filesystem_identity(source_objects)

    sha1 = git_network._authorize_source_object_directory(
        source_objects,
        identity,
        "sha1",
    )
    sha256 = git_network._authorize_source_object_directory(
        source_objects,
        identity,
        "sha256",
    )

    assert sha1.object_format == "sha1"
    assert sha256.object_format == "sha256"
    assert sha1.identity_fingerprint != sha256.identity_fingerprint
    with pytest.raises(git_network.NetworkContextError):
        git_network._authorize_source_object_directory(
            source_objects,
            identity,
            "sha512",  # type: ignore[arg-type]
        )


def _fake_ssh_executable(tmp_path: Path) -> Path:
    executable = tmp_path / "trusted-ssh"
    executable.write_text("not executed\n", encoding="utf-8")
    executable.chmod(0o700)
    return executable


def _recording_ssh_executable(tmp_path: Path) -> tuple[Path, Path]:
    log_path = tmp_path / "ssh-argv.json"
    executable = tmp_path / "recording-ssh"
    executable.write_text(
        (
            f"#!{Path(sys.executable).resolve()} -I\n"
            "import json\n"
            "from pathlib import Path\n"
            "import sys\n"
            f"Path({str(log_path)!r}).write_text(\n"
            "    json.dumps(sys.argv[1:]), encoding='utf-8'\n"
            ")\n"
            "raise SystemExit(73)\n"
        ),
        encoding="utf-8",
    )
    executable.chmod(0o700)
    return executable, log_path


def test_ssh_host_trust_snapshot_materializes_private_file_and_exact_openssh_invocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping any frozen trust or agent pin must make SSH context use fail."""
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint(
        "ssh://git@push.example.test:2222/team/notes.git"
    )
    destination = endpoint.projection
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "isolated-home"
    ssh_directory = home / ".ssh"
    ssh_directory.mkdir(parents=True, mode=0o700)
    (ssh_directory / "known_hosts").write_bytes(b"first-host-key")
    (ssh_directory / "known_hosts").chmod(0o600)
    (ssh_directory / "known_hosts2").write_bytes(b"second-host-key\n")
    (ssh_directory / "known_hosts2").chmod(0o600)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    executable = _fake_ssh_executable(tmp_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path),
    }
    source_objects = Path(repository.git_common_dir) / "objects"
    source_authorization = git_network._authorize_source_object_directory(
        source_objects,
        _filesystem_identity(source_objects),
        "sha1",
    )
    try:
        configuration = git_network._authorize_network_config_snapshot(
            (),
            configuration_fingerprint="f" * 64,
            destination=destination,
            environment=environment,
            repository=repository,
        )
        context = _network_factory(
            tmp_path,
            environment=environment,
            ssh_executable=executable,
            allow_ssh_agent=True,
        ).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )
        invocation = context.openssh_invocation()
        assert invocation is not None
        trust_argument = next(
            argument
            for argument in invocation.argv
            if argument.startswith("UserKnownHostsFile=")
        )
        private_trust = Path(trust_argument.partition("=")[2])

        assert private_trust.read_bytes() == (
            b"first-host-key\nsecond-host-key\n"
        )
        assert stat.S_IMODE(private_trust.stat().st_mode) == 0o400
        assert private_trust.stat().st_nlink == 1
        assert invocation.argv == (
            str(executable.resolve()),
            *git_network._OPENSSH_FIXED_ARGUMENTS,
            "-o",
            f"UserKnownHostsFile={private_trust}",
            "-o",
            "GlobalKnownHostsFile=none",
            "-o",
            "IdentityFile=none",
            "-o",
            "IdentitiesOnly=no",
            "-o",
            f"IdentityAgent={agent_path.resolve()}",
            "-o",
            "HostName=push.example.test",
            "-p",
            "2222",
            "-l",
            "git",
            "--",
            "push.example.test",
        )
        assert context.close() is True
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


def _isolated_system_ssh_trust_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Path, Path]:
    paths = (
        tmp_path / "system-ssh" / "ssh_known_hosts",
        tmp_path / "system-ssh" / "ssh_known_hosts2",
    )
    monkeypatch.setattr(
        git_network,
        "_SYSTEM_SSH_TRUST_PATHS",
        paths,
        raising=False,
    )
    return paths


@contextmanager
def _isolated_ssh_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, str]]:
    """Retain one real agent and isolated standard trust set for a test."""
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "isolated-ssh-home"
    home.mkdir(mode=0o700, exist_ok=True)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    try:
        yield {
            "HOME": str(home.resolve()),
            "PATH": os.defpath,
            "SSH_AUTH_SOCK": str(agent_path.resolve()),
        }
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.fixture
def isolated_ssh_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[dict[str, str]]:
    """Provide one real, cleanup-bound SSH authority for compatibility tests."""
    with _isolated_ssh_environment(tmp_path, monkeypatch) as environment:
        yield environment


def _ssh_network_authorization(
    repository: RepositoryIdentity,
    destination,
    environment: Mapping[str, str],
):
    return git_network._authorize_network_config_snapshot(
        (),
        configuration_fingerprint="f" * 64,
        destination=destination,
        environment=environment,
        repository=repository,
    )


def _private_host_trust_path(
    context: git_network.NetworkGitExecutionContext,
) -> Path:
    invocation = context.openssh_invocation()
    assert invocation is not None
    argument = next(
        value
        for value in invocation.argv
        if value.startswith("UserKnownHostsFile=")
    )
    return Path(argument.partition("=")[2])


def test_ssh_host_trust_missing_sources_make_empty_snapshot_and_distinct_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treating missing trust as an omitted fact would permit policy ABA."""
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint("git@push.example.test:team/notes.git")
    destination = endpoint.projection
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "empty-home"
    home.mkdir(mode=0o700)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path.resolve()),
    }
    source_objects = Path(repository.git_common_dir) / "objects"
    source_authorization = git_network._authorize_source_object_directory(
        source_objects,
        _filesystem_identity(source_objects),
        "sha1",
    )
    try:
        missing = _ssh_network_authorization(
            repository,
            destination,
            environment,
        )
        ssh_directory = home / ".ssh"
        ssh_directory.mkdir(mode=0o700)
        known_hosts = ssh_directory / "known_hosts"
        known_hosts.write_bytes(b"host-key\n")
        known_hosts.chmod(0o600)
        present = _ssh_network_authorization(
            repository,
            destination,
            environment,
        )
        context = _network_factory(
            tmp_path,
            environment=environment,
            ssh_executable=_fake_ssh_executable(tmp_path),
            allow_ssh_agent=True,
        ).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=missing,
            destination=destination,
            endpoint=endpoint,
        )

        private_trust = _private_host_trust_path(context)
        assert private_trust.read_bytes() == b""
        assert stat.S_IMODE(private_trust.stat().st_mode) == 0o400
        assert missing.copy_fingerprint != present.copy_fingerprint
        assert context.close() is True
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


def test_ssh_host_trust_combined_limit_counts_inserted_newlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The private payload, including separators, must remain at most 4 MiB."""
    repository = _network_repository(tmp_path)
    destination = _network_destination(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    system_paths = _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "full-trust-home"
    user_paths = (
        home / ".ssh" / "known_hosts",
        home / ".ssh" / "known_hosts2",
    )
    for source in (*user_paths, *system_paths):
        source.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
        source.write_bytes(b"x" * (1 << 20))
        source.chmod(0o600)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    try:
        with pytest.raises(git_network.NetworkContextError) as error:
            _ssh_network_authorization(
                repository,
                destination,
                {
                    "HOME": str(home.resolve()),
                    "PATH": os.defpath,
                    "SSH_AUTH_SOCK": str(agent_path.resolve()),
                },
            )

        assert error.value.code == "unsafe_filesystem"
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "unsafe_kind",
    ["symlink", "hardlink", "group_write", "unreadable", "unstable", "oversize"],
)
def test_ssh_host_trust_rejects_unsafe_present_source_before_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    unsafe_kind: str,
) -> None:
    """Accepting an unsafe present source would turn local proof into live trust."""
    repository = _network_repository(tmp_path)
    destination = _network_destination(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "unsafe-home"
    ssh_directory = home / ".ssh"
    ssh_directory.mkdir(parents=True, mode=0o700)
    source = ssh_directory / "known_hosts"
    source.write_bytes(b"host-key\n")
    source.chmod(0o600)
    if unsafe_kind == "symlink":
        target = ssh_directory / "target"
        source.rename(target)
        source.symlink_to(target)
    elif unsafe_kind == "hardlink":
        os.link(source, ssh_directory / "second-link")
    elif unsafe_kind == "group_write":
        source.chmod(0o620)
    elif unsafe_kind == "unreadable":
        source.chmod(0o000)
    elif unsafe_kind == "oversize":
        source.write_bytes(b"x" * ((1 << 20) + 1))
    else:
        original_fstat = git_network.os.fstat
        identity = _filesystem_identity(source)
        matching_calls = 0

        def unstable_fstat(descriptor: int):
            nonlocal matching_calls
            metadata = original_fstat(descriptor)
            if (
                metadata.st_dev == identity.device
                and metadata.st_ino == identity.inode
            ):
                matching_calls += 1
                if matching_calls > 1:
                    values = list(metadata)
                    values[8] += 1
                    return os.stat_result(values)
            return metadata

        monkeypatch.setattr(git_network.os, "fstat", unstable_fstat)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    try:
        with pytest.raises(git_network.NetworkContextError) as error:
            _ssh_network_authorization(
                repository,
                destination,
                {
                    "HOME": str(home.resolve()),
                    "PATH": os.defpath,
                    "SSH_AUTH_SOCK": str(agent_path.resolve()),
                },
            )

        assert error.value.code in {"unsafe_filesystem", "invalid_environment"}
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.mark.parametrize("agent_kind", ["missing", "regular_file"])
def test_ssh_host_trust_requires_safe_existing_agent_during_local_proof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    agent_kind: str,
) -> None:
    """Deferring missing-agent refusal to OpenSSH would cross the network boundary."""
    repository = _network_repository(tmp_path)
    destination = _network_destination(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "agentless-home"
    home.mkdir(mode=0o700)
    agent_path = tmp_path / "missing-agent.sock"
    if agent_kind == "regular_file":
        agent_path.write_bytes(b"not an agent")

    with pytest.raises(git_network.NetworkContextError) as error:
        _ssh_network_authorization(
            repository,
            destination,
            {
                "HOME": str(home.resolve()),
                "PATH": os.defpath,
                "SSH_AUTH_SOCK": str(agent_path.resolve()),
            },
        )

    assert error.value.code == "invalid_environment"


def test_https_authorization_does_not_read_ssh_host_trust_or_require_agent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Applying SSH-only local reads to HTTPS would reject an unrelated transport."""
    repository = _network_repository(tmp_path)
    destination = _network_destination()
    unsafe_home = Path(repository.worktree_root) / "unsafe-home"
    ssh_directory = unsafe_home / ".ssh"
    ssh_directory.mkdir(parents=True)
    target = ssh_directory / "target"
    target.write_bytes(b"must-not-be-read")
    (ssh_directory / "known_hosts").symlink_to(target)
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)

    authorization = git_network._authorize_network_config_snapshot(
        (),
        configuration_fingerprint="f" * 64,
        destination=destination,
        environment={
            "HOME": str(unsafe_home),
            "SSH_AUTH_SOCK": str(tmp_path / "missing-agent.sock"),
        },
        repository=repository,
    )

    assert authorization.configuration_fingerprint == "f" * 64


@pytest.mark.parametrize("tamper", ["content", "mode", "substitution"])
def test_private_host_trust_tamper_invalidates_context_and_can_be_cleaned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tamper: str,
) -> None:
    """Failing to pin private trust would let same-session drift reach OpenSSH."""
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    destination = endpoint.projection
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "trust-home"
    ssh_directory = home / ".ssh"
    ssh_directory.mkdir(parents=True, mode=0o700)
    source = ssh_directory / "known_hosts"
    source.write_bytes(b"host-key\n")
    source.chmod(0o600)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path.resolve()),
    }
    source_objects = Path(repository.git_common_dir) / "objects"
    source_authorization = git_network._authorize_source_object_directory(
        source_objects,
        _filesystem_identity(source_objects),
        "sha1",
    )
    try:
        context = _network_factory(
            tmp_path,
            environment=environment,
            ssh_executable=_fake_ssh_executable(tmp_path),
            allow_ssh_agent=True,
        ).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=_ssh_network_authorization(
                repository,
                destination,
                environment,
            ),
            destination=destination,
            endpoint=endpoint,
        )
        private_trust = _private_host_trust_path(context)
        original = private_trust.read_bytes()
        displaced = private_trust.with_suffix(".displaced")
        if tamper == "content":
            private_trust.chmod(0o600)
            private_trust.write_bytes(b"replacement-key\n")
            private_trust.chmod(0o400)
        elif tamper == "mode":
            private_trust.chmod(0o600)
        else:
            private_trust.rename(displaced)
            private_trust.write_bytes(original)
            private_trust.chmod(0o400)

        with pytest.raises(git_network.NetworkContextError):
            context.command_settings()
        assert context.close() is False

        if tamper == "substitution":
            private_trust.unlink()
            displaced.rename(private_trust)
        else:
            private_trust.chmod(0o600)
            private_trust.write_bytes(original)
            private_trust.chmod(0o400)
        assert context.close() is True
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


def _recording_git_dispatch_executable(
    path: Path,
    log_path: Path,
    *,
    credential: bool = False,
) -> None:
    path.write_text(
        (
            f"#!{Path(sys.executable).resolve()} -I\n"
            "import json\n"
            "import os\n"
            "from pathlib import Path\n"
            "import sys\n"
            "scratch_writes = {}\n"
            "for name in ('HOME', 'XDG_CONFIG_HOME', 'TMPDIR'):\n"
            "    scratch = Path(os.environ[name]) / ('helper-' + name)\n"
            "    try:\n"
            "        descriptor = os.open(\n"
            "            scratch, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600\n"
            "        )\n"
            "    except OSError:\n"
            "        scratch_writes[name] = False\n"
            "    else:\n"
            "        os.close(descriptor)\n"
            "        scratch.unlink()\n"
            "        scratch_writes[name] = True\n"
            f"Path({str(log_path)!r}).write_text(\n"
            "    json.dumps({\n"
            "        'argv': sys.argv,\n"
            "        'environment': dict(os.environ),\n"
            "        'scratch_writes': scratch_writes,\n"
            "    }),\n"
            "    encoding='utf-8',\n"
            ")\n"
            + (
                "if sys.argv[1:] == ['get']:\n"
                "    print('username=pinned-user')\n"
                "    print('password=pinned-password')\n"
                if credential
                else "raise SystemExit(73)\n"
            )
        ),
        encoding="utf-8",
    )
    path.chmod(0o700)


def _write_loose_blob(objects: Path, payload: bytes) -> str:
    object_payload = b"blob " + str(len(payload)).encode() + b"\0" + payload
    object_id = hashlib.sha1(object_payload).hexdigest()
    destination = objects / object_id[:2] / object_id[2:]
    destination.parent.mkdir()
    destination.write_bytes(zlib.compress(object_payload))
    return object_id


def test_openssh_invocation_is_exact_literal_direct_argv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executable = _fake_ssh_executable(tmp_path)
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint(
        "ssh://git@[2001:db8::1]:2222/team/notes.git"
    )
    destination = endpoint.projection
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "isolated-home"
    home.mkdir(mode=0o700)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path.resolve()),
    }
    source_objects = Path(repository.git_common_dir) / "objects"
    source_authorization = git_network._authorize_source_object_directory(
        source_objects,
        _filesystem_identity(source_objects),
        "sha1",
    )
    try:
        context = _network_factory(
            tmp_path,
            environment=environment,
            ssh_executable=executable,
            allow_ssh_agent=True,
        ).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=_ssh_network_authorization(
                repository,
                destination,
                environment,
            ),
            destination=destination,
            endpoint=endpoint,
        )
        invocation = context.openssh_invocation()
        assert invocation is not None
        private_trust = _private_host_trust_path(context)
        assert invocation.argv == (
            str(executable.resolve()),
            *git_network._OPENSSH_FIXED_ARGUMENTS,
            "-o",
            f"UserKnownHostsFile={private_trust}",
            "-o",
            "GlobalKnownHostsFile=none",
            "-o",
            "IdentityFile=none",
            "-o",
            "IdentitiesOnly=no",
            "-o",
            f"IdentityAgent={agent_path.resolve()}",
            "-o",
            "HostName=2001:db8::1",
            "-p",
            "2222",
            "-l",
            "git",
            "--",
            "2001:db8::1",
        )
        assert not any("AskPass" in argument for argument in invocation.argv)
        assert "GIT_SSH_COMMAND" not in context.command_settings().environment
        assert destination.host == "2001:db8::1"
        assert repository.git_dir not in repr(invocation)
        with pytest.raises(FrozenInstanceError):
            invocation.argv = ()  # type: ignore[misc]
        assert context.close() is True
    finally:
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.mark.parametrize(
    "unsafe_parent_name",
    ["network-%h-contexts", "network space contexts"],
)
def test_openssh_invocation_rejects_tokenized_private_host_trust_path(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
    unsafe_parent_name: str,
) -> None:
    """OpenSSH token expansion must not redirect the private trust snapshot."""
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint(
        "ssh://git@push.example.test:22/team/notes.git"
    )
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        environment=isolated_ssh_environment,
    )
    unsafe_parent = tmp_path / unsafe_parent_name
    unsafe_parent.mkdir(mode=0o700)
    git_executable, git_exec_path = _test_git_installation()
    factory = git_network.NetworkContextFactory(
        environment=isolated_ssh_environment,
        temporary_parent=unsafe_parent,
        git_executable=str(git_executable),
        git_exec_path=git_exec_path,
        ssh_executable=str(_fake_ssh_executable(tmp_path)),
        allow_ssh_agent=True,
    )

    with pytest.raises(git_network.NetworkContextError) as error:
        factory.create(
            repository=repository,
            source_objects=source_authorization,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )

    assert error.value.code == "invalid_openssh"
    assert tuple(unsafe_parent.iterdir()) == ()


def test_openssh_git_adapter_executes_only_exact_frozen_route_without_network(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    endpoint_value = "ssh://git@[2001:db8::1]:2222/team/notes.git"
    executable, log_path = _recording_ssh_executable(tmp_path)
    _repository, _destination, context = _create_network_context(
        tmp_path,
        endpoint=endpoint_value,
        environment=isolated_ssh_environment,
        ssh_executable=executable,
        allow_ssh_agent=True,
    )
    endpoint = push_contracts._freeze_push_endpoint(endpoint_value, BRANCH_REF)
    settings = context.command_settings()
    invocation = context.openssh_invocation()

    assert invocation is not None
    assert "GIT_SSH_COMMAND" not in settings.environment
    assert settings.environment["GIT_SSH_VARIANT"] == "ssh"
    adapter = Path(settings.environment["GIT_SSH"])
    assert adapter.is_file()
    assert stat.S_IMODE(adapter.stat().st_mode) == 0o700

    result = subprocess.run(
        context.build_query_argv(endpoint),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert tuple(json.loads(log_path.read_text(encoding="utf-8"))) == (
        *invocation.argv[1:],
        "git-upload-pack /team/notes.git",
    )

    log_path.unlink()
    rejected = subprocess.run(
        (
            str(adapter),
            "-p",
            "2222",
            "git@2001:db8::1",
            "git-upload-pack '/other.git'",
        ),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert rejected.returncode == 126
    assert not log_path.exists()

    tampered_environment = dict(settings.environment)
    tampered_environment["CHATBOOK_NETWORK_SSH_PATH"] = "/other.git"
    tampered = subprocess.run(
        (
            str(adapter),
            "-p",
            "2222",
            "git@2001:db8::1",
            "git-upload-pack '/team/notes.git'",
        ),
        cwd=settings.cwd,
        env=tampered_environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert tampered.returncode == 126
    assert not log_path.exists()

    canonicalized = subprocess.run(
        (
            str(adapter),
            "-p",
            "2222",
            "git@2001:db8::1",
            'git-upload-pack "/team/notes.git"',
        ),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert canonicalized.returncode == 73
    assert tuple(json.loads(log_path.read_text(encoding="utf-8"))) == (
        *invocation.argv[1:],
        "git-upload-pack /team/notes.git",
    )
    assert context.close() is True


def test_openssh_git_adapter_accepts_actual_scp_push_receive_pack(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    endpoint_value = "git@push.example.test:team/notes.git"
    repository = _network_repository(tmp_path)
    candidate_oid = _write_loose_blob(
        Path(repository.git_common_dir) / "objects",
        b"guarded candidate",
    )
    endpoint = _network_endpoint(endpoint_value)
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        environment=isolated_ssh_environment,
    )
    executable, log_path = _recording_ssh_executable(tmp_path)
    context = _network_factory(
        tmp_path,
        environment=isolated_ssh_environment,
        ssh_executable=executable,
        allow_ssh_agent=True,
    ).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=configuration,
        destination=destination,
        endpoint=endpoint,
    )
    settings = context.command_settings()
    invocation = context.openssh_invocation()
    assert invocation is not None

    result = subprocess.run(
        context.build_push_argv(
            endpoint,
            "a" * 40,
            candidate_oid,
        ),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert result.returncode != 0
    assert tuple(json.loads(log_path.read_text(encoding="utf-8"))) == (
        *invocation.argv[1:],
        "git-receive-pack team/notes.git",
    )
    assert context.close() is True


def test_real_sha256_network_context_uses_alternate_for_query_and_push(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(
        tmp_path,
        object_format="sha256",
    )
    _owner, _binding, repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    endpoint = _network_endpoint(
        "git@push.example.test:team/notes.git"
    )
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        object_format="sha256",
        environment=isolated_ssh_environment,
    )
    executable, log_path = _recording_ssh_executable(tmp_path)
    context = _network_factory(
        tmp_path,
        environment=isolated_ssh_environment,
        ssh_executable=executable,
        allow_ssh_agent=True,
    ).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=configuration,
        destination=destination,
        endpoint=endpoint,
    )
    settings = context.command_settings()
    invocation = context.openssh_invocation()
    assert invocation is not None
    private_config = Path(settings.environment["GIT_DIR"]) / "config"
    object_query = (
        str(_test_git_installation()[0]),
        f"--git-dir={settings.environment['GIT_DIR']}",
        "--no-replace-objects",
        "cat-file",
        "-e",
        f"{candidate_oid}^{{commit}}",
    )
    missing_alternate_environment = dict(settings.environment)
    missing_alternate_environment["GIT_ALTERNATE_OBJECT_DIRECTORIES"] = str(
        tmp_path / "missing-objects"
    )

    missing_alternate = subprocess.run(
        object_query,
        cwd=settings.cwd,
        env=missing_alternate_environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    resolved_from_authorized_alternate = subprocess.run(
        object_query,
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert missing_alternate.returncode != 0
    assert resolved_from_authorized_alternate.returncode == 0
    assert not log_path.exists()

    query = subprocess.run(
        context.build_query_argv(endpoint),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert query.returncode != 0
    assert tuple(json.loads(log_path.read_text(encoding="utf-8"))) == (
        *invocation.argv[1:],
        "git-upload-pack team/notes.git",
    )
    log_path.unlink()

    push = subprocess.run(
        context.build_push_argv(endpoint, parent_oid, candidate_oid),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )

    assert push.returncode != 0
    assert tuple(json.loads(log_path.read_text(encoding="utf-8"))) == (
        *invocation.argv[1:],
        "git-receive-pack team/notes.git",
    )
    assert private_config.read_text(encoding="utf-8") == (
        "[core]\n"
        "\trepositoryFormatVersion = 1\n"
        "\tfileMode = true\n"
        "\tbare = true\n"
        "\tlogAllRefUpdates = false\n"
        "[extensions]\n"
        "\tobjectFormat = sha256\n"
    )
    assert len(parent_oid) == len(candidate_oid) == 64
    assert context.close() is True


def test_network_context_crash_left_files_exclude_transient_routing(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    agent_path = Path(isolated_ssh_environment["SSH_AUTH_SOCK"])
    endpoint_value = (
        "private-user-canary@private-host-canary.example.test:"
        "private/repository-route-canary.git"
    )
    repository, _destination, context = _create_network_context(
        tmp_path,
        endpoint=endpoint_value,
        environment={
            **isolated_ssh_environment,
            "CHATBOOK_UNRELATED_STATE": "PRIVATE_ENV_CANARY",
        },
        ssh_executable=_fake_ssh_executable(tmp_path),
        allow_ssh_agent=True,
    )
    settings = context.command_settings()
    root = Path(settings.cwd).parent
    forbidden = (
        endpoint_value,
        "private-user-canary",
        "private-host-canary.example.test",
        "private/repository-route-canary.git",
        str(agent_path),
        repository.worktree_root,
        repository.git_dir,
        repository.git_common_dir,
        str(Path(repository.git_common_dir) / "objects"),
        "PRIVATE_ENV_CANARY",
    )

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        payload = path.read_bytes()
        assert all(value.encode() not in payload for value in forbidden)

    assert context.close() is True


def test_https_network_context_has_no_openssh_invocation(
    tmp_path: Path,
) -> None:
    _repository, _destination, context = _create_network_context(tmp_path)

    assert context.openssh_invocation() is None
    assert context.close() is True


def test_openssh_executable_substitution_invalidates_context_capability(
    tmp_path: Path,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    executable = _fake_ssh_executable(tmp_path)
    _repository, _destination, context = _create_network_context(
        tmp_path,
        endpoint="ssh://git@push.example.test:22/team/notes.git",
        environment=isolated_ssh_environment,
        ssh_executable=executable,
        allow_ssh_agent=True,
    )
    displaced = executable.with_suffix(".displaced")
    executable.rename(displaced)
    executable.write_text("replacement\n", encoding="utf-8")
    executable.chmod(0o700)

    with pytest.raises(git_network.NetworkContextError):
        context.openssh_invocation()

    assert context.close() is True


def test_network_context_builders_require_live_exact_context_endpoint(
    tmp_path: Path,
) -> None:
    repository, _destination, context = _create_network_context(tmp_path)
    other_repository = _network_repository(tmp_path / "other")
    other_endpoint = _network_endpoint(
        "https://other.example.test/team/notes.git"
    )
    other_destination = other_endpoint.projection
    other_source, other_config = _network_authorizations(
        other_repository,
        other_destination,
    )
    other_parent = tmp_path / "other-contexts"
    other_parent.mkdir(mode=0o700)
    other = git_network.NetworkContextFactory(
        environment={},
        temporary_parent=other_parent,
        git_exec_path=_test_git_installation()[1],
    ).create(
        repository=other_repository,
        source_objects=other_source,
        configuration=other_config,
        destination=other_destination,
        endpoint=other_endpoint,
    )
    resolved = _resolve(_base_facts())
    endpoint = resolved.transport.endpoint
    assert endpoint is not None

    query = context.build_query_argv(endpoint)
    push = context.build_push_argv(
        endpoint,
        "b" * 40,
        "d" * 40,
    )

    private_git_dir = context.command_settings().environment[
        "GIT_DIR"
    ]
    assert f"--git-dir={private_git_dir}" in query
    assert f"--git-dir={private_git_dir}" in push
    assert "ls-remote" in query
    assert "push" in push
    assert repository.worktree_root not in query
    assert repository.worktree_root not in push
    with pytest.raises(git_network.NetworkContextError):
        context.build_query_argv(other_endpoint)
    assert context.close() is True
    with pytest.raises(git_network.NetworkContextError):
        context.build_query_argv(endpoint)
    assert other.close() is True


def test_network_context_cleanup_waits_for_every_retained_purpose(
    tmp_path: Path,
) -> None:
    _repository, _destination, context = _create_network_context(tmp_path)
    root = Path(
        context.command_settings().environment["GIT_DIR"]
    ).parent
    review = context.retain("review")
    active = context.retain("active")
    recovery = context.retain("recovery")

    assert context.close() is False
    assert root.exists()
    assert context.command_settings().cwd
    with pytest.raises(git_network.NetworkContextError):
        context.retain("active")
    assert review.release() is True
    assert review.release() is False
    assert active.release() is True
    assert root.exists()
    assert recovery.release() is True
    assert not root.exists()
    assert context.cleaned is True


def test_network_context_lease_is_bound_to_its_issuing_context(
    tmp_path: Path,
) -> None:
    _repository, _destination, first = _create_network_context(
        tmp_path / "first"
    )
    _repository, _destination, second = _create_network_context(
        tmp_path / "second"
    )
    lease = first.retain("active")

    assert first.close() is False
    assert lease.release() is True
    assert first.cleaned is True
    assert second.cleaned is False
    assert second.close() is True


def test_network_context_cleanup_refuses_unknown_or_hardlinked_shape(
    tmp_path: Path,
) -> None:
    _repository, _destination, context = _create_network_context(tmp_path)
    settings = context.command_settings()
    root = Path(settings.environment["GIT_DIR"]).parent
    unknown = root / "unexpected"
    unknown.write_text("do not delete broad trees", encoding="utf-8")

    assert context.close() is False
    assert unknown.read_text(encoding="utf-8") == "do not delete broad trees"
    unknown.unlink()
    global_config = Path(settings.environment["GIT_CONFIG_GLOBAL"])
    external_link = root.parent / "external-global-link"
    os.link(global_config, external_link)
    assert context.close() is False
    assert external_link.exists()
    external_link.unlink()
    assert context.close() is True
    assert not root.exists()


def test_network_context_cleanup_retries_its_own_partial_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _repository, _destination, context = _create_network_context(tmp_path)
    settings = context.command_settings()
    root = Path(settings.environment["GIT_DIR"]).parent
    fail_path = Path(settings.environment["GIT_CONFIG_GLOBAL"])
    original_unlink = Path.unlink
    failed = False

    def fail_once(path: Path, *args, **kwargs) -> None:
        nonlocal failed
        if path == fail_path and not failed:
            failed = True
            raise OSError("injected cleanup interruption")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_once)
    assert context.close() is False
    assert failed is True
    assert root.exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    assert context.close() is True
    assert not root.exists()


def test_network_context_cleanup_tracks_file_before_failed_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _network_repository(tmp_path)
    destination = _network_destination()
    source, config = _network_authorizations(repository, destination)
    factory = _network_factory(tmp_path)
    parent = tmp_path / "network-contexts"

    def fail_fchmod(_descriptor: int, _mode: int) -> None:
        raise OSError("injected file initialization failure")

    monkeypatch.setattr(git_network.os, "fchmod", fail_fchmod)
    with pytest.raises(git_network.NetworkContextError):
        factory.create(
            repository=repository,
            source_objects=source,
            configuration=config,
            destination=destination,
            endpoint=_network_endpoint(),
        )

    assert tuple(parent.iterdir()) == ()


def test_network_context_rejects_safe_leaf_under_writable_ancestor(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe-ancestor"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    parent = unsafe / "private-parent"
    parent.mkdir(mode=0o700)
    repository = _network_repository(tmp_path / "repository")
    destination = _network_destination()
    source, config = _network_authorizations(repository, destination)
    git_executable = shutil.which("git")
    assert git_executable is not None
    factory = git_network.NetworkContextFactory(
        environment={},
        temporary_parent=parent,
        git_executable=git_executable,
        git_exec_path=_test_git_installation()[1],
    )

    with pytest.raises(git_network.NetworkContextError):
        factory.create(
            repository=repository,
            source_objects=source,
            configuration=config,
            destination=destination,
            endpoint=_network_endpoint(),
        )


def test_network_context_rejects_executable_under_writable_ancestor(
    tmp_path: Path,
) -> None:
    unsafe = tmp_path / "unsafe-executable-ancestor"
    unsafe.mkdir(mode=0o700)
    unsafe.chmod(0o777)
    executable = unsafe / "git"
    executable.write_text("not executed\n", encoding="utf-8")
    executable.chmod(0o700)

    with pytest.raises(git_network.NetworkContextError):
        git_network.NetworkContextFactory(
            environment={},
            temporary_parent=tmp_path,
            git_executable=str(executable),
            git_exec_path=_test_git_installation()[1],
        )


def _approved_helper_name() -> str | None:
    if sys.platform == "darwin":
        return "osxkeychain"
    return None


def test_network_context_pins_approved_helper_and_removes_unrelated_path(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    proved_exec_path = tmp_path / "proved-exec-path"
    proved_exec_path.mkdir(mode=0o700)
    _recording_git_dispatch_executable(
        proved_exec_path / "git-remote-https",
        tmp_path / "unused-remote-log.json",
    )
    helper = proved_exec_path / f"git-credential-{helper_name}"
    helper.write_text("not executed\n", encoding="utf-8")
    helper.chmod(0o700)
    unrelated = tmp_path / "unrelated-bin"
    unrelated.mkdir(mode=0o700)
    _repository, _destination, context = _create_network_context(
        tmp_path,
        facts=(
            _fact(
                "credential.helper",
                helper_name,
                scope="global",
            ),
        ),
        environment={"PATH": str(unrelated)},
        git_exec_path=proved_exec_path,
    )
    settings = context.command_settings()

    assert str(unrelated) not in settings.environment["PATH"].split(os.pathsep)
    helper.unlink()
    with pytest.raises(git_network.NetworkContextError):
        context.command_settings()
    assert context.close() is True


def test_network_context_proved_git_exec_path_dispatches_pinned_targets(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    git_executable, _installed_exec_path = _test_git_installation()
    proved_exec_path = tmp_path / "proved-git-exec"
    proved_exec_path.mkdir(mode=0o700)
    remote_log = tmp_path / "remote-helper.json"
    _recording_git_dispatch_executable(
        proved_exec_path / "git-remote-https",
        remote_log,
    )
    credential_log = tmp_path / "credential-helper.json"
    _recording_git_dispatch_executable(
        proved_exec_path / f"git-credential-{helper_name}",
        credential_log,
        credential=True,
    )
    repository = _network_repository(tmp_path)
    endpoint = _network_endpoint()
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        facts=(
            _fact(
                "credential.helper",
                helper_name,
                scope="global",
            ),
        ),
    )
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)
    context = git_network.NetworkContextFactory(
        environment={},
        temporary_parent=parent,
        git_executable=str(git_executable),
        git_exec_path=proved_exec_path,
    ).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=configuration,
        destination=destination,
        endpoint=endpoint,
    )
    settings = context.command_settings()
    context_exec_path = Path(settings.environment["GIT_EXEC_PATH"])

    assert context_exec_path == proved_exec_path
    assert {
        path.name for path in context_exec_path.iterdir()
    } == {
        "git-remote-https",
        f"git-credential-{helper_name}",
    }
    assert all(
        stat.S_IMODE(path.stat().st_mode) == 0o700
        for path in context_exec_path.iterdir()
    )
    query = subprocess.run(
        context.build_query_argv(endpoint),
        cwd=settings.cwd,
        env=dict(settings.environment),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert query.returncode != 0
    remote_record = json.loads(remote_log.read_text(encoding="utf-8"))
    endpoint_value = "https://push.example.test/team/notes.git"
    assert remote_record["argv"] == [
        str(proved_exec_path / "git-remote-https"),
        endpoint_value,
        endpoint_value,
    ]
    assert remote_record["environment"]["GIT_EXEC_PATH"] == str(
        context_exec_path
    )
    assert remote_record["scratch_writes"] == {
        "HOME": False,
        "XDG_CONFIG_HOME": False,
        "TMPDIR": False,
    }

    credential = subprocess.run(
        (
            str(git_executable.resolve()),
            f"--git-dir={settings.cwd}",
            "credential",
            "fill",
        ),
        cwd=settings.cwd,
        env=dict(settings.environment),
        input=(
            b"protocol=https\n"
            b"host=push.example.test\n"
            b"path=team/notes.git\n\n"
        ),
        capture_output=True,
        timeout=5,
        check=False,
    )
    assert credential.returncode == 0
    assert b"username=pinned-user" in credential.stdout
    credential_record = json.loads(
        credential_log.read_text(encoding="utf-8")
    )
    assert credential_record["argv"] == [
        str(proved_exec_path / f"git-credential-{helper_name}"),
        "get",
    ]
    assert credential_record["environment"]["GIT_EXEC_PATH"] == str(
        context_exec_path
    )
    assert credential_record["scratch_writes"] == {
        "HOME": False,
        "XDG_CONFIG_HOME": False,
        "TMPDIR": False,
    }
    assert context.close() is True


@pytest.mark.parametrize(
    "executable_kind",
    ["git-exec-path", "git", "python", "ssh", "https-helper"],
)
def test_network_context_rejects_source_repository_network_executables(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    executable_kind: str,
    isolated_ssh_environment: Mapping[str, str],
) -> None:
    repository = _network_repository(tmp_path)
    source_root = Path(repository.worktree_root)
    endpoint_value = (
        "ssh://git@push.example.test:22/team/notes.git"
        if executable_kind in {"python", "ssh"}
        else "https://push.example.test/team/notes.git"
    )
    endpoint = _network_endpoint(endpoint_value)
    destination = endpoint.projection
    environment = (
        isolated_ssh_environment
        if destination.scheme == "ssh"
        else {"PATH": os.defpath}
    )
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        environment=environment,
    )
    git_executable, installed_exec_path = _test_git_installation()
    selected_git = git_executable
    selected_exec_path = installed_exec_path
    selected_ssh: Path | None = None

    source_executable = source_root / f"network-{executable_kind}"
    source_executable.write_text("not executed\n", encoding="utf-8")
    source_executable.chmod(0o700)
    if executable_kind == "git-exec-path":
        source_executable.unlink()
        selected_exec_path = source_root / "git-exec"
        selected_exec_path.mkdir(mode=0o700)
        _recording_git_dispatch_executable(
            selected_exec_path / "git-remote-https",
            tmp_path / "unused-contained-exec-log.json",
        )
    elif executable_kind == "git":
        source_executable = source_executable.rename(source_root / "git")
        selected_git = source_executable
    elif executable_kind == "python":
        monkeypatch.setattr(git_network.sys, "executable", str(source_executable))
    elif executable_kind == "ssh":
        selected_ssh = source_executable
    else:
        selected_exec_path = tmp_path / "proved-exec-path"
        selected_exec_path.mkdir(mode=0o700)
        (selected_exec_path / "git-remote-https").symlink_to(
            source_executable
        )

    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)
    factory = git_network.NetworkContextFactory(
        environment=environment,
        temporary_parent=parent,
        git_executable=str(selected_git),
        git_exec_path=selected_exec_path,
        ssh_executable=(
            None if selected_ssh is None else str(selected_ssh)
        ),
        allow_ssh_agent=destination.scheme == "ssh",
    )

    with pytest.raises(git_network.NetworkContextError):
        factory.create(
            repository=repository,
            source_objects=source_authorization,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )


def test_https_network_context_ignores_source_repository_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = _network_repository(tmp_path)
    source_python = Path(repository.worktree_root) / "python"
    source_python.write_text("not executed\n", encoding="utf-8")
    source_python.chmod(0o700)
    monkeypatch.setattr(git_network.sys, "executable", str(source_python))
    endpoint = _network_endpoint()
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
    )
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)

    context = git_network.NetworkContextFactory(
        environment={},
        temporary_parent=parent,
        git_executable=str(_test_git_installation()[0]),
        git_exec_path=_test_git_installation()[1],
    ).create(
        repository=repository,
        source_objects=source_authorization,
        configuration=configuration,
        destination=destination,
        endpoint=endpoint,
    )

    source_python.unlink()
    assert context.command_settings().cwd
    assert context.close() is True


def test_network_context_rejects_credential_helper_inside_source_repository(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    repository = _network_repository(tmp_path)
    helper_bin = Path(repository.worktree_root) / "helper-bin"
    helper_bin.mkdir(mode=0o700)
    helper = helper_bin / f"git-credential-{helper_name}"
    helper.write_text("not executed\n", encoding="utf-8")
    helper.chmod(0o700)
    proved_exec_path = tmp_path / "proved-exec-path"
    proved_exec_path.mkdir(mode=0o700)
    _recording_git_dispatch_executable(
        proved_exec_path / "git-remote-https",
        tmp_path / "unused-source-remote-log.json",
    )
    (proved_exec_path / f"git-credential-{helper_name}").symlink_to(
        helper
    )
    endpoint = _network_endpoint()
    destination = endpoint.projection
    source_authorization, configuration = _network_authorizations(
        repository,
        destination,
        facts=(
            _fact(
                "credential.helper",
                helper_name,
                scope="global",
            ),
        ),
    )

    with pytest.raises(git_network.NetworkContextError):
        _network_factory(
            tmp_path,
            environment={},
            git_exec_path=proved_exec_path,
        ).create(
            repository=repository,
            source_objects=source_authorization,
            configuration=configuration,
            destination=destination,
            endpoint=endpoint,
        )


def test_network_context_restart_never_discovers_or_reuses_crash_left_directory(
    tmp_path: Path,
) -> None:
    parent = tmp_path / "network-contexts"
    parent.mkdir(mode=0o700)
    orphan = parent / ".chatbook-network-git-crash-left"
    orphan.mkdir(mode=0o700)
    marker = orphan / "opaque-marker"
    marker.write_bytes(b"no credentials or note body")
    repository = _network_repository(tmp_path)
    destination = _network_destination()
    source, config = _network_authorizations(repository, destination)

    context = git_network.NetworkContextFactory(
        environment={},
        temporary_parent=parent,
        git_exec_path=_test_git_installation()[1],
    ).create(
        repository=repository,
        source_objects=source,
        configuration=config,
        destination=destination,
        endpoint=_network_endpoint(),
    )
    root = Path(
        context.command_settings().environment["GIT_DIR"]
    ).parent

    assert root != orphan
    assert marker.read_bytes() == b"no credentials or note body"
    assert context.close() is True
    assert orphan.exists()


def _candidate_owner(
    tmp_path: Path,
) -> tuple[FileNotesSessionOwner, SessionBinding, RepositoryIdentity]:
    root = tmp_path / "notes"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "objects").mkdir()
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (root / "note.md").write_text("candidate\n", encoding="utf-8")
    return _owner_for_candidate(root, "b" * 40, "d" * 40)


def _current_push_operation(
    service: FileNotesGitService,
    binding: SessionBinding,
) -> git_service.RetainedPushOperation:
    operation = service.retained_push_operation(binding)
    assert operation is not None
    return operation


def _authorize_current_push(
    service: FileNotesGitService,
    binding: SessionBinding,
) -> asyncio.Task[git_service.PushPreflightResult]:
    return service.authorize_and_check_push(
        binding,
        _current_push_operation(service, binding),
    )


def _cancel_current_push(
    service: FileNotesGitService,
    binding: SessionBinding,
) -> bool:
    return service.cancel_push(
        binding,
        _current_push_operation(service, binding),
    )


def _owner_for_candidate(
    root: Path,
    parent_oid: str,
    candidate_oid: str,
) -> tuple[FileNotesSessionOwner, SessionBinding, RepositoryIdentity]:
    git_dir = root / ".git"
    repository = RepositoryIdentity(
        worktree_root=str(root.resolve()),
        git_dir=str(git_dir.resolve()),
        git_common_dir=str(git_dir.resolve()),
        worktree_identity=_filesystem_identity(root),
        git_dir_identity=_filesystem_identity(git_dir),
        git_common_dir_identity=_filesystem_identity(git_dir),
    )
    owner = FileNotesSessionOwner()
    binding = owner.select_root(root)
    _publish_candidate_on_owner(
        owner,
        binding,
        repository,
        parent_oid=parent_oid,
        candidate_oid=candidate_oid,
    )
    return owner, binding, repository


def _publish_candidate_on_owner(
    owner: FileNotesSessionOwner,
    binding: SessionBinding,
    repository: RepositoryIdentity,
    *,
    parent_oid: str,
    candidate_oid: str,
) -> None:
    """Publish one guarded candidate on an existing exact root binding."""
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
    sequence = owner.snapshot(binding).changes[-1].sequence
    assert owner.publish_trust(binding, repository)
    head = HeadIdentity.attached(BRANCH_REF, parent_oid)
    ownership = StagingOwnership(
        repository=repository,
        head=head,
        approved_endpoint_topology=("note.md",),
        approved_move_edges=(),
        approved_current_path="note.md",
        original_baselines={
            "note.md": IndexBaseline(
                IndexEntry("note.md", "100644", "a" * 40)
            )
        },
        post_stage_entries={
            "note.md": IndexEntry("note.md", "100644", "c" * 40)
        },
    )
    assert owner.publish_ownership(binding, {sequence: ownership})
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    reviewed = owner._capture_commit_authority_after_review(
        lease,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=head,
        group_sequence_ids={sequence: (sequence,)},
        subject="Guarded note",
        included_notes=(PushIncludedNote(sequence, "note.md"),),
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
            retired_sequence_ids=(sequence,),
            candidate_seed=capture._candidate_seed,
        ),
    )
    assert publication.published
    lease.release()


class _ControlledLocalProofRunner:
    """Interpret only bounded local proof commands and record every boundary."""

    def __init__(
        self,
        repository: RepositoryIdentity,
        *,
        paths: tuple[bytes, ...] = (b"note.md",),
        lfs_paths: frozenset[bytes] = frozenset(),
        malformed_attributes: bool = False,
        change_config_during_read: bool = False,
        index_payload: bytes = b"controlled-index",
        index_directory: bool = False,
        read_tree_returncode: int = 0,
        object_format: str = "sha1",
        git_exec_path: Path | None = None,
        head_oid: str = "d" * 40,
        parent_oid: str = "b" * 40,
        push_url: str = "https://push.example.test/team/notes.git",
    ) -> None:
        self.repository = repository
        self.paths = paths
        self.lfs_paths = lfs_paths
        self.malformed_attributes = malformed_attributes
        self.change_config_during_read = change_config_during_read
        self.index_payload = index_payload
        self.index_directory = index_directory
        self.read_tree_returncode = read_tree_returncode
        self.object_format = object_format
        self.head_oid = head_oid
        self.parent_oid = parent_oid
        self.git_exec_path = (
            _test_git_installation()[1]
            if git_exec_path is None
            else git_exec_path
        )
        self.config_reads = 0
        self.calls: list[
            tuple[tuple[str | bytes, ...], Mapping[str, str], bytes | None]
        ] = []
        config_path = os.fsencode(Path(repository.git_dir) / "config")
        records = (
            ("branch.main.remote", "origin"),
            ("branch.main.merge", BRANCH_REF),
            (
                "remote.origin.pushurl",
                push_url,
            ),
            ("remote.origin.url", "https://fetch.example.test/team/notes.git"),
        )
        self.config_payload = b"".join(
            b"local\0file:"
            + config_path
            + b"\0"
            + key.encode()
            + b"\n"
            + value.encode()
            + b"\0"
            for key, value in records
        )

    async def run(
        self,
        argv: tuple[str | bytes, ...],
        *,
        cwd: str,
        environment: Mapping[str, str],
        stdin: bytes | None = None,
        timeout: float | None = None,
        stdout_limit: int | None = None,
        stderr_limit: int | None = None,
        on_spawn: Callable[[], None] | None = None,
        cancel_before_spawn: bool = False,
    ) -> GitCommandResult:
        del timeout, stdout_limit, stderr_limit, on_spawn, cancel_before_spawn
        assert cwd == self.repository.worktree_root
        self.calls.append((argv, dict(environment), stdin))
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "--exec-path" in command:
            return GitCommandResult(
                0,
                os.fsencode(self.git_exec_path) + b"\n",
                b"",
            )
        if "symbolic-ref" in command:
            return GitCommandResult(0, BRANCH_REF.encode() + b"\n", b"")
        if "rev-parse" in command:
            assert "--show-object-format=storage" in command
            return GitCommandResult(
                0,
                self.object_format.encode("ascii")
                + b"\n"
                + self.head_oid.encode("ascii")
                + b"\n",
                b"",
            )
        if "cat-file" in command:
            return GitCommandResult(
                0,
                (
                    b"tree "
                    + b"e" * 40
                    + b"\nparent "
                    + self.parent_oid.encode("ascii")
                    + b"\nauthor A <a@example.test> 1 +0000"
                    + b"\ncommitter C <c@example.test> 1 +0000"
                    + b"\n\nGuarded note\n"
                ),
                b"",
            )
        if "config" in command:
            self.config_reads += 1
            if self.change_config_during_read and self.config_reads == 2:
                config_path = Path(self.repository.git_dir) / "config"
                config_path.write_text(
                    config_path.read_text(encoding="utf-8") + "# changed\n",
                    encoding="utf-8",
                )
            return GitCommandResult(0, self.config_payload, b"")
        if "diff-tree" in command:
            return GitCommandResult(0, b"\0".join(self.paths) + b"\0", b"")
        if "read-tree" in command:
            index_path = Path(environment["GIT_INDEX_FILE"])
            if self.index_directory:
                index_path.mkdir(mode=0o700)
            else:
                index_path.write_bytes(self.index_payload)
            return GitCommandResult(self.read_tree_returncode, b"", b"")
        if "check-attr" in command:
            assert stdin is not None
            if self.malformed_attributes:
                return GitCommandResult(0, b"malformed", b"")
            requested = tuple(path for path in stdin.split(b"\0") if path)
            return GitCommandResult(
                0,
                b"".join(
                    path
                    + b"\0filter\0"
                    + (b"lfs" if path in self.lfs_paths else b"unspecified")
                    + b"\0"
                    for path in requested
                ),
                b"",
            )
        raise AssertionError(f"unexpected local proof command: {command!r}")

    def shutdown(self) -> None:
        return None


class _ControlledPushPreflightRunner(_ControlledLocalProofRunner):
    """Add one exact remote-ref observation to the local-proof interpreter."""

    def __init__(
        self,
        repository: RepositoryIdentity,
        observation: GitCommandResult,
        **local_proof_options,
    ) -> None:
        super().__init__(repository, **local_proof_options)
        self.observation = observation
        self.network_calls: list[tuple[tuple[str | bytes, ...], dict[str, object]]] = []

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" not in command:
            return await super().run(argv, **kwargs)
        self.network_calls.append((tuple(argv), dict(kwargs)))
        return self.observation


class _ControlledExactPushRunner(_ControlledLocalProofRunner):
    """Interpret the fixed preflight, push, and postflight network sequence."""

    def __init__(
        self,
        repository: RepositoryIdentity,
        *,
        observations: tuple[GitCommandResult, ...],
        push_result: GitCommandResult,
        launch_error: bool = False,
        **local_proof_options,
    ) -> None:
        super().__init__(repository, **local_proof_options)
        self.observations = list(observations)
        self.push_result = push_result
        self.launch_error = launch_error
        self.network_calls: list[
            tuple[tuple[str | bytes, ...], dict[str, object]]
        ] = []

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if not {"ls-remote", "push"}.intersection(command):
            return await super().run(argv, **kwargs)
        self.network_calls.append((tuple(argv), dict(kwargs)))
        if "ls-remote" in command:
            assert self.observations
            return self.observations.pop(0)
        if self.launch_error:
            raise OSError("controlled launch refusal")
        on_spawn = kwargs.get("on_spawn")
        assert callable(on_spawn)
        on_spawn()
        return self.push_result


class _UnprovedCancelledPushRecoveryRunner(_ControlledExactPushRunner):
    """Lose containment proof for the exact cancelled recovery child."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.token = git_service.RetainedGitChildToken(
            git_service._RETAINED_CHILD_TOKEN_SECRET
        )
        self.query_count = 0
        self.claim_calls = 0
        self.shutdown_proved = False
        self.released = False
        self.settlement_started = asyncio.Event()
        self.release_settlement = asyncio.Event()
        self.released_event = asyncio.Event()

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" in command:
            self.query_count += 1
            if self.query_count == 4:
                self.network_calls.append((tuple(argv), dict(kwargs)))
                raise git_service.GitRunCancelled(
                    retained_child=self.token
                ) from None
        return await super().run(argv, **kwargs)

    def claim_retained_child(self, token) -> bool:
        assert token is self.token
        self.claim_calls += 1
        return self.claim_calls > 1

    async def settle_retained_child(self, token, *, timeout):
        del timeout
        assert token is self.token
        self.settlement_started.set()
        await self.release_settlement.wait()
        return git_service.RetainedGitChildSettlement(
            "natural",
            0,
            owned_process_tree=True,
            containment_proved=True,
        )

    def release_retained_child(self, token) -> bool:
        assert token is self.token
        self.released = True
        self.released_event.set()
        return True

    def shutdown(self) -> None:
        self.shutdown_proved = True
        self.release_settlement.set()


class _PushSpawnBarrier:
    """Deterministically pause after final proof and before push admission."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def __call__(self) -> None:
        self.entered.set()
        await self.release.wait()


class _BlockingExactPushRunner(_ControlledExactPushRunner):
    """Pause only after publishing the runner's actual child-start signal."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.push_started = asyncio.Event()
        self.release_push = asyncio.Event()

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "push" not in command:
            return await super().run(argv, **kwargs)
        self.network_calls.append((tuple(argv), dict(kwargs)))
        on_spawn = kwargs.get("on_spawn")
        assert callable(on_spawn)
        on_spawn()
        self.push_started.set()
        await self.release_push.wait()
        return self.push_result


class _NoSpawnCallbackExactPushRunner(_ControlledExactPushRunner):
    """Return a settled push result without claiming actual child admission."""

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "push" not in command:
            return await super().run(argv, **kwargs)
        self.network_calls.append((tuple(argv), dict(kwargs)))
        return self.push_result


class _BlockingPushPreflightRunner(_ControlledPushPreflightRunner):
    """Pause one exact query while its public waiter may disappear."""

    def __init__(
        self,
        repository: RepositoryIdentity,
        observation: GitCommandResult,
    ) -> None:
        super().__init__(repository, observation)
        self.started = asyncio.Event()
        self.release_query = asyncio.Event()

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" not in command:
            return await _ControlledLocalProofRunner.run(self, argv, **kwargs)
        self.network_calls.append((tuple(argv), dict(kwargs)))
        self.started.set()
        await self.release_query.wait()
        return self.observation


class _BlockingPushRevalidationRunner(_ControlledLocalProofRunner):
    """Pause the second exact local proof before authorization/context use."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.exec_path_calls = 0
        self.revalidation_started = asyncio.Event()

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "--exec-path" in command:
            self.exec_path_calls += 1
            if self.exec_path_calls == 2:
                self.revalidation_started.set()
                await asyncio.Event().wait()
        return await super().run(argv, **kwargs)


class _RetainedCancelledPushPreflightRunner(_ControlledLocalProofRunner):
    """Expose one cancelled read-only child until explicit tree settlement."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.token = git_service.RetainedGitChildToken(
            git_service._RETAINED_CHILD_TOKEN_SECRET
        )
        self.started = asyncio.Event()
        self.claimed = asyncio.Event()
        self.tree_settled = asyncio.Event()
        self.shutdown_called = asyncio.Event()
        self.released = False

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" not in command:
            return await super().run(argv, **kwargs)
        assert kwargs["owned_process_tree"] is True
        self.started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            raise git_service.GitRunCancelled(
                retained_child=self.token
            ) from None
        raise AssertionError("unreachable")

    def claim_retained_child(self, token) -> bool:
        assert token is self.token
        self.claimed.set()
        return True

    async def settle_retained_child(self, token, *, timeout):
        assert token is self.token
        try:
            await asyncio.wait_for(
                self.tree_settled.wait(),
                timeout=timeout,
            )
        except TimeoutError:
            return git_service.RetainedGitChildSettlement(
                "alive",
                owned_process_tree=True,
            )
        return git_service.RetainedGitChildSettlement(
            "stop_requested",
            -15,
            owned_process_tree=True,
            containment_proved=True,
            stop_requested=True,
        )

    def release_retained_child(self, token) -> bool:
        assert token is self.token
        assert self.tree_settled.is_set()
        self.released = True
        return True

    def shutdown(self) -> None:
        self.shutdown_called.set()


class _RetainedCancelledConfirmQueryRunner(
    _RetainedCancelledPushPreflightRunner
):
    """Settle cancellation of Confirm's retained final query child."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.query_count = 0
        self.network_calls: list[
            tuple[tuple[str | bytes, ...], dict[str, object]]
        ] = []

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" not in command:
            return await _ControlledLocalProofRunner.run(
                self,
                argv,
                **kwargs,
            )
        self.network_calls.append((tuple(argv), dict(kwargs)))
        self.query_count += 1
        if self.query_count == 1:
            return _remote_observation("b" * 40)
        return await super().run(argv, **kwargs)


class _RetainedCancelledPostflightRunner(
    _RetainedCancelledPushPreflightRunner
):
    """Expose a retained child when the exact postflight is cancelled."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.query_count = 0
        self.network_calls: list[
            tuple[tuple[str | bytes, ...], dict[str, object]]
        ] = []

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if not {"ls-remote", "push"}.intersection(command):
            return await _ControlledLocalProofRunner.run(
                self,
                argv,
                **kwargs,
            )
        self.network_calls.append((tuple(argv), dict(kwargs)))
        if "push" in command:
            on_spawn = kwargs.get("on_spawn")
            assert callable(on_spawn)
            on_spawn()
            return _accepted_push_result()
        self.query_count += 1
        if self.query_count <= 2:
            return _remote_observation("b" * 40)
        self.started.set()
        raise git_service.GitRunCancelled(
            retained_child=self.token
        ) from None


class _NormallyReturnedRetainedPushPreflightRunner(
    _RetainedCancelledPushPreflightRunner
):
    """Return a retained token before its native process tree is proved empty."""

    async def run(self, argv, **kwargs) -> GitCommandResult:
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "ls-remote" not in command:
            return await _ControlledLocalProofRunner.run(self, argv, **kwargs)
        assert kwargs["owned_process_tree"] is True
        self.started.set()
        return GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            retained_child=self.token,
            owned_process_tree=True,
        )

    async def settle_retained_child(self, token, *, timeout):
        assert token is self.token
        try:
            await asyncio.wait_for(
                self.tree_settled.wait(),
                timeout=timeout,
            )
        except TimeoutError:
            return git_service.RetainedGitChildSettlement(
                "alive",
                owned_process_tree=True,
            )
        return git_service.RetainedGitChildSettlement(
            "natural",
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        )


class _UnprovedShutdownPushPreflightRunner(
    _RetainedCancelledPushPreflightRunner
):
    """Report bounded controller failure while the owned tree stays unproved."""

    async def settle_retained_child(self, token, *, timeout):
        assert token is self.token
        if self.tree_settled.is_set():
            return git_service.RetainedGitChildSettlement(
                "stop_requested",
                -15,
                owned_process_tree=True,
                containment_proved=True,
                stop_requested=True,
            )
        return git_service.RetainedGitChildSettlement(
            "contained_uncertain",
            owned_process_tree=True,
            containment_proved=False,
        )

    def shutdown(self):
        self.shutdown_called.set()

        async def settle() -> bool:
            await asyncio.sleep(0)
            return False

        return settle()


class _OwnershipFailurePushPreflightRunner(
    _NormallyReturnedRetainedPushPreflightRunner
):
    """Fail one retained-token ownership seam after the child is exposed."""

    def __init__(self, repository: RepositoryIdentity, failure: str) -> None:
        super().__init__(repository)
        self.failure = failure
        self.failure_observed = asyncio.Event()
        self.shutdown_proved = False

    def claim_retained_child(self, token) -> bool:
        assert token is self.token
        if self.failure == "claim" and not self.shutdown_proved:
            self.failure_observed.set()
            return False
        return super().claim_retained_child(token)

    async def settle_retained_child(self, token, *, timeout):
        assert token is self.token
        if self.failure == "settle" and not self.shutdown_proved:
            self.failure_observed.set()
            raise RuntimeError("REMOTE_HELPER_SECRET_CANARY")
        return await super().settle_retained_child(token, timeout=timeout)

    def shutdown(self):
        self.shutdown_proved = True
        self.tree_settled.set()
        self.shutdown_called.set()

        async def settle() -> bool:
            await asyncio.sleep(0)
            return True

        return settle()


class _DelayedSettlementPushPreflightRunner(
    _NormallyReturnedRetainedPushPreflightRunner
):
    """Cross the shutdown handoff deadline before yielding terminal proof."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.settlement_started = asyncio.Event()
        self.settlement_attempts = 0

    async def settle_retained_child(self, token, *, timeout):
        assert token is self.token
        self.settlement_attempts += 1
        if self.settlement_attempts <= 2:
            self.settlement_started.set()
            await asyncio.Event().wait()
        return git_service.RetainedGitChildSettlement(
            "natural",
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        )

    def shutdown(self):
        self.shutdown_called.set()
        self.tree_settled.set()

        async def settle() -> bool:
            await asyncio.sleep(0)
            return True

        return settle()


class _RecordingAsyncGitRunner:
    """Run real local Git while retaining bounded command/result evidence."""

    def __init__(self) -> None:
        self.delegate = AsyncGitProcessRunner()
        self.calls: list[tuple[tuple[str, ...], GitCommandResult]] = []

    async def run(self, argv, **kwargs) -> GitCommandResult:
        result = await self.delegate.run(argv, **kwargs)
        self.calls.append(
            (
                tuple(os.fsdecode(argument) for argument in argv),
                result,
            )
        )
        return result

    def shutdown(self):
        return self.delegate.shutdown()

    def read_retained_child(self, token):
        return self.delegate.read_retained_child(token)

    def claim_retained_child(self, token):
        return self.delegate.claim_retained_child(token)

    async def settle_retained_child(self, token, *, timeout):
        return await self.delegate.settle_retained_child(token, timeout=timeout)

    def release_retained_child(self, token):
        return self.delegate.release_retained_child(token)


class _ReplacingAttributeProofRunner(_RecordingAsyncGitRunner):
    """Replace the isolated Git directory after read-tree settles."""

    def __init__(self) -> None:
        super().__init__()
        self.replaced = False
        self.replacement_marker: Path | None = None

    async def run(self, argv, **kwargs) -> GitCommandResult:
        result = await super().run(argv, **kwargs)
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "read-tree" not in command or self.replaced:
            return result

        environment = kwargs["environment"]
        git_dir = Path(environment["GIT_DIR"])
        displaced = git_dir.with_name(f"{git_dir.name}.displaced")
        git_dir.rename(displaced)
        git_dir.mkdir(mode=0o700)
        (git_dir / "objects").mkdir(mode=0o700)
        (git_dir / "refs").mkdir(mode=0o700)
        info = git_dir / "info"
        info.mkdir(mode=0o700)
        (git_dir / "HEAD").write_text(
            "ref: refs/heads/replaced\n",
            encoding="utf-8",
        )
        marker = info / "attributes"
        marker.write_text("*.md -filter\n", encoding="utf-8")
        self.replaced = True
        self.replacement_marker = marker
        return result


class _ChangingObjectDirectoryModeRunner(_ControlledLocalProofRunner):
    """Change source-object directory metadata after isolated read-tree."""

    def __init__(self, repository: RepositoryIdentity) -> None:
        super().__init__(repository)
        self.changed = False

    async def run(self, argv, **kwargs) -> GitCommandResult:
        result = await super().run(argv, **kwargs)
        command = tuple(os.fsdecode(argument) for argument in argv)
        if "read-tree" in command and not self.changed:
            (Path(self.repository.git_common_dir) / "objects").chmod(0o777)
            self.changed = True
        return result


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        "0.25",
        None,
        object(),
        0,
        0.0,
        -0.25,
        float("nan"),
        float("inf"),
        float("-inf"),
    ],
    ids=[
        "true",
        "false",
        "string",
        "none",
        "object",
        "zero-int",
        "zero-float",
        "negative",
        "nan",
        "positive-infinity",
        "negative-infinity",
    ],
)
def test_push_query_timeout_validation_rejects_invalid_values(
    value: object,
) -> None:
    with pytest.raises(
        ValueError,
        match="push_query_timeout must be a finite positive number",
    ):
        FileNotesGitService(
            FileNotesSessionOwner(),
            push_query_timeout=value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("value", [0.001, 0.25, 29.999, 30.0, 120.0])
def test_push_query_timeout_validation_preserves_positive_finite_values(
    value: float,
) -> None:
    service = FileNotesGitService(
        FileNotesSessionOwner(),
        push_query_timeout=value,
    )

    assert service._push_query_timeout == value


@pytest.mark.asyncio
async def test_review_push_destination_no_network_and_authorization_contacts_nothing(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={
            "GIT_DIR": "CANARY_GIT_DIR",
            "GIT_INDEX_FILE": "CANARY_INDEX",
            "GIT_SSH_COMMAND": "CANARY_SSH",
            "GIT_ASKPASS": "CANARY_HELPER",
            "HTTPS_PROXY": "CANARY_PROXY",
        },
    )

    review = await service.review_push_destination(binding)
    call_count = len(runner.calls)
    authorization = service.authorize_push_destination(binding)

    assert review.state == "ready"
    assert review.authorization is not None
    assert review.remote_contact_started is False
    assert authorization is not None
    assert len(runner.calls) == call_count
    for argv, environment, _stdin in runner.calls:
        command = tuple(os.fsdecode(argument) for argument in argv)
        assert not {"ls-remote", "push", "fetch", "clone"}.intersection(command)
        assert "CANARY_GIT_DIR" not in command
        assert "CANARY_INDEX" not in command
        assert all("CANARY" not in value for value in environment.values())
        assert "GIT_SSH_COMMAND" not in environment
        assert "GIT_ASKPASS" not in environment
        assert "HTTPS_PROXY" not in environment


@pytest.mark.asyncio
async def test_start_push_review_retains_local_proof_and_decline_preserves_candidate(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    local = await service.start_push_review(binding)

    assert local.state == "ready"
    retained = service.retained_push_operation(binding)
    assert retained is not None
    assert retained.kind == "local_proof"
    assert retained.settled
    assert _cancel_current_push(service, binding)
    assert owner.snapshot(binding).push_candidate is not None
    assert service.authorize_push_destination(binding) is None
    assert service.retained_push_operation(binding) is None
    assert not any(
        {"ls-remote", "push"}.intersection(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )


@pytest.mark.asyncio
async def test_push_authorization_and_cancel_require_exact_retained_operation(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _RetainedCancelledPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    first = service.retained_push_operation(binding)
    assert first is not None
    assert service.cancel_push(binding, first)
    assert (await service.start_push_review(binding)).state == "ready"
    second = service.retained_push_operation(binding)
    assert second is not None and second is not first

    with pytest.raises(git_service.GitMutationAdmissionError):
        service.authorize_and_check_push(binding, first)
    assert not runner.started.is_set()

    waiter = service.authorize_and_check_push(binding, second)
    preflight = service.retained_push_operation(binding)
    assert preflight is not None and preflight is not second
    await asyncio.wait_for(runner.started.wait(), timeout=1)
    assert not service.cancel_push(binding, first)
    assert not service.cancel_push(binding, second)
    assert service.cancel_push(binding, preflight)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)
    runner.tree_settled.set()
    result = await asyncio.wait_for(waiter, timeout=1)
    assert result.state == "cancelled"


@pytest.mark.asyncio
async def test_push_authorization_revalidates_local_policy_before_context_or_query(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    factory = _network_factory(tmp_path)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=factory,
    )
    assert (await service.start_push_review(binding)).state == "ready"
    before = owner.snapshot(binding)
    runner.lfs_paths = frozenset({b"note.md"})

    result = await _authorize_current_push(service, binding)

    after = owner.snapshot(binding)
    assert result.state == "blocked"
    assert after.destination_authorization_epoch == before.destination_authorization_epoch
    assert after.push_candidate is not None
    assert runner.network_calls == []
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_ssh_host_trust_unsafe_source_blocks_local_proof_without_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unsafe trust must block before authorization, context, or network contact."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(1, b"", b"must not run"),
        push_url="ssh://git@push.example.test:22/team/notes.git",
    )
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "unsafe-ssh-home"
    ssh_directory = home / ".ssh"
    ssh_directory.mkdir(parents=True, mode=0o700)
    known_hosts = ssh_directory / "known_hosts"
    known_hosts.write_bytes(b"host-key\n")
    known_hosts.chmod(0o620)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path.resolve()),
    }
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment=environment,
    )
    try:
        before = owner.snapshot(binding)

        local = await service.start_push_review(binding)

        after = owner.snapshot(binding)
        assert local.state == "blocked"
        assert runner.network_calls == []
        assert after.destination_authorization_epoch == (
            before.destination_authorization_epoch
        )
        assert service._push_destination_policy is None
    finally:
        await service.shutdown()
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_ssh_host_trust_source_replacement_before_confirm_revokes_review(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replacing trust with equal bytes must not preserve a reviewed command."""
    owner, binding, repository = _candidate_owner(tmp_path)
    parent_observation = GitCommandResult(
        0,
        b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
        b"",
        owned_process_tree=True,
        containment_proved=True,
    )
    runner = _ControlledExactPushRunner(
        repository,
        observations=(parent_observation,),
        push_result=GitCommandResult(0, b"", b""),
        push_url="ssh://git@push.example.test:22/team/notes.git",
    )
    _isolated_system_ssh_trust_paths(tmp_path, monkeypatch)
    home = tmp_path / "ssh-home"
    ssh_directory = home / ".ssh"
    ssh_directory.mkdir(parents=True, mode=0o700)
    known_hosts = ssh_directory / "known_hosts"
    trust_payload = b"host-key\n"
    known_hosts.write_bytes(trust_payload)
    known_hosts.chmod(0o600)
    agent_path = _short_agent_socket_path(tmp_path)
    agent = _bound_agent_socket(agent_path)
    environment = {
        "HOME": str(home.resolve()),
        "PATH": os.defpath,
        "SSH_AUTH_SOCK": str(agent_path.resolve()),
    }
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment=environment,
        network_context_factory=_network_factory(
            tmp_path,
            environment=environment,
            ssh_executable=_fake_ssh_executable(tmp_path),
            allow_ssh_agent=True,
        ),
    )
    try:
        assert (await service.start_push_review(binding)).state == "ready"
        reviewed = await _authorize_current_push(service, binding)
        assert reviewed.state == "review"
        assert reviewed.handle is not None
        snapshot = service._push_review_snapshots[reviewed.handle]
        authorization = snapshot.authorization
        replacement = ssh_directory / "known_hosts.next"
        replacement.write_bytes(trust_payload)
        replacement.chmod(0o600)
        replacement.replace(known_hosts)
        changed_network = _ssh_network_authorization(
            repository,
            snapshot.policy.configuration.transport.destination,
            environment,
        )
        changed_policy = replace(
            snapshot.policy,
            network_configuration=changed_network,
        )
        settings = snapshot.context.command_settings()

        assert service._push_command_policy_fingerprint(
            changed_policy,
            snapshot.context,
            settings.environment_fingerprint,
        ) != snapshot.command_policy_fingerprint
        result = await service.start_push(binding, reviewed.handle)

        assert result.state == "blocked"
        assert len(runner.network_calls) == 1
        assert not owner._destination_authorization_matches(
            snapshot.policy.owner_capture,
            authorization,
        )
    finally:
        await service.shutdown()
        agent.close()
        agent_path.unlink(missing_ok=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("exec_path_change", ["drift", "invalid"])
async def test_push_authorization_exec_path_drift_blocks_before_context_or_query(
    tmp_path: Path,
    exec_path_change: str,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    changed_exec_path = tmp_path / "changed-git-exec-path"
    if exec_path_change == "drift":
        changed_exec_path.mkdir()
    runner.git_exec_path = changed_exec_path

    result = await _authorize_current_push(service, binding)

    assert result.state == "blocked"
    assert runner.network_calls == []
    assert owner.snapshot(binding).push_candidate is not None
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_authorization_windows_refuses_before_grant_context_or_query(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    before = owner.snapshot(binding)
    monkeypatch.setattr(git_service.os, "name", "nt")

    result = await _authorize_current_push(service, binding)

    assert result.state == "blocked"
    assert (
        owner.snapshot(binding).destination_authorization_epoch
        == before.destination_authorization_epoch
    )
    assert runner.network_calls == []
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_preflight_parent_issues_one_review_and_back_preserves_candidate(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    parent_line = b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n"
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            parent_line,
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
        push_query_timeout=0.25,
    )
    local = await service.start_push_review(binding)
    assert local.state == "ready"

    result = await _authorize_current_push(service, binding)

    assert result.state == "review"
    assert result.handle is not None
    assert result.review is not None
    assert result.review.candidate.candidate_oid == "d" * 40
    assert result.review.destination.destination_ref == BRANCH_REF
    config_path = Path(repository.git_dir) / "config"
    config_path.write_text(
        (
            "[branch \"main\"]\n"
            "\tremote = replacement\n"
            f"\tmerge = {BRANCH_REF}\n"
            "[remote \"replacement\"]\n"
            "\tpushurl = https://replacement.example.test/other.git\n"
        ),
        encoding="utf-8",
    )
    assert result.review.configured_remote_label == "origin"
    assert len(runner.network_calls) == 1
    argv, call = runner.network_calls[0]
    command = tuple(os.fsdecode(argument) for argument in argv)
    assert command[-3:] == (
        "--",
        "https://push.example.test/team/notes.git",
        BRANCH_REF,
    )
    assert "push" not in command
    assert call["stdin"] is None
    assert call["timeout"] == 0.25
    assert Path(call["environment"]["GIT_EXEC_PATH"]) == runner.git_exec_path
    assert call["owned_process_tree"] is True
    assert call["stdout_limit"] is not None
    assert call["stderr_limit"] is not None
    review_snapshot = service._push_review_snapshots[result.handle]
    settings = review_snapshot.context.command_settings()
    service._push_query_timeout = 0.5
    try:
        assert service._push_command_policy_fingerprint(
            review_snapshot.policy,
            review_snapshot.context,
            settings.environment_fingerprint,
        ) != review_snapshot.command_policy_fingerprint
    finally:
        service._push_query_timeout = 0.25
    retained = service.retained_push_operation(binding)
    assert retained is not None and retained.settled

    assert _cancel_current_push(service, binding)
    assert owner.snapshot(binding).push_candidate is not None
    assert service.retained_push_operation(binding) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observation_case", "observation"),
    [
        (
            "missing",
            GitCommandResult(
                0,
                b"",
                b"",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
        (
            "deleted",
            GitCommandResult(
                0,
                b"",
                b"",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
        (
            "divergent",
            GitCommandResult(
                0,
                b"a" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
                b"",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
        (
            "plural",
            GitCommandResult(
                0,
                2 * (b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n"),
                b"",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
        (
            "malformed",
            GitCommandResult(
                0,
                b"REMOTE_SECRET_CANARY\n",
                b"",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
        (
            "inaccessible",
            GitCommandResult(
                128,
                b"",
                b"REMOTE_SECRET_CANARY\n",
                owned_process_tree=True,
                containment_proved=True,
            ),
        ),
    ],
)
async def test_push_preflight_blocked_observations_preserve_candidate(
    tmp_path: Path,
    observation_case: str,
    observation: GitCommandResult,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(repository, observation)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"

    result = await _authorize_current_push(service, binding)

    assert observation_case in {
        "missing",
        "deleted",
        "divergent",
        "plural",
        "malformed",
        "inaccessible",
    }
    assert result.state == "blocked"
    assert result.handle is None
    assert result.review is None
    assert result.outcome is None
    assert owner.snapshot(binding).push_candidate is not None
    assert service._push_review_snapshots == {}
    assert len(runner.network_calls) == 1
    assert not any(
        os.fsdecode(argument) == "push"
        for argv, _call in runner.network_calls
        for argument in argv
    )
    assert list((tmp_path / "network-contexts").iterdir()) == []
    assert "REMOTE_SECRET_CANARY" not in repr(
        (result, service._push_review_snapshots, owner.snapshot(binding))
    )


@pytest.mark.asyncio
async def test_push_preflight_context_cleanup_failure_is_retained_and_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    original_close = git_network.NetworkGitExecutionContext.close
    close_calls = 0

    def fail_first_close(context) -> bool:
        nonlocal close_calls
        close_calls += 1
        if close_calls == 1:
            return False
        return original_close(context)

    monkeypatch.setattr(
        git_network.NetworkGitExecutionContext,
        "close",
        fail_first_close,
    )
    assert (await service.start_push_review(binding)).state == "ready"

    blocked = await _authorize_current_push(service, binding)

    assert blocked.state == "blocked"
    assert len(service._pending_push_contexts) == 1
    assert list((tmp_path / "network-contexts").iterdir())

    assert (await service.start_push_review(binding)).state == "ready"

    assert close_calls >= 2
    assert service._pending_push_contexts == set()
    assert list((tmp_path / "network-contexts").iterdir()) == []
    assert _cancel_current_push(service, binding)


@pytest.mark.asyncio
async def test_push_preflight_public_waiter_cancellation_retains_one_cycle_and_context(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _BlockingPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    waiter = _authorize_current_push(service, binding)
    retained = service.retained_push_operation(binding)
    assert retained is not None
    await asyncio.wait_for(runner.started.wait(), timeout=1)
    context_paths = list((tmp_path / "network-contexts").iterdir())
    assert len(context_paths) == 1

    waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiter

    assert owner.mutation_active(binding)
    assert service.retained_push_operation(binding) is retained
    with pytest.raises(git_service.GitMutationAdmissionError) as duplicate:
        _authorize_current_push(service, binding)
    assert duplicate.value.reason == "stale_binding"
    assert len(runner.network_calls) == 1
    assert owner.record_change(
        binding,
        SessionChange("modified", "while-query.md"),
    )

    runner.release_query.set()
    result = await asyncio.wait_for(retained.wait(), timeout=1)

    assert result.state == "review"
    assert len(runner.network_calls) == 1
    assert list((tmp_path / "network-contexts").iterdir()) == context_paths
    assert _cancel_current_push(service, binding)
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_review_rebinding_releases_stale_context_and_refuses_candidate_less_start(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    reviewed = await _authorize_current_push(service, binding)
    assert reviewed.handle is not None
    stale = service._push_review_snapshots[reviewed.handle]
    assert list((tmp_path / "network-contexts").iterdir())

    rebound = owner.select_root(tmp_path / "other-notes")
    with pytest.raises(git_service.GitMutationAdmissionError) as refused:
        service.start_push_review(rebound)

    assert refused.value.reason == "stale_binding"
    assert owner.snapshot(rebound).push_candidate is None
    assert list((tmp_path / "network-contexts").iterdir()) == []
    assert reviewed.handle not in service._push_review_snapshots
    assert not owner._destination_authorization_matches(
        stale.policy.owner_capture,
        stale.authorization,
    )
    assert (
        owner._consume_push_review(
            reviewed.handle,
            operation_id=stale.operation_id,
            network_context=stale.context,
        )
        is None
    )
    assert service.retained_push_operation(rebound) is None
    assert not owner.mutation_active(rebound)


@pytest.mark.asyncio
async def test_push_authorization_builds_default_context_after_local_exec_proof(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            b"b" * 40 + b"\t" + BRANCH_REF.encode() + b"\n",
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    git_executable, _git_exec_path = _test_git_installation()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=str(git_executable),
        environment={},
    )
    assert (await service.start_push_review(binding)).state == "ready"

    result = await _authorize_current_push(service, binding)

    assert result.state == "review"
    assert result.handle is not None
    assert len(runner.network_calls) == 1
    assert _cancel_current_push(service, binding)


@pytest.mark.asyncio
async def test_push_preflight_cancel_during_local_revalidation_never_authorizes(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _BlockingPushRevalidationRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    before = owner.snapshot(binding)
    waiter = _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.revalidation_started.wait(), timeout=1)

    assert _cancel_current_push(service, binding)
    result = await asyncio.wait_for(waiter, timeout=1)

    after = owner.snapshot(binding)
    assert result.state == "cancelled"
    assert (
        after.destination_authorization_epoch
        == before.destination_authorization_epoch
    )
    assert after.push_candidate is not None
    assert not owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_preflight_cancel_retains_gate_and_context_until_tree_settles(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _RetainedCancelledPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    waiter = _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.started.wait(), timeout=1)

    assert _cancel_current_push(service, binding)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)
    assert owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir())

    runner.tree_settled.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "cancelled"
    assert runner.released
    assert not owner.mutation_active(binding)
    assert owner.snapshot(binding).push_candidate is not None
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_preflight_cancel_during_retained_drain_never_issues_review(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _NormallyReturnedRetainedPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    waiter = _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)

    assert _cancel_current_push(service, binding)
    assert owner.mutation_active(binding)
    runner.tree_settled.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "cancelled"
    assert result.handle is None
    assert service._push_review_snapshots == {}
    assert runner.released
    assert owner.snapshot(binding).push_candidate is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["claim", "settle"])
async def test_push_preflight_ownership_failure_quarantines_token_and_context(
    tmp_path: Path,
    failure: str,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _OwnershipFailurePushPreflightRunner(repository, failure)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    assert (await service.start_push_review(binding)).state == "ready"
    _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.failure_observed.wait(), timeout=1)
    await asyncio.sleep(0)

    assert service._unsettled_push_preflight is not None
    assert service._unsettled_push_preflight.retained_child is runner.token
    assert owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir())
    assert not runner.released
    assert "REMOTE_HELPER_SECRET_CANARY" not in repr(
        (
            service._unsettled_push_preflight,
            service.retained_push_operation(binding),
            service._push_preflight_cycle,
        )
    )

    cycle = service._push_preflight_cycle
    waiter = service._push_preflight_waiter
    assert cycle is not None and waiter is not None
    assert cycle.get_coro().cr_frame is not None
    assert "REMOTE_HELPER_SECRET_CANARY" not in repr(
        cycle.get_coro().cr_frame.f_locals
    )
    cycle.cancel()
    waiter.cancel()
    await asyncio.gather(cycle, waiter, return_exceptions=True)
    await asyncio.sleep(0)

    assert owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir())


@pytest.mark.asyncio
async def test_push_preflight_confirmed_shutdown_releases_quarantine(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _OwnershipFailurePushPreflightRunner(repository, "settle")
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    owner.attach_git_service(service)
    assert (await service.start_push_review(binding)).state == "ready"
    _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.failure_observed.wait(), timeout=1)
    await asyncio.sleep(0)
    assert service._unsettled_push_preflight is not None

    await asyncio.wait_for(owner.shutdown_async(), timeout=1)

    assert runner.released
    assert service._unsettled_push_preflight is None
    assert not owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir()) == []
    assert service.retained_push_operation(binding) is None


@pytest.mark.asyncio
async def test_push_preflight_delayed_settlement_cannot_hang_shutdown_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _DelayedSettlementPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    owner.attach_git_service(service)
    monkeypatch.setattr(
        git_service,
        "_PUSH_QUARANTINE_TRANSFER_TIMEOUT_SECONDS",
        0.01,
    )
    assert (await service.start_push_review(binding)).state == "ready"
    _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.settlement_started.wait(), timeout=1)

    await asyncio.wait_for(owner.shutdown_async(), timeout=1)

    assert runner.settlement_attempts >= 3
    assert runner.released
    assert service._unsettled_push_preflight is None
    assert not owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
async def test_push_preflight_shutdown_waits_for_retained_tree_before_cleanup(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _NormallyReturnedRetainedPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    owner.attach_git_service(service)
    assert (await service.start_push_review(binding)).state == "ready"
    _authorize_current_push(service, binding)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)

    shutdown = asyncio.create_task(owner.shutdown_async())
    await asyncio.wait_for(runner.shutdown_called.wait(), timeout=1)
    await asyncio.sleep(0)

    assert not shutdown.done()
    assert owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir())

    runner.tree_settled.set()
    await asyncio.wait_for(shutdown, timeout=1)

    assert runner.released
    assert not owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir()) == []
    assert service.retained_push_operation(binding) is None


@pytest.mark.asyncio
async def test_push_preflight_shutdown_is_bounded_when_tree_remains_unproved(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _UnprovedShutdownPushPreflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    owner.attach_git_service(service)
    assert (await service.start_push_review(binding)).state == "ready"
    retained = service.retained_push_operation(binding)
    assert retained is not None
    _authorize_current_push(service, binding)
    retained = service.retained_push_operation(binding)
    assert retained is not None
    await asyncio.wait_for(runner.started.wait(), timeout=1)

    shutdown = asyncio.create_task(owner.shutdown_async())
    await asyncio.wait_for(runner.shutdown_called.wait(), timeout=1)
    await asyncio.wait_for(shutdown, timeout=1)

    assert not retained.settled
    assert not owner.mutation_active(binding)
    assert owner.admit_mutation(binding).reason == "shutdown"
    assert list((tmp_path / "network-contexts").iterdir())
    assert service._unsettled_push_preflight is not None
    assert service._unsettled_push_preflight.retained_child is runner.token
    assert not runner.released

    cycle = service._push_preflight_cycle
    waiter = service._push_preflight_waiter
    assert cycle is not None and waiter is not None
    cycle.cancel()
    waiter.cancel()
    await asyncio.gather(cycle, waiter, return_exceptions=True)
    await asyncio.sleep(0)

    assert cycle.done()
    assert waiter.done()
    assert service._push_preflight_cycle is None
    assert not owner.mutation_active(binding)
    assert list((tmp_path / "network-contexts").iterdir())


@pytest.mark.asyncio
async def test_already_published_clears_only_candidate_without_push_or_review(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    candidate_line = b"d" * 40 + b"\t" + BRANCH_REF.encode() + b"\n"
    runner = _ControlledPushPreflightRunner(
        repository,
        GitCommandResult(
            0,
            candidate_line,
            b"",
            owned_process_tree=True,
            containment_proved=True,
        ),
    )
    factory = _network_factory(tmp_path)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=factory,
    )
    assert (await service.start_push_review(binding)).state == "ready"

    result = await _authorize_current_push(service, binding)

    assert result.state == "already_published"
    assert result.outcome is not None
    assert result.outcome.state == "already_published"
    assert result.handle is None
    assert result.review is None
    assert owner.snapshot(binding).push_candidate is None
    assert len(runner.network_calls) == 1
    assert all(
        "push" not in tuple(os.fsdecode(argument) for argument in argv)
        for argv, _call in runner.network_calls
    )
    assert list((tmp_path / "network-contexts").iterdir()) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("indeterminate", [False, True])
async def test_lfs_or_indeterminate_candidate_tree_attributes_block_locally(
    tmp_path: Path,
    indeterminate: bool,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(
        repository,
        lfs_paths=frozenset({b"note.md"}) if not indeterminate else frozenset(),
        malformed_attributes=indeterminate,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert review.authorization is None
    assert service.authorize_push_destination(binding) is None
    assert not any(
        {"ls-remote", "push"}.intersection(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )


@pytest.mark.asyncio
async def test_confirm_revalidation_revokes_authorization_when_lfs_policy_changes(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    review = await service.review_push_destination(binding)
    assert review.state == "ready"
    policy = service._push_destination_policy
    assert policy is not None
    authorization = service.authorize_push_destination(binding)
    assert authorization is not None
    epoch = owner.snapshot(binding).destination_authorization_epoch
    runner.lfs_paths = frozenset({b"note.md"})

    valid = await service.revalidate_push_destination(
        binding,
        authorization,
    )

    assert valid is False
    assert owner.snapshot(binding).destination_authorization_epoch > epoch
    assert not owner._destination_authorization_matches(
        policy.owner_capture,
        authorization,
    )


@pytest.mark.asyncio
async def test_confirm_revalidation_revokes_authorization_on_format_mismatch(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    review = await service.review_push_destination(binding)
    authorization = service.authorize_push_destination(binding)
    assert review.state == "ready"
    assert authorization is not None
    epoch = owner.snapshot(binding).destination_authorization_epoch
    runner.object_format = "sha256"
    first_revalidation_call = len(runner.calls)

    valid = await service.revalidate_push_destination(binding, authorization)

    assert valid is False
    assert owner.snapshot(binding).destination_authorization_epoch > epoch
    assert not any(
        {"ls-remote", "push"}.intersection(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls[
            first_revalidation_call:
        ]
    )


@pytest.mark.asyncio
async def test_destination_authorization_reuses_only_unchanged_revalidation(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )
    review = await service.review_push_destination(binding)
    assert review.state == "ready"
    authorization = service.authorize_push_destination(binding)
    assert authorization is not None
    before = owner.snapshot(binding)

    valid = await service.revalidate_push_destination(binding, authorization)

    assert valid is True
    assert service.authorize_push_destination(binding) is authorization
    after = owner.snapshot(binding)
    assert after.destination_policy_generation == before.destination_policy_generation
    assert (
        after.destination_authorization_epoch
        == before.destination_authorization_epoch
    )


@pytest.mark.asyncio
async def test_destination_configuration_source_change_during_proof_blocks(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(
        repository,
        change_config_during_read=True,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert service.authorize_push_destination(binding) is None
    assert runner.config_reads == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("origin_kind", "key", "value"),
    [
        ("relative_local", "credential.helper", "!touch SHOULD_NOT_RUN"),
        ("absolute_local", "core.sshcommand", "touch SHOULD_NOT_RUN"),
        ("absolute_worktree", "credential.helper", "!touch SHOULD_NOT_RUN"),
        ("external", "core.sshcommand", "touch SHOULD_NOT_RUN"),
    ],
)
async def test_unknown_configuration_scope_cannot_bypass_local_helper_policy(
    tmp_path: Path,
    origin_kind: str,
    key: str,
    value: str,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(repository)
    if origin_kind == "relative_local":
        origin = b".git/config"
    elif origin_kind == "absolute_local":
        origin = os.fsencode(Path(repository.git_common_dir) / "config")
    elif origin_kind == "absolute_worktree":
        worktree_config = Path(repository.git_dir) / "config.worktree"
        worktree_config.write_text("[core]\n", encoding="utf-8")
        origin = os.fsencode(worktree_config)
    else:
        external_config = tmp_path / "external.gitconfig"
        external_config.write_text("[core]\n", encoding="utf-8")
        origin = os.fsencode(external_config)
    runner.config_payload += (
        b"unknown\0file:"
        + origin
        + b"\0"
        + key.encode()
        + b"\n"
        + value.encode()
        + b"\0"
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert service.authorize_push_destination(binding) is None


def _unknown_scope_config_record(
    origin: Path,
    key: str,
    value: str,
) -> bytes:
    return (
        b"unknown\0file:"
        + os.fsencode(origin)
        + b"\0"
        + key.encode()
        + b"\n"
        + value.encode()
        + b"\0"
    )


@pytest.mark.asyncio
async def test_unknown_scope_home_fallback_helper_blocks_when_xdg_is_set(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    home_fallback = home / ".config" / "git" / "config"
    home_fallback.parent.mkdir(parents=True)
    home_fallback.write_text("[credential]\n", encoding="utf-8")
    xdg_config = tmp_path / "xdg" / "git" / "config"
    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        home_fallback,
        "credential.helper",
        "!touch SHOULD_NOT_RUN",
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(xdg_config.parent.parent),
        },
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert service.authorize_push_destination(binding) is None


@pytest.mark.asyncio
async def test_unknown_scope_xdg_global_config_is_admitted_and_fingerprinted(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    xdg_config = tmp_path / "xdg" / "git" / "config"
    xdg_config.parent.mkdir(parents=True)
    xdg_config.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    base_payload = runner.config_payload
    runner.config_payload = base_payload + _unknown_scope_config_record(
        xdg_config,
        "credential.helper",
        "",
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={
            "HOME": str(home),
            "XDG_CONFIG_HOME": str(xdg_config.parent.parent),
        },
    )

    first = await service.review_push_destination(binding)
    first_policy = service._push_destination_policy
    runner.config_payload = base_payload + _unknown_scope_config_record(
        xdg_config,
        "credential.helper",
        helper_name,
    )
    second = await service.review_push_destination(binding)
    second_policy = service._push_destination_policy

    assert first.state == "ready"
    assert second.state == "ready"
    assert first_policy is not None
    assert second_policy is not None
    assert (
        first_policy.configuration.configuration_fingerprint
        != second_policy.configuration.configuration_fingerprint
    )


@pytest.mark.asyncio
async def test_unknown_scope_home_fallback_is_global_when_xdg_is_unset(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    home_fallback = home / ".config" / "git" / "config"
    home_fallback.parent.mkdir(parents=True)
    home_fallback.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        home_fallback,
        "credential.helper",
        helper_name,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"HOME": str(home)},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "ready"
    assert review.authorization is not None


@pytest.mark.asyncio
async def test_network_context_policy_retains_exact_approved_helper_and_objects(
    tmp_path: Path,
) -> None:
    helper_name = _approved_helper_name()
    if helper_name is None:
        pytest.skip("No guarded helper allowlist is defined for this platform")
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    global_config = home / ".config" / "git" / "config"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        global_config,
        "credential.helper",
        helper_name,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"HOME": str(home)},
    )

    review = await service.review_push_destination(binding)
    policy = service._push_destination_policy

    assert review.state == "ready"
    assert policy is not None
    assert policy.source_objects.identity_fingerprint
    assert helper_name not in repr(policy.network_configuration)
    assert repository.git_common_dir not in repr(policy.source_objects)


@pytest.mark.asyncio
async def test_network_context_policy_blocks_global_shell_credential_helper(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    global_config = home / ".config" / "git" / "config"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        global_config,
        "credential.helper",
        "!touch SHOULD_NOT_RUN",
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"HOME": str(home)},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert service.authorize_push_destination(binding) is None
    assert not any(
        {"credential", "ls-remote", "push"}.intersection(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )


@pytest.mark.asyncio
async def test_network_context_policy_captures_and_blocks_scoped_use_http_path(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    global_config = home / ".config" / "git" / "config"
    global_config.parent.mkdir(parents=True)
    global_config.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        global_config,
        "credential.https://push.example.test/team/notes.git.usehttppath",
        "true",
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={"HOME": str(home)},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert service.authorize_push_destination(binding) is None
    assert runner.config_reads == 2
    assert not any(
        {"credential", "ls-remote", "push"}.intersection(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("path_count", [1, 1000])
async def test_candidate_tree_lfs_proof_batches_paths_with_bounded_commands(
    tmp_path: Path,
    path_count: int,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    paths = tuple(f"note-{index}.md".encode() for index in range(path_count))
    runner = _ControlledLocalProofRunner(repository, paths=paths)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "ready"
    assert len(runner.calls) <= 9
    assert sum(
        "--exec-path"
        in tuple(os.fsdecode(argument) for argument in argv)
        for argv, _environment, _stdin in runner.calls
    ) == 1
    attribute_calls = [
        (argv, stdin)
        for argv, _environment, stdin in runner.calls
        if "check-attr" in tuple(os.fsdecode(item) for item in argv)
    ]
    assert len(attribute_calls) == 1
    argv, stdin = attribute_calls[0]
    assert {"--cached", "--stdin", "-z"}.issubset(
        os.fsdecode(item) for item in argv
    )
    assert stdin is not None
    assert tuple(path for path in stdin.split(b"\0") if path) == paths


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    executable = shutil.which("git")
    assert executable is not None
    return subprocess.run(
        (executable, *arguments),
        cwd=root,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def _real_candidate_repository(
    tmp_path: Path,
    *,
    lfs: bool = False,
    object_format: str = "sha1",
) -> tuple[Path, str, str]:
    root = tmp_path / (
        f"real-{object_format}-lfs" if lfs else f"real-{object_format}"
    )
    root.mkdir()
    init_arguments = ["init", "-b", "main"]
    if object_format == "sha256":
        init_arguments.insert(1, "--object-format=sha256")
    _git(root, *init_arguments)
    if lfs:
        (root / ".gitattributes").write_text(
            "*.md filter=lfs diff=lfs merge=lfs -text\n",
            encoding="utf-8",
        )
    (root / "note.md").write_text("parent\n", encoding="utf-8")
    _git(root, "add", "--", ".gitattributes" if lfs else "note.md")
    if lfs:
        _git(root, "add", "--", "note.md")
    _git(
        root,
        "-c",
        "user.name=Test Author",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "parent",
    )
    parent_oid = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    (root / "note.md").write_text("candidate\n", encoding="utf-8")
    _git(root, "add", "--", "note.md")
    _git(
        root,
        "-c",
        "user.name=Test Author",
        "-c",
        "user.email=test@example.test",
        "commit",
        "-m",
        "candidate",
    )
    candidate_oid = _git(root, "rev-parse", "HEAD").stdout.decode().strip()
    _git(root, "config", "branch.main.remote", "origin")
    _git(root, "config", "branch.main.merge", BRANCH_REF)
    _git(
        root,
        "config",
        "remote.origin.url",
        "https://push.example.test/team/notes.git",
    )
    return root, parent_oid, candidate_oid


@pytest.mark.asyncio
@pytest.mark.parametrize(("lfs", "expected"), [(False, "ready"), (True, "blocked")])
async def test_real_candidate_tree_destination_and_lfs_proof_is_local_only(
    tmp_path: Path,
    lfs: bool,
    expected: str,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(
        tmp_path,
        lfs=lfs,
    )
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    runner = _RecordingAsyncGitRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    diagnostic = "\n".join(
        f"{index}: rc={result.returncode} argv={command!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
        for index, (command, result) in enumerate(runner.calls)
    )
    assert review.state == expected, diagnostic
    assert _git(root, "status", "--porcelain").stdout == b""
    await service.shutdown()


@pytest.mark.asyncio
async def test_real_sha256_lfs_proof_reaches_exact_tree_attribute_check(
    tmp_path: Path,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(
        tmp_path,
        lfs=True,
        object_format="sha256",
    )
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    runner = _RecordingAsyncGitRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    read_tree = [
        result
        for command, result in runner.calls
        if "read-tree" in command
    ]
    check_attr = [
        result
        for command, result in runner.calls
        if "check-attr" in command
    ]
    diagnostic = "\n".join(
        f"{index}: rc={result.returncode} argv={command!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
        for index, (command, result) in enumerate(runner.calls)
    )
    assert review.state == "blocked", diagnostic
    assert len(read_tree) == 1 and read_tree[0].returncode == 0, diagnostic
    assert len(check_attr) == 1 and check_attr[0].returncode == 0, diagnostic
    assert len(parent_oid) == len(candidate_oid) == 64
    assert _git(root, "status", "--porcelain").stdout == b""
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("attribute_source", ["info", "global"])
@pytest.mark.parametrize(
    ("candidate_lfs", "external_rule", "expected"),
    [
        (True, "*.md -filter\n", "blocked"),
        (False, "*.md filter=lfs\n", "ready"),
    ],
)
async def test_exact_candidate_tree_lfs_ignores_external_attribute_sources(
    tmp_path: Path,
    attribute_source: str,
    candidate_lfs: bool,
    external_rule: str,
    expected: str,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(
        tmp_path,
        lfs=candidate_lfs,
    )
    environment: dict[str, str] = {}
    if attribute_source == "info":
        info_attributes = root / ".git" / "info" / "attributes"
        info_attributes.write_text(external_rule, encoding="utf-8")
    else:
        config_home = tmp_path / "config-home"
        git_config_home = config_home / "git"
        git_config_home.mkdir(parents=True)
        (git_config_home / "attributes").write_text(
            external_rule,
            encoding="utf-8",
        )
        environment = {
            "HOME": str(tmp_path / "home"),
            "XDG_CONFIG_HOME": str(config_home),
        }
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    runner = _RecordingAsyncGitRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment=environment,
    )

    review = await service.review_push_destination(binding)

    diagnostic = "\n".join(
        f"{index}: rc={result.returncode} argv={command!r} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
        for index, (command, result) in enumerate(runner.calls)
    )
    assert review.state == expected, diagnostic
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("conditional", [False, True], ids=["include", "include-if"])
async def test_configuration_include_edge_aba_revokes_stale_authorization(
    tmp_path: Path,
    conditional: bool,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(tmp_path)
    git_dir = root / ".git"
    included_values = (
        '[branch "main"]\n'
        "\tremote = origin\n"
        f"\tmerge = {BRANCH_REF}\n"
        '[remote "origin"]\n'
        "\turl = https://push.example.test/team/notes.git\n"
    )
    (git_dir / "include-a.conf").write_text(
        included_values,
        encoding="utf-8",
    )
    (git_dir / "include-b.conf").write_text(
        included_values,
        encoding="utf-8",
    )

    def local_config(include_name: str) -> str:
        section = (
            '[includeIf "onbranch:main"]'
            if conditional
            else "[include]"
        )
        return f"{section}\n\tpath = {include_name}\n"

    config = git_dir / "config"
    config.write_text(local_config("include-a.conf"), encoding="utf-8")
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    service = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
    )
    initial = await service.review_push_destination(binding)
    initial_policy = service._push_destination_policy
    assert initial.state == "ready"
    assert initial_policy is not None
    authorization = service.authorize_push_destination(binding)
    assert authorization is not None

    replacement = git_dir / "config.next"
    replacement.write_text(
        local_config("include-b.conf"),
        encoding="utf-8",
    )
    replacement.replace(config)
    replacement.write_text(
        local_config("include-a.conf"),
        encoding="utf-8",
    )
    replacement.replace(config)

    valid = await service.revalidate_push_destination(binding, authorization)
    refreshed = await service.review_push_destination(binding)
    refreshed_policy = service._push_destination_policy

    assert valid is False
    assert refreshed.state == "ready"
    assert refreshed_policy is not None
    assert (
        refreshed_policy.configuration.configuration_fingerprint
        != initial_policy.configuration.configuration_fingerprint
    )
    assert not owner._destination_authorization_matches(
        initial_policy.owner_capture,
        authorization,
    )
    await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.asyncio
async def test_attribute_proof_rejects_unsafe_nonsticky_temp_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unsafe_parent = tmp_path / "unsafe"
    unsafe_parent.mkdir(mode=0o700)
    unsafe_parent.chmod(0o777)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(unsafe_parent))
    root, parent_oid, candidate_oid = _real_candidate_repository(
        unsafe_parent,
    )
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    service = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.parametrize("parent_mode", [0o700, 0o1777], ids=["owner", "sticky"])
@pytest.mark.asyncio
async def test_attribute_proof_accepts_safe_owner_or_sticky_temp_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_mode: int,
) -> None:
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir(mode=0o700)
    safe_parent.chmod(parent_mode)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(safe_parent))
    root, parent_oid, candidate_oid = _real_candidate_repository(
        safe_parent,
    )
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    service = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "ready"
    await service.shutdown()


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_private_proof_routine_validation_does_not_reread_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir(mode=0o700)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(safe_parent))
    _owner, _binding, repository = _candidate_owner(tmp_path)
    proof = git_service._create_private_push_proof_directory(repository)
    index_path = proof.reserve_index("index")
    index_path.write_bytes(b"sealed index")
    index_path.chmod(0o600)
    assert proof.capture_index()
    original_digest = git_service._bounded_file_descriptor_digest
    digest_calls = 0

    def count_digest(file_descriptor: int, expected_size: int) -> bytes | None:
        nonlocal digest_calls
        digest_calls += 1
        return original_digest(file_descriptor, expected_size)

    monkeypatch.setattr(
        git_service,
        "_bounded_file_descriptor_digest",
        count_digest,
    )

    assert proof.validate()
    assert digest_calls == 0

    metadata = index_path.stat()
    os.utime(
        index_path,
        ns=(metadata.st_atime_ns, metadata.st_mtime_ns + 1_000_000_000),
    )
    assert not proof.validate()
    assert digest_calls == 0


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.parametrize("error_type", [OSError, RuntimeError])
def test_safe_private_parent_iterator_falls_back_after_policy_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error_type: type[Exception],
) -> None:
    temp_parent = tmp_path / "temp"
    temp_parent.mkdir(mode=0o700)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(temp_parent))
    _owner, _binding, repository = _candidate_owner(tmp_path)
    worktree = Path(repository.worktree_root)
    repository_device = worktree.stat().st_dev
    checked: list[tuple[Path, int]] = []

    def check_parent(parent: Path, device: int) -> bool:
        checked.append((parent, device))
        if parent == temp_parent.resolve():
            raise error_type("parent policy failed")
        return True

    monkeypatch.setattr(git_service, "_hooks_parent_is_safe", check_parent)

    assert list(git_service._iter_safe_private_directory_parents(repository)) == [
        (worktree, tmp_path.resolve(), repository_device)
    ]
    assert checked == [
        (temp_parent.resolve(), repository_device),
        (tmp_path.resolve(), repository_device),
    ]


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
def test_safe_private_parent_iterator_deduplicates_canonical_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _owner, _binding, repository = _candidate_owner(tmp_path)
    worktree = Path(repository.worktree_root)
    aliased_parent = worktree / ".."
    monkeypatch.setattr(
        git_service.tempfile,
        "tempdir",
        str(aliased_parent),
    )
    repository_device = worktree.stat().st_dev
    checked: list[tuple[Path, int]] = []

    def check_parent(parent: Path, device: int) -> bool:
        checked.append((parent, device))
        return True

    monkeypatch.setattr(git_service, "_hooks_parent_is_safe", check_parent)

    assert list(git_service._iter_safe_private_directory_parents(repository)) == [
        (worktree, tmp_path.resolve(), repository_device)
    ]
    assert checked == [(tmp_path.resolve(), repository_device)]


@pytest.mark.skipif(os.name != "posix", reason="POSIX ownership policy")
@pytest.mark.asyncio
async def test_attribute_proof_directory_substitution_blocks_without_cleanup_follow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir(mode=0o700)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(safe_parent))
    root, parent_oid, candidate_oid = _real_candidate_repository(
        safe_parent,
        lfs=True,
    )
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    runner = _ReplacingAttributeProofRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert runner.replaced is True
    assert runner.replacement_marker is not None
    assert runner.replacement_marker.exists()
    assert not any(
        "check-attr" in command
        for command, _result in runner.calls
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_attribute_proof_rejects_directory_index_without_mutating_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    safe_parent = tmp_path / "safe"
    safe_parent.mkdir(mode=0o700)
    monkeypatch.setattr(git_service.tempfile, "tempdir", str(safe_parent))
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledLocalProofRunner(
        repository,
        index_directory=True,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    read_tree_environments = [
        environment
        for argv, environment, _stdin in runner.calls
        if "read-tree" in tuple(os.fsdecode(argument) for argument in argv)
    ]
    assert review.state == "blocked"
    assert len(read_tree_environments) == 1
    index_path = Path(read_tree_environments[0]["GIT_INDEX_FILE"])
    assert index_path.is_dir()
    assert stat.S_IMODE(index_path.stat().st_mode) == 0o700
    assert not any(
        "check-attr" in tuple(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("index_payload", "read_tree_returncode"),
    [(b"x" * 64, 0), (b"partial", 1)],
    ids=["oversized", "failed-partial"],
)
async def test_attribute_proof_rejected_index_leaves_no_proof_residue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    index_payload: bytes,
    read_tree_returncode: int,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    monkeypatch.setattr(
        git_service,
        "_LOCAL_PUSH_PROOF_FILE_LIMIT_BYTES",
        48,
    )
    runner = _ControlledLocalProofRunner(
        repository,
        index_payload=index_payload,
        read_tree_returncode=read_tree_returncode,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    review = await service.review_push_destination(binding)

    read_tree_environments = [
        environment
        for argv, environment, _stdin in runner.calls
        if "read-tree" in tuple(os.fsdecode(argument) for argument in argv)
    ]
    assert review.state == "blocked"
    assert len(read_tree_environments) == 1
    index_path = Path(read_tree_environments[0]["GIT_INDEX_FILE"])
    assert not index_path.parent.exists()
    await service.shutdown()


@pytest.mark.asyncio
async def test_attribute_proof_rejects_source_object_metadata_change(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    objects = Path(repository.git_common_dir) / "objects"
    original_mode = stat.S_IMODE(objects.stat().st_mode)
    runner = _ChangingObjectDirectoryModeRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
    )

    try:
        review = await service.review_push_destination(binding)
    finally:
        objects.chmod(original_mode)

    assert review.state == "blocked"
    assert runner.changed is True
    read_tree_environments = [
        environment
        for argv, environment, _stdin in runner.calls
        if "read-tree" in tuple(os.fsdecode(argument) for argument in argv)
    ]
    assert len(read_tree_environments) == 1
    index_path = Path(read_tree_environments[0]["GIT_INDEX_FILE"])
    assert not index_path.parent.exists()
    assert not any(
        "check-attr" in tuple(
            os.fsdecode(argument) for argument in argv
        )
        for argv, _environment, _stdin in runner.calls
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_production_local_destination_blocks_while_test_admission_only_proves(
    tmp_path: Path,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(tmp_path)
    local_bare = tmp_path / "remote.git"
    local_bare.mkdir()
    marker = local_bare / "not-contacted"
    marker.write_text("unchanged", encoding="utf-8")
    _git(root, "config", "remote.origin.url", str(local_bare))
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    production = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
    )

    blocked = await production.review_push_destination(binding)
    test_only = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
        transport_admission=(
            push_contracts._local_bare_transport_admission_for_tests()
        ),
    )
    admitted = await test_only.review_push_destination(binding)

    assert blocked.state == "blocked"
    assert admitted.state == "ready"
    assert marker.read_text(encoding="utf-8") == "unchanged"
    await production.shutdown()
    await test_only.shutdown()


@pytest.mark.asyncio
async def test_destination_configuration_value_aba_changes_policy_identity(
    tmp_path: Path,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(tmp_path)
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    service = FileNotesGitService(
        owner,
        git_executable=executable,
        environment={},
    )
    first = await service.review_push_destination(binding)
    assert first.state == "ready"
    first_policy = service._push_destination_policy
    assert first_policy is not None
    authorization = service.authorize_push_destination(binding)
    assert authorization is not None
    first_snapshot = owner.snapshot(binding)

    _git(
        root,
        "config",
        "remote.origin.url",
        "https://changed.example.test/team/notes.git",
    )
    changed = await service.review_push_destination(binding)
    assert changed.state == "ready"
    _git(
        root,
        "config",
        "remote.origin.url",
        "https://push.example.test/team/notes.git",
    )
    restored = await service.review_push_destination(binding)
    restored_policy = service._push_destination_policy

    assert restored.state == "ready"
    assert restored_policy is not None
    assert (
        restored_policy.configuration.configuration_fingerprint
        != first_policy.configuration.configuration_fingerprint
    )
    final_snapshot = owner.snapshot(binding)
    assert (
        final_snapshot.destination_policy_generation
        > first_snapshot.destination_policy_generation
    )
    assert (
        final_snapshot.destination_authorization_epoch
        > first_snapshot.destination_authorization_epoch
    )
    assert not owner._destination_authorization_matches(
        first_policy.owner_capture,
        authorization,
    )
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("credential.helper", "!touch SHOULD_NOT_RUN"),
        ("core.sshCommand", "touch SHOULD_NOT_RUN"),
    ],
)
async def test_repository_executable_configuration_blocks_without_invocation(
    tmp_path: Path,
    key: str,
    value: str,
) -> None:
    root, parent_oid, candidate_oid = _real_candidate_repository(tmp_path)
    _git(root, "config", key, value)
    owner, binding, _repository = _owner_for_candidate(
        root,
        parent_oid,
        candidate_oid,
    )
    executable = shutil.which("git")
    assert executable is not None
    runner = _RecordingAsyncGitRunner()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable=executable,
        environment={},
    )

    review = await service.review_push_destination(binding)

    assert review.state == "blocked"
    assert not (root / "SHOULD_NOT_RUN").exists()
    assert all(
        not {"credential", "ssh", "ls-remote", "push"}.intersection(command)
        for command, _result in runner.calls
    )
    await service.shutdown()


def _settled_network_result(
    returncode: int,
    stdout: bytes,
    *,
    timed_out: bool = False,
    termination_uncertain: bool = False,
    output_overflow: bool = False,
    containment_proved: bool = True,
) -> GitCommandResult:
    return GitCommandResult(
        returncode,
        stdout,
        b"REMOTE_SECRET_CANARY",
        timed_out=timed_out,
        termination_uncertain=termination_uncertain,
        output_overflow=output_overflow,
        owned_process_tree=True,
        containment_proved=containment_proved,
    )


def _remote_observation(oid: str) -> GitCommandResult:
    return _settled_network_result(
        0,
        oid.encode("ascii") + b"\t" + BRANCH_REF.encode("ascii") + b"\n",
    )


def _accepted_push_result() -> GitCommandResult:
    return _settled_network_result(
        0,
        b" \t"
        + b"d" * 40
        + b":"
        + BRANCH_REF.encode("ascii")
        + b"\tb..d\n",
    )


async def _prepare_exact_push_review(
    service: FileNotesGitService,
    binding: SessionBinding,
) -> git_service.PushPreflightResult:
    assert (await service.start_push_review(binding)).state == "ready"
    reviewed = await _authorize_current_push(service, binding)
    assert reviewed.state == "review"
    assert reviewed.handle is not None
    return reviewed


@pytest.mark.asyncio
async def test_workspace_rehydrate_retained_push_phases_keep_exact_candidate(
    tmp_path: Path,
) -> None:
    """Losing phase-owned identity would let UI bind an old op to a new commit."""
    owner, binding, repository = _candidate_owner(tmp_path)
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )

    assert (await service.start_push_review(binding)).state == "ready"
    local_proof = _current_push_operation(service, binding)
    assert local_proof.candidate == candidate

    reviewed = await service.authorize_and_check_push(binding, local_proof)
    assert reviewed.handle is not None
    preflight = _current_push_operation(service, binding)
    assert preflight.candidate == candidate

    result = await service.start_push(binding, reviewed.handle)
    push = _current_push_operation(service, binding)
    assert result.state == "succeeded"
    assert push.candidate == candidate
    assert owner.snapshot(binding).push_candidate is None


@pytest.mark.asyncio
async def test_workspace_rehydrate_recovery_uses_retained_candidate_evidence(
    tmp_path: Path,
) -> None:
    """Recovery identity must come from frozen evidence, not live UI state."""
    owner, binding, repository = _candidate_owner(tmp_path)
    candidate = owner.snapshot(binding).push_candidate
    assert candidate is not None
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"
    uncertain = _current_push_operation(service, binding)
    assert uncertain.candidate == candidate
    evidence = service._uncertain_push
    assert evidence is not None
    assert owner.clear_push_candidate(evidence.candidate_capture)
    recovery_snapshot = owner.snapshot(binding)
    assert recovery_snapshot.push_candidate is None
    assert recovery_snapshot.push_recovery is not None
    assert recovery_snapshot.push_recovery_candidate == candidate

    recovered = await service.check_push_again(binding, uncertain)
    recovery = _current_push_operation(service, binding)

    assert recovered.state == "succeeded"
    assert recovery.kind == "recovery"
    assert recovery.candidate == candidate


@pytest.mark.asyncio
async def test_confirm_consumes_review_once_and_exact_push_postflight_succeeds(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert owner.record_change(
        binding,
        SessionChange("modified", "later-local-note.md"),
    )

    waiter = service.start_push(binding, reviewed.handle)
    operation = _current_push_operation(service, binding)
    result = await waiter

    assert operation.kind == "push"
    assert operation.child_started
    assert result.state == "succeeded"
    assert result.outcome is not None
    assert result.outcome.state == "succeeded"
    assert owner.snapshot(binding).push_candidate is None
    assert not owner.mutation_active(binding)
    network_commands = [
        tuple(os.fsdecode(argument) for argument in argv)
        for argv, _kwargs in runner.network_calls
    ]
    assert [
        "push" if "push" in command else "ls-remote"
        for command in network_commands
    ] == ["ls-remote", "ls-remote", "push", "ls-remote"]
    assert network_commands[2][-2:] == (
        "https://push.example.test/team/notes.git",
        f"{'d' * 40}:{BRANCH_REF}",
    )
    assert f"--force-with-lease={BRANCH_REF}:{'b' * 40}" in network_commands[2]
    assert [call[1]["timeout"] for call in runner.network_calls] == [
        30.0,
        30.0,
        60.0,
        30.0,
    ]
    assert "REMOTE_SECRET_CANARY" not in repr(result)
    with pytest.raises(
        git_service.GitMutationAdmissionError,
        match="invalid or already consumed",
    ):
        service.start_push(binding, reviewed.handle)


@pytest.mark.asyncio
async def test_confirm_cancel_before_actual_push_spawn_sends_no_update(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    barrier = _PushSpawnBarrier()
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
        before_push_spawn=barrier,
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    waiter = service.start_push(binding, reviewed.handle)
    operation = _current_push_operation(service, binding)
    await asyncio.wait_for(barrier.entered.wait(), timeout=1)

    assert not operation.child_started
    assert service.cancel_push(binding, operation)
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "cancelled"
    assert owner.snapshot(binding).push_candidate is not None
    assert not owner.mutation_active(binding)
    assert not any(
        "push" in tuple(os.fsdecode(argument) for argument in argv)
        for argv, _kwargs in runner.network_calls
    )


@pytest.mark.asyncio
async def test_confirm_cancel_settles_retained_final_query_before_release(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _RetainedCancelledConfirmQueryRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    waiter = service.start_push(binding, reviewed.handle)
    operation = _current_push_operation(service, binding)
    await asyncio.wait_for(runner.started.wait(), timeout=1)

    assert service.cancel_push(binding, operation)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)

    assert owner.mutation_active(binding)
    assert not waiter.done()
    assert not runner.released
    runner.tree_settled.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "cancelled"
    assert runner.released
    assert not owner.mutation_active(binding)
    assert owner.snapshot(binding).push_candidate is not None
    assert len(runner.network_calls) == 2
    assert not any(
        "push" in tuple(os.fsdecode(argument) for argument in argv)
        for argv, _kwargs in runner.network_calls
    )


@pytest.mark.asyncio
async def test_uncertain_push_postflight_cancel_settles_retained_child(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _RetainedCancelledPostflightRunner(repository)
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    waiter = service.start_push(binding, reviewed.handle)
    await asyncio.wait_for(runner.started.wait(), timeout=1)
    await asyncio.wait_for(runner.claimed.wait(), timeout=1)

    assert owner.mutation_active(binding)
    assert not waiter.done()
    assert not runner.released
    runner.tree_settled.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "uncertain"
    assert result.outcome is not None
    assert result.outcome.state == "uncertain"
    assert runner.released
    assert owner.mutation_active(binding)
    snapshot = owner.snapshot(binding)
    assert snapshot.push_candidate is not None
    assert snapshot.push_recovery is not None
    assert snapshot.push_recovery_available
    assert len(runner.network_calls) == 4
    await service.shutdown()
    assert not owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_exact_push_cancel_is_rejected_after_actual_child_spawn(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _BlockingExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    waiter = service.start_push(binding, reviewed.handle)
    operation = _current_push_operation(service, binding)
    await asyncio.wait_for(runner.push_started.wait(), timeout=1)

    assert operation.child_started
    assert service.cancel_push(binding, operation) is False
    assert owner.mutation_active(binding)
    runner.release_push.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "succeeded"


@pytest.mark.asyncio
async def test_exact_push_post_spawn_head_drift_preserves_result_and_newer_owner_state(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _BlockingExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    waiter = service.start_push(binding, reviewed.handle)
    operation = _current_push_operation(service, binding)
    await asyncio.wait_for(runner.push_started.wait(), timeout=1)
    before_drift = owner.snapshot(binding)
    availability = before_drift.push_candidate
    assert availability is not None

    (Path(repository.git_dir) / "HEAD").write_text(
        "ref: refs/heads/other\n",
        encoding="utf-8",
    )
    assert (
        owner._capture_push_candidate_after_fresh_proof(
            binding,
            candidate_generation=availability.generation,
            repository=repository,
            head=HeadIdentity.attached("refs/heads/other", "f" * 40),
            sole_parent_oid="b" * 40,
        )
        is None
    )
    assert owner.record_change(
        binding,
        SessionChange("modified", "after-spawn.md"),
    )
    drifted = owner.snapshot(binding)
    assert drifted.push_candidate is None
    assert drifted.push_candidate_generation > availability.generation
    status_generation = owner.next_status_generation(binding)
    assert status_generation is not None
    newer_status = SessionGitStatus(
        binding_generation=binding.generation,
        status_generation=status_generation,
        state="ready",
        repository=repository,
        head=HeadIdentity.attached("refs/heads/other", "f" * 40),
    )
    assert not owner.publish_status(binding, newer_status)

    runner.release_push.set()
    result = await asyncio.wait_for(waiter, timeout=1)

    assert result.state == "succeeded"
    assert await operation.wait() == result
    settled = owner.snapshot(binding)
    assert settled.push_candidate is None
    assert settled.push_candidate_generation == drifted.push_candidate_generation
    assert tuple(change.change.relative_path for change in settled.changes) == (
        "after-spawn.md",
    )
    assert owner.publish_status(binding, newer_status)
    published = owner.snapshot(binding)
    assert published.git_status == newer_status
    assert published.push_candidate_generation == drifted.push_candidate_generation
    assert await operation.wait() == result


@pytest.mark.asyncio
async def test_exact_push_launch_failure_is_blocked_not_uncertain(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
        launch_error=True,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    result = await service.start_push(binding, reviewed.handle)

    assert result.state == "blocked"
    assert result.outcome is None
    assert owner.snapshot(binding).push_candidate is not None
    assert not owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_exact_push_settled_result_without_spawn_callback_is_blocked(
    tmp_path: Path,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _NoSpawnCallbackExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    result = await service.start_push(binding, reviewed.handle)

    assert result.state == "blocked"
    assert result.outcome is None
    assert owner.snapshot(binding).push_candidate is not None
    assert not owner.mutation_active(binding)
    assert len(runner.network_calls) == 3
    assert len(runner.observations) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("push_result", "postflight", "expected"),
    [
        (_accepted_push_result(), _remote_observation("d" * 40), "succeeded"),
        (
            _settled_network_result(
                1,
                b"!\t" + b"d" * 40 + b":" + BRANCH_REF.encode() + b"\t[rejected]\n",
            ),
            _remote_observation("b" * 40),
            "failed_no_update_observed",
        ),
        (
            _settled_network_result(1, b"", timed_out=True),
            _remote_observation("b" * 40),
            "uncertain",
        ),
        (_accepted_push_result(), _remote_observation("b" * 40), "uncertain"),
        (
            _settled_network_result(1, b"", containment_proved=False),
            _remote_observation("b" * 40),
            "uncertain",
        ),
        (
            _accepted_push_result(),
            _settled_network_result(1, b""),
            "uncertain",
        ),
        (
            _accepted_push_result(),
            _settled_network_result(0, b""),
            "uncertain",
        ),
        (
            _accepted_push_result(),
            _remote_observation("e" * 40),
            "uncertain",
        ),
    ],
    ids=[
        "accepted-candidate",
        "rejected-parent",
        "timeout-parent",
        "accepted-parent-contradiction",
        "unproved-descendants",
        "postflight-query-failure",
        "postflight-missing",
        "postflight-other",
    ],
)
async def test_exact_push_classifies_machine_result_and_postflight(
    tmp_path: Path,
    push_result: GitCommandResult,
    postflight: GitCommandResult,
    expected: str,
) -> None:
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            postflight,
        ),
        push_result=push_result,
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    result = await service.start_push(binding, reviewed.handle)

    assert result.state == expected
    assert result.outcome is not None
    assert result.outcome.state == expected
    assert (owner.snapshot(binding).push_candidate is None) is (
        expected == "succeeded"
    )
    assert "REMOTE_SECRET_CANARY" not in repr(result)


@pytest.mark.asyncio
async def test_uncertain_push_retains_only_query_recovery_and_exact_endpoint(
    tmp_path: Path,
) -> None:
    """Losing the frozen endpoint or retaining Confirm authority must fail."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    context_parent = tmp_path / "network-contexts"
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    push_result = await service.start_push(binding, reviewed.handle)

    assert push_result.state == "uncertain"
    assert owner.mutation_active(binding)
    snapshot = owner.snapshot(binding)
    assert snapshot.push_recovery is not None
    assert snapshot.push_recovery_available is True
    evidence = service._uncertain_push
    assert evidence is not None
    assert service._push_destination_policy is None
    assert service._push_authorization is None
    for reusable_authority in (
        "review",
        "review_handle",
        "authorization",
        "policy",
        "push_argv",
    ):
        assert not hasattr(evidence, reusable_authority)

    config_path = Path(repository.git_dir) / "config"
    config_path.write_text(
        "[branch \"main\"]\n"
        "\tremote = replacement\n"
        f"\tmerge = {BRANCH_REF}\n"
        "[remote \"replacement\"]\n"
        "\tpushurl = https://replacement.example.test/other.git\n",
        encoding="utf-8",
    )
    recovery = await service.check_push_again(
        binding,
        _current_push_operation(service, binding),
    )

    assert recovery.state == "succeeded"
    assert recovery.query_only
    assert "cause" in recovery.message
    assert not recovery.can_check_again
    assert not owner.mutation_active(binding)
    settled = owner.snapshot(binding)
    assert settled.push_candidate is None
    assert settled.push_recovery is None
    network_commands = [
        tuple(os.fsdecode(argument) for argument in argv)
        for argv, _kwargs in runner.network_calls
    ]
    assert sum("push" in command for command in network_commands) == 1
    assert network_commands[-1][-2:] == (
        "https://push.example.test/team/notes.git",
        BRANCH_REF,
    )
    assert not any(context_parent.iterdir())


@pytest.mark.asyncio
async def test_push_recovery_rejects_stale_operation_after_new_attempt(
    tmp_path: Path,
) -> None:
    """An A callback must not authorize or consume same-binding recovery B."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("d" * 40),
            _remote_observation("d" * 40),
            _remote_observation("d" * 40),
            _remote_observation("d" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed_a = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed_a.handle)).state == "uncertain"
    operation_a = _current_push_operation(service, binding)
    assert (
        await service.check_push_again(binding, operation_a)
    ).state == "succeeded"

    _publish_candidate_on_owner(
        owner,
        binding,
        repository,
        parent_oid="d" * 40,
        candidate_oid="f" * 40,
    )
    runner.head_oid = "f" * 40
    runner.parent_oid = "d" * 40
    reviewed_b = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed_b.handle)).state == "uncertain"
    operation_b = _current_push_operation(service, binding)
    evidence_b = service._uncertain_push
    assert evidence_b is not None
    grant_b = evidence_b.recovery_handle
    snapshot_b = owner.snapshot(binding)
    calls_before = tuple(runner.network_calls)

    assert service.authorize_push_recovery(binding, operation_a) is False
    with pytest.raises(
        git_service.GitMutationAdmissionError,
        match="exact recovery operation",
    ):
        service.check_push_again(binding, operation_a)

    assert service._uncertain_push is evidence_b
    assert service._uncertain_push.recovery_handle is grant_b
    assert service.retained_push_operation(binding) is operation_b
    assert owner.snapshot(binding) == snapshot_b
    assert tuple(runner.network_calls) == calls_before
    await service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (_remote_observation("b" * 40), "uncertain"),
        (_remote_observation("e" * 40), "needs_attention"),
        (_settled_network_result(0, b""), "needs_attention"),
        (_settled_network_result(1, b""), "needs_attention"),
    ],
    ids=("parent", "other", "missing", "query-failure"),
)
async def test_push_recovery_never_retries_and_unresolved_state_keeps_gate(
    tmp_path: Path,
    observation: GitCommandResult,
    expected: str,
) -> None:
    """One recovery click must never mutate or infer definite failure."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            observation,
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"
    calls_before = len(runner.network_calls)

    recovery = await service.check_push_again(
        binding,
        _current_push_operation(service, binding),
    )

    assert recovery.state == expected
    assert recovery.can_check_again
    assert len(runner.network_calls) == calls_before + 1
    assert sum(
        "push" in tuple(os.fsdecode(argument) for argument in argv)
        for argv, _kwargs in runner.network_calls
    ) == 1
    assert owner.mutation_active(binding)
    assert owner.snapshot(binding).push_recovery == recovery
    assert owner.snapshot(binding).push_candidate is not None
    await service.shutdown()


@pytest.mark.asyncio
async def test_push_recovery_waits_for_terminal_descendants(
    tmp_path: Path,
) -> None:
    """A query must not start while the original process tree is unproved."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_settled_network_result(
            1,
            b"",
            containment_proved=False,
        ),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)

    result = await service.start_push(binding, reviewed.handle)

    assert result.state == "uncertain"
    assert owner.snapshot(binding).push_recovery_available is False
    calls_before = tuple(runner.network_calls)
    with pytest.raises(
        git_service.GitMutationAdmissionError,
        match="descendants",
    ):
        service.check_push_again(
            binding,
            _current_push_operation(service, binding),
        )
    assert tuple(runner.network_calls) == calls_before
    assert owner.mutation_active(binding)
    await service.shutdown()
    assert not owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_push_recovery_retains_unproved_cancelled_child_until_shutdown(
    tmp_path: Path,
) -> None:
    """A cancellation containment failure must remain exact and fail closed."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _UnprovedCancelledPushRecoveryRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"

    recovery = await service.check_push_again(
        binding,
        _current_push_operation(service, binding),
    )

    assert recovery.state == "needs_attention"
    evidence = service._uncertain_push
    assert evidence is not None
    assert evidence.retained_child is runner.token
    assert evidence.descendants_terminal is False
    assert evidence.recovery_handle is None
    assert owner._push_recovery_grant is None
    assert owner.snapshot(binding).push_recovery_available is False
    assert service.authorize_push_recovery(
        binding,
        _current_push_operation(service, binding),
    ) is False
    assert owner.mutation_active(binding)
    assert runner.released is False

    await service.shutdown()

    assert runner.released is True
    assert service._uncertain_push is None
    assert owner.snapshot(binding).push_recovery is None
    assert not owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_push_recovery_settles_retained_child_without_new_query(
    tmp_path: Path,
) -> None:
    """Terminal child proof must re-enable only the exact parked recovery."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _UnprovedCancelledPushRecoveryRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"
    operation = _current_push_operation(service, binding)

    recovery = await service.check_push_again(binding, operation)
    calls_after_recovery = tuple(runner.network_calls)
    await asyncio.wait_for(runner.settlement_started.wait(), timeout=1)
    continuation = service._push_recovery_settlement_cycle

    assert recovery.state == "needs_attention"
    assert continuation is not None
    assert owner.snapshot(binding).push_recovery_available is False
    assert owner._push_recovery_grant is None
    assert owner.mutation_active(binding)
    assert tuple(runner.network_calls) == calls_after_recovery

    runner.release_settlement.set()
    await asyncio.wait_for(asyncio.shield(continuation), timeout=1)

    evidence = service._uncertain_push
    assert evidence is not None
    assert evidence.retained_child is None
    assert evidence.descendants_terminal is True
    assert evidence.recovery_handle is not None
    assert owner.snapshot(binding).push_recovery_available is True
    assert runner.released is True
    assert tuple(runner.network_calls) == calls_after_recovery
    assert owner.mutation_active(binding)
    await service.shutdown()


@pytest.mark.asyncio
async def test_push_shutdown_retains_recovery_waiter_after_cycle_settles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner shutdown must join the shield waiter in the cycle-callback race."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"
    operation = _current_push_operation(service, binding)
    cycle_settled = asyncio.Event()
    release_waiter = asyncio.Event()
    original_shield = service._shield_push_recovery_cycle

    async def delayed_shield(cycle):
        result = await original_shield(cycle)
        cycle_settled.set()
        await release_waiter.wait()
        return result

    monkeypatch.setattr(service, "_shield_push_recovery_cycle", delayed_shield)
    waiter = service.check_push_again(binding, operation)
    await asyncio.wait_for(cycle_settled.wait(), timeout=1)

    assert service._push_recovery_cycle is None
    assert service._push_recovery_waiter is waiter
    settlement = service.shutdown()

    async def await_settlement() -> None:
        await settlement

    shutdown_waiter = asyncio.create_task(await_settlement())
    await asyncio.sleep(0)
    assert not shutdown_waiter.done()

    release_waiter.set()
    assert (await asyncio.wait_for(waiter, timeout=1)).state == "uncertain"
    await asyncio.wait_for(shutdown_waiter, timeout=1)

    assert service._push_recovery_waiter is None
    assert service._uncertain_push is None
    assert not owner.mutation_active(binding)


@pytest.mark.asyncio
async def test_push_recovery_trust_drift_requires_fresh_exact_authorization(
    tmp_path: Path,
) -> None:
    """Trust ABA must not redirect recovery or revive its old capability."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"
    assert owner.clear_trust_if_matches(binding, repository)
    assert owner.publish_trust(binding, repository)
    calls_before = len(runner.network_calls)

    with pytest.raises(
        git_service.GitMutationAdmissionError,
        match="authorization",
    ):
        service.check_push_again(
            binding,
            _current_push_operation(service, binding),
        )

    assert len(runner.network_calls) == calls_before
    operation = _current_push_operation(service, binding)
    assert service.authorize_push_recovery(binding, operation) is True
    recovery = await service.check_push_again(binding, operation)
    assert recovery.state == "uncertain"
    command = tuple(
        os.fsdecode(argument)
        for argument in runner.network_calls[-1][0]
    )
    assert command[-2:] == (
        "https://push.example.test/team/notes.git",
        BRANCH_REF,
    )
    await service.shutdown()


@pytest.mark.asyncio
async def test_uncertain_push_blocks_git_and_rebinding_but_allows_note_changes(
    tmp_path: Path,
) -> None:
    """Releasing the transition gate or freezing ordinary edits must fail."""
    owner, binding, repository = _candidate_owner(tmp_path)
    runner = _ControlledExactPushRunner(
        repository,
        observations=(
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
            _remote_observation("b" * 40),
        ),
        push_result=_accepted_push_result(),
    )
    service = FileNotesGitService(
        owner,
        runner=runner,
        git_executable="git",
        environment={},
        network_context_factory=_network_factory(tmp_path),
    )
    reviewed = await _prepare_exact_push_review(service, binding)
    assert (await service.start_push(binding, reviewed.handle)).state == "uncertain"

    assert owner.admit_mutation(binding).reason == "mutation_active"
    assert owner.try_acquire_transition(binding, "source") is None
    assert owner.try_acquire_transition(binding, "screen") is None
    with pytest.raises(RuntimeError, match="mutation"):
        owner.select_root(tmp_path / "other-notes")
    assert owner.record_change(
        binding,
        SessionChange("modified", "edited-during-uncertainty.md"),
    )
    assert owner.current_binding() == binding
    await service.shutdown()
