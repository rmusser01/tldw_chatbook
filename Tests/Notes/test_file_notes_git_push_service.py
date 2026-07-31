from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import pytest

import tldw_chatbook.Notes.file_notes_git_push as push_contracts
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
    IndexBaseline,
    IndexEntry,
    PushIncludedNote,
    RepositoryIdentity,
    SessionChange,
    StagingOwnership,
)


BRANCH_REF = "refs/heads/main"


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


def _filesystem_identity(path: Path) -> FileSystemIdentity:
    metadata = path.stat()
    return FileSystemIdentity(metadata.st_dev, metadata.st_ino)


def _candidate_owner(
    tmp_path: Path,
) -> tuple[FileNotesSessionOwner, object, RepositoryIdentity]:
    root = tmp_path / "notes"
    git_dir = root / ".git"
    git_dir.mkdir(parents=True)
    (git_dir / "objects").mkdir()
    (git_dir / "config").write_text("[core]\n\tbare = false\n", encoding="utf-8")
    (root / "note.md").write_text("candidate\n", encoding="utf-8")
    return _owner_for_candidate(root, "b" * 40, "d" * 40)


def _owner_for_candidate(
    root: Path,
    parent_oid: str,
    candidate_oid: str,
) -> tuple[FileNotesSessionOwner, object, RepositoryIdentity]:
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
    assert owner.record_change(binding, SessionChange("modified", "note.md"))
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
    assert owner.publish_ownership(binding, {1: ownership})
    lease = owner.try_acquire_mutation(binding)
    assert lease is not None
    reviewed = owner._capture_commit_authority_after_review(
        lease,
        binding=binding,
        authority_generation=owner.snapshot(binding).git_authority_generation,
        repository=repository,
        head=head,
        group_sequence_ids={1: (1,)},
        subject="Guarded note",
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
    return owner, binding, repository


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
    ) -> None:
        self.repository = repository
        self.paths = paths
        self.lfs_paths = lfs_paths
        self.malformed_attributes = malformed_attributes
        self.change_config_during_read = change_config_during_read
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
                "https://push.example.test/team/notes.git",
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
        if "symbolic-ref" in command:
            return GitCommandResult(0, BRANCH_REF.encode() + b"\n", b"")
        if "rev-parse" in command:
            return GitCommandResult(0, b"d" * 40 + b"\n", b"")
        if "cat-file" in command:
            return GitCommandResult(
                0,
                (
                    b"tree "
                    + b"e" * 40
                    + b"\nparent "
                    + b"b" * 40
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
            return GitCommandResult(0, b"", b"")
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
        "store-one",
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
        "store-two",
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
    owner, binding, repository = _candidate_owner(tmp_path)
    home = tmp_path / "home"
    home_fallback = home / ".config" / "git" / "config"
    home_fallback.parent.mkdir(parents=True)
    home_fallback.write_text("[credential]\n", encoding="utf-8")
    runner = _ControlledLocalProofRunner(repository)
    runner.config_payload += _unknown_scope_config_record(
        home_fallback,
        "credential.helper",
        "store",
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
    assert len(runner.calls) <= 8
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
) -> tuple[Path, str, str]:
    root = tmp_path / ("real-lfs" if lfs else "real")
    root.mkdir()
    _git(root, "init", "-b", "main")
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
