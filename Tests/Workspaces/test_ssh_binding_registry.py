"""Registry admission for ``ssh-filesystem`` runtime bindings (Phase 1a).

``add_ssh_binding`` mirrors ``add_folder_binding``: parse + validate the
locator, canonicalize the destination via ``ssh -G`` (a local config dump
— no connection, no DNS, no remote command), reject duplicates/nesting on
the same canonical identity, and persist kind ``ssh-filesystem`` with a
metadata allowlist that never carries credentials.

These tests never execute the real client: the fake ``ssh`` (a shell
script in ``tmp_path``, the Task 6 pattern) dispatches on the host argv
token and prints the fixed ``key value`` block ``ssh -G`` would.
"""

from __future__ import annotations

import shlex
from dataclasses import replace
from pathlib import Path

import pytest

from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
from tldw_chatbook.Workspaces.models import (
    RuntimeBindingKind,
    RuntimeBindingStatus,
    WorkspaceRuntimeBinding,
)
from tldw_chatbook.Workspaces.registry_service import (
    BindingNotFound,
    LocalWorkspaceRegistryService,
    WorkspaceNotFound,
    WorkspaceRegistryServiceError,
)

# The complete set of metadata keys add_ssh_binding may ever write; the
# no-credentials-in-metadata contract is asserted against this allowlist.
_ALLOWED_METADATA_KEYS = {
    "access",
    "python",
    "canonical_fingerprint",
    "canonical_host",
    "canonical_port",
    "canonical_user",
}

# Alias/host -> (hostname, port, user) the fake ``ssh -G`` resolves to.
# ``dev`` and ``devbox`` deliberately share one destination so alias-vs-
# hostname locators can be proven to collide on the canonical identity.
_DEFAULT_RESOLVES: dict[str, tuple[str, int, str]] = {
    "dev": ("devbox.example.com", 22, "me"),
    "devbox": ("devbox.example.com", 22, "me"),
    "alpha": ("alpha.example.com", 22, "me"),
    "beta": ("beta.example.com", 22, "me"),
}


def _fake_ssh(tmp_path: Path, resolves=None) -> Path:
    """Write an executable fake ``ssh``; return its path.

    The canonicalization argv is always ``-G [-p port] [-l user] -- host``
    (Task 6 pins that form). Like the real ``ssh -G``, locator parts
    override the config: the script scans for ``-p``/``-l`` values, takes
    the host as the last argument, and prints the three identity keys
    with the mapping's port/user as defaults. Unknown hosts fail like
    real ssh (stderr + exit 255) so the error mapping is exercised too.
    """
    resolved = dict(_DEFAULT_RESOLVES if resolves is None else resolves)
    lines = [
        "#!/bin/sh",
        "port=''",
        "user=''",
        "host=''",
        "prev=''",
        'for arg in "$@"; do',
        '  if [ "$prev" = "-p" ]; then port="$arg"; fi',
        '  if [ "$prev" = "-l" ]; then user="$arg"; fi',
        '  host="$arg"',
        '  prev="$arg"',
        "done",
        'case "$host" in',
    ]
    for alias, (hostname, default_port, default_user) in resolved.items():
        lines.append(
            f"  {shlex.quote(alias)}) "
            f'[ -n "$port" ] || port={shlex.quote(str(default_port))}; '
            f'[ -n "$user" ] || user={shlex.quote(default_user)}; '
            f"printf 'hostname {hostname}\\n'; "
            f'printf \'port %s\\nuser %s\\n\' "$port" "$user"; exit 0 ;;'
        )
    lines += ["esac", 'echo "unknown host" >&2', "exit 255"]
    script = tmp_path / "fake-ssh"
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script


class _BindingRecorder:
    """Change Review observer stand-in recording binding_added calls."""

    def __init__(self) -> None:
        self.added: list[tuple[str, WorkspaceRuntimeBinding]] = []

    def binding_added(
        self, workspace_id: str, binding: WorkspaceRuntimeBinding
    ) -> None:
        self.added.append((workspace_id, binding))


@pytest.fixture()
def service(tmp_path: Path):
    db = WorkspaceDB(tmp_path / "ws.sqlite", client_id="ssh-registry-tests")
    svc = LocalWorkspaceRegistryService(db)
    svc.create_workspace(workspace_id="workspace-ssh-1", name="SSH Workspace")
    yield svc
    db.close()


# --- Admission happy path ---


def test_add_ssh_binding_on_named_workspace(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    assert binding.binding_kind is RuntimeBindingKind.SSH_FILESYSTEM
    assert binding.binding_id.startswith("ssh-")
    # locator is the user-typed canonical form (default port dropped)
    assert binding.locator == "ssh://dev/srv/app"
    # informational only — admission never reads the stored status column
    assert binding.status is RuntimeBindingStatus.READY
    assert binding.metadata["access"] == "ro"
    assert binding.metadata["python"] == "python3"
    assert binding.metadata["canonical_host"] == "devbox.example.com"
    assert binding.metadata["canonical_port"] == 22
    assert binding.metadata["canonical_user"] == "me"
    assert set(binding.metadata) <= _ALLOWED_METADATA_KEYS
    listed = service.list_ssh_bindings("workspace-ssh-1")
    assert [b.binding_id for b in listed] == [binding.binding_id]


def test_add_ssh_binding_write_access_and_custom_interpreter(
    tmp_path: Path, service
):
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1",
        "ssh://me@dev:2222/srv/app",
        allow_write=True,
        python_interpreter="/usr/bin/python3.11",
        ssh_bin=str(ssh),
    )
    assert binding.metadata["access"] == "rw"
    assert binding.metadata["python"] == "/usr/bin/python3.11"
    assert binding.metadata["canonical_port"] == 2222
    assert binding.metadata["canonical_user"] == "me"


def test_add_ssh_binding_fingerprint_matches_task6_helper(
    tmp_path: Path, service
):
    from tldw_chatbook.Tools.remote_binding_locator import (
        CanonicalTarget,
        canonical_fingerprint,
        parse_remote_locator,
    )

    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    expected = canonical_fingerprint(
        CanonicalTarget(hostname="devbox.example.com", port=22, user="me"),
        parse_remote_locator("ssh://dev/srv/app").path,
    )
    assert binding.metadata["canonical_fingerprint"] == expected


def test_add_ssh_binding_notifies_change_review_owner(
    tmp_path: Path, service
):
    recorder = _BindingRecorder()
    service.attach_change_review_consent_service(recorder)
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    assert [(ws, b.binding_id) for ws, b in recorder.added] == [
        ("workspace-ssh-1", binding.binding_id)
    ]


# --- Workspace gating ---


def test_default_workspace_rejected(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="Default workspace"):
        service.add_ssh_binding(
            "workspace-default", "ssh://dev/srv/app", ssh_bin=str(ssh)
        )
    assert service.list_ssh_bindings("workspace-default") == ()


def test_unknown_workspace_rejected(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceNotFound):
        service.add_ssh_binding(
            "workspace-nope", "ssh://dev/srv/app", ssh_bin=str(ssh)
        )


# --- Locator and interpreter validation ---


@pytest.mark.parametrize(
    "raw",
    [
        "dev/srv/app",  # missing scheme
        "ssh://dev",  # missing absolute path
        "ssh://dev/srv/../app",  # parent escape
        "ssh://dev/srv app",  # whitespace crosses a remote command line
        "ssh://-oProbe/x",  # option injection via leading dash
        "ssh://dev/~root/x",  # home-relative remote path
        123,  # not a string
    ],
)
def test_malformed_locator_rejected(tmp_path: Path, service, raw):
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="SSH locator"):
        service.add_ssh_binding("workspace-ssh-1", raw, ssh_bin=str(ssh))


def test_unresolvable_host_rejected(tmp_path: Path, service):
    # The fake fails unknown hosts like real ssh (exit 255); the static
    # branch reason must surface without echoing anything from stderr.
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="exited with status"):
        service.add_ssh_binding(
            "workspace-ssh-1", "ssh://ghost/srv/app", ssh_bin=str(ssh)
        )


@pytest.mark.parametrize(
    "bad",
    [
        "python 3",  # space
        "-python3",  # leading dash (option injection)
        "python3;touch /tmp/x",  # shell metacharacters
        "$PYTHON",  # metachar
        "py'thon3",  # quote
        "",  # empty
        None,  # not a string
    ],
)
def test_bad_python_interpreter_rejected(tmp_path: Path, service, bad):
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="python interpreter"):
        service.add_ssh_binding(
            "workspace-ssh-1",
            "ssh://dev/srv/app",
            python_interpreter=bad,
            ssh_bin=str(ssh),
        )


def test_locator_validation_before_db_lookup(tmp_path: Path, service):
    """Evaluation-order guard (mirrors the folder binding one).

    With a malformed locator the parser must reject before any workspace
    read: list_ssh_bindings("") would raise ValueError("workspace_id is
    required") if the overlap check ran first.
    """
    ssh = _fake_ssh(tmp_path)
    with pytest.raises(WorkspaceRegistryServiceError, match="SSH locator"):
        service.add_ssh_binding("", "ssh://dev/srv/../app", ssh_bin=str(ssh))


# --- Duplicate / nested roots on one canonical identity ---


def test_duplicate_same_identity_rejected(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    # Explicit user + explicit default port canonicalize to the same
    # destination and the same path: duplicate, regardless of access.
    with pytest.raises(WorkspaceRegistryServiceError, match="already bound"):
        service.add_ssh_binding(
            "workspace-ssh-1",
            "ssh://me@dev:22/srv/app",
            allow_write=True,
            ssh_bin=str(ssh),
        )
    assert len(service.list_ssh_bindings("workspace-ssh-1")) == 1


def test_nested_same_identity_rejected_across_alias_and_hostname(
    tmp_path: Path, service
):
    ssh = _fake_ssh(tmp_path)
    service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    # ``devbox`` is a different spelling of the same destination (the
    # fake resolves both to devbox.example.com:22 as user me): the child
    # path nests inside the alias-spelled parent and must be rejected.
    with pytest.raises(WorkspaceRegistryServiceError, match="inside"):
        service.add_ssh_binding(
            "workspace-ssh-1", "ssh://devbox:22/srv/app/sub", ssh_bin=str(ssh)
        )


def test_existing_ssh_root_inside_candidate_rejected(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app/sub", ssh_bin=str(ssh)
    )
    with pytest.raises(WorkspaceRegistryServiceError, match="remove it first"):
        service.add_ssh_binding(
            "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
        )


def test_different_hosts_same_path_allowed(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    first = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://alpha/srv/app", ssh_bin=str(ssh)
    )
    second = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://beta/srv/app", ssh_bin=str(ssh)
    )
    listed = service.list_ssh_bindings("workspace-ssh-1")
    assert {b.binding_id for b in listed} == {first.binding_id, second.binding_id}


def test_different_ports_same_host_allowed(tmp_path: Path, service):
    # Port is part of the canonical identity: same hostname, different
    # port is a different destination, so identical paths do not collide.
    ssh = _fake_ssh(tmp_path)
    service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    second = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev:2222/srv/app", ssh_bin=str(ssh)
    )
    assert second.metadata["canonical_port"] == 2222


def test_local_folder_and_ssh_binding_coexist(
    tmp_path: Path, service, monkeypatch
):
    # Local vs remote never conflict: the same-looking path bound as a
    # local folder and as an SSH root coexist, and each kind lists only
    # its own bindings. The local sensitive-path denylist is stubbed
    # (same pattern as test_folder_binding_validator.py): this test's
    # subject is cross-kind coexistence, and exercising the real
    # denylist here would drag in its lazy RAG import, which trips the
    # known first-in-process config-participant guard
    # (RecoveryRequired "raw_source_selection_changed", TASK-32908
    # class) that already makes test_folder_binding_validator.py's own
    # first test environment-flaky.
    import tldw_chatbook.Workspaces.registry_service as rs

    monkeypatch.setattr(rs, "find_root_binding_conflict", lambda p: None)
    ssh = _fake_ssh(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    folder = service.add_folder_binding("workspace-ssh-1", project)
    remote = service.add_ssh_binding(
        "workspace-ssh-1", f"ssh://dev{project}", ssh_bin=str(ssh)
    )
    assert [b.binding_id for b in service.list_folder_bindings("workspace-ssh-1")] == [
        folder.binding_id
    ]
    assert [b.binding_id for b in service.list_ssh_bindings("workspace-ssh-1")] == [
        remote.binding_id
    ]


# --- Metadata discipline ---


def test_metadata_stays_within_allowlist_after_round_trip(
    tmp_path: Path, service
):
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1",
        "ssh://dev/srv/app",
        allow_write=True,
        python_interpreter="/usr/bin/python3.11",
        ssh_bin=str(ssh),
    )
    fetched = service.get_runtime_binding(binding.binding_id)
    assert fetched is not None
    assert set(fetched.metadata) <= _ALLOWED_METADATA_KEYS
    assert fetched.metadata == binding.metadata


def test_secret_looking_metadata_keys_are_scrubbed(tmp_path: Path, service):
    # The model-level scrubber (the shared no-credentials discipline)
    # drops secret-looking keys from any binding's metadata, so nothing
    # downstream can smuggle a credential through this channel.
    scrubbed = WorkspaceRuntimeBinding(
        workspace_id="workspace-ssh-1",
        binding_id="ssh-scrub",
        binding_kind=RuntimeBindingKind.SSH_FILESYSTEM,
        label="scrub",
        locator="ssh://dev/x",
        metadata={
            "ssh_password": "hunter2",
            "identity_token": "t0k3n",
            "access": "ro",
        },
    )
    assert "ssh_password" not in scrubbed.metadata
    assert "identity_token" not in scrubbed.metadata
    assert scrubbed.metadata["access"] == "ro"


# --- Access toggle ---


def test_set_ssh_binding_access_toggles(tmp_path: Path, service):
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    assert binding.metadata["access"] == "ro"
    rw = service.set_ssh_binding_access(binding.binding_id, allow_write=True)
    assert rw.metadata["access"] == "rw"
    stored = service.get_runtime_binding(binding.binding_id)
    assert stored is not None and stored.metadata["access"] == "rw"
    ro = service.set_ssh_binding_access(binding.binding_id, allow_write=False)
    assert ro.metadata["access"] == "ro"


def test_set_ssh_binding_access_rejects_other_kinds(
    tmp_path: Path, service, monkeypatch
):
    # Denylist stubbed for the same reason as the coexist test above:
    # the subject here is the kind check, not the sensitive-path gate.
    import tldw_chatbook.Workspaces.registry_service as rs

    monkeypatch.setattr(rs, "find_root_binding_conflict", lambda p: None)
    ssh = _fake_ssh(tmp_path)
    project = tmp_path / "project"
    project.mkdir()
    folder = service.add_folder_binding("workspace-ssh-1", project)
    with pytest.raises(WorkspaceRegistryServiceError, match="SSH binding"):
        service.set_ssh_binding_access(folder.binding_id, allow_write=True)


def test_set_ssh_binding_access_unknown_id_raises(tmp_path: Path, service):
    with pytest.raises(BindingNotFound):
        service.set_ssh_binding_access("ssh-nope", allow_write=True)


# --- Listing semantics and persistence ---


def test_list_ssh_bindings_returns_stored_status_without_recompute(
    tmp_path: Path, service
):
    # Unlike list_folder_bindings (which recomputes status from the local
    # disk), the SSH lister returns rows as stored: the in-memory status
    # cache (Task 13) owns live status.
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    service.save_runtime_binding(replace(binding, status=RuntimeBindingStatus.MISSING))
    listed = service.list_ssh_bindings("workspace-ssh-1")
    assert len(listed) == 1
    assert listed[0].status is RuntimeBindingStatus.MISSING


def test_remove_runtime_binding_accepts_ssh_bindings(tmp_path: Path, service):
    # remove_runtime_binding is kind-agnostic (DELETE by id); pin that an
    # ssh-filesystem row flows through it like any other binding.
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-1", "ssh://dev/srv/app", ssh_bin=str(ssh)
    )
    service.remove_runtime_binding(binding.binding_id)
    assert service.list_ssh_bindings("workspace-ssh-1") == ()
    assert service.get_runtime_binding(binding.binding_id) is None
    with pytest.raises(BindingNotFound):
        service.remove_runtime_binding(binding.binding_id)


def test_ssh_binding_round_trips_through_workspace_db(tmp_path: Path):
    db_path = tmp_path / "roundtrip.sqlite"
    db = WorkspaceDB(db_path, client_id="ssh-roundtrip")
    service = LocalWorkspaceRegistryService(db)
    service.create_workspace(workspace_id="workspace-ssh-2", name="Two")
    ssh = _fake_ssh(tmp_path)
    binding = service.add_ssh_binding(
        "workspace-ssh-2",
        "ssh://me@dev:2222/srv/app",
        allow_write=True,
        python_interpreter="/usr/bin/python3.11",
        ssh_bin=str(ssh),
    )
    db.close()

    reopened = WorkspaceDB(db_path, client_id="ssh-roundtrip")
    service2 = LocalWorkspaceRegistryService(reopened)
    listed = service2.list_ssh_bindings("workspace-ssh-2")
    assert len(listed) == 1
    got = listed[0]
    assert got.binding_id == binding.binding_id
    assert got.binding_kind is RuntimeBindingKind.SSH_FILESYSTEM
    assert got.locator == "ssh://me@dev:2222/srv/app"
    assert got.status is RuntimeBindingStatus.READY
    assert got.metadata == binding.metadata
    fetched = service2.get_runtime_binding(binding.binding_id)
    assert fetched is not None and fetched.binding_kind is (
        RuntimeBindingKind.SSH_FILESYSTEM
    )
    reopened.close()
