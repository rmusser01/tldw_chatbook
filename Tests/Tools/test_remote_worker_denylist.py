"""Remote denylist + worker-side exclusion enforcement (Phase 3c, Task 17).

Two enforcement layers meet in the WORKER's dispatch (the committed
bundle, exercised through the loopback harness):

1. ``REMOTE_SENSITIVE_PATHS`` — the remote-home-relative denylist Task 8
   embedded as data. Task 17 wires its enforcement: entries map onto the
   pinned root when the root sits under the worker host's home directory,
   and the mapped subtrees join every operation's exclusion set (direct
   reads/writes refuse; enumerating tools omit).
2. ``sensitive_exclusions`` — the request-carried serialized binding
   exclusions (raw relative strings per ADR-174). The matcher already
   existed for reads/writes; Task 17 pins it end-to-end through the
   loopback worker and closes the ``stat_path`` gap.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

from tldw_chatbook.Tools.remote_workspace_executor import run_bundle_loopback
from tldw_chatbook.Tools.workspace_tool_protocol import MAX_RESPONSE_BYTES

_PING_PROBE_IDENTITY = {"device": 0, "inode": 0, "mode": 0, "reparse": False}


def _workspace(tmp_path: Path, *, home: bool = False) -> Path:
    """One stand-in remote filesystem: ok file, ``secrets/``, ``.ssh/``.

    ``home=True`` marks the stand-in as the worker host's HOME (the
    loopback child inherits this process's environment), which is what
    makes the home-relative denylist reachable from the pinned root.
    """
    root = tmp_path / ("home-standin" if home else "workspace")
    root.mkdir()
    (root / "ok.txt").write_text("public body\n", encoding="utf-8")
    secrets = root / "secrets"
    secrets.mkdir()
    (secrets / "kv.txt").write_text("token=hidden\n", encoding="utf-8")
    ssh_dir = root / ".ssh"
    ssh_dir.mkdir()
    (ssh_dir / "id_rsa").write_text("PRIVATE\n", encoding="utf-8")
    (ssh_dir / "config").write_text("Host *\n", encoding="utf-8")
    return root


def _request(
    root: Path,
    operation: str,
    arguments: dict[str, Any],
    *,
    chain: dict[str, Any] | None = None,
    intent: str = "read",
) -> dict[str, Any]:
    if chain is None:
        identities = [_PING_PROBE_IDENTITY]
        locator = str(root)
    else:
        identities = [
            {
                "device": entry[1],
                "inode": entry[2],
                "mode": entry[3],
                "reparse": False,
            }
            for entry in chain["identity_chain"]
        ]
        locator = chain["canonical_path"]
    return {
        "version": 1,
        "operation_id": uuid.uuid4().hex,
        "operation": operation,
        "intent": intent,
        "root_locator": locator,
        "root_identity": identities[0],
        "ancestor_identities": identities,
        "arguments": arguments,
        "timeout_seconds": 30,
        "output_max_bytes": MAX_RESPONSE_BYTES,
    }


def _ping_chain(root: Path) -> dict[str, Any]:
    import json

    result = run_bundle_loopback(root, _request(root, "ping", {}))
    assert result["outcome"] == "success", result
    return json.loads(result["result"])


def _loopback(
    root: Path,
    chain: dict[str, Any],
    operation: str,
    arguments: dict[str, Any],
    *,
    intent: str = "read",
) -> dict[str, Any]:
    return run_bundle_loopback(
        root, _request(root, operation, arguments, chain=chain, intent=intent)
    )


# ---------------------------------------------------------------------------
# Remote-home denylist (REMOTE_SENSITIVE_PATHS), first wired by Task 17
# ---------------------------------------------------------------------------


def test_denylist_read_of_home_ssh_refused_via_loopback(
    tmp_path: Path, monkeypatch
) -> None:
    root = _workspace(tmp_path, home=True)
    monkeypatch.setenv("HOME", str(root))
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_read",
        {"path": ".ssh/id_rsa", "sensitive_exclusions": []},
    )

    assert result["outcome"] == "failure"
    assert "PRIVATE" not in (result["error"] or "")
    assert "PRIVATE" not in (result["result"] or "")


def test_denylist_write_into_home_ssh_refused_via_loopback(
    tmp_path: Path, monkeypatch
) -> None:
    root = _workspace(tmp_path, home=True)
    monkeypatch.setenv("HOME", str(root))
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_write",
        {
            "path": ".ssh/authorized_keys",
            "content": "ssh-ed25519 forged\n",
            "sensitive_exclusions": [],
        },
        intent="write",
    )

    assert result["outcome"] == "failure"
    assert not (root / ".ssh" / "authorized_keys").exists()


def test_denylist_glob_never_lists_home_ssh_entries(
    tmp_path: Path, monkeypatch
) -> None:
    root = _workspace(tmp_path, home=True)
    monkeypatch.setenv("HOME", str(root))
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_glob",
        {"pattern": ".ssh/*", "sensitive_exclusions": []},
    )

    assert result["outcome"] == "success", result
    listing = result["result"] or ""
    assert "id_rsa" not in listing
    assert "config" not in listing
    assert "no files matching" in listing


def test_denylist_stat_of_home_ssh_refused_via_loopback(
    tmp_path: Path, monkeypatch
) -> None:
    root = _workspace(tmp_path, home=True)
    monkeypatch.setenv("HOME", str(root))
    chain = _ping_chain(root)

    result = _loopback(root, chain, "stat_path", {"path": ".ssh"})

    assert result["outcome"] == "failure"


def test_denylist_does_not_reach_roots_outside_the_worker_home(
    tmp_path: Path,
) -> None:
    """The denylist maps onto the HOME; an unrelated root keeps a literal
    ``.ssh`` fixture directory readable (name-based denial was considered
    and declined: a project's own ``.ssh`` test fixture is not the home
    credential store — ``remote_sensitive_paths`` is home-relative)."""
    root = _workspace(tmp_path)  # NOT under the real HOME
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_read",
        {"path": ".ssh/config", "sensitive_exclusions": []},
    )

    assert result["outcome"] == "success", result
    assert "Host *" in (result["result"] or "")


# ---------------------------------------------------------------------------
# Review fix (Important 1): the home mapping must hold for roots at/above
# the home directory too. The pre-review ``root.relative_to(home)`` mapping
# RAISED for such roots and silently voided the whole denylist (``ssh://
# host/`` is a legal binding). The general mapping is relpath(home/entry,
# root); a pinned root INSIDE a denylisted entry refuses outright.
# ---------------------------------------------------------------------------


def _home_standin(tmp_path: Path) -> Path:
    """A controlled HOME stand-in with the denylist fixtures inside it."""
    home = tmp_path / "home"
    (home / ".ssh").mkdir(parents=True)
    (home / ".ssh" / "id_rsa").write_text("PRIVATE\n", encoding="utf-8")
    return home


def test_denylist_root_at_filesystem_root_still_denies_home_ssh(
    tmp_path: Path, monkeypatch
) -> None:
    """A binding pinned at ``ssh://host/`` (root ``/``) must not void the
    denylist: ``~/.ssh`` maps to a subtree refusal under the root."""
    home = _home_standin(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    root = Path("/")
    chain = _ping_chain(root)
    rel_id_rsa = PurePosixPath(
        os.path.relpath(home / ".ssh" / "id_rsa", root)
    ).as_posix()

    result = _loopback(
        root,
        chain,
        "fs_read",
        {"path": rel_id_rsa, "sensitive_exclusions": []},
    )

    assert result["outcome"] == "failure"
    assert "PRIVATE" not in (result["result"] or "")
    assert "PRIVATE" not in (result["error"] or "")


def test_denylist_root_inside_a_denylisted_entry_refuses_outright(
    tmp_path: Path, monkeypatch
) -> None:
    """A binding pinned INSIDE ``~/.ssh`` (root ``~/.ssh``) is itself a
    denylisted subtree: every operation refuses — nothing under it is
    reachable, not even an innocuous file the binding itself placed."""
    home = _home_standin(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    root = home / ".ssh"
    (root / "harmless.txt").write_text("nothing to see\n", encoding="utf-8")
    chain = _ping_chain(root)

    read = _loopback(
        root,
        chain,
        "fs_read",
        {"path": "harmless.txt", "sensitive_exclusions": []},
    )
    stat = _loopback(root, chain, "stat_path", {"path": "."})

    assert read["outcome"] == "failure"
    assert stat["outcome"] == "failure"


def test_denylist_root_inside_home_but_outside_entries_is_unaffected(
    tmp_path: Path, monkeypatch
) -> None:
    """A root under the home but outside every denylisted entry works
    normally (no catch-all, no void) — including its OWN ``.ssh`` fixture
    folder, which is not the home credential store."""
    home = _home_standin(tmp_path)
    monkeypatch.setenv("HOME", str(home))
    root = home / "projects"
    (root / ".ssh").mkdir(parents=True)
    (root / ".ssh" / "config").write_text("Host *\n", encoding="utf-8")
    (root / "ok.txt").write_text("public body\n", encoding="utf-8")
    chain = _ping_chain(root)

    ok = _loopback(
        root, chain, "fs_read", {"path": "ok.txt", "sensitive_exclusions": []}
    )
    fixture = _loopback(
        root,
        chain,
        "fs_read",
        {"path": ".ssh/config", "sensitive_exclusions": []},
    )

    assert ok["outcome"] == "success", ok
    assert "public body" in (ok["result"] or "")
    assert fixture["outcome"] == "success", fixture
    assert "Host *" in (fixture["result"] or "")


def test_denylist_grep_skips_home_ssh_content(
    tmp_path: Path, monkeypatch
) -> None:
    root = _workspace(tmp_path, home=True)
    monkeypatch.setenv("HOME", str(root))
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_grep",
        {
            "pattern": "PRIVATE",
            "sensitive_exclusions": [],
            "content_exclusions": [],
        },
    )

    assert result["outcome"] == "success", result
    # The denylisted file's content never leaks; only the echoed pattern
    # may mention it (the no-matches notice repeats the pattern itself).
    assert result["result"] == "(no matches for 'PRIVATE')"


# ---------------------------------------------------------------------------
# Request-carried serialized exclusions (ADR-174 raw relative strings)
# ---------------------------------------------------------------------------

_SUBTREE_EXCLUSIONS = [{"kind": "subtree", "value": "secrets"}]


def test_serialized_exclusion_read_refused_via_loopback(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_read",
        {"path": "secrets/kv.txt", "sensitive_exclusions": _SUBTREE_EXCLUSIONS},
    )

    assert result["outcome"] == "failure"
    assert "token=hidden" not in (result["error"] or "")


def test_serialized_exclusion_write_refused_via_loopback(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_write",
        {
            "path": "secrets/leak.txt",
            "content": "x",
            "sensitive_exclusions": _SUBTREE_EXCLUSIONS,
        },
        intent="write",
    )

    assert result["outcome"] == "failure"
    assert not (root / "secrets" / "leak.txt").exists()


def test_serialized_exclusion_list_omits_excluded_dir(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_list",
        {"path": ".", "sensitive_exclusions": _SUBTREE_EXCLUSIONS},
    )

    assert result["outcome"] == "success", result
    listing = result["result"] or ""
    assert "secrets/" not in listing
    assert "ok.txt" in listing


def test_serialized_exclusion_glob_does_not_list(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_glob",
        {"pattern": "**/*.txt", "sensitive_exclusions": _SUBTREE_EXCLUSIONS},
    )

    assert result["outcome"] == "success", result
    assert "kv.txt" not in (result["result"] or "")
    assert "ok.txt" in (result["result"] or "")


def test_serialized_exclusion_grep_does_not_read_content(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    chain = _ping_chain(root)

    result = _loopback(
        root,
        chain,
        "fs_grep",
        {
            "pattern": "token",
            "sensitive_exclusions": _SUBTREE_EXCLUSIONS,
            "content_exclusions": _SUBTREE_EXCLUSIONS,
        },
    )

    assert result["outcome"] == "success", result
    assert "token=hidden" not in (result["result"] or "")


# ---------------------------------------------------------------------------
# Transport-executor exclusion injection (Task 17): the model-facing spec
# handlers call ``execute(op, {path})`` verbatim, so the ssh/loopback
# executor injects the binding's serialized exclusions into the wire args
# (fail-closed: a throwing exclusions source refuses the call).
# ---------------------------------------------------------------------------


def _sensitive_exclusions_source():
    from tldw_chatbook.Utils.sensitive_paths import SensitiveExclusion

    return lambda: (SensitiveExclusion("subtree", "secrets"),)


def _loopback_executor(
    root: Path, *, exclusions=None
) -> "RemoteWorkspaceToolExecutor":
    import json

    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceToolExecutor,
    )

    prober = RemoteWorkspaceToolExecutor(
        root, root_locator=str(root), identity_chain_source=lambda: None
    )
    payload = prober.ping()
    chain = {
        "identity_chain": payload["identity_chain"],
        "canonical_path": payload["canonical_path"],
    }
    kwargs: dict[str, Any] = {}
    if exclusions is not None:
        kwargs["sensitive_exclusions"] = exclusions
    return RemoteWorkspaceToolExecutor(
        root,
        root_locator=str(root),
        identity_chain_source=lambda: chain,
        **kwargs,
    )


def test_transport_executor_injects_exclusions_into_bare_args(tmp_path: Path) -> None:
    """Model-facing args carry no exclusion field; the executor adds the
    binding's real serialized exclusions so the wire schema is satisfied
    AND the worker enforces the binding's exclusions."""
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
    )

    root = _workspace(tmp_path)
    executor = _loopback_executor(root, exclusions=_sensitive_exclusions_source())

    ok = executor.execute("fs_read", {"path": "ok.txt"}, intent="read")
    assert ok["outcome"] == "success"
    assert "public body" in (ok["result"] or "")

    with pytest.raises(RemoteWorkspaceExecutionError):
        executor.execute("fs_read", {"path": "secrets/kv.txt"}, intent="read")


def test_transport_executor_injection_is_fail_closed(tmp_path: Path) -> None:
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
    )

    root = _workspace(tmp_path)

    def broken():
        raise RuntimeError("exclusion source exploded")

    executor = _loopback_executor(root, exclusions=broken)

    with pytest.raises(RemoteWorkspaceExecutionError) as raised:
        executor.execute("fs_read", {"path": "ok.txt"}, intent="read")

    assert raised.value.code == "invalid_request"


# ---------------------------------------------------------------------------
# Review fix (Critical): model-reachable exclusion override + fs_grep gap.
# The model's tool args reach this executor verbatim (the provider's arg
# cleaning does not drop undeclared fields), so a caller-supplied
# ``sensitive_exclusions`` must NEVER win: the executor OVERRIDES both
# exclusion fields with the binding's serialized set, exactly like the
# sibling remote-mode ``WorkspaceToolExecutor._build_remote_request``.
# (Inverts the pre-review pin ``test_transport_executor_keeps_caller_
# supplied_exclusions``, which asserted caller-supplied values win.)
# ---------------------------------------------------------------------------


def test_transport_executor_overrides_model_supplied_exclusions(
    tmp_path: Path,
) -> None:
    """`fs_read {"path": "secrets/kv.txt", "sensitive_exclusions": []}` from
    the MODEL must still refuse: the binding's set overrides the model's
    empty list (declared fields reach the wire -- expected_sha256 already
    proved it -- so an undropped value here un-excluded everything)."""
    from tldw_chatbook.Tools.remote_workspace_executor import (
        RemoteWorkspaceExecutionError,
    )

    root = _workspace(tmp_path)
    executor = _loopback_executor(root, exclusions=_sensitive_exclusions_source())

    ok = executor.execute(
        "fs_read",
        {"path": "ok.txt", "sensitive_exclusions": []},
        intent="read",
    )
    assert ok["outcome"] == "success"

    with pytest.raises(RemoteWorkspaceExecutionError):
        executor.execute(
            "fs_read",
            {"path": "secrets/kv.txt", "sensitive_exclusions": []},
            intent="read",
        )


def test_transport_executor_overrides_model_supplied_grep_exclusions(
    tmp_path: Path,
) -> None:
    """fs_grep's SECOND exclusion field is overridden too: the model's
    empty ``content_exclusions`` cannot re-enable content scanning of an
    excluded subtree."""
    root = _workspace(tmp_path)
    executor = _loopback_executor(root, exclusions=_sensitive_exclusions_source())

    result = executor.execute(
        "fs_grep",
        {
            "pattern": "token",
            "sensitive_exclusions": [],
            "content_exclusions": [],
        },
        intent="read",
    )

    assert result["outcome"] == "success", result
    assert "token=hidden" not in (result["result"] or "")
    assert result["result"] == "(no matches for 'token')"


def test_transport_executor_injects_content_exclusions_for_bare_grep(
    tmp_path: Path,
) -> None:
    """fs_grep's wire schema REQUIRES both exclusion fields; bare model
    args (``{"pattern": ...}``) used to build an invalid request. Both
    are injected (and enforced) now."""
    root = _workspace(tmp_path)
    executor = _loopback_executor(root, exclusions=_sensitive_exclusions_source())

    result = executor.execute("fs_grep", {"pattern": "public"}, intent="read")

    assert result["outcome"] == "success", result
    assert "public body" in (result["result"] or "")

    excluded = executor.execute("fs_grep", {"pattern": "token"}, intent="read")
    assert excluded["outcome"] == "success"
    assert excluded["result"] == "(no matches for 'token')"

