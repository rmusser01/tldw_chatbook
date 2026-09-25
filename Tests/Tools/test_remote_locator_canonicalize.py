"""canonicalize_locator: ssh -G resolution, argv pinning, error mapping.

``ssh -G`` prints the fully resolved *local* ssh configuration and exits
— no connection, no DNS lookup, no remote command — so canonicalization
is a local config read. These tests never execute the real client: a
fake ``ssh`` (a shell script in ``tmp_path``) records the argv it was
invoked with and prints a fixed ``key value`` block, pinning the exact
call shape ``ssh -G [-p port] [-l user] -- host`` and the parsing /
error rules over its output.
"""

import shlex
from dataclasses import replace
from pathlib import PurePosixPath

import pytest

from tldw_chatbook.Tools import remote_binding_locator
from tldw_chatbook.Tools.remote_binding_locator import (
    CanonicalTarget,
    RemoteLocatorError,
    canonical_fingerprint,
    canonicalize_locator,
    parse_remote_locator,
    sha256_fingerprint,
)

# Real ``ssh -G`` prints dozens of keys; only the three identity keys are
# consumed. The shuffled order proves parsing does not depend on it.
_GOOD_OUTPUT = (
    "user me\n"
    "hostname devbox.example.com\n"
    "port 2222\n"
    "addressfamily any\n"
    "batchmode no\n"
)


def _fake_ssh(tmp_path, *, output: str = "", exit_code: int = 0, delay: float | None = None):
    """Write an executable fake ssh; return ``(script, argv_file)``.

    The script records its argv one token per line (``printf '%s\\n'
    "$@"``), optionally sleeps, prints the fixed output block, and exits
    with ``exit_code``.
    """
    script = tmp_path / "fake-ssh"
    argv_file = tmp_path / "ssh-argv.txt"
    lines = [
        "#!/bin/sh",
        f'printf \'%s\\n\' "$@" > {shlex.quote(str(argv_file))}',
    ]
    if delay is not None:
        lines.append(f"sleep {delay}")
    if output:
        out_file = tmp_path / "ssh-g-out.txt"
        out_file.write_text(output, encoding="utf-8")
        lines.append(f"cat {shlex.quote(str(out_file))}")
    lines.append(f"exit {exit_code}")
    script.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script.chmod(0o755)
    return script, argv_file


def _read_argv(argv_file) -> list[str]:
    return argv_file.read_text(encoding="utf-8").splitlines()


# --- Resolution and argv form ---


def test_canonicalize_parses_ssh_g_output(tmp_path):
    ssh, _ = _fake_ssh(tmp_path, output=_GOOD_OUTPUT)
    loc = parse_remote_locator("ssh://me@devbox:2222/srv/app")
    target = canonicalize_locator(loc, ssh_bin=str(ssh))
    assert target == CanonicalTarget(
        hostname="devbox.example.com", port=2222, user="me"
    )


def test_canonicalize_argv_form_is_pinned(tmp_path):
    # The call shape is part of the contract: -G first, then the locator
    # parts rebuilt exactly like every dispatch spawn (-p/-l/-- / host).
    ssh, argv_file = _fake_ssh(tmp_path, output=_GOOD_OUTPUT)
    loc = parse_remote_locator("ssh://me@devbox:2222/srv/app")
    canonicalize_locator(loc, ssh_bin=str(ssh))
    assert _read_argv(argv_file) == ["-G", "-p", "2222", "-l", "me", "--", "devbox"]


def test_canonicalize_argv_omits_unset_port_and_user(tmp_path):
    ssh, argv_file = _fake_ssh(
        tmp_path, output="hostname devbox.example.com\nport 22\nuser me\n"
    )
    loc = parse_remote_locator("ssh://devbox/srv/app")
    target = canonicalize_locator(loc, ssh_bin=str(ssh))
    assert _read_argv(argv_file) == ["-G", "--", "devbox"]
    # ssh -G expands config defaults: unset locator parts still resolve.
    assert target == CanonicalTarget(hostname="devbox.example.com", port=22, user="me")


def test_canonicalize_argv_strips_ipv6_brackets(tmp_path):
    ssh, argv_file = _fake_ssh(
        tmp_path, output="hostname 2001:db8::1\nport 22\nuser me\n"
    )
    loc = parse_remote_locator("ssh://[2001:db8::1]/srv/app")
    canonicalize_locator(loc, ssh_bin=str(ssh))
    assert _read_argv(argv_file) == ["-G", "--", "2001:db8::1"]


def test_canonicalize_lowercases_hostname_only(tmp_path):
    # DNS names are case-insensitive -> hostname is lowercased; ssh
    # treats usernames case-sensitively -> user is kept verbatim.
    ssh, _ = _fake_ssh(
        tmp_path, output="hostname DevBox.Example.COM\nport 2222\nuser Me\n"
    )
    loc = parse_remote_locator("ssh://me@devbox:2222/srv/app")
    target = canonicalize_locator(loc, ssh_bin=str(ssh))
    assert target.hostname == "devbox.example.com"
    assert target.user == "Me"


def test_canonical_target_is_frozen():
    target = CanonicalTarget(hostname="h.example.com", port=22, user=None)
    with pytest.raises(Exception):
        target.port = 23  # type: ignore[misc]


# --- Error mapping (static reasons, key names only, never output) ---


def test_canonicalize_failing_ssh_g_raises_without_echoing_output(tmp_path):
    ssh, _ = _fake_ssh(tmp_path, output="hostname secrets.example.com\n", exit_code=255)
    with pytest.raises(RemoteLocatorError, match="ssh -G") as excinfo:
        canonicalize_locator(
            parse_remote_locator("ssh://devbox/srv/app"), ssh_bin=str(ssh)
        )
    assert "secrets.example.com" not in str(excinfo.value)


@pytest.mark.parametrize("missing", ["hostname", "port", "user"])
def test_canonicalize_missing_key_raises(tmp_path, missing):
    lines = {
        "hostname": "hostname devbox.example.com",
        "port": "port 2222",
        "user": "user me",
    }
    del lines[missing]
    ssh, _ = _fake_ssh(tmp_path, output="\n".join(lines.values()) + "\n")
    with pytest.raises(RemoteLocatorError, match=missing):
        canonicalize_locator(
            parse_remote_locator("ssh://devbox/srv/app"), ssh_bin=str(ssh)
        )


def test_canonicalize_non_numeric_port_raises(tmp_path):
    ssh, _ = _fake_ssh(
        tmp_path, output="hostname h.example.com\nport seventeen\nuser me\n"
    )
    with pytest.raises(RemoteLocatorError, match="port"):
        canonicalize_locator(
            parse_remote_locator("ssh://devbox/srv/app"), ssh_bin=str(ssh)
        )


def test_canonicalize_timeout_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(remote_binding_locator, "_SSH_G_TIMEOUT_SECONDS", 0.2)
    ssh, _ = _fake_ssh(tmp_path, delay=5)
    with pytest.raises(RemoteLocatorError, match="ssh -G"):
        canonicalize_locator(
            parse_remote_locator("ssh://devbox/srv/app"), ssh_bin=str(ssh)
        )


def test_canonicalize_unrunnable_ssh_bin_raises(tmp_path):
    with pytest.raises(RemoteLocatorError, match="ssh -G"):
        canonicalize_locator(
            parse_remote_locator("ssh://devbox/srv/app"),
            ssh_bin=str(tmp_path / "no-such-ssh"),
        )


# --- Fingerprint over the canonical identity ---


def test_canonical_fingerprint_feeds_sha256_fingerprint():
    target = CanonicalTarget(hostname="devbox.example.com", port=2222, user="me")
    assert canonical_fingerprint(target, PurePosixPath("/srv/app")) == (
        sha256_fingerprint("me", "devbox.example.com", 2222, "/srv/app")
    )


def test_canonical_fingerprint_none_user_serializes_empty():
    # ssh -G always reports a user, but the type admits None (manual
    # construction); it must serialize like an unset user in the hash.
    target = CanonicalTarget(hostname="h.example.com", port=22, user=None)
    assert canonical_fingerprint(target, PurePosixPath("/x")) == (
        sha256_fingerprint("", "h.example.com", 22, "/x")
    )


def test_canonical_fingerprint_lowercases_hostname_defensively():
    upper = CanonicalTarget(hostname="DevBox.Example.COM", port=2222, user="me")
    lower = CanonicalTarget(hostname="devbox.example.com", port=2222, user="me")
    assert canonical_fingerprint(upper, PurePosixPath("/srv/app")) == (
        canonical_fingerprint(lower, PurePosixPath("/srv/app"))
    )


def test_canonical_fingerprint_stable_across_cosmetic_alias_edits():
    # Same resolved symbolic identity (what a cosmetic alias edit looks
    # like after ssh -G) -> same fingerprint -> no re-consent trip.
    a = CanonicalTarget(hostname="devbox.example.com", port=2222, user="me")
    b = CanonicalTarget(hostname="devbox.example.com", port=2222, user="me")
    assert canonical_fingerprint(a, PurePosixPath("/srv/app")) == (
        canonical_fingerprint(b, PurePosixPath("/srv/app"))
    )


def test_canonical_fingerprint_changes_on_genuine_retarget():
    base = CanonicalTarget(hostname="devbox.example.com", port=2222, user="me")
    fp = canonical_fingerprint(base, PurePosixPath("/srv/app"))
    assert fp != canonical_fingerprint(
        replace(base, hostname="staging.example.com"), PurePosixPath("/srv/app")
    )
    assert fp != canonical_fingerprint(
        replace(base, port=2202), PurePosixPath("/srv/app")
    )
    assert fp != canonical_fingerprint(
        replace(base, user="root"), PurePosixPath("/srv/app")
    )
    assert fp != canonical_fingerprint(base, PurePosixPath("/srv/other"))
