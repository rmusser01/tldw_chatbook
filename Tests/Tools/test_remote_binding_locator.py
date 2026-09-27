import hashlib
from pathlib import PurePosixPath

import pytest

from tldw_chatbook.Tools.remote_binding_locator import (
    RemoteLocatorError, build_ssh_argv, parse_remote_locator,
)
from tldw_chatbook.Tools.remote_binding_locator import (
    RemoteLocator, locator_string, sha256_fingerprint,
)

@pytest.mark.parametrize("raw,user,host,port", [
    ("ssh://devbox/srv/app", None, "devbox", None),
    ("ssh://me@devbox:2222/srv/app", "me", "devbox", 2222),
    ("ssh://[2001:db8::1]:2222/srv/app", None, "2001:db8::1", 2222),
])
def test_parse_ok(raw, user, host, port):
    loc = parse_remote_locator(raw)
    assert (loc.user, loc.host, loc.port) == (user, host, port)

@pytest.mark.parametrize("raw", [
    "ssh://devbox",                # no absolute path
    "ssh://devbox/~/x",            # ~ not allowed
    "ssh://-F./evil/srv/app",      # leading dash host
    "ssh://me$@devbox/srv/app",    # shell metachar
    "ssh://devbox:70000/srv/app",  # bad port
    "ssh://[::1]:22/srv/app extra",# trailing junk
])
def test_parse_rejects(raw):
    with pytest.raises(RemoteLocatorError):
        parse_remote_locator(raw)

def test_argv_components_never_mix():
    loc = parse_remote_locator("ssh://me@[2001:db8::1]:2222/srv/app")
    argv = build_ssh_argv(loc, ["-o", "BatchMode=yes"], ["python3", "-I"])
    assert argv == ["-o", "BatchMode=yes", "-p", "2222", "-l", "me",
                    "--", "2001:db8::1", "python3", "-I"]


# --- Additional coverage: normalization, charsets, canonicalization ---


@pytest.mark.parametrize("raw,expected", [
    ("ssh://devbox/srv/./app/", PurePosixPath("/srv/app")),
    ("ssh://devbox/srv//app", PurePosixPath("/srv/app")),
    # POSIX leaves a leading '//' implementation-defined and PurePosixPath
    # preserves it; canonical paths must not carry that ambiguity.
    ("ssh://devbox//srv/app", PurePosixPath("/srv/app")),
    ("ssh://devbox///srv/app", PurePosixPath("/srv/app")),
    ("ssh://devbox/", PurePosixPath("/")),
])
def test_parse_normalizes_path(raw, expected):
    assert parse_remote_locator(raw).path == expected


@pytest.mark.parametrize("raw", [
    "ssh://devbox/srv/../app",     # '..' component
    "ssh://devbox/../srv/app",     # '..' at the front
    "ssh://devbox/srv/~x",         # '~' anywhere in path
    "ssh://devbox:0/srv/app",      # port below range
    "ssh://devbox:65536/srv/app",  # port above range
    "ssh://devbox:abc/srv/app",    # non-numeric port
    "ssh://devbox:-22/srv/app",    # port with sign
    "ssh://2001:db8::1/srv/app",   # unbracketed IPv6
    "ssh://[2001:db8::1/srv/app",  # unterminated bracket
    "ssh://me:22@devbox/srv/app",  # '@' handled by split, ':' inside user part
    "ssh://me@-host/srv/app",      # leading dash host
    "ssh://-u@devbox/srv/app",     # leading dash user
    "ssh://me@/srv/app",           # empty host
    "ssh://@devbox/srv/app",       # empty user
    "ssh://devbox:/srv/app",       # empty port
    "http://devbox/srv/app",       # wrong scheme
    "devbox/srv/app",              # no scheme at all
    "ssh://devbox/srv/ap p",       # whitespace inside path
    "ssh://me\td@devbox/srv/app",  # control char
    "ssh://me@d evbox/srv/app",    # whitespace in host
])
def test_parse_rejects_extended(raw):
    with pytest.raises(RemoteLocatorError):
        parse_remote_locator(raw)


def test_parse_rejects_non_string():
    with pytest.raises(RemoteLocatorError):
        parse_remote_locator(b"ssh://devbox/srv/app")  # type: ignore[arg-type]


def test_parse_underscore_host_allowed():
    # Spec charset: host [A-Za-z0-9._-]+ (underscore permitted; underscores
    # appear in ssh_config Host aliases and mDNS names).
    loc = parse_remote_locator("ssh://dev_box/srv/app")
    assert loc.host == "dev_box"


def test_parse_ipv6_flags_and_strips_brackets():
    loc = parse_remote_locator("ssh://me@[2001:db8::1]:2222/srv/app")
    assert loc.ipv6 is True
    assert loc.host == "2001:db8::1"
    assert "[" not in loc.host and "]" not in loc.host


def test_parse_bare_host_not_flagged_ipv6():
    loc = parse_remote_locator("ssh://devbox/srv/app")
    assert loc.ipv6 is False


def test_locator_is_frozen():
    loc = parse_remote_locator("ssh://devbox/srv/app")
    with pytest.raises(Exception):
        loc.host = "other"  # type: ignore[misc]


def test_argv_omits_unset_parts():
    loc = parse_remote_locator("ssh://devbox/srv/app")
    assert build_ssh_argv(loc, [], []) == ["--", "devbox"]


def test_argv_options_and_command_passthrough_order():
    loc = parse_remote_locator("ssh://devbox/srv/app")
    argv = build_ssh_argv(loc, ["-o", "X=y"], ["python3", "-I", "-c", "pass"])
    assert argv == ["-o", "X=y", "--", "devbox", "python3", "-I", "-c", "pass"]


@pytest.mark.parametrize("raw,expected", [
    ("ssh://me@devbox:2222/srv/app", "ssh://me@devbox:2222/srv/app"),
    ("ssh://devbox/srv/app", "ssh://devbox/srv/app"),
    ("ssh://[2001:db8::1]:2222/srv/app", "ssh://[2001:db8::1]:2222/srv/app"),
    # Rule: explicit default port 22 is omitted in canonical form.
    ("ssh://devbox:22/srv/app", "ssh://devbox/srv/app"),
    ("ssh://[::1]:22/srv/app", "ssh://[::1]/srv/app"),
    # Normalized path: '.' collapsed, trailing slash dropped.
    ("ssh://devbox/srv/./app/", "ssh://devbox/srv/app"),
])
def test_locator_string(raw, expected):
    assert locator_string(parse_remote_locator(raw)) == expected


@pytest.mark.parametrize("raw", [
    "ssh://me@devbox:2222/srv/app",
    "ssh://devbox/srv/app",
    "ssh://[2001:db8::1]/srv/app",
])
def test_locator_string_round_trips(raw):
    # Canonical form re-parses to an equivalent locator (an explicit :22
    # canonicalizes to port=None per the documented omission rule).
    loc = parse_remote_locator(raw)
    assert parse_remote_locator(locator_string(loc)) == loc


def test_fingerprint_is_sha256_hex():
    fp = sha256_fingerprint("me", "devbox", 2222, "/srv/app")
    assert fp == hashlib.sha256(b"me@devbox:2222/srv/app").hexdigest()
    assert len(fp) == 64
    assert fp == fp.lower()


def test_fingerprint_defaults_user_and_port():
    # None user serializes as empty; None port serializes as the SSH
    # default 22 — so those spellings are fingerprint-equivalent.
    assert sha256_fingerprint(None, "h", None, "/x") == sha256_fingerprint("", "h", 22, "/x")


def test_fingerprint_normalizes_path():
    assert sha256_fingerprint("me", "h", 22, "/srv/./app/") == sha256_fingerprint(
        "me", "h", 22, PurePosixPath("/srv/app")
    )


def test_fingerprint_distinguishes_identity_parts():
    base = sha256_fingerprint("me", "devbox", 22, "/srv/app")
    assert base != sha256_fingerprint("me2", "devbox", 22, "/srv/app")
    assert base != sha256_fingerprint("me", "devbox2", 22, "/srv/app")
    assert base != sha256_fingerprint("me", "devbox", 2222, "/srv/app")
    assert base != sha256_fingerprint("me", "devbox", 22, "/srv/other")
