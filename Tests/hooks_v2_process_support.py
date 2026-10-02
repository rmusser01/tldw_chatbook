"""Repository-owned isolation bootstrap for real v2 hook subprocess controls.

Keep module imports stdlib-only: bootstrap installs the child policy before
importing the application or keyring. Python -I ignores ambient PYTHONPATH and
sitecustomize; the explicit checkout and policy are therefore reproducible.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


class TestNetworkRefused(OSError):
    """The child fixture refused network access before any OS connection."""


def _refuse_dns(*_args, **_kwargs):
    raise TestNetworkRefused("hook test DNS refused")


def bootstrap(checkout: str, environment: dict[str, str], data_root: str) -> None:
    """Install policy and assert exact child checkout/profile provenance."""
    os.environ.update(environment)
    connect = socket.socket.connect
    connect_ex = socket.socket.connect_ex

    def guarded_connect(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise TestNetworkRefused("hook test network refused")
        return connect(sock, address)

    def guarded_connect_ex(sock, address):
        if sock.family in (socket.AF_INET, socket.AF_INET6):
            raise TestNetworkRefused("hook test network refused")
        return connect_ex(sock, address)

    socket.socket.connect = guarded_connect
    socket.socket.connect_ex = guarded_connect_ex
    socket.getaddrinfo = _refuse_dns
    for family in (socket.AF_INET, socket.AF_INET6):
        with socket.socket(family) as probe:
            for method in (probe.connect, probe.connect_ex):
                try:
                    method(("127.0.0.1", 9))
                except TestNetworkRefused:
                    pass
                else:
                    raise AssertionError("child network refusal is inactive")
    try:
        socket.getaddrinfo("example.invalid", 9)
    except TestNetworkRefused:
        pass
    else:
        raise AssertionError("child DNS refusal is inactive")

    import keyring

    import tldw_chatbook
    from tldw_chatbook.config import get_user_data_dir

    assert (
        Path(tldw_chatbook.__file__).resolve()
        == Path(checkout) / "tldw_chatbook/__init__.py"
    )
    assert type(keyring.get_keyring()).__module__ == "keyring.backends.null"
    assert Path(data_root).is_dir()
    assert get_user_data_dir().resolve() == Path(data_root)
    assert socket.getaddrinfo is _refuse_dns


def child_argv(code: str) -> list[str]:
    """Create a child with the current pytest fixture's exact private profile."""
    from tldw_chatbook.config import get_user_data_dir

    assert os.environ.get("TLDW_TEST_MODE") == "1"
    checkout = Path(__file__).resolve().parents[1]
    data_root = get_user_data_dir().resolve()
    environment = {
        key: os.environ[key]
        for key in ("HOME", "XDG_DATA_HOME", "XDG_CONFIG_HOME", "TLDW_CONFIG_PATH")
    }
    environment["PYTHON_KEYRING_BACKEND"] = "keyring.backends.null.Keyring"
    # The repository fixture owns these actual roots. No platform-specific
    # temporary-directory prefix or external child bootstrap is assumed.
    assert data_root.is_dir()
    assert any(
        data_root.is_relative_to(Path(environment[key]).resolve())
        for key in ("HOME", "XDG_DATA_HOME")
    )
    source = (
        f"import sys;sys.path.insert(0,{str(checkout)!r});"
        "from Tests.hooks_v2_process_support import bootstrap;"
        f"bootstrap({str(checkout)!r},{environment!r},{str(data_root)!r});" + code
    )
    return [sys.executable, "-I", "-c", source]
