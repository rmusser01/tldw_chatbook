#!/usr/bin/env python3
"""Print a manifest of the REAL tldw_cli profile; read-only. Diff two manifests to prove a
harness session left it untouched.

Usage: isolation_snap.py > before.txt ; ... ; isolation_snap.py > after.txt ; diff before.txt after.txt

Covers ~/.config/tldw_cli and ~/.local/share/tldw_cli (home from the password database, not
$HOME). For every entry up to depth 2 below each root: type, size, mtime (ns); plus a
SHA-256 of every regular file up to 8 MB at that depth (config.toml, ui_state.toml, the
default_user DB sidecars, ...). Larger files (big DBs) are covered by size + mtime.
On macOS it also lists the metadata (service, account, modification time - never a secret)
of every login-keychain item whose service starts with "tldw_chatbook", so a run that
created or rewrote a keychain item shows up in the diff.
"""
from __future__ import annotations

import hashlib
import os
import pwd
import re
import shutil
import stat
import subprocess
import sys

HOME = pwd.getpwuid(os.getuid()).pw_dir
ROOTS = (os.path.join(HOME, ".config", "tldw_cli"), os.path.join(HOME, ".local", "share", "tldw_cli"))
MAX_DEPTH = 2
HASH_LIMIT = 8 * 1024 * 1024


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def entry(path: str) -> str:
    try:
        st = os.lstat(path)
    except FileNotFoundError:
        return f"{path} ABSENT"
    kind = "d" if stat.S_ISDIR(st.st_mode) else "l" if stat.S_ISLNK(st.st_mode) else "f"
    line = f"{path} {kind} size={st.st_size} mtime_ns={st.st_mtime_ns}"
    if kind == "f" and st.st_size <= HASH_LIMIT:
        try:
            line += f" sha256={sha(path)}"
        except OSError as exc:
            line += f" sha256=ERR({exc.__class__.__name__})"
    return line


def walk(root: str, depth: int, out: list[str]) -> None:
    out.append(entry(root))
    if depth >= MAX_DEPTH or not os.path.isdir(root) or os.path.islink(root):
        return
    try:
        names = sorted(os.listdir(root))
    except OSError as exc:
        out.append(f"{root} LISTERR {exc.__class__.__name__}")
        return
    for name in names:
        walk(os.path.join(root, name), depth + 1, out)


def keychain(out: list[str]) -> None:
    """Attributes only (`security dump-keychain` without -d never reads secrets)."""
    if sys.platform != "darwin" or not shutil.which("security"):
        return
    try:
        dump = subprocess.run(["security", "dump-keychain"], capture_output=True, text=True,
                              timeout=30, errors="replace").stdout
    except (OSError, subprocess.SubprocessError) as exc:
        out.append(f"keychain UNAVAILABLE {exc.__class__.__name__}")
        return
    items = []
    for block in dump.split("keychain: ")[1:]:
        attrs = dict(re.findall(r'"(svce|acct|mdat)"<\w+>=(?:0x[0-9A-F]+\s+)?"?([^"\n]*)', block))
        if attrs.get("svce", "").startswith("tldw_chatbook"):
            items.append(f"keychain svce={attrs.get('svce')} acct={attrs.get('acct')} mdat={attrs.get('mdat', '').replace(chr(92) + '000', '')}")
    out.extend(sorted(items) or ["keychain (no tldw_chatbook items)"])


def main() -> int:
    out: list[str] = []
    for root in ROOTS:
        walk(root, 0, out)
    keychain(out)
    sys.stdout.write("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
