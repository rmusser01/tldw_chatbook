#!/usr/bin/env python3
"""Isolation guard for the rp-review live harness.

Every harness script that opens app databases (seed.py, scale_seed.py, persona_more.py,
lore_fix.py) calls this module twice:

  prof = harness_guard.require_profile(...)    # BEFORE importing tldw_chatbook
  harness_guard.check_app_paths(config, prof)  # right AFTER `from tldw_chatbook import config`

The guard is keyed on HARNESS_STATE. It refuses (exit status 1, message starting with
"REFUSING:") when:
  * HARNESS_STATE is unset, relative, "/", the real home, or overlaps the real tldw_cli
    profile (~/.config/tldw_cli, ~/.local/share/tldw_cli);
  * TLDW_CONFIG_PATH, HOME, XDG_CONFIG_HOME or XDG_DATA_HOME is unset, resolves outside
    HARNESS_STATE, or resolves into the real profile;
  * the profile is not HARNESS_STATE/<master> or HARNESS_STATE/runs/<socket>, or is a
    master the calling script may not write;
  * after import, the app's own user-data dir or any core DB path resolves outside the
    profile or into the real profile, or tldw_chatbook was imported from outside APP_WT.

"Real home" comes from the password database, never from $HOME (which the harness
overrides for the app).

CLI (used by env.sh and the self-test):
  harness_guard.py state            -> print the validated, absolute HARNESS_STATE
                                       (env HARNESS_STATE, else <main checkout>/.worktrees/.rp-harness-state)
  harness_guard.py real-profiles    -> print the two real profile directories
"""
from __future__ import annotations

import os
import pwd
import subprocess
import sys
from dataclasses import dataclass

REAL_HOME = os.path.realpath(pwd.getpwuid(os.getuid()).pw_dir)
REAL_PROFILE_DIRS = (
    os.path.join(REAL_HOME, ".config", "tldw_cli"),
    os.path.join(REAL_HOME, ".local", "share", "tldw_cli"),
)
MASTERS = ("golden", "golden.preseed.bak", "empty", "volume")


def refuse(msg: str) -> "None":
    sys.exit(f"REFUSING: {msg}")


def _real(path: str) -> str:
    return os.path.realpath(path)


def _inside(child: str, parent: str) -> bool:
    return child == parent or child.startswith(parent.rstrip(os.sep) + os.sep)


def _forbidden(path: str) -> str | None:
    rp = _real(path)
    for real_dir in REAL_PROFILE_DIRS:
        if _inside(rp, _real(real_dir)):
            return real_dir
    return None


def default_state() -> str:
    """<main checkout>/.worktrees/.rp-harness-state (the main checkout is the parent of
    the git common dir of the checkout that holds this file; .worktrees/ is git-ignored)."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        common = subprocess.run(
            ["git", "-C", here, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        refuse("HARNESS_STATE is unset and the harness is not inside a git checkout")
    return os.path.join(os.path.dirname(common), ".worktrees", ".rp-harness-state")


def _is_tracked_location(path: str) -> bool:
    """True when `path` lies inside a git work tree and is NOT git-ignored there
    (e.g. under Docs/): state must never be committable."""
    probe = path
    while not os.path.isdir(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            return False
        probe = parent
    try:
        top = subprocess.run(["git", "-C", probe, "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False  # not in a git work tree
    rel = os.path.relpath(path, _real(top))
    if rel.startswith(".."):
        return False
    ignored = subprocess.run(["git", "-C", top, "check-ignore", "-q", "--", rel],
                             capture_output=True).returncode == 0
    return not ignored


def check_state_dir(state: str) -> str:
    if not state or not os.path.isabs(state):
        refuse(f"HARNESS_STATE={state!r} is unset or not absolute")
    rs = _real(state)
    if rs in ("/", REAL_HOME) or len(rs.strip(os.sep).split(os.sep)) < 2:
        refuse(f"HARNESS_STATE={rs!r} is too broad")
    for real_dir in REAL_PROFILE_DIRS:
        rd = _real(real_dir)
        if _inside(rs, rd) or _inside(rd, rs):
            refuse(f"HARNESS_STATE={rs!r} overlaps the real profile {real_dir}")
    if _is_tracked_location(rs):
        refuse(f"HARNESS_STATE={rs!r} is inside a git work tree and not git-ignored "
               "(profiles, runs and captures must never be committable; never put them under Docs/)")
    return rs


def state_root() -> str:
    state = os.environ.get("HARNESS_STATE", "")
    if not state:
        refuse("HARNESS_STATE is unset (run this through seedrun.sh)")
    rs = check_state_dir(state)
    if not os.path.isdir(rs):
        refuse(f"HARNESS_STATE={rs!r} does not exist")
    return rs


@dataclass(frozen=True)
class Profile:
    state: str   # resolved HARNESS_STATE
    dir: str     # resolved profile dir (parent of TLDW_CONFIG_PATH)
    kind: str    # "run" or "master"
    name: str    # socket name for a run, master name for a master


def require_profile(*, allow_runs: bool = True, allow_masters: tuple[str, ...] = ()) -> Profile:
    """Validate the isolated env BEFORE tldw_chatbook is imported. Returns the profile."""
    state = state_root()
    for var in ("TLDW_CONFIG_PATH", "HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        val = os.environ.get(var, "")
        if not val or not os.path.isabs(val):
            refuse(f"{var}={val!r} is unset or not absolute (run this through seedrun.sh)")
        hit = _forbidden(val)
        if hit:
            refuse(f"{var}={val!r} resolves into the real profile {hit}")
        if not _inside(_real(val), state):
            refuse(f"{var}={val!r} is not inside HARNESS_STATE={state}")
    prof_dir = os.path.dirname(_real(os.environ["TLDW_CONFIG_PATH"]))
    for var in ("HOME", "XDG_CONFIG_HOME", "XDG_DATA_HOME"):
        if not _inside(_real(os.environ[var]), prof_dir):
            refuse(f"{var} is not inside the profile {prof_dir} that owns TLDW_CONFIG_PATH")
    parts = os.path.relpath(prof_dir, state).split(os.sep)
    if len(parts) == 2 and parts[0] == "runs":
        kind, name = "run", parts[1]
        if not allow_runs:
            refuse(f"this script may not write run copies ({prof_dir})")
    elif len(parts) == 1 and parts[0] not in ("runs", ".", ".."):
        kind, name = "master", parts[0]
        if name not in allow_masters:
            refuse(f"master profile {name!r} is protected for this script "
                   f"(allowed masters: {list(allow_masters) or 'none'})")
    else:
        refuse(f"{prof_dir} is not HARNESS_STATE/<master> or HARNESS_STATE/runs/<socket>")
    return Profile(state=state, dir=prof_dir, kind=kind, name=name)


def check_app_paths(config_module, prof: Profile) -> None:
    """Validate where the imported app will actually read and write."""
    import tldw_chatbook  # noqa: PLC0415 - already imported by the caller

    app_wt = os.environ.get("APP_WT", "")
    pkg = _real(os.path.dirname(tldw_chatbook.__file__))
    if app_wt and not _inside(pkg, _real(app_wt)):
        refuse(f"tldw_chatbook was imported from {pkg}, not from APP_WT={app_wt}")
    checks = [("user data dir", config_module.get_user_data_dir())]
    for label, getter in (("chachanotes db", "get_chachanotes_db_path"),
                          ("prompts db", "get_prompts_db_path"),
                          ("media db", "get_media_db_path")):
        fn = getattr(config_module, getter, None)
        if fn is not None:
            checks.append((label, fn()))
    for label, value in checks:
        rp = _real(str(value))
        hit = _forbidden(rp)
        if hit:
            refuse(f"{label} {rp} resolves into the real profile {hit}")
        if not _inside(rp, prof.dir):
            refuse(f"{label} {rp} is outside the profile {prof.dir}")
    print(f"[guard] profile={prof.dir} ({prof.kind}:{prof.name}) app={pkg}")


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[1] == "state":
        print(check_state_dir(os.environ.get("HARNESS_STATE") or default_state()))
        return 0
    if len(argv) == 2 and argv[1] == "real-profiles":
        print("\n".join(REAL_PROFILE_DIRS))
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
