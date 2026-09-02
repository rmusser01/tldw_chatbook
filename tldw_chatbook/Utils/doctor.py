"""TASK-25906: aggregate health-check ("doctor") surface.

Chatbook is local-first, so the user owns the whole stack. This aggregates the
checks that already exist -- it adds no new probing logic -- into one pass/fail
surface: config load status, optional-dependency availability, DB integrity,
configured-provider readiness, and private-path posture.

Design constraints:
- Each check REUSES an existing implementation (AC#2).
- Network calls are opt-in, never run by default (AC#3).
- A failing check names a remediation where one is known (AC#4).
- No secret ever appears -- provider readiness reports configured/not, never a
  key (AC#5).
- It runs even when config load has failed; one failing check never aborts the
  rest (AC#6).
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

_UNSET = object()

_STATUS_ORDER = {"fail": 0, "warn": 1, "pass": 2, "skip": 3}


@dataclass(frozen=True)
class DoctorCheck:
    """One health check's outcome."""

    name: str
    status: str  # "pass" | "fail" | "warn" | "skip"
    detail: str
    remediation: Optional[str] = None


# --- individual checks (pure; inject the input for testability) -------------

def check_config_load(load_failure: object = _UNSET) -> DoctorCheck:
    """Config parsed and is in effect (AC#1/#4)."""
    if load_failure is _UNSET:
        from ..config import get_config_load_failure
        load_failure = get_config_load_failure()
    if load_failure is None:
        return DoctorCheck("config", "pass", "configuration loaded")
    path = getattr(load_failure, "path", "?")
    message = getattr(load_failure, "message", str(load_failure))
    return DoctorCheck(
        "config",
        "fail",
        f"configuration at {path} did not load: {message}",
        remediation="fix the reported TOML error; the last-known-good/defaults are in effect until you do",
    )


def check_optional_dependencies(available: Optional[Mapping[str, bool]] = None) -> DoctorCheck:
    """Optional extras availability (AC#2). Missing extras WARN, never FAIL."""
    if available is None:
        from .optional_deps import DEPENDENCIES_AVAILABLE
        available = DEPENDENCIES_AVAILABLE
    present = sorted(k for k, v in available.items() if v)
    missing = sorted(k for k, v in available.items() if not v)
    if not missing:
        return DoctorCheck(
            "optional-dependencies", "pass",
            f"all {len(present)} optional feature groups available",
        )
    return DoctorCheck(
        "optional-dependencies", "warn",
        f"{len(missing)} optional feature group(s) not installed: {', '.join(missing)}",
        remediation="install the matching extra (pip install 'tldw_chatbook[<extra>]') if you want that feature",
    )


def check_database_integrity(integrity_fn: Optional[Callable[[], bool]] = None) -> DoctorCheck:
    """Main DB integrity, via the existing PRAGMA integrity_check (AC#2)."""
    if integrity_fn is None:
        integrity_fn = _default_integrity_fn
    try:
        ok = integrity_fn()
    except Exception as exc:  # noqa: BLE001 - a check must report, not crash
        return DoctorCheck(
            "database", "fail", f"integrity check could not run: {exc}",
            remediation="ensure the database file is present and not locked by another process",
        )
    if ok:
        return DoctorCheck("database", "pass", "integrity check passed")
    return DoctorCheck(
        "database", "fail", "integrity check reported corruption",
        remediation="restore from a backup; a corrupt SQLite file cannot be repaired in place",
    )


def check_provider_readiness(providers: Optional[Sequence[str]] = None) -> DoctorCheck:
    """Which providers have a usable key configured -- names only, NEVER the
    key value (AC#5)."""
    if providers is None:
        from ..config import get_detected_api_providers
        providers = get_detected_api_providers()
    names = sorted(str(p) for p in providers)
    if names:
        return DoctorCheck(
            "providers", "pass",
            f"{len(names)} provider(s) configured: {', '.join(names)}",
        )
    return DoctorCheck(
        "providers", "warn", "no API providers are configured",
        remediation="add a key under [api_settings.<provider>] or set the provider's env var",
    )


def check_private_path_posture(postures: Optional[Sequence[tuple[str, bool]]] = None) -> DoctorCheck:
    """Config/data directories are owner-private (AC#1)."""
    if postures is None:
        postures = _default_path_postures()
    insecure = [path for path, secure in postures if not secure]
    if not insecure:
        return DoctorCheck("private-paths", "pass", "config and data directories are owner-private")
    return DoctorCheck(
        "private-paths", "warn",
        f"loose permissions on: {', '.join(insecure)}",
        remediation="chmod 700 the listed directories so other users cannot read your data",
    )


# --- default (impure) input gatherers, each best-effort ---------------------

def _default_integrity_fn() -> bool:
    from ..config import get_chachanotes_db_lazy
    db = get_chachanotes_db_lazy()
    if db is None:
        raise RuntimeError("main database is not available")
    return bool(db.check_integrity())


def _path_is_owner_private(path: str) -> bool:
    try:
        mode = os.stat(path).st_mode
    except OSError:
        # a not-yet-created dir is not an insecurity to report here
        return True
    return not (mode & (stat.S_IRWXG | stat.S_IRWXO))


def _default_path_postures() -> list[tuple[str, bool]]:
    postures: list[tuple[str, bool]] = []
    try:
        from ..config import get_user_data_dir, get_cli_config_path
        for candidate in (get_user_data_dir(), get_cli_config_path().parent):
            p = str(candidate)
            postures.append((p, _path_is_owner_private(p)))
    except Exception:  # noqa: BLE001 - posture is best-effort
        pass
    return postures


# --- orchestration + rendering ----------------------------------------------

def run_doctor(*, include_network: bool = False) -> list[DoctorCheck]:
    """Run every check, each isolated so one failure can't abort the rest.

    ``include_network`` gates any check that would make a network call; none run
    by default (AC#3). Runs regardless of config state (AC#6).
    """
    checks: list[DoctorCheck] = []
    for name, fn in (
        ("config", check_config_load),
        ("optional-dependencies", check_optional_dependencies),
        ("database", check_database_integrity),
        ("providers", check_provider_readiness),
        ("private-paths", check_private_path_posture),
    ):
        try:
            checks.append(fn())
        except Exception as exc:  # noqa: BLE001 - never let one check abort doctor
            checks.append(DoctorCheck(name, "fail", f"check errored: {exc}"))
    # Network-dependent probes (e.g. live provider reachability) go here,
    # guarded by include_network. None are implemented yet -- the gate exists
    # so adding one can never make doctor phone home by default.
    if include_network:
        checks.append(DoctorCheck(
            "network-probes", "skip", "no network probes are implemented yet",
        ))
    return checks


def format_doctor_report(checks: Sequence[DoctorCheck]) -> str:
    """Render the report, worst-status-first, with remediations (AC#1/#4)."""
    if not checks:
        return "Doctor: no checks ran."
    ordered = sorted(checks, key=lambda c: (_STATUS_ORDER.get(c.status, 9), c.name))
    lines = ["Doctor report:"]
    for check in ordered:
        line = f"  [{check.status.upper()}] {check.name} — {check.detail}"
        if check.remediation and check.status in ("fail", "warn"):
            line += f"\n         → {check.remediation}"
        lines.append(line)
    failures = sum(1 for c in checks if c.status == "fail")
    warnings = sum(1 for c in checks if c.status == "warn")
    lines.append(f"{failures} failing, {warnings} warning, {len(checks)} checks total.")
    return "\n".join(lines)
