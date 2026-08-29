"""Shared ratchet policy, snapshots, and reporting for the four boot budgets.

TASK-23029 / ADR-097 (``backlog/decisions/097-boot-budget-ratchets.md``): the
four boot-cost budgets are RATCHETS -- their constants never rise. This module
is the single home for

* the policy footer every ratchet breach message carries,
* the one-line headroom report each guard emits when it PASSES, and
* the pinned snapshots that let a breach name the culprit (module names,
  CSS segments, pre-import routes) instead of just a total.

Snapshots live in ``Tests/Performance/boot_budget_snapshots/`` and are only
ever written by ``scripts/update_boot_budget_snapshots.py`` (a deliberate,
documented one-liner). Nothing in this module or in the guards writes them:
an accidental regeneration would silently bless whatever the tree currently
costs, which is the failure mode the snapshots exist to catch.

This is intentionally NOT a pytest plugin and NOT part of the production
package: the guards load it by file path (see ``conftest.py`` next to it), so
it can never ride the boot closure it helps police.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Mapping

ADR_REF = "ADR-097, backlog/decisions/097-boot-budget-ratchets.md"

SNAPSHOT_DIR = Path(__file__).resolve().parent / "boot_budget_snapshots"

#: The deliberate snapshot-refresh one-liner (from the repo root).
SNAPSHOT_REFRESH = ".venv/bin/python scripts/update_boot_budget_snapshots.py"

#: Snapshot filenames, keyed by guard name (the same names the update script
#: accepts for ``--only``).
SNAPSHOT_FILES = {
    "boot-import-weight": "boot_import_modules.txt",
    "ui-ready-census": "ui_ready_modules.txt",
    "boot-css-bytes": "boot_css_bytes.json",
    "preimport-payload": "preimport_payload.json",
}


def ratchet_policy(constant: str) -> str:
    """The policy footer for a ratchet breach message.

    Args:
        constant: The budget constant's name, so the reader knows exactly
            which number they must NOT touch.

    Returns:
        A short, CI-log-readable statement of the ratchet rule and the three
        legitimate responses to a breach.
    """
    return (
        f"RATCHET ({ADR_REF}): {constant} never rises. Legitimate responses: "
        "(a) defer the cost off this path, (b) shed equivalent cost elsewhere "
        "in the same PR, or (c) an explicit owner exception recorded in the "
        "ADR's exception ledger. Raising the constant is NOT one of the "
        "options -- a PR that raises it without a ledger entry should be "
        "rejected in review."
    )


def emit_headroom(line: str) -> str:
    """Emit a guard's one-line headroom report on PASS.

    Printed (visible under ``pytest -s``) AND raised as a ``UserWarning`` so
    it lands in pytest's warnings summary -- the only per-test channel a
    default CI invocation shows for a PASSING test.

    Args:
        line: The already-formatted stable one-liner.

    Returns:
        The line, unchanged (convenient for tests of the format).
    """
    print(line)
    warnings.warn(line, stacklevel=2)
    return line


def headroom_line(guard: str, pairs: Iterable[tuple[str, int, int]]) -> str:
    """Format the stable one-line headroom report.

    Args:
        guard: Guard name (e.g. ``boot-import-weight``).
        pairs: ``(unit, used, limit)`` tuples, one per budgeted axis.

    Returns:
        e.g. ``boot-import-weight: 650/660 modules (headroom 10)``.
    """
    parts = [
        f"{used}/{limit} {unit} (headroom {limit - used})"
        for unit, used, limit in pairs
    ]
    return f"{guard}: " + "; ".join(parts)


def _snapshot_path(guard: str) -> Path:
    return SNAPSHOT_DIR / SNAPSHOT_FILES[guard]


def load_module_snapshot(guard: str) -> set[str]:
    """Read a pinned module-name snapshot (one name per line, ``#`` comments).

    Args:
        guard: ``boot-import-weight`` or ``ui-ready-census``.

    Returns:
        The pinned module set. Empty only if the file is missing, which the
        snapshot sanity test treats as a failure in its own right.
    """
    path = _snapshot_path(guard)
    if not path.is_file():
        return set()
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def load_json_snapshot(guard: str) -> dict:
    """Read a pinned JSON snapshot (CSS bytes / pre-import payload).

    Args:
        guard: ``boot-css-bytes`` or ``preimport-payload``.

    Returns:
        The parsed snapshot, or ``{}`` if the file is missing.
    """
    path = _snapshot_path(guard)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_module_diff(live: Iterable[str], guard: str) -> str:
    """Directional module-set diff against the pinned snapshot.

    The house pattern (TASK-23028's ``EXPECTED_CLOCK_ROOTS``): additions and
    removals are listed separately, each with its own guidance, so a breach
    names the modules that consumed the headroom instead of just the total.

    Args:
        live: The measured module set.
        guard: Which snapshot to diff against.

    Returns:
        A multi-line report (or a note that the snapshot is missing/stale).
    """
    pinned = load_module_snapshot(guard)
    snapshot = SNAPSHOT_FILES[guard]
    if not pinned:
        return (
            f"(no snapshot at boot_budget_snapshots/{snapshot} -- regenerate "
            f"deliberately with `{SNAPSHOT_REFRESH}`)"
        )
    live_set = set(live)
    added = sorted(live_set - pinned)
    removed = sorted(pinned - live_set)
    lines: list[str] = [f"vs pinned snapshot boot_budget_snapshots/{snapshot}:"]
    if added:
        lines.append(
            f"  NEW modules ({len(added)}) -- these consumed the headroom; "
            "defer them or shed elsewhere:"
        )
        lines.extend(f"    + {name}" for name in added)
    if removed:
        lines.append(
            f"  modules no longer resident ({len(removed)}) -- sheds since "
            "the snapshot was pinned:"
        )
        lines.extend(f"    - {name}" for name in removed)
    if not added and not removed:
        lines.append(
            "  set identical to the snapshot -- the snapshot itself was "
            "captured over budget; find the culprit by diffing against the "
            "last in-budget state instead."
        )
    return "\n".join(lines)


def format_name_delta(
    live: Iterable[str],
    pinned: Iterable[str],
    noun: str,
    added_note: str = "",
) -> str:
    """Directional name-set delta (the ``+``/``-`` house pattern), unpinned.

    Like :func:`format_module_diff` but against a caller-supplied reference
    set instead of a named snapshot file.

    Args:
        live: Measured names.
        pinned: Reference names.
        noun: What a name is (``module``, ``route``, ...).
        added_note: Optional guidance appended to the additions heading.

    Returns:
        The delta block, or a one-line "identical" note.
    """
    live_set, pinned_set = set(live), set(pinned)
    added = sorted(live_set - pinned_set)
    removed = sorted(pinned_set - live_set)
    lines: list[str] = []
    if added:
        heading = f"  NEW {noun}s ({len(added)})"
        if added_note:
            heading += f" -- {added_note}"
        lines.append(heading + ":")
        lines.extend(f"    + {name}" for name in added)
    if removed:
        lines.append(f"  {noun}s no longer present ({len(removed)}):")
        lines.extend(f"    - {name}" for name in removed)
    if not lines:
        lines.append(f"  {noun} set identical to the snapshot.")
    return "\n".join(lines)


def format_byte_diff(
    live: Mapping[str, int],
    pinned: Mapping[str, int],
    label: str,
    limit: int = 20,
) -> str:
    """Per-key byte/LOC delta report against a pinned mapping.

    Args:
        live: Measured ``{name: size}``.
        pinned: Snapshot ``{name: size}``.
        label: What a key is (``segment``, ``route``, ...), for the headings.
        limit: Cap on printed changed rows (largest |delta| first).

    Returns:
        A multi-line report: grown/shrunk keys with signed deltas, then keys
        added or removed outright.
    """
    if not pinned:
        return (
            f"(no pinned {label} snapshot -- regenerate deliberately with "
            f"`{SNAPSHOT_REFRESH}`)"
        )
    changed = [
        (name, live[name] - pinned[name])
        for name in live.keys() & pinned.keys()
        if live[name] != pinned[name]
    ]
    changed.sort(key=lambda item: -abs(item[1]))
    added = sorted(live.keys() - pinned.keys())
    removed = sorted(pinned.keys() - live.keys())
    lines: list[str] = []
    if changed:
        lines.append(f"  {label}s that changed size (largest first):")
        lines.extend(
            f"    {name}: {pinned[name]:,} -> {live[name]:,} ({delta:+,})"
            for name, delta in changed[:limit]
        )
        if len(changed) > limit:
            lines.append(f"    ... and {len(changed) - limit} more")
    if added:
        lines.append(f"  NEW {label}s:")
        lines.extend(f"    + {name}: {live[name]:,}" for name in added)
    if removed:
        lines.append(f"  removed {label}s:")
        lines.extend(f"    - {name}: was {pinned[name]:,}" for name in removed)
    if not lines:
        lines.append(f"  no {label}-level differences vs the snapshot.")
    return "\n".join(lines)


def snapshot_drift_suffix(live: Iterable[str], guard: str) -> str:
    """A compact drift marker for the headroom line (empty when in sync).

    Args:
        live: The measured module set.
        guard: Which snapshot to diff against.

    Returns:
        e.g. ``"; snapshot drift +3/-1"`` or ``""``.
    """
    pinned = load_module_snapshot(guard)
    if not pinned:
        return "; snapshot missing"
    live_set = set(live)
    added = len(live_set - pinned)
    removed = len(pinned - live_set)
    if not added and not removed:
        return ""
    return f"; snapshot drift +{added}/-{removed}"
