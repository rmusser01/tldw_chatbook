"""Payload budget for the whole-registry screen pre-importer (TASK-22214).

The pre-importer (`app.py _preimport_heavy_screens`) compiles every screen
route's import closure on a GIL-holding daemon thread starting 0.2 s after
mount. Between the 2026-08-22 pin and the 2026-08-24 review that payload
grew +99 modules / +74.5k LOC with no guard able to see it: the import-weight
guard measures `import tldw_chatbook.app`, the `_ui_ready` census pins the
pre-importer OFF, and the Packaging closure guards each watch one named seam.
This guard closes that gap: it measures what the PASS ITSELF pays -- the
modules the registry walk adds beyond the boot closure -- so the next +30k
LOC of screen payload lands in review instead of in users' laps.

Census method (mirrors the pass's real starting condition): a fresh
subprocess imports `tldw_chatbook.app`, pre-warms the initial Chat route
(both boot paths complete `_push_initial_screen` before the pass arms, so
chat is always a `sys.modules` dict hit at pass time -- and the Chat leg has
its own guards: `test_ui_ready_module_census.py` and
`test_rag_boot_import_closure.py`), snapshots `sys.modules`, then walks
`TldwCli._screen_preimport_route_order()` calling the exact
`load_screen_class()` the pass calls, recording each route's marginal
tldw_chatbook modules and their LOC.

Measured 2026-08-25 (branch fix/task-22214-perf, exactly reproducible across
runs -- module sets and LOC are deterministic, unlike wall time):

    pass-added (beyond app + chat): 478 modules / 365,692 LOC
    top routes by marginal LOC:
        library                161 mods / 133,517 LOC
        ccp (personas)          66 mods /  53,429 LOC
        settings                50 mods /  56,421 LOC
        watchlists_collections  38 mods /  30,279 LOC
        stts                    26 mods /  19,570 LOC

Re-measured 2026-08-28 (dev b5eaa9cf64, TASK-23029): 488 modules /
374,697 LOC (library alone 166 / 137,494) -- headroom 12 modules /
5,303 LOC.

Budgets sit just above reality with headroom deliberately SMALLER than the
+74.5k LOC growth this task answers (and smaller than the +30k the AC names):

* MAX_PASS_ADDED_LOC / MAX_PASS_ADDED_MODULES -- the drift signal for the
  whole pass.
* MAX_SINGLE_ROUTE_ADDED_LOC -- catches one route ballooning while another
  shrinks (the library route alone grew +43k LOC between the last two pins).

RATCHET (TASK-23029 / ADR-097,
`backlog/decisions/097-boot-budget-ratchets.md`): the three budgets below
never rise. On a breach, defer the screen payload that grew (lazy-import it
behind the seam its route mounts through) or shed equivalent payload
elsewhere in the same PR; the only other path is an explicit owner
exception recorded in the ADR's exception ledger. The breach message diffs
the per-route table AND the pass-wide module set against the pinned
snapshot (`boot_budget_snapshots/preimport_payload.json`) so the grown
route and the exact new modules are named (the module-set diff is
order-independent, so trust it over row-level attribution). Refresh the
snapshot only via `scripts/update_boot_budget_snapshots.py`. When a diet
drops a measured number well below its limit, LOWER the limit to measured
+ standard slack (ADR-097's tightening convention) in that same PR.

The first-send warm-up (TASK-33620.15.13). Since TASK-33620.15.5 the same
daemon thread also imports the Console's first-send modules
(``Chat/console_first_send_modules.py``), which the route walk never ran.
``test_first_send_warmup_payload_stays_within_budget`` walks that list from
the same start (app + chat) in the warm-up's own order and pins its marginal
module count as its own row, ``MAX_FIRST_SEND_WARMUP_ADDED_MODULES``
(snapshot ``boot_budget_snapshots/first_send_warmup.json``). It also fails if
the warm-up newly imports one of ``test_app_import_weight.HEAVY_MODULES``, so
a heavy third-party import (invisible to a tldw-only module count) is named
too. Measured 2026-10-20: 94 modules / 61,281 LOC; re-pinned 2026-10-22:
94 modules / 61,609 LOC (dev grew ``local_tool_provider``); re-pinned
2026-10-30: 95 modules / 61,957 LOC (TASK-33620.15.19 lists the two Library
preparation modules an Automatic first send imports).

Honest blind spots:

* LOC is a proxy for import cost, not a measurement of it -- a small module
  with heavy module-scope work is invisible to this census; the duty-cycle
  pacing (`SCREEN_PREIMPORT_*` in app.py) is what bounds the felt cost.
* Marginal attribution is walk-order-dependent: a module shared by two
  routes bills to whichever the pass reaches first. The walk order here is
  the pass's own order, so the attribution matches what the pass pays, but
  a route-order change can move modules between rows without any real
  growth (totals are order-independent -- trust them over rows).
* Only `tldw_chatbook.*` is counted. A screen adding a heavy third-party
  import grows this census by one line (`import x`) while the real payload
  is invisible; `test_app_import_weight.py`'s HEAVY_MODULES idea applies if
  that becomes a problem.
* The Chat route's own closure is excluded by design (pre-warmed; see
  above): growth there is TASK-22213's census's territory, not this one's.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Total tldw_chatbook modules the registry walk adds beyond app + chat.
#: Measured 478 on 2026-08-25. RATCHET (ADR-097): never rises -- see the
#: module docstring before touching any of the three constants below.
#:
#: TASK-33260 (PERF-01) re-pin, ADR-097 exception ledger 2026-09-28: this
#: guard was not in perf-guard.yml, so it went red on dev unseen (554/500
#: modules, 410,347/378,740 LOC, library 125,111/123,319 LOC at 9cd9aad65f;
#: the 2026-09-27 audit read 409,566 LOC at 840ed2ca58). All three limits
#: are re-pinned at that measurement -- modules exactly, LOC plus ADR-097's standard slack, since
#: LOC moves with every line edited in those 554 modules -- so the guard can
#: run as a PR gate. The paydown back under the old limits is owned by
#: TASK-33276 (PERF-17, boot import diet + pre-import ratchet paydown); that
#: PR lowers these constants again. Modules re-pinned 554 -> 556 on 2026-09-29
#: when #2888 was rebased onto dev 6423c4fbd1: dev had added
#: ``Widgets.Settings_Widgets`` and ``speech_tts_panel_types`` to the Settings
#: route after the first measurement (owner approved 556; ADR-097 ledger).
#: 556 -> 557 on 2026-09-30 after the rebase onto dev 75c06af39a: #2922's
#: ``settings_screen`` imports ``settings_hooks`` at module level (owner
#: approved). 557 -> 556 on 2026-10-03: TASK-33642 made it lazy again.
#: B0 TASK-33910.1: 556 -> 557 (2026-10-03), owner-approved; ADR-097 ledger.
#: B1 TASK-33910.2: 557 -> 558 (2026-10-04), owner-approved; ADR-097 ledger.
MAX_PASS_ADDED_MODULES = 558

#: TASK-31552 pinned 378,740 (363,740 + 15,000 slack). TASK-33260 re-pin:
#: 410,347 measured + 15,000 standard slack; TASK-33276 pays it down.
MAX_PASS_ADDED_LOC = 425_347

#: TASK-31552 pinned 123,319 (Library 113,319 + 10,000 slack). TASK-33260
#: re-pin: Library 125,111 measured + 10,000 slack; TASK-33276 pays it down.
MAX_SINGLE_ROUTE_ADDED_LOC = 135_111

#: tldw_chatbook modules the first-send warm-up adds beyond app + chat.
#: New row, owner decision 2026-10-20 ("Measure, then merge", TASK-33620.15.13):
#: since TASK-33620.15.5 the pass also imports the Console's first-send
#: modules (``Chat/console_first_send_modules.py``) on the same daemon thread,
#: after the last route (TASK-33620.15.14), and the route walk above never ran
#: them. A separate row rather than a raise of MAX_PASS_ADDED_MODULES: the
#: route census stays comparable with its history. This one is measured from
#: the same start (app + chat), so it is an upper bound on what the warm-up
#: adds once the routes have run.
#: Pinned at the measurement on 2026-10-20 (origin/dev 2d071d6d71 plus the
#: warm-up list), modules exactly, no slack. RATCHET (ADR-097): never rises.
#: No LOC row: a LOC pin with no slack would fail every edit that adds a line
#: to one of these Console modules; the per-entry LOC is printed instead.
#: Re-pinned 94 -> 95 on 2026-10-30 (TASK-33620.15.19): a live first send
#: with Library Auto set to Automatic loaded
#: ``Chat.console_library_preparation`` and ``Chat.console_library_search`` on
#: the UI loop, so both joined the list. Measured at origin/dev 7536ac3b87:
#: dev 93 modules, branch 95 (exactly those two); ADR-097 ledger row. Owner
#: decision: PENDING. The task's AC#2 asks for this re-pin, but an agent filed
#: that AC in review; ADR-097 needs the owner's explicit sign-off, quoted in
#: the ledger row, before this merges.
MAX_FIRST_SEND_WARMUP_ADDED_MODULES = 95

_CENSUS_SCRIPT = """
import json
import sys
from types import SimpleNamespace

import tldw_chatbook.app as app_module
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (pre-warm, see docstring)


def tldw_modules():
    return {
        m
        for m in sys.modules
        if m.startswith("tldw_chatbook") and sys.modules[m] is not None
    }


def loc_of(mods):
    total = 0
    for name in mods:
        path = getattr(sys.modules.get(name), "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            with open(path, "rb") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            pass
    return total


routes = app_module.TldwCli._screen_preimport_route_order(SimpleNamespace())
baseline = tldw_modules()
rows = []
for route in routes:
    before = tldw_modules()
    route.load_screen_class()
    added = tldw_modules() - before
    rows.append(
        {
            "route": route.screen_name,
            "added_modules": len(added),
            "added_loc": loc_of(added),
            "modules": sorted(added),
        }
    )
    print(
        "ROUTE:{}:{}:{}".format(
            route.screen_name, len(added), rows[-1]["added_loc"]
        ),
        flush=True,
    )

final = tldw_modules() - baseline
print(
    "BUDGET_JSON:"
    + json.dumps(
        {
            "pass_added_modules": len(final),
            "pass_added_loc": loc_of(final),
            "routes": rows,
        }
    ),
    flush=True,
)
"""


#: The first-send warm-up's census (TASK-33620.15.13): the same starting
#: condition as the route walk (app + the initial Chat route), then
#: ``FIRST_SEND_MODULES`` in the warm-up's own order, one marginal row per
#: listed module. ``argv[1]`` is the heavy third-party set (JSON) whose new
#: arrivals are reported by name.
_WARMUP_CENSUS_SCRIPT = """
import importlib
import json
import sys

import tldw_chatbook.app  # noqa: F401
import tldw_chatbook.UI.Screens.chat_screen  # noqa: F401  (pre-warm, as above)
from tldw_chatbook.Chat.console_first_send_modules import FIRST_SEND_MODULES

heavy = json.loads(sys.argv[1])


def tldw_modules():
    return {
        m
        for m in sys.modules
        if m.startswith("tldw_chatbook") and sys.modules[m] is not None
    }


def loc_of(mods):
    total = 0
    for name in mods:
        path = getattr(sys.modules.get(name), "__file__", None)
        if not path or not path.endswith(".py"):
            continue
        try:
            with open(path, "rb") as fh:
                total += sum(1 for _ in fh)
        except OSError:
            pass
    return total


heavy_before = {m for m in heavy if m in sys.modules}
baseline = tldw_modules()
rows = []
for name in FIRST_SEND_MODULES:
    before = tldw_modules()
    heavy_seen = {m for m in heavy if m in sys.modules}
    importlib.import_module(name)
    added = tldw_modules() - before
    rows.append(
        {
            "entry": name,
            "added_modules": len(added),
            "added_loc": loc_of(added),
            "modules": sorted(added),
            "heavy": sorted(m for m in heavy if m in sys.modules and m not in heavy_seen),
        }
    )

final = tldw_modules() - baseline
print(
    "BUDGET_JSON:"
    + json.dumps(
        {
            "listed": list(FIRST_SEND_MODULES),
            "warmup_added_modules": len(final),
            "warmup_added_loc": loc_of(final),
            "heavy_added": sorted(m for m in heavy if m in sys.modules and m not in heavy_before),
            "entries": rows,
        }
    ),
    flush=True,
)
"""


def _run_census(tmp_path: Path, script: str = _CENSUS_SCRIPT, *args: str) -> dict:
    """Walk the pre-importer's route order in an isolated fresh interpreter.

    Fresh interpreter because `sys.modules` is process-global: an earlier
    test importing any screen module would deflate the census. Isolated
    HOME/XDG so importing the app never touches the developer's live
    profile.

    Args:
        tmp_path: Directory for the subprocess's isolated profile.
        script: The census to run (the route walk by default;
            ``_WARMUP_CENSUS_SCRIPT`` for the first-send warm-up).
        *args: Command-line arguments for ``script``.

    Returns:
        The census payload the script printed after ``BUDGET_JSON:``.
    """
    home = tmp_path / "home"
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700, exist_ok=True)

    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(config_dir),
        "TLDW_TEST_MODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("TLDW_CONFIG_PATH", None)

    result = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=300,
    )
    assert result.returncode == 0, (
        f"payload census subprocess failed (rc={result.returncode}):\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    payloads = [
        line[len("BUDGET_JSON:") :]
        for line in result.stdout.splitlines()
        if line.startswith("BUDGET_JSON:")
    ]
    assert payloads, f"census sentinel missing:\n{result.stdout[-2000:]}"
    return json.loads(payloads[-1])


def _route_table(census: dict) -> str:
    rows = sorted(census["routes"], key=lambda r: -r["added_loc"])
    return "\n".join(
        f"  {r['route']:<28} {r['added_modules']:>4} mods {r['added_loc']:>8} LOC"
        for r in rows
    )


def _snapshot_diff(census: dict, ratchet) -> str:
    """Diff the live census against the pinned snapshot, naming culprits.

    Two views: per-route LOC deltas (attribution is walk-order-dependent, see
    the module docstring) and the pass-wide module-name delta, which is
    order-independent and therefore the authoritative culprit list.

    Args:
        census: The live census payload (``routes`` rows with ``modules``).
        ratchet: shared ratchet helper.

    Returns:
        A multi-line report block.
    """
    snapshot = ratchet.load_json_snapshot("preimport-payload")
    pinned_routes = snapshot.get("routes", {})
    if not pinned_routes:
        return (
            "(no pinned snapshot at boot_budget_snapshots/"
            "preimport_payload.json -- regenerate deliberately with "
            f"`{ratchet.SNAPSHOT_REFRESH}`)"
        )
    live_loc = {r["route"]: r["added_loc"] for r in census["routes"]}
    pinned_loc = {name: row["loc"] for name, row in pinned_routes.items()}
    live_modules = {m for r in census["routes"] for m in r.get("modules", [])}
    pinned_modules = {m for row in pinned_routes.values() for m in row["modules"]}
    return (
        "vs pinned snapshot boot_budget_snapshots/preimport_payload.json:\n"
        + ratchet.format_byte_diff(live_loc, pinned_loc, "route")
        + "\npass-wide module set (order-independent -- trust this over "
        "row attribution):\n"
        + ratchet.format_name_delta(
            live_modules,
            pinned_modules,
            "module",
            added_note="these consumed the headroom; defer them or shed elsewhere",
        )
    )


@pytest.mark.integration
def test_preimport_pass_payload_stays_within_budget(tmp_path: Path, ratchet) -> None:
    """The registry walk's total marginal payload stays at its pinned size.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
        ratchet: shared ratchet helper (see ``conftest.py``).
    """
    census = _run_census(tmp_path)
    # Printed on pass too (visible under `-s`): the raise procedure in the
    # module docstring needs the current table, and a table only visible on
    # failure cannot be diffed against the pinned numbers.
    print(
        f"\nPREIMPORT PAYLOAD CENSUS: {census['pass_added_modules']} modules / "
        f"{census['pass_added_loc']} LOC added by the registry walk\n"
        f"{_route_table(census)}"
    )

    # Anti-vacuity: an empty or near-empty census means the walk did not
    # actually import the registry (e.g. the pre-warm accidentally pulled
    # everything, or route resolution broke) -- a budget trivially "met" by
    # measuring nothing is the tests-that-cannot-fail failure mode.
    assert census["pass_added_modules"] > 300, (
        f"census looks degenerate: only {census['pass_added_modules']} modules "
        f"added by the whole registry walk\n{_route_table(census)}"
    )
    heavy_routes = [r["route"] for r in census["routes"] if r["added_loc"] > 20_000]
    assert "library" in heavy_routes and "settings" in heavy_routes, (
        "census looks degenerate -- library/settings did not import their "
        f"closures:\n{_route_table(census)}"
    )

    diff = _snapshot_diff(census, ratchet)
    refresh = f"Deliberate snapshot refresh: `{ratchet.SNAPSHOT_REFRESH}`"
    assert census["pass_added_modules"] <= MAX_PASS_ADDED_MODULES, (
        f"pre-importer pass adds {census['pass_added_modules']} tldw modules "
        f"(ratchet limit {MAX_PASS_ADDED_MODULES}). Per-route census:\n"
        f"{_route_table(census)}\n{diff}\n"
        f"{ratchet.ratchet_policy('MAX_PASS_ADDED_MODULES')}\n{refresh}"
    )
    assert census["pass_added_loc"] <= MAX_PASS_ADDED_LOC, (
        f"pre-importer pass adds {census['pass_added_loc']} LOC "
        f"(ratchet limit {MAX_PASS_ADDED_LOC}). Per-route census:\n"
        f"{_route_table(census)}\n{diff}\n"
        f"{ratchet.ratchet_policy('MAX_PASS_ADDED_LOC')}\n{refresh}"
    )

    fattest = max(census["routes"], key=lambda r: r["added_loc"])
    assert fattest["added_loc"] <= MAX_SINGLE_ROUTE_ADDED_LOC, (
        f"route '{fattest['route']}' alone adds {fattest['added_loc']} LOC "
        f"(per-route ratchet limit {MAX_SINGLE_ROUTE_ADDED_LOC}). "
        f"Full census:\n{_route_table(census)}\n{diff}\n"
        f"{ratchet.ratchet_policy('MAX_SINGLE_ROUTE_ADDED_LOC')}\n{refresh}"
    )
    ratchet.emit_headroom(
        ratchet.headroom_line(
            "preimport-payload",
            [
                ("modules", census["pass_added_modules"], MAX_PASS_ADDED_MODULES),
                ("LOC", census["pass_added_loc"], MAX_PASS_ADDED_LOC),
                (
                    f"LOC fattest-route ({fattest['route']})",
                    fattest["added_loc"],
                    MAX_SINGLE_ROUTE_ADDED_LOC,
                ),
            ],
        )
    )


def _heavy_modules() -> tuple[str, ...]:
    """The boot guard's heavy third-party set, read from its own file."""
    path = Path(__file__).with_name("test_app_import_weight.py")
    spec = importlib.util.spec_from_file_location("tldw_perf_heavy_modules", path)
    assert spec is not None and spec.loader is not None, f"unloadable: {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return tuple(module.HEAVY_MODULES)


def _warmup_table(census: dict) -> str:
    rows = sorted(census["entries"], key=lambda r: -r["added_modules"])
    return "\n".join(
        f"  {r['entry'].removeprefix('tldw_chatbook.'):<52} "
        f"{r['added_modules']:>4} mods {r['added_loc']:>7} LOC"
        for r in rows
        if r["added_modules"]
    )


def _warmup_snapshot_diff(census: dict, ratchet) -> str:
    """Name what grew: new list entries, and new modules with the entry that pulls each in.

    Args:
        census: The live warm-up census (``entries`` rows with ``modules``).
        ratchet: shared ratchet helper.

    Returns:
        A multi-line report block.
    """
    snapshot = ratchet.load_json_snapshot("first-send-warmup")
    pinned_entries = snapshot.get("entries", {})
    if not pinned_entries:
        return (
            "(no pinned snapshot at boot_budget_snapshots/first_send_warmup.json "
            f"-- regenerate deliberately with `{ratchet.SNAPSHOT_REFRESH}`)"
        )
    pinned_modules = {m for row in pinned_entries.values() for m in row["modules"]}
    brought_by = {m: r["entry"] for r in census["entries"] for m in r["modules"]}
    new_modules = sorted(set(brought_by) - pinned_modules)
    lines = [
        "vs pinned snapshot boot_budget_snapshots/first_send_warmup.json:",
        "listed entries:",
        ratchet.format_name_delta(
            census["listed"],
            snapshot.get("listed", []),
            "listed module",
            added_note="added to FIRST_SEND_MODULES",
        ),
    ]
    if new_modules:
        lines.append(
            f"  NEW modules ({len(new_modules)}) -- these consumed the headroom; "
            "defer them or shed elsewhere (pulled in by):"
        )
        lines.extend(
            f"    + {name} (by {brought_by[name].removeprefix('tldw_chatbook.')})"
            for name in new_modules
        )
    gone = sorted(pinned_modules - set(brought_by))
    if gone:
        lines.append(f"  modules no longer imported ({len(gone)}):")
        lines.extend(f"    - {name}" for name in gone)
    if not new_modules and not gone:
        lines.append("  module set identical to the snapshot.")
    return "\n".join(lines)


@pytest.mark.integration
def test_first_send_warmup_payload_stays_within_budget(tmp_path: Path, ratchet) -> None:
    """The pass's first-send warm-up adds no module and no heavy dependency unseen.

    TASK-33620.15.13: the warm-up runs on the pass's thread after the last
    route (TASK-33620.15.14), so its imports are pass payload too. Counted
    from app + chat, the same start as the route walk: an upper bound on what
    it adds once the routes have run. Red when a module is added to
    ``FIRST_SEND_MODULES`` or a listed module gains a top-level import of
    another tldw module (the message names it and the entry that pulls it
    in), or when the warm-up newly imports one of the boot guard's heavy
    third-party modules.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
        ratchet: shared ratchet helper (see ``conftest.py``).
    """
    heavy = _heavy_modules()
    census = _run_census(tmp_path, _WARMUP_CENSUS_SCRIPT, json.dumps(heavy))
    print(
        f"\nFIRST-SEND WARM-UP CENSUS: {census['warmup_added_modules']} modules / "
        f"{census['warmup_added_loc']} LOC added beyond app + chat\n"
        f"{_warmup_table(census)}"
    )

    # Anti-vacuity: the warm-up's closure is about ninety modules, and the
    # hooks package a first send runs is in it.
    assert census["warmup_added_modules"] > 60, _warmup_table(census)
    hooks = next(
        r
        for r in census["entries"]
        if r["entry"] == "tldw_chatbook.Agents.hooks_v2.lifecycle"
    )
    assert hooks["added_modules"] > 0, "the hooks package was already resident"

    heavy_rows = [(r["entry"], r["heavy"]) for r in census["entries"] if r["heavy"]]
    assert not census["heavy_added"], (
        f"the first-send warm-up imports heavy dependencies {census['heavy_added']} "
        f"(by entry: {heavy_rows}); defer them behind the call that needs them"
    )

    diff = _warmup_snapshot_diff(census, ratchet)
    refresh = f"Deliberate snapshot refresh: `{ratchet.SNAPSHOT_REFRESH}`"
    assert census["warmup_added_modules"] <= MAX_FIRST_SEND_WARMUP_ADDED_MODULES, (
        f"the first-send warm-up adds {census['warmup_added_modules']} tldw modules "
        f"(ratchet limit {MAX_FIRST_SEND_WARMUP_ADDED_MODULES}). Per entry:\n"
        f"{_warmup_table(census)}\n{diff}\n"
        f"{ratchet.ratchet_policy('MAX_FIRST_SEND_WARMUP_ADDED_MODULES')}\n{refresh}"
    )
    ratchet.emit_headroom(
        ratchet.headroom_line(
            "first-send-warmup",
            [
                (
                    "modules",
                    census["warmup_added_modules"],
                    MAX_FIRST_SEND_WARMUP_ADDED_MODULES,
                )
            ],
        )
    )
