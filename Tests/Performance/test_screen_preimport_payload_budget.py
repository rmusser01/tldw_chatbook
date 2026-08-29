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
MAX_PASS_ADDED_MODULES = 500

#: Total LOC of those modules. Measured 365,692 on 2026-08-25. Headroom
#: ~14k LOC -- deliberately under the +30k the AC requires to land in review.
MAX_PASS_ADDED_LOC = 380_000

#: Marginal LOC cap for any single route. Measured max: library 133,517.
MAX_SINGLE_ROUTE_ADDED_LOC = 145_000

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


def _run_census(tmp_path: Path) -> dict:
    """Walk the pre-importer's route order in an isolated fresh interpreter.

    Fresh interpreter because `sys.modules` is process-global: an earlier
    test importing any screen module would deflate the census. Isolated
    HOME/XDG so importing the app never touches the developer's live
    profile.
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
        [sys.executable, "-c", _CENSUS_SCRIPT],
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
    pinned_modules = {
        m for row in pinned_routes.values() for m in row["modules"]
    }
    return (
        "vs pinned snapshot boot_budget_snapshots/preimport_payload.json:\n"
        + ratchet.format_byte_diff(live_loc, pinned_loc, "route")
        + "\npass-wide module set (order-independent -- trust this over "
        "row attribution):\n"
        + ratchet.format_name_delta(
            live_modules,
            pinned_modules,
            "module",
            added_note="these consumed the headroom; defer them or shed "
            "elsewhere",
        )
    )


@pytest.mark.integration
def test_preimport_pass_payload_stays_within_budget(
    tmp_path: Path, ratchet
) -> None:
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
