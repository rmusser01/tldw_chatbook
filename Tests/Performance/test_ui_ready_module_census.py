"""`sys.modules` census at `_ui_ready` (TASK-22213).

The import-weight guard (`test_app_import_weight.py`) and the closure guards
(`Tests/Packaging/test_*_closure.py`) all measure the IMPORT phase -- and the
2026-08-24 holistic perf review (finding 22213) showed warm boot-to-ready
regressing ~11% while every one of them stayed green, because the growth was
on the legs they cannot see: modules imported while the initial Chat screen
MOUNTS. A deferral that merely moves an import from module scope into a
mount-path function keeps every import guard green and the user waiting
exactly as long. This guard closes that class: it boots the real app
headless (Textual Pilot) against a scratch profile, waits for
``TldwCli._ui_ready`` -- the same flag the review's TTI probes measure to --
and censuses this repo's modules at that moment.

What it pins:

* ``MAX_TLDW_MODULES_AT_UI_READY`` -- the drift budget for the WARM boot:
  the profile is created by a throwaway first boot inside the test, and the
  census measures the second boot -- the recurring user experience, the
  same condition as the review's TTI metric. Measured 938-939 across
  consecutive warm boots, 2026-08-25, this branch (run-to-run wobble
  observed: +/-1). Budget 970: ~30 modules of headroom, mirroring the
  import-weight guard's just-above-reality philosophy. Re-measured 963 on
  2026-08-28 (dev b5eaa9cf64, TASK-23029): headroom is down to 7. Do NOT
  re-baseline against a fresh-profile boot: that measures ~975 and includes
  the first-boot residents below.
* The heavy deferred families stay off the whole first-paint window, not
  just off the import phase: ``Chunking``, ``RAG_Search.simplified``
  (TASK-21731's packages), and the trajectory family TASK-22213 deferred.

Known resident, deliberately NOT asserted absent: ``Internal_Prompts``
(10 modules). It is off the Chat IMPORT leg (see
``test_rag_boot_import_closure.py``), but the mount path still resolves it:
``chat_screen._ensure_console_agent_bridge`` imports
``Chat/console_agent_bridge.py``, whose module-scope catalog constants
(``CONSOLE_AGENT_OPERATING_PROMPT``, ``_KNOWN_SUBAGENT_PREFIXES``) need the
catalog. Measured marginal cost of the package with the app already
imported: **1.0-2.4 ms warm** -- not worth touching the security-relevant
``_is_subagent`` prefix-seeding mechanism for (the stability-over-quick-wins
ruling). If the bridge edge is ever made lazy, ADD the prefix to
``ABSENT_AT_READY_PREFIXES`` in the same commit.

RATCHET (TASK-23029 / ADR-097,
``backlog/decisions/097-boot-budget-ratchets.md``):
``MAX_TLDW_MODULES_AT_UI_READY`` never rises. On a breach, defer the new
mount-leg cost or shed equivalent cost elsewhere in the same PR; the only
other path is an explicit owner exception recorded in the ADR's exception
ledger. The breach message diffs the census against the pinned snapshot
(``boot_budget_snapshots/ui_ready_modules.txt``) so the new residents are
named; because a warm boot wobbles +/-1 module run-to-run, the snapshot is
diagnostic (it feeds the breach message and the headroom drift marker), not
a hard equality pin. Refresh it only via
``scripts/update_boot_budget_snapshots.py``. When a diet drops the measured
number well below the limit, LOWER the limit to measured + standard slack
(ADR-097's tightening convention) in that same PR.

First-boot residents (found by this guard's own first RED run, traced with
an ``__import__``-stack wrapper, 2026-08-25): a FRESH profile's very first
boot has the entire ``Chunking`` engine (34 modules) resident at ready,
via ``app.py _init_media_db -> Client_Media_DB_v2._initialize_schema ->
_apply_migration_v6_to_v7 -> Chunking._template_conversion``. That is
legitimate one-time migration work, so this guard warms the profile first
and censuses the SECOND boot -- which means first-boot/post-upgrade
residency is a documented blind spot of this census, not a covered case
(it is the 22200 post-upgrade-window family; a fix there should not need
this guard's permission).

Documented blind spots (be honest about what a census cannot see):

* It measures RESIDENCY, not time. Work moved into a function of an
  already-resident module still slows the mount invisibly; only a TTI probe
  (the review's interleaved method) sees that.
* The whole-registry screen pre-importer is pinned OFF here
  (``TLDW_SCREEN_PREIMPORT=0``): its payload lands seconds after ready on a
  daemon thread and would otherwise race the census nondeterministically.
  Pre-importer payload growth is finding 22214's territory, not this
  guard's.
* Deferred-startup timers (footer status, audio services, media cleanup;
  0.1-0.2 s) fire after the census moment: this is a first-paint snapshot,
  not a steady-state one. A module they import is invisible here.
* Only ``tldw_chatbook.*`` is budgeted. Third-party residency varies with
  installed extras, exactly as ``test_app_import_weight.py`` documents.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Drift budget for this repo's own modules resident at `_ui_ready` on a
#: WARM (second) boot. Measured 938-939 on 2026-08-25. RATCHET (ADR-097):
#: this constant never rises -- see the module docstring before touching it.
#: 970 -> 972 (2026-08-30, PR #2223): the network TLS trust policy
#: (task-21513) legitimately adds exactly one UI-ready resident
#: (``Utils.tls_trust`` -- imported eagerly for the settings category and
#: outbound-client policy); measured 969 macOS / 971 linux CI, so 972 keeps
#: the documented +/-1 wobble headroom. Snapshot refreshed via
#: ``scripts/update_boot_budget_snapshots.py --only ui-ready``.
MAX_TLDW_MODULES_AT_UI_READY = 972

#: Families that must not be resident anywhere in the first-paint window.
#: The two package prefixes are TASK-21731's; the exact module names are the
#: trajectory family TASK-22213 took off the Chat leg.
ABSENT_AT_READY_PREFIXES = (
    "tldw_chatbook.Chunking",
    "tldw_chatbook.RAG_Search.simplified",
)
ABSENT_AT_READY_MODULES = (
    "tldw_chatbook.UI.Screens.trajectory_screen",
    "tldw_chatbook.Chat.trajectory_import",
    "tldw_chatbook.Chat.trajectory_export",
    "tldw_chatbook.UI.Widgets.trajectory_timeline",
    "tldw_chatbook.UI.Widgets.trace_filter_bar",
    # TASK-23023: the Research_Workspace facade is lazy (PEP 562), so the
    # screen-only members and the 26-model pydantic schema module
    # `server_adapter` used to drag in for one integer stay off the whole
    # first-paint window, not just off the import phase. NOT listed:
    # local_adapter/server_adapter/quick_notes/contracts -- `TldwCli.
    # __init__`'s `_wire_research_source_association` legitimately builds
    # readiness adapters at construction, so those stay resident at ready.
    "tldw_chatbook.Research_Workspace.controller",
    "tldw_chatbook.Research_Workspace.layout_state",
    "tldw_chatbook.Research_Workspace.overlay_store",
    "tldw_chatbook.tldw_api.notes_workspace_schemas",
    # TASK-24613 / ADR-097: lesson proposal and portable-organization owners
    # are interaction/sync work, not first-paint dependencies.  Keep their
    # implementation modules behind the post-ready or first-use seams.
    "tldw_chatbook.Agents.agent_lesson_promotion",
    "tldw_chatbook.Notes.agent_lessons",
    "tldw_chatbook.Notes.notes_organization_repository",
    "tldw_chatbook.Sync_Interop.domain_adapters.notes_organization",
    "tldw_chatbook.Sync_Interop.notes_organization",
    "tldw_chatbook.Sync_Interop.notes_organization_sync_service",
    "tldw_chatbook.Sync_Interop.notes_outbox_producer",
    # TASK-23113.7: normalization is idle maintenance. Its adapter and worker
    # import graph must stay outside the first-interactive-frame window even
    # on slow runners where mount settling continues after ``_ui_ready``.
    "tldw_chatbook.Chat.console_trace_legacy",
    "tldw_chatbook.Chat.console_trace_maintenance",
    # TASK-3605: Hub Test Tool admission is first-use work. Importing its
    # execution coordinator and preview registry during service construction
    # spends first-paint budget before an operator opens the MCP Hub.
    "tldw_chatbook.MCP.hub_test_execution",
)

#: Anti-vacuity: if these are not resident, the boot did not actually mount
#: the Chat screen and the census is measuring nothing.
EXPECTED_AT_READY = (
    "tldw_chatbook.UI.Screens.chat_screen",
    "tldw_chatbook.Chat.console_chat_controller",
    "tldw_chatbook.app",
)

#: Scratch profile mirroring the 2026-08-24 review's probe recipe: first-run
#: wizard completed, splash disabled (boot is serial under the splash), and a
#: valid-SHAPED key so the Console boots configured rather than into the
#: setup modal. The key is a nonsense literal -- nothing dials out.
_PROBE_CONFIG_TOML = """\
[first_run]
setup_completed = true

[splash_screen]
enabled = false

[api_settings.openai]
api_key = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"
"""

_CENSUS_SCRIPT = """
import asyncio
import json
import sys


async def main() -> None:
    import tldw_chatbook.app

    app = tldw_chatbook.app.TldwCli()
    async with app.run_test(size=(120, 40)):
        while not getattr(app, "_ui_ready", False):
            await asyncio.sleep(0.005)
        # Snapshot BEFORE iterating: background threads (tick syncs, catalog
        # refresh) keep importing after _ui_ready, and iterating the live
        # dict raised "dictionary changed size during iteration" -- an
        # intermittent guardrails failure on PR #2255 and a sibling branch,
        # 2026-08-31. A point-in-time copy is also the honest census: every
        # module in it existed at the same instant.
        modules_now = list(sys.modules)
        mods = sorted(
            m
            for m in modules_now
            if m.startswith("tldw_chatbook") and sys.modules[m] is not None
        )
        for m in mods:
            print("MOD:" + m, flush=True)
        print("CENSUS_JSON:" + json.dumps({"count": len(mods)}), flush=True)


asyncio.run(main())
"""


def _boot_and_census(tmp_path: Path) -> list[str]:
    """Boot to `_ui_ready` twice in subprocesses; return the WARM census.

    The first boot only exists to create the profile (DB files, migrations,
    seeding) so the second boot is the recurring warm case the budget is
    pinned against -- see the module docstring for the fresh-boot residents
    this deliberately excludes.
    """
    home = tmp_path / "home"
    data = tmp_path / "data"
    config_dir = tmp_path / "config"
    for directory in (home, data, config_dir):
        directory.mkdir(mode=0o700, exist_ok=True)
    (config_dir / "config.toml").write_text(_PROBE_CONFIG_TOML)

    env = {
        **os.environ,
        "HOME": str(home),
        "USERPROFILE": str(home),
        "XDG_DATA_HOME": str(data),
        "XDG_CONFIG_HOME": str(config_dir),
        "TLDW_CONFIG_PATH": str(config_dir / "config.toml"),
        "TLDW_TEST_MODE": "1",
        # Pin the whole-registry pre-importer OFF: its daemon thread would
        # race the census nondeterministically (see module docstring).
        "TLDW_SCREEN_PREIMPORT": "0",
        "PYTHONPATH": str(REPO_ROOT),
    }
    env.pop("PYTEST_CURRENT_TEST", None)

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
        f"profile-warming first boot failed (rc={result.returncode}):\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )

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
        f"warm headless boot to _ui_ready failed (rc={result.returncode}):\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    mods = [
        line[len("MOD:") :]
        for line in result.stdout.splitlines()
        if line.startswith("MOD:")
    ]
    payloads = [
        line[len("CENSUS_JSON:") :]
        for line in result.stdout.splitlines()
        if line.startswith("CENSUS_JSON:")
    ]
    assert payloads, f"census sentinel missing from stdout:\n{result.stdout[-2000:]}"
    assert json.loads(payloads[-1])["count"] == len(mods)
    return mods


@pytest.mark.integration
def test_ui_ready_module_census_stays_at_the_pinned_size(
    tmp_path: Path, ratchet
) -> None:
    """Boot to `_ui_ready`; this repo's resident modules stay within budget.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
        ratchet: shared ratchet helper (see ``conftest.py``).
    """
    mods = _boot_and_census(tmp_path)

    missing = [m for m in EXPECTED_AT_READY if m not in mods]
    assert not missing, (
        f"census looks degenerate -- expected mount-leg members missing: {missing}"
    )

    assert len(mods) <= MAX_TLDW_MODULES_AT_UI_READY, (
        f"{len(mods)} tldw_chatbook modules resident at _ui_ready "
        f"(ratchet limit {MAX_TLDW_MODULES_AT_UI_READY}). Mount-leg growth "
        "is invisible to the import guards.\n"
        f"{ratchet.format_module_diff(mods, 'ui-ready-census')}\n"
        f"{ratchet.ratchet_policy('MAX_TLDW_MODULES_AT_UI_READY')}\n"
        f"Deliberate snapshot refresh: `{ratchet.SNAPSHOT_REFRESH}`"
    )
    ratchet.emit_headroom(
        ratchet.headroom_line(
            "ui-ready-census",
            [("modules", len(mods), MAX_TLDW_MODULES_AT_UI_READY)],
        )
        + ratchet.snapshot_drift_suffix(mods, "ui-ready-census")
    )

    on_leg = [
        m
        for m in mods
        if any(m == p or m.startswith(p + ".") for p in ABSENT_AT_READY_PREFIXES)
        or m in ABSENT_AT_READY_MODULES
    ]
    assert not on_leg, (
        f"deferred families resident at _ui_ready (the first-paint window): "
        f"{on_leg}. Something re-eagered them on the import OR mount leg -- "
        "the closure guards in Tests/Packaging name the intended seams."
    )
