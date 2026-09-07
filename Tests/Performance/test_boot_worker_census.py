"""Boot worker/thread census (TASK-22222; the list TASK-22215 will stagger).

The 2026-08-24 holistic perf review counted the boot-time background worker
fleet growing 4 -> 7 since the prior pin (new: the chachanotes-fts-backfill,
the initial-screen pre-import thread, the actor-pack recovery relocation)
with nothing noticing. Each worker was individually justified; the aggregate
GIL contention during the first seconds after mount is what the user feels.
This guard boots the real app headless (Textual Pilot) against a scratch
profile, records every Textual worker start (via the ``WorkerManager`` seam)
and every OS thread start (via ``threading.Thread.start``) from process
start until ``_ui_ready`` + a settle window, and pins the observed set
against an allowlist.

The property pinned: **starting an unlisted worker or thread during boot is
a reviewed decision**, made by adding a row here with the feature that needs
it -- the eighth worker arrives in review, not silently. The allowlist is
deliberately the superset of everything legitimately observed (fresh-boot
one-offs and stall-triggered threads included), because the guard's job is
to catch NEW UNREVIEWED members, not to flake on members that only
sometimes run. TASK-22215 (stagger/priority policy for this fleet) took
this allowlist as the inventory it reorders; staggering itself stays out of
scope here. That policy now lives in ``tldw_chatbook/Utils/boot_worker_
policy.py`` and ``Tests/App/test_boot_worker_stagger_policy.py`` cross-checks
every policy row against the allowlist below, so the two cannot drift: WHICH
workers may start is pinned here, WHEN and HOW MANY AT ONCE is pinned there.
No allowlist row changed for that task -- the four staggered members kept
their (name, group) identity and merely moved from ``on_mount`` to the
post-``_ui_ready`` tier, which is still inside this census's settle window.

Raising/extending: when this fails, the message prints the unlisted
starters. Name the feature that added each, decide whether it must really
start during boot (first paint is the most contended moment in the app's
life -- deferring is the preferred answer), then add the row with a comment
naming the owner, in the same commit.

Observed inventory (2026-08-25, this branch, base dev f0e896122): 16
(name, group) worker pairs on a warm boot, +2 fresh-boot-only persistence
workers, 3-4 thread families -- see the allowlists below for the rows.

Documented blind spots (what a start-record census cannot see):

* It records STARTS, not cost or concurrency. Seven cheap staggered workers
  can beat four heavy simultaneous ones; only a latency probe (22215's
  before/after AC) sees the difference. A listed worker that grows a bigger
  payload is invisible here.
* The settle window is ``_ui_ready`` + 1.0 s. A boot-adjacent worker first
  started later than that (or one gated off ``TLDW_TEST_MODE=1``, which
  every boot guard sets) is not censused. Members that only START sometimes
  (stall-triggered persistence, fresh-profile one-offs) are allowlisted but
  not asserted present, so their absence never fails and their growth is
  covered only by membership.
* Thread identity is (normalized name, target module, target qualname);
  pool sizes are machine-dependent (cpu_count), so this census deliberately
  does NOT budget thread COUNTS -- a pool that doubles its size passes.
  Worker identity is (name, group): a second start of a listed pair (e.g.
  a duplicated startup call) also passes -- membership, not multiplicity.
* Subprocesses are not threads: work farmed to a child process would be
  invisible here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Every (worker name, worker group) pair that may start during boot.
#: Adding a row is the reviewed decision this guard exists to force.
ALLOWED_BOOT_WORKERS: frozenset[tuple[str, str]] = frozenset(
    {
        # -- TldwCli (app-level startup) --
        ("_backfill_chachanotes_messages_fts", "chachanotes-fts-backfill"),
        ("_backfill_subscription_items_fts", "subscriptions-fts-backfill"),
        ("_reconcile_research_quick_notes_startup", "research-quick-notes-startup-reconciliation"),
        ("_sweep_research_paste_staging", "research_paste_staging_startup"),
        ("deferred_actor_pack_recovery", "actor_pack_recovery"),
        ("deferred_actor_pack_staging_sweep", "actor_pack_staging_sweep"),
        ("restore_ingest_jobs", "ingest_restore"),
        ("resume_startup", "research_source_association_startup"),
        ("run", "scheduling"),
        # -- ChatScreen / Console (initial-screen mount) --
        ("_refresh_console_persisted_rows_cache", "console-persisted-browser-cache"),
        ("_refresh_console_skill_candidates", "default"),
        ("_sync_console_legacy_workspace_context_aliases", "console-workspace-context-legacy-aliases"),
        # TASK-26042: this one off-loop snapshot is required to render truthful
        # Show Files availability without filesystem status work on the UI loop.
        ("_refresh_workspace_files_availability_snapshot", "console-workspace-files-availability"),
        ("build_worker", "console-changed-files"),
        ("load", "console-prompt-history"),
        # Rail-preference persistence: observed on the FIRST boot of a fresh
        # profile (initial state write); allowlisted, not asserted present.
        ("_save_console_rail_preferences", "default"),
        ("_prune_console_rail_preferences", "default"),
        ("_persist_sidebar_state_off_loop", "sidebar-state-persist"),
    }
)

#: Every (normalized thread name, target module, target qualname) family
#: that may start an OS thread during boot. Numeric run/pool suffixes are
#: normalized to ``#`` so pool sizing (machine-dependent) cannot flake.
ALLOWED_BOOT_THREADS: frozenset[tuple[str, str, str]] = frozenset(
    {
        # The parallel service-init pool (app startup).
        ("ThreadPoolExecutor-#_#", "concurrent.futures.thread", "_worker"),
        # asyncio's default executor: Textual thread workers land here.
        ("asyncio_#", "concurrent.futures.thread", "_worker"),
        # Whole-registry screen pre-importer (daemon; finding 22214's list).
        ("tldw-screen-preimport", "tldw_chatbook.app", "TldwCli._preimport_heavy_screens"),
        # Stall persistence: starts only when the responsiveness monitor
        # observes a UI stall during boot -- allowlisted, never asserted.
        ("ui-stall-persist", "tldw_chatbook.Utils.ui_responsiveness", "UIResponsivenessMonitor._drain_stalls"),
    }
)

#: Anti-vacuity: a real boot MUST start these. If none are recorded the
#: probe instrumented nothing (or never reached _ui_ready) and the census
#: is measuring an empty list, which must fail rather than pass.
EXPECTED_BOOT_WORKERS: frozenset[tuple[str, str]] = frozenset(
    {
        ("_backfill_chachanotes_messages_fts", "chachanotes-fts-backfill"),
        ("load", "console-prompt-history"),
        ("run", "scheduling"),
    }
)

_PROBE_CONFIG_TOML = """\
[first_run]
setup_completed = true

[splash_screen]
enabled = false

[api_settings.openai]
api_key = "sk-abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKL"
"""

#: The probe instruments the two start seams BEFORE importing the app, so
#: nothing started during import/construct/mount can escape the record, and
#: snapshots nothing -- it records starts, so a worker that finishes before
#: the settle window ends is still censused.
_CENSUS_SCRIPT = """
import asyncio
import json
import threading

records = {"workers": [], "threads": []}

import textual.worker_manager as _wm

_real_new_worker = _wm.WorkerManager._new_worker


def _recording_new_worker(self, work, node, **kwargs):
    records["workers"].append(
        {
            "name": kwargs.get("name") or getattr(work, "__name__", "") or "",
            "group": kwargs.get("group", "default"),
        }
    )
    return _real_new_worker(self, work, node, **kwargs)


_wm.WorkerManager._new_worker = _recording_new_worker

_real_thread_start = threading.Thread.start


def _recording_thread_start(self):
    target = getattr(self, "_target", None)
    records["threads"].append(
        {
            "name": self.name,
            "module": getattr(target, "__module__", None) if target else None,
            "qualname": getattr(target, "__qualname__", None) if target else None,
        }
    )
    return _real_thread_start(self)


threading.Thread.start = _recording_thread_start


async def main() -> None:
    import tldw_chatbook.app

    app = tldw_chatbook.app.TldwCli()
    async with app.run_test(size=(120, 40)):
        while not getattr(app, "_ui_ready", False):
            await asyncio.sleep(0.005)
        # Settle window: the deferred-startup timers (0.1-0.2 s) fire inside
        # it, so their workers are censused too.
        await asyncio.sleep(1.0)
        print("CENSUS_JSON:" + json.dumps(records), flush=True)


asyncio.run(main())
"""


def _normalize_thread_name(name: str) -> str:
    """Collapse numeric pool/run suffixes so pool sizing cannot flake.

    ``ThreadPoolExecutor-0_2`` -> ``ThreadPoolExecutor-#_#``;
    ``asyncio_3`` -> ``asyncio_#``; ``Thread-7 (run)`` -> ``Thread-# (run)``.
    """
    out = []
    digits = False
    for ch in name:
        if ch.isdigit():
            digits = True
            continue
        if digits:
            out.append("#")
            digits = False
        out.append(ch)
    if digits:
        out.append("#")
    return "".join(out)


def _boot_and_census(tmp_path: Path) -> dict[str, list[dict[str, str | None]]]:
    """Boot headless to `_ui_ready`+settle in a subprocess; return the record."""
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
        f"headless boot to _ui_ready failed (rc={result.returncode}):\n"
        f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-4000:]}"
    )
    payloads = [
        line[len("CENSUS_JSON:") :]
        for line in result.stdout.splitlines()
        if line.startswith("CENSUS_JSON:")
    ]
    assert payloads, f"census sentinel missing from stdout:\n{result.stdout[-2000:]}"
    return json.loads(payloads[-1])


@pytest.mark.integration
def test_boot_worker_and_thread_starts_stay_within_the_allowlist(
    tmp_path: Path,
) -> None:
    """Every worker/thread started during boot is on the reviewed allowlist.

    Args:
        tmp_path: pytest fixture; isolated dir for the subprocess's profile.
    """
    records = _boot_and_census(tmp_path)

    started_workers = {(w["name"], w["group"]) for w in records["workers"]}
    started_threads = {
        (
            _normalize_thread_name(t["name"] or ""),
            t["module"] or "",
            t["qualname"] or "",
        )
        for t in records["threads"]
    }

    missing = EXPECTED_BOOT_WORKERS - started_workers
    assert not missing, (
        f"census looks degenerate -- boot workers that always start were "
        f"not recorded: {sorted(missing)}. Either the boot never mounted "
        "the Chat screen or the probe's instrumentation seam moved."
    )
    assert started_threads, (
        "census looks degenerate -- a real boot starts at least the "
        "parallel-init pool threads, and none were recorded."
    )

    unlisted_workers = started_workers - ALLOWED_BOOT_WORKERS
    assert not unlisted_workers, (
        f"workers started during boot that are not on the reviewed "
        f"allowlist: {sorted(unlisted_workers)}. First paint is the most "
        "contended moment in the app's life (GIL: finding 22215) -- prefer "
        "deferring to first feature use; if it must ride the boot, add the "
        "(name, group) row with the owning feature named, in the same "
        "commit. TASK-22215's stagger policy consumes this list."
    )

    unlisted_threads = started_threads - ALLOWED_BOOT_THREADS
    assert not unlisted_threads, (
        f"OS threads started during boot that are not on the reviewed "
        f"allowlist: {sorted(unlisted_threads)}. Same rule as workers: "
        "defer if possible, otherwise add the reviewed row naming the "
        "owner, in the same commit."
    )
