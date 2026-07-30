"""Selection state and read-side view model for the Evals workbench.

``EvalsSelection`` is the single source of truth for what the detail and
inspector panes show and what the primary action does. ``EvalsViewModel`` is
the read side: it translates ``EvalsDB`` rows into the shapes the library
rail and panes render. Neither imports any Textual widget -- this module is
pure data/read logic, kept separate from ``library_rail.py`` and
``evals_screen.py`` so the selection model can be reasoned about (and
unit-tested) without mounting anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Optional

from ...DB.Evals_DB import EvalsDB
from ...Evals.word_bench.models import PreflightResult
from ...Evals.word_bench.storage import BENCH_TYPE, load_run_preflight

SelectionKind = Literal["none", "bench", "classic", "dataset", "run_group"]

#: EvalsDB.list_tasks/list_datasets/list_runs all page; the Evals workbench
#: has no pagination UI yet, so each read pulls a generous single page
#: rather than the library defaults (100) that would silently hide older
#: rows in an install with more than a handful of benches.
_LIST_LIMIT = 500


@dataclass(frozen=True)
class EvalsSelection:
    """What the library rail currently has selected.

    ``kind="none"`` (the default) is the screen's initial state and
    whatever a stale/deleted id degrades to -- panes and the primary action
    key off ``kind`` first, ``id`` second.
    """

    kind: SelectionKind = "none"
    id: Optional[str] = None


class EvalsViewModel:
    """Read side for the Evals library rail and panes.

    Wraps a single ``EvalsDB`` handle -- or ``None``, when the app's
    evaluation service failed to wire (see ``EvalsScreen._resolve_db``) --
    and exposes the four collections the rail and panes render. Every
    method degrades to an empty list rather than raising when ``db`` is
    ``None``, so a broken evaluation service still renders an (empty)
    workbench instead of crashing the whole Evals destination.
    """

    def __init__(self, db: Optional[EvalsDB]) -> None:
        self._db = db

    @property
    def db(self) -> Optional[EvalsDB]:
        """The wrapped handle, for read-only widgets (``bench_editor.py``,
        ``inspector.py``) that need calls this view model doesn't itself
        expose (``load_bench``, ``get_model``) -- rather than each widget
        resolving its own second handle from ``app_instance``."""
        return self._db

    def _all_tasks(self) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        return self._db.list_tasks(limit=_LIST_LIMIT)

    @staticmethod
    def _is_word_bench(task: dict[str, Any]) -> bool:
        return (task.get("config_data") or {}).get("bench_type") == BENCH_TYPE

    def benches(self) -> list[dict[str, Any]]:
        """Word benches: ``eval_tasks`` rows tagged ``bench_type == "word_bench"``.

        See ``word_bench/storage.py``'s module docstring: a bench IS an
        ``eval_tasks`` row (``task_type="logprob"``), discriminated by
        ``config_data.bench_type`` rather than a dedicated table.
        """
        return [task for task in self._all_tasks() if self._is_word_bench(task)]

    def classic_tasks(self) -> list[dict[str, Any]]:
        """Every other ``eval_tasks`` row (pre-word-bench evaluation tasks)."""
        return [task for task in self._all_tasks() if not self._is_word_bench(task)]

    def datasets(self) -> list[dict[str, Any]]:
        if self._db is None:
            return []
        return self._db.list_datasets(limit=_LIST_LIMIT)

    def run_groups(self) -> list[dict[str, Any]]:
        """One row per distinct ``run_group_id``, newest run first.

        A word bench run (see ``word_bench/storage.create_run_group``)
        always shares one ``run_group_id`` across N per-target ``eval_runs``
        rows; this pivots ``list_runs()`` back into one selectable row per
        group. Runs with no ``run_group_id`` (never grouped) are not
        selectable here -- there is nothing for a "run_group" selection to
        resolve to.

        Each row also carries a rolled-up ``"status"`` (TASK-1480): the
        rail's run-row label needs one status per GROUP, but the DB tracks
        status per per-target run, and those can genuinely disagree --
        ``46d56f371``/``da4967a7a`` wired the primary action to a live
        ``WordBenchRunner`` pass (``runner.py`` moves each run
        pending -> running -> completed/cancelled independently), so a
        group composing mid-run can have one target still "running" while
        another has already finished. Precedence, computed in this same
        pivot pass so it never re-reads ``list_runs()``: any run
        "running" makes the whole group "running"; else any run
        "cancelled" makes it "cancelled"; else "completed" -- this also
        folds "pending" and "failed" runs into "completed", since the
        rail has no separate glyph for those and a group with no run
        still running or explicitly cancelled reads as done from the
        library rail's point of view.
        """
        if self._db is None:
            return []
        groups: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for run in self._db.list_runs(limit=_LIST_LIMIT):
            group_id = run.get("run_group_id")
            if not group_id:
                continue
            group = groups.get(group_id)
            if group is None:
                group = {
                    "id": group_id,
                    "task_id": run.get("task_id"),
                    "task_name": run.get("task_name"),
                    "created_at": run.get("created_at"),
                    "run_count": 0,
                    "_has_running": False,
                    "_has_cancelled": False,
                }
                groups[group_id] = group
                order.append(group_id)
            group["run_count"] += 1
            run_status = run.get("status")
            if run_status == "running":
                group["_has_running"] = True
            elif run_status == "cancelled":
                group["_has_cancelled"] = True
        results: list[dict[str, Any]] = []
        for group_id in order:
            group = groups[group_id]
            has_running = group.pop("_has_running")
            has_cancelled = group.pop("_has_cancelled")
            if has_running:
                group["status"] = "running"
            elif has_cancelled:
                group["status"] = "cancelled"
            else:
                group["status"] = "completed"
            results.append(group)
        return results

    def library_is_empty(self) -> bool:
        """Whether the whole workbench library -- benches, classic tasks,
        datasets, and run groups -- is genuinely empty.

        TASK-1076 review (Qodo): ``EvalsScreen._empty_detail_text()`` used
        to answer this by calling ``benches()`` and ``classic_tasks()``
        back to back -- each independently re-running
        ``EvalsDB.list_tasks(limit=500)`` via its own ``_all_tasks()``
        call -- plus a further ``datasets()`` and ``run_groups()`` read on
        top, all from a compose path that reruns on every selection change
        (``EvalsScreen.select()`` -> ``refresh(recompose=True)``). This
        method reads tasks ONCE and derives both bench and classic-task
        emptiness from that single read: either kind existing on its own
        is enough to make the library non-empty, so no bench/classic split
        is actually needed to answer this yes/no question, only a single
        pass over the one page of rows.

        Datasets use a ``limit=1`` existence check instead of a full page:
        exactly equivalent to ``bool(datasets())`` here, since
        ``list_datasets`` applies no row-shaping beyond the same
        ``deleted_at`` filter -- a matching row at position 1 proves the
        collection is non-empty just as well as a full 500-row page would.

        Run groups still go through ``run_groups()`` rather than a raw
        ``list_runs(limit=1)``: unlike datasets, that would NOT be
        equivalent -- ``run_groups()`` deliberately excludes runs with no
        ``run_group_id`` (see its own docstring), which ``list_runs``
        alone does not filter. Trading correctness for a cheaper query the
        DB API doesn't actually support for this shape is exactly the kind
        of behaviour change this fix must not make.

        Returns:
            ``True`` iff no benches, no classic tasks, no datasets, and no
            run groups exist (or ``db`` is unavailable).
        """
        if self._db is None:
            return True
        tasks = self._all_tasks()
        if any(self._is_word_bench(task) for task in tasks):
            return False
        if any(not self._is_word_bench(task) for task in tasks):
            return False
        if self._db.list_datasets(limit=1):
            return False
        if self.run_groups():
            return False
        return True

    def bench_by_id(self, bench_id: str) -> Optional[dict[str, Any]]:
        for bench in self.benches():
            if bench.get("id") == bench_id:
                return bench
        return None

    def classic_task_by_id(self, task_id: str) -> Optional[dict[str, Any]]:
        for task in self.classic_tasks():
            if task.get("id") == task_id:
                return task
        return None

    def dataset_by_id(self, dataset_id: str) -> Optional[dict[str, Any]]:
        for dataset in self.datasets():
            if dataset.get("id") == dataset_id:
                return dataset
        return None

    def run_group_by_id(self, run_group_id: str) -> Optional[dict[str, Any]]:
        for group in self.run_groups():
            if group.get("id") == run_group_id:
                return group
        return None

    def latest_run_group_for_bench(self, bench_id: str) -> Optional[dict[str, Any]]:
        """The bench's most recent run group, or ``None`` if it has never
        run. ``run_groups()`` is already newest-group-first (see its own
        docstring), so the first match is the latest."""
        for group in self.run_groups():
            if group.get("task_id") == bench_id:
                return group
        return None

    def preflight_for_bench(self, bench_id: str) -> dict[str, PreflightResult]:
        """Per-target readiness from the bench's most recent run snapshot.

        Reads ``word_bench.storage.load_run_preflight``'s stored verdicts
        rather than re-running preflight: a fresh preflight call from a
        render path would fire network requests on every selection change
        and could disagree with the verdict the run itself used (see
        ``runner.RunOutcome``'s own docstring). A bench that has never run
        has no snapshot, so this degrades to ``{}`` -- callers render an
        un-preflighted state for every target rather than treating an
        empty mapping as "all blocked".

        Uses ``load_run_preflight``, not ``load_grid``: the latter also
        pages and JSON-decodes every ``eval_results`` row for every run in
        the group just to reach this same snapshot field, discarding
        everything else it read. This method itself is also resolved once
        per selection by ``EvalsScreen`` (see its ``_preflight_for_
        selection``) and threaded into both the bench editor and the
        readiness inspector, rather than each pane calling this a second
        time on the same render.
        """
        if self._db is None:
            return {}
        group = self.latest_run_group_for_bench(bench_id)
        if group is None:
            return {}
        try:
            return load_run_preflight(self._db, group["id"])
        except ValueError:
            # The run group vanished between listing it and loading it
            # (e.g. concurrent deletion) -- render un-preflighted rather
            # than raising out of a compose().
            return {}

    def runs_for_task(self, task_id: str) -> list[dict[str, Any]]:
        """A task's run history, newest first -- used by the classic-task
        read-only detail pane. Unlike ``run_groups()`` this returns every
        run row, not one row per group: classic tasks have no run-group
        concept (word bench is the only bench type that groups runs)."""
        if self._db is None:
            return []
        return self._db.list_runs(task_id=task_id, limit=_LIST_LIMIT)
