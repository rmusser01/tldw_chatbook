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
from ...Evals.word_bench.storage import BENCH_TYPE

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
                }
                groups[group_id] = group
                order.append(group_id)
            group["run_count"] += 1
        return [groups[group_id] for group_id in order]

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
