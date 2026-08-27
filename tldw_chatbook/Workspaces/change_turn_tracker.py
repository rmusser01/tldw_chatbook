"""Per-turn B/E snapshot orchestration for Agent Change Review (TASK-1971).

Wraps :mod:`tldw_chatbook.Workspaces.change_tracking`'s per-root shadow
repos around one agent run:

* :meth:`ChangeTurnTracker.begin_turn` kicks the baseline (**B**) snapshot on
  a background thread and returns immediately — B rides the model's own
  first-token latency instead of adding send latency (spec §2);
* :meth:`TurnHandle.await_baseline` is the tool-dispatch gate: it must
  complete before the FIRST tool touches disk, or the tool's own write races
  into the baseline and vanishes from the diff;
* :meth:`ChangeTurnTracker.end_turn` takes the end (**E**) snapshot on every
  terminal path — including failed and cancelled runs, which is when review
  matters most — and returns one record per root that actually changed.

Known window (spec §2's trade, characterized while testing): a write that
lands during the FIRST provider stream — before B settles — is swallowed
into the baseline and never appears in the diff. Production has no writer
in that window: every writer, scripts included, is a tool, and tools sit
behind the await-B gate. Only a non-tool writer (another app, the user's
editor) racing the first token can hit it, and those are already documented
attribution limits (spec §5).

Failure posture (spec §2): tracking NEVER blocks or fails the agent reply.
Every error becomes a per-root record carrying ``tracking_error`` for the
card to disclose; nothing here raises out of ``begin_turn``/``end_turn``.

The ``.gitignore`` carve-out (spec §1): paths the run's WRITE tools touched
are force-added before E, so a direct agent edit to an ignored file (`.env`
is the canonical case) always surfaces. Read tools are deliberately NOT
included — force-adding a merely-read, pre-existing ignored file would lie
an "Added" row into the review.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from loguru import logger

from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_MAX_FILE_BYTES,
    change_review_setting,
)
from tldw_chatbook.Workspaces.change_tracking import ShadowRepoService

#: Tools whose path arguments are force-added at E time (the .gitignore
#: carve-out). WRITE tools only — see the module docstring for why reads
#: must not be here.
WRITE_TOOL_NAMES: frozenset[str] = frozenset({"write_file"})

#: Argument keys that carry the written path for tools in
#: :data:`WRITE_TOOL_NAMES`.
_PATH_ARG_KEYS: tuple[str, ...] = ("file_path", "path")

#: Upper bound on waiting for the baseline thread. Generous — snapshots are
#: seconds — but bounded, so a pathological root cannot hang the run.
_BASELINE_TIMEOUT_SECONDS = 120.0


@dataclass(frozen=True)
class TurnChangeRecord:
    """One root's outcome for one turn.

    Attributes:
        root: Canonical root path (string, for storage).
        baseline_sha: The B snapshot tip ("" when tracking failed).
        end_sha: The E snapshot tip ("" when tracking failed).
        files_changed: Count of changed files between B and E.
        adds: Total added lines.
        dels: Total deleted lines.
        tracking_error: Non-empty when tracking failed for this root; the
            record then exists to be DISCLOSED, not rendered as a diff.
    """

    root: str
    baseline_sha: str = ""
    end_sha: str = ""
    files_changed: int = 0
    adds: int = 0
    dels: int = 0
    tracking_error: str = ""
    untracked_oversize: int = 0
    nested_repos: tuple[str, ...] = ()


class TurnHandle:
    """The in-flight state of one turn's baseline snapshots."""

    def __init__(self, roots: list[Path]) -> None:
        self.roots = roots
        self.baselines: dict[str, str] = {}
        self.errors: dict[str, str] = {}
        #: TASK-1975: each root's oversize-excluded set at B, so end_turn
        #: can tell NEW oversize (disclose even changeless) from stable.
        self.baseline_oversize: dict[str, tuple[str, ...]] = {}
        #: TASK-1976: each root's nested-repo set at B, from the SAME walk
        #: the budget gate already runs — new holes disclose even cardless.
        self.baseline_nested: dict[str, tuple[str, ...]] = {}
        #: TASK-1977: per-root REL paths auto-registered as sub-roots this
        #: turn — excluded from that root's nested-repo disclosure.
        self.auto_registered: dict[str, tuple[str, ...]] = {}
        #: PR3a-1 Task 6c: each root's E sha, recorded by ``end_turn`` for
        #: EVERY root it snapshotted — including the unchanged ones, which
        #: yield no record. It is the boundary a follow-on window starts
        #: from (see :meth:`ChangeTurnTracker.continuation`), so a write
        #: made after this turn's E cannot fall between two windows.
        self.end_shas: dict[str, str] = {}
        self._thread: threading.Thread | None = None

    def await_baseline(self, timeout: float = _BASELINE_TIMEOUT_SECONDS) -> None:
        """Block until every root's B snapshot settled (or errored).

        The tool-dispatch gate: called before the first tool executes and
        again defensively by ``end_turn``. Never raises — a timeout is
        recorded as a per-root error and disclosed downstream.
        """
        thread = self._thread
        if thread is None:
            return
        thread.join(timeout=timeout)
        if thread.is_alive():
            # Qodo #1256: the discovery thread may still be APPENDING
            # sub-roots — iterate a snapshot, never the live list.
            for root in tuple(self.roots):
                key = str(root)
                if key not in self.baselines and key not in self.errors:
                    self.errors[key] = (
                        f"baseline snapshot still running after {timeout:.0f}s"
                    )


class ChangeTurnTracker:
    """Orchestrates B/E snapshots for agent turns. One instance per app."""

    def __init__(self, service: ShadowRepoService | None = None) -> None:
        """Create a tracker over a shadow-repo service.

        Args:
            service: Shadow-repo service; a default (app data dir, PATH
                git) is built when omitted.
        """
        self.service = service if service is not None else ShadowRepoService()

    @property
    def available(self) -> bool:
        """Whether tracking can work at all (git binary present)."""
        return self.service.available

    # -- turn lifecycle ----------------------------------------------------

    def begin_turn(
        self,
        roots: Sequence[Path | str],
        touched_paths: Sequence[str] = (),
    ) -> TurnHandle:
        """Kick baseline snapshots for ``roots`` in the background.

        Returns immediately; never raises. Non-directory roots are recorded
        as errors rather than dropped silently.

        Args:
            roots: The run's workspace folder roots.
            touched_paths: Paths eligible for the WRITE-tool ignore carve-out
                at the baseline snapshot.

        Returns:
            A handle for :meth:`TurnHandle.await_baseline` / :meth:`end_turn`.
        """
        # Roots are RESOLVED here (review finding): `_paths_within` resolves
        # each touched path, so an unresolved (symlink-spelled) root would
        # make `relative_to` fail and silently skip the force-add -- the
        # .gitignore carve-out dying without a trace.
        handle = TurnHandle(
            [Path(r).expanduser().resolve() for r in roots]
        )
        frozen_touched_paths = tuple(touched_paths)

        def _baseline() -> None:
            from tldw_chatbook.Workspaces.change_bounds import (
                DEFAULT_MAX_SUB_ROOTS,
                scan_root,
            )

            # TASK-1977: nested repos found inside a GIVEN root become
            # tracked sub-roots of their own (bounded by max_sub_roots).
            # Depth is 1 by construction: only the caller's original roots
            # expand — a grandchild repo stays disclosed via ITS parent's
            # banner rather than recursing unbounded.
            original_keys = {str(root) for root in handle.roots}
            seen = set(original_keys)
            queue = list(handle.roots)
            while queue:
                root = queue.pop(0)
                key = str(root)
                try:
                    # TASK-1975: budget gate BEFORE any snapshot work. Over
                    # budget disables tracking for this root with honest
                    # copy -- never a silent half-track.
                    scan = scan_root(root)
                    if scan.over_budget:
                        handle.errors[key] = (
                            "root over change-tracking budget "
                            f"({scan.files}+ files / {scan.total_bytes}+ "
                            "bytes) — narrow the root or add excludes; "
                            "tracking disabled for this turn"
                        )
                        continue
                    registered: tuple[str, ...] = ()
                    if key in original_keys:
                        max_subs = change_review_setting(
                            "max_sub_roots", DEFAULT_MAX_SUB_ROOTS
                        )
                        candidates = scan.nested_repos[: max(0, max_subs)]
                        kept: list[str] = []
                        for rel in candidates:
                            child = (root / rel).resolve()
                            ckey = str(child)
                            if ckey in seen or not child.is_dir():
                                continue
                            seen.add(ckey)
                            kept.append(rel)
                            handle.roots.append(child)
                            queue.append(child)
                        registered = tuple(kept)
                    handle.auto_registered[key] = registered
                    handle.baseline_nested[key] = tuple(
                        rel
                        for rel in scan.nested_repos
                        if rel not in registered
                    )
                    repo = self.service.repo_for_root(root)
                    eligible = self._eligible_touched_paths(
                        root, frozen_touched_paths
                    )
                    if eligible:
                        baseline = repo.snapshot(
                            "turn baseline", force_paths=eligible
                        )
                    else:
                        baseline = repo.snapshot("turn baseline")
                    handle.baselines[key] = baseline
                    handle.baseline_oversize[key] = repo.last_oversize_excluded
                except Exception as exc:  # noqa: BLE001 -- disclosed, never raised
                    handle.errors[key] = str(exc)[:400]

        if handle.roots:
            thread = threading.Thread(
                target=_baseline, name="change-review-baseline", daemon=True
            )
            handle._thread = thread
            thread.start()
        return handle

    def continuation(self, handle: TurnHandle) -> "TurnHandle | None":
        """A follow-on window starting exactly where ``handle`` ended.

        PR3a-1 Task 6c. A sub-agent that outlives its turn keeps writing
        after that turn's E snapshot, so its work needs a window of its
        own — and that window must START at the previous one's END sha,
        not at a fresh snapshot taken some milliseconds later, or a write
        in between belongs to no window at all.

        No I/O: the returned handle is pre-satisfied (no baseline thread),
        because its baseline shas were already taken by ``end_turn``.

        Args:
            handle: A handle ``end_turn`` has already run over.

        Returns:
            A handle whose baselines are ``handle``'s end shas, or
            ``None`` when that turn recorded no usable end sha (tracking
            failed for every root) — there is nothing to continue from.
        """
        if not handle.end_shas:
            return None
        follow_on = TurnHandle(
            [root for root in tuple(handle.roots) if str(root) in handle.end_shas]
        )
        follow_on.baselines = dict(handle.end_shas)
        # Carry the disclosure baselines forward: a file that was ALREADY
        # oversize / a repo that was ALREADY nested at the turn's end is
        # not news in the follow-on window either.
        follow_on.baseline_oversize = dict(handle.baseline_oversize)
        follow_on.baseline_nested = dict(handle.baseline_nested)
        follow_on.auto_registered = dict(handle.auto_registered)
        return follow_on

    def end_turn(
        self,
        handle: TurnHandle,
        touched_paths: Sequence[str] = (),
        *,
        end_shas: "dict[str, str] | None" = None,
    ) -> list[TurnChangeRecord]:
        """Take E snapshots and return one record per root that changed.

        Runs on every terminal path. Never raises: per-root failures become
        records with ``tracking_error`` set; a clean root with ``B == E``
        yields NO record (spec: no changes, no card).

        Args:
            handle: The turn's :class:`TurnHandle` from :meth:`begin_turn`.
            touched_paths: Absolute paths the run's WRITE tools touched
                (see :meth:`tool_touched_paths`) — force-added before E so
                ignored-but-edited files surface.
            end_shas: PR3a-1 Task 6c. Per-root shas to use as E INSTEAD of
                taking a snapshot — how a survivor's window is closed at
                the exact sha the next turn's baseline recorded, so the
                two windows share a boundary and nothing can fall between
                them. Roots absent from the mapping still snapshot.
                Oversize/nested disclosure is skipped for a provided sha:
                those measurements belong to whoever took that snapshot,
                and re-deriving them here would report the state of a tree
                this window never observed.

        Returns:
            Records for roots with changes or tracking errors.
        """
        handle.await_baseline()
        records: list[TurnChangeRecord] = []
        # Snapshot + dedupe: a timed-out baseline thread may still be
        # appending discovered sub-roots (Qodo #1256) — a live-list
        # iteration could loop on churn, and a duplicate entry would yield
        # duplicate records for one root.
        seen_roots: set[str] = set()
        for root in tuple(handle.roots):
            key = str(root)
            if key in seen_roots:
                continue
            seen_roots.add(key)
            if key in handle.errors:
                records.append(
                    TurnChangeRecord(root=key, tracking_error=handle.errors[key])
                )
                continue
            baseline = handle.baselines.get(key, "")
            if not baseline:
                records.append(
                    TurnChangeRecord(
                        root=key,
                        tracking_error="baseline snapshot missing",
                    )
                )
                continue
            provided = (end_shas or {}).get(key)
            if provided:
                handle.end_shas[key] = provided
            try:
                repo = self.service.repo_for_root(root)
                eligible = self._eligible_touched_paths(root, touched_paths)
                if provided:
                    # The supplied snapshot stays immutable. Priming the
                    # shared index only lets the next fresh snapshot consume
                    # a path that became available after this exact boundary.
                    end = provided
                    if end == baseline:
                        repo.force_add(eligible)
                        continue
                    changed = repo.changed_files(baseline, end)
                    record = TurnChangeRecord(
                        root=key,
                        baseline_sha=baseline,
                        end_sha=end,
                        files_changed=len(changed),
                        adds=sum(c.adds for c in changed),
                        dels=sum(c.dels for c in changed),
                    )
                    repo.force_add(eligible)
                    records.append(record)
                    continue
                if eligible:
                    end = repo.snapshot("turn end", force_paths=eligible)
                else:
                    end = repo.snapshot("turn end")
                handle.end_shas[key] = end
                oversize = repo.last_oversize_excluded
                # TASK-1977: a TRACKED sub-root is not an untracked hole —
                # disclosure covers exactly what is not tracked.
                registered = handle.auto_registered.get(key, ())
                nested = tuple(
                    rel
                    for rel in repo.last_nested_repos
                    if rel not in registered
                )
                if end == baseline:
                    # TASK-1975 (AC#6): an oversized file CREATED during
                    # the turn is the turn's only event -- disclose it with
                    # a zero-change record instead of staying silent. A
                    # STABLE oversize set stays cardless (noise control).
                    new_oversize = set(oversize) - set(
                        handle.baseline_oversize.get(key, ())
                    )
                    # TASK-1976: a repo cloned mid-turn is a NEW hole —
                    # same disclosure rule as new oversize.
                    new_nested = set(nested) - set(
                        handle.baseline_nested.get(key, ())
                    )
                    if new_oversize or new_nested:
                        records.append(
                            TurnChangeRecord(
                                root=key,
                                baseline_sha=baseline,
                                end_sha=end,
                                untracked_oversize=len(oversize),
                                nested_repos=nested,
                            )
                        )
                    continue
                changed = repo.changed_files(baseline, end)
                records.append(
                    TurnChangeRecord(
                        root=key,
                        baseline_sha=baseline,
                        end_sha=end,
                        files_changed=len(changed),
                        adds=sum(c.adds for c in changed),
                        dels=sum(c.dels for c in changed),
                        untracked_oversize=len(oversize),
                        nested_repos=nested,
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- disclosed, never raised
                records.append(
                    TurnChangeRecord(
                        root=key,
                        baseline_sha=baseline,
                        end_sha=provided or "",
                        tracking_error=str(exc)[:400],
                    )
                )
        return records

    # -- helpers -----------------------------------------------------------

    def _eligible_touched_paths(
        self, root: Path, touched_paths: Iterable[str]
    ) -> list[str]:
        """Return root-relative touched paths allowed into a snapshot."""
        in_root = self._paths_within(root, touched_paths)
        if not in_root:
            return []
        # TASK-1975: force-add exists to defeat IGNORE rules, not the size
        # cap -- a tool-written oversized file is disclosed, never committed.
        cap = change_review_setting("max_file_bytes", DEFAULT_MAX_FILE_BYTES)
        return [
            rel for rel in in_root if not self._over_cap(root, rel, cap)
        ]

    @staticmethod
    def _over_cap(root: Path, rel: str, cap: int) -> bool:
        """Whether a root-relative path currently exceeds the size cap."""
        try:
            return (root / rel).stat().st_size > cap
        except OSError:
            return False

    @staticmethod
    def _paths_within(root: Path, paths: Iterable[str]) -> list[str]:
        """Relativize ``paths`` to ``root``, dropping those outside it."""
        out: list[str] = []
        for raw in paths:
            try:
                rel = Path(raw).expanduser().resolve().relative_to(root)
            except (ValueError, OSError):
                continue
            out.append(str(rel))
        return out

    @staticmethod
    def tool_touched_paths(steps: Iterable[Any]) -> list[str]:
        """Extract WRITE-tool path arguments from a run's recorded steps.

        Args:
            steps: ``AgentStep``-shaped objects (``tool_name``/``args``
                attributes) or equivalent dicts.

        Returns:
            Path strings in step order, de-duplicated, WRITE tools only —
            a read touch here would force-add a pre-existing ignored file
            and lie an "Added" row into the review.
        """
        seen: set[str] = set()
        out: list[str] = []
        for step in steps:
            tool = getattr(step, "tool_name", None)
            args = getattr(step, "args", None)
            if tool is None and isinstance(step, dict):
                tool = step.get("tool_name")
                args = step.get("args")
            if tool not in WRITE_TOOL_NAMES or not isinstance(args, dict):
                continue
            for arg_key in _PATH_ARG_KEYS:
                value = args.get(arg_key)
                if isinstance(value, str) and value and value not in seen:
                    seen.add(value)
                    out.append(value)
                    break
        return out


def initial_snapshot_in_background(root: Path | str) -> threading.Thread | None:
    """Best-effort background initial snapshot for a newly registered root.

    Spec §2: the FIRST snapshot of a root happens at registration time, so
    first-send latency never absorbs the cost of hashing a whole tree.
    Failures log and are disclosed on first use instead; never raises.

    Args:
        root: The just-registered folder root.

    Returns:
        The started thread (for tests to join), or ``None`` when tracking
        is unavailable.
    """
    service = ShadowRepoService()
    if not service.available:
        return None

    def _snapshot() -> None:
        try:
            service.repo_for_root(root).snapshot("root registered")
        except Exception:  # noqa: BLE001 -- best-effort by design
            logger.opt(exception=True).warning(
                f"change_review: initial snapshot failed for {root}"
            )

    thread = threading.Thread(
        target=_snapshot, name="change-review-initial-snapshot", daemon=True
    )
    thread.start()
    return thread
