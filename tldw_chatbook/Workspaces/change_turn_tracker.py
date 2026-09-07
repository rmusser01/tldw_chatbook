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

from tldw_chatbook.Utils.path_validation import validate_path
from tldw_chatbook.Workspaces.change_bounds import (
    DEFAULT_MAX_FILE_BYTES,
    change_review_setting,
)
from tldw_chatbook.Workspaces.change_tracking import (
    ChangeTrackingError,
    ShadowRepoService,
)

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


@dataclass(frozen=True)
class BaselineRootPreparation:
    """Immutable bounded-scan result consumed before a B snapshot."""

    root: Path
    registered: tuple[str, ...] = ()
    nested_repos: tuple[str, ...] = ()
    tracking_error: str = ""


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
        #: Root-relative ignored paths owned by this handle but learned only
        #: after its baseline. They are staged atomically with this handle's
        #: E snapshot, never left in the root-shared shadow index.
        self._deferred_force_paths: dict[str, set[str]] = {}
        self._deferred_force_paths_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._baseline_ready = threading.Event()
        self._baseline_lock = threading.Lock()
        self._accepting_baseline = True

    def defer_force_paths(self, root: Path | str, paths: Iterable[str]) -> None:
        """Bind eligible ignored paths to this handle's future E snapshot."""
        if not paths:
            return
        key = str(root)
        with self._deferred_force_paths_lock:
            self._deferred_force_paths.setdefault(key, set()).update(paths)

    def force_paths_for_root(self, root: Path | str) -> tuple[str, ...]:
        """Return a stable copy of deferred paths for one root."""
        key = str(root)
        with self._deferred_force_paths_lock:
            return tuple(sorted(self._deferred_force_paths.get(key, ())))

    def await_baseline(self, timeout: float = _BASELINE_TIMEOUT_SECONDS) -> bool:
        """Block until every root's B snapshot settled (or errored).

        The tool-dispatch gate: called before the first tool executes and
        again defensively by ``end_turn``. Never raises — a timeout is
        recorded as a per-root error and disclosed downstream.
        """
        if self._baseline_ready.wait(timeout=max(0.0, timeout)):
            return True
        with self._baseline_lock:
            if self._baseline_ready.is_set():
                return True
            self._accepting_baseline = False
            # Qodo #1256: the discovery thread may still be APPENDING
            # sub-roots — iterate a snapshot, never the live list.
            for root in tuple(self.roots):
                key = str(root)
                if key not in self.baselines and key not in self.errors:
                    self.errors[key] = (
                        f"baseline snapshot still running after {timeout:.0f}s"
                    )
        return False


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

    def new_turn_handle(self, roots: Sequence[Path | str]) -> TurnHandle:
        """Return an unresolved handle without starting any worker thread."""
        handle = TurnHandle([Path(r).expanduser().resolve() for r in roots])
        if not handle.roots:
            handle._baseline_ready.set()
        return handle

    def populate_baseline(
        self,
        handle: TurnHandle,
        touched_paths: Sequence[str] = (),
    ) -> None:
        """Populate one handle's baseline on the caller-owned worker."""
        preparations = self.discover_baseline(handle)
        self.populate_prepared_baseline(
            handle,
            preparations,
            touched_paths=touched_paths,
        )

    def discover_baseline(
        self, handle: TurnHandle
    ) -> tuple[BaselineRootPreparation, ...]:
        """Scan roots and discover bounded nested roots without snapshotting."""
        from tldw_chatbook.Workspaces.change_bounds import (
            DEFAULT_MAX_SUB_ROOTS,
            scan_root,
        )

        original_keys = {str(root) for root in handle.roots}
        seen = set(original_keys)
        queue = list(handle.roots)
        preparations: list[BaselineRootPreparation] = []
        while queue:
            root = queue.pop(0)
            key = str(root)
            try:
                scan = scan_root(root)
                if scan.over_budget:
                    preparations.append(
                        BaselineRootPreparation(
                            root=root,
                            tracking_error=(
                                "root over change-tracking budget "
                                f"({scan.files}+ files / {scan.total_bytes}+ "
                                "bytes) — narrow the root or add excludes; "
                                "tracking disabled for this turn"
                            ),
                        )
                    )
                    continue
                registered: tuple[str, ...] = ()
                if key in original_keys:
                    max_subs = change_review_setting(
                        "max_sub_roots", DEFAULT_MAX_SUB_ROOTS
                    )
                    kept: list[str] = []
                    for rel in scan.nested_repos[: max(0, max_subs)]:
                        child = (root / rel).resolve()
                        ckey = str(child)
                        if ckey in seen or not child.is_dir():
                            continue
                        seen.add(ckey)
                        kept.append(rel)
                        queue.append(child)
                    registered = tuple(kept)
                preparations.append(
                    BaselineRootPreparation(
                        root=root,
                        registered=registered,
                        nested_repos=tuple(
                            rel for rel in scan.nested_repos if rel not in registered
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 -- disclosed, never raised
                preparations.append(
                    BaselineRootPreparation(root=root, tracking_error=str(exc)[:400])
                )
        return tuple(preparations)

    def populate_prepared_baseline(
        self,
        handle: TurnHandle,
        preparations: Sequence[BaselineRootPreparation],
        touched_paths: Sequence[str] = (),
    ) -> None:
        """Snapshot an already-discovered root set on the caller's worker."""
        try:
            with handle._baseline_lock:
                if handle._accepting_baseline:
                    handle.roots[:] = [item.root for item in preparations]
            for item in preparations:
                root = item.root
                key = str(root)
                if item.tracking_error:
                    with handle._baseline_lock:
                        if handle._accepting_baseline:
                            handle.errors[key] = item.tracking_error
                    continue
                try:
                    repo = self.service.repo_for_root(root)
                    eligible = self._eligible_touched_paths(root, touched_paths)
                    if eligible:
                        baseline = repo.snapshot(
                            "turn baseline", force_paths=eligible
                        )
                    else:
                        baseline = repo.snapshot("turn baseline")
                    oversize = repo.last_oversize_excluded
                    with handle._baseline_lock:
                        if not handle._accepting_baseline:
                            continue
                        handle.auto_registered[key] = item.registered
                        handle.baseline_nested[key] = item.nested_repos
                        handle.baselines[key] = baseline
                        handle.baseline_oversize[key] = oversize
                except Exception as exc:  # noqa: BLE001 -- disclosed, never raised
                    with handle._baseline_lock:
                        if handle._accepting_baseline:
                            handle.errors[key] = str(exc)[:400]
        finally:
            handle._baseline_ready.set()

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
        handle = self.new_turn_handle(roots)
        frozen_touched_paths = tuple(touched_paths)

        if handle.roots:
            thread = threading.Thread(
                target=self.populate_baseline,
                args=(handle, frozen_touched_paths),
                name="change-review-baseline",
                daemon=True,
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
        follow_on._baseline_ready.set()
        return follow_on

    def finish_turn(
        self,
        handle: TurnHandle,
        touched_paths: Sequence[str] = (),
        *,
        end_shas: "dict[str, str] | None" = None,
    ) -> list[TurnChangeRecord]:
        """Take E synchronously on the caller-owned worker."""
        return self.end_turn(handle, touched_paths=touched_paths, end_shas=end_shas)

    def end_turn(
        self,
        handle: TurnHandle,
        touched_paths: Sequence[str] = (),
        *,
        end_shas: "dict[str, str] | None" = None,
        successor_handle: "TurnHandle | None" = None,
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
            successor_handle: The claimed turn whose baseline supplied
                ``end_shas``. Eligible ignored paths learned after that
                baseline are bound to this handle and staged atomically at
                its E snapshot instead of leaking through the shared index.

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
                    # The supplied snapshot stays immutable. Late ignored
                    # paths belong to the claimed successor and must not be
                    # left staged in the root-shared shadow index where an
                    # unrelated conversation could consume them.
                    end = provided
                    if end == baseline:
                        self._defer_to_successor(
                            successor_handle, key, provided, eligible
                        )
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
                    self._defer_to_successor(
                        successor_handle, key, provided, eligible
                    )
                    records.append(record)
                    continue
                force_paths = list(
                    dict.fromkeys(
                        (*eligible, *handle.force_paths_for_root(key))
                    )
                )
                if force_paths:
                    end = repo.snapshot("turn end", force_paths=force_paths)
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

    @staticmethod
    def _defer_to_successor(
        successor_handle: TurnHandle | None,
        root_key: str,
        boundary_sha: str,
        paths: Sequence[str],
    ) -> None:
        """Attach supplied-boundary paths to their exact successor handle."""
        if not paths:
            return
        if (
            successor_handle is None
            or successor_handle.baselines.get(root_key) != boundary_sha
        ):
            raise ValueError(
                "ignored paths have no matching claimed successor boundary"
            )
        successor_handle.defer_force_paths(root_key, paths)

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
                validated = validate_path(
                    Path(raw).expanduser(),
                    root,
                    redact_paths=True,
                    allow_hidden=True,
                )
                rel = validated.relative_to(root)
            except ValueError:
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


def initialize_shadow_root(root: Path | str) -> None:
    """Synchronously create one root's initial shadow snapshot.

    Background ownership belongs to :class:`ChangeReviewConsentService`;
    this helper performs only the filesystem/Git operation and either returns
    normally or raises so the owner can publish honest readiness.

    Args:
        root: Canonical workspace folder root.

    Raises:
        ChangeTrackingError: If shadow Git is unavailable.
        Exception: If repository initialization or snapshotting fails.
    """
    service = ShadowRepoService()
    if not service.available:
        raise ChangeTrackingError("Change Review shadow Git is unavailable.")
    service.repo_for_root(root).snapshot("root registered")
