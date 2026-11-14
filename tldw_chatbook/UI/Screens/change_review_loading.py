"""Native execution methods behind the existing service; ownership stays on the caller."""

from __future__ import annotations

from tldw_chatbook.Utils.log_sanitizer import content_fingerprint

from . import change_review_screen as native_host


def _load_turn(self, turn: native_host.ReviewTurn) -> None:
    # TASK-18060 final-review fix round (Fix 1b + Fix 2): both must
    # happen before anything below rebuilds the tree/leaves. Fix 1b: a
    # comment input opened against the PRIOR turn's leaf is a SIBLING
    # of the diff pane, not a descendant of anything rebuilt here --
    # left mounted, a later Enter on it would save a note whose
    # row/change was captured against a turn this screen has already
    # left. Fix 2: bumping the generation invalidates the per-leaf
    # diff-text memo so a revert-reload of THIS SAME turn (identical
    # row objects, identical path) refetches instead of serving
    # pre-revert content.
    self._close_any_open_comment_input()
    self._diff_cache_generation += 1
    self._diff_cache_key = None
    self._diff_cache_text = None
    self._diff_cache_error = None
    self._active_turn = turn
    multi_root = len(turn.rows) > 1
    window_labels = native_host._window_labels(turn.rows)
    tree = self.query_one("#change-review-tree", native_host.Tree)
    tree.clear()
    tree.root.expand()
    self._leaves = []
    banners: list[str] = []
    grouped: dict[str, list[tuple[dict, native_host.ChangedFile]]] = {}
    for row in turn.rows:
        error = str(row.get("tracking_error") or "")
        if error:
            banners.append(f"⚠ tracking failed for {row['root']}: {error}")
            continue
        nested_raw = row.get("nested_repos") or "[]"
        try:
            import json as _json

            nested = [str(p) for p in _json.loads(nested_raw)]
        except (ValueError, TypeError):
            nested = []
        if nested:
            # TASK-1976: name the holes — changes inside these repos
            # are not tracked at all.
            limit = native_host.NESTED_BANNER_NAMED_LIMIT
            shown = ", ".join(nested[:limit]) + (
                f" (+{len(nested) - limit} more)" if len(nested) > limit else ""
            )
            plural = "ies" if len(nested) != 1 else "y"
            banners.append(
                f"⚠ {len(nested)} nested repositor{plural} inside "
                f"{row['root']} not tracked: {shown}"
            )
        oversize = int(row.get("untracked_oversize") or 0)
        if oversize:
            # TASK-1975 AC#2: cost bounds are honest, not silent.
            plural = "s" if oversize != 1 else ""
            banners.append(
                f"⚠ {oversize} oversized file{plural} untracked for "
                f"{row['root']} (over the size cap)"
            )
        try:
            for change in self._provider.changed_files(row):
                grouped.setdefault(change.status, []).append((row, change))
        except native_host.ChangeTrackingError as exc:
            if self._provider.snapshots_pruned(row):
                banners.append(f"history for {row['root']} was pruned by retention")
            else:
                banners.append(f"⚠ diff unavailable for {row['root']}: {exc}")

    touched_by_row: dict[int, "set[str] | None"] = {}

    def _badged(row: dict, change: native_host.ChangedFile) -> bool:
        rid = id(row)
        if rid not in touched_by_row:
            try:
                touched_by_row[rid] = self._provider.tool_touched_relpaths(row)
            except Exception as exc:  # noqa: BLE001 -- badge must never break review
                native_host.logger.warning(
                    "change_review: badge derivation failed; rendering without "
                    "badges operation=badge root_sha256={} exception_type={}",
                    content_fingerprint(str(row.get("root") or "")),
                    type(exc).__name__,
                )
                touched_by_row[rid] = None
        touched = touched_by_row[rid]
        if touched is None:
            return False
        # Qodo #1262: no file tool can DELETE or RENAME — those rows
        # badge regardless of path membership (a write_file-created
        # path later script-deleted must not launder the deletion).
        if change.status in ("D", "R"):
            return True
        return change.path not in touched

    self._populate_tree(
        tree,
        grouped,
        multi_root,
        _badged,
        window_labels=window_labels,
    )

    self._turn_banner_lines = banners
    # A turn view carries no current-mode per-root failures.
    self._current_root_errors = []
    self._update_banner()
    # TASK-16801 arc B: a snapshot turn is showing -- commit goes away,
    # the snapshot-only affordances come back.
    self._refresh_mode_affordances()

    totals = self.query_one("#change-review-totals", native_host.Static)
    adds = sum(int(r["adds"] or 0) for r in turn.rows)
    dels = sum(int(r["dels"] or 0) for r in turn.rows)
    noun = native_host._file_count_noun(
        len(self._leaves), repeated_root=bool(window_labels)
    )
    totals.update(f"{len(self._leaves)} {noun}  +{adds} −{dels}")

    # TASK-18060 Task 3: the initials are constructor state consumed
    # exactly ONCE -- cleared here, UNCONDITIONALLY, so a later turn
    # switch (this same method, via select_turn/Select.Changed) reverts
    # to the first leaf like today. This must happen before the
    # `self._leaves` branch below: a zero-leaf initial turn (a
    # tracking-error row, or one whose snapshots were pruned) must not
    # leave the initials sitting around to hijack focus on the NEXT
    # `_load_turn` call once a later turn happens to contain a
    # same-named path (reviewer catch on the first cut of this task).
    initial_path = self._initial_path
    initial_snapshot_id = self._initial_snapshot_id
    self._initial_path = None
    self._initial_snapshot_id = None

    if initial_path is None and initial_snapshot_id is not None:
        for index, (row, _change) in enumerate(self._leaves):
            if row.get("id") == initial_snapshot_id:
                self._focus_leaf(index)
                return
        self._focused_leaf = -1
        present = any(row.get("id") == initial_snapshot_id for row in turn.rows)
        self._show_empty(
            "No file changes in the selected snapshot."
            if present
            else "The selected snapshot is unavailable in this turn."
        )
        self._refresh_notes_strip()
        return
    if self._leaves:
        # A path that doesn't exist in this turn (stale rail cache, a
        # revert since) degrades to the first leaf rather than an empty
        # pane -- select_file itself stays a no-op on an unmatched path
        # so its legacy behavior (external callers with no fallback
        # expectation) is untouched.
        if initial_path is not None and any(
            change.path == initial_path for _row, change in self._leaves
        ):
            self.select_file(initial_path, snapshot_id=initial_snapshot_id)
        else:
            self._focus_leaf(0)
    else:
        self._show_empty("No file changes in this turn.")
        # TASK-18060 Task 7: `_focus_leaf` is the strip's usual refresh
        # choke point, but a zero-leaf turn never reaches it — clear
        # (hide) any stale strip content from a previously focused
        # turn explicitly.
        self._refresh_notes_strip()
