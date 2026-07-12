# Follow-ups persistence batch — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Three cross-visit persistence follow-ups (backlog 164, 165, 166). Branch `claude/followups-persistence` off dev b37ce5b2. Anchors exact at branch point; grep symbols, lines drift.

**Goal:** Restore Library per-pane filters across tab switches (164), re-populate the Media viewer + row highlight on restore (165), and make repeat Library visits render instantly from an app-scoped snapshot cache without staleness (166).

**Context (binding):** Since the freeze fix (PR #595) navigation composes a FRESH screen instance per visit; continuity is via the app-owned `_screen_states` (`save_state`/`restore_state`, in-memory) called before mount. A per-SCREEN-INSTANCE cache does NOT survive a tab round-trip — 166 must be app-scoped.

**Global Constraints:** explicit-path staging (NEVER `git add -A`); Fable 5 co-author line; RED-first; parameterized SQL only; `escape_markup` for user text in labels; venv pytest with isolated HOME. Test command: `HOME=/private/tmp/tldw-chatbook-test-home XDG_DATA_HOME=/private/tmp/tldw-chatbook-test-home/.local/share /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest <files> -q -p no:cacheprovider -o addopts="" --timeout=300 --timeout-method=thread`.

## Cluster A — persistence extensions (164 + 165)

### T164 — persist Library per-pane filters/sort
**Files:** Modify `tldw_chatbook/UI/Screens/library_screen.py` (`save_state` :761, `restore_state` :795). Test: `Tests/UI/test_library_shell.py`.
- The existing save/restore (PR #595) persists selection/view + RAG state but NOT the per-pane filters. Add these four attrs:
  - `_library_media_type_filter` (:548, default `"All"`) — the media type cycle.
  - `_library_notes_sort` (:559, default `"newest"`) — notes sort mode.
  - `_library_notes_filter` (:560, default `""`) — notes substring filter VALUE.
  - `_library_conversation_query` (set in `handle_library_conversations_filter_submitted` :7331 via `_safe_text`) — conversations filter query.
- save_state: `state["library_media_type_filter"] = self._library_media_type_filter`, `state["library_notes_sort"] = ...`, `state["library_notes_filter"] = ...`, `state["library_conversation_query"] = getattr(self, "_library_conversation_query", "")`. Do NOT persist `_library_notes_filter_records` (:561, a recomputed cache — leave it None so restore recomputes).
- restore_state: read each back with a type-guard + default (mirror the existing `str(state.get(...) or default)` idiom); coerce `_library_media_type_filter` to a str defaulting `"All"`, `_library_notes_sort` to a str defaulting `"newest"`, the two filters to `""` default. Re-sanitize `library_conversation_query` through `self._safe_text` on restore (defense — it's user text, and a saved-state dict isn't statically typed).
- These attrs are read by the canvas builders at mount (`active_type=self._library_media_type_filter` :2694; notes `sort_mode`/`filter_value` :2544-2550; conversations query in `_build_library_conversations_state`), so setting them pre-mount just works — no on_mount re-kick needed (unlike a fetched detail). Verify by reading each builder call.
- RED pilot: cycle the media type filter off "All" (and set a notes sort/filter), save_state → restore_state on a fresh screen → assert the attrs are restored AND the media canvas renders the restored type (mirror the existing filter pilots' harness). One pilot per pane is fine; assert the round-trip via the real save/restore methods.

### T165 — Media restore re-populates the viewer detail + row highlight
**Files:** Modify `tldw_chatbook/UI/MediaWindow_v2.py` (`apply_restored_view_state` — currently returns after setting `active_media_type`/`selected_media_id` + re-running search, deliberately NOT fetching viewer detail). Test: media window tests (`ls Tests/UI | grep -i media_window`; else `Tests/UI/test_library_content_hub.py` neighbors — adjudicate).
- Today `apply_restored_view_state` sets the scalars + re-runs the list search but leaves the viewer empty and the row un-highlighted (documented as "kept cheap"). T165 wires the missing half: when a `selected_media_id` is restored, trigger the SAME scoped media-detail fetch a live row click triggers (grep the row-click handler in MediaWindow_v2 for the detail-load method — likely `_load_media_details`/`_select_media_item`/a worker) so the viewer panel re-populates, and highlight the restored row in the list.
- Stale-id safety: if the restored `selected_media_id` no longer resolves (deleted while away), the fetch must degrade the same way a live click on a since-deleted row does (grep that path — it likely notifies/clears the viewer). Do NOT crash; do NOT leave a permanent loading placeholder.
- Keep it side-effect-safe before first paint: if the detail fetch is async/worker-based, kick it the same fire-and-forget way the row click does (don't await in the sync restore path). Update the method's docstring (it currently says detail restore is deliberately skipped — reword to describe the new behavior).
- RED test: restore with a `selected_media_id` present in the seeded set → the viewer detail loads for that id (assert the detail-fetch was invoked / the viewer shows the item) and the row is highlighted; restore with a stale id → no crash, viewer degrades. Follow the existing media-window test harness.

**Cluster A commit:** `feat(library,media): persist per-pane filters and restore the media viewer on return (164,165)`

## Cluster B — app-scoped Library snapshot cache (166)

### T166 — instant repeat-visit render via an app-level snapshot memo
**Files:** Modify `tldw_chatbook/UI/Screens/library_screen.py` (`on_mount` :652, `_refresh_local_source_snapshot` :1043, `_apply_local_source_snapshot`); store the cache on the APP (`self.app_instance`), not the screen. Test: `Tests/UI/test_library_shell.py`.
- Problem: `on_mount` calls `_refresh_local_source_snapshot()` on EVERY visit (screens are fresh per visit now), so repeat Library visits re-run the full DB snapshot fetch and show nothing until it returns. A memo on `self` is useless (new instance each visit).
- Design — app-scoped cache + instant-then-reconcile (avoids staleness beyond one refresh cycle):
  - App-level attr, e.g. `app_instance._library_source_snapshot_cache: tuple | None` and `app_instance._library_source_snapshot_cache_stamp: float | None` (init lazily via getattr; do NOT require app.py changes — use `getattr(self.app_instance, "_library_source_snapshot_cache", None)`).
  - `_refresh_local_source_snapshot` (after computing the snapshot tuple) writes it + a `time.monotonic()` stamp to the app cache.
  - `on_mount`: if a cached snapshot exists AND `monotonic() - stamp < LIBRARY_SNAPSHOT_CACHE_TTL_SECONDS` (define, e.g. `5.0`), apply the CACHED snapshot synchronously first (`_apply_local_source_snapshot(*cached)`) so the returning visit renders instantly, THEN still call `_refresh_local_source_snapshot()` to reconcile in the background (its completion re-applies fresh data + refreshes the cache). If no fresh cache, keep today's behavior (just `_refresh_local_source_snapshot()`). This bounds staleness to a single async refresh cycle — the AC's "no stale data beyond the memo window".
  - Invalidation: every existing `_refresh_local_source_snapshot` call site already re-applies + re-caches, so Library-side mutations self-heal. Cross-screen mutations (e.g. ingest-tab note import — task 167) already trigger a Library refresh when Library is mounted; when it is NOT mounted the next visit's background reconcile corrects within one cycle. Document this bound.
- CAUTION: the cache stores the raw snapshot tuple (records/counts/etc.) — the SAME data `_apply_local_source_snapshot` already holds transiently. It is not a NEW fetched-data retention surface (it mirrors what a live screen holds), and it's app-scoped so it's shared, not per-instance-leaked. Do not cache selection/view (that's `_screen_states`' job).
- RED pilot: mount Library (populates cache), unmount, mount again within TTL → assert the second mount applied the cached snapshot BEFORE its own refresh completed (e.g. the canvas has content at first paint / `_apply_local_source_snapshot` was called with the cached tuple before the fresh fetch resolves — use a gated fake for the fetch to widen the window, bounded 30s). Second pilot: after TTL expiry, no stale apply (fresh fetch only). Verify RED: without the cache, the first-paint content assertion fails.

**Cluster B commit:** `perf(library): app-scoped snapshot cache for instant repeat visits (166)`

## Verification & gate

Combined gate: `Tests/UI/test_library_shell.py Tests/UI/test_home_screen.py Tests/UI/test_library_content_hub.py Tests/Media/ Tests/Library/` + media-window tests + `python -c "import tldw_chatbook.app"`. Visual QA: 164 (a restored media type filter still applied after a tab round-trip) and 165 (media viewer populated on return) are visible — capture in the served TUI if the round-trip is scriptable; else rely on the pilots and note it. Mark backlog 164/165/166 Done (tick ACs). PR to dev; merge only on explicit user authorization.
