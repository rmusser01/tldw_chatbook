---
id: TASK-261
title: Performance sundries: bg-effect frame cache, token-count gate, SELECT-1 ping, picker ctor I/O, MCP rebuild diffing, browser indexes
status: Done
assignee: []
created_date: '2026-07-16 14:30'
labels: [performance, ui]
dependencies: []
priority: low
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bounded small fixes from the audit: console_background_effect render_line recomputes the full W×H grid PER LINE = O(W·H²)/repaint (cache per frame tick); the 10s footer token-count re-tokenizes the whole visible history without a dirty check (app.py:5950-5954); get_connection pings SELECT 1 per query (~2× raw call count); BookmarksManager.__init__ does 5 sync Path.exists() (cloud-dir stalls) + first-run TOML write on every picker construction; MCP rail/servers-table/hidden-Tools-canvas full rebuilds ×2 per lifecycle action (N+2 store loads already tracked in task-236); consider indexes on conversations.deleted/last_modified for the browser ORDER BY at scale. Full analysis with measurements: Docs/Design/2026-07-16-performance-audit.md (§P3 D3).
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Each listed item fixed or explicitly declined with reasoning in the task notes
- [x] #2 No behavior changes (existing suites green)
<!-- AC:END -->

## Implementation Plan

1. **console_background_effect** — cache the computed frame lines on the widget instance, keyed by
   (frame serial, width, height); `_advance_frame` and `update_settings` bump the serial, resize
   changes the key. `render_line` reuses the cached lines instead of recomputing the W×H grid per line.
2. **Footer token-count** — extract a `_estimate_tokens_cached` helper in
   `Event_Handlers/Chat_Events/chat_token_events.py` with a cheap input signature
   (provider/model/limits/system prompt + message count/last-length/total chars) cached on the app
   instance; unchanged history reuses the previous counts, the footer is still updated every tick.
3. **SELECT 1 ping** — idle-threshold gate in `_get_thread_connection` for CharactersRAGDB,
   MediaDatabase, and PromptsDatabase (the three copies of the pattern): ping only when the
   thread-local connection has been idle ≥30 s; dead-connection transparent reopen preserved.
4. **BookmarksManager** — constructor becomes I/O-free; defaults (5× `Path.exists()`) computed
   lazily and cached, config load + first-run TOML write deferred to first actual use.
5. **MCP rebuild diffing** — read the code, decline if no clean targeted improvement (expected;
   reasoning pointing at task-236).
6. **Browser indexes** — grep for non-migration DDL precedent in ChaChaNotes; decline in favour of
   the next coordinated migration if none (migration numbers contested across sessions).
7. Regression tests per fixed item (spies over real implementations, unmocked sqlite/config seams);
   run module-local suites; commit + push.

## Implementation Notes

Four items fixed, two declined with reasoning. No behavior changes: every fixed item's test suite
asserts both that the optimization engages AND that output/behavior is byte-identical to the
pre-change path.

| # | Item | Outcome | Detail |
|---|------|---------|--------|
| 1 | console_background_effect O(W·H²) repaint | **FIXED** | `render_line` now reads `_frame_lines()`, a per-widget-instance cache keyed by (frame serial, width, height). The serial bumps on `_advance_frame` (tick) and `update_settings`; a resize changes the key. One `frame_text()` grid build per repaint instead of one per line; rendered strips proven byte-identical to the uncached grid. |
| 2 | Footer token-count re-tokenizing every 10 s tick | **FIXED** | New `_estimate_tokens_cached()` in `chat_token_events.py`: signature over provider/model/max-tokens/system-prompt + message count/last-message length/total chars, cached on the app instance. Unchanged history reuses the previous `(used, limit, remaining)`; the footer widget is STILL updated every tick (per-screen footers, task-264), so display behavior is unchanged. The typing (`_with_pending`) path is untouched — it changes every keystroke by construction. |
| 3 | `SELECT 1` liveness ping per `get_connection` | **FIXED** | The pattern lives in three copies — `CharactersRAGDB` (the audit's measured one), `MediaDatabase`, `PromptsDatabase` — not `base_db.py`. Chose the idle-threshold reduction (`_LIVENESS_PING_IDLE_SECONDS = 30 s`, `time.monotonic` stamp on `_local`): least behavior-changing because connections are thread-local/long-lived and `close_connection()` always clears the thread-local ref, so a recently-used connection is known-good; a long-idle one still gets ping + transparent reopen (covered by a real-sqlite recovery test using `set_trace_callback` to count actual pings). |
| 4 | `BookmarksManager.__init__` sync I/O | **FIXED** | Constructor is now I/O-free (verified by test: zero config reads/writes, zero `Path.home()` calls). Defaults (the 5× `Path.exists()` cloud-stall hazard) compute lazily via a cached property; config load and the first-run defaults TOML write happen on first actual use (`get_bookmarks`/`add`/`remove`/`is_bookmarked`), preserving the old first-use-visible behavior exactly. `save_to_config()` on a never-loaded manager is a no-op. |
| 5 | MCP rebuild diffing (rail/servers-table/hidden Tools canvas ×2 per lifecycle action) | **DECLINED** | Read the code first, per instructions. (a) The ×2 is two *different intentional states*, not a duplicate: `_start_lifecycle` fires an optimistic-CHECKING resync and a completion resync (documented in `_sync_children`'s T7 lock notes), so an equality-skip gate would never fire. (b) The rail's full recompose is load-bearing for the mount-echo guard machinery (`_displayed_scope_*` per-generation slots, per-instance `_mcp_mount_echo_value` tags — each guard's comments document real races that diffing would reopen). (c) `update_overview`'s from-scratch table rebuild is an explicitly documented decision ("simpler than tracking the previously rendered column set... not a hot path"), and the always-refresh of hidden canvases is a stated invariant ("switching INTO Audit mode never shows a stale window", `_sync_audit_mode` docstring). (d) The measured cost driver — N+2 store loads per pass — is already tracked as **task-236**; T10 already dedupes `_collect_hub_tools`/`_resolve_effective_states` to once per pass. A targeted diff here is exactly the risky churn on locked, extensively-adjudicated MCP UI the task warns against, for small residual gain. |
| 6 | Indexes on `conversations.deleted`/`last_modified` | **DECLINED** | Grepped for `CREATE INDEX` outside the migration path first, as required: within ChaChaNotes every index lives inside `_FULL_SCHEMA_SQL_V4` or a versioned `_MIGRATE_*` script, and `_initialize_schema()` returns without executing ANY DDL when the stored version equals `_CURRENT_SCHEMA_VERSION` — there is no existing non-migration DDL hook to ride. (Other DB modules that executescript idempotent schemas per-init are a different design, not precedent for out-of-band DDL layered onto a versioned schema.) Adding a new unconditional DDL step to the repo's most migration-contested schema (v21 now; parallel sessions have collided on version numbers twice) is not worth it for a P3 at-scale concern. Recommendation: the next coordinated ChaChaNotes migration (v21→v22, whoever claims it) should include `CREATE INDEX IF NOT EXISTS idx_conversations_deleted_last_modified ON conversations(deleted, last_modified)` to cover the browser's `WHERE deleted = 0 ... ORDER BY last_modified DESC` list queries in `ChaChaNotes_DB.py`. |

**Files modified**: `tldw_chatbook/Widgets/Console/console_background_effect.py`,
`tldw_chatbook/Event_Handlers/Chat_Events/chat_token_events.py`,
`tldw_chatbook/DB/ChaChaNotes_DB.py`, `tldw_chatbook/DB/Client_Media_DB_v2.py`,
`tldw_chatbook/DB/Prompts_DB.py`, `tldw_chatbook/Widgets/enhanced_file_picker.py`.
**Tests added**: 2 in `Tests/UI/test_console_background_effects.py` (now 11 total),
`Tests/Chat/test_footer_token_dirty_gate.py` (7), `Tests/DB/test_connection_liveness_ping_gate.py`
(9, parametrized over all three DB classes, real on-disk sqlite + trace-callback ping counting),
`Tests/UI/test_file_picker_bookmarks_lazy.py` (5).
**Suites run green**: Tests/DB + Tests/Prompts_DB (162), Tests/Media_DB (59+6s), Tests/Chat
(959+69s), Tests/Event_Handlers + config-console-defaults (112+26s), picker consumers (7).
Tests/ChaChaNotesDB: 156 passed + exactly the 3 known pre-existing legacy-parity failures listed in
the audit brief — no new failures.
