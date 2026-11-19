# S19 — UI/Screens B — validation

Validated against `d0face3ebe` (origin/dev). Review was done at `3722a85748`. Note: `git log --oneline
3722a85748..HEAD -- <file>` returns **no commits** for `stats_screen.py` or `chatbooks_screen.py` — those files
are byte-identical to the review commit, so the large line-number discrepancies noted below (findings 1 and 10/11)
are citation errors in the review document itself, not drift from dev moving on. The underlying defects are real
regardless.

## 1. P1 [D1] — `StatsScreen` crashes when a chat topic contains a non-ASCII word
- Verdict: CONFIRMED
- Site now: `UI/Screens/stats_screen.py:69` (`Container(classes="topic-bar-fill", id=f"bar-{self.topic}")` inside
  `TopicBar.compose()`), reached from `:414` (`TopicBar(topic.capitalize(), count, max_count)`). (Review cited
  `:590`/`:935` — wrong even at the review's own commit: `git show 3722a85748:...stats_screen.py` is 527 lines
  total and has the same code at `:69`/`:414`.)
- Proof: `textual/css/tokenize.py:32` → `IDENTIFIER = r"[a-zA-Z_\-][a-zA-Z0-9_\-]*"` (ASCII-only).
  `python -c "from textual.widgets import Container; Container(id='bar-Über')"` raises `BadIdentifier`. Word
  extraction confirmed Unicode-aware: `re.findall(r'\b\w{4,}\b','über künstliche')` → `['über','künstliche']`,
  both `.isalpha()` True (`Stats/user_statistics.py:370,372`). The `except Exception: pass` guard the file DOES
  have (`stats_screen.py:73-76`, inside `on_mount`) only wraps the `query_one` lookup, not the `Container(id=...)`
  construction inside `compose()`, which is unguarded and runs first.

## 2. P1 [D1] — Settings/Console endpoint probe's chat path skips the egress SSRF check the TTS path in the same function uses
- Verdict: CONFIRMED
- Site now: `UI/Screens/settings_endpoint_probe.py:607-628` (`probe_settings_endpoint` chat branch calling
  `_request_models` with no gate) vs `:514-517` (`_probe_openai_tts_catalog` →
  `check_url_or_raise_async(endpoint.catalog_url, trusted_origins=origin_set(endpoint.origin))`). Credential
  attach at `:295`. (Review's `:607-641`/`:514-517`/`:294-295` all land within a line or two of the real sites.)
- Proof: `grep -n "check_url_or_raise_async" settings_endpoint_probe.py` → exactly one hit, `:514` (TTS branch
  only; the import at `:44` is the only other occurrence). The four production callers
  (`UI/Speech/speech_catalog_mixin.py:145`, `UI/Screens/chat_screen.py:3241`, `UI/Screens/settings_screen.py:15174`,
  `UI/Wizards/FirstRunSetupWizard.py:934`) each `grep`ped individually for `check_url_or_raise|egress` — zero
  hits in any of them. `settings_screen.py:18018` does contain the doc-comment the review quotes ("fetches go
  through the egress SSRF ...") describing a *neighbouring* feature, confirming the asymmetry.
- Note: the review's own conditional P0 escalation ("if a non-user-authored config path can set `base_url`") is
  left as filed — a spot check of `Backup_Recovery/credentials.py`/`credential_policies.py` shows the restore path
  does handle `base_url`-shaped fields (`_config_server_id` reads `api.get("base_url")`), but that is the
  `tldw_api` server binding, not confirmed to be the same `api_settings.<provider>.base_url` /
  `custom_endpoints.*` field the probe reads — insufficient to independently promote to P0 in the time available.

## 3. P1 [D1] — `SchedulesWorkbench._run_sync`'s `finally` dereferences the DOM after an awaited network sync, `exit_on_error=True`
- Verdict: CONFIRMED
- Site now: `UI/Screens/scheduling/schedules_workbench.py:5393` (dispatch, no `exit_on_error=`), `:5395-5428`
  (`_run_sync`), `finally:` at `:5426-5429` with two bare `query_one(btn_id, Button)` calls after
  `await service.sync_now(owner_id)` at `:5407`. Exact same line numbers as filed.
- Proof: read the full method. `screen_registry.py` "schedules" route has no `reusable=True` (only 3 routes in
  the whole registry do), so navigating away unmounts the screen mid-await, and the worker dispatch omits
  `exit_on_error=False`.

## 4. P2 [D3] — `check_textual_worker_contract.py`'s W002 guard treats any ancestor `Try` as "guarded", including `finally`/`except` bodies
- Verdict: CONFIRMED
- Site now: `scripts/check_textual_worker_contract.py:216-224` — `collect_w002`'s guard walk is
  `while cursor is not None and cursor is not func: if isinstance(cursor, ast.Try): guarded = True; break`, with
  no check of which `Try` section (`body`/`handlers`/`orelse`/`finalbody`) the node sits in.
- Proof: read the source directly — confirms the mechanism exactly as described. Spot-checked absence:
  `grep -n "_run_sync" scripts/textual_await_dom_census.tsv` → no row for the P1-3 `_run_sync` finally-body
  lookups, consistent with the census missing them. Did not re-run the full repo-wide count to re-derive "59" —
  the mechanism that produces false negatives is verified; the exact count is not independently reproduced.

## 5. P2 [D1/D2] — `_settle_orphaned_transfers` runs a synchronous DB sweep on the event loop despite a docstring claiming it was moved off-loop
- Verdict: CONFIRMED
- Site now: `scheduling/schedules_workbench.py:4647-4699` (`_settle_orphaned_transfers`), inner `async def
  _settle()` at `:4684-4695` calling `service.sync_engine._settle_orphaned_transfer_mutations(target_owner)`
  directly (no `asyncio.to_thread`), dispatched via `self.run_worker(_settle, exclusive=True,
  group="schedules-orphan-sweep")` — same shape/lines as filed.
- Proof: `Scheduling/services/sync_engine.py:173` confirms `_settle_orphaned_transfer_mutations` is a plain
  `def` (not `async def`) doing DB reads/writes. The docstring at `:4671-4676` (was cited 4669-4676) describes
  only a timing fix ("used to run ... synchronously from `on_mount` ... Deferred onto the same fire-and-forget
  worker path"), never claiming a thread hop — matching the review's reading that the comment implies safety it
  doesn't have. `:1305-1307`-area text ("the same 'local DB read, off-thread' discipline every `service.db.*`
  read here uses") is present verbatim, confirming the contrast with the file's actual convention elsewhere.

## 6. P2 [D2] — `BackupRestoreScreen`'s 0.2s poller re-`update()`s unconditionally, never pauses while covered
- Verdict: CONFIRMED
- Site now: `UI/Screens/backup_restore_screen.py:504` (`set_interval(0.2, self._refresh_status)`), `:1690`
  (`self.query_one("#backup-status", Static).update(text)` — unconditional once `current is not None`). Exact
  same lines as filed.
- Proof: `grep -n "on_screen_suspend\|on_screen_resume"` on the file → zero hits. `_refresh_status`
  (`:1604-1610`) only early-returns on `current is None`; once any operation exists it always reaches `.update`.
  Confirmed the sibling fixes exist in the same slice: `schedules_workbench.py` has both
  `_update_static_content` (`:625-628`, compare-before-update) and `on_screen_suspend`/`on_screen_resume`
  (`:1174`, `:1185`).

## 7. P2 [D3] — Direct `await self.recompose()` bypasses `BaseAppScreen`'s focus-restore seam
- Verdict: CONFIRMED
- Site now: `UI/Screens/workflows_screen.py:157`; repo-wide:
  `UI/Library_Modules/library_unavailable_navigation.py:354,536`,
  `UI/Library_Modules/library_inspection_admission.py:403`, and `UI/Screens/library_screen.py:11322,12403,
  21842,21939,22041,22450` (one more site than the review's list — `:22450` — undercounting, not overcounting).
- Proof: `base_app_screen.py:91-176` (`refresh(recompose=True)`) captures/restores focus via
  `_focus_identity_for_recompose` (`:181`); the `recompose()` override (`:267-...`) releases only mouse capture
  — grepped for "focus" inside its body, zero hits. `textual/widget.Widget.recompose()` removes children and
  restores no focus (base behaviour).

## 8. P2 [D4] — `settings_image_gen_defaults.py` / `settings_video_gen_defaults.py`: parallel data layer, `_coerce_value` drift
- Verdict: CONFIRMED
- Site now: `settings_image_gen_defaults.py` (985 lines, `_coerce_value` at `:428`),
  `settings_video_gen_defaults.py` (439 lines, `_coerce_value` at `:292`) — line counts and function lines match
  the review almost exactly.
- Proof: image `_coerce_value` (`:428-444`) has `int`/`float`/`origin` branches, no `bool` branch at all; video
  `_coerce_value` (`:292-304`) has `int`/`bool` branches, no `float`/`origin`. Video's `FIELD_SCHEMA` declares
  `FieldSpec("allow_uploads", ..., "bool")` at `:68` — a field the image module's `_coerce_value` cannot type
  correctly if mirrored.

## 9. P2 [delete] — `UI/Screens/schedules_screen.py` (516 lines) is unrouted, zero importers, stated reason is false
- Verdict: CONFIRMED
- Site now: whole file — 516 lines exactly, docstring at `:1-6` still reads "DEPRECATED ... unrouted and
  retained only because existing tests still exercise it directly."
- Proof: `grep -rn "schedules_screen" tldw_chatbook/` → only the file itself. `grep -rn "SchedulesScreen\b"
  tldw_chatbook/ Tests/` → the class is defined but never imported anywhere; no test file imports/instantiates
  it (the "tests still exercise it directly" claim is itself false — the 3 test hits are all string literals, as
  the review states). `screen_registry.py:128-133` maps route `"schedules"` → `SchedulesWorkbench` in
  `scheduling/schedules_workbench.py`, not this module.

## 10. P3 [D3] — `stats_screen.py`: stdlib `logging`, dead expression in `compose()`, 3x redundant rebuild
- Verdict: CONFIRMED
- Site now: `:6` (`import logging`), `:27` (`logger = logging.getLogger(__name__)`) — NOT `:527, 548` as filed
  (the file is only 527 lines total; those citations were wrong even at the review's own commit, see header
  note). Dead expression: `:63` inside `TopicBar.compose()` — `(self.count / self.max_count * 100) if
  self.max_count > 0 else 0` as a bare, discarded statement — recomputed identically at `:72` inside
  `on_mount()` where it is actually used (was cited `:585`/`:594`). Triple rebuild:
  `watch_is_loading`/`watch_stats_data`/`watch_error_message` at `:235,241,247`, each calling
  `self.call_after_refresh(self.refresh_stats_display)`; `_apply_statistics_result` (`:123-136`) sets
  `self.stats_data`, `self.error_message`, `self.is_loading` in sequence, each a separate reactive write (was
  cited `:756-772`/`:644-652`).
- Proof: read all cited regions directly at their real line numbers; the `import logging`/`getLogger` shape, the
  duplicated percentage expression, and the three sequential reactive writes are all present exactly as
  described, just far from the review's stated line numbers.

## 11. P3 [D3] — `ChatbooksScreen`: three dead public methods, four write-only reactives
- Verdict: CONFIRMED
- Site now: whole file is 91 lines (matches the review's "~45 of 91 lines" framing exactly). Reactives at
  `:24-27`; dead methods `create_new_chatbook:55`, `open_chatbook:67`, `delete_chatbook:79` (review cited
  `:317-320`/`:348-384`, both far outside this 91-line file — wrong from the start, same pattern as finding 10).
- Proof: `grep -rn "create_new_chatbook\|open_chatbook\|\.delete_chatbook(" tldw_chatbook/ Tests/` → zero hits
  outside the screen file itself; every `delete_chatbook` hit elsewhere is a `service.`/`scope.` object method
  (confirmed by inspection, e.g. `Tests/Chat/test_citation_artifact_ownership.py`). `on_screen_resume`
  (`:47-53`) does call `chatbooks_window._refresh_chatbooks()` — a private cross-module method, as filed.

## 12. P3 [D4] — `research_screen.py` / `writing_screen.py` verbatim twins
- Verdict: CONFIRMED
- Site now: `research_screen.py:40-64`, `writing_screen.py:48-68` — exact match to the review's citation.
- Proof: read both `save_state`/`restore_state` pairs; bodies are identical except for the window class and the
  `#…-window` id, exactly as filed.

## 13. P3 [D3] — `task_detail.set_lifecycle_lock` parses its own output back out and drops it, plus a discarded `query_one`
- Verdict: CONFIRMED
- Site now: `scheduling/task_detail.py:1704-1758` (`set_lifecycle_lock`, the round-trip at `:1750-1758`) and
  `:1469` (discarded `query_one`) — both exact matches to the review's citation.
- Proof: `grep -rn "scheduling-transfer-why" tldw_chatbook/` → only `task_detail.py` writes it (compose `:910`,
  clear `:1482`, write `:1750-1758`); an AST scan for a bare `Expr(Call(... .query_one))` in this file returns
  exactly one hit, `:1469` — `self.query_one("#schedules-follow-in-console", Button)` with no assignment/chain.

## 14. P3 [D2] — `WorkbenchHostScreen._on_message` calls `route_message` for every message, unfiltered against a 14-entry tuple
- Verdict: CONFIRMED
- Site now: `scheduling/workbench_host_screen.py:92-104` (`_on_message`, docstring at `:73` says "deliberately
  unfiltered"); consumer `schedules_workbench.py:1951` checks `isinstance(message, _PUSHED_DETAIL_MESSAGES)`.
  `_PUSHED_DETAIL_MESSAGES` (`:181-194`) has exactly 14 entries — counted directly.
- Proof: read both sites; message-type count confirmed by listing the tuple's 14 names.

## 15. P3 [D2] — `SchedulesWorkbench` runs synchronous `service.db.*` reads on the event loop in badge/overlay paths
- Verdict: CONFIRMED
- Site now: `_unread_result_ids` at `:4399-4413` — `service.db.count_unread_results(owner_id=None)` then an
  unbounded `service.db.list_automation_results(owner_id=None, review_state="unread", limit=unread_total)`, no
  `await`/`to_thread`. `_push_results_overlay`/`_push_conflicts_overlay` (`:2435`/`:2522`) similarly documented
  as pre-reading via `service.db.*` before pushing a view.
- Proof: read the method bodies directly; both plain synchronous calls confirmed, matching the review's claim
  that this contradicts the file's own stated off-thread discipline.

## 16. P3 [D3] — `MCPScreen.on_screen_suspend` mutates two private attributes of `MCPWorkbench`
- Verdict: CONFIRMED
- Site now: `mcp_screen.py:296-303` (was cited `:822-823` — shifted, same code):
  `self.workbench._mcp_recovery_busy` / `self.workbench._mcp_recovery_token = None`.
- Proof: `grep -n "_mcp_recovery_busy\|_mcp_recovery_token"` confirms both private-attribute accesses at the
  cited `on_screen_suspend` method.

TOTALS: confirmed=16 fixed=0 wrong=0 demoted=0 promoted=0
