# S18 — UI/Screens A — validation

Validated against `d0face3ebe` (origin/dev). Review was done at `3722a85748`.

## 1. P1 [D1] — `evals_screen.py` still uses `rich.markup.escape`
- Verdict: CONFIRMED
- Site now: `UI/Screens/evals_screen.py:38` (import), used at `:2947` (`ConfirmationDialog` message), `:3635`,
  `:3670`, `:3732` (Button label/tooltip). (was `:38`/`:2536`/`:3223`/`:3258`/`:3320` — same import line, usage
  sites shifted ~+400 lines from earlier docstring growth in the file, still the same call shapes.)
- Proof: `sed -n '38p'` shows `from rich.markup import escape as escape_markup`; `sed -n '2947p'` shows
  `message=f'Delete "{escape_markup(name)}"? This can\'t be undone.'`. Repro:
  `python -c "from rich.markup import escape as e; from textual.content import Content; print(Content.from_markup(e('Run [TODO] Q3 plan')).plain)"`
  → `'Run  Q3 plan'` (token silently deleted), matching the review's evidence line verbatim.

## 2. P1 [D1] — `change_review_screen.py` raw git stderr into `notify()` with markup ON
- Verdict: CONFIRMED
- Site now: `UI/Screens/change_review_screen.py:2884` (`_land_commit_refused`), `:2897` (`_land_commit_error`),
  `:3211` (`_land_push_refused`), `:3224` (`_land_push_error`) — identical to the lines the review cited.
- Proof: `grep -n 'self.notify(message, severity="warning")\|self.notify(f"Commit could not run\|self.notify(f"Push could not run'`
  returns exactly those four lines, none passing `markup=`. `App.notify` signature confirmed via
  `inspect.signature(App.notify)` → `markup: bool = True` default; `textual/widgets/_toast.py Toast.render`
  confirmed to call `Content.from_markup(notification.message)` whenever `notification.markup` is true.

## 3. P1 [D1] — Three watchlists write workers have no error handling, dispatched with app-exiting `exit_on_error` default
- Verdict: CONFIRMED
- Site now: `UI/Screens/watchlists_collections_screen.py:14214/14217` (`_mark_all_read_worker`, body
  `14217-14252`, no `try`), `:8112/8117` (`_mark_notification_read`), `:8130/8135` (`_dismiss_notification`) —
  line numbers essentially unchanged from the review (14213/8111/8129).
- Proof: read all three bodies — none contains a `try`; all three `run_worker(...)` dispatch calls (`:8109-8113`,
  `:8128-8132`, `:14213-14215`) omit `exit_on_error=`. `inspect.signature(DOMNode.run_worker)` confirms
  `exit_on_error: bool = True` is the default.

## 4. P2 [D1] — `research_workspace_screen.py` omits `exit_on_error=False` on `_start_catalog_refresh`'s worker
- Verdict: CONFIRMED
- Site now: `UI/Screens/research_workspace_screen.py:456-461` (dispatch, no `exit_on_error=`), body
  `_refresh_workspace_catalog:463-467` → `_apply_catalog_state:469-...` — line numbers unchanged from the review.
- Proof: of the file's 8 `run_worker` calls, `:457`, `:1060`, `:1631` omit `exit_on_error=False` while
  `:292`, `:603`, `:726`, `:735`, `:1076` all pass it — exactly the 5-of-8 split the review states. Route
  `"research_workspace"` has no `reusable=True` entry in `screen_registry.py` (`grep -n reusable` shows only 3
  routes are reusable, and it isn't one of them), confirming the unmount-on-navigate premise.

## 5. P2 [D2] — Watchlists item-status write does a redundant triple hop now that the service offloads itself
- Verdict: CONFIRMED
- Site now: `UI/Screens/watchlists_collections_screen.py:13442-13509` (`_update_item_status_off_loop`), same
  location as filed.
- Proof: `LocalWatchlistsService.update_item` (`Subscriptions/local_watchlists_service.py:966-1010`) ends in
  `await run_db_off_loop(db, db.mark_item_status, row_id, normalized_status)`, and
  `Subscriptions/db_offload.py:70-77`'s `run_db_off_loop` itself does `await asyncio.to_thread(run_and_close)` for
  a real `SubscriptionsDB`. The screen wrapper's docstring (`:13446-13457`) still asserts "every layer of that call
  chain ... has no genuine `await` of its own," which the `run_db_off_loop` call directly contradicts — the
  wrapper's own `asyncio.to_thread(lambda: asyncio.run(...))` is layered on top of that, producing the claimed
  thread → throwaway loop → second thread shape.

## 6. P2 [D2] — Skill-eval run worker writes ~66 individually-committed sqlite rows on the event loop
- Verdict: CONFIRMED
- Site now: `UI/Screens/evals_screen.py:2502` (`_run_skill_eval_worker`), dispatched at `:1980-1984` as a bare
  coroutine (`self.run_worker(self._run_skill_eval_worker, exclusive=True, group="evals-run-skill-eval")`, no
  `thread=True`). Persistence loop at `:2604-2607` (judge artifacts) and `:2608-2632` (sim cells).
- Proof: `Evals/skill_eval/storage.py:129-145 save_artifact` → `db.store_result`
  (`DB/Evals_DB.py:1800`), whose body is `with self.connection() as conn: with conn: conn.execute(INSERT ...)` —
  one committed transaction per call. `Evals/skill_eval/models.py:108` confirms `deep_sim_total: int = 50`.

## 7. P2 [D3] — `watchlists_collections_screen.py` has no size-ratchet row and is absent from the decomposition doc
- Verdict: CONFIRMED
- Proof: `grep -n '"tldw_chatbook/' Tests/Architecture/test_screen_size_ratchet.py` returns exactly two rows —
  `chat_screen.py` (25363/762) and `library_screen.py` (35855/1330) — no `watchlists_collections_screen.py` row in
  either `test_screen_size_ratchet.py` or `test_module_size_ratchet.py`.
  `grep -n "watchlists" backlog/docs/size-decomposition-candidates-2026-09-18.md` returns no candidate-row hit
  (only an unrelated line-range mention inside another module's cluster description).

## 8. P3 [D3] — `change_review_screen.py` imports `pathlib`/`git_workspace` repeatedly from function bodies
- Verdict: CONFIRMED
- Site now: `pathlib` — 13 `from pathlib import Path[ as _P]` lines, all inside function bodies (`:437, 686, 717,
  755, 774, 814, 856, 882, 1139, 2548, 2764, 3363, 4804`) — exact count match with the review. `git_workspace` —
  17 `from tldw_chatbook.Workspaces.git_workspace import ...` lines (1 inside `if TYPE_CHECKING:` at `:29`, 16
  inside function bodies) vs the review's "20 times."
- Proof: `grep -n "^\s*from.*git_workspace import"` → 17 hits, not 20. `Workspaces/git_workspace.py:24-38` imports
  only stdlib/loguru/`Utils.log_sanitizer`/`Workspaces.change_tracking` — confirms no import cycle, matching the
  review's "no circular-import reason" claim.
- Note: the review's git_workspace count (20) overstates by 3 against a literal `grep` of import statements today;
  the qualitative claim (redundant per-function imports, no cycle) is unaffected. Kept as CONFIRMED, not WRONG,
  since the headline defect and its fix are unchanged by the exact count.

## 9. P3 [D1] — 9 of 13 `except Exception: pass` blocks in watchlists wrap multi-statement bodies
- Verdict: CONFIRMED
- Site now: AST-located at lines `4836, 5527, 5589, 5707, 6429, 7243, 7249, 7255, 7926, 7986, 12942, 13008, 13751`
  (was `4832, 5524, 5584, 5704, 7915, 7979, 12939, 13005, 13748` plus 4 more the review didn't line-list) — total
  count 13, matching the review exactly.
- Proof: an AST walk (`ExceptHandler` with `type.id == "Exception"` and a single bare `pass` body, then counting
  its sibling `Try.body` statements) finds exactly 9 handlers guarding a >1-statement try body (`4836`:3,
  `5527`:2, `5589`:3, `5707`:2, `7926`:5, `7986`:3, `12942`:2, `13008`:2, `13751`:2) and 4 guarding a single
  statement (`6429`, `7243`, `7249`, `7255`) — the exact 9/4 split the review states, and `7926`'s 5-statement
  body matches the review's cited example (formerly `:7915`). `grep -c "except NoMatches"` → 61, matching the
  review's cited count exactly.

## 10. P3 [D4] — Three "is this process alive?" copies disagree in failure direction
- Verdict: CONFIRMED
- Site now: `Event_Handlers/LLM_Management_Events/server_lifecycle.py:69-77` (`except Exception: return True`),
  `UI/Lab_Modules/lab_server_status.py:61-76` (`return False`), `UI/Screens/llm_screen.py:2931-2938`
  (`_vllm_process_liveness_proven`, `return False`) — same lines as filed.
- Proof: read all three bodies directly; `reserve_server_launch` (`server_lifecycle.py:80-96`) does
  `if process_is_running(process): return None` — confirms the "already running" refusal-on-exception path.

## 11. P3 [D4] — `_safe_text`/`_safe_skill_text` byte-identical bodies drift on truncation cap (500 vs 1000)
- Verdict: CONFIRMED
- Site now: `watchlists_collections_screen.py:1953-1959` (`max_length: int = 500`),
  `skills_screen.py:433-439` (`max_length: int = 1000`) — same lines as filed (off by ~1).
- Proof: both bodies read `sanitize_string(...).strip()` → `validate_text_input(..., allow_html=False)` →
  `fallback`, byte-for-byte identical except the default `max_length`.

## 12. P3 [D3] — `UI/Screens/settings_speech_tts.py` (2,408 lines) defines no `Screen` at all
- Verdict: CONFIRMED
- Site now: whole file, 2,408 lines (exact match to the review's count).
- Proof: `wc -l` → 2408; `grep -n "class.*Screen\|class.*Widget"` → zero matches.

TOTALS: confirmed=12 fixed=0 wrong=0 demoted=0 promoted=0
