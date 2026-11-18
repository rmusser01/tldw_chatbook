# CHAT-controller — tldw_chatbook/Chat/console_chat_controller.py, 29,048 lines

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| tldw_chatbook/Chat/console_chat_controller.py | 29,048 | **read in full**, sequentially in ≤500-line `sed -n` chunks (1–29048; chunks 2000–2700, 4200–4700, 13400–13999 were read from persisted tool output). No range skipped. |
| tldw_chatbook/DB/base_db.py | 38–330 | read (operation_owned_connection, SQLiteConnectionQuiescenceRegistry) — evidence for findings 1/2 |
| tldw_chatbook/DB/ChaChaNotes_DB.py | get_connection, _get_thread_connection, execute_query, quiesce_connections, close_connection, get_message_by_id_without_blob | read (evidence) |
| tldw_chatbook/Chat/console_chat_store.py | messages_for_session, _materialize_stream_buffer, _persist_pending_message_if_ready, get_message_version | read (evidence) |
| tldw_chatbook/Chat/console_capture_policy_repository.py | grep only (`replace`/`replace_privacy` use `self.db.transaction(immediate=True)`) | mechanical |
| UI/Console_Modules/send_price.py 150–200, console_spend_projection.py, wiring.py 2270–2320, session.py 3318–3430 (grep), Chat/console_turn_context.py 38–100, console_interrupt_rounds.py (grep) | callers of controller seams | sampled (to settle tick/keystroke exposure) |
| Tests/ | grep + one file excerpt (test_permission_summary_wiring.py 40–75) | EVIDENCE only |

Environment: worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` at `d8fb4053f9`. `ruff check --select E9,F63,F7,F82 tldw_chatbook/Chat/console_chat_controller.py` → `All checks passed!` (exit 0). App never run; no full-suite run.

### Responsibility clusters (god-module map; line ranges are from the method index, `grep -nE '^    (async )?def '` → 470 methods in `ConsoleChatController` + 59 module-level defs + 32 classes)
| cluster | lines | notes |
|---|---|---|
| Module-level: provider-selection builders, settings-field support tables | 590–760 | `build_console_provider_selection_from_settings`, `_supported_console_settings_fields` |
| Module-level: project-instruction binding authority (resolve/validate/commit) | 827–1625 | 20 free functions; pure + registry reads |
| Module-level: turn-context captures (prompt transforms, skills, MCP maxima) | 1166–1290 | sync sqlite/file reads; called per turn snapshot |
| Module-level: tool review hooks (MCP/builtin/local/skill-promotion/virtual-cli/raw-shell/combined) | 1626–3040 | 7 hook builders, `ApprovalDecisions` |
| Module-level: dataclasses, decorators (`_maintenance_boundary`, generation handoff, profile lease) | 3140–3677 | |
| Controller `__init__` (831 lines) | 3758–4588 | 60+ attributes, 5 interrupt registries aliased to the host |
| Capture policy / trace privacy / purge | 4589–5390 | includes finding 1 (raw thread at 5044) |
| Run-state facades, lifecycle revision, maintenance pause/drain | 5390–5690 | |
| Prompt queue mutations | 5687–5960 | |
| Pending-round accounting, fleet markers/counts, send refusal | 5977–6536 | |
| Provider-continuation recovery | 6537–6790 | |
| Library preparation state machine (prepare/retry/bypass/cancel/continue) | 6840–8138 | |
| Voice: speculative attempt preparation | 8139–8542 | includes raw `connection.execute` at 8438 |
| Submit lifecycle (`submit_draft` → `_submit_draft_body`, 1,575 lines) | 8543–10375 | |
| Durable turn commit / postcommit effects / dispatch recovery / discard | 10375–12447 | |
| Session lifecycle (new/switch/close/settings rebase) | 12448–13381 | `rebase_console_settings_draft` alone 12648–12885 |
| Stop/cancel signalling + interrupt bridges (approval, skill-install, skill-script, chat-create, ask-user, worktree-merge) | 13382–17874 | chat-create is the one kind outside `InterruptRoundHost` (finding 8) |
| Steer/redirect/abandon/stop/shutdown/visit lifecycle | 17875–18306 | |
| Retry / continue / regenerate / edit-and-resend | 18325–19807 | |
| Summaries: manual /rewind, notes, impersonate | 18793–19565 | |
| Context snapshot preview + personal-context + project-instruction preview | 19808–20806 | two ~200-line provider-composition copies (see verified-fine note) |
| Redaction / turn configuration snapshot / library authority | 20807–21543 | |
| Payload transforms: skills, world-info, chat dictionaries, RAG evidence | 21544–22574 | |
| Run hooks, prompt history, failure rows | 22575–22742 | |
| Branch memory fences / manual+automatic admission / compaction preflight | 22743–24411 | `_apply_conversation_memory_preflight` 520 lines |
| Stream dispatch: `_stream_assistant_response(_inner)`, usage/fleet-usage fold, direct-provider stream, citation repair | 24412–26258 | |
| Agent reply (`_run_agent_reply`, 963 lines) + finalizers | 26259–27755 | |
| Presentation / system prompt / emote / provider-message payload builders | 27756–28491 | |
| Run-state writer (`_set_run_state`), outcome publication, rejection helpers | 28492–29048 | |

## Findings   (ordered P0→P3, then D1→D4; one ### each)

### P2 [D1] — `replace_conversation_capture_detail` leaks one registered ChaChaNotes connection per call on a file-backed DB
- Where: `tldw_chatbook/Chat/console_chat_controller.py:5028-5050` (`run_repository_call` spawned as a bare `threading.Thread(name="console-capture-policy-write")`); contrast `:10448-10456` `_run_owned_chat_db_operation` / `:10458-10480` `_run_durable_db_call`, which wrap the same class of write in `operation_owned_connection`.
- Evidence: repository-level probe (scratchpad, run under `env.sh` with cwd=worktree; `ConsoleCapturePolicyRepository(db).replace(conv, CaptureDetail.SAFE)` on a real conversation in a tmp-file `CharactersRAGDB`):
  ```
  PROBE baseline registered handles: 1
  PROBE after 3 raw threads (controller.py:5044 shape): 4
  PROBE after 3 owned threads (_run_owned_chat_db_operation shape): 4
  ```
  Same result with a bare `db.get_connection().execute("SELECT 1")` per thread (1 → 4 → 4). Registry is a strong `dict[int, sqlite3.Connection]` (`DB/base_db.py:112`), so the handle and its file descriptor survive until a backup `quiesce_connections()` → `close_registered()` (`base_db.py:253-272`) closes strays. `console_capture_policy_repository.py:114` confirms `replace` opens a thread-local handle via `self.db.transaction(immediate=True)`.
- Why it matters: every conversation-level Capture Full/Safe change by the user leaks a live sqlite connection (fd + WAL reader) for the lifetime of the process (or until the next backup pass); the codebase pins the opposite invariant elsewhere (`Tests/UI/test_console_send_refresh_scope.py:129`, `test_console_send_diagnostic_flow.py:185`, `test_console_composer_blink_wrap.py:198` assert `registered_connection_count() == 0` after a send).
- Recommended correction: wrap `run_repository_call`'s body in `operation_owned_connection(getattr(self.store.persistence, "db", None))` (the helper `_run_owned_chat_db_operation` already exists 10448); keep the hand-rolled Event/`call_soon_threadsafe` handoff if the cancel-survival semantics are wanted, or replace the thread with `asyncio.to_thread(self._run_owned_chat_db_operation, repository.replace, ...)` + the existing shield loop.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none — `Tests/Chat/test_console_chat_controller_exchanges.py::test_conversation_policy_reservation_blocks_race_and_reconciles_cancel` (and siblings at 989–1347) drive this method against a file-backed `CharactersRAGDB` (line 749) but never assert the handle count.
- Already covered: none (task-31502 is the registry's shared lock, not stray handles).

### P2 [D1] — `_permission_summary_worker` reads (and can persist) the owner-thread-only `ConsoleChatStore` from a raw worker thread
- Where: `:14045-14085` (`threading.Thread(target=self._permission_summary_worker …).start()` → `_summary_tail_messages` `:14087-14106` → `self._provider_messages_for_session(session_id)` `:28061` → `self.store.messages_for_session` → `console_chat_store.py` `_materialize_stream_buffer` → `_persist_pending_message_if_ready` → `_persist_new_message`).
- Evidence: `grep -nE 'single-threaded|owns the store' console_chat_controller.py` → `4389: store mutation always runs on the thread that owns the store.`, `25107: The store is single-threaded, so the actual fold …` (the fleet-drain consumer hops onto the loop via `call_soon_threadsafe` for exactly this reason, `:25102-25129`). `console_chat_store.py::_materialize_stream_buffer` folds buffered chunks and calls `_persist_pending_message_if_ready` (→ `_persist_new_message`, a persistence write) — i.e. `messages_for_session` is not a pure read. The only tests of this lane stub the thread out: `Tests/Chat/test_permission_summary_wiring.py:43-59` (`monkeypatch.setattr(ccc.threading, "Thread", _ThreadStub)`), so the real cross-thread path is never exercised.
- Why it matters: with `[permission_summary]` active (ADR-090), every approval round folds/persists the session's streaming rows from a non-owner thread concurrently with the loop's own `messages_for_session`/`append_stream_chunk` calls — torn snapshots or `RuntimeError: list changed size` are swallowed by the `except Exception: return []` at 14103 (summary silently missing), and a persist on that thread also leaks a registered handle by the finding-1 mechanism (raw thread, no `operation_owned_connection`).
- Recommended correction: build `tail` on the UI thread inside `_maybe_fire_permission_summary` (it already runs there for the head-mount path) and pass the ready-made list into the worker; the worker should only do the network summarizer call. S.
- Size: S · ADR: no (ADR-090 covers the feature, not thread ownership) · Confidence: verified (path and ownership contract); whether `_persist_pending_message_if_ready` actually fires in a given round depends on an unflushed buffer being present — see UNVERIFIED.
- Pinning test: none (the Thread stub above means no test can go red on this).
- Already covered: none.

### P2 [D2] — `_durable_context_snapshots` performs one synchronous sqlite point read per active-path message on the event loop, on every dispatch
- Where: `:22743-22862` (`version = version_reader(persisted_id)` inside the per-message loop, `:22775`); `version_reader` = `Chat/chat_persistence_service.py:923 get_message_version` → `self.db.get_message_by_id_without_blob(message_id)` (one `SELECT … FROM messages WHERE id=?` each). Callers: `:23940` `_apply_conversation_memory_preflight` (every `_stream_assistant_response_inner` dispatch with a `ConsoleProviderResolution`, `:24545`), `:23577` `context_control_inputs` (settings UI, `prepare_speculative_voice_attempt` `:8409`, `compact_context_now` twice), `:23540` `_project_session_effective_memory` (context preview), `:23601`/`:23850`, `:23649` `undo_context_memory_reset` (loops **all** sessions).
- Evidence: `grep -n '_durable_context_snapshots(' console_chat_controller.py` → 8 call sites listed above; `get_message_version` body quoted from `chat_persistence_service.py:923-942`. Cost not measured (read only).
- Why it matters: TASK-22205 offloaded the two per-send `BEGIN IMMEDIATE` transactions because "tens of ms steady-state, up to the 15 s busy timeout under write-lock contention" — this loop re-exposes the loop to N such reads per send (N = transcript length), each subject to the same busy-timeout stall (e.g. during the messages_fts backfill window named in that task).
- Recommended correction: add a batched `get_message_versions(ids) -> dict[str,int]` to `ChatPersistenceService` (one `SELECT id, version, deleted FROM messages WHERE id IN (…)`) and call it once per snapshot; or run the existing loop through `self._run_durable_db_call` (already handles the `:memory:` case). M.
- Size: M · ADR: no · Confidence: inferred (magnitude) — literal command to settle: `cd <worktree> && source <SCRATCH>/env.sh && $PY - <<'EOF'` building a file-backed store with ~200 persisted messages and timing `controller._durable_context_snapshots(session_id)` with `time.perf_counter()` (Tests/Chat/test_console_rewind_summarize.py's fixtures build such a controller).
- Pinning test: `Tests/Chat/test_console_rewind_summarize.py` / `test_console_context_compaction.py` exercise the snapshots (behaviour), none pins the read count.
- Already covered: none.

### P2 [D3] — God module: 29,048 lines, 470 methods on one class, 831-line `__init__`, 1,575-line `_submit_draft_body`, 963-line `_run_agent_reply`, and no size governance
- Where: whole file; cluster map in Coverage above. Longest bodies: `_submit_draft_body` 8800–10375, `__init__` 3758–4588, `_run_agent_reply` 26259–27221, `resume_durable_postcommit` 11206–11883, `_apply_conversation_memory_preflight` 23892–24411.
- Evidence: `grep -cE '^    (async )?def '` → 470; `grep -cE '^class '` → 32; `grep -cE 'except Exception'` → 152; `ls Tests/Architecture | grep -i ratchet` → `test_library_modules_size_ratchet.py`, `test_screen_size_ratchet.py`; `grep -ln 'console_chat_controller\|Chat/' Tests/Architecture/*ratchet*.py` → (none). `grep -rliE 'console_chat_controller' backlog/tasks | xargs grep -liE 'split|decompos|size.?ratchet|god.?module'` → (none).
- Why it matters: 60 synchronous `self.store.<persisting call>` sites (`grep -cE` of append/persist/mark/create_sibling/record_trace_event/set_message_usage) and 25 `asyncio.to_thread` sites live in one file with no per-subsystem field ownership; every review (this one included) has to re-derive which of the ~70 `__init__` attributes each cluster owns.
- Recommended correction: out of scope to redesign here — apply `backlog/docs/library-decomposition-recipe.md` §1 (per-subsystem PR series), §2 (field-ownership script, ≥2-subsystems rule), §3 (monkeypatch-name routing — this file is patched by name in dozens of tests), §17 (add a `_BUDGETS` row for `Chat/console_chat_controller.py` in a new sibling ratchet, since neither existing ratchet covers `Chat/`). Candidate first extractions with the fewest shared fields: the module-level review-hook builders (1626–3040, no `self`), project-instruction authority (827–1625, no `self`), the settings rebase (12648–12885, pure), and the interrupt bridges (13382–17874, already mostly delegated to `InterruptRoundHost`).
- Size: L · ADR: yes (new — a decomposition ADR in the shape of the Library series) · Confidence: verified
- Pinning test: none (no ratchet).
- Already covered: none (task-1378/task-31202 govern `settings_screen.py`, not this file).

### P2 [D4b] — `_manual_json_digest` is a verbatim third copy of a storage-bound digest
- Where: `:22946-22954` (`ConsoleChatController._manual_json_digest`), `Chat/console_context_repository.py:1876-1883` (`_digest_json`), `Chat/console_context_compaction.py:2657-2664` (`_digest_json`). Controller callers: `:22957` `_repository_prefix_digest`, `:23091` `content_digest=` on `PersistedLineageFenceRow`, `:23143` `legacy_summary_digest=` on `MemorySelectionFence`.
- Evidence: `sed -n` of all three bodies — byte-identical (`json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")` → `sha256().hexdigest()`). `grep -rnE 'sort_keys=True' tldw_chatbook/Utils` → no shared helper exists. Repository writers use its own copy at `console_context_repository.py:1455,1590,1686`; the controller's copy produces the `expected_*` fences the repository compares against (`append_current_branch_reset_if_current`, `BranchMemoryCommit.durable_lineage`).
- Why it matters: the controller-side and repository-side digests must agree byte-for-byte or every branch-memory admission fails "conversation changed" — the drift would reach storage comparisons silently (no test compares the two implementations).
- Recommended correction: export one `digest_json` from `Chat/console_context_repository.py` (the storage owner) and import it in the controller and `console_context_compaction.py`; delete the two private copies.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_console_rewind_summarize.py` exercises the fences end-to-end (would go red on drift, not on duplication).
- Already covered: none.

### P3 [D4b] — `get_internal_prompt` lazy wrapper duplicated in two modules that already import each other
- Where: `:526-550` and `Agents/agent_service.py:213-236` (identical body: lazy `from tldw_chatbook.Internal_Prompts import get_internal_prompt as _resolve`).
- Evidence: `sed -n` both; identical. `:147` already does `from tldw_chatbook.Agents.agent_service import append_personal_context` at module scope.
- Why it matters: two boot-ratchet wrappers to keep in sync (ADR-097 census reasoning is in both docstrings).
- Recommended correction: `from tldw_chatbook.Agents.agent_service import get_internal_prompt` in the controller (same name, so module-namespace patches in tests keep working).
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Packaging/test_rag_boot_import_closure.py` (per both docstrings) pins the boot census, not the duplication.
- Already covered: none.

### P3 [D4b] — `_empty_profile_context_snapshot` duplicated between controller and `console_chat_models`
- Where: `:3537-3542` and `Chat/console_chat_models.py:1395-1400` (identical lazy `ProfileContextSnapshot.empty()`); controller uses it at `:19974, 20130, 20291, 20297, 20422, 20467`.
- Evidence: `sed -n` both bodies; identical. The controller imports ~40 names from `console_chat_models` at `:60-101`.
- Recommended correction: import it from `console_chat_models` (make it public or keep the private-name import with a comment; the models module is the owner of `ConsoleContextSnapshot`'s default_factory).
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D4b] — chat-create confirm is the one interrupt kind hand-rolled outside `InterruptRoundHost`, and it has drifted from the five host kinds
- Where: `:16907-17106` `request_chat_create_confirm` (own registry `_pending_chat_create_rounds` + `_pending_chat_create_lock` `:4554-4560`, own `event.wait` poll loop `:17040-17049`), `:16297-16341` `_revoke_chat_create_rounds` ("Mirrors `_revoke_skill_script_rounds` exactly"), `:17137-17175` `resolve_pending_chat_create`, `:17107-17126` `_remount_parked_chat_create`.
- Evidence (drift): deadline arming — chat-create `:17010` `deadline = time.monotonic() + timeout_seconds if timeout_seconds > 0 else None` (hard clock even when parked in a background session) vs approval/skill-install/skill-script `:14402-14406`, `:16412-16416`, `:16610-16614` `… if session_id is None and timeout_seconds > 0` (session-bound rounds get the host's pausable "answerable" clock, `:14141-14345`); `resolve_pending_chat_create` `:17170-17175` sets decision+event with no `settled` guard, whereas `resolve_pending_skill_script` `:16745-16751` refuses a second resolve; chat-create payloads are excluded from `_pending_decision_payloads_locked` `:14205-14219` (documented live-UAT workaround at `:14413-14418`).
- Why it matters: with a non-default positive `chat_create_confirm_timeout_seconds`, a card parked behind another session times out unseen; the code comments themselves flag the special-casing (`:14413 "rides its OWN standalone registry (pre-host design)"`). Default timeout is 0, so no user-visible defect today.
- Recommended correction: register `"chat_create"` as a sixth kind in `Chat/console_interrupt_rounds.py` (`KIND_SETTER_ATTRS`, revocation stamp) and route `request_chat_create_confirm` through `run_round`, deleting the private lock/registry/poll/revoke copies.
- Size: M · ADR: yes (`150-agent-chat-fork-and-spawn.md` owns the feature; the host exemption is not decided there — cite it, do not redesign) · Confidence: verified (reading) · Pinning test: `Tests/Chat/test_console_chat_create_confirm.py`, `Tests/Chat/test_chat_create_confirm_card.py` (behaviour; none asserts the deadline semantics) · Already covered: none.

### P3 [D3] — raw `connection.execute` SELECT outside `db.transaction()`/`execute_query` in `prepare_speculative_voice_attempt`
- Where: `:8418-8460` (`database.get_connection()` then `connection.execute("SELECT revision_id … FROM console_trace_semantic_revisions WHERE live_message_id = ?")`); the sibling read at `:10661` correctly sits inside `with database.transaction(immediate=True) as cursor:`.
- Evidence: `grep -nE '\.execute\('` → exactly these two sites; index present (`DB/recovery_core_schema.py:171` `uq_console_trace_semantic_revisions_live_message` partial unique) so the read is a point lookup; `ChaChaNotes_DB.execute_query` (`:3989-4080`) adds the lazy SQL debug log + `chachanotes_db_query_*` metrics the raw path skips.
- Why it matters: consistency only — the read bypasses the DB layer's metrics/logging and the `db.transaction()` pattern CLAUDE.md prescribes; cost is fine (indexed, ≤ one row per saved voice message).
- Recommended correction: `with database.transaction() as cursor: row = cursor.execute(...)` (mirrors 10661) — S.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D1] — silent `except Exception: pass` (no log) on four data-adjacent paths
- Where: `:22529-22536` `_notify_submission_accepted` (composer-clear hook), `:22546-22549` `_notify_queued_submission_accepted`, `:24356-24364` hybrid visual-compaction fallback (the follow-up `logger.info("…fell_back_to_text")` at 24366 drops the exception type), `:1174-1194` `capture_prompt_transform_inputs` (swallows ImportError **and** DB errors → chat dictionaries / world-info silently absent from the turn, no log at all).
- Evidence: `sed -n` of each; none logs. The remaining excerpt rows (2540, 2715, 14126, 15023, 17446, 18134) carry a `noqa: BLE001` rationale and are teardown/advisory (see Verified-fine).
- Why it matters: a broken dictionary/world-book read or a failing composer hook is indistinguishable from "nothing configured"; the 1174/1182 case masks a real import regression of a mandatory module (`Character_Chat.Chat_Dictionary_Lib`, `world_info_resolver`).
- Recommended correction: `logger.debug("…", exception_type=type(exc).__name__)` in each; narrow 1174/1182 to `except (sqlite3.Error, KeyError, ValueError)` so an ImportError surfaces.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D4b] — documented deliberate duplicate `_split_skill_command_word`
- Where: `:3041-3058`; copies `Chat/console_command_grammar.py:151-166` `_split_leading_token`, `UI/Console_Modules/skill.py:111-116` `_split_console_skill_name_args`. Bodies identical (single-whitespace split).
- Evidence: `sed -n` all three; the controller docstring states it is "a deliberate small duplicate rather than an import" because the grammar helper is module-private by design. Reported as *documented* duplication, no drift; leave unless the grammar module exports it.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape `clear_governed_state@3165` vs `_wire_llamacpp_snapshot_service@app.py`, `_cancel_countdown`, `invalidate_refresh_scope` | retired — 3-line "set two attrs to None" shape match; unrelated semantics, nothing to share |
| dup_shape/dup_verbatim `_split_skill_command_word@3041` (3 copies) | confirmed → P3 documented duplication (finding above) |
| dup_shape/dup_verbatim `_manual_json_digest@22946` (3 copies) | confirmed → P2 [D4b] (storage-bound digest) |
| dup_verbatim `get_internal_prompt@526` vs `agent_service.py:213` | confirmed → P3 [D4b] |
| dup_verbatim `_empty_profile_context_snapshot@3537` vs `console_chat_models.py:1395` | confirmed → P3 [D4b] |
| except_exception_pass 2540, 2715 | retired — best-effort `clear_lesson_approvals` cleanup inside an already-failing/raising path, `noqa: BLE001` rationale present |
| except_exception_pass 14126 | retired — advisory summary patch (`update_pending_approval_summary`), `noqa` rationale |
| except_exception_pass 15023, 17446 | retired — injected timeout seam fails open to the documented default, `noqa` rationale |
| except_exception_pass 18134 | retired — `shutdown()` awaiting cancelled tasks; the task's own failure path already logged; teardown must not raise |
| except_exception_pass 22533, 22548, 24364 | confirmed → P3 [D1] silent swallow (no log) |
| except_exception_return_per_file (42) | examined while reading; all carry `noqa: BLE001` + rationale or return a typed refusal; the un-logged ones are folded into the P3 [D1] finding (1174/1182); no further action |
| function_body_import_per_file (72; ast found 108 import names) | **confirmed resolvable, none missing** — ast script + `importlib.import_module`+`hasattr` under `env.sh`: 104/108 ok on first pass; the 4 flagged (`Agents.activation`, `Chat.console_worktree_recovery`, `Backup_Recovery.config_participants`, `Character_Chat.Chat_Dictionary_Lib`) were `find_spec` false positives — `ls` shows all four `.py` files exist and a direct `import_module` + `hasattr(name)` printed `OK` for each (`worker_guard`, `capture_intent`, `operation`, `collect_active_chatdict_entries`) |
| legacy_markers_per_file (44) | retired — "legacy"/"pre-Task-9"/"deprecated shim" are docstring history notes; the one real shim (`set_run_pending_approval` 6133) is documented and has one live caller |
| lock_and_execute (locks=7, executes=2) | retired as a thread-safety concern — none of the 7 locks (3490 handoff, 4043 watchlists, 4049 lifecycle revision, 4133 workspace roots, 4140/4141 RLocks for submit tasks / capture quiescence, 4554 chat-create) wraps any SQL; both `.execute(` sites are unlocked — 10661 inside `transaction(immediate=True)` (fine), 8438 raw (→ P3 [D3]) |
| strftime 19241 `%Y-%m-%d %H:%M UTC` | retired — Python `datetime.strftime` for a note provenance header (`_note_provenance_header`); minute precision is the intended display, not SQLite |
| try_import_guard 1174, 1182 (`capture_prompt_transform_inputs`) | confirmed → folded into P3 [D1] (silent swallow of ImportError/DB errors) |
| try_import_guard 1253 (`capture_mcp_definition_maximum`) | retired — `except Exception: return {}` documented "uncertainty freezes an empty maximum" (fail-closed for a security maximum; correct direction) |
| try_import_guard 9070 (`_submit_draft_body` references) | retired — `logger.opt(exception=True).warning(...)` logs the failure |
| try_import_guard 9901 (`run_hooks.truncate_hook_text`) | retired — `except BaseException:` re-raises after cleanup |
| try_import_guard 14022 (`permission_summary_service`) | retired — advisory-only resolver, `resolution=None` → summary skipped |
| try_import_guard 17362 (`console_conversation_hydration`) | retired — `logger.opt(exception=True).warning(...)` logs; degrades to empty transcript by contract |

## Verified-fine
- **Locks vs sqlite**: none of the 7 `threading.Lock/RLock`s guards raw SQL; each protects an in-memory map, with lock order documented at 4118-4130 and 14192-14201; `_capture_quiescence_lock`/`_active_submit_tasks_lock` are RLocks because `_capture_purge_blocker` nests them (4715→4728). Evidence: `grep -nE 'with self\._(lock|…)\b'` → 36 sites, all around dict/set/tuple mutations.
- **`run_worker(coroutine)`**: the file contains no `run_worker(` call (the single hit at 8849 is a docstring). Off-loop work goes through `asyncio.to_thread` (25 sites) and `_run_durable_db_call`; the coroutine-side sync store calls are the store's documented owner-thread contract (see finding 3 for the one unbounded loop).
- **`_maybe_await`-style helper**: `_run_durable_postcommit_effect` (10375-10429) runs the callback inline and awaits only if awaitable — the two heavy effects (`transition_checkpoint`, `publish_owners`' commit) offload inside via `_run_durable_db_call`; the rest are in-memory store ops. Not a deferral bug.
- **Negative-predicate offload `_durable_db_call_offloadable` (10430-10446)**: `not getattr(db, "is_memory_db", False)` threads a fake with no `.db`; in production `store.persistence.db` is always a `CharactersRAGDB` exposing `is_memory_db` (`ChaChaNotes_DB.py:3344-3347`), and the `:memory:` branch (inline) is the correct one because a worker thread would see an empty per-connection DB. Documented in the docstring; no shipped path hits the "unknown shape" arm.
- **8438 raw SELECT cost**: partial unique index on `live_message_id` (`recovery_core_schema.py:171`) → point read; only the bypass of `execute_query`'s logging/metrics is worth a P3.
- **"sqlite on every keystroke/tick" hypothesis** (from `provider_messages_for_next_send_estimate` → `resolve_turn_configuration_snapshot` → `capture_prompt_transform_inputs` sqlite reads): **retired** — `UI/Console_Modules/send_price.py:150-157` docstring: "Expensive … Never call this on the keystroke path"; the 0.2 s `ConsoleDraftSpendRefresh` and the 10 s `CONSOLE_COST_TTL_TICK_SECONDS` timer (`chat_screen.py:808, 11942`) call `_sync_console_cost_chip` → `_build_console_cost_state`, which reads the in-memory store with a memoised tokenizer (`chat_screen.py:11726-11742`), not the turn snapshot. The snapshot's sqlite reads run per submit/queue/preview — acceptable.
- **`_permission_summary_worker`'s other reads**: `store.messages_for_session` is in-memory (`console_chat_store.py::messages_for_session`) — the problem is ownership + the materialize/persist side effect (finding 2), not sqlite reads per se.
- **Quiescence registry does not deadlock on stray handles**: `close_registered` (`base_db.py:253-272`) closes any registered handle not `in_transaction`, so finding 1's leak is bounded by backup cadence, not fatal to backups.
- **Boot-ratchet lazy imports** (`Chat.console_interrupt_rounds`, `console_auxiliary_routing`, providers at 15517/15686/15776, `console_visual_transcript`): each is annotated with the ADR-097/task id and the module resolves (script above) — deliberate, not "guarded-but-not-lazy".
- **`_replace_image_data_with_placeholders` inline `re.sub(r"data:[^\s\"'<>]+")` (20852)**: preview-only path; the `re` module's 512-entry compile cache makes the literal pattern free after first use. The hot regex (`_SECRET_REDACTION_PATTERN`) is class-level compiled (20918-20930).
- **Two ~200-line provider-composition copies** (`_build_personal_context_snapshot` 20274-20467 and `_build_project_instruction_preview_for_session` 20469-20730 vs the live `_run_agent_reply` 26611-26920): same 5-provider compose sequence + `bound_messages_to_window` + system-prompt split. Preview paths must mirror live composition by design (docstring 20643-20648); refactor belongs to the decomposition finding, not a separate D4 (no drift found: both previews call `_compose_agent_request_providers` with `publish_mcp_counts=False`).
- **Lint / loggers / workers**: `ruff --select E9,F63,F7,F82` clean; no stdlib `logging` import (loguru only); no `exclusive=True` workers in the file.
- **Function-body imports**: all 108 resolve (see dispositions).
- **`_note_provenance_header` strftime**: display timestamp, intentional.
- **Cross-package private imports** (`Library.library_rag_service._outcome_from_service_result` at 457, `Character_Chat.world_info_resolver._collect_active_world_books` at 1183, `local._load_index()`/`_summary_for_record` at 1230-1233 with `noqa: SLF001`): real but pre-existing and annotated; folded into the decomposition finding rather than listed separately (a private-name export contract is what the recipe §3 governs).

## Retired
- **4 "missing module" function-body imports** (`Agents.activation`, `Chat.console_worktree_recovery`, `Backup_Recovery.config_participants`, `Character_Chat.Chat_Dictionary_Lib`): `find_spec` returned None inside my ast script, but `ls` shows all four files (`Agents/activation.py` 8718 B, `Chat/console_worktree_recovery.py` 16173 B, `Backup_Recovery/config_participants.py` 17324 B) and a direct `importlib.import_module` + `hasattr` printed `OK` for every target name. Symptom real (script output), cause was my probe (`find_spec` before parent-package import), retired with evidence.
- **"Locks guard sqlite outside `db.transaction()`"** (excerpt row): every `with self._*lock` body is in-memory; retired with the 36-site grep.
- **"Per-keystroke sqlite via the send-price projection"**: retired — see Verified-fine (send_price.py docstring + cost-chip build path).
- **`shutdown()` except-pass (18134)**: teardown path; retired.
- **`dup_shape clear_governed_state`**: trivial shape; retired.
- **`legacy_markers` 44**: docstring history; retired.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| Finding 2's persist branch (`_persist_pending_message_if_ready`) actually fires during a real approval round (needs an unflushed stream buffer on a pending row at the moment the summary thread starts) | requires a live run with `[permission_summary]` active; app not run per brief | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify` launch with `[permission_summary] mode="always"` and an MCP "ask" tool, send a turn that triggers an approval card, then `capture-pane` the Logs (F3) for `exchange_attach_failed`/`RuntimeError` lines and check `db.registered_connection_count()` via the diagnostics inventory |
| Magnitude of finding 3 (N point reads per send) | not measured | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY -m pytest Tests/Chat/test_console_rewind_summarize.py -q` to confirm the fixture, then time `controller._durable_context_snapshots(session_id)` with `time.perf_counter()` on a 200-message file-backed session |
| Whether a backup quiescence pass actually reclaims the finding-1 handles in production (only read `close_registered`) | not exercised | `$PY -c` building the probe DB, leaking 3 handles, then `with db.quiesce_connections(timeout_seconds=5): pass` and printing `db.registered_connection_count()` |
