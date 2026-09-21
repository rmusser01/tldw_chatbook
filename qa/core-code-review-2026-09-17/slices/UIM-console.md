# UIM-console — tldw_chatbook/UI/Console_Modules/ (42 files), 47,714 lines

Worktree: /Users/macbook-dev/Documents/GitHub/tldw-review @ d8fb4053f9 (read-only). Run 1 was killed by a usage limit mid-run; this file is the completed report (run-1 reads are marked in the coverage table).
`<SCRATCH>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-chatbook/5cdb48ca-db0f-47a4-930a-9ef5b33bceed/scratchpad`. All probes: `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$WT $PY <SCRATCH>/repro/<file>.py` (each printed `module: .../tldw-review/tldw_chatbook/__init__.py`, i.e. the worktree copy).

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| workspace.py | 7483 | read in full (run 1); re-read 5380-5450 this run (raw SQL) |
| session.py | 5340 | read in full (run 1); re-read 3528-3550, 3680-3695, 4400-4440 |
| message.py | 3012 | read in full (run 1); re-read 180-215, 2080-2110, 597-610 |
| left_rail.py | 2720 | read in full (run 1) |
| realtime.py | 2337 | read in full (run 1); re-read 600-640, 700-780 |
| wiring.py | 2310 | read in full (run 1); re-read 184-230, 800-812, 1220-1245, 1530-1545 |
| agent.py | 2296 | read in full (run 1) |
| hands_free.py | 2244 | read in full |
| dictation.py | 2231 | read in full |
| prompts.py | 1949 | read in full |
| right_rail.py | 1787 | read in full |
| video.py | 1508 | read in full |
| image.py | 1401 | read in full |
| character_context.py | 1337 | read in full |
| raw_cli.py | 900 | read in full (ADR-093/094 check) |
| terminal.py | 539 | read in full |
| library_activity.py | 274 | read in full |
| worktree.py | 70 | read in full |
| review_selection.py | 645 | sampled: 370-390, 450-480, 590-600 (inline-truncate, strftime, `_sessions` rows) |
| send_price.py | 336 | sampled: the three `except Exception` sites (142-146, 205-215, 220-228) |
| archive.py | 475 | sampled: 80-100, 390-410 (both `try_import_guard` rows) |
| retrieval.py | 721 | sampled: 528-555 (`try_import_guard` row) |
| prompt_queue.py, environment.py, provider_continuation_recovery.py, character.py, fleet.py, transcript.py, status_row.py, skill.py, dispatch_recovery.py, console_spend_projection.py, library_policy.py, capture_policy_bindings.py, frame.py, character_avatar_layout.py, reaction_preview.py, research_command.py, rail_section_layout.py, conversation_token_preparation.py, trace_call_recovery.py, __init__.py | ~5,600 total | mechanical only — covered by four whole-slice scans, each run over all 42 files: (1) an `ast` scan for `run_worker(exclusive=True)` without `group=`; (2) an `ast` scan for `query_one`/`query_exactly_one` outside a `try`; (3) greps for `re.compile`, `get_cli_setting`, `connection.execute`/`fetchall`, `store._`/`_sessions` private access, `logging.`; (4) the excerpt's own candidate rows. `provider_continuation_recovery.py` was additionally read at 200-300. |

## Findings (P0→P3, then D1→D4)

### P1 [D2] — a `workspace_records` SELECT runs on the UI thread 5×/s for the whole of every streaming run, for a title `ensure_session` throws away
- Where: `UI/Console_Modules/session.py:4419-4426` (`_sync_console_session_draft`); same shape at `session.py:3688` (`_replace_active_console_session_settings`). Chain: `UI/Screens/chat_screen.py:18797 set_interval(0.2, _poll_transcript)` → `:18746 await self._sync_native_console_chat_ui()` → `:18547 self._session._sync_console_session_draft()` → `self._workspace_initial_session_title(...)` → (`wiring.py:1535` lambda) `workspace.py:5248 _console_initial_session_title_for_workspace` → `workspace.py:5232 registry_service.get_workspace()` → `Workspaces/registry_service.py` sync `SELECT ... FROM workspace_records`.
- The value is discarded: `Chat/console_chat_store.py:2122-2124` — `ensure_session` returns `self._sessions[self.active_session_id]` untouched when a session is active; `title` is consumed only by `create_session`.
- Evidence: `<SCRATCH>/repro/uim_console_probe3.py` →
  ```
  'workspace-default': median=0.1us -> per-second at 5Hz = 0.00ms
  'global':            median=0.1us -> per-second at 5Hz = 0.00ms
  'w-1':               median=2296.2us -> per-second at 5Hz = 11.48ms
  ```
  `uim_console_probe2.py` on the same machine: `title-for-workspace(non-default): median=2.18ms p95=2.36ms max=2.54ms` over 200 calls against a fresh `WorkspaceDB`. Run-1 probe measured the underlying `LocalWorkspaceRegistryService.get_workspace` at median 3.63 ms / p95 5.47 ms and showed `_console_initial_session_title_for_workspace` issuing one `get_workspace` per call with no memo (10 calls → 10 queries).
- Scope: only when the active workspace is a real workspace — `workspace-default` and the `global` sentinel return early at `workspace.py:5252-5258` and cost nothing. `_poll_transcript` runs only while a run is live (`chat_screen.py:18720 _console_transcript_poll_needed`), i.e. exactly while the UI must stay responsive for streaming.
- Why it matters: ~11 ms of main-thread SQLite per second of streaming, entirely wasted, on the loop that paints the transcript.
- Recommended correction: the fix already exists three methods up — `session.py:3534-3546` carries the TASK-26839 comment and computes the title only `if creating_blank_session`. `_sync_console_session_draft` already computes `creating_blank_session` at `session.py:4418`; apply the same conditional there and at `session.py:3688`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found asserting the unconditional call.
- Already covered: none (task-26839 fixed the sibling call site only).

### P1 [D1] — the Console OS-video fallback is dead: every `_open_video_with_os` call raises `TypeError`, swallowed into a misleading notice
- Where: callers `UI/Console_Modules/video.py:1308` (`_play_console_video`, the fallback when ffmpeg/ffplay are absent) and `video.py:978` (after an external save); wiring `UI/Console_Modules/wiring.py:808 open_video_with_os=lambda path: screen._open_video_with_os(path)`; definition `UI/Screens/chat_screen.py:19648 def _open_video_with_os(path: Path) -> None:` — an instance method with no `self` and no `@staticmethod`.
- Evidence: `<SCRATCH>/repro/uim_console_probe4.py` →
  ```
  descriptor type: function signature: (path: pathlib.Path) -> None
  TypeError: ChatScreen._open_video_with_os() takes 1 positional argument but 2 were given
  ```
  (the probe performs the exact `wiring.py:808` call, `screen._open_video_with_os(path)`, on a real `ChatScreen`.)
- Why it matters: with ffmpeg/ffplay missing, `/generate-video` playback silently never works — the user gets "Could not open the video with the system player." (`video.py:1315`); after a successful external save the user is told "Video saved, but could not open it automatically" (`video.py:986`) on every save. Both `except Exception` blocks (`video.py:979`, `:1309`) hide a plain programming error.
- Recommended correction: add `@staticmethod` at `chat_screen.py:19647` (the body uses no instance state).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none.
- Already covered: none. Independently found from the definition side as UI-chat P1; this confirms it from the caller side, which is where the two swallowing handlers are.

### P1 [D1] — the realtime pre-connect credential gate is defeated by a placeholder key, and the placeholder is handed to the transport
- Where: `UI/Console_Modules/realtime.py:730` (`if not self._console_realtime_api_key():`) reading `realtime.py:616 get_api_key(...)`; the value is then put on the wire at `realtime.py:783-786` (`RealtimeSessionConfig(api_key=self._console_realtime_api_key(), ...)` → `_build_console_realtime_session` → `OpenAIRealtimeSession`).
- Evidence: run-1 probe (`uim_console_probe.py`): with `get_api_key` returning `"YOUR_KEY"`, `_start_console_realtime_connect` dispatched the `console-realtime-connect` worker — the gate passed. This run, `<SCRATCH>/repro/uim_console_probe5.py` →
  ```
  resolve_provider_api_key(placeholder) -> None
  api_key handed to the provider session: '<API_KEY_HERE>'
  ```
  i.e. the shipped placeholder is rejected by `config.py:1400 resolve_provider_api_key` but reaches `RealtimeSessionConfig.api_key` and the provider session constructor verbatim.
- Why it matters: the gate's own comment (`realtime.py:723-729`) states its purpose — "dispatching one anyway would spend the connect timeout to come back with whatever 401 text the provider chose, and the fallback toast would quote THAT instead of the one thing the user can act on". A placeholder in `[api_settings.openai] api_key` produces exactly the outcome the gate was written to prevent, plus a credential-shaped placeholder sent to a third party.
- Recommended correction: gate on `config.resolve_provider_api_key(...)` (or `is_valid_provider_api_key`, `config.py:1411`) instead of raw truthiness, in `_console_realtime_api_key` at `realtime.py:610-621`; the root cause (`get_api_key` not applying that check) is ENTRY-config P1.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none; root cause tracked as ENTRY-config P1.

### P2 [D1] — `terminal.py` swallows every runtime failure with no log anywhere: nine `except Exception: return False/None` in a file that imports no logger
- Where: `UI/Console_Modules/terminal.py` — `request_focus:298`, `request_retry_cleanup:317`, `send_key:328`, `send_paste:345`, `request_resize:366`, `_refresh:428`, `_session_view:481`, `_selected_session_id:500`, and the nested `detach_workspace:113/137/141`. Five of these (`open_workspace:98/110`, `request_arm:152/165`, `request_new_session:189/225/230`, `request_rename:262`, `request_close:286`) at least call `self._status(...)`; the nine above return silently.
- Evidence: `grep -c "logger\|logging" tldw_chatbook/UI/Console_Modules/terminal.py` → `0`. Every other controller in this slice logs at its boundaries (`dictation.py:1223`, `video.py:130`, `image.py:461`, `retrieval.py:547`, `hands_free.py:1269`, …).
- Why it matters: a keystroke or paste dropped by a manager-side error (including an `AttributeError` from a refactor) is indistinguishable from a working terminal that ignored you; `_refresh` returning early leaves the projection frozen. Nothing reaches the log, the status line, or persistent diagnostics, so there is no way to diagnose it after the fact.
- Recommended correction: bind `loguru` in this module and add a one-line `logger.opt(exception=True).debug(...)` in each of the nine handlers (narrowing the catches would be better still, but the manager boundary has no shared error base — the same reason `character_context.py:375` gives).
- Size: S · ADR: no · Confidence: verified (grep + full read)
- Pinning test: none.
- Already covered: none.

### P2 [D4] — `_apply_console_message_attachments` is a byte-identical copy of `ConsoleChatStore._set_message_attachments`, and `Chat/` imports the UI copy to get it
- Where: `UI/Console_Modules/message.py:186-210` vs `Chat/console_chat_store.py:10708-10729`; the inverted import is `Chat/console_conversation_hydration.py:137-139` (`from tldw_chatbook.UI.Console_Modules.message import _apply_console_message_attachments`, inside `_batch_fetch_resume_attachments`).
- Evidence: both bodies read (above); the store copy is decorated `@staticmethod` at `console_chat_store.py:10708`, so it needs no instance. The UI copy's own docstring justifies itself with "outside the store, where that helper isn't reachable" — that premise is false for a staticmethod.
- Why it matters: the helper enforces a storage invariant ("every attachments mutation sets the tuple AND the scalar image fields together"). Two copies of a storage invariant drift into two different on-screen/persisted attachment states; and a `Chat/` module reaching into `UI/` inverts the layering, so the business layer cannot be used (or tested) without the UI package.
- Recommended correction: delete `message.py:186-210`, call `ConsoleChatStore._set_message_attachments` from both call sites (canonical home: `Chat/console_chat_store.py`, where the invariant is documented), and drop the function-body import in `console_conversation_hydration.py`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none. Same defect reported from the store side as CHAT-store P2 — confirmed from this side, with the staticmethod fact that removes its stated justification.

### P2 [D4] — `_console_message_role_from_persisted` is duplicated verbatim across the Chat/UI boundary
- Where: `Chat/console_conversation_hydration.py:109-122` (module function) and `UI/Console_Modules/message.py:597-610` (staticmethod). Bodies read side by side: identical except the parameter annotation (`Mapping[str, Any]` vs `dict[str, Any]`).
- Why it matters: this maps persisted `role`/`sender` columns onto `ConsoleMessageRole`; drift between the two copies shows a resumed conversation with different message roles depending on which path hydrated it (user-visible, storage-derived).
- Recommended correction: keep the `Chat/console_conversation_hydration.py` copy as canonical and have `message.py` import it (that direction does not invert the layering).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none.

### P2 [D3] — a UI controller owns three hand-written schema queries over four ChaChaNotes tables
- Where: `UI/Console_Modules/workspace.py:5417-5442` (`_revalidate_character_conversation_target_sync`) — `SELECT local_authority_id FROM rag_identity_context`, a `conversations`⋈`character_cards` join, and `SELECT data_revision FROM character_conversation_search_revision`.
- Evidence: `grep -rln "connection.execute(\|cursor.execute(" tldw_chatbook/UI/` → exactly two files repo-wide (`UI/Console_Modules/workspace.py`, `UI/Persona_Modules/personas_conversations_controller.py`); 3 statements in this slice.
- Not a D1: the block is inside `db.transaction()`, parameterized, and reached only through `asyncio.to_thread` (`workspace.py:5395-5397`), so neither the thread-safety nor the loop-blocking rule is broken.
- Why it matters: table/column knowledge for four tables lives in a Console screen controller. A migration that renames `conversations.assistant_authority_id` or the revision singleton passes every check in `DB/` and silently breaks character-conversation activation here; `sql_validation.py`/`VALID_TABLES` governance does not see it either.
- Recommended correction: move the query into `DB/ChaChaNotes_DB.py` as one revalidation method returning the typed row the controller already consumes, and call it from the same `to_thread`.
- Size: M · ADR: no · Confidence: verified
- Pinning test: none found.
- Already covered: none.

### P3 [D4] — `_utf8_prefix` exists twice, verbatim
- Where: `Chat/console_raw_cli.py:84` and `UI/Console_Modules/raw_cli.py:74`.
- Evidence: run-1 probe `diff` of the two bodies → identical.
- Why it matters: it is the byte-budget truncation both the durable run-log record and the visible preview are clipped with; two copies is one edit away from a transcript preview and a persisted record disagreeing about where a command's output was cut.
- Recommended correction: keep `Chat/console_raw_cli.py`'s (the module the UI controller already imports six names from) and import it.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D4] — `_blank_console_session_settings` + `_console_new_chat_default_generation` copied between the session and workspace controllers
- Where: `UI/Console_Modules/session.py:3499-3513` and `UI/Console_Modules/workspace.py:1215-1229` — identical bodies (the session copy's return annotation is `ConsoleSessionSettings`, the workspace copy's is `Any`); both wrap the already-shared `blank_console_session_settings(app_config)`.
- Recommended correction: one mixin-free module function taking `app_instance`, in `Chat/console_session_settings.py` (where `blank_console_session_settings` already lives), called by both controllers.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D4] — `list_image_models_for_catalog` lazy-import shim copied into two screens
- Where: `UI/Console_Modules/image.py:55-60` and `UI/Screens/personas_screen.py:388-393` — identical three-line bodies around `Image_Generation.listing.list_image_models_for_catalog`, docstrings differing only in which caller they name.
- Recommended correction: leave the import lazy but put the one shim in `Image_Generation/listing.py`'s consumers' shared place, or simply do the function-body import at each of the two call sites without a named wrapper.
- Size: S · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D4] — the `maintenance_ready`/`maintenance_drain`/`maintenance_resume` poll loop is re-rolled three times while a helper for the shape exists and is used in this same slice
- Where: `UI/Console_Modules/dictation.py:874-906`, `Event_Handlers/STTS_Events/stts_events.py:709-722`, `Event_Handlers/TTS_Events/tts_events.py:718-731` — the three `maintenance_drain` bodies are identical apart from their `RuntimeError` string. The existing shape is `Backup_Recovery/runtime_producer_lifetime.ProducerLifetime` (`close`/`drain(deadline)`/`resume`), which `UI/Console_Modules/video.py:76,98-105` uses.
- Why it matters (only): three copies of a deadline loop that decides whether shutdown waits or gives up; `ProducerLifetime.drain` polls every 10 ms, the copies every 20 ms, so the same "settle" means two different things depending on the subsystem.
- Recommended correction: a `drain_until(predicate, deadline)` in `runtime_producer_lifetime.py` (the copies poll a predicate, not a call registry, so they cannot use `ProducerLifetime` as-is) and three 3-line call sites.
- Size: S · ADR: no · Confidence: verified (three bodies read) · Pinning test: none · Already covered: none.

### P3 [D4] — `_maybe_await` is defined 66 times in the tree; one copy is in this slice
- Where: `UI/Console_Modules/character_context.py:187-188`. Census: `grep -rn "def _maybe_await" tldw_chatbook/ | wc -l` → `66`; body census shows 59 share the identical two-line body.
- Recommended correction: one `Utils/` helper (this is a tree-wide cleanup, not a Console one — reported here only because the slice owns one copy).
- Size: M (tree-wide) · ADR: no · Confidence: verified · Pinning test: none · Already covered: none.

### P3 [D3] — six sites reach into another object's private state across a package boundary
- Where: `right_rail.py:745` and `:763` (`section._reconcile_scheduled`) and `:1191` (assigning `section._on_focus_recovery`, which `Widgets/Console/console_bounded_section.py:97` takes as a constructor argument); `hands_free.py:1564` (`session.controller._send_delay_seconds`, from `Chat/console_hands_free.py`); `hands_free.py:1815` (`self.app_instance._thread_id`); `review_selection.py:381, 472, 598` (`getattr(store, "_sessions", {})` where `store.sessions()` is public and used that way in `image.py:860`); `session.py:2320` (`store._pending_workspace_projections[...]`, carrying its own `# noqa: SLF001`); `message.py:2984` (`store._leaf_under`).
- Why it matters: `review_selection.py`'s three are the live ones — `getattr(..., {})` means a rename degrades silently to "Console" in a saved note's byline rather than failing; the rest pin private names that a refactor of `Widgets/Console` or `Chat/` will break with no compiler help.
- Recommended correction: use `store.sessions()` in `review_selection.py`; expose `reconcile_scheduled` (read-only property) and a `set_focus_recovery()` on `ConsoleBoundedSection`; expose `send_delay_seconds` on `HandsFreeController`.
- Size: S · ADR: no · Confidence: verified (all six read) · Pinning test: none · Already covered: none.

### P3 [D1] — 26 `query_one` calls in the slice sit outside any `try`, against a package idiom of `except (NoMatches, QueryError)`
- Where: `provider_continuation_recovery.py` (16: 216, 231, 232, 235, 294, 476, 485, 486, 499-502, 532, 574, 648, 651), `left_rail.py` (5: 727, 733, 1646, 1653, 1910), `right_rail.py` (3: 978, 979, 1055 — inside `on_key`), `workspace.py` (2: 968, 5532).
- Evidence: `ast` scan over all 42 files (command in "Left UNVERIFIED"). Read of `provider_continuation_recovery.py:205-300`: they are `self.query_one` calls on the widget's own unconditionally-composed children, from `on_mount`, `sync_recovery` and a `Button.Pressed` handler — all mounted-by-construction contexts, and `sync_recovery` is called from the screen at `chat_screen.py:18137, 22137`.
- Why it matters: none of these is a timer callback (the crash shape the brief names), so this is a consistency finding, not a live crash — but `sync_recovery` is reachable from screen code, and the same file's neighbours all guard.
- Recommended correction: none required; if touched, match the file-local `except (NoMatches, QueryError): return` idiom.
- Size: S · ADR: no · Confidence: inferred (no reproduction of an actual raise)
- Pinning test: none · Already covered: none.

## Candidate dispositions
| candidate (file:line pattern) | disposition |
|---|---|
| dup_verbatim `_utf8_prefix` Chat/console_raw_cli.py:84 ≡ raw_cli.py:74 | confirmed → P3 [D4] |
| dup_verbatim `_set_message_attachments` ≡ `_apply_console_message_attachments` message.py:186 | confirmed → P2 [D4] (plus the inverted import) |
| dup_verbatim `_console_message_role_from_persisted` message.py:597 | confirmed → P2 [D4] |
| dup_verbatim `_blank_console_session_settings` session.py:3499 / workspace.py:1215 | confirmed → P3 [D4] |
| dup_verbatim `_console_new_chat_default_generation` session.py:3506 / workspace.py:1222 | confirmed → folded into the row above |
| dup_verbatim `list_image_models_for_catalog` image.py:55 | confirmed → P3 [D4] |
| dup_verbatim/dup_shape `_split_console_skill_name_args` skill.py:111 | unverified (check: `sed -n '105,125p' UI/Console_Modules/skill.py Chat/console_command_grammar.py:151 Chat/console_chat_controller.py:3041` and diff the three bodies) — three-token-split helpers, not examined |
| dup_shape `maintenance_drain` dictation.py:893 | confirmed → P3 [D4] |
| dup_shape `_request_retry` right_rail.py:307 (19 copies) | retired — `event.stop()` + `post_message(self.RetryRequested())`, two lines with no shared semantics; a shared helper would be longer than the code. Mechanical shape noise. |
| dup_shape `_play_console_video`/`_save_console_video_copy`/`_regenerate_console_video_message` message.py:555-567 | retired — one-line delegations from the message controller to `self._video.X(...)`; that IS the decomposition contract. |
| dup_shape `_handle_workspace_create_result` workspace.py:4739 (3 copies) | unverified (check: `grep -n -A20 "_handle_workspace_create_result" UI/Console_Modules/workspace.py UI/Screens/library_screen.py UI/Screens/settings_screen.py`) — cross-slice, not examined |
| other `dup_shape` rows (wiring.py:342/386/397/1964, character.py:482, dictation.py:1157/1164/1172, hands_free.py:1588, video.py:107) | retired — read in full in their files; each is a 2-5 line delegation/timer-stop/status-set with no shared behaviour to extract. |
| dotted_section_setting video.py:1361 (`chat.videos`) | retired — resolves (TASK-1771) AND the resolved path goes through `Utils/path_validation.validate_path_simple` at `video.py:1359` before `mkdir`/`copy2`. Trust boundary covered. |
| dotted_section_setting message.py:2092 (`chat.images`) | retired — same shape; `validate_path_simple` at `message.py:2089` before `mkdir`/`write_bytes`. |
| raw_mkdir video.py:1364 / message.py:2095 | retired — both on an already-validated path (above). |
| os_replace_no_atomic video.py:700 | retired — the staged sibling is fsynced before commit (`video.py:633 os.fsync`), the commit is `os.link`/`os.replace` with `src_dir_fd`/`dst_dir_fd` pinning plus pre-commit identity revalidation (`video.py:645-705`), and capabilities are checked fail-closed (`:507-529`). `Utils/atomic_file_ops.py` cannot express dir-fd pinning or the create-if-absent hardlink commit, and its own `atomic_write_bytes` (`:178-184`) fsyncs no more than this does. |
| except_exception_pass prompts.py:1529/1533/1539/1543 | retired — all four are the rollback arms of `_apply_guarded_prompt_application`; the user is always told via `_warn_prompt_application` on the same path (`prompts.py:1545-1549`). |
| except_exception_pass raw_cli.py:333/348/405/442 | retired — refusal-delivery and run-row-terminalisation arms; each has a `noqa: BLE001` plus a stated reason, and the durable failure is reported through `report_persistence_error`/`refuse_for_persistence_error`. |
| except_exception_pass session.py:5189, terminal.py:113/137 | terminal.py rows folded into the P2 above; session.py:5189 retired (teardown arm, read in run 1). |
| except_exception_return_per_file terminal.py (9) | confirmed → P2 [D1] |
| except_exception_return_per_file send_price.py (2) | retired — `send_price.py:145` and `:226` return the documented degraded copy (`UNAVAILABLE_PRICE`), which the presentation layer renders. |
| except_exception_return_per_file image.py (4), message.py (6), session.py (7), workspace.py (2), wiring.py (3), left_rail.py, character.py, character_context.py, prompts.py, raw_cli.py, review_selection.py, status_row.py | retired — every one read (image/message/session/workspace/wiring/left_rail/character_context/prompts/raw_cli in full, review_selection/status_row sampled): each logs (`logger.opt(exception=True)`) or notifies before returning. |
| function_body_import_per_file (24 files, ~160 imports) | retired as a class — spot-resolved every target named in the excerpt's largest files (workspace 28, video 18, hands_free 18, image 15, message 14, wiring 12): all resolve, and the ones with a stated reason are cycle-breaking (`..Navigation.character_conversation_navigation` in `character_context.py`) or cost-deferring on the UI-ready path (`hands_free.py:1290` cites the PR #2638 census ratchet by name). No import of a module that no longer exists. |
| inline_truncate review_selection.py:460 | retired — `first_line[:47] + "…"` builds a 48-char note TITLE, not a transcript preview; `Chat/console_session_titles`' helper is about session titles and is not a drop-in. P3-tier noise. |
| legacy_markers_per_file (12 files) | retired — sampled the markers in agent/left_rail/message/session/workspace during the full reads: they are TASK-id provenance comments, not dead-code markers. |
| lock_and_execute raw_cli.py:0 | retired — the lock is `threading.Lock()` guarding the repaint-coalescing closure state (`raw_cli.py:536-610`); the "execute" is `runtime.execute` (process launch), not SQL. No DB statement in the file. |
| loguru_and_logging dictation.py, realtime.py | retired — `import logging` is used for exactly one thing in each file, the level constant `logging.ERROR` passed to `Utils/persistent_diagnostics.persist_event` (`dictation.py:1219`, `realtime.py:918`), whose signature is `level: int = logging.INFO` and whose docstring explains why persistence deliberately uses stdlib levels. `grep -n "logging\." ` → those two lines only. |
| plain_readback session.py:343 `.plain` | retired — read in run 1; not a user-text read-back through a Textual 8 label. |
| raw_1024x1024 image.py:43 | retired — the match is `REMOTE_IMAGE_MAX_BYTES = 8 * 1024 * 1024`, a byte budget, not an image dimension. |
| run_worker_coroutine_per_file (12 files, 89 calls) | retired as a class — `ast` scan found ZERO `run_worker(exclusive=True)` without `group=` across all 42 files; the six group-less calls (`dictation.py:1890`, `session.py:2715`, `wiring.py:644`, `wiring.py:2149`, `worktree.py:44`, `worktree.py:66`) are all non-exclusive. |
| seed_name `_maybe_await` character_context.py:187 | confirmed as duplication (P3 [D4]); retired as a D1 — its one call site (`character_context.py:1310`) awaits `wiring.py:184 _start_character_context_chat`, which is `async def`, so no sync sqlite runs on the loop. |
| strftime message.py:2096, review_selection.py:474, right_rail.py:208 | retired — filename stamp, note byline date, and a UTC clock label; none is a persisted/wire format. |
| try_import_guard archive.py:89/399, hands_free.py:859/1289, image.py:441, retrieval.py:538, workspace.py:1002/4903/5964 | retired — all nine read: each wraps a real awaited operation (`storage_call`, `change_conversation_archive`, `fetch_image_bytes`, `summarize_active_world_books` in `to_thread`, the workspace openers), not a bare import, and each logs or notifies. |

## Verified-fine
- **`raw_cli.py` vs ADR-093/094**: this module never spawns a process — it validates (`validate_raw_cli_request`, `:265`), checks `runtime.permitted`/`runtime.armed` before anything else (`:238-241`), passes `timeout_seconds=MAX_RAW_TIMEOUT_SECONDS`, and hands the request to `Tools/raw_cli_executor`'s runtime on a `thread=True`, non-exclusive worker (`:279-291`). argv/shell policy, process-group kill and bounded waits live in the executor, outside this slice. No boundary violation here.
- **`fetch_image_bytes` (image.py:439-464)** is egress-validated: `Image_Generation/adapters/image_format_utils.py:211` calls `_validate_egress_or_raise` on every redirect hop and strips credentials cross-origin (`:218-223`). The remote-image path is also opt-in (`resolve_render_remote_images`, `image.py:399`), capped at 8 MB, and attempt-bounded by an `OrderedDict` LRU of 256.
- **External video save (`video.py:562-760`)** is the most careful file-write in the slice: dir-fd pinned parent, `O_NOFOLLOW`, random staged sibling at 0600, fsync, size revalidation, identity revalidation immediately before commit, hardlink commit when the target is believed absent, and a `finally` that only unlinks a sibling whose (dev, ino, mode) still matches.
- **Hands-free store tap is uninstalled**: `hands_free.py:1744 uninstall_console_hands_free_store_tap` is called from `chat_screen.py:17008` inside `on_unmount`, and restores only seams still holding this screen's wrapper. The documented leak (store outlives screen) is closed.
- **Timer callbacks in this slice do no unguarded DOM work**: `dictation.py:1172 _tick_console_dictation_elapsed` and `hands_free.py:1362 _tick_console_hands_free` both go through `_console_composer_or_none()` with a `None` check; `raw_cli.py`'s `threading.Timer` repaint marshals to the UI thread and drops itself on `terminal`.
- **`character_context.py`** puts every DB call through `DB/base_db.run_owned_db_call` with a three-attempt scope fence and authority/revision revalidation on both sides of each await — no sqlite on the loop anywhere in the file.
- **`library_activity.py:264`** runs `store.retry_library_activity` through `asyncio.to_thread`; `right_rail.py:666`'s `run_worker` for it carries `group="console-library-activity-retry"`.
- **No `run_worker(exclusive=True)` without `group=`** anywhere in the 42 files (ast scan).
- **No `re.compile` inside a function body** in the slice (5 patterns, all module-level), no `fetchall()` without a bounding `WHERE`/`LIMIT`, and `get_cli_setting` appears twice — both on explicit user save actions, never in `compose()` or a loop.

## Retired
- *"`video.py:700` hand-rolls `os.replace` next to `Utils/atomic_file_ops`"* — symptom real (a hand-rolled commit), cause wrong: it is strictly stronger than the helper (fsync + dir-fd pinning + identity revalidation + hardlink-if-absent), and the helper cannot express what it needs. Evidence in the dispositions table.
- *"the dotted `get_cli_setting("chat.videos"/"chat.images", "save_location", …)` writes user attachments to an unvalidated path"* — symptom absent: both paths pass through `Utils/path_validation.validate_path_simple` before `mkdir` (`video.py:1359`, `message.py:2089`).
- *"`dictation.py`/`realtime.py` mix loguru and stdlib logging"* — cause wrong: one stdlib level constant each, for `persist_event`, whose documented contract requires stdlib levels.
- *"`_maybe_await(sync_call())` runs sqlite on the loop in `character_context.py`"* — cause wrong: the awaited callable is `async def _start_character_context_chat` (`wiring.py:184`).
- *"`image.py:43` hard-codes a 1024×1024 image size"* — mechanical false positive (`8 * 1024 * 1024` bytes).

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| The 26 unguarded `query_one` sites can actually raise in production (P3 [D1]) | needs a live screen mid-recompose; the brief forbids running the app | `tmux -L verify new -d -s c 'cd /Users/macbook-dev/Documents/GitHub/tldw-review && TLDW_CONFIG_PATH=<SCRATCH>/home/.config/tldw_cli/config.toml .venv/bin/python -m tldw_chatbook.app'` then drive Console ▸ trace-recovery region per `.claude/skills/verify/SKILL.md` and `tmux -L verify capture-pane -p -e` |
| The AST scans above (reproduce the coverage claims for the 20 mechanical-only files) | n/a — these are the commands themselves | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY - <<'EOF'` … the two `ast` walkers printed in this report's method (run_worker exclusive/group; `query_one` outside `try`) `EOF` |
| `skill.py:111 _split_console_skill_name_args` vs its two siblings | not examined (file sampled mechanically only) | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && sed -n '105,125p' tldw_chatbook/UI/Console_Modules/skill.py; sed -n '145,165p' tldw_chatbook/Chat/console_command_grammar.py; sed -n '3035,3055p' tldw_chatbook/Chat/console_chat_controller.py` |
| `_handle_workspace_create_result` three-copy row (workspace.py:4739 + two screens) | cross-slice; the two screen copies are outside this slice | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && grep -n -A25 "def _handle_workspace_create_result" tldw_chatbook/UI/Console_Modules/workspace.py tldw_chatbook/UI/Screens/library_screen.py tldw_chatbook/UI/Screens/settings_screen.py` |
| Whether any pinning test asserts the current (unconditional) `_sync_console_session_draft` title computation | targeted collection not run for this symbol | `cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && $PY -m pytest --collect-only -q Tests/UI 2>/dev/null \| rg -i "session_draft\|initial_session_title"` |
