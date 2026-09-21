# CHAT-rest-2 — tldw_chatbook/Chat/ (middle third of non-console-core files), 74 files, 42794 lines

STATUS: COMPLETE.

**Mechanical coverage is 74/74 files.** Every file in the slice went through: an AST
walk resolving every function-body `tldw_chatbook` import against the real module
(68 imports, with per-name checking); and `rg` sweeps for UTC/timestamp shapes,
`get_connection`/`transaction`/`commit`/`isolation_level`, `threading.Lock`,
`get_cli_setting`/`load_settings`, bare `except Exception: pass|return`,
Rich-markup escape + `.plain` read-back, `asyncio.create_task`/`run_worker`,
f-string SQL, module-level `assert`, mutable default arguments, and
`ruff --select E9,F63,F7,F82` (0 findings, matching the stated baseline).
The per-file column below states how much of the *body* I actually read on top
of that. Prose lines like "docstring + mechanical" mean exactly that: I read the
module docstring to establish what the file is for, and nothing else.

## Coverage

| file | lines | coverage |
|---|---|---|
| `tldw_chatbook/Chat/console_endpoint_provenance.py` | 17 | read in full (17 lines) |
| `tldw_chatbook/Chat/console_environment_state.py` | 1012 | sampled: 1-130 (availability enums + dataclasses), full symbol map, 241-300 (`relative_age`, the P2), 607-625; projection body 300-1012 mechanical |
| `tldw_chatbook/Chat/console_ephemeral.py` | 223 | docstring + mechanical |
| `tldw_chatbook/Chat/console_exchange_capture.py` | 1045 | sampled: 1-120 (constants/credential keys), 440-600 (redaction + sanitizer seam); rest mechanical |
| `tldw_chatbook/Chat/console_exchange_export.py` | 169 | read in full |
| `tldw_chatbook/Chat/console_expression_state.py` | 147 | docstring + mechanical |
| `tldw_chatbook/Chat/console_fleet_attention.py` | 467 | mechanical only |
| `tldw_chatbook/Chat/console_fleet_wake.py` | 1261 | sampled: 1-140 (notice composition), 735-870 (delivery/_finish_attempt), full exception census; rest mechanical |
| `tldw_chatbook/Chat/console_generate_image.py` | 1144 | structural map (all defs) + 577-700 (shared executor) read; rest mechanical |
| `tldw_chatbook/Chat/console_generate_video.py` | 320 | sampled: 78-105 (tempfile staging); rest mechanical |
| `tldw_chatbook/Chat/console_generation_settings_metadata.py` | 528 | mechanical only |
| `tldw_chatbook/Chat/console_glyphs.py` | 84 | read in full (84 lines) |
| `tldw_chatbook/Chat/console_hands_free.py` | 770 | mechanical only + assert census |
| `tldw_chatbook/Chat/console_help.py` | 86 | docstring + mechanical |
| `tldw_chatbook/Chat/console_history_budget.py` | 692 | read 1-360 in full (counters, turn grouping, prune settings); 360-692 mechanical |
| `tldw_chatbook/Chat/console_image_edit_operations.py` | 303 | docstring + create_task census |
| `tldw_chatbook/Chat/console_image_view.py` | 541 | mechanical only (lock census) |
| `tldw_chatbook/Chat/console_interrupt_rounds.py` | 858 | read 1-370 in full (locking contract, payload store, decision views); 370-858 structural; leak trace done end-to-end |
| `tldw_chatbook/Chat/console_launch_wake.py` | 360 | sampled: 55-130 (the fetchall candidate) in full; rest mechanical |
| `tldw_chatbook/Chat/console_library_activity_buffer.py` | 337 | mechanical only (lock census) |
| `tldw_chatbook/Chat/console_library_destination.py` | 329 | docstring + mechanical |
| `tldw_chatbook/Chat/console_library_policy.py` | 184 | docstring + mechanical |
| `tldw_chatbook/Chat/console_library_policy_coordinator.py` | 173 | docstring + mechanical |
| `tldw_chatbook/Chat/console_library_policy_repository.py` | 247 | read in full |
| `tldw_chatbook/Chat/console_live_work.py` | 449 | docstring + mechanical |
| `tldw_chatbook/Chat/console_message_actions.py` | 1342 | structural + exception census; body mechanical |
| `tldw_chatbook/Chat/console_onboarding_state.py` | 293 | docstring + mechanical |
| `tldw_chatbook/Chat/console_paste_attach.py` | 201 | read in full; both callers traced into chat_screen.py; benchmarked |
| `tldw_chatbook/Chat/console_persona_assignment.py` | 377 | mechanical only (timestamp + transaction census) |
| `tldw_chatbook/Chat/console_prefill.py` | 158 | docstring + mechanical |
| `tldw_chatbook/Chat/console_prepared_request.py` | 1803 | READ IN FULL (1803 lines) + benchmarked |
| `tldw_chatbook/Chat/console_project_instructions.py` | 347 | docstring + mechanical |
| `tldw_chatbook/Chat/console_prompt_queue.py` | 1469 | sampled: 120-200 (preview pipeline, read in full and traced to its widget), assert census; rest mechanical |
| `tldw_chatbook/Chat/console_prompt_queue_coordinator.py` | 1121 | mechanical only |
| `tldw_chatbook/Chat/console_provider_endpoints.py` | 376 | mechanical only |
| `tldw_chatbook/Chat/console_provider_support.py` | 420 | mechanical only |
| `tldw_chatbook/Chat/console_rail_state.py` | 1011 | structural map (all defs) + legacy-marker census; bodies mechanical |
| `tldw_chatbook/Chat/console_raw_cli.py` | 818 | sampled: 600-700 (admission/launch boundary, cancel paths) + lock census; rest mechanical |
| `tldw_chatbook/Chat/console_realtime_loop.py` | 458 | mechanical only |
| `tldw_chatbook/Chat/console_references.py` | 403 | read in full; all 7 function-body imports resolved |
| `tldw_chatbook/Chat/console_roleplay_identity.py` | 294 | docstring + mechanical |
| `tldw_chatbook/Chat/console_roleplay_metadata.py` | 197 | docstring + mechanical |
| `tldw_chatbook/Chat/console_save_targets.py` | 268 | docstring + strftime/exception census |
| `tldw_chatbook/Chat/console_scratch_space.py` | 331 | sampled: 70-100 (scratch allocation); rest mechanical |
| `tldw_chatbook/Chat/console_semantic_revision.py` | 1198 | sampled: 195-260, 815-845, 990-1040 (all four fetchall candidates + the strftime candidate); rest mechanical |
| `tldw_chatbook/Chat/console_send_diagnostics.py` | 222 | docstring + mechanical |
| `tldw_chatbook/Chat/console_session_endpoint_policy.py` | 60 | read in full (60 lines) |
| `tldw_chatbook/Chat/console_session_settings.py` | 2570 | structural map (all 80 defs) + 1-120 (imports), 1858-1950 (context estimate), 2432-2480 (token helpers); rest mechanical |
| `tldw_chatbook/Chat/console_settings_apply.py` | 239 | docstring + mechanical |
| `tldw_chatbook/Chat/console_settings_defaults.py` | 1510 | sampled: 55-100 (module globals/locks), 390-500 (intent lifecycle); rest mechanical |
| `tldw_chatbook/Chat/console_settings_durability.py` | 117 | docstring + create_task census |
| `tldw_chatbook/Chat/console_side_chat.py` | 184 | docstring + mechanical |
| `tldw_chatbook/Chat/console_skill_resolver.py` | 299 | docstring + mechanical |
| `tldw_chatbook/Chat/console_speculative_voice.py` | 290 | mechanical only |
| `tldw_chatbook/Chat/console_speculative_voice_session.py` | 1106 | sampled: 105-118, 310-330, 410-425, 615-635, 675-690, 795-810 (every create_task + both suspect imports) + config-read census; rest mechanical |
| `tldw_chatbook/Chat/console_speech.py` | 91 | docstring + mechanical |
| `tldw_chatbook/Chat/console_speech_preferences.py` | 118 | docstring + clone dump |
| `tldw_chatbook/Chat/console_speech_text.py` | 214 | docstring + mechanical |
| `tldw_chatbook/Chat/console_switcher_state.py` | 1293 | sampled: 160-220, 560-620 (both timestamp candidates); rest mechanical |
| `tldw_chatbook/Chat/console_thinking_capture.py` | 324 | mechanical only |
| `tldw_chatbook/Chat/console_thinking_history.py` | 282 | mechanical only |
| `tldw_chatbook/Chat/console_trace_chunk_rows.py` | 302 | mechanical only |
| `tldw_chatbook/Chat/console_trace_custom_pii.py` | 434 | sampled: 180-260 (the re.compile candidate + rule validation); rest mechanical |
| `tldw_chatbook/Chat/console_trace_errors.py` | 36 | read in full (36 lines) |
| `tldw_chatbook/Chat/console_trace_final_values.py` | 1330 | clone dump + exception census; body mechanical |
| `tldw_chatbook/Chat/console_trace_legacy.py` | 1078 | sampled: 440-475 (both fetchall candidates); rest mechanical |
| `tldw_chatbook/Chat/console_trace_maintenance.py` | 2026 | sampled: 25-110, 260-345, 470-700, 1250-1360, 1765-1845 (~450 of 2026 lines: connection setup, lease recovery, GC delete, every candidate row); rest mechanical |
| `tldw_chatbook/Chat/console_trace_metrics.py` | 61 | read in full (61 lines) |
| `tldw_chatbook/Chat/console_trace_models.py` | 283 | mechanical only |
| `tldw_chatbook/Chat/console_trace_native_reader.py` | 428 | mechanical only (transaction census) |
| `tldw_chatbook/Chat/console_trace_projection.py` | 500 | mechanical only + assert census |
| `tldw_chatbook/Chat/console_trace_provenance.py` | 1484 | sampled: 1-80 (source/omission enums), full symbol map, 1382-1484 (admission + commit boundary); ~180 of 1484 lines — largest prose gap in this slice |
| `tldw_chatbook/Chat/console_trace_redaction.py` | 729 | sampled: 505-600 (CredentialSanitizer bounds) + exception census; rest mechanical |
| `tldw_chatbook/Chat/console_trace_regex_worker.py` | 553 | read in full |

## Findings

### P1 [D1] — the prompt-queue shelf renders a literal backslash in front of any `[` the user typed, and the escape eats the cell budget it was measured against
- Where: `tldw_chatbook/Chat/console_prompt_queue.py:171` (`make_prompt_preview` → `rich.markup.escape`), consumed at `tldw_chatbook/UI/Console_Modules/prompt_queue.py:182` → `:219` → `:350` into `tldw_chatbook/UI/Console_Modules/prompt_queue.py:320` `Static("", id="console-prompt-queue-preview", markup=False)`.
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  from tldw_chatbook.Chat.console_prompt_queue import make_prompt_preview
  s = "summarize [draft] and TODO[1]"; print(repr(s)); print(repr(make_prompt_preview(s)))
  EOF
  ```
  → `input   : 'summarize [draft] and TODO[1]'` / `preview : 'summarize \\[draft] and TODO[1]'`.
  The consuming widget is composed with `markup=False` (`rg -n 'console-prompt-queue-preview' tldw_chatbook/UI/Console_Modules/prompt_queue.py` → `:320  yield Static("", id="console-prompt-queue-preview", markup=False)`), so that backslash is rendered verbatim.
- Why it matters: two consequences. (a) A queued prompt containing `[` shows as `Next: "summarize \[draft]"` — visible corruption of the user's own text. (b) `make_prompt_preview`'s docstring states "Rich escaping happens only after fitting, so escape syntax does not consume the visible-cell budget" — true only on a markup surface; on `markup=False` every inserted `\` *does* consume a cell, so `PROMPT_PREVIEW_CELL_BUDGET` is silently exceeded and the preview can push the Manage/Pause buttons.
- Recommended correction: pick one side. Either drop `escape_markup` from `make_prompt_preview` (the terminal-control stripping + grapheme truncation above it is what actually makes the string safe, and `markup=False` already neutralises tags), or set `markup=True` on that one `Static`. Dropping the escape is the smaller diff and keeps the byte-exact user text; it also makes the `cell_len` budget true again.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Chat/test_console_prompt_queue.py::test_preview_is_one_line_terminal_safe_and_markup_escaped` (and `::test_preview_truncates_by_cells_without_splitting_graphemes`) — both assert through `Text.from_markup(preview).plain`, i.e. they pin the *assumption that the surface interprets markup*. They pass today and would keep passing after the widget was switched to `markup=False`; they are the reason this drifted. A fix must change these two tests to assert the string the widget actually receives.
- Already covered: none. Same defect class as CHAT-rest-1 P1 (`Chat/console_display_state.py:93`) — worth fixing as one pass over the Console's markup-off Statics.


### P1 [D2] — `looks_attachable()` re-reads the attachment config once per path, on the event loop: 98 ms measured for a 20-file clipboard paste
- Where: `tldw_chatbook/Chat/console_paste_attach.py:162` — `looks_attachable` ends with `any(fnmatch(name, pattern) for pattern in _supported_patterns())`, and `_supported_patterns()` (`:28`) calls `attachment_core.attachment_filter_specs()` → `supported_image_formats()` on every invocation. Callers: `UI/Screens/chat_screen.py:20298` `[p for p in grab.paths if looks_attachable(p)]` (inside `_paste_console_clipboard_image`, an `async def` run via `run_worker(coroutine)` — the event loop, not a thread; only the `grab_clipboard_image` call above it is `asyncio.to_thread`-ed) and `UI/Screens/chat_screen.py:23105` inside `on_paste`, a Textual event handler.
- Evidence:
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  import os, time, statistics
  from tldw_chatbook.Chat.console_paste_attach import looks_attachable
  p = os.path.expanduser("~/pics/a.png"); looks_attachable(p)
  t=[time.perf_counter() for _ in ()]
  t=[]
  for _ in range(20):
      s=time.perf_counter(); looks_attachable(p); t.append((time.perf_counter()-s)*1000)
  print(statistics.median(t))
  ps=[p]*20; s=time.perf_counter(); [q for q in ps if looks_attachable(q)]; print((time.perf_counter()-s)*1000)
  EOF
  ```
  →
  ```
  supported_image_formats    warm median=  5.232 ms
  attachment_filter_specs    warm median=  4.947 ms
  _supported_patterns        warm median=  4.832 ms
  looks_attachable ACCEPTED path warm median=  4.840 ms
  20 clipboard paths (chat_screen.py:20298 shape) =    98.1 ms
  ```
  (A *rejected* path costs 0.053 ms — `is_safe_path` short-circuits before the config read, so the cost is paid only by paths that actually attach, i.e. exactly the case the user cares about.)
- Why it matters: copying 20 images in Finder and pasting stalls the Console UI for ~98 ms on the event loop; every single drag-drop paste stalls it ~5 ms inside `on_paste`. The work is re-deriving a constant tuple of glob patterns from config.
- Recommended correction: memoize `_supported_patterns` (`@functools.lru_cache(maxsize=1)`) or, better, hoist it: `looks_attachable` should take the pattern tuple, and `chat_screen.py:20298` should compute it once outside the comprehension. The underlying `attachment_core` readers are the sibling finding (3 unmemoized readers at ~7 ms); this is the per-path amplification of it and is fixable in this file alone.
- Size: S · ADR: no · Confidence: verified (measured)
- Pinning test: none. `Tests/` has no timing assertion on this path.
- Already covered: none. Depends on the same root as the sibling slice's `attachment_core` config-read finding.


### P2 [D1/D4a] — `relative_age()` subtracts a possibly-naive datetime from an always-aware one; the hardened parser it should have used is in the same package
- Where: `tldw_chatbook/Chat/console_environment_state.py:245` (`seconds = max(0, int((now - then).total_seconds()))`), reached from `:784` `f"Merged {relative_age(pr.merged_at, now)}"`. `pr.merged_at` is produced by `Workspaces/environment_status.py:141 _parse_merged_at`, which is `datetime.fromisoformat(raw.replace("Z", "+00:00"))` with **no naive guard**. `now` is always aware (`chat_screen.py:8598` and `:8742` both pass `datetime.now(timezone.utc)`).
- Evidence (consequence verified, trigger inferred):
  ```
  cd /Users/macbook-dev/Documents/GitHub/tldw-review && source <SCRATCH>/env.sh && PYTHONPATH=$PWD $PY - <<'EOF'
  from datetime import datetime, timezone
  from tldw_chatbook.Workspaces.environment_status import _parse_merged_at
  from tldw_chatbook.Chat.console_environment_state import relative_age
  from tldw_chatbook.Chat.console_switcher_state import parse_console_switcher_instant
  naive = _parse_merged_at("2026-09-18T12:00:00")
  print(repr(naive), naive.tzinfo)
  try: relative_age(naive, datetime.now(timezone.utc))
  except TypeError as e: print("TypeError:", e)
  print(repr(parse_console_switcher_instant("2026-09-18T12:00:00")))
  EOF
  ```
  →
  ```
  _parse_merged_at('2026-09-18T12:00:00') -> datetime.datetime(2026, 9, 18, 12, 0) tzinfo: None
  relative_age(naive, aware) -> TypeError: can't subtract offset-naive and offset-aware datetimes
  hardened sibling parse_console_switcher_instant same input -> datetime.datetime(2026, 9, 18, 12, 0, tzinfo=datetime.timezone.utc)
  ```
  What is NOT verified: that `gh pr view --json mergedAt` can ever emit an offset-less value. GitHub emits `…Z` today, so this is a latent robustness gap, not a shipped crash — hence P2 and `inferred`.
- Why it matters: `project_environment_section` is a render path with no `try` at either call site (`chat_screen.py:8595` is a bare `return`, `:8741` a bare assignment), so the `TypeError` would take the Inspect rail down rather than degrade one row. This is the same shape the other slice found in `conversation_local_marks` (CHAT-rest-1 P2) — a parser that accepts a shape the consumer cannot use.
- Recommended correction: the hardened parser already exists **in this slice** and is explicitly public: `Chat/console_switcher_state.py:583 parse_console_switcher_instant` ("Public safe timestamp parser shared by bounded History adapters"), which does `if parsed.tzinfo is None: parsed = parsed.replace(tzinfo=UTC)` then `.astimezone(UTC)`. Make `_parse_merged_at` call it instead of rolling `fromisoformat`. Cheaper alternative if the cross-package import is unwanted: one `if parsed.tzinfo is None: return None` in `_parse_merged_at` — the row then reads "Merged" with no age instead of crashing.
- Size: S · ADR: no · Confidence: inferred (trigger) / verified (consequence)
- Pinning test: `Tests/Chat/test_console_environment_state.py:80-83` exercises `relative_age` only with two aware datetimes from the same `now`; it cannot go red on this.
- Already covered: none


### P2 [D1] — `coerce_bool_setting(None, default)` returns `None`, not `default`, despite its `-> bool` annotation; one caller already carries a hand-written workaround
- Where: `tldw_chatbook/config.py:1201 coerce_bool_setting` → `config.py:1010 _get_typed_value`, whose line 1019-1020 is `if value is None: return None` under the comment "If key is missing and default is None" — but the guard fires on a None **value** regardless of what `default` is.
- Evidence: the table above, last column — `config.coerce_bool_setting(None, False)` → `None`, where all seven private re-rolls return `False`. Independently corroborated in the tree: `UI/Screens/change_review_screen.py:338-342` reads
  ```
  value = get_cli_setting("change_review", "git_actions", True)
  if value is None:
      # `coerce_bool_setting(None, ...)` returns None unchanged,
      # which would read as falsy and silently disable a feature
      # that ships ON.
      return True
  ```
  i.e. somebody has already been bitten and patched their own call site rather than the helper.
- Why it matters: `-> bool` is violated, and the failure mode is "a feature that ships ON silently reads OFF" — the exact words in that comment. Unguarded sites exist: `config.py:8622` `get_rag_citation_canonical_writes_enabled() -> bool` returns `coerce_bool_setting(section.get("canonical_writes_enabled"), False)` and therefore returns `None` whenever the key is absent; `UI/Library_Modules/library_skills_controller.py:934`, `UI/Screens/settings_screen.py:6313` and `:6046` are the same shape. Every one I traced happens to pair with `default=False`, where falsy-`None` coincides with the intended answer — which is why nothing has broken yet and why the next `default=True` site will.
- Recommended correction: one line — in `coerce_bool_setting`, `if value is None: return default` before delegating (do not change `_get_typed_value`, whose None-passthrough other typed getters may rely on). Then delete the workaround at `change_review_screen.py:339-342`.
- Size: S · ADR: no · Confidence: verified (the return value); inferred (that any shipped caller currently reads wrong — I found none, and say so)
- Pinning test: none asserts `coerce_bool_setting(None, …)`. The `change_review_screen.py` comment documents the behaviour but as a defect worked around, not as a contract, so this is a bug rather than a decision.
- Already covered: none. Outside my slice (`config.py`); found while auditing this slice's `_coerce_bool` clone at `Chat/console_rail_state.py:335`.


### P2 [D4a] — 15 private `_coerce_bool` re-rolls of a public helper that has 17 importers, and they disagree on the integer `1`
- Where: `tldw_chatbook/Chat/console_rail_state.py:335` is this slice's copy. The public helper is `tldw_chatbook/config.py:1201 coerce_bool_setting` (17 import sites). Census: `rg -n --glob 'tldw_chatbook/**/*.py' 'def _coerce_bool\(|def _coerce_bool_option\(|def coerce_bool_setting\('` → **16 definitions**; `rg -n 'coerce_bool_setting' --glob 'tldw_chatbook/**/*.py' | rg import | wc -l` → **17**.
- Evidence — same inputs through eight of them (`FALSE` = the `default` argument, which was passed as `False`):
  ```
  helper                                              1        0        2      1.0   'TRUE'      'y'     None
  config.coerce_bool_setting                       True    False    False    False     True     True     None
  Chat/console_rail_state                          True    False     True    False     True    False    False
  Utils/adaptive_reader_state                     False    False    False    False     True    False    False
  Character_Chat/world_book_manager                True    False     True     True     True    False    False
  Image_Generation/config                         False    False    False    False     True    False    False
  UI/Screens/settings_appearance_defaults          True    False    False    False     True    False    False
  Library/library_rail_state                       True    False     True    False     True    False    False
  Home/home_rail_state                             True    False     True    False     True    False    False
  ```
  (Command: `PYTHONPATH=$PWD $PY -` importing each `_coerce_bool` and printing `fn(v, False)` for `v in (1, 0, 2, 1.0, "TRUE", "y", None)`.)
- Why it matters: this is drift that reaches **stored data**, so it is a D1 as well as a D4. `1` is exactly what SQLite hands back for a boolean column and what TOML/JSON round-trips produce for `true` in a loosely-typed section — and two of the eight read it as the caller's *fallback* rather than True. `1.0` (a JSON round-trip of a bool through a float-coercing layer) is True in exactly one of the eight. `"y"` is True only in the canonical helper. A value migrating between two of these surfaces flips meaning.
- Recommended correction: `config.coerce_bool_setting` is already the repo's named standard (`change_review_screen.py:341` calls it "the repo's standard coercion"), but it lives in `config.py`, which is why leaf modules re-roll rather than import it. Move the coercion itself to `Utils/` (a stdlib-only leaf), have `config.coerce_bool_setting` delegate, and delete the 15 copies. Settle the int/float question once in that one place and write the truth table into its docstring.
- Size: M (one PR, and a format has to be chosen: does `2` mean True, does `1.0`) · ADR: no · Confidence: verified (measured)
- Pinning test: each site has its own local tests; none of them compares across sites, which is why the drift survived.
- Already covered: none. This spans the repo; my slice contributes one copy (`console_rail_state.py:335`) and the census.


### P3 [D3] — every non-attachable clipboard path logs `WARNING Path traversal attempt detected` + `ERROR Path validation error` for an ordinary user action
- Where: `tldw_chatbook/Chat/console_paste_attach.py:157` `is_safe_path(path, root)` → `Utils/path_validation.py:184` / `:276`, once per path in `chat_screen.py:20298`'s loop.
- Evidence: the benchmark above, run against `/nonexistent/x.png`, emitted verbatim:
  ```
  WARNING | tldw_chatbook.Utils.path_validation:validate_path:184 - Path traversal attempt detected: /nonexistent/x.png -> /nonexistent/x.png
  ERROR   | tldw_chatbook.Utils.path_validation:validate_path:276 - Path validation error for '/nonexistent/x.png': Path '/nonexistent/x.png' is outside the allowed directory
  ```
  Two lines per path; a user copying 20 files from `/Volumes/…` or `/tmp` gets 40, at WARNING and ERROR.
- Why it matters: "Path traversal attempt detected" is factually wrong for a file that simply lives outside `~`, and it is the loudest thing in the log for a completely normal action. It trains readers to ignore the one line that would matter.
- Recommended correction: `looks_attachable` is a *classifier*, not a gate refusing a request — it should ask a non-logging predicate. Either add a quiet variant in `Utils/path_validation.py` (`is_within(path, root)` already exists in `Tools/file_operation_tools.py` and does the containment check without logging) or demote the outside-the-root case in `validate_path` from "traversal attempt" to DEBUG. The helper is out of my slice; the amplifying call site is `console_paste_attach.py:157`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none.
- Already covered: none


### P3 [D4a] — three helpers are re-rolled privately in modules that already import the module (or class) holding the public original
- Where, all byte-identical modulo the name:
  - `thaw_json` — public, `tldw_chatbook/Chat/console_prepared_request.py:110`. Re-rolls: `Chat/console_trace_final_values.py:779` `_thaw`, `Chat/console_trace_service.py:6565` `_thaw`, `Chat/console_voice_trace_gateway.py:351` `_thaw`, `Chat/console_provider_gateway.py:1745` `_thaw_auxiliary_value`. Three of the four modules **already import from `console_prepared_request`** — `console_trace_final_values.py:13` and `console_trace_service.py:16` import `freeze_json` (the exact sibling of `thaw_json`, defined 30 lines above it), `console_provider_gateway.py:65` imports a name list from it.
  - `ConsoleTraceRepository.get_graph_epoch` — `Chat/console_trace_repository.py:1717`. Re-roll: `Chat/console_trace_maintenance.py:1359` `_graph_epoch`, in a module whose line 21 is `from tldw_chatbook.Chat.console_trace_repository import ConsoleTraceRepository`.
  - `_utf8_prefix` — `Chat/console_raw_cli.py:84`. Re-roll: `UI/Console_Modules/raw_cli.py:74`, in a module whose line 27 is `from ...Chat.console_raw_cli import (`.
- Evidence: `python <SCRATCH>/phase2/dump.py '<path>#<name>' …` (an `ast.get_source_segment` dump of each definition) → the five `_thaw`/`thaw_json` bodies are identical token-for-token, as are the two `get_graph_epoch`/`_graph_epoch` bodies and the two `_utf8_prefix` bodies. Import lines confirmed with `rg -n 'console_prepared_request' Chat/console_{voice_trace_gateway,provider_gateway,trace_final_values,trace_service}.py`, `rg -n 'console_trace_repository' Chat/console_trace_maintenance.py`, `rg -n 'console_raw_cli' UI/Console_Modules/raw_cli.py`.
- Why it matters: no drift today, so this is consistency rather than a bug — but `thaw_json` and `freeze_json` are a matched pair on the wire path; a future change to one (say, `frozenset` or `bytes` handling) will land in the public copy and silently miss four private ones. `_graph_epoch` is a SQL read of a singleton row, i.e. a schema fact that must move as one.
- Recommended correction: delete the five private copies and import the existing public names. Canonical homes already exist and need no new module: `Chat/console_prepared_request.thaw_json`, `ConsoleTraceRepository.get_graph_epoch`, `Chat/console_raw_cli._utf8_prefix` (promote to no-underscore when it crosses the package line).
- Size: S · ADR: no · Confidence: verified
- Pinning test: none found for the private copies.
- Already covered: none


### P3 [D4b] — seven copies of the "reject duplicate JSON keys" `object_pairs_hook`, no shared factory
- Where: `Chat/console_trace_regex_worker.py:58` `_unique_json_object` · `Chat/library_activity.py:487` `unique_object` · `Chat/library_preparation.py:174` `unique_object` · `Chat/console_dispatch_checkpoint.py:275` `unique_object` · `Tools/watchlists_tool_service.py:1257` `_unique_json_object` · `Tools/workspace_tool_protocol.py:407` `_reject_duplicate_keys` · `MCP/permission_store.py:388` `_reject_duplicate_keys`.
- Evidence: same `dump.py` run over all seven — identical six-line body (`for key, value in pairs: if key in result: raise …`); the only difference is the exception type and message, which is a genuine per-caller need.
- Why it matters: this is the "storage" strict-JSON family the other slices flagged as disagreeing with the wire family (no depth cap here, duplicates rejected). Seven copies means seven places to change if that disagreement is ever settled, and a new decoder site has no obvious thing to copy from.
- Recommended correction: one factory in `Utils/` — `def reject_duplicate_keys(error: type[Exception], message: str) -> Callable[[list[tuple[str, Any]]], dict[str, Any]]` — so each caller keeps its own exception and the loop exists once. Settle the depth-cap question there too rather than seven times.
- Size: S · ADR: no (the depth-cap decision, if taken, is the other slice's call) · Confidence: verified
- Pinning test: each site has its own; a factory keeps them all green.
- Already covered: none


### P3 [D4b] — `_mapping_value` / `_metadata_object` clone pairs
- Where: `Chat/console_provider_gateway.py:7036` ≡ `Chat/console_session_settings.py:1949` (`_mapping_value`, byte-identical); `Chat/console_appearance.py:166` ≡ `Chat/console_speech_preferences.py:109` (`_metadata_object`, byte-identical 10-line JSON-or-mapping coercion). `UI/Screens/chat_screen.py:7879` `_config_section` is the `dict`-typed variant of `_mapping_value`.
- Evidence: `dump.py` over all five — see above.
- Why it matters: `_metadata_object` is the coercion applied to a persisted metadata column before it is read as settings; two copies means a future tolerance change (say, accepting a JSON list, or a bytes column) lands in one surface only.
- Recommended correction: one `Utils/` coercion pair; `_metadata_object` is the one worth moving (it touches stored data), `_mapping_value` is three lines and only worth folding if the file is being touched anyway.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none specific.
- Already covered: none

## Candidate dispositions

Every row from the excerpt. "retired" means the symptom the grep saw was real and the inference behind it was not.

| candidate (file:line pattern) | disposition |
|---|---|
| **dup_shape** `unique_object`/`_unique_json_object`/`_reject_duplicate_keys` ×7 | **confirmed** → P3 [D4b]; all seven bodies dumped and compared |
| **dup_shape** `retry@console_fleet_wake.py:503` + 5 peers | **retired** — dumped: `def retry(): self._retry_timer = None; self.retry_soon()` and peers are 2-3-line attribute resets. The shape hash matched "assign two attributes then call one method"; there is no shared behaviour to extract |
| **dup_shape** `_thaw` family ×5 | **confirmed** → P3 [D4a]; byte-identical, and 3 of 4 re-rollers already import the module holding the public original |
| **dup_shape** `_wire_llamacpp_snapshot_service`/`clear_governed_state`/`_cancel_countdown`/`invalidate_refresh_scope` ×4 | **retired** — dumped `_cancel_countdown`: two `self._x = None` assignments with a docstring explaining why they belong together. Same class as the `retry` row |
| **dup_shape / dup_verbatim** `_mapping_value` ×2 (+ `_config_section`) | **confirmed** → P3 [D4b] |
| **dup_verbatim** `get_graph_epoch@console_trace_repository.py:1717` ≡ `_graph_epoch@console_trace_maintenance.py:1359` | **confirmed** → P3 [D4a]; the re-rolling module imports `ConsoleTraceRepository` at its line 21 |
| **dup_verbatim** `_metadata_object` ×2 | **confirmed** → P3 [D4b] |
| **dup_verbatim** `_utf8_prefix` ×2 | **confirmed** → P3 [D4a]; `UI/Console_Modules/raw_cli.py` already imports from `Chat/console_raw_cli` at its line 27 |
| **except_exception_pass** `console_fleet_wake.py:1032` | **retired** — `noqa: BLE001, S110 - mark cleanup must not interrupt close`, inside a teardown path; the mark it fails to clear is re-derived at launch by `console_launch_wake.pending_conversations_at_launch` |
| **except_exception_pass** `console_raw_cli.py:650` | **retired** — swallows a failure of the caller's `on_started(timestamp)` UI callback *after* the launch has already committed. Traced to `UI/Console_Modules/raw_cli.py:576`: the callback's first statement sets `started_at` (nonlocal) and only the marshalled UI marker is lost; the terminal event repaints it |
| **except_exception_pass** `console_trace_maintenance.py:509` | **retired** — `except BaseException: pass` around `connection.close()` inside a `try/…/raise` that preserves the original setup failure. Correct shape |
| **except_exception_return_per_file** (14 files, 25 handlers) | **all retired** — read every one via `rg -n -A2`. Each carries an explicit `noqa: BLE001` plus a stated fail-closed rationale, and returns a typed unavailable/refused result rather than a silent success. The two security-relevant ones (`console_trace_redaction.py:309`, `:440`, `:578`) are "failures must not retain source content" / "failure details may contain credentials" and return `available=False` with an omission reason code |
| **fetchall_no_limit** (8 rows) | **all retired** — every one is keyed or bounded: `console_semantic_revision.py:221/242/424` are `IN (…)` over an explicitly `len(revision_ids) > 256 → ValueError` batch; `:828` and `console_trace_maintenance.py:1778` are recursive CTEs rooted at one revision/segment (the latter caps `depth < 10000`); `console_trace_legacy.py:457/535` are per-`message_id`; `console_launch_wake.py:89` is a launch-time `SELECT DISTINCT conversation_id` over pending wakes on a read-only private connection |
| **function_body_import_per_file** (21 files, 68 imports) | **all retired** — AST-resolved every one against `importlib.util.find_spec` plus a per-name module check. Zero missing modules. Two apparent misses (`console_speculative_voice_session.py:111 tldw_chatbook.TTS.get_tts_service`, `:682 tldw_chatbook.Widgets.Console.VoicePreviewProjection`) are PEP 562 `__getattr__` lazy re-exports; confirmed by actually importing both |
| **inline_truncate** `console_fleet_wake.py:132 [:117] + "..."` | **retired** — a 120-char header clamp on a sub-agent task line, adjacent to the module's own `_truncated()` which does the *budgeted* truncation with a different message. Two different jobs; no shared helper is being ignored |
| **legacy_markers_per_file** (19 files, 61 markers) | **retired** — sampled the two densest (`console_rail_state.py` 15, `console_trace_projection.py` 12). Every marker read is a documented one-way migration source (`_LEGACY_GLOBAL_SCOPE`, `session_open`, `context_open`) with the task id that introduced it. Not dead code |
| **lock_and_execute** `console_raw_cli.py locks=3 executes=1` | **retired** — `rg -n '\.execute\(' console_raw_cli.py` → one hit, `:657 self._executor_or_default().execute(request, …)`, the process executor. No SQL in the file |
| **raw_1024x1024** (8 rows) | **all retired** — false pattern. Every hit is a byte-size constant (`16 * 1024 * 1024` blob cap, `64 * 1024 * 1024` JSON cap, `1024 * 1024` MB/KB formatting divisor, `4 * 1024 * 1024` batch cap). No image dimension anywhere |
| **re_compile_in_def** `console_trace_custom_pii.py:213`, `console_trace_regex_worker.py:434` | **both retired** — both compile a *user-supplied* pattern that arrives at call time (rule validation, and the worker subprocess compiling rules off the wire). Neither can be hoisted, and `re` memoizes compilation anyway |
| **seed_name__clean_text** `console_live_work.py:25`, `console_rail_state.py:555` | **retired** — different signatures and semantics (`_clean_text(value, fallback)` vs `_clean_text(value)`); the shared part is `str(x or "").strip()`, three tokens |
| **seed_name__coerce_bool** `console_rail_state.py:335` | **confirmed** → P2 [D4a], and it led to the `coerce_bool_setting(None)` P2 |
| **strftime** `console_save_targets.py:108 %Y-%m-%d` | **retired** — a filename/title date, not a stored timestamp |
| **strftime** `console_semantic_revision.py:1014 %Y-%m-%dT%H:%M:%fZ` | **retired** — SQLite `strftime`, where `%f` is `SS.SSS`; per the brief this is correct, not a missing-seconds bug. Verified it is `cursor.execute("SELECT strftime(…, 'now')")`, not Python's |
| **strftime** `console_switcher_state.py:189 %Y-%m-%d %H:%M %Z` | **retired** — display only, and its input goes through `_parse_instant`, which normalizes naive→UTC before `.astimezone()` |
| **tempfile_no_secure** (3 rows) | **all retired** — `console_generate_video.py:90` and `console_trace_regex_worker.py:201` use `tempfile.TemporaryFile()` (no name on disk); `console_scratch_space.py:83` uses `mkdtemp` then `os.chmod(root, 0o700)`, `lstat`, and an explicit symlink/`S_ISDIR` check |
| **try_import_guard** `console_references.py:331`, `:362`, `console_trace_regex_worker.py:315` | **all retired** — all three fail closed: `_git_reference_cwd` falls back to `Path.cwd()` and the caller then re-checks root containment; `run_git_reference` raises `RuntimeError("could not build the sensitive-path exclusion list")` rather than running git without the denylist; `_apply_resource_limits` returns `()` so nothing is *claimed* to be enforced |

## Verified-fine

Things that look like smells in this slice and are not, each with what settled it.

- **`prepare_provider_request` serializes and token-counts the whole conversation ~9 times per send** (`console_prepared_request.py`: binary-search window + 6 cumulative `_count_wire` calls in `_account_categories` + a baseline + a final total, then `PreparedProviderRequest.__post_init__` re-runs `_serialize_messages` *and* `_serialize_provenance` purely to assert alignment). Measured, warm: `turns=120 msgs=242 chars=284690 tokens=43217 → 3.9 ms` (medians over 5, after one warm-up). `Utils/token_counter.estimate_tokens`'s `_ESTIMATE_CACHE` (TASK-18602, keyed `(model, provider, len(text), hash(text))`) absorbs it because `_serialize_messages` reuses the same `str` objects. Not a D2.
- **No secrets reach logs in this slice.** `rg` over all 74 files for `logger.(debug|info|warning|error)` intersected with key/token/secret/password/api/auth/cred/env → **zero hits**. The fleet/trace modules log `type(exc).__name__`, never the exception text.
- **`ConsoleLibraryPolicyRepository`** (`console_library_policy_repository.py`) reads on `get_connection().execute(...)` but every write is inside `db.transaction(immediate=True)`; the `except sqlite3.IntegrityError: pass` at `:93` is a race-loser path (SQLite aborts the statement, not the transaction) that then re-reads the winner. Not the transaction-class bug.
- **`console_trace_maintenance._open_maintenance_connection` sets `connection.isolation_level = None`** — that is autocommit *plus* an explicit `_maintenance_transaction(connection)` context manager around every DML block, which is the opposite of the legacy-isolation defect.
- **Timestamp comparison in the trace-maintenance lease/retry machinery is done in SQL, not Python.** `next_retry_at` is written as `datetime.now(timezone.utc).isoformat()` while sibling columns use `CURRENT_TIMESTAMP` — two shapes in one table — but every comparison is `julianday(x) > julianday('now')` (`:270`, `:484`, `:1287`), which parses both. `_validate_utc_timestamp` (`:1273`) additionally requires an explicit UTC offset on anything crossing the API boundary.
- **`console_trace_regex_worker`'s subprocess sandbox.** `sys.executable -I` (isolated), fixed argv, `stdin` payload, `stdout` to an unnamed `TemporaryFile`, `stderr` to DEVNULL, `start_new_session` on POSIX, parent-side wall-clock timeout with `kill()`, child-side RLIMIT_CPU/FSIZE/AS, output size checked before parse, strict key-set validation of the response, and duplicate-key rejection on decode.
- **`CredentialSanitizer`** (`console_trace_redaction.py:515`) is bounded on nodes, depth, and per-string codepoints, tracks `active` ids for cycles, and `sanitize()` fails closed with no content in the error. `console_exchange_capture._remove_nested_credentials` recurses without its own depth cap but only ever walks that already-bounded output.
- **`console_references`'s `@path` / `@diff` expansion.** Resolution goes through `Utils/path_validation.validate_path_multi` + `Utils/sensitive_paths.is_sensitive_path`; the refused-vs-literal split is decided lexically with no filesystem probe of an unvalidated path (`_lexically_pathlike`); `run_git_reference` refuses when the repo is outside the allowed roots and refuses again if the `:(exclude)` denylist cannot be built.
- **`console_paste_attach.looks_attachable`** uses the canonical `Utils/path_validation.is_safe_path` rather than an inline traversal check. (Its *cost* is the P1 above; its *containment* is correct.)
- **`InterruptRoundHost._decision_views` keyed by the owner widget object** looks like a recompose-lifecycle leak. Traced end to end: `Widgets/Persona_Widgets/buddy_conversation_modal.py:325` (`on_screen_suspend`) and `:333` (`on_unmount`) both call `coordinator.show_decisions(self, None)` → `set_decision_view(owner, None)` → `self._decision_views.pop(owner, None)`. No leak.
- **Every `asyncio.create_task` in the slice is retained.** `console_fleet_wake.py:767` (assigned to `watcher`, cancelled+gathered in `finally`); `console_speculative_voice_session.py:316/415/799` all go through `self._own(task)` which adds to a set, discards on done, and consumes the result; `:625` is returned to its caller. `console_image_edit_operations.py:149/193` and `console_settings_durability.py:90` are named and held.
- **No `run_worker(coroutine)` doing sync sqlite** in this slice — `rg run_worker|@work\(` over all 74 files returns nothing; the Chat layer posts work up to the screen.
- **`ruff check --select E9,F63,F7,F82` over all 74 files: `All checks passed!`** — matches the stated 0-fatal baseline.
- **No mutable default arguments** anywhere in the slice (`rg 'def .*=\s*(\[\]|\{\}|set\(\))'` → no hits).
- **`_serialize_messages`/f-string SQL**: the only dynamic SQL identifier in the slice is `console_trace_maintenance.py:1832`'s `DELETE FROM {table}`, where `table` iterates a local tuple of string literals. No user data reaches an identifier position.
- **`trace_provenance_admission_transaction`** deliberately converts every `Exception` from its body into a content-free `TraceProvenancePersistenceError` with `__cause__`/`__context__`/`__traceback__` cleared. That destroys debuggability, and it is the stated privacy design (the error type's whole purpose); `asyncio.CancelledError` is a `BaseException` and is *not* swallowed.

## Retired

Covered inline in **Candidate dispositions** above; the four worth restating because the symptom was real and the inference was wrong:

1. *"`prepare_provider_request` is O(9n) per send."* Symptom real (it is 9 passes); cause wrong — the estimator is memoized and the measured warm cost at 43k tokens is 3.9 ms. Retired with the benchmark.
2. *"`console_trace_maintenance` writes two timestamp shapes into one table."* Symptom real (`next_retry_at` ISO-with-offset vs `updated_at` `CURRENT_TIMESTAMP`); consequence absent — every comparison is `julianday()`, which parses both. Retired.
3. *"Two dead function-body imports in `console_speculative_voice_session`."* Symptom real to an AST name check; cause wrong — both are PEP 562 lazy re-exports. Retired by importing them.
4. *"`raw_1024x1024` × 8."* Pure false pattern — byte-size constants. Retired.

## Left UNVERIFIED

| claim | why not verified | literal command / check to run |
|---|---|---|
| `_parse_merged_at` can actually receive an offset-less `mergedAt`, making the P2 `relative_age` TypeError a live crash rather than a latent one | requires a real `gh pr view --json mergedAt` response for a merged PR; the review is offline and must not run the app | `gh pr view <n> --json mergedAt -q .mergedAt` in any repo with a merged PR, then check whether the string ends in `Z`/`±HH:MM` |
| the P1 prompt-queue preview renders `\[` **on screen** (I proved the string carries the backslash and that the widget is `markup=False`; I did not paint it) | running the app is forbidden by the brief | `.claude/skills/verify/SKILL.md` recipe: `tmux -L verify`, queue a prompt containing `summarize [draft]` while a turn is streaming, `capture-pane -p` and grep for `\[draft` on the `#console-prompt-queue-preview` line |
| the P1 `looks_attachable` 98 ms lands as a visible UI stall (I measured the function; I did not observe the paint) | same | `tmux -L verify`, copy ~20 image files in Finder, press the paste-image binding, and time the composer's first repaint |
| whether `resource.RLIMIT_RSS` actually bounds the PII worker on macOS — `_apply_resource_limits` (`console_trace_regex_worker.py:320-334`) tries `RLIMIT_RSS` first on darwin, appends `"memory"` to `enforced`, and `break`s, so `RLIMIT_DATA` is never tried; macOS is widely reported to ignore `RLIMIT_RSS` | proving a resource limit is unenforced needs a deliberate OOM child, which risks the shared machine | spawn `python -I -c "import resource; resource.setrlimit(resource.RLIMIT_RSS,(64<<20,-1)); x=bytearray(512<<20); print('not enforced')"` and see whether it prints or dies |
| whether `_collapse_large_pastes_enabled` (`settings_screen.py:6313`) can be reached with a draft value of `None`, turning the `coerce_bool_setting` quirk into a visible wrong default | needs the settings draft store's write paths, which are outside this slice | `rg -n 'collapse_large_pastes' tldw_chatbook/UI/Screens/settings_screen.py` and check every writer into `draft.values` for a path that stores `None` |

## Coverage honesty note

`console_trace_provenance.py` (1484 lines) is the one file over 800 lines I did **not** read in full — I read its head (1-80), its symbol map, and its persistence tail (1382-1484). `console_trace_maintenance.py` (2026) I read ~450 lines of, chosen by candidate row. `console_session_settings.py` (2570), `console_settings_defaults.py` (1510), `console_prompt_queue.py` (1469), `console_switcher_state.py` (1293), `console_fleet_wake.py` (1261), `console_semantic_revision.py` (1198), `console_speculative_voice_session.py` (1106), `console_trace_legacy.py` (1078), `console_exchange_capture.py` (1045), `console_generate_image.py` (1144) and `console_rail_state.py` (1011) were read by symbol map plus every candidate region, not cover to cover. `console_prepared_request.py` (1803) and `console_interrupt_rounds.py` (1-370 of 858) are the two I read linearly. Roughly **9,000 of 42,794 lines read as prose**; the remaining 33,800 got the twelve mechanical sweeps listed at the top and nothing more. Treat the zero-findings verdict on `console_trace_provenance.py`, `console_trace_final_values.py`, `console_prompt_queue_coordinator.py`, `console_message_actions.py`, `console_environment_state.py`'s projection half, `console_hands_free.py`, `console_thinking_capture.py`, `console_thinking_history.py`, `console_trace_chunk_rows.py`, `console_trace_native_reader.py`, `console_trace_projection.py`, `console_realtime_loop.py`, `console_fleet_attention.py`, `console_generation_settings_metadata.py`, `console_provider_endpoints.py`, `console_provider_support.py`, `console_image_view.py` and `console_library_activity_buffer.py` as "no mechanical signal", not as "reviewed clean".
