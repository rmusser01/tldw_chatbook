# AGENTS — tldw_chatbook/Agents/ (53 files), 40,468 lines

Worktree `/Users/macbook-dev/Documents/GitHub/tldw-review` @ d8fb4053f9. Read-only; two probe scripts written to the scratchpad only (`agents_probe_gates.py`, `agents_probe_persist.py`). No app run, no full-suite run.

## Coverage
| file | lines | read in full / sampled (which ranges) / mechanical only |
|---|---|---|
| agent_service.py | 8746 | read in full |
| local_tool_provider.py | 4406 | read in full |
| agent_runtime.py | 3213 | read in full |
| tool_catalog.py | 2263 | read in full |
| mcp_tool_provider.py | 1427 | read in full |
| canvas_tool_provider.py | 1281 | read in full |
| agent_models.py | 1183 | read in full |
| run_log_search.py | 1144 | read in full |
| run_log.py | 981 | read in full |
| builtin_tool_gate.py | 948 | read in full |
| fleet_coordinator.py | 945 | read in full |
| run_hooks.py | 909 | read in full |
| virtual_cli_provider.py | 675 | read in full |
| raw_shell_tool_provider.py | 601 | read in full |
| run_webhooks.py | 504 | read in full |
| agent_worktree.py | 455 | read in full |
| activation.py | 231 | read in full (+ diffed against MCP/activation.py, RAG_Search/activation.py) |
| native_tools.py | 198 | read in full |
| agent_stream.py | 180 | read in full |
| agent_worktree_git.py | 178 | read in full |
| agent_worktree_recovery.py | 644 | sampled: 85-125 (grep context), 505-520; mechanical scan |
| profile_tool_provider.py | 431 | sampled: the 3 `except Exception: return` sites (160-167, 208-215, 305-312); mechanical scan |
| ask_user_questions.py | 432 | sampled: 170-200; mechanical scan |
| recovery.py | 215 | sampled: 60-100; mechanical scan |
| agent_lesson_promotion.py, agent_presets.py, agent_routing.py, approval_provenance.py, automatic_work_budget.py, automatic_work_runtime.py, bulk_reader_corpus.py, execution_capacity.py, fallback_chain.py, fleet_message_tools.py, fleet_messages.py, fs_read_ledger.py, history_projection.py, human_input_wait.py, library_rag_tool_provider.py, library_tool_provider.py, model_retry.py, persona_policy.py, project_instruction_resolver.py, project_instruction_runtime.py, run_context.py, run_log_eviction.py, run_log_format.py, run_log_paging.py, run_tool_policy.py, session_todo_store.py, tool_arg_coercion.py, tool_refusals.py, __init__.py | 8,278 | mechanical only: `rg` for `except Exception: pass/return`, `threading.Lock`, indented `re.compile`, raw SQL `.execute("`, `httpx/requests/urlopen`, `tempfile/os.replace/mkdir`, `run_worker/query_one/asyncio.run` (all zero except: 1 lock each in agent_lesson_promotion/fleet_messages/fs_read_ledger/human_input_wait/project_instruction_runtime/run_tool_policy, 2 in library_tool_provider; `asyncio.run` in library_rag_tool_provider:395 (worker-thread provider invoke, same shape as tool_catalog:1487); `recovery.py:80` PRAGMA on a private read-only sqlite inside `closing()`) |

Totals: 30,468 lines read in full (75%), ~1,700 sampled, 8,278 mechanical. Lint: not re-run (brief baseline is 0 fatal for Tier-1; nothing here is a lint claim).

## Findings

### P2 [D1] — A read failure inside `AgentService._persist` skips the terminal write and leaves the PRIMARY run row `running` until the next app launch
- Where: `tldw_chatbook/Agents/agent_service.py:4627-4642` (`exact_terminal_event_id`), called at 4642 and 4656; the terminal write it gates is `_set_terminal_status` at 4690.
- Evidence: `cd $WT && source $SCRATCH/env.sh && $PY $SCRATCH/agents_probe_persist.py` (stub DB whose `get_run` raises `sqlite3.OperationalError("database is locked")`) →
  ```
  RESULT: _persist RAISED OperationalError: database is locked
  db calls: [('insert_steps_at_indices', 1), ('get_run',)]
  terminal write attempted: False
  ```
  `exact_terminal_event_id` catches only `(KeyError, StopIteration, TypeError)` (4638). `AgentRunsDB.get_run` (DB/AgentRuns_DB.py:2552-2566) runs through `connection()` → `_held_connection` → `_core_access` (Backup_Recovery/participants.py:392-417), so the escaping classes are `sqlite3.OperationalError` (busy_timeout 5000 ms, AgentRuns_DB.py:313) and `RecoveryRequired(RuntimeError)` (`storage_locally_paused`, participants.py:417). Every sibling read in this module wraps the same call in `except Exception` (`_latest_durable_event_id` 4056, `_next_owner_seq` 4016, `_service_error_step` 4556, `_resolved_control_event_id` 4066). The bridge's `finally` after `service.run_turn(` (Chat/console_agent_bridge.py:7266-7300) writes no terminal status. A fleet CHILD is covered by `run_child`'s `finally` fallback (`_set_terminal_status`, agent_service.py:5747-5753); the primary has no equivalent. `AgentRunsDB.reconcile_orphaned_runs` (DB/AgentRuns_DB.py:2340) flips stranded `running` rows to `error`, but only once per DB open (docstring: "On open ... tracked via `_swept_paths`").
- Why it matters: for the rest of the session the rail shows a run that never finishes, `steer_primary`/`send_to_agent` and `list_running_run_ids` treat it as live, and the reply's `run_turn` call raises into the bridge after the answer was already produced — a transient lock during fleet concurrency (children write steps to the same file) is the realistic trigger.
- Recommended correction: widen 4638 to `except Exception` (matching 4056) so a failed read degrades to the existing diagnostic/recovery path instead of aborting `_persist`; optionally wrap the whole of `_persist`'s pre-terminal section so `_set_terminal_status` always runs.
- Size: S · ADR: no · Confidence: verified (code path, stub repro); the production trigger (real lock/pause at exactly that read) is inferred, hence P2 not P1.
- Pinning test: `Tests/Agents/test_agent_service.py::test_terminal_recovery_does_not_duplicate_lifecycle_transition` (happy path only; nothing pins a raising `get_run` — `rg -n "OperationalError|RecoveryRequired" Tests/Agents | rg -i "persist|get_run"` → 0).
- Already covered: none.

### P2 [D1] — Agent worktrees are created under a fixed-name, default-mode directory in the shared system temp dir
- Where: `tldw_chatbook/Agents/agent_worktree.py:69-70` (`_worktrees_base` → `tempfile.gettempdir()/"tldw_agent_worktrees"`), `:126-129` (`mkdir(parents=True, exist_ok=True)` then `dest.parent.resolve(strict=True)`), consumed by `git worktree add` at 134-144.
- Evidence: read only — see UNVERIFIED. `rg -n "tldw_agent_worktrees|_worktrees_base" Tests tldw_chatbook` → every test monkeypatches `_worktrees_base` to `tmp_path`; `Tests/Agents/test_agent_worktree.py::test_checkout_created_under_symlinked_temp_base_can_be_recovered` pins that a symlinked base IS followed (stated as a requirement for app-owned aliases).
- Why it matters: on a multi-user host the first process to create `/tmp/tldw_agent_worktrees` owns it; `exist_ok=True` accepts a pre-existing directory or symlink of any owner/mode and `resolve(strict=True)` follows it, so every child checkout (user code, later merged back with `merge_agent_worktree`) lands wherever that name points. The per-run leaf name is uuid4 (unguessable) — the exposure is the parent, not the leaf. This is the classic fixed-name-in-tmp shape `tempfile.mkdtemp` exists to avoid.
- Recommended correction: root the base under the profile's data dir (`profile_paths.user_data_dir()/agent_worktrees`, app-owned, created 0o700) or `tempfile.mkdtemp(prefix="tldw-agent-")` per checkout; keep the `resolve(strict=True)` canonicalisation. The `agent_worktrees` DB records `child_path`, so no layout contract breaks.
- Size: M · ADR: no · Confidence: inferred
- Pinning test: `Tests/Agents/test_agent_worktree.py::test_checkout_created_under_symlinked_temp_base_can_be_recovered` (pins symlink-following as desired for owned aliases; does not pin the shared-tmp location).
- Already covered: none.

### P2 [D4b] — The per-run approval stamp store is hand-rolled in five providers plus the built-in gate, with documented behavioural drift between copies
- Where (all copies): `builtin_tool_gate.py:141-233` (`begin_turn`/`stamp_scope`/`stamped`/`_stamp_detail`), `mcp_tool_provider.py:612-766` (`apply_batch_decisions`/`stamped_decision`/`_stamped_decision_detail`/`stamp_scope`), `local_tool_provider.py:1232-1296`, `virtual_cli_provider.py:365-418` (`apply_batch_decisions`/`_pop_stamp`/`_pop_stamp_detail`/`stamp_scope`), `raw_shell_tool_provider.py:345-459` (same + `authority_generation`); the `_root_is_valid` twin at `virtual_cli_provider.py:594` / `local_tool_provider.py:2456` and `_kill_switch_engaged` ×4 belong to the same provider-mirror family.
- Evidence: `rg -n "^    def (stamp_scope|apply_batch_decisions|stamped|stamped_decision|_stamp_detail|_stamped_decision_detail|_pop_stamp|_pop_stamp_detail)\(" tldw_chatbook/Agents/*.py` → 20 defs across 6 files. Drift, by reading: (a) `stamp_scope` CLEARS the run's slice on entry in local/virtual_cli/raw_shell (local_tool_provider.py:1265 calls this "a deliberate divergence from a pure snapshot") but NOT in mcp/builtin_gate (mcp:751-766, builtin:198-211); (b) peek semantics (`stamped`) in builtin/local/mcp vs pop-on-read (`_pop_stamp`) in virtual_cli/raw_shell; (c) stamps keyed by tool NAME in local/mcp (`for name, verdict in decisions`) vs `row.call_id or row.tool_name` in virtual_cli/raw_shell; (d) raw_shell alone guards restore with `authority_generation` (raw_shell:442-459).
- Why it matters: four copies of a lock-guarded dict with three different clear/peek/key policies is exactly the place the next "child clobbered the parent's verdict" regression (PR2a Task 5's original bug, documented at builtin_tool_gate.py:100-109) re-enters; each fix so far (per-run keying, F1 peek, call-id keying, generation guard) landed in a subset of the copies.
- Recommended correction: one `RunScopedStamps` in `Agents/approval_provenance.py` (already the shared home of `ApprovalStamp`/`approval_stamp`/`approval_key_unanswered`, 42 lines): `replace_run(run_id, decisions, *, key=...)`, `peek(run_id, key)`, `pop(run_id, key, fallback=None)`, `scope(run_id, *, clear_on_enter: bool)`, `clear_run(run_id)`, optional generation guard. Providers keep their public method names as one-line delegations so `Tests/Agents/test_gate_run_scoping.py` and the parity tests stay green.
- Size: M · ADR: yes — `032-local-agent-tool-permission-boundary.md` mandates the mirroring ("Mirrors MCPToolProvider's approval discipline"), not the copy; no ADR forbids a shared store. · Confidence: verified (reading; drift points cited by line)
- Pinning test: `Tests/Agents/test_gate_run_scoping.py` (per-run keying), `Tests/Agents/test_local_tool_provider.py::test_approve_once_stamp_does_not_persist` / `::test_persist_failure_does_not_block_execution`, `Tests/Agents/test_virtual_cli_provider.py:846` (all state the current per-provider behaviour as requirements — the consolidation must preserve each copy's policy, not unify it).
- Already covered: none.

### P3 [D4a] — `_write_spill` re-implements `Utils/atomic_file_ops.atomic_write_text`
- Where: `tldw_chatbook/Agents/local_tool_provider.py:456-471`.
- Evidence: helper at `Utils/atomic_file_ops.py:41-134` (mkstemp in the same dir → write → fsync → chmod → `os.replace`, cleanup on failure); `rg -ln "atomic_file_ops import|atomic_write_text" tldw_chatbook -g '*.py'` → 11 importers. The re-roll differs only in skipping `fsync` and in the helper's default `logger.debug(f"... to {file_path}")` (path in log; `privacy_safe_log=True` suppresses it).
- Why it matters: one more copy of the atomic-write dance to audit for the crash-safety and permission-widening properties the helper already encodes (`preserve_existing_mode`, `overwrite=False` no-clobber link).
- Recommended correction: `atomic_write_text(final, text, mode=0o600, privacy_safe_log=True)`; keep the `spill_dir.mkdir(mode=0o700)` line. Note the added per-spill `fsync` (spills are already rare: ≥32 KiB results only).
- Size: S · ADR: no · Confidence: verified (reading)
- Pinning test: `Tests/Agents/test_local_tool_provider.py` spill tests (name not enumerated; `rg -n "spill" Tests/Agents/test_local_tool_provider.py`) — they assert content/permissions, not the mechanism.
- Already covered: none. (The excerpt's `os_replace_no_atomic` row is retired below — the write IS atomic.)

### P3 [D4b] — `_now_iso`/`_utc_now` are re-declared in four places in this slice (five with `DB/AgentRuns_DB.py`); no drift reaches storage, but no shared helper exists
- Where: `agent_runtime.py:129` (`_utc_now → datetime`, feeds `safe_utc_timestamp` 134-144), `agent_service.py:1181` (`_now_iso → str`, used once at 7858), `agent_service.py:1185` (`_utc_now → datetime`, the `wall_clock` default), `run_log.py:978-981` (`_now_iso → str`, function-body import), `DB/AgentRuns_DB.py:145` (`_now_iso → str`).
- Evidence: every string form is `strftime("%Y-%m-%dT%H:%M:%S.%fZ")` (read at each site), so `AgentRuns` rows (`created_at`/`updated_at` via AgentRuns_DB `_now_iso`, step `created_at` via `safe_utc_timestamp`, context-trace `created_at` via agent_service `_now_iso`) carry ONE format — the excerpt's `strftime` rows (agent_runtime:142,144; agent_service:1182; run_log:981) confirm. `rg -n "^def (_now_iso|_utc_now)\(" tldw_chatbook -g '*.py'` → 25 definitions repo-wide; `rg "^def (utc_now|now_iso|...)" tldw_chatbook/Utils` → none.
- Why it matters: in-slice, none (formats agree). Repo-wide the SAME name `_utc_now` returns `datetime` in 3 modules and `str` in ~18 — a contract drift that will bite the first cross-module import (outside this slice; reported for the roll-up).
- Recommended correction: one `Utils/clock.py` (`utc_now() -> datetime`, `utc_iso_ms_z() -> str`) and import it from the four sites (run_log's function-body import becomes a module import — `datetime` is stdlib, ADR-097 does not apply).
- Size: S (slice) / L (repo-wide rename) · ADR: no · Confidence: verified (reading)
- Pinning test: none specific to format; `Tests/Agents/test_run_log_service_wiring.py::test_on_record_returns_the_assigned_record_number` pins the int return, not the timestamp.
- Already covered: none (the orchestrator's seed list flags these names; this is the in-slice disposition).

### P3 [D4b] — `_identity()` is byte-identical in three activation modules
- Where: `Agents/activation.py:24-29`, `MCP/activation.py:25-30`, `RAG_Search/activation.py:141-146`.
- Evidence: md5 of each 6-line body → `c1a6123a7ea437b426929ac5d8c97496` ×3. difflib: Agents↔MCP share 73/231 identical lines; only `_identity` is a whole-function match (`execution`, `guarded` differ; `_sources` is legitimately per-service).
- Why it matters: the (pid, thread, task) execution identity is the admission key `execution()` compares (`active[0] != identity`); three copies means a future change to the identity tuple (e.g. adding a loop id) must be made thrice or admission silently disagrees across subsystems.
- Recommended correction: export it from `Backup_Recovery/activation.py` (all three modules already import `execution_scope` from there) and delete the copies.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Backup_Recovery/test_activation_agents.py` (exercises admission; does not pin the helper's location).
- Already covered: none.

### P3 [D4b] — Two hand-rolled bounded `run_git` subprocess runners (three git runners counting `Workspaces/git_workspace._run_user_git`)
- Where: `Agents/agent_worktree_git.py:33-176` (`run_git(root, *args)` → bytes; hooksPath/gpgsign hardening; 32 MiB cap; 30 s; process-group kill; `OperationError` codes), `Tools/git_tool_impls.py:202-~330` (`run_git(argv, ...)` → `GitCommandResult`; argv allowlist; sanitized env; cap+timeout; own process group), and `Workspaces/git_workspace._run_user_git` (used by `agent_worktree.py:84-92`).
- Evidence: `rg -n "run_git\b|agent_worktree_git" tldw_chatbook -g '*.py'` → the Agents copy is consumed only by `agent_worktree_recovery.py`; both docstrings describe the same properties (sanitized env, DEVNULL stdin, bounded pipes read on threads, kill on cap/timeout).
- Why it matters: the pipe-reader/kill/reap logic (the part that goes wrong under a wedged textconv child) is duplicated with different reaping policies (`cleanup_unproven` latching exists only in the Agents copy).
- Recommended correction: give `Tools/git_tool_impls.run_git` an `extra_config: tuple[str, ...]` / `validate_argv: bool` parameter and have `agent_worktree_git.run_git` delegate, keeping its `OperationError` translation. Needs a short ADR because the Agents copy's `cleanup_unproven` latch is a recovery-boundary contract.
- Size: L · ADR: new · Confidence: verified (reading both)
- Pinning test: `Tests/Agents/test_agent_worktree_confirmed_recovery.py` (behavioural; would survive a delegation).
- Already covered: none.

### P3 [D4b] — `cap-with-marker` truncation micro-helper duplicated across Agents and Widgets/Console
- Where: `Agents/run_hooks.py:193-197` (`_truncate`, public alias `truncate_hook_text`), `Widgets/Console/console_feedback_comment_modal.py:32-36` (`_preview_text`), `Widgets/Console/console_selection.py:136-139` (`cap_quote`).
- Evidence: the three bodies are identical modulo the `(cap, marker)` constant pair (read all three).
- Why it matters: trivial today; listed because the excerpt flagged it and it is a real 3-copy helper, not a shape coincidence.
- Recommended correction: `Utils/text_utils.cap_with_marker(text, cap, marker)` if a fourth copy appears; not worth a PR alone.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Agents/test_run_hooks.py` (budget tests pin `HOOK_IO_BUDGET_CHARS` inclusive-of-marker semantics).
- Already covered: none.

### P3 [D4b] — `get_internal_prompt` boot-leg shim duplicated verbatim
- Where: `Agents/agent_service.py:213-235`, `Chat/console_chat_controller.py:526-548`.
- Evidence: both are the TASK-22213 lazy-import wrapper (same body, same docstring rationale), guarded by `Tests/Packaging/test_rag_boot_import_closure.py`.
- Why it matters: cross-package (Chat/Agents) copy of a 3-line shim whose only job is to keep `Internal_Prompts` off first paint; a third copy is inevitable.
- Recommended correction: `Internal_Prompts/lazy.py` (no catalog import at module scope) exporting `get_internal_prompt_lazy`, or `Utils/lazy_prompts.py`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Packaging/test_rag_boot_import_closure.py` (pins the boot closure, not the shim's location).
- Already covered: none.

### P3 [D3] — Module-scope `try/except` guarding an INTERNAL import in `run_webhooks.py`
- Where: `tldw_chatbook/Agents/run_webhooks.py:39-44` (`from ..Metrics.metrics_logger import log_counter` inside `try: ... except Exception:` with a no-op fallback).
- Evidence: `rg -n "Metrics\.metrics_logger import" tldw_chatbook -g '*.py' -g '!Third_Party/**' | wc -l` → 69 importers; 3 guarded (this, `RAG_Search/parallel_processor.py:31`, `TTS_Events/tts_events.py:2687`), 66 unguarded. `metrics_logger.py:9` imports `psutil`, which is a core dependency (`pyproject.toml` `[project] dependencies`, line 65). The guard is not lazy either — the import still executes at module load.
- Why it matters: the fallback can only fire when a core dependency is broken, in which case 66 other modules fail first; meanwhile a real regression in `metrics_logger` would be masked here as "metrics silently off". D3 "guarded-but-not-lazy import of a non-optional module".
- Recommended correction: plain `from ..Metrics.metrics_logger import log_counter`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: `Tests/Agents/test_run_webhooks.py` (delivery tests; none pins the guard).
- Already covered: none.

### P3 [D1] — `_retire_agent_worktree` failures are swallowed with a bare `pass` on two teardown paths
- Where: `tldw_chatbook/Agents/agent_service.py:5737-5744` (`run_child` finally) and `:5819-5822` (thread-start failure).
- Evidence: the callee (`_retire_agent_worktree` 4520-4546) only calls `registry.resolve_owner_for_name("fs_read")` → `_ensure_catalog_cache` (can raise `RuntimeError("tool catalog changed during cache build")`, tool_catalog.py:1990, or a provider `list_catalog` error) and then the locked pop. A swallowed raise leaves `LocalToolProvider._agent_roots[run_id]` and its `_path_specs_by_alias` entry admitted for the process lifetime — the exact leak M1 (4534-4541) fixed once. The adjacent `_set_terminal_status` wrap at 5748-5753 logs; these two do not.
- Why it matters: a dead child's worktree root stays routable for its run id; invisible without a log line.
- Recommended correction: `except Exception: logger.warning("could not retire agent worktree routing for run {}", run_id)` (matching 5750).
- Size: S · ADR: no · Confidence: verified (reading)
- Pinning test: none (`Tests/Agents/test_fleet_runtime.py` covers retire on the happy path).
- Already covered: none.

### P3 [D1] — `except BaseException: pass` ×6 around diagnostics on the webhook delivery worker
- Where: `tldw_chatbook/Agents/run_webhooks.py:252-300` (`_record_drop`, `_record_start_failure`, `_record_delivery_failure`).
- Evidence: read; each wraps a `logger.warning` or `log_counter` call.
- Why it matters: swallowing `KeyboardInterrupt`/`SystemExit` inside a logging call on a daemon thread is harmless in practice but wider than the "diagnostics are best effort" comment needs; `except Exception` is the correct width.
- Recommended correction: narrow to `except Exception`.
- Size: S · ADR: no · Confidence: verified
- Pinning test: none.
- Already covered: none.

## Candidate dispositions
| candidate (file:line pattern) | confirmed / retired (why) / unverified (check) |
|---|---|
| dup_shape: `replace_disclosed_names@agent_service.py:5345` (+7 widget handlers) | retired — a 2-line `clear()`+`update()`; the other 7 copies are Persona/Console widget button handlers outside this slice sharing only the shape |
| dup_shape: `_pop_stamp@virtual_cli:390`, `stamped_decision@mcp:659`, `_pop_stamp@raw_shell:419`, `stamped@builtin_tool_gate:213`, `stamped@local:1252` | confirmed → P2 [D4b] stamp store |
| dup_shape / dup_verbatim: `_identity@Agents/activation:24` ×3 | confirmed → P3 [D4b] (md5-identical) |
| dup_shape: `_truncate@run_hooks:193`, `_preview_text@console_feedback_comment_modal:32`, `cap_quote@console_selection:136` | confirmed → P3 [D4b] (identical modulo constants) |
| dup_verbatim: `get_internal_prompt@console_chat_controller:526` / `@agent_service:213` | confirmed → P3 [D4b] |
| dup_verbatim: `stamp_scope@virtual_cli:400` / `@local:1262` | confirmed → folded into P2 [D4b] stamp store |
| dup_verbatim: `_root_is_valid@virtual_cli:594` / `@local:2456` | confirmed → folded into P2 [D4b] (same provider-mirror family) |
| except_exception_pass: agent_runtime.py:1339, 1376, 1762 | retired — `on_step`/`on_trace_step`/`reserve_context_trace` observation callbacks; containment is documented at each site and the DB-side callee (`observe_step` 7766-7809) already logs its own failures |
| except_exception_pass: agent_service.py:4056, 4119, 4224, 4297 | retired — each wraps a pre-check READ (`db.get_run`) whose failure falls through to the write path that logs (4138-4145, 4284-4291) |
| except_exception_pass: agent_service.py:5743, 5821 | confirmed → P3 [D1] `_retire_agent_worktree` swallow |
| except_exception_pass: fleet_coordinator.py:315 | retired — `_on_reserve` observer after admission; documented "observer cannot unwind admission" |
| except_exception_pass: local_tool_provider.py:1832 | retired — fs_read ledger observation; documented "observation must never affect dispatch" (`_record_fs_read_observation` itself never raises) |
| except_exception_pass: mcp_tool_provider.py:1366, 1376 | retired — best-effort `coroutine.close()`/`future.cancel()` inside the never-raise error path |
| except_exception_pass: run_webhooks.py:255, 262, 272, 282, 292, 299 | confirmed → P3 [D1] (width, not data) |
| except_exception_return_per_file: agent_service.py (15) | retired — all read: injected-callable boundaries (`_budget_weighted_tokens` 1676/1684/1710 → flat count; `_app_config` 2499 → `{}` with warning; token-fit probes 2624/2668 → refuse; `_build_for_candidate` 7749 → `None`, traced by the loop at agent_runtime:1878-1895; `_notify_*`/`_revoke` → logged) |
| except_exception_return_per_file: agent_runtime.py (10) | retired — continuation validators (`_continuation_calls_match` 696, `persist_continuation` 1439, `transition_call` 1480, `expand_restore_history` 1491) return False → `continuation_error()` records a STEP_ERROR + RUN_ERROR; `projection_opt_in` 1183 → fail closed |
| except_exception_return_per_file: tool_catalog.py (8) | retired — `_deferred_names` 831 lists nothing; provider registration 1766/1810 fails closed; `_coerce_arguments` 2107 documented best-effort; `execution_policy_for` 2211 → bounded; `probe_initial_catalog` 2252 → discovery |
| except_exception_return_per_file: local_tool_provider.py (7), virtual_cli_provider.py (8), raw_shell_tool_provider.py (5), mcp_tool_provider.py (4), canvas_tool_provider.py (4), profile_tool_provider.py (3), run_log.py (1), run_webhooks.py (1), fleet_coordinator.py (1) | retired — all read (profile_tool_provider: the 3 sites only); every one is a fail-closed return at a provider protocol boundary (`invoke()` must never raise) or a documented config fallback (`run_log._setting` 149 → default, F3) |
| except_exception_return_per_file: agent_lesson_promotion.py, automatic_work_runtime.py, model_retry.py (1 each) | unverified — mechanical scan only; my regex (`except Exception...:\n\s+(pass\|return)`) found 0 immediate-return sites in these three, so the excerpt's rows match a wider shape. Check: `rg -n -U -A3 "except Exception" tldw_chatbook/Agents/{agent_lesson_promotion,automatic_work_runtime,model_retry}.py` |
| function_body_import_per_file (21 files) | retired — ADR `097-boot-budget-ratchets.md`; each read site carries the citation (agent_runtime:37-40, 391-396, 1093; agent_service:147, 7753; tool_catalog:76, 83; mcp_tool_provider:85; run_webhooks:338/401 lazy `httpx`/`coerce_bool_setting`); files not read in full: mechanical only |
| get_cli_setting_hot: builtin_tool_gate.py:797, :895; tool_catalog.py:1242 | retired — measured (`agents_probe_gates.py`): `get_cli_setting` ×1 median 4.7 µs; 8-gate loop 37.5 µs; `all_tool_gates()` 60.0 µs; `_off_tool_gate_status()` 53.9 µs; `BuiltinToolProvider()` 40.3 µs (n=200/50). None is on a tick/compose/per-query path (`__init__` once per run; `all_tool_gates` per hub-pane render) |
| id_keyed_dict: agent_runtime.py:2285, 2338, 2340, 2372, 2374, 2415, 2445, 2471, 2626, 2994 | retired — one dict, `call_trace`, created per loop iteration (2262) and keyed on the `calls` list built at 1939/1951/1667; `calls` is held for the whole iteration (last use 3195) and never reassigned inside it, so no keyed object can be collected while the dict is live; `preauthorized_call_ids` (2317) has the same lifetime. Nothing survives the iteration |
| id_keyed_dict: canvas_tool_provider.py:456 | retired — `self._authorities[id(a)] = weakref.ref(a)`; `authenticates_registration_authority` (459-474) dereferences and requires `reference() is authority`, so a reused id with a dead ref returns False. Dead entries are never pruned but bounded (one per `issue_registration_authority`, called once per registration) |
| inline_path_check_no_pv: agent_worktree_git.py:46 | retired — env-var hygiene (drop HOME/XDG_CONFIG_HOME/SYSTEMROOT/TMPDIR that resolve INSIDE the repo so repo-authored git config cannot be injected), not a user-path trust check; `Utils/path_validation.py` has no "must be outside base" predicate |
| legacy_markers_per_file (9) | not examined (comment markers) |
| lock_and_execute: local_tool_provider (4/12), raw_shell (1/1), tool_catalog (1/1), virtual_cli (1/1) | retired — `rg -n "\.execute\(\s*[\"']" tldw_chatbook/Agents` → only `recovery.py:80` (PRAGMA on a private read-only sqlite inside `closing()`); the counted `.execute(` are `workspace_executor.execute` (12), `runtime.execute`, `tool.execute(**args)`, `registry.execute` — no SQL under any of the locks |
| os_replace_no_atomic: local_tool_provider.py:466 | retired as a D1 (it IS mkstemp-in-dir + chmod + `os.replace`); confirmed as P3 [D4a] helper-ignored |
| raw_1024x1024: agent_worktree_git.py:14, project_instruction_resolver.py:174, run_hooks.py:49 | retired — no shared MiB constant exists anywhere (`rg "1024 \* 1024" tldw_chatbook -g '*.py'` → 311 literals); a P3 consistency row for 3 of 311 is noise |
| raw_mkdir: run_log.py:619, 633 | retired — both after `is_within(base/run_dir, root)` containment and under `acquire_storage(root)` |
| raw_mkdir: agent_worktree.py:126 | confirmed → P2 [D1] shared-tmp base |
| re_compile_in_def: run_log_search.py:245, 358 | retired — the pattern is the model's per-call argument (cannot hoist); `_looks_catastrophic` pre-screens; `re`'s internal 512-entry cache covers repeats |
| seed_name__clean_text: ask_user_questions.py:178 | retired — in-slice singleton; its `(value, *, required)` / raise-ValueError-for-Pydantic contract differs from the 6 other `_clean_text(value, fallback)` defs (Chat ×3, Library, ACP_Interop, Character_Chat — outside slice, noted for the roll-up) |
| seed_name__identity: activation.py:24 | confirmed → P3 [D4b] |
| seed_name__now_iso: agent_service.py:1181, run_log.py:978; seed_name__utc_now: agent_runtime.py:129, agent_service.py:1185; strftime: agent_runtime.py:142, 144, agent_service.py:1182, run_log.py:981 | confirmed → P3 [D4b]; one format reaches `AgentRuns` rows (verified by reading each site) |
| tempfile_no_secure: agent_worktree.py:262 | retired — `NamedTemporaryFile(delete=False)` (mkstemp-backed, 0600) with `unlink(missing_ok=True)` in `finally` (270-271) |
| tempfile_no_secure: agent_worktree_recovery.py:512 | retired — `NamedTemporaryFile` context manager (auto-delete) |
| tempfile_no_secure: local_tool_provider.py:461 | retired — `mkstemp(dir=spill_dir)` + `chmod 0o600` + `os.replace` |
| tempfile_no_secure: agent_worktree.py:70 | confirmed → P2 [D1] |
| try_import_guard: run_webhooks.py:39 (module) | confirmed → P3 [D3] |
| try_import_guard: agent_service.py:1673, 2495, 3730, 7709; local_tool_provider.py:2098; mcp_tool_provider.py:1323; run_log.py:145, 333, 700; tool_catalog.py:1229, 2102 | retired — function-body, ADR-097 lazy imports whose `except` also bounds the CALL that follows (accounting fallback, config fallback, webhook emit, candidate build, promotion refusal, automatic-work check, log-root resolution, legacy-dir migration, gate registration [logged with traceback 1258], arg repair) |
| try_import_guard: persona_policy.py:67 | unverified — mechanical only. Check: `sed -n 55,80p tldw_chatbook/Agents/persona_policy.py` |

## Verified-fine
- **`call_trace[id(call)]` lifetime** (the brief's check 1): see the id_keyed_dict row — per-iteration dict, keyed objects held by `calls` for the dict's entire life; no id reuse window.
- **Timestamp formats reaching `AgentRuns_DB`** (check 2/`_now_iso` vs `_utc_now`): all string forms are `%Y-%m-%dT%H:%M:%S.%fZ`; `agent_service._utc_now` is only the `wall_clock` default consumed through `safe_utc_timestamp`; `agent_service._now_iso` is used once (7858) for context-trace `created_at`. Same format as `DB/AgentRuns_DB._now_iso` (145). No drift in the rows.
- **Webhooks and `Utils/egress.py`** (check 2): `run_webhooks.deliver_webhook` calls `check_url_or_raise_async(config.url)` (443) before any POST and refuses on `EgressBlockedError`; `rg -n "httpx\.|requests\.|urlopen" tldw_chatbook/Agents` → the only network call in the package is `run_webhooks.py:403`. Payload is identifiers-only (350-377); secret-less config refuses (432-438); origin-only logging (386-395). Not a task-586/609 gap.
- **Run persistence swallow audit** (check 5): there is no `upsert_run` in this code; terminal truth is written by `_set_terminal_status` (4207-4329): pre-check read → `set_terminal_with_step` ×2 attempts → diagnostic row → `set_status` fallback, every failure logged with `error_type`. Step writes (`observe_step` 7766-7807, `observe_trace_step` 7811-7850, `_persist` 4606-4614) log and write a `capture_failed` diagnostic. Fleet child terminal fallback: `run_child` finally 5745-5753. The only gap found is the P2 read-exception finding above.
- **Review-hook raise fails OPEN for non-continuation batches** (agent_runtime.py:2354-2368, logged at WARNING): mitigated by design — every provider re-gates at `invoke()` (builtin_tool_gate.check_detailed 427-492 fails closed on "ask" without a stamp; mcp `_invoke_locked` 957-1051 falls back to the single-call card or `DENY_REFUSAL`; local `_verdict_for` 2484-2623; virtual_cli 433-474; raw_shell 491-516), and the in-loop runtime tools each carry their own confirm card (install_skill, run_skill_script, fork/new_chat, merge/discard worktree). Pinned by `Tests/Agents/test_agent_runtime_preparation.py::test_raised_review_hook_still_runs_dispatch_gate_before_fail_open_dispatch` (states the behaviour as a requirement).
- **Kill-switch fail direction differs across providers** (mcp `_kill_switch_engaged` 1105-1122 and builtin_tool_gate `_kill_switch` 314-325 → False on read failure; local 2465-2482 and virtual_cli 608-612 → True): documented at local_tool_provider.py:2466-2477 — the controller's composition closure swallows `get_kill_switch` errors and returns False for all four in production, so the provider-level branch is a protocol boundary only. Documented divergence; not filed.
- **`stamp_scope` clear-on-entry divergence** is documented (local_tool_provider.py:1265-1270) — reported inside the P2 stamp-store finding as drift, not as a bug.
- **`agent_runtime` purity**: no Textual/DB/I-O imports at module scope (docstring claim holds; `render_tool_protocol` and the loop's helpers import lazily with ADR-097 citations).
- **`LoopDeps` positional-slot discipline**: every new field is appended (622-660) — the comments' contract holds by reading.
- **`_call_with_timeout`** (agent_service.py:1887-2011): per-call daemon thread, `_CANCEL_POLL_SECONDS` slicing, ADR-067 deadline re-arm while `human_input_wait_active`; `operation.finish()` in `finally` on both the runner and the failed-start path.
- **Thread-safety of provider stamp dicts**: every access under `_stamps_lock`/`_decisions_lock`; no lock held across `yield` or across a call into `self._service` (asserted by comment at builtin_tool_gate.py:132-138 and verified at every site read).
- **`_GATEABLE_BUILTINS` title/blurb copy** — registration contract per brief; not flagged.
- **`config.py:8447` dotted `get_cli_setting`** — not flagged per brief; the one dotted read here is `agent_service.py:7718` (`f"api_settings.{candidate_endpoint}"`).

## Retired
- **"`_persist` may leave the row `running` forever"** — narrowed: `AgentRunsDB.reconcile_orphaned_runs` (DB/AgentRuns_DB.py:2340, auto-called from `__init__` once per path per process) flips it to `error` on the NEXT launch. Kept as P2 for the in-session window, not P1.
- **`_tool_protocol_cache` dict written from fleet-child threads** (`_build_model_request` 2528-2532): CPython dict set/get under the GIL; worst case a duplicate render; cleared per `run_turn` (8561). Not a finding.
- **`_primary_cut_suppress` unlocked set** (2208): single writer (the run's own thread), racing readers poll again — documented at 2201-2207.
- **`local_tool_provider.py` "4 locks + 12 `.execute(`"** — see the lock_and_execute row; no SQL exists in the file.
- **`agent_worktree_git.py:46` as a path-validation bypass** — see the inline_path_check row.
- **`run_log_search` per-call `re.compile`** — see the re_compile row.
- **`get_cli_setting` in `_GATEABLE_BUILTINS` loops** — measured; see the get_cli_setting_hot row.

## Left UNVERIFIED
| claim | why not verified | literal command to run |
|---|---|---|
| A pre-existing `/tmp/tldw_agent_worktrees` (or symlink) owned by another local user redirects agent checkouts | needs a second OS user on this host; no test fixture models it | `sudo -u <other> ln -s /some/dir /tmp/tldw_agent_worktrees; cd $WT && source $SCRATCH/env.sh && $PY - <<'EOF'\nfrom pathlib import Path\nfrom tldw_chatbook.Agents.agent_worktree import create_agent_worktree\nprint(create_agent_worktree(Path("$WT"), "probe-run-1"))\nEOF` then `readlink -f /tmp/tldw_agent_worktrees` |
| The `_persist` read gap fires under REAL sqlite contention (not a stub) | reproducing a 5 s `BEGIN IMMEDIATE` hold exactly across `_persist`'s `get_run` needs a second process pinned to the timing; out of scope for a read-only pass | `cd $WT && source $SCRATCH/env.sh && $PY -m pytest Tests/Agents/test_agent_service.py -q` after adding a test that monkeypatches `AgentRunsDB.get_run` to raise `sqlite3.OperationalError` once inside `_persist` and asserts `db.get_run(run_id)["status"] != "running"` |
| `reconcile_orphaned_runs` actually repairs a row stranded by the gap on the next open | read from the docstring only | `cd $WT && source $SCRATCH/env.sh && $PY - <<'EOF'\nfrom tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB\np="$SCRATCH/home/runs-probe.db"\ndb=AgentRunsDB(p); rid=db.create_run(conversation_id="c", agent_kind="primary"); db.close()\ndb2=AgentRunsDB(p); print(db2.get_run(rid)["status"])\nEOF` (expect `error`) |
| The 3 `except_exception_return` rows in agent_lesson_promotion / automatic_work_runtime / model_retry and `persona_policy.py:67` | mechanical scan only (see dispositions) | commands given in the dispositions table |
