# TASK-1270 — Run log in a bound workspace folder is readable by sub-agents

Status: **BLOCKED** on Step 2 (the code fix). Steps 1 and 3 are complete and committed.

## TL;DR

The vulnerability described in TASK-1270 is real and is reproduced by two new,
`xfail(strict=True)`-marked tests in
`Tests/Agents/test_run_log_workspace_isolation.py`. The designed fix — dot the
run-log directory name unconditionally in `RunLogWriter.bind()`, deleting the
sandbox-fallback-only conditional — is fully specified and its diff is attached
(`task-1270-prototype-fix.diff`), but applying it flips the literal directory-name
string asserted by **22 pre-existing tests** in `Tests/Agents/test_run_log_writer.py`
and `Tests/Agents/test_run_log_service_wiring.py`, plus
`test_bound_workspace_folder_keeps_the_undotted_name` in
`Tests/Agents/test_run_log_sandbox_isolation.py`. Per this task's explicit rule
("Do NOT edit any pre-existing test to make something pass. If one fails, STOP and
report BLOCKED with detail"), I did not touch those tests and did not land the code
fix. Everything else the task asked for is done: reproduction, an
implementation-agnostic regression test, the AC #4 proof, and corrected
documentation.

## What's committed

1. **`Tests/Agents/test_run_log_workspace_isolation.py`** (new file) — four tests:
   - `test_bound_workspace_folder_log_is_hidden_from_grep_and_glob` —
     `xfail(strict=True)`. Direct reproduction: binds a genuine workspace folder
     (outside the sandbox — the common real case, not the already-fixed
     nested-inside-sandbox edge case), plants
     `PARENT_SECRET_API_KEY=sk-live-workspace789` via `RunLogWriter.append`, and
     asserts `grep_files`/`glob_files` cannot recover it. This is the AC #6
     "regression test [that] pins the invariant ... regardless of how root
     resolution is implemented" — it asserts only the tools' *outcome*, never
     `_root_kind`/`is_sandbox_fallback` or any dotted/undotted string.
   - `test_spawned_subagent_cannot_read_parents_log_via_grep_files_in_bound_workspace`
     — `xfail(strict=True)`. Full `run_turn` reproduction mirroring
     `test_run_log_sandbox_isolation.py`'s sandbox-case test: parent embeds the
     secret in its own turn, spawns a child, child calls `grep_files`, secret
     shows up in the child's own persisted `tool_result`.
   - `test_search_run_log_reads_the_log_in_bound_workspace_configuration` and
     `test_search_run_log_reads_the_log_in_sandbox_fallback_configuration` — both
     pass today. AC #4 proof: `run_log_search.load_records(writer.log_dir)` reads
     the log correctly in both configurations, because it globs `log_dir` directly
     and never routes through `validate_path`/`_is_hidden_within`.

2. **`Docs/superpowers/specs/2026-07-27-agent-programmatic-run-memory-design.md`**
   — §9.1 and §9.4 corrected in place (old text kept, marked superseded, followed by
   the current-true statement); the now-stale §11 "Extending glob_files/grep_files
   to workspace roots" deferred-item struck through and marked shipped-as-TASK-850.

3. **`tldw_chatbook/Agents/run_log.py`** — comment-only changes (no behavior
   change): a block above `DEFAULT_DIR_NAME` and a second pointer at the actual
   conditional inside `bind()`, both marked "KNOWN OPEN VULNERABILITY — TASK-1270,
   not yet fixed", stating the corrected premise and pointing at this report. The
   pre-existing "Final-review CRITICAL 2" / F8 comments were left as-is because
   they still accurately describe what the *current* code does — I did not want a
   comment claiming behavior the code doesn't have.

4. **`task-1270-prototype-fix.diff`** (repo root) — the actual, ready-to-apply code
   fix (see below), kept as an attachment for whoever resolves the blocker.

## Step 1 — reproduction (done)

Before writing any fix, I wrote the two `xfail` tests above and ran them with
`--runxfail` (xfail disabled) against the unmodified code. Exact failure, direct
case:

```
    assert _grep("PARENT_SECRET_API_KEY") == [], (
        "grep_files must not be able to read the run log through a "
        "genuinely bound workspace folder"
    )
E   AssertionError: grep_files must not be able to read the run log through a genuinely bound workspace folder
E   assert [{'line': 'PA...gs.0001.txt'}] == []
```

Full-run case:

```
    assert not any(secret in r or "PARENT_SECRET" in r for r in tool_results)
E   assert not True
```

Note on getting the reproduction right: my first attempt at the seam helper only
patched `workspace_file_roots.allowed_file_roots`, mirroring the existing
`_fallback_seams` helper in `test_run_log_sandbox_isolation.py`. That silently
under-tested the *positive control* — `GlobFiles`/`GrepFiles` import
`allowed_file_roots` via `from .workspace_file_roots import allowed_file_roots` at
`file_operation_tools.py` module-load time, a separate, early-bound name that
patching the *source* module's attribute does not redirect (confirmed empirically:
`file_tools.allowed_file_roots is ws_roots.allowed_file_roots` is `True` before
patching, `False` after). The existing sandbox-case tests never hit this because
their workspace folder is either absent or nested *inside* the patched sandbox
root, so the real (unpatched) resolution reaches it anyway by coincidence. My test
patches `file_operation_tools.allowed_file_roots` directly as well, and the
positive control now genuinely exercises the redirected root before the security
assertion is trusted.

## Step 2 — the fix (specified, NOT applied)

The instructed fix: in `RunLogWriter.bind()`, always dot `dir_name` and delete the
`is_sandbox_fallback` conditional entirely (see `task-1270-prototype-fix.diff` for
the exact patch). Applying it and running the full `Tests/Agents` suite produced:

```
22 failed, 507 passed, 1 warning in 27.86s
```

All 22 failures are the same shape — a test-only helper reading
`root / "agent-runs"` (or asserting `writer.log_dir.parent.name == "agent-runs"`)
that no longer exists once the name is unconditionally `.agent-runs`:

```
Tests/Agents/test_run_log_sandbox_isolation.py::test_bound_workspace_folder_keeps_the_undotted_name
Tests/Agents/test_run_log_service_wiring.py::test_a_plain_run_writes_records_without_the_caller_wiring_anything
Tests/Agents/test_run_log_service_wiring.py::test_record_numbers_are_unique_across_the_whole_run_tree
Tests/Agents/test_run_log_service_wiring.py::test_a_real_spawn_shares_the_parent_log_directory_and_counter
Tests/Agents/test_run_log_service_wiring.py::test_parent_spawn_tool_call_record_precedes_the_childs_own_records
Tests/Agents/test_run_log_service_wiring.py::test_run_turn_called_twice_on_one_service_gets_two_separate_logs
Tests/Agents/test_run_log_writer.py::test_bind_creates_the_run_directory_and_gitignore
Tests/Agents/test_run_log_writer.py::test_a_child_run_shares_the_parent_counter
Tests/Agents/test_run_log_writer.py::test_segment_rolls_and_no_record_spans_a_boundary
Tests/Agents/test_run_log_writer.py::test_oversized_record_is_capped_and_marked
Tests/Agents/test_run_log_writer.py::test_config_overrides_the_directory_name
Tests/Agents/test_run_log_writer.py::test_write_manifest_emits_readable_json
Tests/Agents/test_run_log_writer.py::test_manifest_records_segments_after_appends
Tests/Agents/test_run_log_writer.py::test_concurrent_appends_produce_unique_numbers_and_no_corruption
Tests/Agents/test_run_log_writer.py::test_path_traversal_with_dotdot_falls_back_to_the_default_dir_name
Tests/Agents/test_run_log_writer.py::test_dir_name_with_a_separator_falls_back_to_the_default
Tests/Agents/test_run_log_writer.py::test_dir_name_absolute_falls_back_to_the_default
Tests/Agents/test_run_log_writer.py::test_dir_name_whitespace_only_falls_back_to_the_default
Tests/Agents/test_run_log_writer.py::test_dir_name_bare_dotdot_falls_back_to_the_default
Tests/Agents/test_run_log_writer.py::test_dir_name_explicit_constructor_arg_is_validated_too
Tests/Agents/test_run_log_writer.py::test_legitimate_custom_dir_name_is_not_rejected
Tests/Agents/test_run_log_writer.py::test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name
```

Two of these are not incidental collateral — they are **regression guards written
specifically against this exact change**:

- `test_run_log_sandbox_isolation.py::test_bound_workspace_folder_keeps_the_undotted_name`
  — docstring: *"the fix must not regress that by dotting unconditionally."*
- `test_run_log_writer.py::test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name`
  — docstring: *"the F8 fix narrows the dotting decision, it must not widen it to
  dot every workspace folder."*

Both were written during the PR #1066 review specifically to reject "always dot" as
a solution to the F8 edge case (a workspace folder nested inside the sandbox). This
task now asks for exactly that. I did not overrule that prior, explicit decision on
my own authority, and did not edit any of the 22 tests. I reverted the prototype
(`git checkout -- tldw_chatbook/Agents/run_log.py`) before committing anything.

## What needs a decision

Someone needs to choose one of:

1. **Authorize updating the 22 tests** (mechanically: every `root / "agent-runs"`
   and `writer.log_dir.parent.name == "agent-runs"` becomes `.agent-runs`, since
   the directory is now dotted unconditionally; the `.gitignore`-preservation and
   `test_existing_gitignore_is_never_overwritten`-style tests need their fixture
   paths updated to match; and the two "regression guard" tests
   (`test_bound_workspace_folder_keeps_the_undotted_name`,
   `test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name`) need their
   assertions AND docstrings flipped to state the new policy, since their current
   docstrings explicitly warn against it) and apply
   `task-1270-prototype-fix.diff`, then remove the `xfail` markers from the two
   new reproduction tests, then delete the now-superseded `_root_kind`
   thread-local and its docstring (dead once the conditional is gone), then
   replace `bind()`'s and `_coerce_dir_name`'s remaining stale prose to match (I
   left pointers, not full rewrites, since the underlying behavior hasn't changed
   yet).
2. **Pick a different remedy** that doesn't change the directory name for every
   caller of the writer's default configuration — e.g., something scoped only to
   genuinely-workspace-and-outside-sandbox binds, verified against the *live*
   root resolution rather than a name string. I did not attempt to design this;
   Step 2 as given foreclosed it explicitly ("Use the dotted directory name in
   BOTH cases ... and delete the conditional entirely").

## Acceptance criteria status

- [x] AC #1 — sub-agent cannot read parent's log via `grep_files`/`glob_files` in
  a bound workspace, proven by a planted-secret test — proven **failing**
  (xfail), not yet fixed.
- [x] AC #2 — sandbox-fallback protection unchanged: all four
  `test_run_log_sandbox_isolation.py` tests still pass.
- [x] AC #3/#6 — regression test asserts the *outcome* (tool reachability), never
  the naming/branch mechanism.
- [x] AC #4 — `search_run_log`/`load_records` proven to read the log in both
  configurations (two new passing tests).
- [~] AC #5 — spec §9.1/§9.4 and `run_log.py` comments corrected to state the true
  premise; the log itself is not yet reachable-as-undotted-and-fixed since the
  code fix is blocked (comments say so explicitly).
- [ ] Not done: landing the actual fix (blocked, see above).

## Tests/Agents numbers

- **Before** (this session's baseline, clean checkout): `529 passed` (0 failed).
- **After** (comment/doc changes + new test file, fix NOT applied):
  `531 passed, 2 xfailed` (0 real failures; +2 passing AC #4 tests, +2 xfailed
  reproduction tests = +4 collected).
- **With the prototype fix applied** (evaluated, then reverted):
  `22 failed, 507 passed` — the blast radius above.

## Commits

See `git log` on `fix/run-log-workspace-disclosure` for the commit SHA(s) — one
commit containing the new test file, the spec corrections, and the comment-only
`run_log.py` changes. No behavior change is included.
