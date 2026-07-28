"""Writer: binding, segmentation, durability, degradation."""

from pathlib import Path

import pytest

from tldw_chatbook.Agents import run_log as run_log_module
from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Agents.run_log_format import iter_records


@pytest.fixture
def root(tmp_path, monkeypatch):
    """Pin the writer's resolved root to a temp dir."""
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: tmp_path)
    return tmp_path


def make(root_dir, **kw):
    writer = RunLogWriter(**kw)
    writer.bind("run-abc")
    return writer


def test_unbound_writer_writes_nothing(root):
    writer = RunLogWriter()
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None
    assert not (root / "agent-runs").exists()


def test_bind_creates_the_run_directory_and_gitignore(root):
    make(root)
    assert (root / "agent-runs" / "run-abc").is_dir()
    assert (root / "agent-runs" / ".gitignore").read_text() == "*\n"


def test_existing_gitignore_is_never_overwritten(root):
    (root / "agent-runs").mkdir()
    (root / "agent-runs" / ".gitignore").write_text("keep me\n")
    make(root)
    assert (root / "agent-runs" / ".gitignore").read_text() == "keep me\n"


def test_records_are_numbered_monotonically_from_one(root):
    writer = make(root)
    numbers = [
        writer.append(run_id="r", kind="primary", type="model", content=str(i))
        for i in range(3)
    ]
    assert numbers == [1, 2, 3]


def test_a_child_run_shares_the_parent_counter(root):
    writer = make(root)
    writer.append(run_id="parent", kind="primary", type="model", content="a")
    writer.append(run_id="child", kind="subagent", type="model", content="b")
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    parsed = list(iter_records(data))
    assert [(p.number, p.run_id) for p in parsed] == [(1, "parent"), (2, "child")]


def test_second_bind_is_ignored(root):
    writer = make(root)
    writer.bind("run-other")
    assert writer.log_dir.name == "run-abc"


def test_segment_rolls_and_no_record_spans_a_boundary(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    run_dir = root / "agent-runs" / "run-abc"
    segments = sorted(run_dir.glob("logs.*.txt"))
    assert len(segments) > 1
    # Every segment parses standalone: no record straddles a boundary.
    total = 0
    for segment in segments:
        parsed = list(iter_records(segment.read_bytes()))
        assert parsed, f"{segment.name} parsed to nothing"
        total += len(parsed)
    assert total == 6


def test_oversized_record_is_capped_and_marked(root):
    writer = make(root, max_record_bytes=50)
    writer.append(run_id="r", kind="primary", type="tool_result", content="y" * 500)
    data = (root / "agent-runs" / "run-abc" / "logs.0001.txt").read_bytes()
    (parsed,) = list(iter_records(data))
    assert len(parsed.content.encode()) <= 50
    assert parsed.truncated_from == 500


def test_unresolvable_root_deactivates_the_writer(monkeypatch):
    monkeypatch.setattr(run_log_module, "resolve_log_root", lambda: None)
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.append(run_id="r", kind="primary", type="model", content="x") is None


def test_write_failure_deactivates_rather_than_raising(root, monkeypatch):
    writer = make(root)
    assert writer.append(run_id="r", kind="primary", type="model", content="a") == 1

    def boom(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(writer, "_write_bytes", boom)
    assert writer.append(run_id="r", kind="primary", type="model", content="b") is None
    assert writer.is_active is False


def test_config_can_disable_logging_entirely(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: False if key == "run_log_enabled" else default
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert not (root / "agent-runs").exists()


def test_config_overrides_the_directory_name(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "my-logs" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert (root / "my-logs" / "run-abc").is_dir()


def test_write_manifest_emits_readable_json(root):
    writer = make(root)
    writer.write_manifest({"run_id": "run-abc", "status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert manifest["status"] == "done"
    assert manifest["segments"] == []  # nothing appended yet


def test_manifest_records_segments_after_appends(root):
    writer = make(root, segment_bytes=400)
    for _ in range(6):
        writer.append(run_id="r", kind="primary", type="model", content="x" * 100)
    writer.write_manifest({"status": "done"})
    import json

    manifest = json.loads((root / "agent-runs" / "run-abc" / "MANIFEST").read_text())
    assert len(manifest["segments"]) > 1
    assert manifest["record_count"] == 6


def test_manifest_failure_never_raises(root, monkeypatch):
    writer = make(root)
    monkeypatch.setattr(writer, "_write_bytes", lambda *a, **k: (_ for _ in ()).throw(OSError))
    writer.write_manifest({"status": "done"})  # must not raise


def test_close_is_safe_to_call_twice_and_on_an_inactive_writer(root):
    writer = make(root)
    writer.close()
    writer.close()
    RunLogWriter().close()


def test_concurrent_appends_produce_unique_numbers_and_no_corruption(root):
    import threading

    writer = make(root)

    def worker(index):
        for _ in range(20):
            writer.append(
                run_id=f"r{index}", kind="primary", type="model", content=f"payload-{index}"
            )

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    records = []
    for segment in sorted((root / "agent-runs" / "run-abc").glob("logs.*.txt")):
        records.extend(iter_records(segment.read_bytes()))
    numbers = sorted(r.number for r in records)
    assert numbers == list(range(1, 81))


def test_non_numeric_segment_bytes_uses_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "not-a-number" if key == "run_log_segment_bytes" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer._segment_bytes == 4_000_000  # default


def test_negative_max_record_bytes_uses_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: -999 if key == "run_log_max_record_bytes" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer._max_record_bytes == 1_000_000  # default


def test_bind_idempotent_after_failed_first_bind(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: False if key == "run_log_enabled" else default
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert writer.log_dir is None

    # Second bind with different run_id must not activate or create directory
    monkeypatch.setattr(
        run_log_module, "_setting", lambda key, default: True if key == "run_log_enabled" else default
    )
    writer.bind("run-other")
    assert writer.is_active is False
    assert writer.log_dir is None
    assert not (root / "agent-runs" / "run-other").exists()


def test_path_traversal_with_dotdot_falls_back_to_the_default_dir_name(root, monkeypatch):
    """F1 (Qodo #1, PR #1066 review ruling): CHANGED BEHAVIOR, deliberately.

    Before this fix, a `..`-bearing dir_name reached `bind()` unvalidated
    and was only caught by the SECOND-LAYER `is_within` containment check,
    which disabled logging entirely (is_active stayed False) -- this test
    used to pin exactly that. The ruling on Qodo finding #1 requires
    validating the dir_name COMPONENT itself up front and falling back to
    the safe default (logged at warning) instead: a bad *config* value
    should not be able to silently kill a crash-durability feature. The
    security property this test guards -- `..` can NEVER walk the log
    outside its root -- still holds; only the AVAILABILITY outcome changed,
    from "logging disabled" to "logging continues under 'agent-runs'".
    `bind()`'s `is_within` check remains as defense in depth underneath.
    """
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "../escape" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"
    assert not (root / "escape").exists()
    assert not (root.parent / "escape").exists()


def test_dir_name_with_a_separator_falls_back_to_the_default(root, monkeypatch):
    """F1: a dir_name containing a separator (but no literal `..`) must be
    rejected too -- it is still an untrusted value joined into a path, and
    a nested path could otherwise create unexpected intermediate
    directories even when it stays within root."""
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "sub/dir" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"
    assert not (root / "sub").exists()


def test_dir_name_absolute_falls_back_to_the_default(root, monkeypatch):
    """F1: pathlib's `/` operator REPLACES the whole path when the
    right-hand side is absolute (`Path("/root") / "/etc" == Path("/etc")`),
    so an absolute dir_name is uniquely dangerous -- it does not merely
    nest somewhere unexpected, it can retarget the log tree entirely."""
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "/etc/evil" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"


def test_dir_name_whitespace_only_falls_back_to_the_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "   " if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"


def test_dir_name_bare_dotdot_falls_back_to_the_default(root, monkeypatch):
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: ".." if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"


def test_dir_name_explicit_constructor_arg_is_validated_too(root):
    """F1: the SAME validation applies to an explicit constructor arg, not
    only the config-sourced value -- both go through `_coerce_dir_name`."""
    writer = RunLogWriter(dir_name="../escape")
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "agent-runs" / "run-abc"


def test_legitimate_custom_dir_name_is_not_rejected(root, monkeypatch):
    """A safe, single-component custom name (no separators, no traversal,
    not absolute) must pass through unchanged -- the validation must not
    be so aggressive it breaks the existing `run_log_dir_name` override."""
    monkeypatch.setattr(
        run_log_module,
        "_setting",
        lambda key, default: "my-logs" if key == "run_log_dir_name" else default,
    )
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is True
    assert writer.log_dir == root / "my-logs" / "run-abc"


def test_non_numeric_segment_bytes_explicit_arg_uses_default():
    """Test explicit segment_bytes="not-a-number" uses default."""
    writer = RunLogWriter(segment_bytes="not-a-number")
    assert writer._segment_bytes == 4_000_000  # default


def test_zero_segment_bytes_explicit_arg_uses_default():
    """Test explicit segment_bytes=0 uses default."""
    writer = RunLogWriter(segment_bytes=0)
    assert writer._segment_bytes == 4_000_000  # default


def test_negative_segment_bytes_explicit_arg_uses_default():
    """Test explicit segment_bytes=-999 uses default."""
    writer = RunLogWriter(segment_bytes=-999)
    assert writer._segment_bytes == 4_000_000  # default


def test_non_numeric_max_record_bytes_explicit_arg_uses_default():
    """Test explicit max_record_bytes="not-a-number" uses default."""
    writer = RunLogWriter(max_record_bytes="not-a-number")
    assert writer._max_record_bytes == 1_000_000  # default


def test_zero_max_record_bytes_explicit_arg_uses_default():
    """Test explicit max_record_bytes=0 uses default."""
    writer = RunLogWriter(max_record_bytes=0)
    assert writer._max_record_bytes == 1_000_000  # default


def test_negative_max_record_bytes_explicit_arg_uses_default():
    """Test explicit max_record_bytes=-5 uses default."""
    writer = RunLogWriter(max_record_bytes=-5)
    assert writer._max_record_bytes == 1_000_000  # default


def test_legitimate_explicit_args_still_work():
    """Test that valid explicit args (e.g., 400, 50) continue to work."""
    writer = RunLogWriter(segment_bytes=400, max_record_bytes=50)
    assert writer._segment_bytes == 400
    assert writer._max_record_bytes == 50


def test_real_resolve_log_root_prefers_workspace_over_sandbox(monkeypatch):
    """Test real resolve_log_root() calls and prefers workspace folder over sandbox root."""
    from pathlib import Path

    tmp_sandbox = Path("/tmp/sandbox")
    tmp_workspace = Path("/tmp/workspace")

    def mock_tool_sandbox_root():
        return tmp_sandbox

    def mock_allowed_file_roots(write=False, sandbox_root=None):
        # Return (sandbox, workspace) tuple; resolve_log_root should prefer workspace
        return [tmp_sandbox, tmp_workspace]

    # Stub the underlying imports that resolve_log_root uses
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", mock_tool_sandbox_root)
    monkeypatch.setattr(ws_roots, "allowed_file_roots", mock_allowed_file_roots)

    # Call the REAL resolve_log_root() with stubbed seams
    result = run_log_module.resolve_log_root()
    assert result == tmp_workspace  # Prefers workspace over sandbox


def test_real_resolve_log_root_falls_back_to_sandbox_when_no_workspace(monkeypatch):
    """Test real resolve_log_root() falls back to sandbox root when no workspace folder bound."""
    from pathlib import Path

    tmp_sandbox = Path("/tmp/sandbox")

    def mock_tool_sandbox_root():
        return tmp_sandbox

    def mock_allowed_file_roots(write=False, sandbox_root=None):
        # Return only sandbox; no workspace folders bound
        return [tmp_sandbox]

    # Stub the underlying imports
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", mock_tool_sandbox_root)
    monkeypatch.setattr(ws_roots, "allowed_file_roots", mock_allowed_file_roots)

    # Call the REAL resolve_log_root()
    result = run_log_module.resolve_log_root()
    assert result == tmp_sandbox  # Falls back to sandbox


def test_real_resolve_log_root_returns_none_on_exception(monkeypatch):
    """Test real resolve_log_root() returns None when underlying call raises."""
    from pathlib import Path

    tmp_sandbox = Path("/tmp/sandbox")

    def mock_tool_sandbox_root():
        return tmp_sandbox

    def mock_allowed_file_roots_raises(write=False, sandbox_root=None):
        raise RuntimeError("cannot access workspace roots")

    # Stub the underlying imports
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", mock_tool_sandbox_root)
    monkeypatch.setattr(ws_roots, "allowed_file_roots", mock_allowed_file_roots_raises)

    # Call the REAL resolve_log_root(); it catches and logs, returns None
    result = run_log_module.resolve_log_root()
    assert result is None  # Returns None (logging off) on exception


# -- F3 (Qodo #3): env vars -> config.toml -> defaults -----------------------


def test_setting_env_var_overrides_a_conflicting_toml_value(monkeypatch):
    """CLAUDE.md's documented priority is "env vars -> config.toml ->
    defaults"; `_setting` previously never consulted the environment at all,
    silently skipping the highest-priority tier. `TLDW_AGENTS_<KEY>` (this
    repo's existing per-setting override convention -- see
    `TLDW_CONSOLE_LLAMA_CPP_BASE_URL` in UI/Screens/chat_screen.py) must win
    over a conflicting TOML-backed value.
    """
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(config_module, "get_cli_setting", lambda *a, **k: "toml-value")
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", "env-value")
    assert run_log_module._setting("run_log_dir_name", "default-value") == "env-value"


def test_setting_falls_back_to_toml_when_env_unset(monkeypatch):
    import tldw_chatbook.config as config_module

    monkeypatch.delenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", raising=False)
    monkeypatch.setattr(config_module, "get_cli_setting", lambda *a, **k: "toml-value")
    assert run_log_module._setting("run_log_dir_name", "default-value") == "toml-value"


def test_setting_env_var_blank_string_does_not_count_as_set(monkeypatch):
    """An env var present but empty must not shadow a real TOML value --
    same as unset, not "set to empty"."""
    import tldw_chatbook.config as config_module

    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", "")
    monkeypatch.setattr(config_module, "get_cli_setting", lambda *a, **k: "toml-value")
    assert run_log_module._setting("run_log_dir_name", "default-value") == "toml-value"


def test_setting_boolean_env_var_is_parsed_not_just_truthy(monkeypatch):
    """`run_log_enabled`'s default is a Python bool; the raw env STRING
    "false" is truthy in Python (`bool("false") is True`), so a naive
    `_setting(...)` return would have silently failed to disable logging.
    """
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(config_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "false")
    assert run_log_module._setting("run_log_enabled", True) is False

    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "1")
    assert run_log_module._setting("run_log_enabled", True) is True


def test_setting_boolean_env_var_unrecognized_value_uses_default(monkeypatch):
    import tldw_chatbook.config as config_module

    monkeypatch.setattr(config_module, "get_cli_setting", lambda *a, **k: True)
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "maybe")
    assert run_log_module._setting("run_log_enabled", True) is True


def test_env_var_disables_logging_end_to_end(root, monkeypatch):
    """Integration check: the env override actually reaches `bind()`, not
    just the unit-level `_setting` function."""
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "false")
    writer = RunLogWriter()
    writer.bind("run-abc")
    assert writer.is_active is False
    assert not (root / "agent-runs").exists()


def test_explicit_constructor_arg_still_wins_over_env_var(monkeypatch):
    """Explicit constructor args remain the highest-priority override --
    ABOVE both the env var and TOML tiers `_setting` resolves between."""
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_DIR_NAME", "env-value")
    writer = RunLogWriter(dir_name="explicit-value")
    assert writer._dir_name == "explicit-value"


# -- F7 (Qodo #7): a capped record must say so, not claim completeness -------


def test_append_reports_truncated_on_the_returned_record_number(root):
    writer = make(root, max_record_bytes=50)
    number = writer.append(run_id="r", kind="primary", type="tool_result", content="y" * 500)
    assert number.truncated is True
    assert int(number) == 1  # still a plain int for every ordinary purpose


def test_append_reports_not_truncated_when_under_the_cap(root):
    writer = make(root)
    number = writer.append(run_id="r", kind="primary", type="model", content="short")
    assert number.truncated is False
    assert int(number) == 1


def test_record_number_behaves_like_a_plain_int_everywhere(root):
    """RunLogRecordNumber must be indistinguishable from int for every
    existing consumer -- formatting, equality, hashing/set membership --
    since `test_on_record_returns_the_assigned_record_number` (and every
    caller before this fix) only ever expects a plain int."""
    writer = make(root)
    number = writer.append(run_id="r", kind="primary", type="model", content="x")
    assert isinstance(number, int)
    assert number == 1
    assert f"{number:06d}" == "000001"
    assert {number, 1} == {1}


# -- F8 (Qodo #8, task-1251): a workspace folder inside the sandbox must ----
# -- still get the dotted name -----------------------------------------------


def test_workspace_folder_resolving_inside_the_sandbox_still_gets_dotted_name(
    tmp_path, monkeypatch
):
    """The fallback FLAG alone only reports which BRANCH resolve_log_root()
    took -- but a bound WORKSPACE folder can itself resolve inside (or
    equal to) the sandbox root, in which case the "workspace" branch fires,
    the flag stays False, and (pre-fix) the log would get the undotted
    name while still being fully reachable via grep_files/glob_files
    (which root at the sandbox and never consult allowed_file_roots,
    §9.4) -- exactly the sub-agent log-disclosure bug the dotted name
    exists to prevent. This drives the REAL resolve_log_root()/bind() (no
    monkeypatching either of those directly) with a workspace folder
    planted INSIDE a fake sandbox root, and asserts the dotted name is
    still chosen.
    """
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    workspace_folder = sandbox_root / "bound-workspace"
    workspace_folder.mkdir()

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox_root)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox_root, workspace_folder),
    )

    writer = RunLogWriter()
    writer.bind("run-abc")

    assert writer.is_active is True
    assert writer.log_dir == workspace_folder / ".agent-runs" / "run-abc"
    # The undotted name must never have been created at all.
    assert not (workspace_folder / "agent-runs").exists()


def test_workspace_folder_equal_to_the_sandbox_root_gets_dotted_name(
    tmp_path, monkeypatch
):
    """Same as above for the degenerate case: the bound workspace folder
    IS the sandbox root, not merely nested inside it."""
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox_root)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox_root, sandbox_root),
    )

    writer = RunLogWriter()
    writer.bind("run-abc")

    assert writer.is_active is True
    assert writer.log_dir == sandbox_root / ".agent-runs" / "run-abc"


def test_workspace_folder_outside_the_sandbox_keeps_the_undotted_name(
    tmp_path, monkeypatch
):
    """Regression guard for the fix above: a GENUINE workspace folder
    (outside the sandbox entirely) must still get the user-visible,
    undotted name -- the F8 fix narrows the dotting decision, it must not
    widen it to dot every workspace folder."""
    import tldw_chatbook.Tools.file_operation_tools as file_tools
    import tldw_chatbook.Tools.workspace_file_roots as ws_roots

    sandbox_root = tmp_path / "sandbox"
    sandbox_root.mkdir()
    workspace_folder = tmp_path / "genuine-workspace"
    workspace_folder.mkdir()

    monkeypatch.setattr(file_tools, "_tool_sandbox_root", lambda: sandbox_root)
    monkeypatch.setattr(
        ws_roots,
        "allowed_file_roots",
        lambda write=False, sandbox_root=None: (sandbox_root, workspace_folder),
    )

    writer = RunLogWriter()
    writer.bind("run-abc")

    assert writer.is_active is True
    assert writer.log_dir == workspace_folder / "agent-runs" / "run-abc"
