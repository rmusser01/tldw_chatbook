"""Finite run-log path preparation preserves original path decisions."""

import sys
from contextlib import contextmanager

import pytest

from tldw_chatbook.Agents.run_log import RunLogWriter
from tldw_chatbook.Tools import file_operation_tools
from tldw_chatbook.Utils import sensitive_paths


# The real sensitive-path resolver keeps its admitted collection profile.
pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "1")
    monkeypatch.setenv("RAG_PERSIST_DIR", str(tmp_path / "vectors"))
    root = tmp_path / "run-root"
    root.mkdir()
    return root


def _writer(root, *, dir_name="agent-runs"):
    return RunLogWriter(
        root=root,
        dir_name=dir_name,
        segment_bytes=4096,
        max_record_bytes=1024,
    )


@contextmanager
def _observe_path_checks():
    """Observe original bodies without replacing a stock-qualified callback."""
    resolver_code = sensitive_paths.resolve_sensitive_context.__code__
    containment_code = file_operation_tools.is_within.__code__
    contexts = []
    checks = []

    def observe(frame, event, result):
        if frame.f_code is resolver_code and event == "return":
            contexts.append(result)
        elif frame.f_code is containment_code and event == "call":
            checks.append(
                (
                    frame.f_locals["candidate"],
                    frame.f_locals["root"],
                    frame.f_locals.get("context"),
                )
            )

    previous = sys.getprofile()
    assert previous is None
    sys.setprofile(observe)
    try:
        yield contexts, checks
    finally:
        installed = sys.getprofile()
        sys.setprofile(previous)
        assert installed is observe


def test_bind_resolves_one_context_for_both_original_path_checks(root):
    writer = _writer(root)

    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert writer.is_active
    assert writer.log_dir == root / ".agent-runs" / "first"
    assert (root / ".agent-runs" / ".gitignore").read_text() == "*\n"
    assert len(contexts) == 1
    assert [(path, base) for path, base, _ in checks] == [
        (root / ".agent-runs", root),
        (root / ".agent-runs" / "first", root),
    ]
    assert all(context is contexts[0] for _, _, context in checks)


def test_new_bind_observes_changed_sensitive_paths(root, monkeypatch):
    first = _writer(root)
    with _observe_path_checks() as (first_contexts, _):
        first.bind("first")
    assert first.is_active

    # A new direct child of this state directory must now be refused.
    monkeypatch.setenv("RAG_PERSIST_DIR", str(root / ".agent-runs"))
    second = _writer(root)
    with _observe_path_checks() as (second_contexts, checks):
        second.bind("second")

    assert not second.is_active
    assert second.log_dir is None
    assert not (root / ".agent-runs" / "second").exists()
    assert len(first_contexts) == len(second_contexts) == 1
    assert second_contexts[0] is not first_contexts[0]
    assert root / ".agent-runs" in second_contexts[0].direct_child_denied_dirs
    assert all(context is second_contexts[0] for _, _, context in checks)


def test_repeated_bind_does_not_resolve_or_check_another_path(root):
    writer = _writer(root)
    writer.bind("first")
    assert writer.is_active

    with _observe_path_checks() as (contexts, checks):
        writer.bind("second")

    assert writer.log_dir == root / ".agent-runs" / "first"
    assert not (root / ".agent-runs" / "second").exists()
    assert contexts == checks == []


def test_disabled_writer_does_not_prepare_sensitive_paths(root, monkeypatch):
    monkeypatch.setenv("TLDW_AGENTS_RUN_LOG_ENABLED", "0")
    writer = _writer(root)

    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert not writer.is_active
    assert not (root / ".agent-runs").exists()
    assert contexts == checks == []


@pytest.mark.parametrize("target", ["base", "run"])
def test_sensitive_base_and_run_directory_remain_refused(root, target):
    writer = _writer(root, dir_name=".npmrc" if target == "base" else "agent-runs")
    run_id = "first" if target == "base" else ".npmrc"

    with _observe_path_checks() as (contexts, checks):
        writer.bind(run_id)

    denied = root / ".npmrc" if target == "base" else root / ".agent-runs" / ".npmrc"
    assert not writer.is_active
    assert writer.log_dir is None
    assert not denied.exists()
    assert len(contexts) == 1
    assert len(checks) == (1 if target == "base" else 2)
    assert checks[-1][0] == denied
    assert all(context is contexts[0] for _, _, context in checks)


@pytest.mark.parametrize("target", ["base", "run"])
def test_each_path_check_still_refuses_escape_from_root(root, target):
    writer = _writer(root)
    if target == "base":
        # Exercise containment independently of constructor name validation.
        writer._dir_name = "../escaped"
    run_id = "first" if target == "base" else "../../escaped"

    with _observe_path_checks() as (contexts, checks):
        writer.bind(run_id)

    assert not writer.is_active
    assert writer.log_dir is None
    assert not (root.parent / "escaped").exists()
    assert len(contexts) == 1
    assert len(checks) == (1 if target == "base" else 2)
    assert not checks[-1][0].resolve().is_relative_to(root)


def test_legacy_migration_keeps_its_separate_path_preparation(root):
    legacy = root / "agent-runs" / "old-run"
    legacy.mkdir(parents=True)
    (legacy / "logs.0001.txt").write_text("preserved history", encoding="utf-8")
    writer = _writer(root)

    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert writer.is_active
    assert not (root / "agent-runs").exists()
    assert (root / ".agent-runs" / "old-run" / "logs.0001.txt").read_text() == (
        "preserved history"
    )
    assert len(contexts) == 2
    assert [path for path, _, _ in checks] == [
        root / ".agent-runs",
        root / "agent-runs",
        root / ".agent-runs" / "first",
    ]
    assert checks[0][2] is checks[2][2] is contexts[0]
    assert checks[1][2] is None
    assert contexts[1] is not contexts[0]


@pytest.mark.parametrize("replacement", ["callback", "original-body"])
def test_custom_containment_keeps_two_argument_calls_without_eager_context(
    root, monkeypatch, replacement
):
    def two_argument_containment(candidate, root):
        return candidate.resolve().is_relative_to(root.resolve())

    original = file_operation_tools.is_within
    if replacement == "callback":
        monkeypatch.setattr(file_operation_tools, "is_within", two_argument_containment)
    else:
        monkeypatch.setattr(original, "__code__", two_argument_containment.__code__)
    writer = _writer(root)

    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert writer.is_active
    assert writer.log_dir == root / ".agent-runs" / "first"
    assert contexts == []
    assert checks == [
        (root / ".agent-runs", root, None),
        (root / ".agent-runs" / "first", root, None),
    ]


def test_custom_migration_keeps_fresh_sensitive_checks(root, monkeypatch):
    writer = _writer(root)
    migrations = []

    def migrate(migration_root, legacy_name, dotted):
        migrations.append((migration_root, legacy_name, dotted))
        monkeypatch.setenv("RAG_PERSIST_DIR", str(dotted))

    monkeypatch.setattr(writer, "_migrate_legacy_dir", migrate)
    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    base = root / ".agent-runs"
    assert migrations == [(root, "agent-runs", base)]
    assert not writer.is_active
    assert writer.log_dir is None
    assert not (base / "first").exists()
    assert len(contexts) == 2
    assert base not in contexts[0].direct_child_denied_dirs
    assert base in contexts[1].direct_child_denied_dirs
    assert checks == [(base, root, None), (base / "first", root, None)]


def test_changed_containment_default_keeps_its_sensitive_denial(root, monkeypatch):
    base = root / ".agent-runs"
    denied = sensitive_paths.SensitivePathContext((), (base,), (), None, ())
    monkeypatch.setattr(file_operation_tools.is_within, "__defaults__", (denied,))
    writer = _writer(root)

    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert not writer.is_active
    assert writer.log_dir is None
    assert not base.exists()
    assert contexts == []
    assert len(checks) == 1
    assert checks[0][:2] == (base, root)
    assert checks[0][2] is denied


def test_custom_sensitive_predicate_receives_original_none_context(root, monkeypatch):
    original = file_operation_tools.is_sensitive_path
    received = []

    def refuses_without_context(candidate, context=None):
        received.append((candidate, context))
        return context is None or original(candidate, context=context)

    monkeypatch.setattr(
        file_operation_tools, "is_sensitive_path", refuses_without_context
    )
    writer = _writer(root)
    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    base = root / ".agent-runs"
    assert not writer.is_active
    assert writer.log_dir is None
    assert not base.exists()
    assert contexts == []
    assert received == [(base, None)]
    assert checks == [(base, root, None)]


def test_custom_sensitive_resolver_keeps_each_original_path_observation(
    root, monkeypatch
):
    base = root / ".agent-runs"
    run_dir = base / "first"
    allowed = sensitive_paths.SensitivePathContext((), (), (), None, ())
    denied = allowed._replace(dirs=(run_dir,))
    returned = []

    def changing_context():
        result = allowed if not returned else denied
        returned.append(result)
        return result

    monkeypatch.setattr(sensitive_paths, "resolve_sensitive_context", changing_context)
    writer = _writer(root)
    with _observe_path_checks() as (contexts, checks):
        writer.bind("first")

    assert not writer.is_active
    assert writer.log_dir is None
    assert base.is_dir()
    assert not run_dir.exists()
    assert returned == contexts == [allowed, denied]
    assert checks == [(base, root, None), (run_dir, root, None)]
