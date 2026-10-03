"""TASK-33665: test runs can never write the real user's tldw profile.

The mechanism tests point the guard at a temporary stand-in for the real profile
(``_roots`` is patched), so a broken guard writes into ``tmp_path``, never into
the owner's home. Each test that trips the guard on purpose takes its recorded
refusals before returning. The ``fake_profile`` teardown drops only refusals
about the stand-in and fails the test on any other refusal.
"""

from __future__ import annotations

import ast
import importlib
import os
import shutil
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests import real_profile_guard as guard

REPO_ROOT = Path(__file__).resolve().parents[1]
OWNER_STATE = '[sidebar]\nsearch_query = "owner"\n'


def _about(root: Path, refusal: str) -> bool:
    return f"{root}{os.sep}" in refusal or f"{root}'" in refusal


@pytest.fixture
def fake_profile(tmp_path, monkeypatch):
    """A stand-in for ``~/.config/tldw_cli`` that the guard protects instead."""
    root = tmp_path / "home" / ".config" / "tldw_cli"
    (root / "sub").mkdir(parents=True)
    (root / "existing.txt").write_text("keep")
    (root / "ui_state.toml").write_text(OWNER_STATE)
    monkeypatch.setattr(guard, "_roots", (guard._norm(str(root)),))
    yield root
    others = [r for r in guard.take_violations() if not _about(root, r)]
    if others:
        pytest.fail(
            "refusals outside the stand-in profile:\n" + "\n".join(others),
            pytrace=False,
        )


def _take_about(root: Path) -> list[str]:
    taken = guard.take_violations()
    assert taken and all(_about(root, r) for r in taken), taken
    return taken


def test_the_guard_protects_the_real_homes_profile_dirs_not_home_env():
    real_home = guard._real_home()
    roots = guard.protected_roots()
    assert guard._norm(str(real_home / ".config" / "tldw_cli")) in roots
    assert guard._norm(str(real_home / ".local" / "share" / "tldw_cli")) in roots
    # The suite's HOME is a sandbox; the guard must not follow it.
    assert not os.environ["HOME"].startswith(str(real_home / ".config"))
    assert guard._roots, "the root conftest installs the guard before any test"


def test_the_real_home_is_exported_so_workers_and_children_inherit_it(
    monkeypatch, tmp_path
):
    # The first process exported it before the conftest sandboxed HOME.
    assert os.environ[guard.REAL_HOME_ENV] == str(guard._real_home())
    # A worker or child whose HOME is a sandbox reads the inherited value
    # first, so it never protects its own sandbox instead of the real home.
    monkeypatch.setenv(guard.REAL_HOME_ENV, str(tmp_path))
    assert guard._real_home() == tmp_path


def _dir_fd(call):
    def run(root):
        fd = os.open(root, os.O_RDONLY)
        try:
            call(fd)
        finally:
            os.close(fd)

    return run


def _skip_without(name):
    return pytest.mark.skipif(not hasattr(os, name), reason=f"no os.{name} here")


@pytest.mark.parametrize(
    "write",
    [
        pytest.param(lambda r: (r / "ui_state.toml").write_text("x"), id="write_text"),
        pytest.param(lambda r: open(r / "existing.txt", "a").close(), id="append"),
        pytest.param(
            lambda r: os.open(r / "new.bin", os.O_CREAT | os.O_WRONLY), id="os.open"
        ),
        pytest.param(lambda r: os.mkdir(r / "dir"), id="mkdir"),
        pytest.param(lambda r: os.remove(r / "existing.txt"), id="remove"),
        pytest.param(
            lambda r: os.replace(r / "existing.txt", r / "moved.txt"), id="replace"
        ),
        pytest.param(lambda r: shutil.rmtree(r), id="rmtree"),
        pytest.param(
            lambda r: sqlite3.connect(r / "library_collections.db"), id="sqlite"
        ),
        pytest.param(
            lambda r: sqlite3.connect(f"file:{r / 'x.db'}?mode=rwc", uri=True),
            id="sqlite-uri",
        ),
        pytest.param(
            lambda r: sqlite3.connect(
                f"file://localhost{r / 'x.db'}?mode=rwc", uri=True
            ),
            id="sqlite-uri-localhost",
        ),
        pytest.param(
            lambda r: sqlite3.connect(
                "file:" + str(r / "x.db").replace("tldw_cli", "tldw%5Fcli"), uri=True
            ),
            id="sqlite-uri-percent-encoded",
        ),
        # M1: metadata writes.
        pytest.param(lambda r: os.chmod(r / "existing.txt", 0o600), id="chmod"),
        pytest.param(lambda r: os.utime(r / "existing.txt"), id="utime"),
        pytest.param(
            lambda r: os.chown(r / "existing.txt", os.getuid(), -1),
            id="chown",
            marks=_skip_without("chown"),
        ),
        pytest.param(
            lambda r: os.chflags(r / "existing.txt", 0),
            id="chflags",
            marks=_skip_without("chflags"),
        ),
        pytest.param(
            lambda r: os.setxattr(r / "existing.txt", "user.tldw", b"1"),
            id="setxattr",
            marks=_skip_without("setxattr"),
        ),
        pytest.param(
            lambda r: os.removexattr(r / "existing.txt", "user.tldw"),
            id="removexattr",
            marks=_skip_without("removexattr"),
        ),
        # C1: writes relative to a directory fd carry only the leaf name.
        pytest.param(_dir_fd(lambda fd: os.mkdir("dir", dir_fd=fd)), id="mkdir-dirfd"),
        pytest.param(
            _dir_fd(lambda fd: os.remove("existing.txt", dir_fd=fd)), id="remove-dirfd"
        ),
        pytest.param(_dir_fd(lambda fd: os.rmdir("sub", dir_fd=fd)), id="rmdir-dirfd"),
        pytest.param(
            _dir_fd(
                lambda fd: os.replace(
                    "existing.txt", "moved.txt", src_dir_fd=fd, dst_dir_fd=fd
                )
            ),
            id="replace-dirfd",
        ),
        pytest.param(
            _dir_fd(
                lambda fd: os.link(
                    "existing.txt", "linked.txt", src_dir_fd=fd, dst_dir_fd=fd
                )
            ),
            id="link-dirfd",
        ),
        pytest.param(
            _dir_fd(lambda fd: os.symlink("existing.txt", "lnk", dir_fd=fd)),
            id="symlink-dirfd",
        ),
        pytest.param(
            _dir_fd(lambda fd: os.chmod("existing.txt", 0o600, dir_fd=fd)),
            id="chmod-dirfd",
        ),
        pytest.param(
            _dir_fd(lambda fd: os.utime("existing.txt", dir_fd=fd)), id="utime-dirfd"
        ),
        pytest.param(
            _dir_fd(lambda fd: shutil.rmtree("sub", dir_fd=fd)), id="rmtree-dirfd"
        ),
    ],
)
def test_every_write_kind_into_the_profile_is_refused_and_recorded(fake_profile, write):
    with pytest.raises(guard.RealProfileWriteError):
        write(fake_profile)
    assert (fake_profile / "existing.txt").read_text() == "keep"
    assert (fake_profile / "sub").is_dir()
    assert sorted(p.name for p in fake_profile.iterdir()) == [
        "existing.txt",
        "sub",
        "ui_state.toml",
    ]
    assert len(_take_about(fake_profile)) == 1


@pytest.mark.parametrize(
    "destroy",
    [
        pytest.param(lambda a, tmp: shutil.rmtree(a), id="rmtree"),
        pytest.param(lambda a, tmp: os.rename(a, tmp / "moved"), id="rename-src"),
        pytest.param(lambda a, tmp: shutil.move(a, tmp / "moved"), id="move-src"),
        pytest.param(lambda a, tmp: os.rmdir(a), id="rmdir"),
        pytest.param(lambda a, tmp: os.remove(a), id="remove"),
    ],
)
def test_destroying_a_directory_that_holds_the_profile_is_refused(
    fake_profile, tmp_path, destroy
):
    # M2: shutil.rmtree(~/.config) must not get past the guard.
    ancestor = fake_profile.parent
    with pytest.raises(guard.RealProfileWriteError):
        destroy(ancestor, tmp_path)
    assert (fake_profile / "existing.txt").read_text() == "keep"
    taken = guard.take_violations()
    assert len(taken) == 1 and f"'{ancestor}'" in taken[0]


def test_writes_through_a_symlink_into_the_profile_are_refused(fake_profile, tmp_path):
    # M3: the abspath is outside the profile; only the realpath is inside.
    link = tmp_path / "innocent"
    link.symlink_to(fake_profile)
    with pytest.raises(guard.RealProfileWriteError):
        (link / "ui_state.toml").write_text("x")
    assert (fake_profile / "ui_state.toml").read_text() == OWNER_STATE
    assert len(_take_about(fake_profile)) == 1


@pytest.mark.skipif(sys.platform != "darwin", reason="APFS is case-insensitive")
def test_a_case_variant_of_the_profile_path_is_refused_on_apfs(fake_profile):
    variant = Path(str(fake_profile).replace("tldw_cli", "TLDW_CLI"))
    with pytest.raises(guard.RealProfileWriteError):
        (variant / "ui_state.toml").write_text("x")
    assert (fake_profile / "ui_state.toml").read_text() == OWNER_STATE
    assert len(guard.take_violations()) == 1


def test_reads_and_writes_elsewhere_are_untouched(fake_profile, tmp_path):
    assert (fake_profile / "existing.txt").read_text() == "keep"
    outside = tmp_path / "elsewhere.txt"
    outside.write_text("ok")
    # Next to the profile, not over it: still allowed.
    (fake_profile.parent / "sibling.txt").write_text("ok")
    os.mkdir(fake_profile.parent / "sibling")
    shutil.rmtree(fake_profile.parent / "sibling")
    sqlite3.connect(tmp_path / "ok.db").close()
    sqlite3.connect(":memory:").close()
    sqlite3.connect("file::memory:?cache=shared", uri=True).close()
    assert guard.take_violations() == []


def test_the_fake_profile_teardown_only_drops_its_own_refusals(tmp_path):
    root = tmp_path / "home" / ".config" / "tldw_cli"
    own = f"open on the real profile path '{root / 'x'}' in thread 'MainThread'"
    whole = f"shutil.rmtree on the real profile path '{root}' in thread 'MainThread'"
    other = "open on the real profile path '/Users/someone/.config/tldw_cli/x'"
    assert [r for r in (own, whole, other) if not _about(root, r)] == [other]


def test_a_late_thread_is_refused_and_still_recorded_when_it_swallows_the_error(
    fake_profile,
):
    def late_writer():
        try:
            (fake_profile / "ui_state.toml").write_text("[sidebar]\n")
        except Exception:  # noqa: BLE001, S110 - production writers log and move on
            pass

    thread = threading.Thread(target=late_writer, name="late-writer-33665")
    thread.start()
    thread.join()
    assert (fake_profile / "ui_state.toml").read_text() == OWNER_STATE
    (refusal,) = _take_about(fake_profile)
    assert "late-writer-33665" in refusal  # M5: which thread wrote


def test_the_incidents_own_dir_fd_writer_cannot_replace_the_profile_file(
    fake_profile, monkeypatch
):
    """C1: ChatScreen's sidebar writer publishes with dir-fd renames.

    It opens its temporary with ``os.open(name, dir_fd=parent)``, whose audit
    event carries only the leaf name, so a stray ``ui_state.toml.tmp`` can
    remain. The replace and the cleanup carry dir fds the guard resolves, so
    the owner's file is never replaced.
    """
    from types import SimpleNamespace as NS

    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen

    monkeypatch.setenv("TLDW_CONFIG_PATH", str(fake_profile / "config.toml"))
    screen = ChatScreen.__new__(ChatScreen)
    screen.ui_state = NS(
        collapsible_states={}, sidebar_search_query="", last_active_section=None
    )
    assert screen._save_sidebar_state() is False  # it swallows the refusal
    assert (fake_profile / "ui_state.toml").read_text() == OWNER_STATE
    assert _take_about(fake_profile)


def test_an_open_relative_to_a_dir_fd_is_invisible_to_the_guard(fake_profile):
    """Documents the known gap: the ``open`` audit event has no dir fd.

    Not restoring the real environment after the session (Tests/conftest.py)
    is what keeps such a writer out of the real profile.
    """
    fd = os.open(fake_profile, os.O_RDONLY)
    try:
        os.close(os.open("stray.tmp", os.O_CREAT | os.O_WRONLY, 0o600, dir_fd=fd))
    finally:
        os.close(fd)
    assert (fake_profile / "stray.tmp").exists()
    assert guard.take_violations() == []


def _sandbox_env(monkeypatch, names):
    """Let monkeypatch restore ``names`` however the code under test sets them."""
    for name in names:
        if name in os.environ:
            monkeypatch.setenv(name, os.environ[name])
        else:
            monkeypatch.delenv(name, raising=False)
    return {name: os.environ.get(name) for name in names}


def test_session_end_leaves_the_environment_in_the_sandbox(monkeypatch, tmp_path):
    """C1(a): late threads must write into the dead sandbox, never the profile."""
    root_conftest = sys.modules["Tests.conftest"]
    names = root_conftest._SANDBOXED_ENV_NAMES
    before = _sandbox_env(monkeypatch, names)
    sandbox = tmp_path / "owned_sandbox"
    sandbox.mkdir()
    monkeypatch.setattr(root_conftest, "_XDIST_WORKER", None)
    monkeypatch.setattr(root_conftest, "_OWNS_BOOTSTRAP_CONFIG_ROOT", True)
    monkeypatch.setattr(root_conftest, "_BOOTSTRAP_CONFIG_ROOT", sandbox)
    # Stand-in "caller" values, so a regression restores these, not real ones.
    monkeypatch.setattr(
        root_conftest, "_PREVIOUS_TEST_ENV", {n: f"/previous/{n}" for n in names}
    )
    session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    root_conftest.pytest_sessionfinish(session, 0)
    assert {n: os.environ.get(n) for n in names} == before
    assert not sandbox.exists(), "the owned sandbox is still removed"


def _ui_conftest(monkeypatch):
    _sandbox_env(
        monkeypatch,
        ("HOME", "USERPROFILE", "TLDW_CONFIG_PATH", "TLDW_TEST_CONFIG_ROOT"),
    )
    return importlib.import_module("Tests.UI.conftest")


def test_the_ui_conftest_session_end_keeps_tldw_config_path(monkeypatch, tmp_path):
    ui_conftest = _ui_conftest(monkeypatch)
    sandbox = tmp_path / "ui_sandbox"
    config_path = sandbox / "config" / "config.toml"
    config_path.parent.mkdir(parents=True)
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    monkeypatch.setattr(ui_conftest, "_OWNS_BOOTSTRAP_CONFIG_ROOT", True)
    monkeypatch.setattr(ui_conftest, "_BOOTSTRAP_CONFIG_ROOT", sandbox)
    monkeypatch.setattr(ui_conftest, "_BOOTSTRAP_CONFIG_PATH", config_path)
    ui_conftest.pytest_sessionfinish(SimpleNamespace(config=SimpleNamespace()), 0)
    assert os.environ["TLDW_CONFIG_PATH"] == str(config_path)
    assert not sandbox.exists()


def test_the_ui_conftest_session_end_fails_the_run_on_a_refusal(
    fake_profile, monkeypatch, capsys
):
    # M6: a run rooted at Tests/UI still fails on an unclaimed refusal.
    ui_conftest = _ui_conftest(monkeypatch)
    monkeypatch.setattr(ui_conftest, "_OWNS_BOOTSTRAP_CONFIG_ROOT", False)
    with pytest.raises(guard.RealProfileWriteError):
        (fake_profile / "ui_state.toml").write_text("x")
    session = SimpleNamespace(config=SimpleNamespace(), exitstatus=0)
    ui_conftest.pytest_sessionfinish(session, 0)
    assert session.exitstatus == 1
    assert str(fake_profile) in capsys.readouterr().out
    assert guard.take_violations() == []


_CHILD_CONFTEST = """
import atexit
from pathlib import Path

import pytest

from Tests import real_profile_guard as guard
from Tests.conftest import pytest_sessionfinish, pytest_testnodedown  # noqa: F401

FAKE = Path({fake!r})
guard._roots = guard._roots + (guard._norm(str(FAKE)),)


@pytest.fixture(scope="session")
def late_session_writer():
    yield
    try:
        (FAKE / "ui_state.toml").write_text("late")
    except OSError:
        pass  # production writers log and move on


@atexit.register  # runs before the guard's own exit report (LIFO)
def _after_the_session():
    try:
        (FAKE / "ui_state.toml").write_text("after exit")
    except OSError:
        pass
"""


def test_an_xdist_workers_session_end_refusal_fails_the_run(fake_profile, tmp_path):
    """I1: a worker's own exit status and stdout are discarded by xdist.

    The same run checks that a refusal after the session-end check is still
    reported at exit.
    """
    child = tmp_path / "child"
    child.mkdir()
    (child / "pytest.ini").write_text("[pytest]\n")
    (child / "conftest.py").write_text(_CHILD_CONFTEST.format(fake=str(fake_profile)))
    (child / "test_late.py").write_text("def test_ok(late_session_writer):\n    pass\n")
    (tmp_path / "sandbox").mkdir()
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTEST_")}
    env.pop("TLDW_TEST_CONFIG_ROOT_OWNER", None)
    env["TLDW_TEST_CONFIG_ROOT"] = str(tmp_path / "sandbox")
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(REPO_ROOT), os.environ.get("PYTHONPATH")))
    )
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-n", "2", "-p", "no:cacheprovider", "-q"],
        cwd=child,
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = result.stdout + result.stderr
    assert result.returncode == 1, output
    assert "1 passed" in output, output
    assert "were refused:" in output and str(fake_profile) in output, output
    assert "refused after the session-end check" in output, output
    assert (fake_profile / "ui_state.toml").read_text() == OWNER_STATE


@pytest.mark.bootstrap_profile
def test_the_factory_app_keeps_every_database_in_its_own_user_data_dir():
    from Tests.UI import app_factory

    app = app_factory._build_test_app()
    user_data_dir = app_factory._created_dirs[-1]
    held = {}
    for name, value in vars(app).items():
        if isinstance(value, Path) and name.endswith("_path"):
            held[name] = value
        db_path = vars(value).get("db_path") if hasattr(value, "__dict__") else None
        if isinstance(db_path, (str, Path)) and str(db_path) != ":memory:":
            held[f"{name}.db_path"] = Path(db_path)
    assert "local_library_collections_db.db_path" in held
    assert "_tts_profile_repository_path" in held
    outside = {
        name: path
        for name, path in held.items()
        if user_data_dir not in Path(path).resolve().parents
    }
    assert outside == {}


def test_the_ui_conftest_moves_home_before_any_tldw_import():
    source = Path(__file__).parent.joinpath("UI", "conftest.py").read_text()
    tree = ast.parse(source)
    home_line = next(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(t, ast.Subscript) and getattr(t.slice, "value", None) == "HOME"
            for t in node.targets
        )
    )
    first_tldw_import = min(
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and any(
            name in ast.unparse(node) for name in ("tldw_chatbook", "Tests.", "textual")
        )
    )
    assert home_line < first_tldw_import
