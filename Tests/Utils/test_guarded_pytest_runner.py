"""Exercise the guarded command builder with a stub child and disposable paths."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tools import run_guarded_pytest as runner


@pytest.fixture
def invocation(monkeypatch, tmp_path):
    child = Mock(return_value=SimpleNamespace(returncode=17))
    monkeypatch.setattr(runner, "subprocess", SimpleNamespace(run=child))
    monkeypatch.setattr(runner, "os", SimpleNamespace(environ={}))
    verified = []
    monkeypatch.setattr(runner, "verify_temp_parent", verified.append)
    target = "Tests/Utils/test_guarded_pytest_runner.py"

    def invoke(*args):
        monkeypatch.setattr(
            runner.sys, "argv", ["runner", str(tmp_path), target, *args]
        )
        return runner.main()

    return invoke, child, tmp_path, verified


@pytest.mark.parametrize(
    "options",
    [
        ["--basetemp", "unverified"],
        ["--basetemp=unverified"],
        ["--base=unverified"],
        ["--noconftest"],
        ["--confcutdir", "elsewhere"],
        ["-c", "alternate.ini"],
        ["-oaddopts=--noconftest"],
        ["-p", "extra_plugin"],
        ["@arguments.txt"],
        ["-k", "@arguments.txt"],
        ["Tests/../tools"],
    ],
)
def test_conflicting_caller_arguments_never_reach_child(invocation, options):
    invoke, child, _, _ = invocation
    with pytest.raises((RuntimeError, SystemExit)):
        invoke(*options)
    child.assert_not_called()


def test_guarded_child_has_verified_storage_and_normal_options(invocation):
    invoke, child, parent, verified = invocation
    runner.os.environ.update(
        PYTEST_ADDOPTS="--noconftest --basetemp=unverified",
        PYTEST_PLUGINS="extra_plugin",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONUTF8="1",
    )
    node = "Tests/Utils/test_guarded_pytest_runner.py::test_conflicting_caller_arguments_never_reach_child[options0]"
    options = [
        node,
        "-k",
        "one or two",
        "-m",
        "unit",
        "-W",
        "error",
        "--tb=short",
        "-v",
        "--show-capture=no",
    ]
    assert invoke(*options) == 17
    child.assert_called_once()
    (argv,) = child.call_args.args
    env = child.call_args.kwargs["env"]
    assert set(env).isdisjoint(
        {"PYTEST_ADDOPTS", "PYTEST_PLUGINS", "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    )
    assert env["PYTHONUTF8"] == "1"
    root, temp = verified
    assert root == parent and temp.parent == parent and not temp.exists()
    assert env["TMPDIR"] == env["TEMP"] == env["TMP"] == str(temp)
    assert argv.count("--basetemp") == 1
    assert argv[argv.index("--basetemp") + 1] == str(temp / "case")
    assert argv[argv.index("-c") + 1] == str(runner.REPO / "pyproject.toml")
    assert argv[-len(options) :] == options
