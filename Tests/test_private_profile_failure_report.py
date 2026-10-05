"""What a failed private-profile child tells its parent (TASK-33621.13).

``private_profile_test`` quotes the child's own failure from its JUnit report
between the log path and the log tail. PR #2945 review: a report that parsed
but held no failure/error element -- exit 5 when the child selected nothing,
or a crash after the report was written -- printed an empty section there.
"""

import pytest

from Tests.private_profile import _child_failure_report

_NO_FAILURE = "(the child's JUnit report records no failure or error; see the log tail)"


def test_a_child_failure_is_quoted_from_its_junit_report(tmp_path):
    report = tmp_path / "pytest.xml"
    report.write_text(
        '<testsuites><testsuite><testcase name="test_case">'
        '<failure message="assert 1 == 2">zq-child-trace</failure>'
        "</testcase></testsuite></testsuites>"
    )
    text = _child_failure_report(report)
    assert text == "--- child failure: assert 1 == 2\nzq-child-trace"


@pytest.mark.parametrize(
    "body",
    [
        '<testsuites><testsuite tests="0" /></testsuites>',
        '<testsuites><testsuite><testcase name="test_case" /></testsuite></testsuites>',
    ],
    ids=["nothing-selected", "passed"],
)
def test_a_report_without_a_failure_says_so_instead_of_printing_nothing(tmp_path, body):
    report = tmp_path / "pytest.xml"
    report.write_text(body)
    assert _child_failure_report(report) == _NO_FAILURE


def test_a_missing_report_says_so(tmp_path):
    assert (
        _child_failure_report(tmp_path / "pytest.xml")
        == "(the child wrote no JUnit report)"
    )


def test_a_utf8_child_failure_reaches_a_locale_bound_parent(tmp_path, monkeypatch):
    """The wrapper must report the failed child instead of failing to decode it."""
    import asyncio
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import Tests.private_profile as private_profile

    monkeypatch.delenv(private_profile._CHILD_NODE, raising=False)
    original_read_text = Path.read_text
    original_create_process = asyncio.create_subprocess_exec
    child_environment = {}
    child_node = "test_selected_utf8_child"

    def locale_bound_read_text(path, encoding=None, errors=None):
        # Faithfully reproduce the Windows isolated parent's default CP1252.
        return original_read_text(path, encoding=encoding or "cp1252", errors=errors)

    async def reporting_child(*command, **kwargs):
        assert command[:3] == (sys.executable, "-m", "pytest")
        assert command[3] == child_node
        child_environment.update(kwargs["env"])
        report = next(
            arg.removeprefix("--junitxml=")
            for arg in command
            if arg.startswith("--junitxml=")
        )
        script = (
            "from pathlib import Path\n"
            "import sys\n"
            "Path(sys.argv[1]).write_text(\n"
            "    '<testsuites><testsuite><testcase name=\"test_selected_utf8_child\">'\n"
            "    '<failure message=\"real-child-assertion\">real-child-trace</failure>'\n"
            "    '</testcase></testsuite></testsuites>', encoding='utf-8')\n"
            "sys.stdout.buffer.write(b'\\xc4\\x81 real-child-log\\n')\n"
            "raise SystemExit(1)\n"
        )
        return await original_create_process(
            sys.executable,
            "-I",
            "-S",
            "-c",
            script,
            report,
            cwd=tmp_path,
            env=kwargs["env"],
            stdout=kwargs["stdout"],
            stderr=kwargs["stderr"],
        )

    monkeypatch.setattr(Path, "read_text", locale_bound_read_text)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", reporting_child)
    request = SimpleNamespace(
        node=SimpleNamespace(
            nodeid=child_node,
            name=child_node,
            get_closest_marker=lambda _name: SimpleNamespace(args=(8,), kwargs={}),
        ),
        config=SimpleNamespace(
            pluginmanager=SimpleNamespace(getplugin=lambda _name: None)
        ),
        getfixturevalue=lambda name: tmp_path if name == "tmp_path" else None,
    )

    @private_profile.private_profile_test
    async def selected_child(request):
        pytest.fail("the original parent must execute its selected subprocess")

    with pytest.raises(pytest.fail.Exception) as failure:
        asyncio.run(selected_child(request=request))
    message = str(failure.value)
    assert "--- child failure: real-child-assertion\nreal-child-trace" in message
    assert "\u0101 real-child-log" in message
    assert "UnicodeDecodeError" not in message
    assert child_environment["PYTHONIOENCODING"] == "utf-8"
