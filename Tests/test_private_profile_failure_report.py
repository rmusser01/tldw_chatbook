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
