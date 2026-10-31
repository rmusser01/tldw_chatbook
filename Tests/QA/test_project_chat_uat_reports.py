"""Exact UAT gate contract: no missing/skipped cases or stale partial receipts."""

import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

check_reports = runpy.run_path(
    str(Path(__file__).resolve().parents[2] / "scripts/check_project_chat_uat.py")
)["check_reports"]

CASES = ("fenced", "native", "compact-recovery", "wrong-family")


@pytest.fixture
def reports(tmp_path):
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite")
    artifacts = tmp_path / "artifacts"
    for variant in CASES:
        ET.SubElement(
            suite,
            "testcase",
            classname="Tests.QA.test_project_chat_uat",
            name=f"test_project_chat_uat[{variant}]",
        )
        folder = artifacts / variant
        folder.mkdir(parents=True)
        (folder / "receipt.json").write_text(
            json.dumps(
                {
                    "case": "UAT-02",
                    "variant": variant,
                    "status": "passed",
                    "phase": "complete",
                    "run_id": f"run-{variant}",
                    "conversation_id": f"chat-{variant}",
                    "assistant_message_id": f"answer-{variant}",
                    "request_count": 5,
                    "model_errors": [],
                    "blocked_destinations": [],
                }
            )
        )
        (folder / "viewport.svg").write_text("<svg/>")
        (folder / "viewport.txt").write_text("synthetic answer")
    junit = tmp_path / "results.xml"
    ET.ElementTree(root).write(junit)
    return junit, artifacts


def test_complete_pilot_report_passes(reports):
    check_reports(*reports)


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "skipped", "failure", "error"]
)
def test_incomplete_or_nonpassing_pytest_cases_fail(reports, defect):
    junit, artifacts = reports
    tree = ET.parse(junit)
    suite = tree.getroot()[0]
    if defect == "missing":
        suite.remove(suite[0])
    elif defect == "duplicate":
        suite.append(ET.fromstring(ET.tostring(suite[0])))
    else:
        ET.SubElement(suite[0], defect)
    tree.write(junit)
    with pytest.raises(AssertionError):
        check_reports(junit, artifacts)


@pytest.mark.parametrize(
    "defect",
    [
        "missing",
        "duplicate",
        "failed",
        "running",
        "no-run",
        "protocol-error",
        "network-escape",
        "no-screenshot",
        "no-text",
    ],
)
def test_missing_or_failed_receipts_fail(reports, defect):
    junit, artifacts = reports
    path = artifacts / "fenced" / "receipt.json"
    data = json.loads(path.read_text())
    if defect == "missing":
        path.unlink()
    elif defect == "duplicate":
        other = artifacts / "stale"
        other.mkdir()
        (other / "receipt.json").write_text(json.dumps(data))
    elif defect in ("no-screenshot", "no-text"):
        path.with_name(
            "viewport.svg" if defect == "no-screenshot" else "viewport.txt"
        ).unlink()
    else:
        if defect in ("failed", "running"):
            data["status"] = defect
        elif defect == "no-run":
            data.pop("run_id")
        elif defect == "protocol-error":
            data["model_errors"] = ["unexpected request"]
        else:
            data["blocked_destinations"] = [["127.0.0.1", 9000]]
        path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        check_reports(junit, artifacts)


@pytest.fixture
def negative_reports(tmp_path):
    junit = tmp_path / "negative.xml"
    root = ET.Element("testsuite")
    case = ET.SubElement(
        root,
        "testcase",
        classname="Tests.QA.test_project_chat_uat",
        name="test_project_chat_uat[fenced]",
    )
    ET.SubElement(
        case, "failure", message="README read did not supply a verification code"
    )
    ET.ElementTree(root).write(junit)
    artifacts = tmp_path / "negative"
    folder = artifacts / "fenced"
    folder.mkdir(parents=True)
    (folder / "receipt.json").write_text(
        json.dumps(
            {
                "variant": "fenced",
                "status": "failed",
                "phase": "await-answer",
                "negative_control": "missing-readme",
                "blocked_destinations": [],
                "model_errors": [
                    "AssertionError: README read did not supply a verification code"
                ],
                "requests": [
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": "Tool result for fs_read: README deliberately lacks verification evidence.",
                            }
                        ]
                    }
                ],
            }
        )
    )
    return junit, artifacts


def test_expected_red_control_requires_the_actual_read_failure(negative_reports):
    check_negative = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/check_project_chat_uat.py")
    )["check_negative_control"]
    check_negative(*negative_reports)
    path = negative_reports[1] / "fenced" / "receipt.json"
    data = json.loads(path.read_text())
    data["model_errors"] = ["connection failed before a read"]
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        check_negative(*negative_reports)


def test_expected_red_cannot_pass_without_read_evidence(negative_reports):
    check_negative = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/check_project_chat_uat.py")
    )["check_negative_control"]
    path = negative_reports[1] / "fenced" / "receipt.json"
    data = json.loads(path.read_text())
    data["requests"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        check_negative(*negative_reports)


@pytest.mark.parametrize("extra", ["error", "skipped"])
def test_expected_red_rejects_teardown_errors_and_skips(negative_reports, extra):
    check_negative = runpy.run_path(
        str(Path(__file__).resolve().parents[2] / "scripts/check_project_chat_uat.py")
    )["check_negative_control"]
    tree = ET.parse(negative_reports[0])
    ET.SubElement(next(tree.iter("testcase")), extra, message="unrelated failure")
    tree.write(negative_reports[0])
    with pytest.raises(AssertionError):
        check_negative(*negative_reports)


@pytest.mark.parametrize("optimization", ["flag", "environment"])
@pytest.mark.parametrize("negative", [False, True])
@pytest.mark.parametrize("defect", [None, "junit", "receipt"])
def test_cli_validation_survives_python_optimization(
    request, optimization, negative, defect
):
    junit, artifacts = request.getfixturevalue(
        "negative_reports" if negative else "reports"
    )
    if defect == "junit":
        tree = ET.parse(junit)
        ET.SubElement(next(tree.iter("testcase")), "skipped")
        tree.write(junit)
    elif defect == "receipt":
        path = artifacts / "fenced" / "receipt.json"
        row = json.loads(path.read_text())
        row["blocked_destinations"] = [["127.0.0.1", 9000]]
        path.write_text(json.dumps(row))
    command = [sys.executable]
    environment = dict(os.environ)
    environment.pop("PYTHONOPTIMIZE", None)
    if optimization == "flag":
        command.append("-O")
    else:
        environment["PYTHONOPTIMIZE"] = "1"
    command += [
        str(Path(__file__).resolve().parents[2] / "scripts/check_project_chat_uat.py"),
        "--junit",
        str(junit),
        "--artifacts",
        str(artifacts),
    ]
    if negative:
        command.append("--negative-control")
    result = subprocess.run(
        command, env=environment, capture_output=True, text=True, timeout=15
    )
    if defect is None:
        assert result.returncode == 0, result.stderr
        assert "Project-chat UAT:" in result.stdout
    else:
        assert result.returncode != 0, result.stdout
        assert "AssertionError" in result.stderr
        assert "Project-chat UAT:" not in result.stdout
