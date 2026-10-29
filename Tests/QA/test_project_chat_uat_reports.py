"""Exact UAT gate contract: no missing/skipped cases or stale partial receipts."""

import json
import runpy
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
    elif defect == "no-screenshot":
        path.with_name("viewport.svg").unlink()
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
