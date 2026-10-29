from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_project_chat_uat_is_required_without_extra_runners():
    workflow = yaml.safe_load(
        (ROOT / ".github/workflows/derived-artifacts.yml").read_text(encoding="utf-8")
    )
    jobs = workflow["jobs"]
    assert "pr-fast-lane-admission-2" in jobs["derived-artifacts"]["needs"]
    job = jobs["pr-fast-lane-admission-2"]
    assert job["timeout-minutes"] == 35
    steps = {step.get("name"): step for step in job["steps"]}
    run = steps["Run project-chat UAT"]
    assert "if" not in run and not run.get("continue-on-error")
    assert run["timeout-minutes"] == 18
    for name in (
        "test_project_chat_uat.py",
        "test_project_chat_uat_support.py",
        "test_project_chat_uat_reports.py",
    ):
        assert f"Tests/QA/{name}" in run["run"]
    assert "-x" in run["run"].split()
    assert job["steps"][3] is run
    assert "--junitxml=project-chat-uat.xml" in run["run"]
    assert "--basetemp=project-chat-uat-pytest" in run["run"]
    assert run["env"] == {"TLDW_UAT_ARTIFACTS": "project-chat-uat"}
    gate = steps["Require complete project-chat UAT evidence"]
    assert gate["if"] == "always()" and not gate.get("continue-on-error")
    assert (
        gate["run"]
        == "python scripts/check_project_chat_uat.py --junit project-chat-uat.xml --artifacts project-chat-uat"
    )
    upload = steps["Retain project-chat UAT evidence"]
    assert upload["with"]["include-hidden-files"] is True
    assert upload["if"] == "always()" and upload["with"]["if-no-files-found"] == "error"
    for artifact in (
        "project-chat-uat/",
        "project-chat-uat.xml",
        "pytest.log",
        "stacks.txt",
        "tldw_cli_app.log",
    ):
        assert artifact in upload["with"]["path"]

    negative = steps["Prove missing project evidence fails UAT"]
    assert "if" not in negative and not negative.get("continue-on-error")
    assert negative["timeout-minutes"] == 5
    assert negative["env"] == {
        "TLDW_UAT_ARTIFACTS": "project-chat-negative",
        "TLDW_UAT_NEGATIVE_CONTROL": "missing-readme",
    }
    assert 'test "$status" -eq 1' in negative["run"]
    assert "scripts/check_project_chat_uat.py --negative-control" in negative["run"]
    assert "project-chat-negative/" in upload["with"]["path"]
