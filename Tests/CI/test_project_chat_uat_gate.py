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
    assert upload["if"] == "always()" and upload["with"]["if-no-files-found"] == "error"
    for artifact in (
        "project-chat-uat/",
        "project-chat-uat.xml",
        "pytest.log",
        "stacks.txt",
        "tldw_cli_app.log",
    ):
        assert artifact in upload["with"]["path"]
