"""ADR-359: opt-in local capacity must preserve hosted and fork routing."""

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

WORKFLOWS = Path(__file__).resolve().parents[2] / ".github" / "workflows"
TEST_JOBS = (
    "pr-fast-lane",
    "pr-fast-lane-admission",
    "pr-fast-lane-admission-2",
    "pr-fast-lane-admission-3",
    "ui-fast-lane",
    "console-p0-gate",
    "dreams-boundary",
    "dreams-tracking",
    "architecture-guards",
    "pr-fast-lane-arch",
    "recovered-suites",
)
ROUTES = [
    ("derived-artifacts.yml", job, "test", "ubuntu-latest") for job in TEST_JOBS
] + [
    ("derived-artifacts.yml", "preflight", "control", "ubuntu-24.04"),
    ("derived-artifacts.yml", "derived-artifacts", "control", "ubuntu-latest"),
    ("derived-artifacts.yml", "queue-tick", "control", "ubuntu-latest"),
    ("merge-queue.yml", "cancel-superseded", "control", "ubuntu-latest"),
    ("merge-queue.yml", "queue", "control", "ubuntu-latest"),
    ("lint.yml", "gate-new", "control", "ubuntu-latest"),
    ("lint.yml", "gate-ratchet", "control", "ubuntu-latest"),
    ("derived-artifacts.yml", "library-shell-dev", "hosted", "ubuntu-latest"),
]


@pytest.fixture(scope="module")
def workflows():
    return {
        name: yaml.safe_load((WORKFLOWS / name).read_text())
        for name in {route[0] for route in ROUTES}
    }


def _runner(expression: str, context: dict[str, str]) -> str:
    if not expression.startswith("${{"):
        return expression
    expression = expression.removeprefix("${{").removesuffix("}}").strip()
    for name, value in context.items():
        expression = expression.replace(name, repr(value))
    # These expressions use only string equality and short-circuit operators,
    # whose value-returning semantics match Python for these string inputs.
    return eval(
        expression.replace("&&", " and ").replace("||", " or "), {"__builtins__": {}}
    )


@pytest.mark.parametrize("workflow,job,pool,hosted", ROUTES)
@pytest.mark.parametrize(
    "event,head_repo",
    [
        ("pull_request", "owner/repo"),
        ("pull_request", "fork/repo"),
        ("pull_request", ""),
        ("push", ""),
        ("workflow_dispatch", ""),
    ],
)
@pytest.mark.parametrize("test_label", ["", "local-test"])
@pytest.mark.parametrize("control_label", ["", "local-control"])
def test_runner_routing(
    workflows, workflow, job, pool, hosted, event, head_repo, test_label, control_label
):
    selected = _runner(
        workflows[workflow]["jobs"][job]["runs-on"],
        {
            "github.event_name": event,
            "github.event.pull_request.head.repo.full_name": head_repo,
            "github.repository": "owner/repo",
            "vars.CI_TEST_RUNNER": test_label,
            "vars.CI_CONTROL_RUNNER": control_label,
        },
    )
    expected = hosted
    if event == "pull_request":
        if head_repo == "owner/repo":
            if pool == "test":
                expected = test_label or hosted
            elif pool == "control":
                expected = control_label or hosted
    elif pool == "control":
        expected = control_label or hosted
    assert selected == expected
