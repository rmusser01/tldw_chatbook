"""Native qualification admission and the actual read-only front classifier."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
ROOT = Path(__file__).resolve().parents[2]
LANES = (
    "pr-fast-lane",
    "pr-fast-lane-admission",
    "pr-fast-lane-admission-2",
    "ui-fast-lane",
    "console-p0-gate",
    "dreams-boundary",
    "dreams-tracking",
)
HARNESS = r"""
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const calls = [], warnings = [], outputs = {};
let page = 0;
const github = {async graphql(query, args) {
  calls.push({query, args});
  if (input.error) throw new Error('API unavailable');
  if (args.number) return {repository: {pullRequest: input.live}};
  return {repository: {pullRequests: input.pages[page++]}};
}};
const core = {setOutput(k,v) {outputs[k]=String(v);}, warning(s) {warnings.push(s);}};
const context = {repo: {owner:'owner', repo:'repo'}, eventName: input.eventName,
  payload: {pull_request: input.event}};
process.env.MERGE_QUEUE = input.mode;
(async () => {
  let error = null;
  try {await new (Object.getPrototypeOf(async function(){}).constructor)(
    'github','context','core',input.script)(github,context,core);}
  catch(e) {error=e.message;}
  process.stdout.write(JSON.stringify({calls,warnings,outputs,error}));
})();
"""


def _workflow(name="derived-artifacts.yml"):
    return yaml.safe_load(
        (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")
    )


def _classifier():
    return next(
        s
        for s in _workflow()["jobs"]["preflight"]["steps"]
        if s.get("id") == "admission"
    )


def _pr(number=42, **changes):
    return {
        "number": number,
        "state": "OPEN",
        "isDraft": False,
        "baseRefName": "dev",
        "headRefOid": "a" * 40,
        "headRefName": f"feat/{number}",
        "headRepository": {"nameWithOwner": "owner/repo"},
        "author": {"__typename": "User"},
        "autoMergeRequest": {"enabledAt": "2026-10-30T00:00:00Z"},
        **changes,
    }


def _page(nodes, cursor=None):
    return {
        "nodes": nodes,
        "pageInfo": {"hasNextPage": cursor is not None, "endCursor": cursor},
    }


def _run(pages=None, **extra):
    node = shutil.which("node")
    assert node, "Node executes the real classifier"
    payload = {
        "script": _classifier()["with"]["script"],
        "pages": pages or [_page([_pr()])],
        "live": _pr(),
        "mode": "on",
        "eventName": "pull_request",
        "event": {"number": 42, "head": {"sha": "a" * 40, "ref": "feat/42"}},
        **extra,
    }
    result = subprocess.run(
        [node, "-e", HARNESS],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=15,
        check=True,
    )
    return json.loads(result.stdout)


def test_front_is_verified_after_complete_pagination():
    result = _run([_page([_pr(50)], "next"), _page([_pr()])])
    assert result["error"] is None and result["outputs"] == {"queue_front": "true"}
    assert len(result["calls"]) == 3
    assert result["calls"][1]["args"]["after"] == "next"
    assert result["calls"][-1]["args"]["number"] == 42
    assert all("mutation" not in c["query"].lower() for c in result["calls"])


@pytest.mark.parametrize(
    "older",
    [
        _pr(41),
        _pr(99, autoMergeRequest={"enabledAt": "2026-10-29T23:59:00Z"}),
    ],
)
def test_queue_order_matches_arming_time_then_pr_number(older):
    assert _run([_page([_pr(), older])])["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "changes",
    [
        {"autoMergeRequest": None},
        {"isDraft": True},
        {"author": {"__typename": "Bot"}},
        {"author": None},
        {"headRepository": {"nameWithOwner": "fork/repo"}},
        {"headRepository": None},
        {"state": "CLOSED"},
        {"baseRefName": "main"},
    ],
)
def test_ineligible_pr_does_not_displace_the_front(changes):
    assert _run([_page([_pr(1, **changes), _pr()])])["outputs"] == {
        "queue_front": "true"
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"headRefOid": "b" * 40},
        {"headRefName": "different"},
        {"state": "CLOSED"},
        {"isDraft": True},
        {"autoMergeRequest": None},
        {"autoMergeRequest": {"enabledAt": "2026-10-30T00:01:00Z"}},
        {"headRepository": {"nameWithOwner": "fork/repo"}},
        {"baseRefName": "main"},
        {"author": {"__typename": "Bot"}},
        {"number": 9},
    ],
)
def test_live_candidate_changes_keep_the_background_slot(changes):
    assert _run(live=_pr(**changes))["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "extra",
    [
        {"mode": "off"},
        {"mode": "dry"},
        {"eventName": "workflow_dispatch"},
        {"eventName": "push"},
        {"event": {"number": 42, "head": {"sha": "b" * 40, "ref": "feat/42"}}},
    ],
)
def test_other_events_modes_or_heads_cannot_claim_front(extra):
    assert _run(**extra)["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "pages,extra",
    [
        ([_page([_pr()])], {"error": True}),
        ([{}], {}),
        ([_page([_pr(autoMergeRequest={"enabledAt": "bad"})])], {}),
        (
            [{"nodes": [_pr()], "pageInfo": {"hasNextPage": True, "endCursor": None}}],
            {},
        ),
        ([_page([_pr()], "repeat")] * 10, {}),
        ([_page([_pr()])], {"live": None}),
    ],
)
def test_api_or_incomplete_metadata_falls_back_without_failing_preflight(pages, extra):
    result = _run(pages, **extra)
    assert result["error"] is None
    assert result["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "field",
    ["state", "isDraft", "baseRefName", "author", "headRepository", "autoMergeRequest"],
)
def test_unknown_earlier_pr_metadata_cannot_prove_this_run_is_front(field):
    earlier = _pr(1)
    del earlier[field]
    assert _run([_page([earlier, _pr()])])["outputs"] == {"queue_front": "false"}


def test_snapshot_bound_and_repeated_pr_fail_closed():
    pages = [_page([_pr(50 + i)], str(i)) for i in range(10)]
    assert _run(pages)["outputs"] == {"queue_front": "false"}
    assert _run([_page([_pr(), _pr()])])["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "timestamp", ["1", "2026-02-30T00:00:00Z", "2026-10-30", "2026-10-30T24:00:00Z"]
)
def test_malformed_arming_timestamp_cannot_grant_front(timestamp):
    pr = _pr(autoMergeRequest={"enabledAt": timestamp})
    assert _run([_page([pr])], live=pr)["outputs"] == {"queue_front": "false"}


@pytest.mark.parametrize(
    "changes",
    [
        {"author": {"__typename": ""}},
        {"headRepository": {"nameWithOwner": ""}},
        {"state": "UNKNOWN"},
        {"baseRefName": ""},
    ],
)
def test_invalid_earlier_eligibility_metadata_cannot_grant_front(changes):
    assert _run([_page([_pr(1, **changes), _pr()])])["outputs"] == {
        "queue_front": "false"
    }


def test_manual_pr_input_and_all_lane_conditions_are_preserved():
    parent, child = _workflow(), _workflow("ci-qualification.yml")
    caller = parent["jobs"]["qualification"]
    assert child.get("on", child.get(True))["workflow_call"]["inputs"]["pr"] == {
        "type": "string",
        "required": False,
        "default": "",
    }
    assert caller["with"]["pr"] == "${{ inputs.pr || '' }}"
    assert all(job["if"] == caller["if"] for job in child["jobs"].values())


def assert_graph(parent, child):
    jobs = parent["jobs"]
    assert list(jobs) == [
        "preflight",
        "qualification",
        "derived-artifacts",
        "queue-tick",
    ]
    call = jobs["qualification"]
    assert call["uses"] == "./.github/workflows/ci-qualification.yml"
    assert call["needs"] == ["preflight"]
    assert call["with"] == {
        "queue_front": "${{ needs.preflight.outputs.queue_front == 'true' }}",
        "pr": "${{ inputs.pr || '' }}",
    }
    assert call["concurrency"] == {
        "group": "ci-qualification-${{ needs.preflight.outputs.queue_front == 'true' && 'front' || 'background' }}",
        "queue": "max",
        "cancel-in-progress": False,
    }
    assert not call.get("continue-on-error", False)
    assert jobs["derived-artifacts"]["needs"] == ["preflight", "qualification"]
    assert (
        jobs["derived-artifacts"]["name"]
        == "Derived artifacts reproduce from their sources"
    )
    assert jobs["derived-artifacts"]["if"] == "${{ always() }}"
    required = jobs["derived-artifacts"]["steps"]
    wrapper = next(
        s for s in required if s.get("name") == "Require successful qualification"
    )
    assert wrapper["if"].startswith("${{ always() && ")
    assert (
        "needs.qualification.result != 'success'" in wrapper["if"]
        and "exit 1" in wrapper["run"]
    )
    on = child.get("on", child.get(True))
    outputs = on["workflow_call"]["outputs"]
    assert set(child["jobs"]) == set(outputs) == set(LANES)
    for name in LANES:
        assert outputs[name]["value"] == "${{ jobs." + name + ".result }}"
        verdict = next(
            s
            for s in required
            if f"needs.qualification.outputs.{name} != 'success'" in s.get("if", "")
        )
        assert "exit 1" in verdict["run"] and not verdict.get(
            "continue-on-error", False
        )
        assert "needs" not in child["jobs"][name]
    assert (
        child["jobs"]["ui-fast-lane"]["strategy"]["max-parallel"]
        == "${{ inputs.queue_front && 17 || 8 }}"
    )
    assert child["jobs"]["ui-fast-lane"]["strategy"]["matrix"] == {
        "shard": list(range(1, 29))
    }
    assert "concurrency" not in child and "concurrency" not in jobs["queue-tick"]
    assert jobs["preflight"]["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
    }
    assert jobs["preflight"]["outputs"] == {
        "queue_front": "${{ steps.admission.outputs.queue_front }}"
    }


def test_native_slots_preserve_required_lane_results():
    assert_graph(_workflow(), _workflow("ci-qualification.yml"))


@pytest.mark.parametrize(
    "damage", ["bypass", "cancel", "single", "per-pr", "missing-output", "only-failure"]
)
def test_graph_rejects_admission_or_required_gate_bypasses(damage):
    parent, child = (
        copy.deepcopy(_workflow()),
        copy.deepcopy(_workflow("ci-qualification.yml")),
    )
    call = parent["jobs"]["qualification"]
    if damage == "bypass":
        call["needs"] = []
    elif damage == "cancel":
        call["concurrency"]["cancel-in-progress"] = True
    elif damage == "single":
        call["concurrency"]["queue"] = "single"
    elif damage == "per-pr":
        call["concurrency"]["group"] += "-${{ github.ref }}"
    elif damage == "missing-output":
        child.get("on", child.get(True))["workflow_call"]["outputs"].pop("ui-fast-lane")
    else:
        step = next(
            s
            for s in parent["jobs"]["derived-artifacts"]["steps"]
            if s.get("name") == "Require successful qualification"
        )
        step["if"] = step["if"].replace("!= 'success'", "== 'failure'")
    with pytest.raises((AssertionError, StopIteration)):
        assert_graph(parent, child)
