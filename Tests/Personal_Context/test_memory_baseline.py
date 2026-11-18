"""Independent contracts for the synthetic Personal Context baseline."""

from __future__ import annotations

import importlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest

MANIFEST = Path(__file__).parent / "fixtures" / "memory_baseline_v1.json"


def api():
    name = "Tests.Personal_Context.memory_baseline"
    assert importlib.util.find_spec(name) is not None, "baseline module is missing"
    return importlib.import_module(name)


def test_denied_tool_contract_uses_error_without_json_body():
    from tldw_chatbook.Agents.agent_models import ToolResult

    result = api()._response(
        ToolResult(ok=False, error="permission_denied"), "get", "permission_denied"
    )
    assert result == {"operation": "get", "status": "permission_denied"}


def test_successful_tool_contract_cannot_carry_an_error():
    from tldw_chatbook.Agents.agent_models import ToolResult

    result = ToolResult(
        ok=True,
        content=json.dumps({"operation": "search", "status": "applied"}),
        error="unexpected error detail",
    )
    with pytest.raises(api().BaselineContractError):
        api()._response(result, "search", "applied")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"ok": True, "error": "permission_denied"},
        {"ok": False, "error": "not_found"},
        {"ok": False, "error": "permission_denied", "content": "unexpected data"},
    ],
)
def test_denial_must_be_exact_and_body_free(kwargs):
    from tldw_chatbook.Agents.agent_models import ToolResult

    with pytest.raises(api().BaselineContractError):
        api()._response(ToolResult(**kwargs), "get", "permission_denied")


def test_search_metrics_use_fixed_k_and_distinct_relevance():
    result = api().score_search(("noise", "brief"), frozenset({"brief", "local"}), k=3)
    assert result == {
        "precision_at_k": pytest.approx(1 / 3),
        "recall_at_k": 0.5,
        "reciprocal_rank": 0.5,
        "false_positive_count": 1,
        "empty_result": False,
    }


def test_empty_relevance_is_not_perfect_recall():
    result = api().score_search(("noise",), frozenset(), k=3)
    assert result["recall_at_k"] is None
    assert result["reciprocal_rank"] is None
    assert result["false_positive_count"] == 1
    assert result["empty_result"] is False


def test_no_hits_with_relevance_scores_zero():
    result = api().score_search((), frozenset({"brief"}), k=3)
    assert result["precision_at_k"] == result["recall_at_k"] == 0
    assert result["reciprocal_rank"] == 0
    assert result["empty_result"] is True


@pytest.mark.parametrize("k", [0, 21, True])
def test_metrics_reject_invalid_k(k):
    with pytest.raises(ValueError, match="k"):
        api().score_search((), frozenset(), k=k)


def test_duplicate_results_are_invalid_not_extra_credit():
    with pytest.raises(ValueError, match="duplicate"):
        api().score_search(("brief", "brief"), frozenset({"brief"}), k=3)


def test_metrics_reject_more_than_k_results():
    with pytest.raises(ValueError, match="exceeds"):
        api().score_search(("a", "b"), frozenset(), k=1)


def test_manifest_is_frozen_and_contains_both_partitions():
    manifest = api().load_manifest(MANIFEST)
    assert manifest["synthetic"] is True
    assert manifest["k"] == 3
    assert [row["id"] for row in manifest["cases"]] == [
        f"{prefix}{index:02d}" for prefix in ("d", "h") for index in range(1, 13)
    ]
    assert {row["split"] for row in manifest["cases"][:12]} == {"development"}
    assert {row["split"] for row in manifest["cases"][12:]} == {"held_out"}
    assert len(manifest["records"]["large_bytes"]["payload"]["value"]) == 13_000


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate_id",
        "unknown_label",
        "unsupported_action",
        "bad_relevance",
        "bool_k",
        "unknown_key",
        "path_id",
        "bad_split",
        "bad_action_target",
        "bad_record_scope",
        "bad_record_kind",
        "bad_message_ref",
        "bad_metadata",
    ],
)
def test_manifest_rejects_invalid_nested_contract(tmp_path, mutation):
    manifest = deepcopy(api().load_manifest(MANIFEST))
    first = manifest["cases"][0]
    if mutation == "duplicate_id":
        manifest["cases"][1]["id"] = first["id"]
    elif mutation == "unknown_label":
        first["record_labels"] = ["absent"]
    elif mutation == "unsupported_action":
        first["actions"] = [{"op": "execute", "command": "ignored"}]
    elif mutation == "bad_relevance":
        first["relevant_labels"] = ["control"]
    elif mutation == "bool_k":
        manifest["k"] = True
    elif mutation == "unknown_key":
        manifest["provider"] = "remote"
    elif mutation == "path_id":
        first["id"] = "../d01"
    elif mutation == "bad_split":
        first["split"] = "held_out"
    elif mutation == "bad_action_target":
        first["actions"] = [{"op": "delete", "label": "other"}]
    elif mutation == "bad_record_scope":
        manifest["records"]["brief"]["scope"] = "unknown"
    elif mutation == "bad_record_kind":
        manifest["records"]["brief"]["payload"]["kind"] = "execute"
    elif mutation == "bad_message_ref":
        manifest["records"]["ambiguous"]["provenance"]["source_references"] = ["absent"]
    elif mutation == "bad_metadata":
        first["metadata_expectations"] = {"other": {"source_references": []}}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        api().load_manifest(path)


@pytest.mark.parametrize("content", [b"{", b"x" * (256 * 1024 + 1), b"\xff"])
def test_manifest_rejects_malformed_or_oversized_input(tmp_path, content):
    path = tmp_path / "bad.json"
    path.write_bytes(content)
    with pytest.raises(ValueError):
        api().load_manifest(path)


def test_production_runner_is_available():
    assert callable(getattr(api(), "run_suite", None)), "production runner is missing"


@pytest.fixture(scope="module")
def completed_report(tmp_path_factory):
    return api().run_suite(MANIFEST, tmp_path_factory.mktemp("memory-baseline"))


def row(report, case_id):
    return next(case for case in report["cases"] if case["id"] == case_id)


def test_write_memory_baseline_report(completed_report):
    import os

    assert completed_report["harness_errors"] == []
    destination = os.environ.get("TLDW_MEMORY_BASELINE_REPORT")
    if destination:
        api().write_report(completed_report, Path(destination))


def test_private_case_has_a_successful_visible_control(completed_report):
    case = row(completed_report, "d05")
    assert case["search_status"] == "applied"
    assert case["returned_labels"] == ["control"]
    assert case["selected_labels"] == ["control"]
    assert case["positive_control_passed"] is True
    assert case["authority_failures"] == []


def test_workspace_exception_is_not_a_search_permission_filter(completed_report):
    case = row(completed_report, "d06")
    assert set(case["returned_labels"]) == {"brief", "local"}
    assert case["selected_labels"] == ["local"]


@pytest.mark.parametrize("case_id", ["h05", "h06", "h07", "h08", "h10", "h11"])
def test_lifecycle_and_authority_use_positive_controls(completed_report, case_id):
    case = row(completed_report, case_id)
    assert case["positive_control_passed"] is True
    assert case["harness_errors"] == case["authority_failures"] == []
    assert case["context_checks"] == []
    assert case["selected_labels"] == ([] if case_id == "h10" else ["control"])
    if case_id == "h10":
        assert case["search_status"] == "permission_denied"
        assert case["search_metrics"] is None


@pytest.mark.parametrize("case_id", ["d08", "d12", "h12"])
def test_context_budgets_are_whole_record_checks(completed_report, case_id):
    case = row(completed_report, case_id)
    assert case["context_checks"] == []
    assert case["context_bytes"] <= 12_288
    assert case["selected_labels"] == ([] if case_id == "d08" else ["control"])


def test_metadata_is_observed_but_support_remains_unmeasured(completed_report):
    assert row(completed_report, "d09")["metadata"]["brief"]["source_references"] == []
    assert row(completed_report, "h09")["metadata"]["ambiguous"][
        "source_references"
    ] == ["message-a", "message-b"]
    assert completed_report["unmeasured"] == [
        "provenance_ui",
        "semantic_support",
        "edit_history",
        "answer_quality",
    ]


def test_baseline_keeps_current_policy_gaps_visible(completed_report):
    # Observe actual behavior without requiring it to stay broken after future fixes.
    for case_id, label, gap in [("d10", "device", "device_only_in_provider_context")]:
        case = row(completed_report, case_id)
        assert (gap in case["policy_gaps"]) == (label in case["selected_labels"])
    assert completed_report["harness_errors"] == []
    assert completed_report["disclosure_checks_passed"] == (
        not completed_report["authority_failures"]
        and not completed_report["policy_gaps"]
    )


def test_runner_calls_real_search_and_snapshot(tmp_path, monkeypatch):
    baseline = api()
    calls = []
    invoke = baseline.ProfileToolProvider.invoke
    build = baseline.ProfileContextService.build_snapshot

    def spy_invoke(self, name, args):
        calls.append(name)
        return invoke(self, name, args)

    def spy_build(self, request):
        calls.append("snapshot")
        return build(self, request)

    monkeypatch.setattr(baseline.ProfileToolProvider, "invoke", spy_invoke)
    monkeypatch.setattr(baseline.ProfileContextService, "build_snapshot", spy_build)
    manifest = baseline.load_manifest(MANIFEST)
    result = baseline.run_case(manifest["cases"][0], manifest, tmp_path / "case")
    assert result["returned_labels"] == ["brief"]
    assert calls.count("profile_search") >= 2
    assert "profile_get" in calls and calls.count("snapshot") >= 2


def test_forbidden_canaries_in_control_responses_are_not_ignored(tmp_path, monkeypatch):
    from dataclasses import replace

    baseline = api()
    invoke = baseline.ProfileToolProvider.invoke

    def leak_in_control(self, name, args):
        result = invoke(self, name, args)
        if name == "profile_search" and args["query"] == "response.control":
            payload = json.loads(result.content)
            payload["diagnostic"] = "PRIVATE_MARKER"
            return replace(result, content=json.dumps(payload))
        return result

    monkeypatch.setattr(baseline.ProfileToolProvider, "invoke", leak_in_control)
    manifest = baseline.load_manifest(MANIFEST)
    case = next(case for case in manifest["cases"] if case["id"] == "d05")
    measured = baseline.run_case(case, manifest, tmp_path / "case")
    assert measured["positive_control_passed"] is True
    assert measured["authority_failures"] == ["forbidden_record:private"]


@pytest.mark.parametrize(
    "poison",
    [
        "unknown_id",
        "duplicate_id",
        "malformed_json",
        "denied",
        "unknown_version",
        "mismatched_payload",
        "get_data",
    ],
)
def test_bad_production_outputs_cannot_become_successful_empty_results(
    tmp_path, monkeypatch, poison
):
    from dataclasses import replace

    baseline = api()
    invoke = baseline.ProfileToolProvider.invoke
    build = baseline.ProfileContextService.build_snapshot

    def poisoned_invoke(self, name, args):
        result = invoke(self, name, args)
        if name == "profile_search" and args["query"] == "concise":
            payload = json.loads(result.content)
            if poison == "unknown_id":
                payload["data"]["records"][0]["record_id"] = "unrecognized"
            elif poison == "duplicate_id":
                payload["data"]["records"] *= 2
            elif poison == "malformed_json":
                return replace(result, content="{")
            elif poison == "denied":
                return replace(result, ok=False)
            result = replace(result, content=json.dumps(payload))
        if name == "profile_get" and poison == "get_data":
            return replace(result, content="{}")
        return result

    def poisoned_build(self, request):
        result = build(self, request)
        if request.available_input_tokens == 20_000:
            if poison == "unknown_version":
                return replace(result, source_version_ids=("unknown-version",))
            if poison == "mismatched_payload":
                return replace(
                    result,
                    serialized_block=result.serialized_block.replace(
                        "concise replies", "different text"
                    ),
                )
        return result

    monkeypatch.setattr(baseline.ProfileToolProvider, "invoke", poisoned_invoke)
    monkeypatch.setattr(
        baseline.ProfileContextService, "build_snapshot", poisoned_build
    )
    manifest = baseline.load_manifest(MANIFEST)
    result = baseline.run_case(manifest["cases"][0], manifest, tmp_path / "case")
    assert result["harness_errors"]
    assert result["search_metrics"] is None


def test_tokenizer_state_is_restored_on_success_and_error(
    tmp_path, monkeypatch, completed_report
):
    baseline = api()
    original = (
        baseline.token_counter.TIKTOKEN_AVAILABLE,
        baseline.token_counter.CUSTOM_TOKENIZERS_AVAILABLE,
    )
    cases = {case["id"]: case for case in completed_report["cases"]}
    monkeypatch.setattr(
        baseline, "run_case", lambda case, manifest, root: deepcopy(cases[case["id"]])
    )
    baseline.run_suite(MANIFEST, tmp_path / "success")
    assert original == (
        baseline.token_counter.TIKTOKEN_AVAILABLE,
        baseline.token_counter.CUSTOM_TOKENIZERS_AVAILABLE,
    )

    def fail(*args):
        raise RuntimeError("injected runner failure")

    monkeypatch.setattr(baseline, "run_case", fail)
    with pytest.raises(RuntimeError, match="injected"):
        baseline.run_suite(MANIFEST, tmp_path / "failure")
    assert original == (
        baseline.token_counter.TIKTOKEN_AVAILABLE,
        baseline.token_counter.CUSTOM_TOKENIZERS_AVAILABLE,
    )


def test_report_writer_uses_stable_utf8_json(tmp_path):
    output = tmp_path / "report.json"
    report = {"z": None, "a": "東京"}
    api().write_report(report, output)
    assert output.read_text(encoding="utf-8") == '{\n  "a": "東京",\n  "z": null\n}\n'


@pytest.mark.parametrize("destination", ["existing", "directory", "missing_parent"])
def test_report_writer_refuses_unsafe_destinations(tmp_path, destination):
    output = tmp_path / "output"
    if destination == "existing":
        output.write_text("retain existing evidence")
    elif destination == "directory":
        output.mkdir()
    else:
        output = tmp_path / "absent" / "output.json"
    with pytest.raises(OSError):
        api().write_report({"synthetic": True}, output)
    if destination == "existing":
        assert output.read_text() == "retain existing evidence"
    elif destination == "directory":
        assert output.is_dir()
    else:
        assert not output.parent.exists()


def test_report_writer_removes_only_its_partial_new_output(tmp_path, monkeypatch):
    from contextlib import contextmanager

    output = tmp_path / "report.json"
    original = Path.open

    @contextmanager
    def fail_write(path, *args, **kwargs):
        with original(path, *args, **kwargs) as stream:

            class BrokenWriter:
                def write(self, text):
                    stream.write(text[:3])
                    raise OSError("injected disk failure")

            yield BrokenWriter()

    monkeypatch.setattr(Path, "open", fail_write)
    with pytest.raises(OSError, match="injected disk failure"):
        api().write_report({"synthetic": True}, output)
    assert not output.exists()


def test_report_writer_serializes_before_creating_output(tmp_path):
    output = tmp_path / "report.json"
    with pytest.raises(TypeError):
        api().write_report({"invalid": object()}, output)
    assert not output.exists()


def test_report_hash_describes_the_manifest_actually_loaded(tmp_path, monkeypatch):
    import hashlib

    path = tmp_path / "manifest.json"
    original = MANIFEST.read_bytes()
    path.write_bytes(original)

    def measured(case, manifest, root):
        path.write_bytes(original + b"\n")
        return {
            "id": case["id"],
            "split": case["split"],
            "category": case["category"],
            "search_metrics": None,
            "harness_errors": [],
            "authority_failures": [],
            "policy_gaps": [],
            "context_checks": [],
        }

    monkeypatch.setattr(api(), "run_case", measured)
    report = api().run_suite(path, tmp_path / "cases")
    assert report["fixture_sha256"] == hashlib.sha256(original).hexdigest()


def test_reports_are_reproducible_across_temporary_roots(tmp_path, completed_report):
    second = api().run_suite(MANIFEST, tmp_path / "second")
    output = tmp_path / "second-report.json"
    api().write_report(second, output)
    assert json.loads(output.read_text(encoding="utf-8")) == completed_report


def test_report_aggregation_preserves_failure_and_clean_control(
    tmp_path, monkeypatch, completed_report
):
    cases = {case["id"]: deepcopy(case) for case in completed_report["cases"]}
    for case in cases.values():
        for key in (
            "harness_errors",
            "authority_failures",
            "policy_gaps",
            "context_checks",
        ):
            case[key] = []
    monkeypatch.setattr(
        api(), "run_case", lambda case, manifest, root: deepcopy(cases[case["id"]])
    )
    for field in ("harness_errors", "authority_failures", "policy_gaps"):
        cases["d01"][field] = ["injected_failure"]
        failed = api().run_suite(MANIFEST, tmp_path / field)
        assert failed["disclosure_checks_passed"] is False
        assert failed[field] == [{"case": "d01", "code": "injected_failure"}]
        cases["d01"][field] = []
    clean = api().run_suite(MANIFEST, tmp_path / "clean")
    assert clean["disclosure_checks_passed"] is True
    assert clean["context_checks_passed"] is True


def test_report_does_not_include_runtime_identifiers_or_source_bodies(completed_report):
    import re

    raw = json.dumps(completed_report)
    assert re.search(r"(?:record|version|scope|profile)-\d{4}", raw) is None
    for forbidden in (
        "PRIVATE_MARKER",
        "OTHER_MARKER",
        "DEVICE_MARKER",
        "concise replies",
        "synthetic-profile.db",
        str(MANIFEST.parent),
    ):
        assert forbidden not in raw


def test_quarantine_case_has_no_unscoped_signal(completed_report) -> None:
    case = row(completed_report, "h11")
    assert case["positive_control_passed"] is True
    assert case["selected_labels"] == ["control"]
    assert case["harness_errors"] == case["authority_failures"] == []
    assert "unscoped_quarantine_signal" not in case["policy_gaps"]
    assert all(
        gap["code"] != "unscoped_quarantine_signal"
        for gap in completed_report["policy_gaps"]
    )


def test_device_only_correction_keeps_frozen_historical_mismatch_visible(
    completed_report,
):
    case = row(completed_report, "d10")
    assert case["returned_labels"] == case["selected_labels"] == ["control"]
    assert case["positive_control_passed"] is True
    assert (
        case["policy_gaps"]
        == case["authority_failures"]
        == case["harness_errors"]
        == []
    )
    assert case["context_checks"] == ["selection_mismatch"]
    assert case["search_metrics"]["recall_at_k"] == 0.5
    frozen = next(
        case for case in api().load_manifest(MANIFEST)["cases"] if case["id"] == "d10"
    )
    assert (
        frozen["expected_context_labels"]
        == frozen["relevant_labels"]
        == ["control", "device"]
    )


def test_device_only_case_uses_syncable_control_without_changing_labels(tmp_path):
    baseline = api()
    manifest = baseline.load_manifest(MANIFEST)
    case = next(case for case in manifest["cases"] if case["id"] == "d10")
    measured = baseline.run_case(case, manifest, tmp_path / "device-case")
    assert measured["positive_control_passed"] is True
    assert measured["harness_errors"] == []
    assert measured["returned_labels"] == measured["selected_labels"] == ["control"]
    assert measured["context_checks"] == ["selection_mismatch"]
    assert measured["search_metrics"]["recall_at_k"] == 0.5
    assert (
        case["expected_context_labels"]
        == case["relevant_labels"]
        == ["control", "device"]
    )
