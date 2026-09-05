"""Captured evidence comparisons; no drafts or inferred source mappings."""

import json

import pytest

from tldw_chatbook.Chunking.lab_models import ExecutionReport, RunResult
from tldw_chatbook.Chunking.lab_state import (
    capture_batch,
    edit_json,
    new_session,
    replace_sample,
)
from tldw_chatbook.Chunking.template_runtime import execute_prepared


def make_result(text="one two three", body=None):
    session = replace_sample(new_session("test"), text, {"kind": "paste"})
    if body:
        session = edit_json(session, next(iter(session.candidates)), json.dumps(body))
    (request,) = capture_batch(session, tuple(session.candidates))
    return RunResult(
        request=request,
        status="completed",
        report=execute_prepared(request.recipe, request.sample.text),
        started_at="2026-09-04T00:00:00Z",
        finished_at="2026-09-04T00:00:01Z",
        elapsed_ms=1000,
        error=None,
    )


def with_chunks(result, texts, spans=None):
    chunks = tuple(
        {
            "text": text,
            "metadata": {},
            "provenance": {
                "mapping": {
                    "status": "exact" if spans else "unavailable",
                    "reason": "No verified map",
                }
            },
            "span": spans[index] if spans else None,
        }
        for index, text in enumerate(texts)
    )
    return result.model_copy(
        update={
            "report": ExecutionReport(
                chunks=chunks,
                transformed_text=result.request.sample.text,
            )
        }
    )


def test_identity_difference_is_not_itself_incompatibility():
    from tldw_chatbook.Chunking.lab_comparison import comparison_reason

    a = make_result()
    b = a.model_copy(
        update={"request": a.request.model_copy(update={"run_id": "another-run"})}
    )
    assert comparison_reason(a, b) is None


@pytest.mark.parametrize("field", ["backend", "engine_version", "execution_version"])
def test_older_matching_versions_compare_but_mixed_versions_do_not(field):
    from tldw_chatbook.Chunking.lab_comparison import comparison_reason

    a = make_result()
    recipe = a.request.recipe.model_copy(
        update={"runtime": a.request.recipe.runtime.model_copy(update={field: "older"})}
    )
    old = a.model_copy(
        update={"request": a.request.model_copy(update={"recipe": recipe})}
    )
    assert comparison_reason(old, old) is None
    assert field.replace("_", " ") in comparison_reason(a, old).lower()


def test_sample_and_failed_results_have_recovery_reasons():
    from tldw_chatbook.Chunking.lab_comparison import comparison_reason

    a = make_result()
    assert "sample" in comparison_reason(a, make_result("other")).lower()
    failed = a.model_copy(
        update={"status": "failed", "report": None, "error": {"message": "limit"}}
    )
    assert "successful" in comparison_reason(a, failed).lower()


def test_methods_options_and_chunking_assets_are_experimental_variables():
    from tldw_chatbook.Chunking.lab_comparison import comparison_reason, diff_configs

    a = make_result(
        body={"chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}}}
    )
    b = make_result(
        body={
            "chunking": {
                "method": "fixed_size",
                "config": {"max_size": 3, "overlap": 1},
            }
        }
    )
    runtime = b.request.recipe.runtime.model_copy(
        update={
            "assets": (
                {
                    "kind": "tokenizer",
                    "name": "different-local",
                    "version": "2",
                    "content_digest": "abc",
                },
            )
        }
    )
    b = b.model_copy(
        update={
            "request": b.request.model_copy(
                update={
                    "recipe": b.request.recipe.model_copy(update={"runtime": runtime})
                }
            )
        }
    )
    assert comparison_reason(a, b) is None
    assert {
        "path": "/chunking/method",
        "kind": "changed",
        "A": "words",
        "B": "fixed_size",
    } in diff_configs(a, b)


def test_distribution_unicode_words_nearest_rank_and_expansion():
    from tldw_chatbook.Chunking.lab_comparison import summarize_result

    summary = summarize_result(
        with_chunks(make_result("ab"), ["é", "👩‍💻", "a b", "12345"])
    )
    assert summary["characters"] == {
        "minimum": 1,
        "median": 3.0,
        "p95": 5,
        "maximum": 5,
        "total": 12,
    }
    assert summary["words"]["total"] == 5
    assert summary["expansion_ratio"] == 6
    assert summary["overlap_characters"] is None


def test_zero_output_has_unavailable_quantiles_and_no_division_by_zero():
    from tldw_chatbook.Chunking.lab_comparison import summarize_result

    summary = summarize_result(with_chunks(make_result(""), []))
    assert summary["chunk_count"] == 0
    assert summary["characters"] == {
        "minimum": None,
        "median": None,
        "p95": None,
        "maximum": None,
        "total": 0,
    }
    assert summary["expansion_ratio"] is None


def test_measurement_requires_identity_and_does_not_reuse_chunking_tokenizer():
    from tldw_chatbook.Chunking.lab_comparison import (
        comparison_deltas,
        summarize_result,
    )

    result = with_chunks(make_result(), ["a", "b"])
    with pytest.raises(ValueError, match="identity"):
        summarize_result(result, token_counts=(1, 2))
    with pytest.raises(ValueError, match="count"):
        summarize_result(result, token_counts=(1,), measurement_id="local:one")
    a = summarize_result(result, token_counts=(1, 2), measurement_id="local:one")
    b = summarize_result(result, token_counts=(2, 3), measurement_id="local:two")
    assert "tokens" not in comparison_deltas(a, b)
    assert (
        comparison_deltas(
            a, summarize_result(result, token_counts=(2, 3), measurement_id="local:one")
        )["tokens"]
        == 2
    )


def test_unlike_method_budgets_never_become_deltas():
    from tldw_chatbook.Chunking.lab_comparison import (
        comparison_deltas,
        summarize_result,
    )

    a = summarize_result(
        make_result(
            body={
                "chunking": {"method": "words", "config": {"max_size": 2, "overlap": 0}}
            }
        )
    )
    b = summarize_result(
        make_result(
            body={
                "chunking": {
                    "method": "fixed_size",
                    "config": {"max_size": 4, "overlap": 0},
                }
            }
        )
    )
    assert a["budget"]["unit"] == "words"
    assert b["budget"]["unit"] == "characters"
    assert set(comparison_deltas(a, b)) == {"chunk_count", "characters", "words"}


def test_diff_uses_complete_captured_documents_with_position_sensitive_operations():
    from tldw_chatbook.Chunking.lab_comparison import diff_configs

    a = make_result()
    left = {
        "preprocessing": [{"type": "x"}, {"type": "y"}],
        "metadata": {"note": "[bold]" * 100},
        "classifier": {"rules": [1]},
    }
    right = {
        "preprocessing": [{"type": "y"}, {"type": "x"}],
        "metadata": {"added": None},
        "classifier": {"rules": [2]},
    }

    def authored(result, doc):
        return result.model_copy(
            update={
                "request": result.request.model_copy(
                    update={
                        "recipe": result.request.recipe.model_copy(
                            update={"authored_json": json.dumps(doc)}
                        )
                    }
                )
            }
        )

    a, b = authored(a, left), authored(a, right)
    assert diff_configs(a, b) == ()
    diffs = diff_configs(a, b, authored=True)
    assert {
        "path": "/preprocessing/0/type",
        "kind": "changed",
        "A": "x",
        "B": "y",
    } in diffs
    assert {
        "path": "/metadata/note",
        "kind": "removed",
        "A": "[bold]" * 100,
        "B": None,
    } in diffs
    assert {"path": "/metadata/added", "kind": "added", "A": None, "B": None} in diffs
    assert any(d["path"] == "/classifier/rules/0" for d in diffs)


def test_repeated_text_never_gets_guessed_mapping_and_transformed_is_distinct():
    from tldw_chatbook.Chunking.lab_comparison import (
        chunk_mapping,
        linked_chunks,
        summarize_result,
    )

    a = with_chunks(make_result("same same"), ["same"])
    b = with_chunks(a, ["same"], [{"start": 5, "end": 9, "coordinate_space": "source"}])
    assert chunk_mapping(a, 0)["coordinate_space"] is None
    assert linked_chunks(a, 0, b) == ()
    assert linked_chunks(b, 0, b) == (0,)
    changed = with_chunks(
        a, ["same"], [{"start": 0, "end": 4, "coordinate_space": "transformed"}]
    )
    assert chunk_mapping(changed, 0)["coordinate_space"] == "transformed"
    assert linked_chunks(changed, 0, b) == ()
    assert summarize_result(changed)["overlap_characters"] is None


def test_verified_overlap_is_union_based_not_expansion():
    from tldw_chatbook.Chunking.lab_comparison import summarize_result

    result = with_chunks(
        make_result("abcdef"),
        ["abcd", "cdef"],
        [
            {"start": 0, "end": 4, "coordinate_space": "source"},
            {"start": 2, "end": 6, "coordinate_space": "source"},
        ],
    )
    assert summarize_result(result)["overlap_characters"] == 2
