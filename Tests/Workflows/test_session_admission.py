"""Whole-definition refusal must precede every workflow effect."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from Tests.Workflows.helpers import prompt_definition
from tldw_chatbook.Workflows.models import Revision


def revision_of(document):
    identity = document["metadata"]["tldw_workflow"]
    return Revision(
        identity["workflow_id"],
        identity["revision_id"],
        tuple(identity["parent_revision_ids"]),
        json.dumps(document),
    )


def test_retry_is_refused_not_silently_removed():
    from tldw_chatbook.Workflows.session import SessionError, admit_definition

    document = prompt_definition()
    document["steps"][1]["retry"] = 1
    with pytest.raises(SessionError, match="retry_unsupported"):
        admit_definition(revision_of(document))
    assert document["steps"][1]["retry"] == 1


def file_definition():
    return json.loads(Path("Tests/fixtures/workflows/file_to_note.json").read_text())


@pytest.mark.parametrize("factory", [prompt_definition, file_definition])
def test_admitted_document_is_detached_and_preserves_saved_bytes(factory):
    from tldw_chatbook.Workflows.session import admit_definition

    document = factory()
    revision = revision_of(document)
    admitted = admit_definition(revision)
    assert admitted == document
    admitted["steps"].clear()
    assert json.loads(revision.raw_json) == document


@pytest.mark.parametrize(
    "mutation,code",
    [
        (lambda d: d.update(on_complete="private-secret"), "definition_fields"),
        (lambda d: d["steps"][0].update(condition=True), "step_fields"),
        (lambda d: d["steps"][0].update(parallel_group="x"), "step_fields"),
        (lambda d: d["steps"][0].update(retry=True), "retry_unsupported"),
        (lambda d: d["steps"][0].update(timeout_seconds=True), "attempt_timeout"),
        (lambda d: d["steps"][0].update(timeout_seconds=0), "attempt_timeout"),
        (lambda d: d["steps"][0].pop("timeout_seconds"), "attempt_timeout"),
        (lambda d: d["steps"][1].update(id="prepare"), "definition_invalid"),
        (
            lambda d: d["steps"][0]["config"].update(template="{{ finish.text }}"),
            "reference",
        ),
        (
            lambda d: d["steps"][0]["config"].update(template="{{ prepare.text }}"),
            "reference",
        ),
        (
            lambda d: d["steps"][1]["config"].update(template="{{ prepare.missing }}"),
            "reference",
        ),
        (
            lambda d: d["steps"][1]["config"].update(template="{{ inputs.missing }}"),
            "reference",
        ),
        (
            lambda d: d["steps"][0]["config"].update(extra="private-secret"),
            "config_fields",
        ),
        (lambda d: d["metadata"].update(hook="private-secret"), "metadata_fields"),
        (
            lambda d: d["metadata"]["tldw_workflow"].update(capabilities=["shell"]),
            "metadata_fields",
        ),
        (
            lambda d: d["metadata"]["tldw_workflow"].update(
                input_schema={"$ref": "https://private-secret"}
            ),
            "input_schema",
        ),
        (
            lambda d: d["metadata"]["tldw_workflow"].update(
                requirements={"x": {"kind": "network"}}
            ),
            "requirements",
        ),
    ],
)
def test_closed_definition_refusals_are_payload_free(mutation, code):
    from tldw_chatbook.Workflows.session import SessionError, admit_definition

    document = prompt_definition()
    mutation(document)
    revision = revision_of(document)
    with pytest.raises(SessionError, match=code) as error:
        admit_definition(revision)
    assert "private-secret" not in str(error.value)
    assert revision.raw_json == json.dumps(document)


@pytest.mark.parametrize("value", ["1e400", "-0", "0.5"])
def test_display_opaque_numbers_never_become_runtime_data(value):
    from tldw_chatbook.Workflows.session import SessionError, admit_definition

    revision = revision_of(prompt_definition())
    revision = replace(revision, raw_json=revision.raw_json.replace('"hello"', value))
    with pytest.raises(SessionError, match="definition_invalid"):
        admit_definition(revision)


def test_size_and_step_bounds_precede_effects():
    from tldw_chatbook.Workflows.session import SessionError, admit_definition

    document = prompt_definition()
    document["inputs"]["source_text"] = "x" * (2 * 1024 * 1024)
    with pytest.raises(SessionError, match="definition_limit"):
        admit_definition(revision_of(document))
    document = prompt_definition()
    document["steps"] = [dict(document["steps"][0], id=f"step{i}") for i in range(101)]
    with pytest.raises(SessionError, match="step_limit"):
        admit_definition(revision_of(document))


@pytest.mark.parametrize(
    "index,change",
    [
        (0, {"sources": [{"uri": "x"}, {"uri": "y"}]}),
        (0, {"extraction": {"extract_text": False}}),
        (2, {"max_tokens": True}),
        (2, {"request_timeout_seconds": 301}),
        (3, {"timeout_seconds": True}),
        (4, {"action": "update"}),
        (4, {"title": "  "}),
    ],
)
def test_closed_step_configuration(index, change):
    from tldw_chatbook.Workflows.session import SessionError, admit_definition

    document = file_definition()
    document["steps"][index]["config"].update(change)
    with pytest.raises(SessionError):
        admit_definition(revision_of(document))
