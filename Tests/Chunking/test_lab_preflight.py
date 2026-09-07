"""Lab admission must refuse work that the parity validator silently accepts."""

import importlib
import importlib.util
import json
import socket

import pytest


def preflight():
    name = "tldw_chatbook.Chunking.lab_preflight"
    assert importlib.util.find_spec(name) is not None, "Lab capability gate is missing"
    return importlib.import_module(name)


def test_unknown_operation_cannot_be_silently_skipped():
    gate = preflight()
    body = {
        "chunking": {"method": "words", "config": {"max_size": 4}},
        "preprocessing": [{"operation": "unregistered_operation", "config": {}}],
    }
    with pytest.raises(gate.PreviewUnsupportedError, match="preprocessing"):
        gate.prepare_recipe(body, runtime=gate.current_local_runtime())


@pytest.mark.parametrize(
    "body,field",
    [
        ({"chunking": {"method": "words"}, "pipeline": []}, "pipeline"),
        ({"chunking": {"method": "words", "condition": True}}, "chunking.condition"),
        (
            {"chunking": {"method": "words", "config": {"typo": 1}}},
            "chunking.config.typo",
        ),
        (
            {
                "chunking": {"method": "words"},
                "preprocessing": [{"operation": "detect_language", "condition": True}],
            },
            "preprocessing.0.condition",
        ),
        (
            {
                "chunking": {"method": "words"},
                "preprocessing": [{"operation": "filter_empty"}],
            },
            "preprocessing.0.operation",
        ),
        (
            {"chunking": {"method": "words", "config": {"language": "ja"}}},
            "chunking.config.language",
        ),
        (
            {"chunking": {"method": "words", "config": {"align_text_to_source": True}}},
            "chunking.config.align_text_to_source",
        ),
        (
            {"chunking": {"method": "words", "config": {"max_size": True}}},
            "chunking.config.max_size",
        ),
        (
            {"chunking": {"method": "words", "config": {"max_size": 4, "overlap": 4}}},
            "chunking.config.overlap",
        ),
        (
            {
                "chunking": {
                    "method": "tokens",
                    "config": {"tokenizer_name": "/missing"},
                }
            },
            "chunking.method",
        ),
        ({"chunking": {"method": "rolling_summarize"}}, "chunking.method"),
        (
            {
                "chunking": {"method": "words", "config": {"hierarchical": True}},
                "postprocessing": [{"operation": "merge_small"}],
            },
            "chunking.config.hierarchical",
        ),
    ],
)
def test_unsupported_executable_fields_have_actionable_paths(body, field):
    gate = preflight()
    with pytest.raises(gate.PreviewUnsupportedError) as caught:
        gate.prepare_recipe(body, runtime=gate.current_local_runtime())
    assert caught.value.field == field
    assert caught.value.reason


def test_prepared_snapshot_preserves_authoring_and_captures_defaults():
    gate = preflight()
    body = {
        "chunking": {"method": "words"},
        "metadata": {"custom": [1]},
        "classifier": {"media_types": ["document"]},
    }
    recipe = gate.prepare_recipe(body, runtime=gate.current_local_runtime())
    body["metadata"]["custom"].append(2)
    assert json.loads(recipe.authored_json)["metadata"] == {"custom": [1]}
    effective = json.loads(recipe.effective_json)
    assert effective["chunking"]["config"] == {
        "max_size": 400,
        "overlap": 50,
        "language": "en",
        "preserve_sentences": False,
        "min_chunk_size": 0,
    }
    assert "classifier" not in effective
    assert "metadata" not in effective
    assert (
        gate.prepare_recipe(json.loads(recipe.authored_json), runtime=recipe.runtime)
        == recipe
    )


def test_local_admission_and_execution_never_open_network(monkeypatch):
    def no_network(*args, **kwargs):
        pytest.fail("Local preview attempted network or asset download")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", no_network)
    gate = preflight()
    from tldw_chatbook.Chunking.template_runtime import execute_prepared

    recipe = gate.prepare_recipe(
        {"chunking": {"method": "words"}}, runtime=gate.current_local_runtime()
    )
    assert [
        chunk["text"] for chunk in execute_prepared(recipe, "local sample").chunks
    ] == ["local sample"]


def test_sentence_preservation_that_loses_words_is_refused_without_rewriting_body():
    gate = preflight()
    body = {"chunking": {"method": "words", "config": {"preserve_sentences": True}}}
    with pytest.raises(gate.PreviewUnsupportedError) as caught:
        gate.prepare_recipe(body, runtime=gate.current_local_runtime())
    assert caught.value.field == "chunking.config.preserve_sentences"
    assert body["chunking"]["config"]["preserve_sentences"] is True


def test_runtime_assets_are_revalidated_before_publication():
    from tldw_chatbook.Chunking.lab_models import RuntimeIdentity

    identity = RuntimeIdentity(
        backend="local",
        engine_version="engine",
        execution_version="execution",
        assets=(
            {
                "kind": "tokenizer",
                "name": "test",
                "version": "1",
                "content_digest": "digest",
            },
        ),
    )
    identity.assets[0]["private_path"] = "must not be serialized"
    with pytest.raises(ValueError, match="Assets"):
        identity.model_dump_json()
