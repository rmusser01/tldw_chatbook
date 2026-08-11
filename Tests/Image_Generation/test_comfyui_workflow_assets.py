"""Independent contract tests for the packaged H3 image-edit workflow."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

WORKFLOW_FILENAME = "minimax_h3_image_edit.json"
WORKFLOW_PATH = (
    Path(__file__).resolve().parents[2]
    / "tldw_chatbook"
    / "Image_Generation"
    / "workflows"
    / WORKFLOW_FILENAME
)

# Transcribed from the approved design, not derived from the packaged graph.
EXPECTED_NODE_CLASSES = {
    "114": "LoadImage",
    "121": "VAELoader",
    "124": "VAEDecode",
    "125": "KSamplerSelect",
    "126": "BasicScheduler",
    "127": "SamplerCustomAdvanced",
    "128": "BasicGuider",
    "129": "UNETLoader",
    "130": "CLIPLoader",
    "131": "RandomNoise",
    "133": "MiniMaxH3ImageToVideo",
    "139": "PrimitiveInt",
    "140": "GetImageSize",
    "141": "ImageScaleToTotalPixels",
    "144": "ImageFromBatch",
    "149": "ResizeImageMaskNode",
    "150": "GetImageSize",
    "165": "SaveImage",
}

# Transcribed from the approved design, not inferred from graph connectivity.
EXPECTED_DIRECT_LINKS = {
    "124.samples": ("127", 0),
    "124.vae": ("121", 0),
    "126.model": ("129", 0),
    "127.guider": ("128", 0),
    "127.latent_image": ("133", 1),
    "127.noise": ("131", 0),
    "127.sampler": ("125", 0),
    "127.sigmas": ("126", 0),
    "128.conditioning": ("133", 0),
    "128.model": ("129", 0),
    "133.clip": ("130", 0),
    "133.first_frame": ("114", 0),
    "133.height": ("140", 1),
    "133.length": ("139", 0),
    "133.vae": ("121", 0),
    "133.width": ("140", 0),
    "140.image": ("141", 0),
    "141.image": ("114", 0),
    "144.image": ("124", 0),
    "149.input": ("144", 0),
    "149.resize_type.height": ("150", 1),
    "149.resize_type.width": ("150", 0),
    "150.image": ("114", 0),
    "165.images": ("149", 0),
}

EXPECTED_CONTROLLED_LITERALS = {
    "114.image",
    "125.sampler_name",
    "126.steps",
    "131.noise_seed",
    "133.prompt",
    "165.filename_prefix",
}

EXPECTED_NEUTRAL_LITERALS = {
    "114.image": "h3_edit_input.png",
    "133.prompt": "Apply the requested image edit.",
    "165.filename_prefix": "h3_edit",
}

_CONTROL_INPUT_NAMES = {
    "image",
    "sampler_name",
    "steps",
    "noise_seed",
    "prompt",
    "filename_prefix",
}
_PATHISH_FIELD = re.compile(
    r"(?:^|_)(?:path|filepath|source|original|provenance|export)(?:$|_)",
    re.IGNORECASE,
)
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")


def _load_workflow() -> dict[str, Any]:
    assert WORKFLOW_PATH.is_file(), f"missing packaged workflow: {WORKFLOW_FILENAME}"
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    assert isinstance(graph, dict) and graph
    return graph


def _walk_input_leaves(
    value: Any,
    path: tuple[str, ...],
) -> Iterator[tuple[str, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_input_leaves(child, (*path, str(key)))
        return
    yield ".".join(path), value


def _input_leaves(graph: Mapping[str, Any]) -> Iterator[tuple[str, Any]]:
    for node_id, node in graph.items():
        inputs = node.get("inputs", {})
        assert isinstance(inputs, Mapping), f"node {node_id} inputs must be an object"
        yield from _walk_input_leaves(inputs, (node_id,))


def _is_direct_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _walk_json(value: Any) -> Iterator[tuple[str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            yield str(key), child
            yield from _walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield None, child
            yield from _walk_json(child)


def test_workflow_has_exact_normative_node_class_inventory() -> None:
    graph = _load_workflow()

    actual = {
        node_id: node.get("class_type")
        for node_id, node in graph.items()
        if isinstance(node, Mapping)
    }

    assert actual == EXPECTED_NODE_CLASSES
    assert set(graph) == set(EXPECTED_NODE_CLASSES)
    assert "154" not in graph
    assert "166" not in graph


def test_workflow_has_exact_normative_direct_links() -> None:
    graph = _load_workflow()

    actual = {
        destination: (value[0], value[1])
        for destination, value in _input_leaves(graph)
        if _is_direct_link(value)
    }

    assert actual == EXPECTED_DIRECT_LINKS


def test_node_165_is_the_only_output_and_uses_the_restored_edit_path() -> None:
    graph = _load_workflow()
    outputs = {
        node_id: node["class_type"]
        for node_id, node in graph.items()
        if node.get("class_type") in {"SaveImage", "PreviewImage"}
        or str(node.get("class_type", "")).startswith("Save")
    }

    assert outputs == {"165": "SaveImage"}
    assert graph["165"]["inputs"]["images"] == ["149", 0]
    assert {
        key: value
        for key, value in graph["165"]["inputs"].items()
        if _is_direct_link(value)
    } == {"images": ["149", 0]}
    assert graph["149"]["inputs"]["input"] == ["144", 0]
    assert graph["149"]["inputs"]["resize_type.width"] == ["150", 0]
    assert graph["149"]["inputs"]["resize_type.height"] == ["150", 1]
    assert graph["150"]["inputs"]["image"] == ["114", 0]


def test_workflow_has_only_the_approved_controlled_literals() -> None:
    graph = _load_workflow()
    leaves = dict(_input_leaves(graph))
    controlled = {
        path
        for path, value in leaves.items()
        if path.rsplit(".", 1)[-1] in _CONTROL_INPUT_NAMES
        and not _is_direct_link(value)
    }

    assert controlled == EXPECTED_CONTROLLED_LITERALS
    for path, expected in EXPECTED_NEUTRAL_LITERALS.items():
        assert leaves[path] == expected
    assert {
        path for path in leaves if path.rsplit(".", 1)[-1] == "prompt"
    } == {"133.prompt"}


def test_workflow_contains_no_path_like_provenance_fields_or_values() -> None:
    graph = _load_workflow()

    for key, value in _walk_json(graph):
        if key is not None:
            assert not _PATHISH_FIELD.search(key), f"provenance-like field: {key}"
        if isinstance(value, str):
            assert not value.startswith(("/", "~/", "\\\\"))
            assert not _WINDOWS_ABSOLUTE_PATH.match(value)


def test_resource_directory_contains_only_the_sanitized_workflow() -> None:
    workflow_dir = WORKFLOW_PATH.parent
    resources = (
        {path.name for path in workflow_dir.glob("*.json")}
        if workflow_dir.is_dir()
        else set()
    )

    assert resources == {WORKFLOW_FILENAME}
