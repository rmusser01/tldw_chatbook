"""Independent contract tests for the packaged H3 image-edit workflow."""

from __future__ import annotations

import copy
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

# Transcribed independently from the approved graph contract. Literal values are
# deliberately excluded so schema failures cannot reveal operational data.
EXPECTED_INPUT_KEYS = {
    "114": frozenset({"image"}),
    "121": frozenset({"vae_name"}),
    "124": frozenset({"samples", "vae"}),
    "125": frozenset({"sampler_name"}),
    "126": frozenset({"denoise", "model", "scheduler", "steps"}),
    "127": frozenset({"guider", "latent_image", "noise", "sampler", "sigmas"}),
    "128": frozenset({"conditioning", "model"}),
    "129": frozenset({"unet_name", "weight_dtype"}),
    "130": frozenset({"clip_name", "device", "type"}),
    "131": frozenset({"noise_seed"}),
    "133": frozenset({"clip", "first_frame", "height", "length", "prompt", "vae", "width"}),
    "139": frozenset({"value"}),
    "140": frozenset({"image"}),
    "141": frozenset({"image", "megapixels", "resolution_steps", "upscale_method"}),
    "144": frozenset({"batch_index", "image", "length"}),
    "149": frozenset(
        {
            "input",
            "resize_type",
            "resize_type.crop",
            "resize_type.height",
            "resize_type.width",
            "scale_method",
        }
    ),
    "150": frozenset({"image"}),
    "165": frozenset({"filename_prefix", "images"}),
}
EXPECTED_NODE_KEYS = frozenset({"_meta", "class_type", "inputs"})
EXPECTED_METADATA_KEYS = frozenset({"title"})
EXPECTED_NODE_TITLES = {
    "114": "Load Image",
    "121": "Load VAE",
    "124": "VAE Decode",
    "125": "KSamplerSelect",
    "126": "BasicScheduler",
    "127": "SamplerCustomAdvanced",
    "128": "Basic Guider",
    "129": "Load Diffusion Model",
    "130": "Load CLIP",
    "131": "RandomNoise",
    "133": "MiniMax H3 Image to Video",
    "139": "Frame Length",
    "140": "Get Image Size",
    "141": "Scale Image to Total Pixels",
    "144": "Get Image from Batch",
    "149": "Resize Image/Mask",
    "150": "Get Image Size",
    "165": "Save Output Image",
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
_PROVENANCE_KEY_TOKENS = frozenset(
    {
        "source",
        "original",
        "provenance",
        "export",
        "path",
        "filepath",
        "hash",
        "digest",
        "checksum",
    }
)
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_TOKEN = re.compile(r"[a-z0-9]+")
_URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
_WINDOWS_ABSOLUTE_PATH = re.compile(r"^[A-Za-z]:[\\/]")
_OPERATIONAL_FILE_SELECTOR_INPUTS = frozenset(
    {
        "121.inputs.vae_name",
        "129.inputs.unet_name",
        "130.inputs.clip_name",
    }
)
_APPROVED_LITERAL_PATHS = {
    "114.inputs.image": EXPECTED_NEUTRAL_LITERALS["114.image"],
    "165.inputs.filename_prefix": EXPECTED_NEUTRAL_LITERALS["165.filename_prefix"],
}
_APPROVED_TITLE_PATHS = {
    f"{node_id}._meta.title": title
    for node_id, title in EXPECTED_NODE_TITLES.items()
}

LOAD_ERROR = "Packaged workflow could not be loaded as a nonempty JSON object"
STRUCTURE_ERROR = "Packaged workflow structure does not match the approved contract"
LINK_ERROR = "Packaged workflow direct links do not match the approved contract"
OUTPUT_ERROR = "Packaged workflow output path does not match the approved contract"
CONTROL_ERROR = "Packaged workflow controlled literals do not match the approved contract"
PRIVACY_ERROR = "Packaged workflow contains prohibited provenance data"
RESOURCE_ERROR = "Image workflow resource inventory does not match the approved contract"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _load_workflow() -> dict[str, Any]:
    _require(WORKFLOW_PATH.is_file(), LOAD_ERROR)
    graph = json.loads(WORKFLOW_PATH.read_text(encoding="utf-8"))
    _require(isinstance(graph, dict) and bool(graph), LOAD_ERROR)
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
        _require(isinstance(inputs, Mapping), STRUCTURE_ERROR)
        yield from _walk_input_leaves(inputs, (node_id,))


def _is_direct_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


def _walk_json(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str | None, Any]]:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            child_path = (*path, key_text)
            yield child_path, key_text, child
            yield from _walk_json(child, child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_path = (*path, str(index))
            yield child_path, None, child
            yield from _walk_json(child, child_path)


def _key_has_provenance_token(key: str) -> bool:
    normalized = _CAMEL_BOUNDARY.sub("_", key)
    tokens = set(_KEY_TOKEN.findall(normalized.casefold()))
    return bool(tokens & _PROVENANCE_KEY_TOKENS)


def _is_absolute_or_uri(value: str) -> bool:
    return (
        bool(_URI_SCHEME.match(value))
        or value.startswith(("/", "~/", "\\\\", "//"))
        or bool(_WINDOWS_ABSOLUTE_PATH.match(value))
    )


def _is_relative_path_like(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    return bool(Path(value).suffix)


def _is_separator_free_basename(value: str) -> bool:
    return "/" not in value and "\\" not in value and Path(value).name == value


def _validate_workflow_structure(graph: Mapping[str, Any]) -> None:
    _require(set(graph) == set(EXPECTED_NODE_CLASSES), STRUCTURE_ERROR)
    for node_id, expected_class in EXPECTED_NODE_CLASSES.items():
        node = graph.get(node_id)
        _require(isinstance(node, Mapping), STRUCTURE_ERROR)
        _require(set(node) == EXPECTED_NODE_KEYS, STRUCTURE_ERROR)
        _require(node.get("class_type") == expected_class, STRUCTURE_ERROR)
        inputs = node.get("inputs")
        metadata = node.get("_meta")
        _require(isinstance(inputs, Mapping), STRUCTURE_ERROR)
        _require(set(inputs) == EXPECTED_INPUT_KEYS[node_id], STRUCTURE_ERROR)
        _require(isinstance(metadata, Mapping), STRUCTURE_ERROR)
        _require(set(metadata) == EXPECTED_METADATA_KEYS, STRUCTURE_ERROR)
        _require(metadata.get("title") == EXPECTED_NODE_TITLES[node_id], STRUCTURE_ERROR)


def _validate_workflow_privacy(graph: Mapping[str, Any]) -> None:
    for path, key, value in _walk_json(graph):
        if key is not None:
            _require(not _key_has_provenance_token(key), PRIVACY_ERROR)
        if not isinstance(value, str):
            continue
        _require(not _is_absolute_or_uri(value), PRIVACY_ERROR)
        if not _is_relative_path_like(value):
            continue
        dotted_path = ".".join(path)
        approved_value = _APPROVED_LITERAL_PATHS.get(dotted_path)
        approved_title = _APPROVED_TITLE_PATHS.get(dotted_path)
        operational_basename = (
            dotted_path in _OPERATIONAL_FILE_SELECTOR_INPUTS
            and _is_separator_free_basename(value)
        )
        allowed = (
            operational_basename
            or approved_value == value
            or approved_title == value
        )
        _require(allowed, PRIVACY_ERROR)


def _validate_resource_inventory(resources: set[str]) -> None:
    _require(resources == {WORKFLOW_FILENAME}, RESOURCE_ERROR)


def test_workflow_has_exact_normative_node_class_inventory() -> None:
    graph = _load_workflow()

    _validate_workflow_structure(graph)
    _require("154" not in graph and "166" not in graph, STRUCTURE_ERROR)


def test_workflow_has_exact_normative_direct_links() -> None:
    graph = _load_workflow()
    _validate_workflow_structure(graph)

    actual = {
        destination: (value[0], value[1])
        for destination, value in _input_leaves(graph)
        if _is_direct_link(value)
    }

    _require(actual == EXPECTED_DIRECT_LINKS, LINK_ERROR)


def test_node_165_is_the_only_output_and_uses_the_restored_edit_path() -> None:
    graph = _load_workflow()
    _validate_workflow_structure(graph)
    outputs = {
        node_id: node["class_type"]
        for node_id, node in graph.items()
        if node.get("class_type") in {"SaveImage", "PreviewImage"}
        or str(node.get("class_type", "")).startswith("Save")
    }

    linked_output_inputs = {
        key: value
        for key, value in graph["165"]["inputs"].items()
        if _is_direct_link(value)
    }
    restored_path_matches = (
        outputs == {"165": "SaveImage"}
        and graph["165"]["inputs"]["images"] == ["149", 0]
        and linked_output_inputs == {"images": ["149", 0]}
        and graph["149"]["inputs"]["input"] == ["144", 0]
        and graph["149"]["inputs"]["resize_type.width"] == ["150", 0]
        and graph["149"]["inputs"]["resize_type.height"] == ["150", 1]
        and graph["150"]["inputs"]["image"] == ["114", 0]
    )
    _require(restored_path_matches, OUTPUT_ERROR)


def test_workflow_has_only_the_approved_controlled_literals() -> None:
    graph = _load_workflow()
    _validate_workflow_structure(graph)
    leaves = dict(_input_leaves(graph))
    controlled = {
        path
        for path, value in leaves.items()
        if path.rsplit(".", 1)[-1] in _CONTROL_INPUT_NAMES
        and not _is_direct_link(value)
    }

    _require(controlled == EXPECTED_CONTROLLED_LITERALS, CONTROL_ERROR)
    for path, expected in EXPECTED_NEUTRAL_LITERALS.items():
        _require(leaves.get(path) == expected, CONTROL_ERROR)
    prompt_paths = {
        path for path in leaves if path.rsplit(".", 1)[-1] == "prompt"
    }
    _require(prompt_paths == {"133.prompt"}, CONTROL_ERROR)


def test_workflow_contains_no_path_like_provenance_fields_or_values() -> None:
    graph = _load_workflow()

    _validate_workflow_structure(graph)
    _validate_workflow_privacy(graph)


def test_resource_directory_contains_only_the_sanitized_workflow() -> None:
    workflow_dir = WORKFLOW_PATH.parent
    resources = (
        {path.name for path in workflow_dir.glob("*.json")}
        if workflow_dir.is_dir()
        else set()
    )

    _validate_resource_inventory(resources)


def _assert_constant_refusal(
    operation: Any,
    expected_message: str,
) -> None:
    try:
        operation()
    except AssertionError as exc:
        if str(exc) != expected_message:
            raise AssertionError("Workflow guard refusal was not sanitized") from None
    else:
        raise AssertionError("Workflow guard accepted prohibited test data")


def _workflow_with_unexpected_input() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["inputs"]["harmlessSentinelInput"] = "harmless-value"
    return graph


def _workflow_with_relative_path() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["_meta"]["title"] = "relative/harmless-sentinel.txt"
    return graph


def _workflow_with_spaced_relative_path() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["_meta"]["title"] = "relative folder/harmless sentinel.txt"
    return graph


def _workflow_with_unapproved_title() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["_meta"]["title"] = "Harmless Display Title"
    return graph


def _workflow_with_operational_selector_path() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["121"]["inputs"]["vae_name"] = (
        "relative folder/harmless selector.safetensors"
    )
    return graph


def _workflow_with_uri() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["_meta"]["title"] = "harmless-scheme://example.invalid/item"
    return graph


def _workflow_with_camel_hash_metadata() -> dict[str, Any]:
    graph = copy.deepcopy(_load_workflow())
    graph["114"]["_meta"]["sourceChecksum"] = "harmless-marker"
    return graph


def _resource_inventory_with_unexpected_filename() -> set[str]:
    return {WORKFLOW_FILENAME, "unexpected-harmless-sentinel.json"}


def test_structure_validator_rejects_unexpected_input_without_echo() -> None:
    graph = _workflow_with_unexpected_input()

    _assert_constant_refusal(
        lambda: _validate_workflow_structure(graph),
        STRUCTURE_ERROR,
    )


def test_privacy_validator_rejects_relative_path_without_echo() -> None:
    graph = _workflow_with_relative_path()

    _assert_constant_refusal(
        lambda: _validate_workflow_privacy(graph),
        PRIVACY_ERROR,
    )


def test_privacy_validator_rejects_spaced_relative_path_without_echo() -> None:
    graph = _workflow_with_spaced_relative_path()

    _assert_constant_refusal(
        lambda: _validate_workflow_privacy(graph),
        PRIVACY_ERROR,
    )


def test_structure_validator_rejects_unapproved_title_without_echo() -> None:
    graph = _workflow_with_unapproved_title()

    _assert_constant_refusal(
        lambda: _validate_workflow_structure(graph),
        STRUCTURE_ERROR,
    )


def test_privacy_validator_rejects_operational_selector_path_without_echo() -> None:
    graph = _workflow_with_operational_selector_path()

    _assert_constant_refusal(
        lambda: _validate_workflow_privacy(graph),
        PRIVACY_ERROR,
    )


def test_privacy_validator_rejects_uri_without_echo() -> None:
    graph = _workflow_with_uri()

    _assert_constant_refusal(
        lambda: _validate_workflow_privacy(graph),
        PRIVACY_ERROR,
    )


def test_privacy_validator_rejects_camel_hash_metadata_without_echo() -> None:
    graph = _workflow_with_camel_hash_metadata()

    _assert_constant_refusal(
        lambda: _validate_workflow_privacy(graph),
        PRIVACY_ERROR,
    )


def test_resource_inventory_refusal_does_not_echo_unexpected_filename() -> None:
    resources = _resource_inventory_with_unexpected_filename()

    _assert_constant_refusal(
        lambda: _validate_resource_inventory(resources),
        RESOURCE_ERROR,
    )
