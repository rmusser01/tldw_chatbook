"""Strict packaged-workflow seam for ComfyUI image editing."""

from __future__ import annotations

import json
from copy import deepcopy
from importlib.resources import files
from typing import Any

H3_IMAGE_EDIT_WORKFLOW_KEY = "minimax_h3_image_edit"
_WORKFLOW_RESOURCE_DIRECTORY = "workflows"


def _load_packaged_workflow(
    workflow_key: str = H3_IMAGE_EDIT_WORKFLOW_KEY,
) -> dict[str, Any]:
    """Load a fresh copy of the one supported packaged workflow."""
    if (
        not isinstance(workflow_key, str)
        or "/" in workflow_key
        or "\\" in workflow_key
        or workflow_key != H3_IMAGE_EDIT_WORKFLOW_KEY
    ):
        raise ValueError("Unsupported packaged workflow key")

    resource = files("tldw_chatbook.Image_Generation").joinpath(
        _WORKFLOW_RESOURCE_DIRECTORY,
        f"{workflow_key}.json",
    )
    try:
        with resource.open("r", encoding="utf-8") as stream:
            graph = json.load(stream)
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ValueError("Packaged workflow is unavailable or invalid") from None

    if not isinstance(graph, dict) or not graph:
        raise ValueError("Packaged workflow must be a nonempty JSON object")
    return deepcopy(graph)
