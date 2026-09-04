"""Strict, deliberately bounded Lab admission, separate from server parity.

Only the concrete pinned text paths below are qualified. Registry membership
alone is not a capability promise: other paths can load assets, call an LLM,
ignore options, or lose structured output. No assets are needed by this subset.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import string
import unicodedata

from ..chunking_engine_version import ENGINE_VERSION
from ..RAG_Admin.template_validation import validate_template
from .engine.chunker import Chunker
from .engine.regex_safety import check_pattern
from .lab_models import PreparedRecipe, RuntimeIdentity, canonical_json
from .template_runtime import registered_template_operations


class PreviewUnsupportedError(ValueError):
    """An authored field cannot be executed faithfully by this local backend."""

    def __init__(self, field: str, reason: str):
        self.field = field
        self.reason = reason
        super().__init__(f"{field}: {reason}")


METHOD_DEFAULTS = {
    "words": {
        "max_size": 400,
        "overlap": 50,
        "language": "en",
        "preserve_sentences": False,
        "min_chunk_size": 0,
    },
    "fixed_size": {"max_size": 400, "overlap": 50, "language": "en"},
}
PRE_DEFAULTS = {
    "normalize_whitespace": {"max_line_breaks": 2},
    "remove_headers": {"patterns": []},
    "extract_sections": {"pattern": r"^#+\s+(.+)$"},
    "clean_markdown": {
        "remove_links": False,
        "remove_images": False,
        "remove_formatting": False,
    },
    "detect_language": {},
}
POST_DEFAULTS = {
    "filter_empty": {"min_length": 10},
    "merge_small": {"min_size": 100, "separator": "\n\n"},
    "add_overlap": {"size": 50, "marker": ""},
    # Upstream's name is misleading: this inserts text, not record metadata.
    "add_metadata": {"prefix": "", "suffix": ""},
    "format_chunks": {"template": "{chunk}"},
}


def current_local_runtime() -> RuntimeIdentity:
    """Identify the pinned engine and stdlib-only qualified execution paths."""
    return RuntimeIdentity(
        backend="local",
        engine_version=ENGINE_VERSION,
        execution_version=f"lab-1/{platform.python_implementation()}-{platform.python_version()}/unicode-{unicodedata.unidata_version}",
        assets=(),
    )


def _keys(value: dict, allowed: dict | set, path: str) -> None:
    for key in value:
        if key not in allowed:
            raise PreviewUnsupportedError(
                f"{path}.{key}" if path else key,
                "Unsupported executable field; remove it to preview locally",
            )


def _config(value: object, defaults: dict, path: str) -> dict:
    if not isinstance(value, dict):
        raise PreviewUnsupportedError(path, "Expected an object")
    _keys(value, defaults, path)
    effective = {**defaults, **value}
    for key, item in effective.items():
        default = defaults[key]
        field = f"{path}.{key}"
        if type(item) is not type(default):
            raise PreviewUnsupportedError(field, f"Expected {type(default).__name__}")
        if isinstance(item, int) and not isinstance(item, bool) and item < 0:
            raise PreviewUnsupportedError(field, "Must be nonnegative")
    return effective


def _pattern(pattern: object, field: str) -> None:
    if not isinstance(pattern, str):
        raise PreviewUnsupportedError(field, "Expected a regex string")
    error = check_pattern(pattern, max_len=256)
    if error:
        raise PreviewUnsupportedError(field, str(error))
    try:
        re.compile(pattern)
    except re.error as exc:
        raise PreviewUnsupportedError(field, str(exc)) from exc


def _format(value: str, field: str, allowed: set[str]) -> None:
    try:
        for _, name, spec, conversion in string.Formatter().parse(value):
            if name is not None and (name not in allowed or spec or conversion):
                raise ValueError(
                    "Only plain placeholders "
                    + ", ".join(sorted(allowed))
                    + " are supported"
                )
    except ValueError as exc:
        raise PreviewUnsupportedError(field, str(exc)) from exc


def prepare_recipe(body: dict, *, runtime: RuntimeIdentity) -> PreparedRecipe:
    """Validate, freeze defaults, and hash one faithful local recipe.

    Classifier rules remain authored selection metadata; direct preview never
    evaluates them. Unknown metadata survives in authored_json. Call off the UI
    loop: validation, serialization, and hashing may exceed 100 ms.
    """
    verdict = validate_template(body)
    if not verdict["valid"]:
        issue = verdict["errors"][0]
        raise PreviewUnsupportedError(issue["field"], issue["message"])
    try:
        authored_json = canonical_json(body)
        body = json.loads(authored_json)
    except (ValueError, TypeError) as exc:
        raise PreviewUnsupportedError(
            "template", "Only finite JSON values are supported"
        ) from exc
    _keys(
        body,
        {"preprocessing", "chunking", "postprocessing", "classifier", "metadata"},
        "",
    )
    if runtime != current_local_runtime():
        raise PreviewUnsupportedError(
            "runtime",
            "Runtime or assets changed; prepare again using the current local runtime",
        )
    chunking = body["chunking"]
    _keys(chunking, {"method", "config"}, "chunking")
    method = chunking["method"]
    if (
        not isinstance(method, str)
        or method not in METHOD_DEFAULTS
        or method not in Chunker().get_available_methods()
    ):
        raise PreviewUnsupportedError(
            "chunking.method",
            "Method unavailable for faithful offline preview; use words (English) or fixed_size. Asset-dependent, hierarchical, and LLM paths are not qualified",
        )
    config = _config(
        chunking.get("config", {}), METHOD_DEFAULTS[method], "chunking.config"
    )
    if config["language"] != "en":
        raise PreviewUnsupportedError(
            "chunking.config.language",
            "Only en is qualified; other tokenizers/assets are unavailable in Lab",
        )
    if config.get("preserve_sentences"):
        raise PreviewUnsupportedError(
            "chunking.config.preserve_sentences",
            "The pinned word processor can omit words with this option enabled; use false for local preview",
        )
    if config["max_size"] <= 0:
        raise PreviewUnsupportedError("chunking.config.max_size", "Must be positive")
    # Capture the engine's documented clamp only for an omitted default.
    if "overlap" not in chunking.get("config", {}):
        config["overlap"] = min(config["overlap"], config["max_size"] - 1)
    if config["overlap"] >= config["max_size"]:
        raise PreviewUnsupportedError(
            "chunking.config.overlap", "Must be smaller than max_size"
        )
    effective = {"chunking": {"method": method, "config": config}}
    registry = registered_template_operations()
    for stage, defaults in (
        ("preprocessing", PRE_DEFAULTS),
        ("postprocessing", POST_DEFAULTS),
    ):
        operations = body.get(stage, [])
        if not isinstance(operations, list):
            raise PreviewUnsupportedError(stage, "Expected an operation array")
        effective[stage] = []
        may_emit_empty = False
        for index, operation in enumerate(operations):
            path = f"{stage}.{index}"
            _keys(operation, {"operation", "config"}, path)
            name = operation.get("operation")
            if (
                not isinstance(name, str)
                or name not in defaults
                or name not in registry
            ):
                raise PreviewUnsupportedError(
                    f"{path}.operation",
                    "Operation unavailable in this stage; use a supported local operation",
                )
            options = _config(
                operation.get("config", {}), defaults[name], f"{path}.config"
            )
            if name == "remove_headers":
                for i, pattern in enumerate(options["patterns"]):
                    _pattern(pattern, f"{path}.config.patterns.{i}")
            if name == "extract_sections":
                _pattern(options["pattern"], f"{path}.config.pattern")
            if name == "normalize_whitespace" and options["max_line_breaks"] < 1:
                raise PreviewUnsupportedError(
                    f"{path}.config.max_line_breaks", "Must be at least one"
                )
            if name == "merge_small" and not options["separator"]:
                raise PreviewUnsupportedError(
                    f"{path}.config.separator",
                    "Use a nonempty separator for contributor attribution",
                )
            if name == "merge_small" and may_emit_empty:
                raise PreviewUnsupportedError(
                    path,
                    "Merging after a formatter that can erase chunks is not qualified; remove the formatter or merge first",
                )
            if name == "format_chunks" and options["template"] == "":
                may_emit_empty = True
            for key in ("prefix", "suffix", "template"):
                if key in options:
                    _format(
                        options[key],
                        f"{path}.config.{key}",
                        {"index", "total", "chunk"}
                        if key == "template"
                        else {"index", "total"},
                    )
            effective[stage].append({"operation": name, "config": options})
    effective_json = canonical_json(effective)
    identity = canonical_json(
        {
            "authored": body,
            "effective": effective,
            "runtime": runtime.model_dump(mode="json"),
        }
    )
    return PreparedRecipe(
        authored_json=authored_json,
        effective_json=effective_json,
        runtime=runtime,
        recipe_hash=hashlib.sha256(identity.encode("utf-8")).hexdigest(),
    )
