"""Immutable Canvas V1 render-plan and bridge wire contracts."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeAlias

from .limits import (
    CanvasLimitError,
    CanvasLimits,
    JsonValue,
    decode_data_url,
    raster_signature_matches,
    sha256_utf8,
    validate_asset_payloads,
    validate_count,
    validate_json_value,
    validate_opaque_identifier,
    validate_unique_identifiers,
    validate_utf8_text,
    validate_utf8_text_parts,
    verify_sha256_utf8,
)

RuntimeProfile: TypeAlias = Literal["canvas-v1"]
BridgeRequestKind: TypeAlias = Literal["submit", "download"]

_PASSIVE_DOWNLOAD_TYPES: Mapping[str, tuple[str, ...]] = MappingProxyType(
    {
        "text/plain": (".txt",),
        "text/csv": (".csv",),
        "application/json": (".json",),
        "image/png": (".png",),
        "image/jpeg": (".jpg", ".jpeg"),
        "image/gif": (".gif",),
        "image/webp": (".webp",),
    }
)
_WINDOWS_RESERVED_BASENAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)
_UNSAFE_FILENAME_CHARACTER = re.compile(r'[\x00-\x1f\x7f<>:"|?*]')


@dataclass(frozen=True, slots=True)
class CanvasScope:
    """Server-owned durable Canvas authority captured for one Console run."""

    session_id: str
    conversation_id: str
    active_message_ids: tuple[str, ...]
    selected_canvas_id: str | None
    selected_revision_id: str | None
    run_id: str


@dataclass(frozen=True, slots=True)
class CanvasOrigin:
    """Compact source-free origin metadata for one reachable revision."""

    message_id: str
    run_id: str


@dataclass(frozen=True, slots=True)
class CanvasListItem:
    """One source-free Canvas projection resolved against an active branch."""

    canvas_id: str
    revision_id: str
    parent_revision_id: str | None
    title: str
    runtime_profile: RuntimeProfile
    content_sha256: str
    source_bytes: int
    sequence: int
    origin: CanvasOrigin
    is_selected: bool
    is_historical_selection: bool


@dataclass(frozen=True, slots=True)
class CanvasRevisionInfo:
    """Source-free identity and revision metadata for one exact Canvas state."""

    canvas_id: str
    revision_id: str
    parent_revision_id: str | None
    title: str
    runtime_profile: RuntimeProfile
    content_sha256: str
    source_bytes: int
    sequence: int
    origin: CanvasOrigin


@dataclass(frozen=True, slots=True)
class CanvasReadResult:
    """One exact reachable revision including its complete inert source."""

    revision: CanvasRevisionInfo
    source: str


@dataclass(frozen=True, slots=True)
class CanvasCreateResult:
    """A created root revision with source and bounded compiler diagnostics."""

    revision: CanvasRevisionInfo
    source: str
    compatibility_issues: tuple[CanvasCompatibilityIssue, ...]


@dataclass(frozen=True, slots=True)
class CanvasMutationResult:
    """Source-free metadata for one appended update or rename revision."""

    revision: CanvasRevisionInfo
    compatibility_issues: tuple[CanvasCompatibilityIssue, ...] = ()


@dataclass(frozen=True, slots=True)
class CanvasConflictResult:
    """The bounded selected/resolved base returned for an optimistic conflict."""

    code: str
    canvas_id: str
    current_revision_id: str
    content_sha256: str
    title: str
    sequence: int
    origin: CanvasOrigin


@dataclass(frozen=True, slots=True)
class CanvasCompatibilityIssue:
    """A bounded compiler/runtime incompatibility without source disclosure."""

    code: str
    message: str
    location: str | None = None

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.code, field_name="compatibility issue code")
        validate_utf8_text(self.message, limit=4 * 1024, field_name="compatibility issue message")
        if self.location is not None:
            validate_utf8_text(self.location, limit=512, field_name="compatibility issue location")


@dataclass(frozen=True, slots=True)
class RenderAsset:
    """An opaque compiler-extracted asset; never a browser-resolvable URL."""

    asset_id: str
    mime_type: str
    data: bytes

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.asset_id, field_name="asset ID")
        if not isinstance(self.mime_type, str) or "/" not in self.mime_type:
            raise CanvasLimitError("asset MIME type must be a MIME string")
        validate_utf8_text(self.mime_type, limit=256, field_name="asset MIME type")
        if not isinstance(self.data, bytes):
            raise CanvasLimitError("asset data must be bytes")


@dataclass(frozen=True, slots=True)
class RenderNode:
    """A compiler-normalized DOM node, represented without browser markup sinks."""

    node_id: str
    tag: str
    attributes: tuple[tuple[str, str], ...] = ()
    text: str | None = None
    children: tuple[RenderNode, ...] = ()

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.node_id, field_name="node ID")
        if not isinstance(self.tag, str) or not self.tag:
            raise CanvasLimitError("node tag must be a non-empty string")
        validate_utf8_text(self.tag, limit=128, field_name="node tag")
        if not isinstance(self.attributes, tuple):
            raise CanvasLimitError("node attributes must be an immutable tuple")
        attribute_names: list[str] = []
        for item in self.attributes:
            if not isinstance(item, tuple) or len(item) != 2:
                raise CanvasLimitError("node attribute must be a string pair")
            name, value = item
            if not isinstance(name, str) or not isinstance(value, str):
                raise CanvasLimitError("node attribute must be a string pair")
            validate_utf8_text(name, limit=256, field_name="node attribute name")
            validate_utf8_text(value, limit=16 * 1024, field_name="node attribute value")
            attribute_names.append(name)
        validate_unique_identifiers(tuple(attribute_names), field_name="node attribute names")
        if self.text is not None:
            validate_utf8_text(self.text, limit=CanvasLimits().html_bytes, field_name="node text")
        if not isinstance(self.children, tuple) or not all(
            isinstance(child, RenderNode) for child in self.children
        ):
            raise CanvasLimitError("node children must be an immutable tuple of render nodes")


@dataclass(frozen=True, slots=True, init=False)
class CanvasSourceIdentity:
    """A factory-created, lossless identity for source retained outside the plan."""

    source_bytes: int
    sha256: str

    @classmethod
    def from_source(cls, source: str) -> CanvasSourceIdentity:
        """Create the only valid identity for exact UTF-8 Canvas source text."""
        identity = object.__new__(cls)
        object.__setattr__(
            identity,
            "source_bytes",
            validate_utf8_text(source, limit=CanvasLimits().html_bytes, field_name="HTML source"),
        )
        object.__setattr__(identity, "sha256", sha256_utf8(source))
        return identity

    def verify_source(self, source: str) -> None:
        """Fail closed unless *source* exactly recreates this byte/digest identity."""
        source_bytes = validate_utf8_text(
            source, limit=CanvasLimits().html_bytes, field_name="HTML source"
        )
        if source_bytes != self.source_bytes or not verify_sha256_utf8(source, self.sha256):
            raise CanvasLimitError("source identity does not match source")


@dataclass(frozen=True, slots=True)
class CanvasRenderPlan:
    """A closed, derived render plan for exactly one supported runtime profile."""

    runtime_profile: RuntimeProfile
    source_identity: CanvasSourceIdentity
    root: RenderNode
    assets: tuple[RenderAsset, ...] = ()
    css_rules: tuple[str, ...] = ()
    scripts: tuple[str, ...] = ()
    compatibility_issues: tuple[CanvasCompatibilityIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.runtime_profile != "canvas-v1":
            raise CanvasLimitError("unsupported Canvas runtime profile")
        if not isinstance(self.source_identity, CanvasSourceIdentity):
            raise CanvasLimitError("render plan source identity must be a Canvas source identity")
        if not isinstance(self.root, RenderNode):
            raise CanvasLimitError("render plan root must be a render node")
        _require_tuple_of(self.assets, RenderAsset, "render plan assets")
        _require_tuple_of(self.compatibility_issues, CanvasCompatibilityIssue, "compatibility issues")
        if not isinstance(self.css_rules, tuple) or not all(
            isinstance(rule, str) for rule in self.css_rules
        ):
            raise CanvasLimitError("CSS rules must be an immutable tuple of strings")
        if not isinstance(self.scripts, tuple) or not all(isinstance(script, str) for script in self.scripts):
            raise CanvasLimitError("scripts must be an immutable tuple of strings")

        limits = CanvasLimits()
        validate_unique_identifiers(tuple(asset.asset_id for asset in self.assets), field_name="asset IDs")
        validate_asset_payloads(
            tuple(_asset_payload(asset) for asset in self.assets),
            per_asset_limit=limits.asset_bytes,
            aggregate_limit=limits.aggregate_asset_bytes,
        )
        validate_count(len(self.css_rules), limit=limits.css_rules, field_name="CSS rules")
        validate_utf8_text_parts(self.scripts, limit=limits.script_bytes, field_name="script")
        nodes = _all_nodes(self.root)
        validate_count(len(nodes), limit=limits.dom_nodes, field_name="DOM nodes")
        validate_unique_identifiers(
            tuple(node.node_id for node in nodes), field_name="node IDs"
        )
        validate_utf8_text_parts(
            _render_plan_text_values(
                nodes=nodes,
                assets=self.assets,
                css_rules=self.css_rules,
                scripts=self.scripts,
                compatibility_issues=self.compatibility_issues,
            ),
            limit=limits.html_bytes,
            field_name="render plan text",
        )


@dataclass(frozen=True, slots=True, repr=False)
class CanvasBridgeRequest:
    """One untrusted browser-to-shell bridge request with a closed V1 schema."""

    version: RuntimeProfile
    request_id: str
    kind: BridgeRequestKind
    value: JsonValue

    def __post_init__(self) -> None:
        _validate_bridge_request(
            version=self.version,
            request_id=self.request_id,
            kind=self.kind,
            value=self.value,
            limits=CanvasLimits(),
        )
        object.__setattr__(self, "value", _freeze_json_value(self.value))

    @classmethod
    def from_wire(cls, message: object, *, limits: CanvasLimits | None = None) -> CanvasBridgeRequest:
        """Decode exactly the Canvas V1 request fields and reject all extras."""
        if not isinstance(message, Mapping):
            raise ValueError("Canvas bridge request must be an object")
        expected_fields = {"version", "request_id", "kind", "value"}
        actual_fields = set(message.keys())
        if actual_fields != expected_fields:
            unknown = actual_fields - expected_fields
            missing = expected_fields - actual_fields
            if unknown:
                raise ValueError("Canvas bridge request contains unknown fields")
            raise ValueError(f"Canvas bridge request is missing fields: {', '.join(sorted(missing))}")

        try:
            _validate_bridge_request(
                version=message["version"],
                request_id=message["request_id"],
                kind=message["kind"],
                value=message["value"],
                limits=limits or CanvasLimits(),
            )
        except CanvasLimitError as exc:
            raise ValueError(str(exc)) from exc
        return cls(
            version=message["version"],
            request_id=message["request_id"],
            kind=message["kind"],
            value=message["value"],
        )

    def submit_text(self) -> str:
        """Return exact text or deterministic compact JSON for a submit request."""

        if self.kind != "submit":
            raise ValueError("Canvas bridge request is not a submit request")
        if isinstance(self.value, str):
            return self.value
        return json.dumps(
            _thaw_json_value(self.value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def download_payload(self) -> CanvasDownloadPayload:
        """Decode one validated passive generated-download request."""

        if self.kind != "download":
            raise ValueError("Canvas bridge request is not a download request")
        return _decode_download_payload(self.value, limits=CanvasLimits())

    def __repr__(self) -> str:
        return (
            "CanvasBridgeRequest("
            f"request_id={self.request_id!r}, kind={self.kind!r}, payload=<redacted>)"
        )


@dataclass(frozen=True, slots=True, repr=False)
class CanvasDownloadPayload:
    """Sanitized passive browser download with decoded bytes kept out of repr."""

    filename: str
    mime_type: str
    data: bytes = field(repr=False)
    text_preview: str | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"CanvasDownloadPayload(mime_type={self.mime_type!r}, payload=<redacted>)"


@dataclass(frozen=True, slots=True)
class CanvasRuntimeFailure:
    """A bounded, content-free runtime failure sent to trusted UI only."""

    code: str
    message: str
    retryable: bool = False

    def __post_init__(self) -> None:
        validate_opaque_identifier(self.code, field_name="runtime failure code")
        validate_utf8_text(self.message, limit=4 * 1024, field_name="runtime failure message")
        if not isinstance(self.retryable, bool):
            raise CanvasLimitError("runtime failure retryable must be a boolean")


def _asset_payload(asset: RenderAsset):
    from .limits import DecodedDataUrl

    return DecodedDataUrl(mime_type=asset.mime_type, data=asset.data)


def _validate_bridge_request(
    *, version: object, request_id: object, kind: object, value: object, limits: CanvasLimits
) -> None:
    if version != "canvas-v1":
        raise CanvasLimitError("unsupported Canvas bridge request version")
    validate_opaque_identifier(request_id, field_name="bridge request ID")
    if kind not in ("submit", "download"):
        raise CanvasLimitError("unsupported Canvas bridge request kind")
    validate_json_value(value, max_depth=limits.json_depth, field_name="bridge request value")
    if kind == "submit":
        if isinstance(value, str):
            encoded_value = value
        else:
            try:
                encoded_value = json.dumps(
                    value,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise CanvasLimitError("bridge request value must be JSON-compatible") from exc
        validate_utf8_text(
            encoded_value,
            limit=limits.submit_payload_bytes,
            field_name="submit payload",
        )
        return
    _decode_download_payload(value, limits=limits)


def _decode_download_payload(value: object, *, limits: CanvasLimits) -> CanvasDownloadPayload:
    if not isinstance(value, Mapping):
        raise CanvasLimitError("download request must be an object")
    expected = {"filename", "mime_type", "data"}
    actual = set(value)
    if actual != expected:
        if actual - expected:
            raise CanvasLimitError("download request contains unknown fields")
        raise CanvasLimitError("download request is missing required fields")
    raw_filename = value["filename"]
    mime_type = value["mime_type"]
    raw_data = value["data"]
    if not all(isinstance(item, str) for item in (raw_filename, mime_type, raw_data)):
        raise CanvasLimitError("download filename, MIME type, and data must be text")
    assert isinstance(raw_filename, str)
    assert isinstance(mime_type, str)
    assert isinstance(raw_data, str)
    if _UNSAFE_FILENAME_CHARACTER.search(raw_filename):
        raise CanvasLimitError("download filename contains unsafe characters")
    filename = raw_filename.strip()
    validate_utf8_text(filename, limit=255, field_name="download filename")
    if not filename:
        raise CanvasLimitError("download filename must not be empty")
    if "/" in filename or "\\" in filename:
        raise CanvasLimitError("download filename must not contain path separators")
    if _UNSAFE_FILENAME_CHARACTER.search(filename) or filename in {".", ".."}:
        raise CanvasLimitError("download filename contains unsafe characters")
    if filename.startswith(".") or filename.endswith((".", " ")):
        raise CanvasLimitError("download filename is reserved or hidden")
    stem = filename.split(".", 1)[0].casefold()
    if stem in _WINDOWS_RESERVED_BASENAMES:
        raise CanvasLimitError("download filename is reserved")
    extensions = _PASSIVE_DOWNLOAD_TYPES.get(mime_type)
    if extensions is None:
        raise CanvasLimitError("download MIME type is not an allowed passive V1 MIME type")
    if not filename.casefold().endswith(extensions):
        raise CanvasLimitError("download filename extension does not match MIME type")

    if mime_type.startswith("image/"):
        decoded = decode_data_url(raw_data, field_name="download image")
        if decoded.mime_type != mime_type:
            raise CanvasLimitError("download image MIME type does not match request MIME type")
        data = decoded.data
        if not raster_signature_matches(mime_type, data):
            raise CanvasLimitError("download image bytes do not match declared signature")
        text_preview = None
    else:
        if raw_data.startswith("data:"):
            raise CanvasLimitError("text downloads use literal UTF-8 data, not data URLs")
        try:
            data = raw_data.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise CanvasLimitError("download data must contain valid Unicode") from exc
        text_preview = raw_data
        if mime_type == "application/json":
            try:
                json.loads(raw_data)
            except json.JSONDecodeError as exc:
                raise CanvasLimitError("JSON download data must be valid JSON") from exc
    if len(data) > limits.download_payload_bytes:
        raise CanvasLimitError(
            f"download payload exceeds {limits.download_payload_bytes} decoded bytes"
        )
    try:
        encoded_value = json.dumps(
            _thaw_json_value(value),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanvasLimitError("download request must be JSON-compatible") from exc
    encoded_limit = ((limits.download_payload_bytes + 2) // 3) * 4 + 4 * 1024
    validate_utf8_text(
        encoded_value,
        limit=encoded_limit,
        field_name="download encoded payload",
    )
    return CanvasDownloadPayload(filename, mime_type, data, text_preview)


def _freeze_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_json_value(child) for key, child in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json_value(child) for child in value)  # type: ignore[return-value]
    return value


def _thaw_json_value(value: JsonValue) -> JsonValue:
    if isinstance(value, Mapping):
        return {key: _thaw_json_value(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json_value(child) for child in value]  # type: ignore[return-value]
    return value


def _all_nodes(root: RenderNode) -> tuple[RenderNode, ...]:
    nodes: list[RenderNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        nodes.append(node)
        stack.extend(reversed(node.children))
    return tuple(nodes)


def _render_plan_text_values(
    *,
    nodes: tuple[RenderNode, ...],
    assets: tuple[RenderAsset, ...],
    css_rules: tuple[str, ...],
    scripts: tuple[str, ...],
    compatibility_issues: tuple[CanvasCompatibilityIssue, ...],
):
    for node in nodes:
        yield node.node_id
        yield node.tag
        for name, value in node.attributes:
            yield name
            yield value
        if node.text is not None:
            yield node.text
    for asset in assets:
        yield asset.asset_id
        yield asset.mime_type
    yield from css_rules
    yield from scripts
    for issue in compatibility_issues:
        yield issue.code
        yield issue.message
        if issue.location is not None:
            yield issue.location


def _require_tuple_of(values: object, item_type: type[object], field_name: str) -> None:
    if not isinstance(values, tuple) or not all(isinstance(value, item_type) for value in values):
        raise CanvasLimitError(f"{field_name} must be an immutable tuple")
