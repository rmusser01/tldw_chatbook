"""One source-owned policy/catalog observation for stock Console consumers."""

from __future__ import annotations

import json
import math
import sys
import weakref
from collections.abc import Mapping
from dataclasses import dataclass, field

from loguru import logger

from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

from . import console_snapshot as snapshot
from . import hub_tool_catalog as hub_catalog
from . import unified_control_plane_service as control
from .client import (
    MAX_DESCRIPTOR_NAME_LENGTH,
    MAX_DESCRIPTOR_STRING_LENGTH,
    MAX_SCHEMA_BYTES,
    MAX_SCHEMA_DEPTH,
)
from .hub_tool_catalog import (
    DIRECT_RUNTIME_UNAVAILABLE_TOOLS,
    HubTool,
    builtin_tools_from_inventory,
    local_tools_from_record,
)
from .permission_store import EffectiveToolState
from .unified_control_plane_service import UnifiedMCPControlPlaneService


# Sharing eligibility only: larger/custom catalogs keep the ordinary route and
# its complete output. Conservative JSON/hash charges bound owner-loop work.
_MAX_SHARED_TOOLS = 256
_MAX_SHARED_CATALOG_BYTES = 1_048_576
_MAX_SHARED_NODES = 16_384
_MAX_SHARED_INTEGER_BITS = 4096


@dataclass(frozen=True)
class PreparedConsoleTool:
    """Detached tool data; each consumer receives its own mutable schema."""

    server_key: str
    server_label: str
    source: str
    name: str
    description: str
    tags: tuple[str, ...]
    stale: bool
    executable: bool
    schema_json: str | None
    effective: EffectiveToolState

    def to_hub_tool(self) -> HubTool:
        """Return run-owned data without touching any native source."""
        return HubTool(
            server_key=self.server_key,
            server_label=self.server_label,
            source=self.source,
            name=self.name,
            description=self.description,
            input_schema=json.loads(self.schema_json)
            if self.schema_json is not None
            else None,
            tags=self.tags,
            stale=self.stale,
            executable=self.executable,
        )


@dataclass(frozen=True, eq=False)
class ConsoleToolPreparation:
    """Issued data for one live composition, never an execution permission."""

    profile_id: str
    kill_switch: bool
    tools: tuple[PreparedConsoleTool, ...]
    _captured_sources: snapshot._CapturedSources = field(repr=False)
    _builtin_raw_name_exclusions: frozenset[str] = field(repr=False)
    _owned_profile_ids: frozenset[str] = field(repr=False)
    _includes_external_catalog: bool = field(repr=False)

    def require_current(self, service: UnifiedMCPControlPlaneService) -> None:
        """Refuse unissued or displaced source data before catalog adoption."""
        checker, code, namespace, inputs = _CONSOLE_PREPARATION_CHECK
        if (
            preparation_pipeline_current is not checker
            or checker.__code__ is not code
            or checker.__globals__ is not namespace
            or namespace is not globals()
            or not snapshot._controller_inputs_current(checker, inputs)
            or not checker()
            or self not in _ISSUED_PREPARATIONS
            or self._captured_sources.service is not service
            or not _resolver_current()
        ):
            raise RecoveryRequired("mcp_source_selection_changed")
        self._captured_sources.require_current()


_ISSUED_PREPARATIONS: weakref.WeakSet[ConsoleToolPreparation] = weakref.WeakSet()
_RESOLVER_BINDINGS = control._CONSOLE_TOOL_STATE_RESOLVER_BINDINGS
_CONTROL_MODULE = control


def _resolver_current() -> bool:
    """Check the stock pure helper chain without a file read or policy cache."""
    if (
        sys.modules.get(control.__name__) is not _CONTROL_MODULE
        or control._CONSOLE_TOOL_STATE_RESOLVER_BINDINGS is not _RESOLVER_BINDINGS
        or not snapshot.controller_precheck_pipeline_current()
    ):
        return False
    for namespace, name, function, code, inputs in _RESOLVER_BINDINGS:
        module = sys.modules.get(function.__module__)
        if (
            module is None
            or vars(module) is not namespace
            or namespace.get(name) is not function
            or function.__code__ is not code
            or function.__globals__ is not namespace
            or not snapshot._controller_inputs_current(function, inputs)
        ):
            return False
    return control.resolve_effective_state is _RESOLVER_BINDINGS[1][2]


def _shared_normalization_eligible(records, inventory, *, owned_profile_ids) -> bool:
    """Bound raw normalization before copying, encoding or policy hashing.

    The walk accepts only plain JSON and consumes aggregate budgets across all
    participating tools. It never materializes dict/list children. Scalar size
    checks precede encoding; conservative escape charges also cover the policy
    hash's ASCII JSON. These limits select sharing, not advertised capability.
    """
    total_bytes = _MAX_SHARED_CATALOG_BYTES
    nodes = _MAX_SHARED_NODES
    tools = _MAX_SHARED_TOOLS
    schema_bytes = MAX_SCHEMA_BYTES

    def charge(amount):
        nonlocal total_bytes, schema_bytes
        total_bytes -= amount
        schema_bytes -= amount
        if total_bytes < 0 or schema_bytes < 0:
            raise ValueError

    def visit(value, depth=1):
        nonlocal nodes
        nodes -= 1
        if nodes < 0 or depth > MAX_SCHEMA_DEPTH:
            raise ValueError
        kind = type(value)
        if kind is dict:
            if len(value) > nodes // 2:
                raise ValueError
            charge(2)
            for key, child in value.items():
                if type(key) is not str:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                    raise ValueError
                visit(key, depth)
                charge(2)  # Colon and separator, including a conservative last comma.
                visit(child, depth + 1)
        elif kind is list:
            if len(value) > nodes:
                raise ValueError
            charge(2)
            for child in value:
                charge(1)
                visit(child, depth + 1)
        elif kind is str:
            if len(value) > min(schema_bytes, total_bytes):
                raise ValueError
            charge(2 + len(value) * (6 if value.isascii() else 12))
        elif value is None or kind is bool:
            charge(5)
        elif kind is int:
            if value.bit_length() > _MAX_SHARED_INTEGER_BITS:
                raise ValueError
            charge(max(1, value.bit_length()) + 1)
        elif kind is float and math.isfinite(value):
            charge(32)
        else:
            raise ValueError

    def inspect_tools(raw_tools):
        nonlocal tools, schema_bytes
        if raw_tools is None:
            return
        if type(raw_tools) is not list or len(raw_tools) > tools:  # noqa: E721 -- exact plain JSON excludes custom callbacks
            raise ValueError
        tools -= len(raw_tools)
        for raw in raw_tools:
            if type(raw) is not dict:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                raise ValueError
            for field_name, maximum in (
                ("name", MAX_DESCRIPTOR_NAME_LENGTH),
                ("description", MAX_DESCRIPTOR_STRING_LENGTH),
            ):
                value = raw.get(field_name)
                value = "" if value is None else value
                if type(value) is not str or len(value) > maximum:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                    raise ValueError
                schema_bytes = MAX_SCHEMA_BYTES
                visit(value)
            schema = raw.get("inputSchema")
            if schema is not None and type(schema) is not dict:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                raise ValueError
            schema_bytes = MAX_SCHEMA_BYTES
            visit(schema)

    try:
        if type(records) is not list or len(records) > _MAX_SHARED_TOOLS:  # noqa: E721 -- exact plain JSON excludes custom callbacks
            return False
        for record in records:
            if type(record) is not dict:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                return False
            profile_id = record.get("profile_id")
            if (
                type(profile_id) is not str  # noqa: E721 -- exact plain JSON excludes custom callbacks
                or len(profile_id) > MAX_DESCRIPTOR_NAME_LENGTH
            ):
                return False
            if (
                record.get("plugin_owner") is not None
                and profile_id not in owned_profile_ids
            ):
                continue
            schema_bytes = MAX_SCHEMA_BYTES
            visit(profile_id)
            discovery = record.get("discovery_snapshot")
            if discovery is not None:
                if type(discovery) is not dict:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                    return False
                inspect_tools(discovery.get("tools"))
        if inventory is not None:
            if type(inventory) is not dict:  # noqa: E721 -- exact plain JSON excludes custom callbacks
                return False
            inspect_tools(inventory.get("tools"))
    except ValueError:
        return False
    return True


def _schema_json(schema: dict | None) -> str | None:
    if schema is None:
        return None
    encoded = json.dumps(
        schema, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    # JSON conversion must not silently change custom tuples or mapping keys.
    if json.loads(encoded) != schema:
        raise ValueError("unsupported_console_tool_schema")
    return encoded


async def prepare_console_tools(
    service: UnifiedMCPControlPlaneService,
    *,
    profile_id: str = "default",
    include_mcp_catalog: bool,
    need_local_switch: bool,
    builtin_raw_name_exclusions: frozenset[str],
    owned_profile_ids: frozenset[str],
    maximum_tool_ids: frozenset[str] | None = None,
) -> ConsoleToolPreparation | None:
    """Prepare shared stock data while preserving loop affinity/native custody.

    Unsupported source adapters or schemas return ``None`` for the ordinary
    route. Native source failures and cancellation keep their original outcome.
    A builtin-only frozen maximum omits unused external reads and audits.
    Actual tool invocation still performs its existing fresh execution checks.
    """
    if not include_mcp_catalog and not need_local_switch:
        return None
    checker, code, namespace, inputs = _CONSOLE_PREPARATION_CHECK

    def pipeline_current():
        return (
            preparation_pipeline_current is checker
            and checker.__code__ is code
            and checker.__globals__ is namespace
            and namespace is globals()
            and snapshot._controller_inputs_current(checker, inputs)
            and checker()
        )

    if not pipeline_current():
        return None

    def require_pipeline_current():
        if not pipeline_current():
            raise RecoveryRequired("mcp_source_selection_changed")

    qualifier = (
        snapshot.standard_console_catalog_sources
        if include_mcp_catalog
        else snapshot.standard_console_sources
    )
    if not _resolver_current() or not qualifier(service):
        return None
    include_external = (
        include_mcp_catalog
        and hub_catalog.maximum_needs_external_catalog(maximum_tool_ids)
    )
    captured = snapshot._CapturedSources(service, capture_catalog=include_mcp_catalog)
    with service._producer_lifetime.operation():
        payload = await snapshot._owned_worker(captured.read_permission_payload)
        captured.require_current()
        require_pipeline_current()
        killed = bool(payload.get("kill_switch", False))
        tools: list[HubTool] = []
        records = []
        inventory = None
        if include_mcp_catalog and not killed:
            if include_external:
                # Governance/connection callbacks stay on this loop; the
                # catalog owner keeps its finite native read in a worker.
                records = await captured.catalog_callback()
                captured.require_current()
            try:
                inventory = captured.inventory_reader()
            except Exception as exc:  # same optional inventory behavior as provider
                logger.warning(
                    "Console built-in inventory read failed (exception_type={})",
                    type(exc).__name__,
                )
                inventory = None
            captured.require_current()
        require_pipeline_current()
        if not _shared_normalization_eligible(
            records, inventory, owned_profile_ids=owned_profile_ids
        ):
            return None
        for record in records:
            if (
                record.get("plugin_owner") is not None
                and record.get("profile_id") not in owned_profile_ids
            ):
                continue
            tools.extend(local_tools_from_record(record))
        if isinstance(inventory, Mapping):
            tools.extend(
                tool
                for tool in builtin_tools_from_inventory(inventory)
                if tool.name not in DIRECT_RUNTIME_UNAVAILABLE_TOOLS
                and tool.name not in builtin_raw_name_exclusions
            )
        require_pipeline_current()
        try:
            schemas = [_schema_json(tool.input_schema) for tool in tools]
        except (TypeError, ValueError, RecursionError):
            return None
        if not _resolver_current():
            raise RecoveryRequired("mcp_source_selection_changed")
        states = control._resolve_tool_states_from_payload(
            payload, tools, profile_id=profile_id
        )
        changed = [
            tool
            for tool in tools
            if states[(tool.server_key, tool.name)].config_changed
        ]
        if changed:

            def audit_changes():
                for tool in changed:
                    captured.require_current()
                    captured.audit_writer(
                        captured.permission,
                        tool,
                        profile_id=profile_id,
                        _captured_execution_log=captured.log_owner,
                    )
                    captured.require_current()

            await snapshot._owned_worker(
                lambda: captured.permission_call(lambda owner: audit_changes())
            )
        captured.require_current()
        require_pipeline_current()
        result = ConsoleToolPreparation(
            profile_id=profile_id,
            kill_switch=killed,
            tools=tuple(
                PreparedConsoleTool(
                    server_key=tool.server_key,
                    server_label=tool.server_label,
                    source=tool.source,
                    name=tool.name,
                    description=tool.description,
                    tags=tool.tags,
                    stale=tool.stale,
                    executable=tool.executable,
                    schema_json=schema,
                    effective=states[(tool.server_key, tool.name)],
                )
                for tool, schema in zip(tools, schemas)
            ),
            _captured_sources=captured,
            _builtin_raw_name_exclusions=builtin_raw_name_exclusions,
            _owned_profile_ids=owned_profile_ids,
            _includes_external_catalog=include_external,
        )
        _ISSUED_PREPARATIONS.add(result)
        result.require_current(service)
        return result


# Bounded callable provenance for the new pure result/conversion boundary.
# Native authority remains with the source owners and invocation gates.
_PREPARATION_MODULE = sys.modules[__name__]
_PREPARATION_CLASSES = (PreparedConsoleTool, ConsoleToolPreparation)

_PREPARATION_SLOT_MISSING = object()
_PREPARATION_CLASS_SHAPES = tuple(
    (
        cls,
        cls.__bases__,
        cls.__mro__,
        _PREPARATION_SLOT_MISSING,
        tuple((name, vars(cls).get(name, _PREPARATION_SLOT_MISSING)) for name in names),
    )
    for cls, names in (
        (
            PreparedConsoleTool,
            (
                "__new__",
                "__getattribute__",
                "__getattr__",
                "server_key",
                "server_label",
                "source",
                "name",
                "description",
                "tags",
                "stale",
                "executable",
                "schema_json",
                "effective",
            ),
        ),
        (
            ConsoleToolPreparation,
            (
                "__new__",
                "__getattribute__",
                "__getattr__",
                "__hash__",
                "__eq__",
                "profile_id",
                "kill_switch",
                "tools",
                "_captured_sources",
                "_builtin_raw_name_exclusions",
                "_owned_profile_ids",
                "_includes_external_catalog",
            ),
        ),
    )
)
_HUB_CATALOG_MODULE = hub_catalog
_HUB_CONSTRUCTION_ANCHOR = hub_catalog._HUB_TOOL_CONSTRUCTION_ANCHOR
_EXTERNAL_CATALOG_ANCHOR = hub_catalog._MAXIMUM_EXTERNAL_CATALOG_ANCHOR


def _class_shape_current(shape) -> bool:
    """Check only these three declared classes, without invoking descriptors."""
    cls, bases, mro, missing, slots = shape
    if type(cls) is not type or cls.__bases__ is not bases or cls.__mro__ is not mro:
        return False
    namespace = vars(cls)
    return all(namespace.get(name, missing) is original for name, original in slots)


def preparation_pipeline_current() -> bool:
    """Qualify original helpers before invoking result or converter methods."""
    if (
        sys.modules.get(__name__) is not _PREPARATION_MODULE
        or PreparedConsoleTool is not _PREPARATION_CLASSES[0]
        or ConsoleToolPreparation is not _PREPARATION_CLASSES[1]
        or sys.modules.get(hub_catalog.__name__) is not _HUB_CATALOG_MODULE
        or hub_catalog._HUB_TOOL_CONSTRUCTION_ANCHOR is not _HUB_CONSTRUCTION_ANCHOR
        or hub_catalog.HubTool is not _HUB_CONSTRUCTION_ANCHOR[0]
        or HubTool is not _HUB_CONSTRUCTION_ANCHOR[0]
    ):
        return False
    rule, code, namespace = _EXTERNAL_CATALOG_ANCHOR
    if (
        hub_catalog._MAXIMUM_EXTERNAL_CATALOG_ANCHOR is not _EXTERNAL_CATALOG_ANCHOR
        or hub_catalog.maximum_needs_external_catalog is not rule
        or rule.__code__ is not code
        or rule.__globals__ is not namespace
        or namespace is not vars(hub_catalog)
        or rule.__defaults__ is not None
        or rule.__kwdefaults__ is not None
        or rule.__closure__ is not None
    ):
        return False
    for owner, name, original, code, namespace, inputs in _PREPARATION_BINDINGS:
        if (
            owner.get(name) is not original
            or original.__code__ is not code
            or original.__globals__ is not namespace
            or not snapshot._controller_inputs_current(original, inputs)
        ):
            return False
    if not all(
        _class_shape_current(shape)
        for shape in (*_PREPARATION_CLASS_SHAPES, _HUB_CONSTRUCTION_ANCHOR[:5])
    ):
        return False
    for function, code, namespace, inputs in _HUB_CONSTRUCTION_ANCHOR[5]:
        if (
            function.__code__ is not code
            or function.__globals__ is not namespace
            or namespace is not vars(_HUB_CATALOG_MODULE)
            or not snapshot._controller_inputs_current(function, inputs)
        ):
            return False
    return True


_PREPARATION_BINDINGS = tuple(
    (
        owner,
        name,
        function,
        function.__code__,
        function.__globals__,
        snapshot._capture_controller_inputs(function),
    )
    for owner, name in (
        (globals(), "prepare_console_tools"),
        (vars(hub_catalog), "maximum_needs_external_catalog"),
        (globals(), "preparation_pipeline_current"),
        (globals(), "_resolver_current"),
        (globals(), "_class_shape_current"),
        (globals(), "_schema_json"),
        (globals(), "_shared_normalization_eligible"),
        (vars(PreparedConsoleTool), "__init__"),
        (vars(PreparedConsoleTool), "to_hub_tool"),
        (vars(ConsoleToolPreparation), "__init__"),
        (vars(ConsoleToolPreparation), "require_current"),
    )
    for function in (owner[name],)
)
_CONSOLE_PREPARATION_CHECK = (
    preparation_pipeline_current,
    preparation_pipeline_current.__code__,
    preparation_pipeline_current.__globals__,
    snapshot._capture_controller_inputs(preparation_pipeline_current),
)
