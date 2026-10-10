from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable, Mapping
from uuid import uuid4

from loguru import logger

from tldw_chatbook.Backup_Recovery.runtime_producer_lifetime import (
    ProducerLifetime,
    producer_call,
)

from tldw_chatbook.runtime_policy.registry import CAPABILITY_REGISTRY
from tldw_chatbook.runtime_policy.types import RuntimeSourceState
from tldw_chatbook.Library.library_tool_contract import (
    LIBRARY_TOOL_DESCRIPTORS,
    LibraryToolDescriptor,
)

from .activation import batch_guard, guarded, request_guard
from .client import MCPClient
from .tool_results import MCPToolResult, current_dispatch, transport_failure
from .local_runtime_delegate import LocalMCPRuntimeDelegate
from .local_store import (
    LocalApprovalRequest,
    LocalExternalMCPProfile,
    LocalGovernanceRule,
    LocalMCPStore,
)

if TYPE_CHECKING:
    from tldw_chatbook.Plugins.authority import DataRoot
    from tldw_chatbook.Plugins.models import PackageInspection


_ENV_PLACEHOLDER_PATTERN = re.compile(
    r"^\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))$"
)
_SPAWN_ENV_BASELINE_KEYS = (
    "PATH",
    "HOME",
    "LANG",
    "LC_ALL",
    "TMPDIR",
    "TMP",
    "TEMP",
    "SYSTEMROOT",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
)

# task-1337 (plan Task 9): each descriptor-backed ``library_*`` tool resolves
# to a read action owned by its Library item type. Media/Notes/Prompts/Skills/
# Conversations reuse their registered local list/detail actions.
_LIBRARY_ITEM_TYPE_ACTION_NAMESPACE = {
    "media": "media.reading",
    "note": "notes",
    "prompt": "prompts",
    "skill": "skills",
    "conversation": "chat",
}

# chunking-agent-tools (Tasks 4-5, spec §6) + student-workflow (Task 1,
# spec §4/§6): writing operations map to their OWN registered action
# instead of the type-owned read. ``spec_save`` resolves to the dedicated
# ``library.templates/save`` verb, ``rechunk`` to ``library.media/rechunk``,
# and ``save`` to ``library.notes/save`` -- the Task-3 provisional derived
# READ mapping had to stop the moment the save handler went live (a live
# write resolving to a read action under policy would be wrong even though
# the v7 CRUD validator still guards the write itself).
_LIBRARY_TOOL_ACTION_OVERRIDES = {
    "spec_save": "library.templates.save.local",
    "rechunk": "library.media.rechunk.local",
    "save": "library.notes.save.local",
}


# chunking-agent-tools (Qodo review, PR #1976): the two media chunk READ
# tools are single-item detail reads (by id, exactly like ``get``), so their
# operations derive the DETAIL action -- not the browse-level list the
# provisional non-get fallback resolved them to.
_DETAIL_READ_OPERATIONS = frozenset({"get", "structure", "chunk"})


def _library_tool_action_id(descriptor: LibraryToolDescriptor) -> str:
    """Map one Library descriptor to its policy action id.

    list/search/spec_list are browse-level reads; get and the single-item
    chunk reads (structure, chunk) are detail-level reads; writing
    operations carry an explicit override above. Derived from the descriptor
    table so the policy surface can never drift from the tool contract.
    """
    override = _LIBRARY_TOOL_ACTION_OVERRIDES.get(descriptor.operation)
    if override is not None:
        return override
    action = "detail" if descriptor.operation in _DETAIL_READ_OPERATIONS else "list"
    namespace = _LIBRARY_ITEM_TYPE_ACTION_NAMESPACE[descriptor.item_type]
    return f"{namespace}.{action}.local"


_TOOL_ACTION_IDS = {
    "chat_with_llm": "chat.launch.local",
    "chat_with_character": "character.sessions.launch.local",
    "search_rag": "media.reading.list.local",
    "search_conversations": "chat.list.local",
    "create_note": "notes.create.local",
    "search_notes": "notes.list.local",
    "list_characters": "character.persona.list.local",
    "create_character": "character.persona.create.local",
    "update_character": "character.persona.update.local",
    "get_conversation_history": "chat.detail.local",
    "export_conversation": "chat.detail.local",
    **{
        name: _library_tool_action_id(descriptor)
        for name, descriptor in LIBRARY_TOOL_DESCRIPTORS.items()
    },
}
_RESOURCE_ACTION_IDS = (
    ("conversation://", "chat.detail.local"),
    ("note://", "notes.detail.local"),
    ("character://", "character.persona.detail.local"),
    ("media://", "media.reading.detail.local"),
    ("rag-chunk://", "media.reading.detail.local"),
)
_REQUEST_METHOD_ACTION_IDS = {
    "initialize": "mcp.runtime.observe.local",
    "status/get": "mcp.runtime.observe.local",
    "tools/list": "mcp.inventory.list.local",
    "resources/list": "mcp.inventory.list.local",
    "prompts/list": "mcp.inventory.list.local",
}


def _default_manifest_provider() -> dict[str, Any]:
    from .server import describe_local_mcp_capabilities

    return describe_local_mcp_capabilities()


class MCPGovernanceDenied(PermissionError):
    """The in-process runtime-governance profile refused this call outright.

    Raised ONLY by :meth:`LocalMCPControlService._require_runtime_
    governance_allowed` -- the ONE seam where a governance RULE (a
    different permission system from the Hub's own Allow/Ask/Off gate,
    ``unified_control_plane_service.py``'s ``gate_tool_test()``) denies an
    action before it ever reaches the tool. Subclasses ``PermissionError``
    so any existing ``except PermissionError`` handler upstream keeps
    working unchanged.

    task-2537 (PR-T3 fix round B, item 3): exists so a caller
    (``mcp_workbench._is_permission_refusal()``) can tell "governance
    refused this call" apart from an unrelated ``PermissionError`` a
    TOOL'S OWN body might raise (e.g. a genuine OS EACCES reading a
    permission-denied path) -- both used to be plain ``PermissionError``,
    indistinguishable by type, so a real per-tool failure risked
    misrendering as a refusal that never reached the tool.
    """


class LocalMCPControlService:
    def _maintenance_close_admission(self):
        """Fence new calls before lower storage admission closes."""
        self._producer_lifetime.close()

    async def _maintenance_drain(self, deadline):
        """Wait for accepted calls without cancelling their native work."""
        return await self._producer_lifetime.drain(deadline)

    def _maintenance_resume(self):
        """Reopen only after accepted work and ordinary storage have settled."""
        self._producer_lifetime.resume()

    def __init__(
        self,
        *,
        store: LocalMCPStore,
        client: MCPClient | None = None,
        manifest_provider: Callable[[], dict[str, Any]] | None = None,
        policy_enforcer: Any | None = None,
        runtime_delegate: LocalMCPRuntimeDelegate | None = None,
        credential_service=None,
    ) -> None:
        self._producer_lifetime = ProducerLifetime()
        self.store = store
        self.credential_service = credential_service
        self.client = client
        self.manifest_provider = manifest_provider or _default_manifest_provider
        self.policy_enforcer = policy_enforcer
        # chunking-agent-tools (Task 5, spec §6): the default delegate rides
        # the SAME enforcer so the lazily composed shared Library service
        # (and through it the writing chunk tools) is service-level gated --
        # exactly the handle the runtime-gate methods below enforce with.
        self.runtime_delegate = runtime_delegate or LocalMCPRuntimeDelegate(
            manifest_provider=self.manifest_provider,
            policy_enforcer=self.policy_enforcer,
        )
        if isinstance(self.client, MCPClient):
            self.client._definition_store = self.store
        if isinstance(self.runtime_delegate, LocalMCPRuntimeDelegate):
            self.runtime_delegate._definition_store = self.store
        self._runtime_activity_limit = 50
        self.connection_ownership = None

    def save_owned_profile(
        self,
        *,
        installation_id: str,
        inspection: PackageInspection,
        component_id: str,
        data_root: DataRoot | None = None,
        session_isolation: str = "separate",
        protocol_version: str = "2026-07-28",
        credential_reference: str | None = None,
        credential_generation: int | None = None,
        development_loopback: bool = False,
    ) -> dict:
        """Save exact host configuration data. This never connects or grants tools."""
        from dataclasses import asdict
        from pathlib import Path

        from tldw_chatbook.Plugins.package_files import canonical_json

        self._require_allowed("mcp.external_profiles.configure.local")
        component = inspection.inventory[component_id]
        if (
            component.kind != "mcp"
            or component.activation_blockers
            or component.support != "supported"
        ):
            raise PermissionError("plugin_mcp_unavailable")
        definition = json.loads(component.definition_json)
        root = str(Path(inspection.materialized_identity).resolve(strict=True))
        if data_root is not None and data_root.installation_id != installation_id:
            raise PermissionError("plugin_data_owner_changed")
        command, args, cwd, environment = self._owned_launch_values(
            definition, root, str(data_root.path) if data_root is not None else None
        )
        transport = "stdio" if definition["type"] == "stdio" else "streamable_http"
        ref = asdict(data_root) if data_root is not None else None
        if ref is not None:
            ref["path"] = str(ref["path"])
        owner = {
            "installation_id": installation_id,
            "revision_digest": inspection.effective_digest,
            "component_id": component_id,
            "definition_digest": hashlib.sha256(
                canonical_json(definition).encode()
            ).hexdigest(),
            "environment": environment,
            "data_root": ref,
            "session_isolation": session_isolation,
            "literal_headers": definition.get("headers", {}),
        }
        configuration = {
            "transport": transport,
            "command": command,
            "args": args,
            "cwd": cwd,
            "url": definition.get("url", ""),
            "protocol_version": protocol_version,
            "development_loopback": development_loopback,
            "credential_reference": credential_reference,
            "credential_generation": credential_generation,
            "plugin_owner": owner,
        }
        profile_id = (
            "plugin-"
            + hashlib.sha256(canonical_json(configuration).encode()).hexdigest()[:56]
        )
        record = LocalExternalMCPProfile(profile_id=profile_id, **configuration)
        return self.store.save_profile(record).to_dict()

    @staticmethod
    def _owned_launch_values(definition, root, data_path, baseline=None):
        """Expand retained portable fields once, with host variables assigned last."""
        import shutil
        from pathlib import Path

        if definition["type"] == "stdio" and data_path is None:
            raise ValueError("plugin_configuration_unresolved")
        variables = {"PLUGIN_ROOT": root}
        if data_path is not None:
            variables["PLUGIN_DATA"] = data_path

        def expand(value):
            def substitute(match):
                if match.group(1) not in variables:
                    raise ValueError("plugin_configuration_unresolved")
                return variables[match.group(1)]

            return re.sub(r"\$\{(PLUGIN_ROOT|PLUGIN_DATA)\}", substitute, value)

        source_env = definition.get("env", {})
        if source_env.keys() & {"PLUGIN_ROOT", "PLUGIN_DATA"}:
            raise ValueError("plugin_reserved_variable")
        baseline = os.environ if baseline is None else baseline
        environment = {
            key: baseline[key] for key in _SPAWN_ENV_BASELINE_KEYS if key in baseline
        }
        environment.update({key: expand(value) for key, value in source_env.items()})
        environment.update(variables)
        if definition["type"] != "stdio":
            return "", (), None, {}
        token = definition["command"]
        command = (
            str(Path(root) / token)
            if token.startswith("./")
            else shutil.which(token, path=environment.get("PATH"))
        )
        if not command:
            raise ValueError("plugin_executable_unavailable")
        args = tuple(expand(value) for value in definition.get("args", []))
        cwd = expand(definition.get("cwd", "${PLUGIN_ROOT}"))
        if not Path(cwd).is_absolute():
            cwd = str(Path(root) / cwd)
        return command, args, cwd, environment

    def _credentials(self):
        if self.credential_service is None:
            from tldw_chatbook.config import create_mcp_credential_service

            self.credential_service = create_mcp_credential_service(
                self.store.path.parent
            )
        return self.credential_service

    def capture_connection_mapping(
        self,
        *,
        installation_id: str,
        mapping_id: str,
        inspection,
        component_id: str,
        profile_id: str,
    ) -> dict:
        """Capture an existing host profile and retained MCP definition.

        This is metadata capture, not reviewed publication or execution permission.
        M4 owns registration into the coordinator's durable mutation protocol.
        """
        from tldw_chatbook.Plugins.authority import Mapping as PluginMapping
        from tldw_chatbook.Skills_Interop.skill_trust_crypto import (
            canonical_json,
            sha256_hex,
        )

        from .credential_bindings import CredentialError, endpoint_origin

        try:
            component = inspection.inventory[component_id]
            profile = self.store.get_profile(profile_id)
            if (
                component.kind != "mcp"
                or component.support != "supported"
                or component.selection != "selected"
                or component.activation_blockers
                or profile is None
            ):
                raise CredentialError("credential_mapping_invalid")
            definition = json.loads(component.definition_json)
            if profile.plugin_owner is not None:
                owner = profile.plugin_owner
                if (
                    owner["installation_id"],
                    owner["component_id"],
                    owner["revision_digest"],
                    owner["definition_digest"],
                ) != (
                    installation_id,
                    component_id,
                    inspection.effective_digest,
                    sha256_hex(canonical_json(definition)),
                ):
                    raise CredentialError("credential_mapping_invalid")
                data_root = owner["data_root"]
                if (
                    data_root is not None
                    and data_root["installation_id"] != installation_id
                ):
                    raise CredentialError("credential_mapping_invalid")
                expected = self._owned_launch_values(
                    definition,
                    inspection.materialized_identity,
                    data_root["path"] if data_root else None,
                    owner["environment"],
                )
                if (
                    expected
                    != (
                        profile.command,
                        profile.args,
                        profile.cwd,
                        owner["environment"],
                    )
                    or profile.url != definition.get("url", "")
                    or owner["literal_headers"] != definition.get("headers", {})
                ):
                    raise CredentialError("credential_mapping_invalid")
            elif (
                definition.get("type") != "streamable-http"
                or definition.get("url") != profile.url
            ):
                raise CredentialError("credential_mapping_invalid")
            configuration = profile.to_input_dict()
            configuration.pop("created_at")
            configuration.pop("updated_at")
            bindings = []
            if profile.credential_reference is not None:
                bindings.append(
                    self._credentials().snapshot_binding(
                        profile.credential_reference,
                        profile.credential_generation,
                        endpoint_origin(profile.url),
                    )
                )
            result = PluginMapping.model_validate(
                {
                    "installation_id": installation_id,
                    "mapping_id": mapping_id,
                    "component_id": component_id,
                    "revision_digest": inspection.effective_digest,
                    "kind": "connection",
                    "target_reference": "local:" + profile.profile_id,
                    "definition_digest": sha256_hex(canonical_json(definition)),
                    "configuration_digest": sha256_hex(canonical_json(configuration)),
                    "credential_bindings": bindings,
                }
            )
            return result.model_dump(mode="json")
        except CredentialError:
            raise
        except Exception:  # noqa: BLE001 -- sanitized authority boundary
            raise CredentialError("credential_mapping_invalid") from None

    def capture_tool_mapping(
        self, connection: dict, inspection, tool_name: str
    ) -> dict:
        """Bind a discovered tool to its reviewed connection and exact definition."""
        from .credential_bindings import CredentialError
        from .permission_store import definition_hash

        self.validate_connection_mapping(connection, inspection)
        profile_id = connection["target_reference"].removeprefix("local:")
        snapshot = self.store.get_discovery_snapshot(profile_id) or {}
        tools = [
            tool for tool in snapshot.get("tools", ()) if tool.get("name") == tool_name
        ]
        if (
            len(tools) != 1
            or not isinstance(tool_name, str)
            or not tool_name
            or "::" in tool_name
        ):
            raise CredentialError("mcp_tool_mapping_invalid")
        tool = tools[0]
        return {
            **connection,
            "kind": "tool",
            "mapping_id": "tool-"
            + hashlib.sha256(
                (connection["mapping_id"] + "\0" + tool_name).encode()
            ).hexdigest(),
            "target_reference": connection["target_reference"] + "::" + tool_name,
            "definition_digest": definition_hash(
                tool.get("description", ""), tool.get("inputSchema", {})
            ),
        }

    def validate_connection_mapping(self, mapping: dict, inspection) -> None:
        """Validate every mapping field against retained material/current owners."""
        from .credential_bindings import CredentialError

        try:
            reference = mapping["target_reference"]
            if mapping["kind"] == "tool":
                profile, separator, name = reference.partition("::")
                if not separator:
                    raise CredentialError("mcp_tool_mapping_invalid")
                connection = self.capture_connection_mapping(
                    installation_id=mapping["installation_id"],
                    mapping_id=mapping["component_id"],
                    inspection=inspection,
                    component_id=mapping["component_id"],
                    profile_id=profile.removeprefix("local:"),
                )
                if self.capture_tool_mapping(connection, inspection, name) != mapping:
                    raise CredentialError("credential_mapping_changed")
                return
            if mapping["kind"] != "connection" or not reference.startswith("local:"):
                raise CredentialError("credential_mapping_unsupported")
            current = self.capture_connection_mapping(
                installation_id=mapping["installation_id"],
                mapping_id=mapping["mapping_id"],
                inspection=inspection,
                component_id=mapping["component_id"],
                profile_id=reference.removeprefix("local:"),
            )
            if current != mapping:
                raise CredentialError("credential_mapping_changed")
        except CredentialError:
            raise
        except Exception:  # noqa: BLE001 -- sanitized authority boundary
            raise CredentialError("credential_mapping_invalid") from None

    def get_overview(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        inventory = self.get_inventory()
        external_servers = self.get_external_servers()
        governance = self.get_governance()
        return {
            "inventory": {
                "tools": len(inventory.get("tools", [])),
                "resources": len(inventory.get("resources", [])),
                "prompts": len(inventory.get("prompts", [])),
            },
            "external_servers": {
                "profiles": len(external_servers),
                "discovery_snapshots": sum(
                    1 for item in external_servers if item.get("discovery_snapshot")
                ),
            },
            "governance": {
                "rules": len(governance),
            },
        }

    def get_inventory(self) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.list.local")
        manifest = self.manifest_provider() or {}
        inventory = dict(manifest)
        inventory["server_id"] = manifest.get("server_id", "local:tldw_chatbook")
        inventory["tools"] = list(manifest.get("tools", []))
        inventory["resources"] = list(manifest.get("resources", []))
        inventory["prompts"] = list(manifest.get("prompts", []))
        return inventory

    def get_external_servers(self) -> list[dict[str, Any]]:
        """List external server profiles with their discovery snapshots.

        task-236: reads the whole catalog from ONE store load via
        ``get_external_catalog`` (was 1 + N loads: ``list_profiles`` plus a
        ``get_discovery_snapshot`` per profile). Governance gating and the
        returned shape are unchanged.

        Returns:
            One dict per profile: the profile fields plus
            ``discovery_snapshot`` and ``is_connected``.
        """
        self._require_allowed("mcp.external_profiles.list.local")
        client = self.client
        active_sessions = getattr(client, "sessions", {}) if client is not None else {}
        catalog_reader = getattr(self.store, "get_external_catalog", None)
        if catalog_reader is not None:
            catalog = catalog_reader()
        else:
            # Duck-typed store double without the joined reader (the store is
            # a constructor-injected dependency): legacy per-item reads.
            catalog = [
                (profile, self.store.get_discovery_snapshot(profile.profile_id))
                for profile in self.store.list_profiles()
            ]
        records = [
            {**profile.to_dict(), "discovery_snapshot": snapshot}
            for profile, snapshot in catalog
        ]
        return self._project_external_catalog(records, active_sessions=active_sessions)

    def _project_external_catalog(
        self,
        records: list[dict[str, Any]],
        *,
        active_sessions: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Project checked public catalog fields using live caller connections."""
        servers: list[dict[str, Any]] = []
        for record in records:
            profile_id = record["profile_id"]
            servers.append(
                {
                    **record,
                    "is_connected": (
                        self.connection_ownership.is_connected(profile_id)
                        if record.get("plugin_owner") is not None
                        and self.connection_ownership is not None
                        else profile_id in active_sessions
                        and not getattr(active_sessions[profile_id], "_closed", False)
                    ),
                }
            )
        return servers

    def save_external_profile(
        self,
        profile: Mapping[str, Any] | LocalExternalMCPProfile,
    ) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.configure.local")
        strict_input = (
            profile.to_input_dict()
            if isinstance(profile, LocalExternalMCPProfile)
            else profile
        )
        record = LocalExternalMCPProfile.from_input_dict(strict_input)
        existing = self.store.get_profile(record.profile_id)
        if record.plugin_owner is not None or (
            existing is not None and existing.plugin_owner is not None
        ):
            raise PermissionError("plugin_owned_profile")
        return self.store.save_profile(record).to_dict()

    @producer_call
    @guarded
    async def connect_profile(self, profile_id: str) -> dict[str, Any]:
        """Connect or replace a profile session and persist fresh discovery.

        Discovery and persistence failures clean up the session established by
        this call only if its identity still owns the profile. Connection
        failures are reported as RuntimeError; discovery and persistence
        exceptions propagate. The last saved catalog is retained when discovery
        fails before saving.

        Args:
            profile_id: ID of the stored local stdio profile.

        Returns:
            The newly discovered tools, resources and prompts snapshot.

        Raises:
            PermissionError: Local launch permission is denied.
            KeyError: The profile is unknown.
            RuntimeError: Spawn environment resolution, connection or usable
                capability discovery fails.
            OSError: Snapshot persistence fails.
        """
        from tldw_chatbook.Agents.mcp_tool_provider import (
            check_mcp_invocation_policies,
            current_mcp_invocation_policies,
        )

        check_mcp_invocation_policies()
        if current_mcp_invocation_policies():
            raise PermissionError("hook_mcp_connection_required")
        self._require_allowed("mcp.external_profiles.launch.local")
        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id: {profile_id}")
        profile_id = profile.profile_id
        if profile.plugin_owner is not None:
            from .connection_ownership import owned_invocation

            context = owned_invocation.get()
            if context is None or context.ownership is not self.connection_ownership:
                raise PermissionError("plugin_connection_authority_required")
            identity = await context.ownership.connect(
                context.snapshot, context.component_id, profile_id
            )
            return await self._get_client().describe_server(identity)

        client = self._get_client()
        if profile.credential_reference is not None and isinstance(client, MCPClient):
            client.credential_service = self._credentials()
        from .local_store import TransportProfile

        if hasattr(client, "connect_profile"):
            connected = await client.connect_profile(
                TransportProfile(
                    profile_id=profile.profile_id,
                    transport=profile.transport,
                    protocol_version=profile.protocol_version,
                    command=profile.command,
                    args=profile.args,
                    cwd=profile.cwd,
                    env=(
                        self._build_spawn_env(profile)
                        if profile.transport == "stdio"
                        else None
                    ),
                    url=profile.url,
                    development_loopback=profile.development_loopback,
                    credential_reference=profile.credential_reference,
                    credential_generation=profile.credential_generation,
                )
            )
        elif profile.transport == "stdio" and profile.protocol_version == "2025-03-26":
            # Explicit legacy injected-client compatibility, without claiming HTTP.
            connected = await client.connect_to_server(
                profile.profile_id,
                profile.command,
                args=list(profile.args),
                env=self._build_spawn_env(profile),
            )
        else:
            raise RuntimeError("mcp_transport_unsupported")
        if connected is False:
            diagnostic = getattr(client, "connection_diagnostics", {}).get(
                profile.profile_id
            )
            raise RuntimeError(
                diagnostic or f"Failed to connect profile: {profile.profile_id}"
            )

        session = getattr(client, "sessions", {}).get(profile_id)
        try:
            snapshot = await client.describe_server(profile.profile_id)
            if not self._has_capabilities(snapshot):
                raise RuntimeError(
                    f"Connected profile '{profile.profile_id}' returned no discoverable capabilities"
                )
            self.store.save_discovery_snapshot(profile.profile_id, snapshot)
            return snapshot
        except BaseException:
            # Only clean up the connection this call established, never a
            # different caller's pending connection or replacement session.
            if (
                session is not None
                and getattr(client, "sessions", {}).get(profile_id) is session
            ):
                await self._disconnect_best_effort(client, profile_id)
            raise

    @producer_call
    async def disconnect_profile(self, profile_id: str) -> bool:
        self._require_allowed("mcp.external_profiles.launch.local")
        profile = self.store.get_profile(profile_id)
        if profile is not None and profile.plugin_owner is not None:
            raise PermissionError("plugin_owned_profile")
        client = self._get_client()
        return await client.disconnect_from_server(profile_id)

    @producer_call
    @guarded
    async def test_external_profile(self, profile_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.external_profiles.trigger.local")
        snapshot = await self._describe_profile(profile_id, keep_connected=False)
        return {
            "ok": True,
            "profile_id": profile_id,
            "tools": len(snapshot.get("tools", [])),
            "resources": len(snapshot.get("resources", [])),
            "prompts": len(snapshot.get("prompts", [])),
        }

    @producer_call
    @guarded
    async def execute_external_tool_result(
        self,
        profile_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> MCPToolResult:
        """Execute through the existing gates, retaining the complete MCP result."""
        return await self._execute_external_tool(
            profile_id, tool_name, arguments, typed_result=True
        )

    @producer_call
    @guarded
    async def execute_external_tool(
        self,
        profile_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute a tool on an external stdio profile, connecting if needed.

        Args:
            profile_id: Id of the stored external profile.
            tool_name: Name of the tool to call on that server.
            arguments: Tool arguments; defaults to an empty dict.

        Returns:
            The raw result payload from ``MCPClient.call_tool``.

        Raises:
            PermissionError: If governance denies the trigger action.
            RuntimeError: If the client reports an error payload.
        """
        return await self._execute_external_tool(profile_id, tool_name, arguments)

    async def _execute_external_tool(
        self,
        profile_id: str,
        tool_name: str,
        arguments: dict[str, Any] | None = None,
        *,
        typed_result: bool = False,
    ) -> MCPToolResult | dict[str, Any]:
        from tldw_chatbook.Agents.mcp_tool_provider import (
            check_mcp_invocation_policies,
            current_mcp_invocation_policies,
        )

        check_mcp_invocation_policies()
        self._require_allowed("mcp.external_profiles.trigger.local")
        from tldw_chatbook.Agents.automatic_work_runtime import current_automatic_work

        automatic_work = current_automatic_work()
        if automatic_work is not None:
            automatic_work.check()
        client = self._get_client()
        profile = self.store.get_profile(profile_id)
        if profile is not None and profile.plugin_owner is not None:
            from .connection_ownership import owned_invocation

            context = owned_invocation.get()
            if context is None or context.ownership is not self.connection_ownership:
                raise PermissionError("plugin_connection_authority_required")
            result = await context.ownership.invoke(
                context,
                profile_id,
                tool_name,
                arguments or {},
                client.call_tool_result,
                automatic_work=automatic_work,
            )
            if typed_result:
                return result
            from .tool_results import project_tool_result

            return project_tool_result(result)
        sessions = getattr(client, "sessions", {})
        if current_mcp_invocation_policies():
            from .connection_ownership import ConnectionOwnership

            session = sessions.get(profile_id)
            if session is None or ConnectionOwnership._session_unavailable(session):
                raise PermissionError("hook_mcp_connection_required")
        if profile_id not in sessions:
            await self.connect_profile(profile_id)
        check_mcp_invocation_policies()
        if automatic_work is not None:
            automatic_work.check()
        if typed_result:
            call = getattr(client, "call_tool_result", None)
            if not callable(call):
                return transport_failure(
                    "mcp_typed_result_unavailable", current_dispatch()
                )
            return await call(profile_id, tool_name, arguments or {})
        payload = await client.call_tool(profile_id, tool_name, arguments or {})
        if isinstance(payload, dict) and "error" in payload:
            raise RuntimeError(payload["error"])
        return payload

    @producer_call
    @guarded
    async def refresh_external_profile(self, profile_id: str) -> dict[str, Any]:
        """Reconnect to discover and persist the current profile catalog.

        Successful refresh preserves whether the profile was connected at
        entry, removing its temporary session when initially disconnected.
        Observe and launch denials leave the existing connection untouched.
        A failed reconnect may leave the profile disconnected; the last saved
        catalog remains available when discovery fails before saving.
        Failed temporary cleanup reports an error with the fresh catalog already
        saved, retaining the owned session so disconnection can be retried.

        Args:
            profile_id: ID of the stored local stdio profile.

        Returns:
            The refreshed tools, resources and prompts snapshot.

        Raises:
            PermissionError: Local observe or launch permission is denied.
            KeyError: The profile is unknown.
            RuntimeError: Spawn environment resolution, connection, usable
                capability discovery or temporary-session cleanup fails.
            OSError: Snapshot persistence fails.
        """
        self._require_allowed("mcp.external_profiles.observe.local")
        profile = self.store.get_profile(profile_id)
        if profile is not None:
            profile_id = profile.profile_id
        client = self._get_client()
        was_connected = profile_id in getattr(client, "sessions", {})
        # describe_server reads cached discovery; reconnect to refresh it.
        snapshot = await self.connect_profile(profile_id)
        if not was_connected:
            # The real client's cached describe and synchronous store save do
            # not yield after session publication. Disconnect captures that
            # session's identity before its first suspension, so subsequent
            # replacement sessions are not owned by this temporary cleanup.
            session = getattr(client, "sessions", {}).get(profile_id)
            await self._disconnect_best_effort(client, profile_id)
            if (
                session is not None
                and getattr(client, "sessions", {}).get(profile_id) is session
            ):
                raise RuntimeError(
                    f"Failed to disconnect temporary profile session: {profile_id}"
                )
        return snapshot

    def delete_external_profile(self, profile_id: str) -> bool:
        self._require_allowed("mcp.external_profiles.configure.local")
        profile = self.store.get_profile(profile_id)
        if profile is not None and profile.plugin_owner is not None:
            raise PermissionError("plugin_owned_profile")
        return self.store.delete_profile(profile_id)

    def get_governance(self) -> list[dict[str, Any]]:
        self._require_allowed("mcp.governance.list.local")
        return [rule.to_dict() for rule in self.store.list_governance_rules()]

    def list_approval_requests(
        self,
        status: str | None = None,
        resolved_action_id: str | None = None,
    ) -> list[dict[str, Any]]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_status = str(status or "").strip()
        normalized_resolved_action_id = str(resolved_action_id or "").strip()
        requests = [
            request.to_dict() for request in self.store.list_approval_requests()
        ]
        if normalized_status:
            requests = [
                request
                for request in requests
                if request.get("status") == normalized_status
            ]
        if normalized_resolved_action_id:
            requests = [
                request
                for request in requests
                if request.get("resolved_action_id") == normalized_resolved_action_id
            ]
        return requests

    def get_advanced(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        governance = self.get_governance()
        approval_requests = self.list_approval_requests()
        governance_summary = {
            "rules": len(governance),
            "deny_rules": sum(
                1 for rule in governance if rule.get("decision") == "deny"
            ),
            "allow_rules": sum(
                1 for rule in governance if rule.get("decision") == "allow"
            ),
        }
        ask_rules = sum(1 for rule in governance if rule.get("decision") == "ask")
        pending_approvals = sum(
            1 for request in approval_requests if request.get("status") == "pending"
        )
        if ask_rules:
            governance_summary["ask_rules"] = ask_rules
        if pending_approvals:
            governance_summary["pending_approvals"] = pending_approvals
        payload = {
            "source": "local",
            "section": "advanced",
            "runtime_status": self.runtime_delegate.get_status(),
            "runtime_health": self.runtime_delegate.get_runtime_health(),
            "protocol": self.runtime_delegate.get_protocol_capabilities(),
            "protocol_diagnostics": self.runtime_delegate.get_protocol_diagnostics(),
            "governance": governance_summary,
        }
        recent_activity = self._recent_runtime_activity_entries(limit=5)
        if recent_activity:
            payload["recent_activity_count"] = len(
                self.store.list_runtime_activity(limit=self._runtime_activity_limit)
            )
            payload["recent_activity"] = recent_activity
        return payload

    def save_governance_rule(
        self,
        rule: Mapping[str, Any] | LocalGovernanceRule,
    ) -> dict[str, Any]:
        self._require_allowed("mcp.governance.configure.local")
        record = (
            rule
            if isinstance(rule, LocalGovernanceRule)
            else LocalGovernanceRule.from_dict(rule)
        )
        return self.store.save_governance_rule(record).to_dict()

    def delete_governance_rule(self, rule_id: str) -> bool:
        self._require_allowed("mcp.governance.configure.local")
        return self.store.delete_governance_rule(str(rule_id or ""))

    def preview_governance_decision(self, capability_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_capability_id = str(capability_id or "").strip()
        matched_rule = self._find_governance_rule(normalized_capability_id)
        return {
            "source": "local",
            "capability_id": normalized_capability_id,
            "decision": matched_rule.decision
            if matched_rule is not None
            else "inherit",
            "matched_rule_id": matched_rule.rule_id
            if matched_rule is not None
            else None,
            "notes": matched_rule.notes if matched_rule is not None else None,
        }

    def preview_runtime_access(
        self, action_name: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_allowed("mcp.governance.observe.local")
        normalized_action_name = str(action_name or "").strip()
        normalized_payload = dict(payload or {})
        if normalized_action_name == "runtime.batch":
            requests = normalized_payload.get("requests")
            if not isinstance(requests, list):
                raise ValueError("Runtime batch preview requires 'requests'.")
            return {
                "source": "local",
                "action_name": normalized_action_name,
                "items": [
                    self._governance_preview_for_runtime_action(
                        "runtime.request",
                        {
                            "method": request.get("method"),
                            "params": request.get("params")
                            if isinstance(request.get("params"), Mapping)
                            else {},
                        },
                    )
                    for request in requests
                    if isinstance(request, Mapping)
                ],
            }
        return self._governance_preview_for_runtime_action(
            normalized_action_name, normalized_payload
        )

    def approve_approval_request(self, request_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.approve.local")
        resolved = self.store.resolve_approval_request(
            str(request_id or ""), "approved"
        )
        if resolved is None:
            raise KeyError(f"Unknown approval request: {request_id}")
        return resolved.to_dict()

    def deny_approval_request(self, request_id: str) -> dict[str, Any]:
        self._require_allowed("mcp.governance.approve.local")
        resolved = self.store.resolve_approval_request(str(request_id or ""), "denied")
        if resolved is None:
            raise KeyError(f"Unknown approval request: {request_id}")
        return resolved.to_dict()

    def delete_approval_request(self, request_id: str) -> bool:
        self._require_allowed("mcp.governance.approve.local")
        return self.store.delete_approval_request(str(request_id or ""))

    def get_runtime_status(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "status": self.runtime_delegate.get_status(),
        }

    def get_runtime_health(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "health": self.runtime_delegate.get_runtime_health(),
        }

    def get_runtime_activity(self, limit: int = 20) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        normalized_limit = max(1, min(int(limit or 20), self._runtime_activity_limit))
        return {
            "source": "local",
            "limit": normalized_limit,
            "entries": self._recent_runtime_activity_entries(limit=normalized_limit),
        }

    def get_runtime_protocol_diagnostics(self) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.observe.local")
        return {
            "source": "local",
            "diagnostics": self.runtime_delegate.get_protocol_diagnostics(),
        }

    @producer_call
    @request_guard
    async def run_runtime_request(
        self, method: str, params: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_method = str(method or "").strip()
        normalized_params = dict(params or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "runtime.request",
                {"method": normalized_method, "params": normalized_params},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "runtime.request",
                {"method": normalized_method, "params": normalized_params},
            )
            self._record_runtime_activity(
                action_name="runtime.request",
                target=normalized_method,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        result = await self.runtime_delegate.request(
            normalized_method, normalized_params
        )
        self._record_runtime_activity(
            action_name="runtime.request",
            target=normalized_method,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "method": normalized_method,
            "params": normalized_params,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    @producer_call
    @batch_guard
    async def run_runtime_batch(
        self, requests: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...]
    ) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_requests = [dict(request) for request in requests]
        results: list[dict[str, Any]] = []
        for index, request in enumerate(normalized_requests):
            method = str(request.get("method") or "").strip()
            params = (
                request.get("params")
                if isinstance(request.get("params"), Mapping)
                else {}
            )
            governance = self._governance_preview_for_runtime_action(
                "runtime.request",
                {"method": method, "params": params},
            )
            if governance["decision"] == "deny":
                self._record_runtime_activity(
                    action_name="runtime.request",
                    target=method,
                    governance=governance,
                    ok=False,
                    blocked=True,
                    error=f"Denied by local governance: {governance['resolved_action_id']}",
                )
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": False,
                        "blocked": True,
                        "error": f"Denied by local governance: {governance['resolved_action_id']}",
                        "governance": self._compact_governance_preview(governance),
                    }
                )
                continue
            if (
                governance["decision"] == "ask"
                and governance.get("approval_status") != "approved"
            ):
                if governance.get("approval_request_id") is None:
                    governance = self._create_pending_runtime_approval(
                        "runtime.request", {"method": method, "params": params}
                    )
                    error_message = (
                        f"Approval required: {governance.get('approval_request_id')}"
                    )
                else:
                    error_message = self._approval_error_message(governance)
                self._record_runtime_activity(
                    action_name="runtime.request",
                    target=method,
                    governance=governance,
                    ok=False,
                    blocked=True,
                    error=error_message,
                )
                results.append(
                    {
                        "index": index,
                        "method": method,
                        "ok": False,
                        "blocked": True,
                        "error": error_message,
                        "governance": self._compact_governance_preview(governance),
                    }
                )
                continue
            result = await self.runtime_delegate.request(method, params)
            self._record_runtime_activity(
                action_name="runtime.request",
                target=method,
                governance=governance,
                ok=True,
            )
            results.append(
                {
                    "index": index,
                    "method": method,
                    "ok": True,
                    "result": result,
                    "governance": self._compact_governance_preview(governance),
                }
            )
        return {
            "source": "local",
            "results": results,
        }

    @producer_call
    @guarded
    async def execute_tool(
        self, tool_name: str, arguments: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_allowed("mcp.runtime.trigger.local")
        normalized_tool_name = str(tool_name or "").strip()
        normalized_arguments = dict(arguments or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "tool.execute",
                {"tool_name": normalized_tool_name, "arguments": normalized_arguments},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "tool.execute",
                {"tool_name": normalized_tool_name, "arguments": normalized_arguments},
            )
            self._record_runtime_activity(
                action_name="tool.execute",
                target=normalized_tool_name,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        observation = current_dispatch()
        observation.state = "uncertain"
        try:
            result = await self.runtime_delegate.execute_tool(
                normalized_tool_name, normalized_arguments
            )
        except Exception:
            # A known raised invocation is settled; cancellation stays uncertain.
            observation.state = "settled"
            raise
        observation.state = "settled"
        self._record_runtime_activity(
            action_name="tool.execute",
            target=normalized_tool_name,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "tool_name": normalized_tool_name,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    @producer_call
    @guarded
    async def read_resource(self, resource_uri: str) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.observe.local")
        normalized_resource_uri = str(resource_uri or "").strip()
        try:
            governance = self._require_runtime_governance_allowed(
                "resource.read",
                {"resource_uri": normalized_resource_uri},
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "resource.read",
                {"resource_uri": normalized_resource_uri},
            )
            self._record_runtime_activity(
                action_name="resource.read",
                target=normalized_resource_uri,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        result = await self.runtime_delegate.read_resource(normalized_resource_uri)
        self._record_runtime_activity(
            action_name="resource.read",
            target=normalized_resource_uri,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "resource_uri": normalized_resource_uri,
            "result": result,
            "governance": self._compact_governance_preview(governance),
        }

    @producer_call
    @guarded
    async def get_prompt(
        self, prompt_name: str, arguments: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        self._require_allowed("mcp.inventory.observe.local")
        normalized_prompt_name = str(prompt_name or "").strip()
        normalized_arguments = dict(arguments or {})
        try:
            governance = self._require_runtime_governance_allowed(
                "prompt.get",
                {
                    "prompt_name": normalized_prompt_name,
                    "arguments": normalized_arguments,
                },
            )
        except PermissionError as exc:
            governance = self._governance_preview_for_runtime_action(
                "prompt.get",
                {
                    "prompt_name": normalized_prompt_name,
                    "arguments": normalized_arguments,
                },
            )
            self._record_runtime_activity(
                action_name="prompt.get",
                target=normalized_prompt_name,
                governance=governance,
                ok=False,
                blocked=True,
                error=str(exc),
            )
            raise
        messages = await self.runtime_delegate.get_prompt(
            normalized_prompt_name, normalized_arguments
        )
        self._record_runtime_activity(
            action_name="prompt.get",
            target=normalized_prompt_name,
            governance=governance,
            ok=True,
        )
        return {
            "source": "local",
            "prompt_name": normalized_prompt_name,
            "arguments": normalized_arguments,
            "messages": messages,
            "governance": self._compact_governance_preview(governance),
        }

    def _get_client(self) -> MCPClient:
        if self.client is None:
            client = MCPClient()
            # TASK-27019 (AC#4): every connection gets a per-server dispatcher
            # for server-initiated sampling/elicitation -- policy from config
            # (default deny), sampling through the live chat provider,
            # elicitation as a confirmation through this store's approvals.
            try:
                from tldw_chatbook.MCP.live_server_request_wiring import (
                    build_server_request_dispatcher_factory,
                )

                client._server_request_dispatcher_factory = (
                    build_server_request_dispatcher_factory(self.store)
                )
            except Exception:  # noqa: BLE001 - wiring failure degrades to -32601
                logger.opt(exception=True).warning(
                    "could not wire MCP server-request handlers; "
                    "server-initiated requests will get method-not-found"
                )
            client._definition_store = self.store
            self.client = client
        return self.client

    def _build_spawn_env(self, profile: LocalExternalMCPProfile) -> dict[str, str]:
        resolved_env = {
            key: value
            for key in _SPAWN_ENV_BASELINE_KEYS
            if (value := os.environ.get(key)) not in (None, "")
        }
        resolved_env.update(profile.legacy_env_literals)
        resolved_env.update(profile.env_literals)
        for key, placeholder in profile.env_placeholders.items():
            match = _ENV_PLACEHOLDER_PATTERN.fullmatch(placeholder)
            if not match:
                raise RuntimeError(
                    f"Invalid env placeholder for '{key}': {placeholder}"
                )
            env_key = match.group("braced") or match.group("plain")
            env_value = os.environ.get(env_key)
            if env_value in (None, ""):
                raise RuntimeError(
                    f"Missing required environment variable '{env_key}' for profile '{profile.profile_id}'"
                )
            resolved_env[key] = env_value
        return resolved_env

    def _has_capabilities(self, snapshot: Mapping[str, Any]) -> bool:
        return any(
            snapshot.get(section) for section in ("tools", "resources", "prompts")
        )

    async def _disconnect_best_effort(self, client: MCPClient, profile_id: str) -> None:
        disconnect = getattr(client, "disconnect_from_server", None)
        if disconnect is None:
            return
        try:
            await disconnect(profile_id)
        except Exception:
            return

    async def _describe_profile(
        self, profile_id: str, *, keep_connected: bool
    ) -> dict[str, Any]:
        profile = self.store.get_profile(profile_id)
        if profile is None:
            raise KeyError(f"Unknown profile_id: {profile_id}")

        client = self._get_client()
        if profile.plugin_owner is not None:
            from .connection_ownership import owned_invocation

            context = owned_invocation.get()
            if context is None or context.ownership is not self.connection_ownership:
                raise PermissionError("plugin_connection_authority_required")
            ownership = context.ownership
            owner_id = ownership._owner_id(context.snapshot)
            existing = {
                connection.connection_id
                for connection in ownership.connections.values()
                if connection.profile is not None
                and connection.profile.profile_id == profile_id
                and owner_id in connection.owners
            }
            identity = await ownership.connect(
                context.snapshot, context.component_id, profile_id
            )
            try:
                return await client.describe_server(identity)
            finally:
                if not keep_connected and identity not in existing:
                    await ownership.detach(identity, owner_id)
        sessions = getattr(client, "sessions", {})
        was_connected = profile_id in sessions
        if not was_connected:
            await self.connect_profile(profile_id)
        snapshot = await client.describe_server(profile_id)
        self.store.save_discovery_snapshot(profile_id, snapshot)
        if (not was_connected) or (was_connected and not keep_connected):
            await self._disconnect_best_effort(client, profile_id)
        return snapshot

    def _find_governance_rule(self, action_id: str) -> LocalGovernanceRule | None:
        return next(
            (
                rule
                for rule in self.store.list_governance_rules()
                if rule.capability_id == action_id
            ),
            None,
        )

    def _governance_preview_for_runtime_action(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        resolved_action_id, fallback_action_ids = self._resolve_runtime_action_ids(
            action_name, payload
        )
        matched_rule = self._find_governance_rule(resolved_action_id)
        if matched_rule is None:
            matched_rule = next(
                (
                    rule
                    for fallback_action_id in fallback_action_ids
                    if (rule := self._find_governance_rule(fallback_action_id))
                    is not None
                ),
                None,
            )
        capability_entry = CAPABILITY_REGISTRY.get(resolved_action_id)
        governance = {
            "source": "local",
            "action_name": action_name,
            "resolved_action_id": resolved_action_id,
            "registry_capability_id": capability_entry.capability_id
            if capability_entry is not None
            else None,
            "decision": matched_rule.decision
            if matched_rule is not None
            else "inherit",
            "matched_rule_id": matched_rule.rule_id
            if matched_rule is not None
            else None,
            "notes": matched_rule.notes if matched_rule is not None else None,
        }
        if governance["decision"] == "ask":
            approval_request = self._find_latest_approval_request(
                self._approval_fingerprint(
                    str(governance["resolved_action_id"]), payload
                )
            )
            governance["approval_request_id"] = (
                approval_request.request_id if approval_request is not None else None
            )
            governance["approval_status"] = (
                approval_request.status if approval_request is not None else None
            )
        return governance

    def _find_latest_approval_request(
        self, payload_fingerprint: str
    ) -> LocalApprovalRequest | None:
        matches = [
            request
            for request in self.store.list_approval_requests()
            if request.payload_fingerprint == payload_fingerprint
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda request: (
                request.updated_at
                or request.created_at
                or datetime.min.replace(tzinfo=timezone.utc)
            ),
        )

    def _create_pending_runtime_approval(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        governance = self._governance_preview_for_runtime_action(action_name, payload)
        saved_request = self.store.save_approval_request(
            LocalApprovalRequest(
                request_id=f"approval-{uuid4().hex[:12]}",
                action_name=action_name,
                resolved_action_id=str(governance["resolved_action_id"]),
                registry_capability_id=str(governance["registry_capability_id"] or "")
                or None,
                payload=dict(payload),
                payload_fingerprint=self._approval_fingerprint(
                    str(governance["resolved_action_id"]), payload
                ),
                status="pending",
                matched_rule_id=str(governance["matched_rule_id"] or "") or None,
                notes=str(governance["notes"] or "") or None,
            )
        )
        governance["approval_request_id"] = saved_request.request_id
        governance["approval_status"] = saved_request.status
        return governance

    @staticmethod
    def _approval_error_message(governance: Mapping[str, Any]) -> str:
        if governance.get("approval_status") == "pending":
            return f"Approval pending: {governance.get('approval_request_id')}"
        if governance.get("approval_status") == "denied":
            return f"Approval denied: {governance.get('approval_request_id')}"
        return f"Approval required: {governance.get('approval_request_id')}"

    @staticmethod
    def _approval_fingerprint(
        resolved_action_id: str, payload: Mapping[str, Any]
    ) -> str:
        canonical_payload = json.dumps(
            {
                "resolved_action_id": resolved_action_id,
                "payload": payload,
            },
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
        return hashlib.sha256(canonical_payload.encode("utf-8")).hexdigest()

    def _require_runtime_governance_allowed(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        # task-2537 (fix round B, item 3): every raise below is
        # `MCPGovernanceDenied`, not bare `PermissionError` -- this is the
        # ONE seam that decides "governance refused this call outright",
        # and callers (`mcp_workbench._is_permission_refusal()`) need to
        # tell it apart from an unrelated `PermissionError` a tool's own
        # body might raise.
        governance = self._governance_preview_for_runtime_action(action_name, payload)
        if governance["decision"] == "deny":
            raise MCPGovernanceDenied(
                f"Denied by local governance: {governance['resolved_action_id']}"
            )
        if governance["decision"] == "ask":
            if governance.get("approval_status") == "approved":
                return governance
            if governance.get("approval_request_id") is None:
                governance = self._create_pending_runtime_approval(action_name, payload)
                raise MCPGovernanceDenied(
                    f"Approval required: {governance.get('approval_request_id')}"
                )
            raise MCPGovernanceDenied(self._approval_error_message(governance))
        return governance

    @staticmethod
    def _compact_governance_preview(governance: Mapping[str, Any]) -> dict[str, Any]:
        compact = {
            "resolved_action_id": governance.get("resolved_action_id"),
            "registry_capability_id": governance.get("registry_capability_id"),
            "decision": governance.get("decision"),
            "matched_rule_id": governance.get("matched_rule_id"),
            "notes": governance.get("notes"),
        }
        if (
            governance.get("decision") == "ask"
            or governance.get("approval_request_id") is not None
            or governance.get("approval_status") is not None
        ):
            compact["approval_request_id"] = governance.get("approval_request_id")
            compact["approval_status"] = governance.get("approval_status")
        return compact

    def _recent_runtime_activity_entries(self, *, limit: int) -> list[dict[str, Any]]:
        normalized_limit = max(1, min(int(limit or 20), self._runtime_activity_limit))
        return self.store.list_runtime_activity(limit=normalized_limit)

    def _record_runtime_activity(
        self,
        *,
        action_name: str,
        target: str,
        governance: Mapping[str, Any] | None,
        ok: bool,
        blocked: bool = False,
        error: str | None = None,
    ) -> None:
        from .activation import _INSPECTION, MCPActivationRequired
        from .recovery_activation import require_store_write

        if action_name == "runtime.request" and target in _INSPECTION:
            try:
                require_store_write(self.store, "mcp.local")
            except MCPActivationRequired:
                # Descriptor inspection stays passive until fresh owner setup.
                return
        entry = {
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "action_name": str(action_name or "").strip(),
            "target": str(target or "").strip(),
            "ok": bool(ok),
            "blocked": bool(blocked),
            "error": str(error) if error is not None else None,
            "resolved_action_id": governance.get("resolved_action_id")
            if governance is not None
            else None,
            "decision": governance.get("decision") if governance is not None else None,
            "matched_rule_id": governance.get("matched_rule_id")
            if governance is not None
            else None,
            "approval_request_id": governance.get("approval_request_id")
            if governance is not None
            else None,
            "approval_status": governance.get("approval_status")
            if governance is not None
            else None,
        }
        self.store.record_runtime_activity(entry, limit=self._runtime_activity_limit)

    def _resolve_runtime_action_ids(
        self,
        action_name: str,
        payload: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        if action_name == "tool.execute":
            tool_name = str(payload.get("tool_name") or "").strip()
            resolved_action_id = _TOOL_ACTION_IDS.get(
                tool_name, "mcp.runtime.trigger.local"
            )
            return resolved_action_id, ("mcp.runtime.trigger.local",)
        if action_name == "resource.read":
            resource_uri = str(payload.get("resource_uri") or "").strip()
            resolved_action_id = next(
                (
                    action_id
                    for prefix, action_id in _RESOURCE_ACTION_IDS
                    if resource_uri.startswith(prefix)
                ),
                "mcp.inventory.observe.local",
            )
            return resolved_action_id, ("mcp.inventory.observe.local",)
        if action_name == "prompt.get":
            return "prompts.preview.local", ("mcp.inventory.observe.local",)
        if action_name == "runtime.status.get":
            return "mcp.runtime.observe.local", ()
        if action_name == "runtime.request":
            method = str(payload.get("method") or "").strip()
            params = (
                payload.get("params")
                if isinstance(payload.get("params"), Mapping)
                else {}
            )
            return self._resolve_runtime_request_action_ids(method, params)
        raise ValueError(f"Unsupported local runtime action preview: {action_name}")

    def _resolve_runtime_request_action_ids(
        self,
        method: str,
        params: Mapping[str, Any],
    ) -> tuple[str, tuple[str, ...]]:
        if method in _REQUEST_METHOD_ACTION_IDS:
            return _REQUEST_METHOD_ACTION_IDS[method], ()
        if method == "tools/call":
            tool_name = str(params.get("name") or params.get("tool_name") or "").strip()
            resolved_action_id = _TOOL_ACTION_IDS.get(
                tool_name, "mcp.runtime.trigger.local"
            )
            return resolved_action_id, ("mcp.runtime.trigger.local",)
        if method == "resources/read":
            resource_uri = str(
                params.get("uri") or params.get("resource_uri") or ""
            ).strip()
            resolved_action_id = next(
                (
                    action_id
                    for prefix, action_id in _RESOURCE_ACTION_IDS
                    if resource_uri.startswith(prefix)
                ),
                "mcp.inventory.observe.local",
            )
            return resolved_action_id, ("mcp.inventory.observe.local",)
        if method == "prompts/get":
            return "prompts.preview.local", ("mcp.inventory.observe.local",)
        return "mcp.runtime.trigger.local", ()

    def _require_allowed(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(
            action_id=action_id,
            runtime_state_override=RuntimeSourceState(active_source="local"),
        )

# Callable provenance captured at definition time; no native authority is retained.
_CONSOLE_STANDARD_METHODS = (
    ("get_external_servers", LocalMCPControlService.get_external_servers),
    ("_project_external_catalog", LocalMCPControlService._project_external_catalog),
    ("get_inventory", LocalMCPControlService.get_inventory),
)
