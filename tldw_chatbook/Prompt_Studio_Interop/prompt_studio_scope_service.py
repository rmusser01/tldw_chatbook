"""Source-aware routing for server-owned Prompt Studio resources."""

from __future__ import annotations

import inspect
from enum import Enum
from typing import Any, AsyncGenerator

from .server_prompt_studio_service import ServerPromptStudioService


class PromptStudioBackend(str, Enum):
    LOCAL = "local"
    SERVER = "server"


_LOCAL_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "prompt_studio.remote_only.local",
        "source": "local",
        "supported": False,
        "reason_code": "remote_only_surface",
        "user_message": "Server Prompt Studio projects, prompts, test cases, evaluations, optimizations, status, and events are unavailable in local/offline mode.",
        "affected_action_ids": [],
    }
]

_SERVER_UNSUPPORTED_CAPABILITIES = [
    {
        "operation_id": "prompt_studio.websocket_realtime.server",
        "source": "server",
        "supported": False,
        "reason_code": "client_adapter_missing",
        "user_message": "The server exposes Prompt Studio websocket realtime endpoints; this Chatbook adapter currently exposes REST operations and SSE event observation only.",
        "affected_action_ids": [],
    }
]


class PromptStudioScopeService:
    """Route Prompt Studio operations through the active server without local authority."""

    def __init__(self, *, server_service: Any = None, policy_enforcer: Any = None):
        self.server_service = server_service
        self.policy_enforcer = policy_enforcer

    def _normalize_mode(self, mode: PromptStudioBackend | str | None) -> PromptStudioBackend:
        if mode is None:
            return PromptStudioBackend.SERVER
        if isinstance(mode, PromptStudioBackend):
            return mode
        try:
            return PromptStudioBackend(str(mode))
        except ValueError as exc:
            raise ValueError(f"Invalid Prompt Studio backend: {mode}") from exc

    def _require_server_service(self, mode: PromptStudioBackend) -> Any:
        if mode == PromptStudioBackend.LOCAL:
            raise ValueError("Server Prompt Studio records are server-only; switch to server mode to manage them.")
        if self.server_service is None:
            raise ValueError("Server Prompt Studio backend is unavailable.")
        return self.server_service

    def _enforce_policy(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        self.policy_enforcer.require_allowed(action_id=action_id)

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        if inspect.isawaitable(value):
            return await value
        return value

    @staticmethod
    def _dump(payload: Any) -> Any:
        if hasattr(payload, "model_dump"):
            return payload.model_dump(mode="python", by_alias=True)
        if isinstance(payload, dict):
            return {key: PromptStudioScopeService._dump(value) for key, value in payload.items()}
        if isinstance(payload, list):
            return [PromptStudioScopeService._dump(item) for item in payload]
        return payload

    @classmethod
    def _normalize_response(
        cls,
        mode: PromptStudioBackend,
        payload: Any,
        *,
        kind: str | None = None,
        identifier: Any | None = None,
        parent_id: Any | None = None,
    ) -> Any:
        payload = cls._dump(payload)
        normalized = ServerPromptStudioService._normalize_response(
            payload,
            kind=kind,
            identifier=identifier,
            parent_id=parent_id,
        )
        return cls._rewrite_backend(normalized, mode.value)

    @classmethod
    def _rewrite_backend(cls, payload: Any, backend: str) -> Any:
        if isinstance(payload, list):
            return [cls._rewrite_backend(item, backend) for item in payload]
        if isinstance(payload, dict):
            record = {key: cls._rewrite_backend(value, backend) for key, value in payload.items()}
            if record.get("backend") == "server":
                record["backend"] = backend
            return record
        return payload

    def list_unsupported_capabilities(
        self,
        *,
        mode: PromptStudioBackend | str | None = None,
    ) -> list[dict[str, Any]]:
        normalized_mode = self._normalize_mode(mode)
        if normalized_mode == PromptStudioBackend.LOCAL:
            return [dict(item) for item in _LOCAL_UNSUPPORTED_CAPABILITIES]
        report: list[dict[str, Any]] = []
        for item in _SERVER_UNSUPPORTED_CAPABILITIES:
            if item["operation_id"] == "prompt_studio.websocket_realtime.server" and self._has_websocket_realtime_adapter():
                continue
            report.append(dict(item))
        return report

    def _has_websocket_realtime_adapter(self) -> bool:
        service = self.server_service
        if service is None:
            return False
        explicit_support = getattr(service, "supports_websocket_realtime", None)
        if explicit_support is not None:
            return bool(explicit_support)
        websocket_methods = (
            "connect_prompt_studio_websocket",
            "stream_prompt_studio_websocket",
            "stream_realtime_websocket",
        )
        return any(callable(getattr(service, method_name, None)) for method_name in websocket_methods)

    async def _call(
        self,
        *,
        mode: PromptStudioBackend | str | None,
        action_id: str,
        method_name: str,
        normalize_kind: str | None = None,
        identifier: Any | None = None,
        parent_id: Any | None = None,
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> Any:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy(action_id)
        result = await self._maybe_await(getattr(service, method_name)(*args, **(kwargs or {})))
        return self._normalize_response(
            normalized_mode,
            result,
            kind=normalize_kind,
            identifier=identifier,
            parent_id=parent_id,
        )

    async def create_project(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None, idempotency_key: str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.create.server",
            method_name="create_project",
            normalize_kind="project",
            args=(request_data,),
            kwargs={"idempotency_key": idempotency_key},
        )

    async def list_projects(self, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.list.server",
            method_name="list_projects",
            normalize_kind="project",
            kwargs=kwargs,
        )

    async def get_project(self, project_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.detail.server",
            method_name="get_project",
            normalize_kind="project",
            args=(project_id,),
        )

    async def update_project(self, project_id: int, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.update.server",
            method_name="update_project",
            normalize_kind="project",
            identifier=project_id,
            args=(project_id, request_data),
        )

    async def delete_project(self, project_id: int, *, mode: PromptStudioBackend | str | None = None, permanent: bool = False) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.delete.server",
            method_name="delete_project",
            identifier=project_id,
            args=(project_id,),
            kwargs={"permanent": permanent},
        )

    async def archive_project(self, project_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.archive.server",
            method_name="archive_project",
            normalize_kind="project",
            identifier=project_id,
            args=(project_id,),
        )

    async def unarchive_project(self, project_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.projects.restore.server",
            method_name="unarchive_project",
            normalize_kind="project",
            identifier=project_id,
            args=(project_id,),
        )

    async def get_project_stats(self, project_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.project_stats.detail.server",
            method_name="get_project_stats",
            normalize_kind="project_stats",
            identifier=project_id,
            args=(project_id,),
        )

    async def create_prompt(
        self,
        request_data: Any,
        *,
        mode: PromptStudioBackend | str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.create.server",
            method_name="create_prompt",
            normalize_kind="prompt",
            args=(request_data,),
            kwargs={"idempotency_key": idempotency_key},
        )

    async def list_prompts(self, project_id: int, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.list.server",
            method_name="list_prompts",
            normalize_kind="prompt",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_prompt(self, prompt_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.detail.server",
            method_name="get_prompt",
            normalize_kind="prompt",
            identifier=prompt_id,
            args=(prompt_id,),
        )

    async def update_prompt(self, prompt_id: int, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.update.server",
            method_name="update_prompt",
            normalize_kind="prompt",
            identifier=prompt_id,
            args=(prompt_id, request_data),
        )

    async def get_prompt_history(self, prompt_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompt_versions.list.server",
            method_name="get_prompt_history",
            normalize_kind="prompt_version",
            parent_id=prompt_id,
            args=(prompt_id,),
        )

    async def revert_prompt(self, prompt_id: int, *, version: int, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.restore.server",
            method_name="revert_prompt",
            normalize_kind="prompt",
            identifier=prompt_id,
            args=(prompt_id,),
            kwargs={"version": version},
        )

    async def preview_prompt(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.preview.server",
            method_name="preview_prompt",
            normalize_kind="prompt_preview",
            args=(request_data,),
        )

    async def convert_prompt(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.process.server",
            method_name="convert_prompt",
            normalize_kind="prompt_conversion",
            args=(request_data,),
        )

    async def execute_prompt(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.prompts.launch.server",
            method_name="execute_prompt",
            normalize_kind="prompt_execution",
            args=(request_data,),
        )

    async def create_test_case(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.create.server",
            method_name="create_test_case",
            normalize_kind="test_case",
            args=(request_data,),
        )

    async def create_test_cases_bulk(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.create.server",
            method_name="create_test_cases_bulk",
            normalize_kind="test_case",
            args=(request_data,),
        )

    async def list_test_cases(self, project_id: int, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.list.server",
            method_name="list_test_cases",
            normalize_kind="test_case",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_test_case(self, test_case_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.detail.server",
            method_name="get_test_case",
            normalize_kind="test_case",
            identifier=test_case_id,
            args=(test_case_id,),
        )

    async def update_test_case(self, test_case_id: int, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.update.server",
            method_name="update_test_case",
            normalize_kind="test_case",
            identifier=test_case_id,
            args=(test_case_id, request_data),
        )

    async def delete_test_case(self, test_case_id: int, *, mode: PromptStudioBackend | str | None = None, permanent: bool = False) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.delete.server",
            method_name="delete_test_case",
            identifier=test_case_id,
            args=(test_case_id,),
            kwargs={"permanent": permanent},
        )

    async def import_test_cases(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.import.server",
            method_name="import_test_cases",
            normalize_kind="test_case_import",
            args=(request_data,),
        )

    async def import_test_cases_csv_upload(
        self,
        project_id: int,
        csv_content: str | bytes,
        *,
        mode: PromptStudioBackend | str | None = None,
        filename: str = "prompt_studio_test_cases.csv",
        signature_id: int | None = None,
        auto_generate_names: bool = True,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.import.server",
            method_name="import_test_cases_csv_upload",
            normalize_kind="test_case_import",
            identifier=project_id,
            args=(project_id, csv_content),
            kwargs={
                "filename": filename,
                "signature_id": signature_id,
                "auto_generate_names": auto_generate_names,
            },
        )

    async def get_test_cases_csv_template(self, *, mode: PromptStudioBackend | str | None = None, signature_id: int | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.export.server",
            method_name="get_test_cases_csv_template",
            normalize_kind="test_case_template",
            identifier=signature_id,
            kwargs={"signature_id": signature_id},
        )

    async def export_test_cases(self, project_id: int, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.export.server",
            method_name="export_test_cases",
            normalize_kind="test_case_export",
            identifier=project_id,
            args=(project_id, request_data),
        )

    async def generate_test_cases(self, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.launch.server",
            method_name="generate_test_cases",
            normalize_kind="test_case",
            kwargs=kwargs,
        )

    async def run_test_cases(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.test_cases.launch.server",
            method_name="run_test_cases",
            normalize_kind="test_case_run",
            args=(request_data,),
        )

    async def create_evaluation(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.evaluations.create.server",
            method_name="create_evaluation",
            normalize_kind="evaluation",
            args=(request_data,),
        )

    async def list_evaluations(self, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.evaluations.list.server",
            method_name="list_evaluations",
            normalize_kind="evaluation",
            kwargs=kwargs,
        )

    async def get_evaluation(self, evaluation_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.evaluations.detail.server",
            method_name="get_evaluation",
            normalize_kind="evaluation",
            identifier=evaluation_id,
            args=(evaluation_id,),
        )

    async def delete_evaluation(self, evaluation_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.evaluations.delete.server",
            method_name="delete_evaluation",
            identifier=evaluation_id,
            args=(evaluation_id,),
        )

    async def create_optimization(
        self,
        request_data: Any,
        *,
        mode: PromptStudioBackend | str | None = None,
        idempotency_key: str | None = None,
    ) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.create.server",
            method_name="create_optimization",
            normalize_kind="optimization",
            args=(request_data,),
            kwargs={"idempotency_key": idempotency_key},
        )

    async def create_optimization_simple(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.launch.server",
            method_name="create_optimization_simple",
            normalize_kind="optimization_job",
            args=(request_data,),
        )

    async def list_optimizations(self, project_id: int, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.list.server",
            method_name="list_optimizations",
            normalize_kind="optimization",
            args=(project_id,),
            kwargs=kwargs,
        )

    async def get_optimization(self, optimization_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.detail.server",
            method_name="get_optimization",
            normalize_kind="optimization",
            identifier=optimization_id,
            args=(optimization_id,),
        )

    async def get_optimization_job_status(self, job_id: str, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.detail.server",
            method_name="get_optimization_job_status",
            normalize_kind="optimization_job",
            identifier=job_id,
            args=(job_id,),
        )

    async def cancel_optimization(self, optimization_id: int, *, mode: PromptStudioBackend | str | None = None, reason: str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimizations.cancel.server",
            method_name="cancel_optimization",
            normalize_kind="optimization",
            identifier=optimization_id,
            args=(optimization_id,),
            kwargs={"reason": reason},
        )

    async def get_optimization_strategies(self, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimization_strategies.list.server",
            method_name="get_optimization_strategies",
            normalize_kind="optimization_strategies",
            identifier="catalog",
        )

    async def get_optimization_history(self, optimization_id: int, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimization_iterations.list.server",
            method_name="get_optimization_history",
            normalize_kind="optimization_iteration",
            parent_id=optimization_id,
            args=(optimization_id,),
        )

    async def add_optimization_iteration(self, optimization_id: int, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimization_iterations.create.server",
            method_name="add_optimization_iteration",
            normalize_kind="optimization_iteration",
            parent_id=optimization_id,
            args=(optimization_id, request_data),
        )

    async def list_optimization_iterations(self, optimization_id: int, *, mode: PromptStudioBackend | str | None = None, **kwargs: Any) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimization_iterations.list.server",
            method_name="list_optimization_iterations",
            normalize_kind="optimization_iteration",
            parent_id=optimization_id,
            args=(optimization_id,),
            kwargs=kwargs,
        )

    async def compare_optimization_strategies(self, request_data: Any, *, mode: PromptStudioBackend | str | None = None) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.optimization_strategies.launch.server",
            method_name="compare_optimization_strategies",
            normalize_kind="optimization_comparison",
            args=(request_data,),
        )

    async def get_status(self, *, mode: PromptStudioBackend | str | None = None, warn_seconds: int = 30) -> dict[str, Any]:
        return await self._call(
            mode=mode,
            action_id="prompt_studio.status.detail.server",
            method_name="get_status",
            normalize_kind="status",
            kwargs={"warn_seconds": warn_seconds},
        )

    async def stream_events(
        self,
        *,
        mode: PromptStudioBackend | str | None = None,
        client_id: str,
        project_id: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        normalized_mode = self._normalize_mode(mode)
        service = self._require_server_service(normalized_mode)
        self._enforce_policy("prompt_studio.events.observe.server")
        async for event in service.stream_events(client_id=client_id, project_id=project_id):
            payload = self._dump(event)
            if isinstance(payload, dict):
                payload.setdefault("backend", normalized_mode.value)
            yield payload
