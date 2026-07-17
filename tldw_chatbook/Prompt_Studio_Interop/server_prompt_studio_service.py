"""Policy-gated active-server Prompt Studio service."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, AsyncGenerator, Mapping, Optional

from ..runtime_policy.bootstrap import build_runtime_api_client_provider_from_config
from ..runtime_policy.types import PolicyDeniedError
from ..tldw_api import (
    PromptStudioCompareStrategiesRequest,
    PromptStudioEvaluationCreate,
    PromptStudioOptimizationCreate,
    PromptStudioOptimizationIterationCreate,
    PromptStudioOptimizationSimpleCreateRequest,
    PromptStudioPromptConvertRequest,
    PromptStudioPromptCreate,
    PromptStudioPromptExecuteRequest,
    PromptStudioPromptPreviewRequest,
    PromptStudioPromptUpdate,
    PromptStudioProjectCreate,
    PromptStudioProjectUpdate,
    PromptStudioRunTestCasesRequest,
    PromptStudioTestCaseBulkCreate,
    PromptStudioTestCaseCreate,
    PromptStudioTestCaseExportRequest,
    PromptStudioTestCaseGenerateRequest,
    PromptStudioTestCaseImportRequest,
    PromptStudioTestCaseUpdate,
)
if TYPE_CHECKING:
    from ..tldw_api import TLDWAPIClient


class ServerPromptStudioService:
    """Execute REST/SSE-backed Prompt Studio operations against the active server."""

    supports_websocket_realtime = False

    def __init__(
        self,
        client: Optional[TLDWAPIClient] = None,
        *,
        policy_enforcer: Any | None = None,
        client_provider: Any | None = None,
    ) -> None:
        self.client = client
        self.policy_enforcer = policy_enforcer
        self.client_provider = client_provider

    @classmethod
    def from_config(
        cls,
        app_config: Mapping[str, Any],
        *,
        client_provider: Any | None = None,
        policy_enforcer: Any | None = None,
    ) -> "ServerPromptStudioService":
        if client_provider is not None:
            return cls(
                client=None,
                client_provider=client_provider,
                policy_enforcer=policy_enforcer,
            )
        return cls(
            client=None,
            client_provider=build_runtime_api_client_provider_from_config(app_config),
            policy_enforcer=policy_enforcer,
        )

    @classmethod
    def from_server_context_provider(
        cls,
        provider: Any,
        *,
        policy_enforcer: Any | None = None,
    ) -> "ServerPromptStudioService":
        return cls(client_provider=provider, policy_enforcer=policy_enforcer)

    def _require_client(self) -> TLDWAPIClient:
        if self.client is not None:
            return self.client
        if self.client_provider is not None:
            return self.client_provider.build_client()
        raise ValueError("TLDW API client is required for server Prompt Studio operations.")

    def _enforce(self, action_id: str) -> None:
        if self.policy_enforcer is None:
            return
        require_allowed = getattr(self.policy_enforcer, "require_allowed", None)
        require_ui_action_allowed = getattr(self.policy_enforcer, "require_ui_action_allowed", None)
        if callable(require_allowed):
            require_allowed(action_id=action_id)
            return
        if callable(require_ui_action_allowed):
            decision = require_ui_action_allowed(action_id=action_id)
            if decision is not None and getattr(decision, "allowed", True) is False:
                raise PolicyDeniedError(
                    action_id=action_id,
                    reason_code=getattr(decision, "reason_code", None) or "authority_denied",
                    user_message=getattr(decision, "user_message", None) or "Server Prompt Studio action is not allowed.",
                    effective_source=getattr(decision, "effective_source", None) or "server",
                    authority_owner=getattr(decision, "authority_owner", None) or "server",
                )

    @classmethod
    def _dump(cls, response: Any) -> Any:
        if hasattr(response, "model_dump"):
            return response.model_dump(mode="python", by_alias=True)
        if isinstance(response, list):
            return [cls._dump(item) for item in response]
        if isinstance(response, dict):
            return {key: cls._dump(value) for key, value in response.items()}
        return response

    @staticmethod
    def _model(request_data: Any, model_type: type[Any]) -> Any:
        if isinstance(request_data, model_type):
            return request_data
        return model_type(**dict(request_data or {}))

    @staticmethod
    def _with_record_id(
        kind: str,
        payload: dict[str, Any],
        identifier: Any | None = None,
    ) -> dict[str, Any]:
        record = dict(payload or {})
        record.setdefault("backend", "server")
        if identifier is None:
            identifier = record.get("id") or record.get("project_id") or record.get("prompt_id")
        if identifier is not None:
            record.setdefault("record_id", f"server:{kind}:{identifier}")
        return record

    @classmethod
    def _normalize_record(
        cls,
        payload: Any,
        *,
        kind: str | None = None,
        identifier: Any | None = None,
        parent_id: Any | None = None,
    ) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [
                cls._normalize_record(item, kind=kind, identifier=identifier, parent_id=parent_id)
                for item in payload
            ]
        if not isinstance(payload, dict):
            return payload

        record = dict(payload)
        record.setdefault("backend", "server")
        if kind == "project":
            return cls._with_record_id("prompt_studio_project", record, identifier)
        if kind == "project_stats":
            return cls._with_record_id("prompt_studio_project_stats", record, identifier or record.get("project_id"))
        if kind == "prompt":
            return cls._with_record_id("prompt_studio_prompt", record, identifier)
        if kind == "prompt_version":
            prompt_id = record.get("prompt_id") or parent_id or identifier
            version = record.get("version") or record.get("id")
            if prompt_id is not None and version is not None:
                record.setdefault("record_id", f"server:prompt_studio_prompt_version:{prompt_id}:{version}")
            return record
        if kind == "prompt_preview":
            return cls._with_record_id("prompt_studio_prompt_preview", record, identifier or record.get("project_id"))
        if kind == "prompt_conversion":
            return cls._with_record_id("prompt_studio_prompt_conversion", record, identifier or record.get("project_id"))
        if kind == "prompt_execution":
            return cls._with_record_id("prompt_studio_prompt_execution", record, identifier or record.get("prompt_id"))
        if kind == "test_case":
            return cls._with_record_id("prompt_studio_test_case", record, identifier)
        if kind == "test_case_import":
            return cls._with_record_id("prompt_studio_test_case_import", record, identifier or record.get("project_id"))
        if kind == "test_case_template":
            return cls._with_record_id("prompt_studio_test_case_template", record, identifier or "default")
        if kind == "test_case_export":
            return cls._with_record_id("prompt_studio_test_case_export", record, identifier or record.get("project_id"))
        if kind == "test_case_run":
            return cls._with_record_id("prompt_studio_test_case_run", record, identifier or record.get("prompt_id"))
        if kind == "evaluation":
            return cls._with_record_id("prompt_studio_evaluation", record, identifier)
        if kind == "optimization":
            return cls._with_record_id("prompt_studio_optimization", record, identifier)
        if kind == "optimization_job":
            return cls._with_record_id("prompt_studio_optimization_job", record, identifier)
        if kind == "optimization_strategies":
            return cls._with_record_id("prompt_studio_optimization_strategies", record, identifier or "catalog")
        if kind == "optimization_iteration":
            optimization_id = record.get("optimization_id") or parent_id or identifier
            iteration_id = record.get("id") or record.get("iteration_number")
            if optimization_id is not None and iteration_id is not None:
                record.setdefault("record_id", f"server:prompt_studio_optimization_iteration:{optimization_id}:{iteration_id}")
            return record
        if kind == "optimization_comparison":
            return cls._with_record_id("prompt_studio_optimization_comparison", record, identifier or record.get("prompt_id"))
        if kind == "status":
            return cls._with_record_id("prompt_studio_status", record, "queue")
        return record

    @classmethod
    def _normalize_response(
        cls,
        payload: Any,
        *,
        kind: str | None = None,
        identifier: Any | None = None,
        parent_id: Any | None = None,
    ) -> Any:
        payload = cls._dump(payload)
        if isinstance(payload, list):
            return [
                cls._normalize_response(item, kind=kind, identifier=identifier, parent_id=parent_id)
                for item in payload
            ]
        if not isinstance(payload, dict):
            return payload

        if "evaluations" in payload and isinstance(payload["evaluations"], list):
            record = dict(payload)
            record.setdefault("backend", "server")
            record["evaluations"] = [
                cls._normalize_record(item, kind="evaluation") if isinstance(item, dict) else item
                for item in record["evaluations"]
            ]
            return record

        if "success" in payload and "data" in payload:
            data = payload.get("data")
            if isinstance(data, list):
                record = dict(payload)
                record.setdefault("backend", "server")
                if all(isinstance(item, dict) for item in data) and kind != "optimization_strategies":
                    record["data"] = [
                        cls._normalize_record(item, kind=kind, identifier=identifier, parent_id=parent_id)
                        for item in data
                    ]
                    return record
                return cls._normalize_record(record, kind=kind, identifier=identifier, parent_id=parent_id)
            if isinstance(data, dict):
                return cls._normalize_record(data, kind=kind, identifier=identifier, parent_id=parent_id)
            return cls._normalize_record(payload, kind=kind, identifier=identifier, parent_id=parent_id)

        return cls._normalize_record(payload, kind=kind, identifier=identifier, parent_id=parent_id)

    async def create_project(self, request_data: PromptStudioProjectCreate | dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.create.server")
        request = self._model(request_data, PromptStudioProjectCreate)
        return self._normalize_response(
            await self._require_client().create_prompt_studio_project(request, idempotency_key=idempotency_key),
            kind="project",
        )

    async def list_projects(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.list.server")
        return self._normalize_response(await self._require_client().list_prompt_studio_projects(**kwargs), kind="project")

    async def get_project(self, project_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.detail.server")
        return self._normalize_response(await self._require_client().get_prompt_studio_project(project_id), kind="project")

    async def update_project(self, project_id: int, request_data: PromptStudioProjectUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.update.server")
        request = self._model(request_data, PromptStudioProjectUpdate)
        return self._normalize_response(await self._require_client().update_prompt_studio_project(project_id, request), kind="project", identifier=project_id)

    async def delete_project(self, project_id: int, *, permanent: bool = False) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.delete.server")
        return self._normalize_response(
            await self._require_client().delete_prompt_studio_project(project_id, permanent=permanent),
            identifier=project_id,
        )

    async def archive_project(self, project_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.archive.server")
        return self._normalize_response(await self._require_client().archive_prompt_studio_project(project_id), kind="project", identifier=project_id)

    async def unarchive_project(self, project_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.projects.restore.server")
        return self._normalize_response(await self._require_client().unarchive_prompt_studio_project(project_id), kind="project", identifier=project_id)

    async def get_project_stats(self, project_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.project_stats.detail.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_project_stats(project_id),
            kind="project_stats",
            identifier=project_id,
        )

    async def create_prompt(self, request_data: PromptStudioPromptCreate | dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.create.server")
        request = self._model(request_data, PromptStudioPromptCreate)
        return self._normalize_response(
            await self._require_client().create_prompt_studio_prompt(request, idempotency_key=idempotency_key),
            kind="prompt",
        )

    async def list_prompts(self, project_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.list.server")
        return self._normalize_response(await self._require_client().list_prompt_studio_prompts(project_id, **kwargs), kind="prompt")

    async def get_prompt(self, prompt_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.detail.server")
        return self._normalize_response(await self._require_client().get_prompt_studio_prompt(prompt_id), kind="prompt", identifier=prompt_id)

    async def update_prompt(self, prompt_id: int, request_data: PromptStudioPromptUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.update.server")
        request = self._model(request_data, PromptStudioPromptUpdate)
        return self._normalize_response(await self._require_client().update_prompt_studio_prompt(prompt_id, request), kind="prompt", identifier=prompt_id)

    async def get_prompt_history(self, prompt_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.prompt_versions.list.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_prompt_history(prompt_id),
            kind="prompt_version",
            parent_id=prompt_id,
        )

    async def revert_prompt(self, prompt_id: int, *, version: int) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.restore.server")
        return self._normalize_response(
            await self._require_client().revert_prompt_studio_prompt(prompt_id, version),
            kind="prompt",
            identifier=prompt_id,
        )

    async def preview_prompt(self, request_data: PromptStudioPromptPreviewRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.preview.server")
        request = self._model(request_data, PromptStudioPromptPreviewRequest)
        return self._normalize_response(
            await self._require_client().preview_prompt_studio_prompt(request),
            kind="prompt_preview",
            identifier=request.project_id,
        )

    async def convert_prompt(self, request_data: PromptStudioPromptConvertRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.process.server")
        request = self._model(request_data, PromptStudioPromptConvertRequest)
        return self._normalize_response(
            await self._require_client().convert_prompt_studio_prompt(request),
            kind="prompt_conversion",
            identifier=request.project_id,
        )

    async def execute_prompt(self, request_data: PromptStudioPromptExecuteRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.prompts.launch.server")
        request = self._model(request_data, PromptStudioPromptExecuteRequest)
        return self._normalize_response(
            await self._require_client().execute_prompt_studio_prompt(request),
            kind="prompt_execution",
            identifier=request.prompt_id,
        )

    async def create_test_case(self, request_data: PromptStudioTestCaseCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.create.server")
        request = self._model(request_data, PromptStudioTestCaseCreate)
        return self._normalize_response(await self._require_client().create_prompt_studio_test_case(request), kind="test_case")

    async def create_test_cases_bulk(self, request_data: PromptStudioTestCaseBulkCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.create.server")
        request = self._model(request_data, PromptStudioTestCaseBulkCreate)
        return self._normalize_response(await self._require_client().create_prompt_studio_test_cases_bulk(request), kind="test_case")

    async def list_test_cases(self, project_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.list.server")
        return self._normalize_response(await self._require_client().list_prompt_studio_test_cases(project_id, **kwargs), kind="test_case")

    async def get_test_case(self, test_case_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.detail.server")
        return self._normalize_response(await self._require_client().get_prompt_studio_test_case(test_case_id), kind="test_case", identifier=test_case_id)

    async def update_test_case(self, test_case_id: int, request_data: PromptStudioTestCaseUpdate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.update.server")
        request = self._model(request_data, PromptStudioTestCaseUpdate)
        return self._normalize_response(
            await self._require_client().update_prompt_studio_test_case(test_case_id, request),
            kind="test_case",
            identifier=test_case_id,
        )

    async def delete_test_case(self, test_case_id: int, *, permanent: bool = False) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.delete.server")
        return self._normalize_response(
            await self._require_client().delete_prompt_studio_test_case(test_case_id, permanent=permanent),
            identifier=test_case_id,
        )

    async def import_test_cases(self, request_data: PromptStudioTestCaseImportRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.import.server")
        request = self._model(request_data, PromptStudioTestCaseImportRequest)
        return self._normalize_response(
            await self._require_client().import_prompt_studio_test_cases(request),
            kind="test_case_import",
            identifier=request.project_id,
        )

    async def import_test_cases_csv_upload(
        self,
        project_id: int,
        csv_content: str | bytes,
        *,
        filename: str = "prompt_studio_test_cases.csv",
        signature_id: int | None = None,
        auto_generate_names: bool = True,
    ) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.import.server")
        return self._normalize_response(
            await self._require_client().import_prompt_studio_test_cases_csv_upload(
                project_id=project_id,
                csv_content=csv_content,
                filename=filename,
                signature_id=signature_id,
                auto_generate_names=auto_generate_names,
            ),
            kind="test_case_import",
            identifier=project_id,
        )

    async def get_test_cases_csv_template(self, *, signature_id: int | None = None) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.export.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_test_cases_csv_template(signature_id=signature_id),
            kind="test_case_template",
            identifier=signature_id,
        )

    async def export_test_cases(self, project_id: int, request_data: PromptStudioTestCaseExportRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.export.server")
        request = self._model(request_data, PromptStudioTestCaseExportRequest)
        return self._normalize_response(
            await self._require_client().export_prompt_studio_test_cases(project_id, request),
            kind="test_case_export",
            identifier=project_id,
        )

    async def generate_test_cases(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.launch.server")
        request_data = kwargs.pop("request_data", None)
        if request_data is not None and not isinstance(request_data, PromptStudioTestCaseGenerateRequest):
            request_data = self._model(request_data, PromptStudioTestCaseGenerateRequest)
        if request_data is None:
            result = await self._require_client().generate_prompt_studio_test_cases(**kwargs)
        else:
            result = await self._require_client().generate_prompt_studio_test_cases(request_data, **kwargs)
        return self._normalize_response(
            result,
            kind="test_case",
        )

    async def run_test_cases(self, request_data: PromptStudioRunTestCasesRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.test_cases.launch.server")
        request = self._model(request_data, PromptStudioRunTestCasesRequest)
        return self._normalize_response(
            await self._require_client().run_prompt_studio_test_cases(request),
            kind="test_case_run",
            identifier=request.prompt_id,
        )

    async def create_evaluation(self, request_data: PromptStudioEvaluationCreate | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.evaluations.create.server")
        request = self._model(request_data, PromptStudioEvaluationCreate)
        return self._normalize_response(await self._require_client().create_prompt_studio_evaluation(request), kind="evaluation")

    async def list_evaluations(self, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.evaluations.list.server")
        return self._normalize_response(await self._require_client().list_prompt_studio_evaluations(**kwargs), kind="evaluation")

    async def get_evaluation(self, evaluation_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.evaluations.detail.server")
        return self._normalize_response(await self._require_client().get_prompt_studio_evaluation(evaluation_id), kind="evaluation", identifier=evaluation_id)

    async def delete_evaluation(self, evaluation_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.evaluations.delete.server")
        return self._normalize_response(await self._require_client().delete_prompt_studio_evaluation(evaluation_id), identifier=evaluation_id)

    async def create_optimization(self, request_data: PromptStudioOptimizationCreate | dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.create.server")
        request = self._model(request_data, PromptStudioOptimizationCreate)
        return self._normalize_response(
            await self._require_client().create_prompt_studio_optimization(request, idempotency_key=idempotency_key),
            kind="optimization",
        )

    async def create_optimization_simple(self, request_data: PromptStudioOptimizationSimpleCreateRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.launch.server")
        request = self._model(request_data, PromptStudioOptimizationSimpleCreateRequest)
        return self._normalize_response(await self._require_client().create_prompt_studio_optimization_simple(request), kind="optimization_job")

    async def list_optimizations(self, project_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.list.server")
        return self._normalize_response(await self._require_client().list_prompt_studio_optimizations(project_id, **kwargs), kind="optimization")

    async def get_optimization(self, optimization_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.detail.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_optimization(optimization_id),
            kind="optimization",
            identifier=optimization_id,
        )

    async def get_optimization_job_status(self, job_id: str) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.detail.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_optimization_job_status(job_id),
            kind="optimization_job",
            identifier=job_id,
        )

    async def cancel_optimization(self, optimization_id: int, *, reason: str | None = None) -> dict[str, Any]:
        self._enforce("prompt_studio.optimizations.cancel.server")
        return self._normalize_response(
            await self._require_client().cancel_prompt_studio_optimization(optimization_id, reason=reason),
            kind="optimization",
            identifier=optimization_id,
        )

    async def get_optimization_strategies(self) -> dict[str, Any]:
        self._enforce("prompt_studio.optimization_strategies.list.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_optimization_strategies(),
            kind="optimization_strategies",
            identifier="catalog",
        )

    async def get_optimization_history(self, optimization_id: int) -> dict[str, Any]:
        self._enforce("prompt_studio.optimization_iterations.list.server")
        return self._normalize_response(
            await self._require_client().get_prompt_studio_optimization_history(optimization_id),
            kind="optimization_iteration",
            parent_id=optimization_id,
        )

    async def add_optimization_iteration(
        self,
        optimization_id: int,
        request_data: PromptStudioOptimizationIterationCreate | dict[str, Any],
    ) -> dict[str, Any]:
        self._enforce("prompt_studio.optimization_iterations.create.server")
        request = self._model(request_data, PromptStudioOptimizationIterationCreate)
        return self._normalize_response(
            await self._require_client().add_prompt_studio_optimization_iteration(optimization_id, request),
            kind="optimization_iteration",
            parent_id=optimization_id,
        )

    async def list_optimization_iterations(self, optimization_id: int, **kwargs: Any) -> dict[str, Any]:
        self._enforce("prompt_studio.optimization_iterations.list.server")
        return self._normalize_response(
            await self._require_client().list_prompt_studio_optimization_iterations(optimization_id, **kwargs),
            kind="optimization_iteration",
            parent_id=optimization_id,
        )

    async def compare_optimization_strategies(self, request_data: PromptStudioCompareStrategiesRequest | dict[str, Any]) -> dict[str, Any]:
        self._enforce("prompt_studio.optimization_strategies.launch.server")
        request = self._model(request_data, PromptStudioCompareStrategiesRequest)
        return self._normalize_response(
            await self._require_client().compare_prompt_studio_optimization_strategies(request),
            kind="optimization_comparison",
            identifier=request.prompt_id,
        )

    async def get_status(self, *, warn_seconds: int = 30) -> dict[str, Any]:
        self._enforce("prompt_studio.status.detail.server")
        return self._normalize_response(await self._require_client().get_prompt_studio_status(warn_seconds=warn_seconds), kind="status")

    async def stream_events(
        self,
        *,
        client_id: str,
        project_id: int | None = None,
    ) -> AsyncGenerator[dict[str, Any], None]:
        self._enforce("prompt_studio.events.observe.server")
        async for event in self._require_client().stream_prompt_studio_events(client_id=client_id, project_id=project_id):
            payload = self._dump(event)
            if isinstance(payload, dict):
                payload.setdefault("backend", "server")
            yield payload
