# tldw_server Automation-Definition API Contract

> Audited from `tldw_server2/tldw_Server_API/app/api/v1/endpoints/scheduled_tasks_control_plane.py` and `tldw_server2/tldw_Server_API/app/api/v1/schemas/scheduled_tasks_automation_schemas.py`.
> All routes are mounted under `/api/v1/scheduled-tasks`.

## Capabilities

| Method | Path | Auth / Rate-limit scope | Response schema |
|--------|------|------------------------|-----------------|
| GET | `/api/v1/scheduled-tasks/capabilities` | `tasks.read` | `ScheduledTaskAutomationCapabilitiesResponse` |

Returns availability and action capabilities for each automation family (`recurring_question`, `agent_task`).

## Previews

| Method | Path | Auth / Rate-limit scope | Request | Response |
|--------|------|------------------------|---------|----------|
| GET | `/api/v1/scheduled-tasks/previews` | `tasks.read` | Query: `limit`, `offset`, `family`, `mode`, `status`, `definition_id`, `expired` | `ScheduledTaskPreviewListResponse` |
| POST | `/api/v1/scheduled-tasks/previews` | `tasks.control` | `ScheduledTaskPreviewCreateRequest` | `ScheduledTaskPreviewResponse` (201) |
| GET | `/api/v1/scheduled-tasks/previews/{preview_id}` | `tasks.read` | — | `ScheduledTaskPreviewResponse` |

## Definitions

| Method | Path | Auth / Rate-limit scope | Request | Response |
|--------|------|------------------------|---------|----------|
| GET | `/api/v1/scheduled-tasks/definitions` | `tasks.read` | Query: `limit`, `offset`, `family`, `lifecycle`, `health`, `visibility_policy`, `q`, `created_from`, `created_to` | `ScheduledTaskDefinitionListResponse` |
| POST | `/api/v1/scheduled-tasks/definitions` | `tasks.control` | `ScheduledTaskDefinitionCreateRequest` | `ScheduledTaskDefinitionResponse` (201) |
| GET | `/api/v1/scheduled-tasks/definitions/{definition_id}` | `tasks.read` | — | `ScheduledTaskDefinitionResponse` |
| PATCH | `/api/v1/scheduled-tasks/definitions/{definition_id}` | `tasks.control` | `ScheduledTaskDefinitionUpdateRequest` | `ScheduledTaskDefinitionResponse` |
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/pause` | `tasks.control` | — | `ScheduledTaskDefinitionResponse` |
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/resume` | `tasks.control` | — | `ScheduledTaskDefinitionResponse` |
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/archive` | `tasks.control` | — | `ScheduledTaskDefinitionResponse` |
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/duplicate` | `tasks.control` | `ScheduledTaskDuplicateRequest` | `ScheduledTaskDefinitionResponse` |
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/run` | `tasks.control` | — (optional `Idempotency-Key` header) | `ScheduledTaskRunNowResponse` |

The `/run` route (server TASK-13022 / client ADR-077) triggers one
immediate execution through the standard Jobs path — the same pipeline the
scheduler feed enqueues into, with the same run-slot idempotency
semantics. Lifecycle refusals reuse the error codes below; the response
carries the created run reference (`definition_id`, `run_slot_utc`,
`job_id`, `deduped`) for correlating with the eventual result
notification.

## Runs & results (recurring-question service)

| Method | Path | Auth / Rate-limit scope | Request | Response |
|--------|------|------------------------|---------|----------|
| POST | `/api/v1/scheduled-tasks/definitions/{definition_id}/runs` | `tasks.control` | — (optional `Idempotency-Key` header) | `ScheduledTaskRunResponse` (201) |
| GET | `/api/v1/scheduled-tasks/definitions/{definition_id}/runs` | `tasks.read` | Query: `limit` (≤200), `offset`, `status` | `ScheduledTaskRunListResponse` |
| GET | `/api/v1/scheduled-tasks/runs/{run_id}` | `tasks.read` | — | `ScheduledTaskRunResponse` |
| GET | `/api/v1/scheduled-tasks/results` | `tasks.read` | Query: `limit`, `offset`, `definition_id`, `run_id`, `review_state`, `kind` | `ScheduledTaskResultListResponse` |
| GET | `/api/v1/scheduled-tasks/results/{result_id}` | `tasks.read` | — | `ScheduledTaskResultResponse` |
| POST | `/api/v1/scheduled-tasks/results/{result_id}/review` | `tasks.control` | `ScheduledTaskResultReviewRequest` | `ScheduledTaskResultResponse` |

Two distinct manual-run routes exist: `POST …/runs` queues a
recurring-question run through `ScheduledTaskRecurringQuestionService`,
while `POST …/run` dispatches any automation family through
`ScheduledTaskAutomationService.run_now`.

## Audit events

| Method | Path | Auth / Rate-limit scope | Query / Request | Response |
|--------|------|------------------------|-----------------|----------|
| GET | `/api/v1/scheduled-tasks/definitions/{definition_id}/audit` | `tasks.read` | `limit`, `offset`, `event_type`, `actor`, `created_from`, `created_to`, `idempotency_key`, `request_id` | `ScheduledTaskAuditListResponse` |

## Literals

- `ScheduledTaskAutomationFamily`: `"recurring_question"` | `"agent_task"`
- `ScheduledTaskPreviewMode`: `"create"` | `"update"`
- `ScheduledTaskPreviewStatus`: `"valid"` | `"invalid"` | `"expired"` | `"consumed"`
- `ScheduledTaskDefinitionLifecycle`: `"configured"` | `"paused"` | `"archived"` | `"disabled"`
- `ScheduledTaskDefinitionCreateLifecycle`: `"configured"` | `"paused"`
- `ScheduledTaskDefinitionHealth`: `"ready"` | `"execution_unavailable"` | `"capability_unavailable"` | `"needs_attention"` | `"permission_required"`
- `ScheduledTaskDefinitionDisabledLockKind`: `"none"` | `"admin"` | `"security"` | `"system"`
- `ScheduledTaskAutomationActionStatus`: `"available"` | `"unavailable"` | `"planned"` | `"disabled"`
- `ScheduledTaskAutomationFamilyAvailability`: `"available"` | `"planned"` | `"unavailable"` | `"degraded"`

## Request schemas

### `ScheduledTaskPreviewCreateRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `mode` | `"create"` \| `"update"` | no | default `"create"` |
| `family` | `ScheduledTaskAutomationFamily` | yes | |
| `definition_id` | `str` | no | required for `mode="update"` |
| `definition_version` | `int >= 1` | no | |
| `name` | `str` (1-255) | no | |
| `description` | `str` | no | |
| `config` | `dict[str, Any]` | no | default `{}` |
| `input` | `dict[str, Any]` | no | default `{}` |
| `schedule` | `dict[str, Any]` | no | default `{}` |
| `visibility_policy` | `dict[str, Any]` | no | default `{}` |
| `notification_policy` | `dict[str, Any]` | no | default `{}` |
| `approval_policy` | `dict[str, Any]` | no | default `{}` |

### `ScheduledTaskDefinitionCreateRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `preview_id` | `str` | yes | references a valid preview |
| `initial_lifecycle` | `"configured"` \| `"paused"` | no | default `"configured"` |

### `ScheduledTaskDefinitionUpdateRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `preview_id` | `str` | yes | references a valid update preview |

### `ScheduledTaskDuplicateRequest`

| Field | Type | Required | Notes |
|-------|------|----------|-------|
| `name` | `str` (1-255) | no | override name for the duplicate |
| `description` | `str` | no | override description |

## Response schemas

### `ScheduledTaskPreviewResponse`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | |
| `owner_id` | `str` | nullable |
| `mode` | `ScheduledTaskPreviewMode` | |
| `family` | `ScheduledTaskAutomationFamily` | |
| `definition_id` | `str` | nullable |
| `definition_version` | `int >= 1` | nullable |
| `status` | `ScheduledTaskPreviewStatus` | |
| `payload_hash` | `str` | nullable |
| `normalized_config` | `dict[str, Any]` | |
| `validation_errors` | `list[dict[str, Any]]` | |
| `warnings` | `list[dict[str, Any]]` | |
| `risk_class` | `str` | nullable |
| `visibility_policy` | `dict[str, Any]` | |
| `schedule_preview` | `dict[str, Any]` | |
| `redaction_policy` | `dict[str, Any]` | |
| `expires_at` | `datetime` | nullable |
| `created_by` | `str` | nullable |
| `created_at` | `datetime` | nullable |
| `consumed_at` | `datetime` | nullable |
| `created_definition_id` | `str` | nullable |

### `ScheduledTaskDefinitionResponse`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | |
| `owner_id` | `str` | nullable |
| `version` | `int >= 1` | default 1 |
| `family` | `ScheduledTaskAutomationFamily` | |
| `name` | `str` | |
| `description` | `str` | nullable |
| `lifecycle` | `ScheduledTaskDefinitionLifecycle` | |
| `health` | `ScheduledTaskDefinitionHealth` | |
| `disabled_lock_kind` | `ScheduledTaskDefinitionDisabledLockKind` | nullable |
| `disabled_reason` | `str` | nullable |
| `schedule` | `dict[str, Any]` | |
| `input` | `dict[str, Any]` | |
| `config` | `dict[str, Any]` | |
| `visibility_policy` | `dict[str, Any]` | |
| `notification_policy` | `dict[str, Any]` | |
| `approval_policy` | `dict[str, Any]` | |
| `preview_id` | `str` | nullable |
| `created_by` | `str` | nullable |
| `updated_by` | `str` | nullable |
| `created_at` | `datetime` | nullable |
| `updated_at` | `datetime` | nullable |
| `archived_at` | `datetime` | nullable |

### `ScheduledTaskAuditEventResponse`

| Field | Type | Notes |
|-------|------|-------|
| `id` | `str` | |
| `definition_id` | `str` | |
| `event_type` | `str` | |
| `actor` | `str` | nullable |
| `summary` | `str` | nullable |
| `before` | `dict[str, Any]` | nullable |
| `after` | `dict[str, Any]` | nullable |
| `created_at` | `datetime` | nullable |
| `request_id` | `str` | nullable |
| `idempotency_key` | `str` | nullable |

### `ScheduledTaskAutomationCapability`

| Field | Type | Notes |
|-------|------|-------|
| `family` | `ScheduledTaskAutomationFamily` | |
| `family_availability` | `ScheduledTaskAutomationFamilyAvailability` | |
| `actions` | `dict[str, ScheduledTaskActionCapability]` | |
| `missing_dependencies` | `list[str]` | |
| `related_capabilities` | `dict[str, Any]` | |
| `reason` | `str` | nullable |
| `schema_version` | `str` | default `"2026-06-09"` |

### `ScheduledTaskActionCapability`

| Field | Type | Notes |
|-------|------|-------|
| `status` | `ScheduledTaskAutomationActionStatus` | |
| `reason` | `str` | nullable |
| `required_permissions` | `list[str]` | |

### List envelope fields

`ScheduledTaskPreviewListResponse`, `ScheduledTaskDefinitionListResponse`, and `ScheduledTaskAuditListResponse` all contain:

| Field | Type | Notes |
|-------|------|-------|
| `items` | `list[<T>]` | |
| `total` | `int >= 0` | default 0 |
| `limit` | `int >= 1` | default 50 |
| `offset` | `int >= 0` | default 0 |
| `has_more` | `bool` | default false |
| `next_offset` | `int >= 0` | nullable |

## Error contract

Errors are returned as FastAPI `HTTPException` detail with the shape:

```json
{
  "code": "scheduled_task_definition_not_found",
  "message": "Scheduled task definition was not found.",
  "details": {"reason": "definition_not_found"},
  "field_errors": [],
  "retryable": false,
  "correlation_id": "<request_id>"
}
```

Common automation error codes mapped by the server:

| Server reason | HTTP status | Client code |
|---------------|-------------|-------------|
| `scheduled_task_family_unavailable` | 409 | `scheduled_task_family_unavailable` |
| `scheduled_task_preview_required` | 400 | `scheduled_task_preview_required` |
| `scheduled_task_definition_not_found` | 404 | `scheduled_task_definition_not_found` |
| `scheduled_task_preview_mismatch` | 409 | `scheduled_task_preview_mismatch` |
| `scheduled_task_preview_expired` | 409 | `scheduled_task_preview_expired` |
| `scheduled_task_schedule_invalid` | 422 | `scheduled_task_schedule_invalid` |
| `scheduled_task_scope_invalid` | 422 | `scheduled_task_scope_invalid` |
| `scheduled_task_agent_ref_invalid` | 422 | `scheduled_task_agent_ref_invalid` |
| `scheduled_task_permission_policy_invalid` | 422 | `scheduled_task_permission_policy_invalid` |
| `scheduled_task_execution_unavailable` | 409 | `scheduled_task_execution_unavailable` |
| `scheduled_task_definition_version_conflict` | 409 | `scheduled_task_definition_version_conflict` |
| `scheduled_task_definition_archived` | 409 | `scheduled_task_definition_archived` |
| `scheduled_task_lifecycle_transition_invalid` | 409 | `scheduled_task_lifecycle_transition_invalid` |
| `scheduled_task_idempotency_conflict` | 409 | `scheduled_task_idempotency_conflict` |

## Notes

- Run and result endpoints under `/api/v1/scheduled-tasks` landed after the
  original audit (`/definitions/{id}/run` in TASK-13022; the `runs`/`results`
  family with the recurring-question service) and are documented above. This
  note replaces an earlier claim that no dedicated run endpoints existed —
  that was true of the audited revision but not of current dev.
- Date-time filters (`created_from`, `created_to`) must include a timezone; the server normalizes to UTC.
- Mutating endpoints accept an optional `Idempotency-Key` request header.
- The `capabilities` endpoint is synchronous (`def`, not `async def`) and returns the current server-side capability matrix.
