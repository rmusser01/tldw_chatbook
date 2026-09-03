# Server Reminder API Fixtures

This directory holds representative server responses for the existing reminder-task endpoints consumed by the scheduling module.

## Pydantic models

All models are imported from `tldw_chatbook.tldw_api.notifications_reminders_schemas`.

### Request / response models by client method

| Client method (`TLDWAPIClient`) | HTTP | Path | Request model | Response model |
|--------------------------------|------|------|---------------|----------------|
| `create_reminder_task` | POST | `/api/v1/tasks` | `ReminderTaskCreateRequest` | `ReminderTaskResponse` |
| `list_reminder_tasks` | GET | `/api/v1/tasks` | — | `ReminderTaskListResponse` |
| `get_reminder_task` | GET | `/api/v1/tasks/{task_id}` | — | `ReminderTaskResponse` |
| `update_reminder_task` | PATCH | `/api/v1/tasks/{task_id}` | `ReminderTaskUpdateRequest` | `ReminderTaskResponse` |
| `delete_reminder_task` | DELETE | `/api/v1/tasks/{task_id}` | — | `ReminderTaskDeleteResponse` |

### `ReminderTaskCreateRequest`

- `title`: `str` (required, 1–200 chars)
- `body`: `str | None`
- `schedule_kind`: `"one_time" | "recurring"` (required)
- `run_at`: `str | None` (required when `schedule_kind == "one_time"`)
- `cron`: `str | None` (required when `schedule_kind == "recurring"`)
- `timezone`: `str | None` (required when `schedule_kind == "recurring"`)
- `link_type`: `str | None`
- `link_id`: `str | None`
- `link_url`: `str | None`
- `enabled`: `bool` (default `True`)

### `ReminderTaskUpdateRequest`

Same fields as the create request, but all optional and with `extra="forbid"`.

### `ReminderTaskResponse`

- `id`: `str`
- `user_id`: `str`
- `tenant_id`: `str`
- `title`: `str`
- `body`: `str | None`
- `link_type`: `str | None`
- `link_id`: `str | None`
- `link_url`: `str | None`
- `schedule_kind`: `"one_time" | "recurring"`
- `run_at`: `str | None`
- `cron`: `str | None`
- `timezone`: `str | None`
- `enabled`: `bool`
- `last_run_at`: `str | None`
- `next_run_at`: `str | None`
- `last_status`: `str | None`
- `created_at`: `str`
- `updated_at`: `str`

### `ReminderTaskListResponse`

- `items`: `list[ReminderTaskResponse]`
- `total`: `int`

### `ReminderTaskDeleteResponse`

- `deleted`: `bool`

## Service wrapper methods

In `tldw_chatbook.Notifications.server_notifications_service.ServerNotificationsService` the exact method names are:

- `create_reminder`
- `list_reminders`
- `get_reminder`
- `update_reminder`
- `delete_reminder`

These wrap the `TLDWAPIClient` methods listed above and enforce the corresponding permissions:

- `notifications.reminders.configure.server` → `create_reminder`, `update_reminder`, `delete_reminder`
- `notifications.reminders.list.server` → `list_reminders`, `get_reminder`

## Search for other scheduling-related endpoints

A search of `tldw_chatbook/tldw_api/client.py` for the strings `scheduled-tasks`, `automation`, and `definition` returned **no matches**. The only server-side task endpoints currently available are the `/api/v1/tasks` reminder endpoints documented above.

## Fixture scope

Only the `list_reminder_tasks` response fixture (`reminder_list.json`) is present at this commit. Future fixtures for create/get/update/delete responses should follow the same format and be validated against the models documented above.

## Drift guard: `owner_id` is the server's RAW user id, not a `server:` scope

`automation_definition_list.json` and `automation_results_list.json` carry
`"owner_id": "1"`. That is the shape a real tldw_server actually puts on the
wire — live-verified 2026-09-02 against `tldw_server2` @ `25fb0eca59`
(single-user, `GET /api/v1/scheduled-tasks/definitions` returned
`{"id": "705657ff…", "owner_id": "1", …}`).

These fixtures previously said `"owner_id": "server:42"`, which is the
**client's** owner-scope convention (`is_server_scoped_owner`, `server:<active
server id>`) and not something a server ever sends. That drift masked a HIGH
defect for a whole slice: `_load_server_automations` only stamped the scoped
owner when the payload omitted one, so against the real server every server
automation was treated as local — wrong `[This device]` label, run-now routed
to the local executor, and "move to this device" refused with "This automation
no longer exists" (task 6 report, D1).

**Do not "fix" these back to a prefixed value.** A server-sourced row's owner
scope is derived from the CONNECTION at the ingestion boundary and overwritten
there; the payload's own `owner_id` is never trusted. `"server:…"` values in
test code are correct where they stand for a LOCAL row's owner column — the
distinction is wire payload vs. stored row.
