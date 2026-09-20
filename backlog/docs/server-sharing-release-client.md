# Connected sharing and Notes release contracts

TASK-32881; [ADR-174](../decisions/174-server-sharing-release-contracts.md).

## Clone and retry

Create one `CloneWorkspaceRequest(name="Copy")` per logical copy. The request owns
an `idempotency_key`; save that key if the workflow must survive app restart.
Reusing the same request with `client.clone_shared_workspace(share_id, request)`
after a timeout or uncertain 503 replays the original admission. Do not construct
a fresh request/key merely because a response was lost. `new_name` remains an
accepted Python input alias; the wire body always uses `name`.

Both `Sharing` and `Sharing_Interop` services accept `idempotency_key=...` together
with the requested name. Service callers should retain and supply that key across
explicit retries; creating a request without a key begins a new logical operation.
The Sharing panel keeps its keys for the app lifetime, including panel remounts.
Keys are scoped to the configured server and authenticated user, so account/server
switches preserve earlier retries and credential refresh does not change identity.
Names use the server's whitespace normalization. The panel retains at most 100
intents; reaching the limit blocks new intents without evicting earlier requests.
Use **Clone / retry** to recover the same receipt and **Start another clone** before
requesting another copy for the same share/name and active account. A process
restart does not retain this panel memory; it is not a durable local operation journal.

A clone response may contain a canonical `operation_id` or an older server's
`job_id`. Canonical `queued`/`running` responses do not mean a copy is complete.
Inspect `result`, its publication/readiness fields and warnings on `succeeded`,
and `error`/cleanup state on `failed`. Read a known canonical receipt through
`get_shared_workspace_clone_operation(share_id, operation_id)`. This constructs
the authenticated local API path; it does not follow a response URL to another host.
Matching replay and receipt reads remain available after share revocation, as the
server's recipient-owned receipt contract requires. New admissions remain subject
to current server authorization. Poll backoff and persistence belong to the caller.

## Shared sources

`list_shared_workspace_source_page(share_id, offset=0, limit=50, q=None, state=None)`
returns `items`, `pagination`, `summary`, and `partial_errors`. Use this method when
you need status or pagination controls. Source rows expose canonical `source_id`
and `origin_url`, with legacy `id` and `url` compatibility accessors/dictionary
fields. Legacy server lists still parse. The existing `list_shared_workspace_sources`
convenience method traverses pages instead of silently truncating to the first
server page; it fails if pagination stops advancing.

## Notes links

Pass the version of the link the user selected:

```python
await notes_scope.delete_note_link(
    scope="server_note",
    edge_id=selected_link["id"],
    dataset_id=selected_dataset_id,
    expected_version=selected_link["version"],
    idempotency_key=delete_request_key,
    reason="User removed link",
)
```

The scope, service, and client forward these preconditions unchanged. They do not
fetch the newest link and replace the selected version. Let 409 conflicts return
to the user for review. A Sync-backed deletion without a version retains the
server's 428 response; older non-Sync servers may still accept an unversioned delete.

Page parameters are validated before transport: offset is nonnegative, limit is
1–200, query text is 1–512 characters when provided, and state is 1–64 characters.
State remains a free-text server filter so new server states work without a client
upgrade. Empty advancing pages are followed; repeated/nonadvancing cursors fail.
The mounted Sharing panel uses the production `sharing_scope_service` wrapper and
passes its share ID/name through shared input validation before retaining a key.
