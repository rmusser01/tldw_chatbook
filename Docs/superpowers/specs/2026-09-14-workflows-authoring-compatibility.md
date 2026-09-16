# Workflows definition compatibility — authoring-only port

Recorded 2026-09-14 for TASK-32601. This is source-inspection evidence, not a
server round-trip or runtime conformance result.

## Reference

- Chatbook base: `77eb2601a63ba473318b8ec1e4edb53f8ac5899e`.
- Reused editor/document checkpoint: `b34eda3d64`.
- Server fetched `origin/dev`: `2e1a5e58d3344a1efd578efb4dbfb1c9465e8767`.
- Previous server reference: `6cd2745f696af04668a61c20b84ab8a9e69ca5e4`.

The following read-only comparison returned no changed files:

```sh
git -C /Users/macbook-dev/Documents/GitHub/tldw_server diff --stat \
  6cd2745f696af04668a61c20b84ab8a9e69ca5e4..2e1a5e58d3344a1efd578efb4dbfb1c9465e8767 -- \
  tldw_Server_API/app/api/v1/schemas/workflows.py \
  tldw_Server_API/app/api/v1/endpoints/workflows.py \
  tldw_Server_API/app/core/Workflows
```

Inspected primary sources: [API schemas](https://github.com/rmusser01/tldw_server/blob/2e1a5e58d3344a1efd578efb4dbfb1c9465e8767/tldw_Server_API/app/api/v1/schemas/workflows.py),
[definition endpoints](https://github.com/rmusser01/tldw_server/blob/2e1a5e58d3344a1efd578efb4dbfb1c9465e8767/tldw_Server_API/app/api/v1/endpoints/workflows.py),
and [step registry](https://github.com/rmusser01/tldw_server/blob/2e1a5e58d3344a1efd578efb4dbfb1c9465e8767/tldw_Server_API/app/core/Workflows/registry.py).
No server process or model was contacted.

An AST comparison of the pinned server's StepType registry and the reused
catalog's DISCOVERY literal found 130 names in each, no duplicates, no missing
names and no source-only names. This checks inventory spelling, not adapter
behavior or availability.

## Contract to retain

| Definition content | Server contract | Authoring boundary |
| --- | --- | --- |
| Definition envelope | name, version, description, tags, inputs, steps, metadata, visibility, on_completion_webhook | Preserve saved JSON values; a valid local draft is not certification that the server will accept it. |
| Step envelope | id, name, type, config, retry, timeout_seconds, on_success, on_failure, on_timeout | Stable IDs and opaque config retained. Only supported sequential forms are offered in this slice. |
| Portable identity | Free-form metadata object | Chatbook uses metadata.tldw_workflow for workflow/revision UUIDs and revision ancestry; these are not the server's database row IDs. |
| Branch and parallel configuration | Registry includes branch, map and parallel | Preserve imported documents without flattening them. Branch authoring is v2; map/parallel authoring is v3. |
| Unknown data | config and metadata are open dictionaries; API envelopes use default Pydantic extra handling | Local preservation is broader than server envelope preservation. Never promise that unknown root/step fields survive a server publish. |
| Run data and secrets | RunRequest is a different API model | No run IDs, grants, credentials, local bindings or draft/view state are added to a definition export. User-authored opaque content still requires review before sharing. |

The reviewed form catalog offers subsets for media_ingest, prompt, llm,
wait_for_human and notes. Its larger inventory is discovery information,
not evidence of installed or executable adapters. All new execution stays
unavailable in this delivery.

## What this delivery does not establish

Local import/edit/save/export tests can establish local preservation of stable
step IDs, opaque metadata and non-edited values. They cannot establish server
acceptance, complete per-step configuration validation, execution behavior,
capability negotiation, sync conflict policy, or cross-server sharing.

In particular, the server's create/version endpoints validate a
WorkflowDefinitionCreate and then use model_dump(). Unknown envelope fields may
be discarded by that model even though the local document service retains them.
A future publish/sync slice must detect and explain such loss before transmission;
it must not silently normalize an authored document into a narrower schema.

The current UI must not advertise server publish or synchronization as working.
Future local inference verification, when explicitly in scope, uses the supplied
llama.cpp endpoint at localhost:9099.
