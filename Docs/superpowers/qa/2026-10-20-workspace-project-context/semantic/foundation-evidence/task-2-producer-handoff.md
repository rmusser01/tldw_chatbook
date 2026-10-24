# Task 2 producer handoff — issued project embedding scope

The project-only issuer is `Workspaces/project_index/semantic.py::_issue_project_projection_scope(service, scope, settings, semantic_revision)`. Persist only `ProjectSemanticSettings` from `_settings_for_project_model(model_id)` after the separate approval action sets `approved_destination_fingerprint` to the exact candidate destination fingerprint. Configured `embedding_config.models[model_id]` takes precedence over the built-in catalog and only that entry is schema validated; current config and credentials are re-resolved on every issuance. A candidate catalog entry never grants authority. A selected configured keyless OpenAI-compatible base URL is valid, while the actual OpenAI endpoint refuses without a key before provider session construction; credentials remain outside identity. Selected Hugging Face `cache_dir` takes precedence over the global cache, and the safe settings fingerprint binds its selected absolute root without storing the path. Unsupported/unsafe endpoints and absent complete local snapshots refuse. Nothing configures, migrates, downloads, or consents implicitly.

Issue inside a *synchronous admitted project operation* on its owning execution. The issuer checks the exact live binding, scope, persisted settings, semantic revision and approval using an existing-only `_store(..., cleanup_only=True)` and `_read_fenced`; those SQL/gate/fence handles close before snapshot closure or model construction. Each later `execution()` rereads persisted semantic state and the current selected catalog/config identity after its finite DB/fence unit closes; a changed endpoint, model or selected cache root refuses before embedding effects. Issuance and compatible rebind bind the exact resolved artifact fingerprint. Project scope adds no duplicate full model-byte hash; existing recovered_encode closure/receipt review remains unchanged. Its retained `registry_connection_scope`, `_factory.retain(key)` pin and `pin.bind()` are the **actual** accepted parent until `ProjectionExecutionScope.close()`. Do not issue in an async outer task then copy its context to a worker. If a tool worker is synchronous, obtain its own fresh admission on that worker. Async embedding/model calls use `wrapper._projection_execution_scope`'s accepted transfer through the installed activation adapter; construction and later work must run in supported native/accepted execution, with child joins before parent close. Never use a copied or expired carrier as authority.

Construct `EmbeddingsServiceWrapper(model_name=carrier.model_name, api_key=carrier.api_key, base_url=carrier.base_url, device=<selected device>, execution_scope=carrier, local_files_only=carrier.local_files_only)` before any Library source inference. Then `create_embedding(s)` or `acreate_embedding(s)` uses project admission. Close the wrapper's native/model work before `carrier.close()` and operation release. The wrapper/factory may be cached **without** an idle project pin. For a later independent operation, issue a new carrier and call `wrapper.rebind_execution_scope(successor: ProjectionExecutionScope) -> None` only after the previous carrier is positively closed and child work retired. It verifies an actual live accepted current execution, same service/key and compatible settings, model, artifact, endpoint, credentials and offline target; valid later revision/epoch may differ. The replacement is serialized under a wrapper lock. New credentials require a fresh wrapper/client while destination identity remains credential-free. Library/default wrappers cannot be upgraded. Wrong or invalid successors refuse without Library fallback. A later query thus needs a fresh admission, not an idle lease or copied carrier revival.

The actual offline factory path loads a complete pre-existing Hugging Face snapshot with `local_files_only=True`, disabled dynamic fallback and no network. `RAG_Search/model_recovery.py` now imports the existing `Utils.platform_files.os` adapter (ADR-126), matching native_files so Windows `listdir(fd)` enumerates the pinned native handle. No path fallback or closure/alias/manifest/identity gate changed. The qualified cached MiniLM composition test loaded once via the production project wrapper/factory, embedded synthetic text in two independent admitted operations, and reused the same factory model record with HTTP network blocked. Model quality, actual UI integration, and 20k-file scale are later tasks.

### Finite caller sequence

An async build owns one task on an off-UI worker loop. The issuer, constructor, pages and final close all run in that same task; the test-backed transfer is **inside** `create_embeddings_async`, not a copied `asyncio.to_thread(wrapper.create_embeddings, ...)` call:

```python
async def admitted_build(service, scope, settings, revision, pages):
    carrier = _issue_project_projection_scope(service, scope, settings, revision)
    wrapper = None
    try:
        wrapper = EmbeddingsServiceWrapper(
            model_name=carrier.model_name,
            api_key=carrier.api_key,
            base_url=carrier.base_url,
            execution_scope=carrier,
            local_files_only=carrier.local_files_only,
        )
        for page in pages:
            vectors = await wrapper.create_embeddings_async(page)
            # Apply this page through the separately fenced Task 3 publication unit.
    finally:
        if wrapper is not None:
            wrapper.close()
        carrier.close()
```

The constructor's installed `activation.guarded` entrance validates the carrier before source/model effects in the issuer task. Each async page enters `activation.async_guarded`, which makes an accepted child; `_async_factory_embed` uses `activation.native_worker` to transfer that accepted child to the actual model thread and `join_retained_task` to await its physical retirement despite repeated caller cancellation. Only after that join may the `finally` close wrapper/model and carrier. If an async build wants a reusable wrapper, finish every in-flight page and close **the carrier**, keep only idle wrapper/model state, then rebind it under the next independent admission; close the wrapper under the final live carrier. Scheduling the one finite coroutine on an off-UI worker is the Task 3 owner's choice. Do not start a new event loop per page/tool, call the synchronous `create_embeddings` within a running loop, or copy an issuer's context to a different task/thread.

A synchronous query tool worker has its own accepted execution and issues a fresh carrier **on that worker**. It may reuse an idle compatible wrapper:

```python
def embed_query_on_admitted_worker(service, scope, settings, revision, wrapper, text):
    carrier = _issue_project_projection_scope(service, scope, settings, revision)
    try:
        if wrapper is None:
            wrapper = EmbeddingsServiceWrapper(
                model_name=carrier.model_name,
                api_key=carrier.api_key,
                base_url=carrier.base_url,
                execution_scope=carrier,
                local_files_only=carrier.local_files_only,
            )
        else:
            wrapper.rebind_execution_scope(carrier)
        vector = wrapper.create_embedding(text)
        return wrapper, vector
    finally:
        # All synchronous/native embedding work has returned before this point.
        carrier.close()
```

At cache/model retirement, close the wrapper while the last carrier is still live, then close that carrier. If the wrapper was kept idle after its last carrier closed, issue a fresh compatible carrier and rebind it before final wrapper close. Calling `wrapper.close()` with only a closed carrier refuses before touching the factory, so no unowned close attempt occurs. A query that keeps the wrapper idle closes only its carrier. A different credential, artifact, settings or endpoint creates a new wrapper. The query test uses this fresh-issuer/rebind pattern twice and checks the same production factory model record; the async scope test proves accepted child/native cancellation and copied-context refusal. Real cached MiniLM composition currently exercises the synchronous query form. The real-async model path and Task 3 publication/quality/scale are not claimed by Task 2 receipts.

### Failed project model close and retry

`wrapper.close()` serializes with rebind and registers the actual wrapper/factory/cache record in its accepted `ProjectIndexService._Operation` before model close. If factory/model close fails, it raises only `project_embedding_close_failed` (no raw cause or raw log); the service roots operation → wrapper → carrier → retained parent even if callers drop their references and GC runs. `carrier.close()`, later semantic issuance/use and rebind for that binding refuse while pending. The independent keyword `service.search` path stays admitted and returns real indexed evidence through its live file/hash check. `service.aclose()` reports `project_embedding_close_pending` promptly instead of awaiting the unresolved operation forever; it retains ownership and does not settle unrelated native/store/capability uncertainty. Retry **the same** `wrapper.close()` after resolving the physical close error. Only a positive factory close removes that exact pending token; then close its carrier and retry service shutdown if needed. Never discard or replace the failing factory/cache record to manufacture retirement. A successful project close leaves the wrapper retired and prevents new embedding or rebind; later independent work must construct a new wrapper.

The known working offline qualification runner is the private `run-qualified-targeted.ps1` beside this handoff. From the Task2 worktree, invoke `& .superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-qualified-targeted.ps1 -RunName <unique-name> -Selectors @('<exact pytest node>')`. Change only the unique run name and selectors. It uses the existing qualification Python, read-only main-checkout shared site with parent `.pth` initialization and child-inherited pywin32 paths, the inventoried cached MiniLM, and offline flags. It writes pre-run source hashes, raw parent output and available private child log/XML under `task2-receipts`; it never downloads or installs a model. `task2-fix1-runner-smoke` passed its targeted catalog control, and `task2-fix1-offline-final-04` passed both real offline factory/composition controls with this exact environment. The prior fix1-offline-final-01/02/03 bootstrap failures remain separately recorded; they are not product assertions.
