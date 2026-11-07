# Independent review: root-owned size extractions

Verdict: no actionable Critical, Important, or Minor findings in the reviewed changes.

Base: ba1cafc55dd5ac9b6b1f066c936206e652ed6b65.
Reviewed working tree: /Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook.

## Byte qualification and limits

This review inspected the uncommitted working-tree source against that base. It is qualified to the source bytes returned by the review's reads and AST comparisons, not a new commit or a recorded content-hash manifest. The parent declared its product files frozen during the review. The parent identified the remaining permitted changes as two PLE1205 explanatory comments on live Loguru calls in the runtime helper and local-import sorting in prompt_export; those do not change the reviewed calls. No claim is made about later source changes.

This was a read-only review. No tests, performance runs, product edits, commits, pushes, or manifest generation were performed. Reported test results from the parent were not independently rerun. The previously prepared ba1 archive was provided for the parent's separate stale-fixture attribution: /private/tmp/mixins-size-ba1-baseline. Only its drag-perf test had the previously documented fixture overlay; citation/schema files were exact ba1. That archive was not used to run tests during this review.

## Scope checked

- app.py and app_config_notifications.py: config-load and newer-schema warning extraction.
- app_service_wiring.py and Notes/notes_sync_startup.py: task outcome, owner-starting projection, and non-coroutine awaitable helpers.
- DB/ChaChaNotes_DB.py and DB/sql_validation.py: migration statement splitting and leading SQL comment/whitespace handling.
- DB/Prompts_DB.py and DB/prompt_export.py: existing CSV/Markdown export surface, dependencies, and error handling.
- app_command_providers.py and UI/Workbench/help.py: binding-to-shortcut projection.
- Agents/agent_service.py and Chat/provider_readiness.py: configured fallback-provider readiness.
- Chat/console_provider_gateway.py and Chat/refused_stream.py: tool-definition rejection, model recovery, and sanitized model-id copy.
- Chat/console_runtime.py and Chat/console_trace_maintenance_runtime.py: post-readiness legacy normalization and physical maintenance loop.
- Chat/citation_trace_repository.py and Chat/citation_artifact_ownership.py: governed trace-row identity comparison.
- app.py, app_idle_warmup.py, and Chat/console_first_send_modules.py: warmup completion callback and current UI-loop runtime/view ownership. The bounded caller search also checked the updated test call sites.

## Evidence actually checked

1. Config warning callers retain their original early-return conditions. Notification text, severity, title, timeout, and live self.notify dispatch remain unchanged. Helpers load only for a detected warning.
2. Notes startup helper bodies are AST-equivalent to the originals after extraction. Done/cancelled/exception ordering, optional snapshot handling, caught exception scope, and awaiting the supplied object remain unchanged.
3. SQL splitting retains sqlite3.complete_statement, accumulated original statement text, incomplete-tail refusal, and the caller's SchemaError class passed explicitly. Leading-noise removal retains its original loop and comment behavior; it does not rewrite SQL executed by SQLite.
4. Prompt export's executable body is AST-equivalent after normalizing the explicit defining-module dependency reads and disregarding local import ordering. Argument forwarding preserves the old positional order. The actual defining module supplies PromptsDatabase, logging, DatabaseError, sqlite3.Error, and the Markdown archive-name helper. Query construction, optional fields/keywords, temporary-file behavior, output formatting, and error return shapes remain unchanged.
5. Binding projection still receives the caller's current Binding class and display-key function. Branches for Binding versus legacy tuple/list entries, description fallback, and ordering are unchanged.
6. Provider readiness still loads current configuration and calls the shared readiness function at invocation time. Provider-copy helper bodies are AST-equivalent after mapping explicit owner reads back to the original globals. Recursive recovery and tool-copy calls retain the gateway seam. CredentialSanitizer, escaping, model-id limits, helper functions, and safe-copy branches remain live defining-module reads.
7. The full extracted maintenance body is AST-equivalent after mapping explicit source-owner reads back to original globals. The original outer readiness wait, delay, and disposal checks stay in place. Provider activity, pause/resume callbacks, work generation, retry delays, pending GC state, epoch invalidation, owned database calls, cancellation checks, and logging retain their order and ownership. Moving provider_active creation does not snapshot its controller: it still reads the current runtime controller when called.
8. Citation comparison retains the same connection acquisition in the repository and passes that connection plus the same summary/profile/trace identities to the helper. All three SELECTs, parameters, expected identity sets, early refusals, and final comparison are unchanged. The AST difference inspected in SQL constants was indentation whitespace only. The helper import is deferred until runtime, avoiding a module-initialization cycle with the ownership module's existing repository imports.
9. Warmup completion remains dispatched through app.call_from_thread. The app callback performs current shutdown, runtime, and view reads on the UI loop. Explicit AttributeError fallbacks match the replaced getattr defaults. The production caller and all located test callers pass the new completion callback. Duplicate warmup entries were removed without dropping their unique module entries.

## Conclusion

The reviewed extractions preserve the examined behavior and boundaries. No corrective change was requested. This verdict does not assert that the full suite or performance cohort is green; those qualifications remain with their separately recorded evidence.
