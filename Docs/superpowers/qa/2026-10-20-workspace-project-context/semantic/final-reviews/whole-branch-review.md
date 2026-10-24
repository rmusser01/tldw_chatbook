Reviewed `5b5920b8…0d618403` read-only. No tests, imports, models, databases, Git mutations, or receipt inventories were rerun.

## Strengths

- Semantic consent and revisions are separate from keyword scope and saved Library controls.
- The implementation bounds source pages, native vector commits, candidate heaps, and result size.
- Model/provider work generally occurs outside SQLite units. Failed model retirement has explicit ownership and retry paths.
- CPU precision follows the selected device, so explicit CPU uses FP32.
- Release documentation distinguishes actual qualification, adverse attempts, historical source attribution, sampled memory, and analytical allocation limits.

## Important findings

1. **[P1] Filter cached sources before sending them to the embedding provider.**  
   [Workspace_Project_Index_DB.py:1001](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/DB/Workspace_Project_Index_DB.py:1001)

   `semantic_source_page()` selects every cached chunk without applying current processing policy or binding exclusions. Meanwhile, [service.py:1055](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/Workspaces/project_index/service.py:1055) permits semantic builds from keyword `Partial` coverage. Keyword refresh deliberately retains old members when their parent enumeration is incomplete.

   Trigger: index a file, exclude it or narrow the processing policy, then finish a refresh as `Partial` before that parent completes. Approving and building hosted semantics sends the retained, now-forbidden text to the provider. Query-time filtering cannot protect this disclosure.

   Apply current policy and authoritative exclusions before loading source text. Make checkpoint/completeness accounting use the same eligible source set. Cover retained forbidden rows after a partial refresh.

2. **[P1] Enforce the approved destination through HTTP redirects.**  
   [Embeddings_Lib.py:601](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/Embeddings/Embeddings_Lib.py:601)

   Project OpenAI-compatible embeddings reach `session.post()` with Requests’ default redirect handling. A `307` or `308` response can resend the complete JSON body—including project text—to another URL without new destination consent. Installed Requests source confirms that those redirects preserve the body.

   This behavior predates the branch for Library callers, but the new project path promises exact per-binding destination approval and currently inherits it unchanged.

   Make project transport refuse redirects before following them, or explicitly validate and authorize each destination before resending. Preserve the intended Library contract separately. Add a controlled two-endpoint redirect test proving the unapproved endpoint receives no project body.

3. **[P2] Do not let another wrapper’s close prevent an admitted carrier from retiring.**  
   [projection_scope.py:218](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/RAG_Search/projection_scope.py:218)

   `close()` calls `require()`, whose `_check_live()` calls `_require_projection_open()`. That service check refuses whenever **any** same-binding operation has a pending model-close token. Tokens are registered before ordinary factory close, even when close eventually succeeds.

   A query admitted before a semantic refresh can overlap that refresh. If wrapper A is closing while wrapper B finishes, B’s `carrier.close()` refuses solely because A is closing. B then enters `_retain_uncertain_model()`, permanently marking its operation `retirement_unproven`. Subsequent service admission refuses—including keyword operations—and retrying A cannot settle B’s separately retained carrier.

   Separate admission for new semantic work from retirement of already-admitted owners. Preserve ownership/thread/child checks during close without treating another owner’s pending close as grounds to prevent cleanup. Cover overlapping query/build retirement, including both ordinary slow close and failed close.

4. **[P2] Recheck semantic consent before final disclosure.**  
   [service.py:886](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/tldw_chatbook/Workspaces/project_index/service.py:886)

   After semantic ranking, candidate validation and the final disclosure fence check only registry authority and keyword scope/state/manifest. Semantic Disable or destination reconfiguration changes the independent semantic revision while deliberately preserving those keyword fields.

   Trigger: pause a live candidate read after semantic ranking, disable or reconfigure semantics, then release the read. The invocation still returns semantically selected hits with `semantic_status="ready"` under the revoked revision.

   Revalidate the captured semantic revision/settings/compatible generation at disclosure. On mismatch, safely discard the semantic contribution or return unavailable; simply changing the status label leaves stale semantic selection in the result. Add the race test through the public search path.

5. **[P2] Classify native-positive tests for unsupported SQLite runtimes.**  
   [test_project_semantic_settings.py:42](C:/Users/GDesktop-1/.codex/worktrees/workspace-project-guidance/tldw_tui/Tests/Workspaces/test_project_semantic_settings.py:42)

   Positive semantic settings/generation/search tests call real semantic mutations without a qualification guard. Production `admit_connection()` requires exact SQLite `3.49.1`, source ID, and compile options. Neither these test modules nor the shared fixtures provide a matching runtime guard. The comprehensive Ubuntu workflow still collects `Tests` and `Tests/UI` using unpinned Python `3.12`.

   On a supported interpreter linked to another SQLite build, these positives fail with `index_allocation_sqlite_unqualified` instead of expressing the intentional unsupported-runtime boundary. This is source-backed; I did not execute or claim an observed Ubuntu failure.

   Explicitly classify source-qualified positives and skip them with a precise reason on unsupported runtimes. Keep portable refusal/keyword controls active, and keep the required Windows lane’s qualification and positive assertions fail-closed.

## Declined to judge

- **General latency or interactive cancellation guarantees:** measured 62–74-second queries are a material documented limitation. The actual default tool budget is 300 seconds; a supposed 30-second default would be incorrect. Existing synchronous ranking does not establish prompt interruption throughout that work.
- **Universal relevance, performance, or platform support:** selected MiniLM cases and single local scale runs do not establish those claims.
- **Hard memory/filesystem quotas or hostile concurrent writers:** sampled RSS and the source-conditioned allocation argument have explicitly narrower scope.
- **Closing during unfinished Settings startup:** the retained positive qualifies settled startup, not that separate scenario.
- **Historical test-source equivalence or a mounted positive Ready-count display:** the retained limitations do not justify either inference.
- **Re-auditing immutable proof inventories:** no concrete inventory gap warranted repeating the independently reviewed raw-byte audits.

## Assessment

**Changes required.** The five findings concern provider disclosure, retirement isolation, revocation, and test classification.

The root’s newly authenticated hosted Project index contracts job succeeded at `d012`, whose production source matches this QA-only head. That supersedes the older queued snapshot for that job; it does not resolve these findings or qualify a subsequent fix head. Keep PR #3237 draft and auto-merge disabled until fixes, scoped rereview, and required final-head checks complete.
