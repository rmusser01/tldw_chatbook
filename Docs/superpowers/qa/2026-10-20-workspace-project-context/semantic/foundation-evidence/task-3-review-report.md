### Spec Compliance

- ❌ Issues found. `ProjectIndexService._semantic_query_vector` creates a wrapper and sends the query to it at `tldw_chatbook/Workspaces/project_index/service.py:620-674` before `semantic_ready_generation` checks the stored generation's artifact at `tldw_chatbook/DB/Workspace_Project_Index_DB.py:1126-1169`. A changed local artifact with the same approved catalog settings leaves control at Ready, but the request still constructs a model and embeds the query even though no compatible generation can be searched. Preflight the ready generation against the current complete artifact identity before query model effects; retain the later validation for races.
- ❌ Required verification is incomplete. `Tests/Workspaces/test_project_semantic_generations.py:447` mutates the manifest before one checkpoint, but the new tests do not hold a build across a mutation between source pages or at final publication. `Tests/Workspaces/test_project_semantic_search.py:195` tests close retry and shutdown, but never cancels the active query tool invocation repeatedly while the original worker owns a failed close. Add focused real-SQLite and worker controls for those explicit Task 3 cases.
- ⚠️ Cannot verify from this task diff alone: the unchanged scanner's complete file capture and tool authority remain the final disclosure guard, and the actual cached model's broader quality and 20k-file scale are reserved for Task 4. The controller should retain the separate whole-branch and release qualification gates.

### Strengths

- `tldw_chatbook/DB/Workspace_Project_Index_DB.py:911-1124` fences claim, page, checkpoint and publication to scope, semantic revision, claim and manifest; a failed refresh can retain a compatible old Ready.
- `tldw_chatbook/Workspaces/project_index/semantic.py:22-96` applies bounded exact cosine after the shared eligibility filter, then fuses ranked candidates before the service's live validation and byte limit at `tldw_chatbook/Workspaces/project_index/service.py:753-854`.
- `tldw_chatbook/Workspaces/project_index/service.py:1014-1209` owns the async build, close retry and publication ordering; `tldw_chatbook/Embeddings/Embeddings_Lib.py:891-944` retains failed OpenAI session eviction records. The report records real cached-model and held localhost-provider qualification, including adverse runs, at `.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task-3-report.md:24-60`.

### Issues

#### Critical (Must Fix)

- None.

#### Important (Should Fix)

- `tldw_chatbook/Workspaces/project_index/service.py:620-674`: Query embedding precedes the compatible-Ready lookup. An otherwise valid semantic consent can cause model construction and provider query work for an unusable stored generation. Check `semantic_ready_generation` with the current catalog/artifact identity before creating the wrapper or embedding the query, then keep the existing post-embed/read validation.
- `Tests/Workspaces/test_project_semantic_generations.py:447` and `Tests/Workspaces/test_project_semantic_search.py:195`: The explicit between-page/final-CAS source mutation and repeated cancelled query-owner tests are missing. These are plan-mandated concurrency controls, so the current passing tests do not establish those two required boundaries. Add focused tests without rerunning broad selections.
- `Tests/Workspaces/test_project_semantic_generations.py:1`: This new Task 3 file fails `ruff format --check` at final HEAD, as the implementation report distinguishes from pre-existing formatting failures in two production files at `.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task-3-report.md:37`. Format the new file and verify that focused check before marking the task Done.

#### Minor (Nice to Have)

- `.superpowers/sdd/2026-10-20-workspace-project-semantic-index/task-3-report.md:37`: Reported targeted pytest output still contains Pydantic and pytest-asyncio deprecation warnings. Identify their owners or suppress only known external noise so verification output is clean.

### Assessment

**Task quality:** Needs fixes.

**Reasoning:** The finite generation and retirement design has meaningful SQLite and real-provider evidence. The preflight compatibility error and two expressly required test controls need correction, and the new test file has not passed the required formatter gate.

**Severity counts:** Critical 0; Important 3; Minor 1.

**Checks:** Read the complete three-commit packaged diff in bounded chunks. Focused unchanged-seam inspection checked `configure_semantic` approval and `_eligibility_predicate` in `tldw_chatbook/DB/Workspace_Project_Index_DB.py:776-824,1767-1808`, and `_require_current_catalog` in `tldw_chatbook/Workspaces/project_index/semantic.py:256-270`. No tests were rerun; the implementation report's retained targeted receipts supplied the execution evidence.
