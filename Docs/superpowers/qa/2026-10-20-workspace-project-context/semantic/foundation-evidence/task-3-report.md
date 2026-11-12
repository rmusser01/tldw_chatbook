# Task 3 implementation report



## Result and ownership



Task 3 was implemented from base 677ddb4760cfbc57a54e294243218f4a23a3c6e1 in the isolated worktree, consuming named Task 1/2 handoffs. Initial implementation commit: 805701f93ada3797e9b7525c320cd1467b17904c. Returned held-lifecycle and transport delta commit: 5ad64a28ac. Eviction retirement correction commit: 286be157ae. Managed working tree clean after both deltas. The parent owns independent review, acceptance criteria and Done. The parent recorded all Ready-after-failed-refresh, same-owner close-retry and registry-retirement seam amendments in ADR241, accepted plan, Task 3 brief and CLI task plan before source edits. No new durable schema or owner was introduced. No full suite, model download, PR, merge or GitHub action occurred.



## Behavior and interfaces



- Finite float32 codec refuses empty, nonfinite, overflow and malformed vectors. Real SQLite generation source pages hold at most 32 records and 1 MiB text; vector commit pages are at most 1 MiB. The existing 1 GiB total binding allocation gate is authoritative; 4 MiB bounds aggregate queued source text. Membership preserves immutable source identity and metadata.

- Claim, append, checkpoint, failure and Ready publication use scope, semantic revision/settings, artifact, claim and manifest CAS. Changed manifest refuses staging publication. Failure settles only the exact still-owned staging generation and current Building control; later claim or reconfigure cannot be overwritten.

- Failed staging remains Error. Control may project Ready with a safe refresh-failed reason only for an independently compatible retained Ready generation; otherwise Error. A newer keyword manifest alone does not disqualify old Ready, but query still checks current consent, scope, settings, catalog, artifact and live file capture.

- Ranking filters forbidden members before the bounded top-100 cosine heap, checks vector shape/dimension, then RRF60 fuses keyword and semantic candidates with deduplication. At most ten hits are disclosed, bounded by the existing 16,384-byte tool result and live Capture for every file.

- One async build reuses one wrapper across pages; each finite query creates one wrapper. There is no idle cache. Per-query construction cost is for Task 4 measurement. One nonblocking synchronous semantic query slot returns ordinary keyword results under contention; no query model or pending close holds search-store/read admission. Held provider work leaves keyword lookup usable.

- Exact-key retry_semantic_close(key) signals only the original build/query worker for same-owner wrapper close. It tries once immediately before persistent signal wait. Model-free semantic_close_pending(key) and ProjectSemanticState.close_pending let Task 4 display Retry cleanup without model construction or private operation traversal, even after failed shutdown. Prompt shutdown reports pending retirement. An exceptional held worker can require process exit.

- The actual outer registry context yields owning-thread retirement callback. Every new/replacement WorkspaceDB handle is positively physically retired before model effects; a borrowed caller handle stays open and causes semantic refusal. Unknown native close roots actual wrapper/carrier/store/run lease/pin in existing operation, blocks model effects and never claims native retirement. Generic unknown close may require process exit.

- Wrapper/carrier positive close precedes final artifact/config/CAS revalidation and Ready publication. A positively retired semantic run clears its exception traceback before joining bounded 16-record history; pruning preserves a still-live owner-thread join record. Unresolved owners retain errors/capabilities.



## Files



Production: tldw_chatbook/Workspaces/project_index/semantic_vectors.py, models.py, semantic.py, service.py, runtime.py; tldw_chatbook/DB/Workspace_Project_Index_DB.py; tldw_chatbook/RAG_Search/projection_scope.py; tldw_chatbook/Embeddings/Embeddings_Lib.py private OpenAI transport callback. Tests: Tests/Workspaces/test_project_semantic_generations.py, test_project_semantic_search.py; Tests/Backup_Recovery/test_project_semantic_lifetime.py; Tests/Agents/test_project_index_tool.py exact-byte fixture; Tests/RAG/test_embeddings_lib_logging.py.



Parent-authored amendment files to commit with Task 3: backlog/decisions/241-workspace-project-context-and-folder-bound-index.md, Docs/superpowers/plans/2026-10-20-workspace-project-semantic-index.md, backlog/tasks/task-36422 - Build-bounded-semantic-generations-and-live-validated-project-search.md. Private SDD progress/brief/current-state files are coordination evidence.



## Verification receipts and adverse evidence



All receipts named below are in .superpowers/sdd/2026-10-20-workspace-project-semantic-index. Native runs kept pre-run source hashes/argv, bounded parent and raw child stdout/stderr/XML outside numbered pytest retention. Selections overlap; pass counts are not additive unique-test counts.



- TDD: task3-codec-red-2.log then task3-codec-green.log (5 passed); task3-store-red-3.log then task3-store-green.log (7 passed); task3-rank-red-2.log then task3-rank-green.log (1); task3-build-red.log then task3-build-green-2.log (1); task3-search-red.log then task3-search-green-1.log (1); task3-fusion-red.log then task3-fusion-green.log (3). Earlier codec/store/build bootstrap or collection failures are setup, not product RED evidence.

- task3-registry-green-2.log passed direct real WorkspaceDB owned/replacement/borrowed/denied closure. task3-query-close-green-1.log passed same-worker query retry and live keyword. task3-crossprocess-04 passed binding-wide exclusion across namespace/process plus competing keyword refresh refusal. task3-quota-04 passed real allocator refusal and staging Error. Earlier fixture-design failures remain retained.

- task3-real-async-04 and task3-http-qualified-02 passed first actual cached MiniLM async build with requests/httpx blocked, and held localhost OpenAI-compatible provider with live keyword hit. task3-final-real-async-01 and task3-final-http-close-01 passed after Ready publication ordering correction. The latter injected two native factory-close failures, saw prompt pending shutdown and exact-owner retry, and no Ready after shutdown. These precede later uncertainty/history edits.

- task3-manifest-red-01 exposed changed-manifest failure settlement; task3-manifest-green-01 passed changed-manifest and superseding claim/reconfigure. task3-final-extra-01 passed zero-member Ready versus unavailable, binding-separated identical vectors, corrupt/oversized stored vector refusal, and actual WorkspaceDB service entry/replacement closure. task3-provider-registry-01 overlapped a source edit and is not final-source evidence.

- task3-tool-budget-red-01 saw 16,416 bytes after semantic_status was added. Only its payload fixture changed; task3-tool-budget-green-01 passed the exact 16,384-byte boundary, overflow/JSON controls isolated.

- task3-final-deterministic-01 yielded 19 passed, 2 setup failures. Synonym and tool timed out in Building at existing Rig.settle 15 seconds; task3-search-isolated-01 repeated synonym startup timeout, with chunking starting around 14.5 seconds. No deadline or production scanner change. The serial tool failure remains an intermittent keyword startup observation even though its exact-byte assertion passed isolated. The synonym case now seeds real SQLite committed Ready keyword state under actual lease, still executes production semantic fusion/live Capture, and passed task3-search-seeded-01. Full failed output is retained.

- task3-final-qualified-03 passed actual offline cached MiniLM async build after uncertainty/manifest/carrier fixes. task3-final-qualified-http-03 passed held localhost provider, concurrent live keyword and exact-owner close retry on that source. Both have task3-pre hashes and parent/child receipts under task2-receipts.

- task3-final-exception-01 passed direct callback, but unknown-registry and service-registry fixtures failed before assertions in unrelated keyword scanner setup (one BindingLifetime busy; one Building at 15 seconds). Its raw and copied child receipts remain. Those two fixtures now seed committed Ready keyword rows under real lease. task3-final-exception-02 passed 4/4: uncertain registry owner retained after GC and unresolved shutdown, actual service WorkspaceDB entry/replacement close, direct callback close, and retired error traceback clearing with live join-record retention. Prehash, raw main log and eight copied child XML/log receipts remain.

- Before the returned lifecycle/transport delta, only post-positive-retirement error clearing/pruning changed production after qualified-03/http-03. task3-final-history-model-01 passed the actual offline cached MiniLM async build at that initial implementation boundary. The later delta changes service close handling and the OpenAI HTTP callback only; no local-model construction/quality path changed or was replayed merely for uniform hashes.

- Scoped ruff check --no-cache passed all changed production and tests; git diff --check passed. Ruff format --check --no-cache passes the focused RAG logging test file after a blank-line-only post-test edit. `tldw_chatbook/Embeddings/Embeddings_Lib.py` and `tldw_chatbook/Workspaces/project_index/service.py` existed and failed formatter check at Task 3 base 677ddb4760. `Tests/Workspaces/test_project_semantic_generations.py` did not exist at that base; Task 3 introduced it at 805701f93a, where it already failed formatter check. The Task 3 generation and search test files are now formatted in review fix round 1; the inherited production formatter failures remain, with no broad production formatting churn. Existing Pydantic/pytest-asyncio deprecation warnings remain. Full suite intentionally not run per AGENTS.md.



## Review focus and limits



Review Ready compatibility and refresh-failed display separately from failed staging, manifest/CAS settlement, exact-key close ownership, physical WorkspaceDB closure, unknown-close retention, and publication after positive model retirement. Scanner startup has intermittent retained 15-second fixture timeouts; the Task 3 semantic/provider cases use either committed Ready SQLite setup or their own live keyword assertion. The isolated tool exact-budget pass and serial setup failure are both retained. No broad scanner diagnosis, model quality/20k scale study or UI wiring is claimed. The fixed-target private trust helper only removed two parent-verified sandbox Modify grants when they recurred; before/after and unchanged trust-checker receipts are retained. Initial auto-review rejected the ACL action; after script/ACL inspection the narrow authorized action was approved.





## Task 3 lifecycle return (pre-review)



The parent identified an acceptance gap after commit 805701f93ada3797e9b7525c320cd1467b17904c: no prior new semantic-run tests held the worker across repeated aclose cancellation, Clear, detach, exclusion and settings changes. The returned delta exercises those service-owned paths. `task3-clear-close-red-01` exposed only one build close attempt and its child failed; incomplete test cleanup then caused a 180-second forced abort, which proves no physical retirement. `task3-clear-close-green-01` exposed known close pending incorrectly promoted to generic retirement uncertainty by the outer registry scope. After narrow source fixes, `task3-clear-close-green-02` passed: durable revocation, prompt known-pending error, readable status, old-scope keyword refusal, prompt aclose pending report, and positive same-owner close/lease retirement. `task3-held-mutations-01` passed detach and exclusion but its settings fixture timed out through test-side semantic_state polling and allocation contention. The event-only observation in `task3-held-settings-02` passed. All original raw failure and final passing receipts are retained.



The first actual held localhost-provider cancellation control, `task3-held-http-cancel-01`, reached two cancelled aclose observers and pending model close but force-aborted at the 300-second outer bound. Its child XML/log and parent/prehash receipts remain; forced exit is not graceful retirement. The test attempted to invoke the captured OpenAI CacheRecord close callback, which was `None`, on retry. This explains the TypeError and is not evidence of an exception from a real Session.close. Producer inspection found `_openai_embedder` owns a `requests.Session` in its closure while `_build` had stored `close=None`; prior `task3-final-qualified-http-03` used a no-op successful callback and demonstrated held provider/keyword/owner scheduling only. Parent recorded the explicit transport retirement amendment in ADR241, accepted plan, Task 3 brief and CLI plan before source edits.



The parent-authorized correction returns `(embed_callable, session.close)` from the private `_openai_embedder` and stores that callback in the existing factory CacheRecord. No transport manager or owner was added. The direct test consumer was adapted. `task3-transport-factory-01` passed 7/7 focused default-factory/logging controls, including sanitized failure paths, exactly one Session.close after ordinary use, retained cache after injected close failure, and successful second close. `task3-transport-held-01` passed 1/1 on the final service/producer source: actual localhost HTTP work stayed held during two aclose cancellations, the observer remained responsive, live keyword lookup succeeded, the worker did not retire or publish early, known close pending was reported promptly, and explicit same-owner retry invoked the real bound `requests.Session.close` after two injected failures. It then emptied the factory cache, closed the wrapper/carrier, retired the run/lease/store, and refused Ready publication after shutdown. The qualified runner's `task2-receipts/task3-transport-{factory,held}-01-pre.json`, parent logs, and held child XML/log retain exact source hashes/argv. The factory-only run had no private child profile. Its test file later received one blank-line-only formatting edit; production source and assertions did not change, so the retained 7/7 run is its behavioral evidence while the formatter check covers the final text. The pre-ordering, pre-transport and final-source receipts are distinct; no counts from overlapping selections are summed.



Structural deviation approved by parent for review: the plan named `ProjectSemanticIndex`, but generation methods live on the existing `ProjectIndexService` and codec/ranking in `semantic.py`/`semantic_vectors.py`. This keeps the existing lifecycle and authority owner instead of adding a delegating wrapper class; the reviewer still judges cohesion and size. Storage allocation is a 1 GiB total per-binding gate. The 4 MiB figure bounds aggregate queued source ownership across admitted jobs, not storage allocation. The finite one-wrapper-per-build and one-per-query policy can incur query construction cost; Task 4 measures that real boundary. Exceptional generic unknown native retirement may still need process exit, and unresolved owners remain rooted/reported rather than silently closed.



## Task 3 eviction retention follow-up (pre-review)



Self-review after 5ad64a28ac found that the existing `EmbeddingFactory.embed` idle and LRU branches popped a zero-refcount CacheRecord before invoking its close callback. The new real OpenAI Session callback made an exception there a concrete loss of the only retirement handle. Parent recorded the narrow close-before-pop amendment in ADR241, accepted plan, Task 3 brief and CLI task plan before this source edit. Both branches now invoke the existing close callback under the existing condition before removing the exact record. A failure propagates, retains that record/callback and constructs no replacement; later embed or factory.close can retry on the owner. Refcount and successful capacity behavior are unchanged. No new owner or schema was added.



`task3-eviction-red-01` was an adverse fixture run: idle expiration was not deterministic and its cleanup masked the intended assertion; LRU exposed missing record retention. After setting the real record's `last` time explicitly and bounding fake-session cleanup, `task3-eviction-red-02` failed 2/2 specifically because both callbacks' records had been removed after injected close errors. On the corrected producer source, `task3-eviction-green-01` passed both idle and LRU callback-failure/retention/retry cases (2/2). The directly affected `task3-eviction-concurrency-01` module passed 5/5, including checked-out model protection and in-flight close waiting. Qualified-runner prehash/argv and parent logs are under `task2-receipts/` for every run. These factory tests have no native private child profile, so there is no child XML/log. A post-test one-line formatting wrap in the new test changed only whitespace; the green behavioral run predates that wrap. The producer remained byte-identical to both green prehashes. No cached local-model or held HTTP control was replayed because this delta changes only factory eviction ordering, which those paths did not enter.



Scoped Ruff lint `check --no-cache`, test-file `format --check --no-cache`, and `git diff --check` pass on this delta. The producer file already failed broad format --check at 5ad64a28ac, and was not reformatted wholesale. The prior stopped 300-second provider run, initial fixture RED and corrected product RED remain adverse evidence, never counted as green.



## Independent review fix round 1 — initial boundary



FIX_BASE 286be157ae92e2552cd70f2ae3fc06da256770e5. Reviewer found query model work before compatible-Ready proof, missing explicit multipage/final-CAS and repeated active tool observer cancellation controls, and formatting of the Task3-introduced generation test. The incompatible-artifact real-SQLite RED is `task3-r1-preflight-red-03` (keyword returned, but actual issuer/wrapper reached); `red-01` failed fixture admission setup and `red-02` failed test indentation, not product behavior. The finite issued carrier now supplies complete artifact identity for a real SQLite compatible-Ready preflight, with positive registry/carrier retirement before any wrapper construction on refusal; the existing post-embed catalog/artifact/Ready/live-capture checks remain. `task3-r1-preflight-green-01` passed 1/1. The multipage/final-CAS and production-issued tool cancellation controls have since passed; exact final-source receipts and limits follow. No full suite or local model replay was run.





## Independent review fix round 1 — completed handoff



**1. Compatible Ready before query model work.** `ProjectIndexService._semantic_query_vector` issues a finite production carrier to obtain the current complete artifact identity. It then opens one real cleanup-only SQLite store and finite read fence to require `semantic_ready_generation(scope,state,carrier.artifact_fingerprint)` plus the live root/chain. That SQL/fence/store closes, and the owning registry retirement callback proves native closure before any wrapper/provider effect. An incompatible Ready positively closes the carrier and falls back to keyword results. Unknown registry/store retirement retains the actual carrier; generic callback exceptions keep the prior raise behavior. Existing post-embed catalog/artifact/Ready/live-Capture validation still catches later races. No second full artifact hash was added. The `test_incompatible_ready_artifact_skips_query_model` control seeds a genuinely incompatible Ready in real SQLite under admitted storage, observes a keyword hit, and asserts zero wrapper constructions. `task3-r1-preflight-red-03` failed at the asserted issuer/model side-effect boundary on FIX_BASE source (`service.py` prehash 01960E3ABEDC); `red-01` had a test-only missing factory admission and `red-02` a test indentation error. `task3-r1-preflight-green-01` passed 1/1 on intermediate source. On final service SHA256 `6BDED02A5BB778B966689965DE3D767C0BA4BDC4819DB73498B6C7F1D4B024C9`, `task3-r1-final-preflight-01` passed 1/1 with new search test SHA256 `FE9121B5784963B96FAB7B8595E5C7277B1BDFA59D74783AEFA752313F82517E`. `task3-r1-synonym-01` passed 1/1 compatible Ready semantic fusion and live validation at those same final hashes.



**2. Mutation and active query owner.** `test_multipage_generation_refuses_changed_manifest_at_page_or_final_cas` seeds 33 real SQLite chunks, commits the first bounded 32-row page, then mutates the manifest either before the second page or after the final append. The next page or final CAS refuses the stale generation; neither branch gets a Ready pointer, and exact staging failure settles Error. `task3-r1-mutations-01` passed 2/2 with generation test SHA256 `A6990D3C47DF32A3E67A994CE14E98A65E4FE3262F208EAAAE9908B7F776787D` (same as current). That control precedes the final query exception-handling split, which does not touch generation/storage code. The active `test_cancelled_query_tool_observers_keep_native_close_owner` uses real admitted storage, actual live scanner Capture identity, production projection issuance, `LocalToolProvider.invoke`, a localhost OpenAI-compatible provider and the real bound `requests.Session.close` callback. Two cancelled asyncio observers leave the already-running tool worker and its carrier/factory rooted. Concurrent tool keyword lookup returns the live file while model close is pending. The first explicit retry still fails; the second reaches actual Session.close on the original owner thread. The test asserts callback thread identities, cache evacuation, wrapper/carrier closure, no pending owner and empty operation set. `task3-r1-query-owner-01` failed only because its synthetic file identity did not match the actual tool Capture; cleanup then positively retired the provider. `task3-r1-query-owner-02` passed 1/1 after seeding actual Capture identity. After the exception-routing correction, `task3-r1-final-query-owner-01` passed 1/1 on the final service/search hashes above, with child XML `tests=1 failures=0 errors=0 skipped=0`. `task3-r1-existing-query-close-01` passed 1/1 the directly affected prior query retry/keyword control on those same hashes. No synthetic carrier or `asyncio.to_thread` is the authority for the new active tool test; the executor only retains the genuine `LocalToolProvider.invoke` worker while observers cancel.



**3. Formatting and static checks.** Ruff formatted the two Task 3-introduced semantic test files. `ruff check --no-cache` passed the three changed files; `ruff format --check --no-cache` reports both tests already formatted; in-memory compilation of all three and `git diff --check` passed. A preliminary `python -m py_compile` under the read-only sandbox failed to write `service.py` bytecode (`Permission denied`); its replacement in-memory compile passed and did not edit source. Inherited production `service.py` and `Embeddings_Lib.py` formatter status at base 677ddb4760 is unchanged. The test file formatting is this task's correction, not inherited baseline.



For every named qualified run, the exact invocation was `pwsh -File .superpowers/sdd/2026-10-20-workspace-project-semantic-index/run-qualified-targeted.ps1 -RunName <name> -Selectors <named test>`, under a hidden bounded parent process. Exact argv/source hashes are in `task2-receipts/<name>-pre.json`; parent output is `task2-receipts/<name>-parent.log`. Private-profile cases also have copied raw child XML/log under `task2-receipts/<name>-<case>-child.{xml,log}`. The generation mutation control does not enter a private child profile; its parent log is the raw result. `task3-r1-final-query-owner-01-test_cancelled_query_tool_obse0-child.xml` and `task3-r1-final-preflight-01-test_incompatible_ready_artifa0-child.xml` independently show one passed case with zero failures/errors/skips. Warnings are retained and inherited: unset `asyncio_default_fixture_loop_scope` in `pyproject.toml` and Pydantic 2.11 `Field(exclude_if)` from `TTS/audio_cpp_guided_config.py`; no suppression or unrelated edit. The full suite and cached local-model quality work were not replayed for this scoped query/generation-test correction. Final-source service/search/generation SHA256 are `6BDED02A5BB7`, `FE9121B57849`, `A6990D3C47DF` respectively (full values above or in prehash manifests). Parent owns fresh independent review, AC and Done.


Review fix round 1 committed at `538faa103ed84fa56d96986795df6e3355123bb5` (FIX_BASE `286be157ae92e2552cd70f2ae3fc06da256770e5`); managed worktree is clean. Parent now owns fresh scoped review, AC, and Done.
