# Task 9 implementation report

Status: DONE_WITH_CONCERNS; TASK-38544 remains In Progress for independent review.
Base: dbc3819f33eb92ff386513ea63da437f07eb91b6.
Commit: see local task9 commit and evidence manifest; no push or merge.

# Prompt mix-ins: integrated qualification

Date: 2026-11-01–02. Task: TASK-38544. Governance: [ADR-286](../../../../backlog/decisions/286-console-prompt-mix-ins.md). Task9 base: `dbc3819f33eb92ff386513ea63da437f07eb91b6`. Implementation is ready for independent review; the task remains **In Progress**. No full suite was run.

## Result and scope

The real local source/store/controller/gateway journey passes with diagnostic capture explicitly off, both ordinary and character chat. It verifies accepted queue A survives source edit B, toggle off and tab switch; failed A → original Retry A; current-off regeneration; current-B regeneration; a clean independent next send; persisted reference/state reopen; and empty new-chat selections. Separate native controls verify stopped-empty Resend, explicit Continue's synthetic request anchor, unavailable new-format Retry refusal, snapshot-write failure before any adapter entry, uncertain post-dispatch settlement recovery, and a calculator tool successor sharing one attempt. Receipt checks compare operational user/assistant anchors, source versions and provider semantic requests, not merely call counts. Promotion/copy/fork/restore/purge are covered in the affected DB/admission/restore modules.

The named generation-switch audit found a real display gap: adding a manual generation retained the previous generation's `mixin_usage_count`. The three-line fix carries the transient count through `ConsoleVariant` capture/apply. An unqualified manual replacement carries `None`; operational receipts alone continue to authorize Retry. The persisted-store regression failed before the fix and passes in the final integrated run. No schema, rendering, permission or provider-policy change was introduced.

## Final targeted commands

All commands ran in the managed worktree with `PY=/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`. The bootstrap-profile fixture is the repository's existing fixture; no new test app or framework was added.

```sh
$PY -m pytest Tests/Chat/test_console_prompt_mixin_journey.py Tests/Chat/test_console_prompt_mixin_retry.py Tests/Chat/test_console_prompt_mixin_admission.py Tests/Chat/test_console_prompt_mixin_caching.py Tests/Chat/test_console_prompt_mixin_composition.py Tests/Chat/test_console_prompt_mixin_values.py Tests/DB/test_console_prompt_mixins.py Tests/Backup_Recovery/test_prompt_mixin_restore.py Tests/UI/test_console_prompt_mixins.py Tests/UI/test_library_prompt_mixins.py Tests/UI/test_design_token_governance.py Tests/UI/test_css_bundle_sync_guard.py -q -p no:cacheprovider --basetemp=/private/tmp/task9-final
```

**249 passed in 197.26s**; [exact log](private-logs/task9-final.log). Includes all seven new journey cases, actual persistence/restore ownership, final UI fixtures, token governance and generated CSS consistency.

```sh
$PY -m pytest Tests/Prompt_Management/test_prompt_mixins.py Tests/Prompt_Management/test_prompt_artifact_codec.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Prompt_Management/test_local_prompt_service.py Tests/Prompts_DB/test_prompts_db_retained_history.py Tests/Library/test_prompt_export_roundtrip.py Tests/Prompt_Management/test_prompt_chatbook_record.py Tests/Chat/test_console_turn_execution_context.py Tests/Chat/test_console_native_commit_integration.py Tests/Chat/test_anthropic_caching_degrade.py Tests/Backup_Recovery/test_sqlite_validation.py::test_prompt_v5_to_v6_recovery_migration_preserves_rows -q -p no:cacheprovider --basetemp=/private/tmp/task9-predecessors2
```

**540 passed in 37.83s**; [exact log](private-logs/task9-predecessors2.log). The first predecessor command additionally named nonexistent `Tests/Chat/test_anthropic_prompt_caching_flags.py`: collection ran no tests, preserved in [failed invocation](private-logs/task9-predecessors.log). This is not counted as a pass.

```sh
$PY -m pytest Tests/Packaging/test_installed_distribution.py::test_built_artifacts_match_distribution_contract -q -p no:cacheprovider --basetemp=/private/tmp/task9-packaging
```

**1 passed in 15.28s**; [exact log](private-logs/task9-packaging.log). The test builds a wheel and sdist from a disposable source copy and checks installed distribution/migration resources; it is not merely a source-tree file-existence assertion.

```sh
$PY -m ruff check Tests/Chat/test_console_prompt_mixin_journey.py
$PY -m ruff format --check Tests/Chat/test_console_prompt_mixin_journey.py
```

New test passes Ruff and format. For the two changed product files, `ruff check --output-format=json --stdin-filename <path> -` read `git show dbc3819f:<path>` versus current bytes. **142 diagnostics on both, zero introduced by per-file code/message multiplicity**, not a changed-line filter: [delta](task9/lint-delta.json), [baseline](task9/lint-base.json), [current](task9/lint-current.json). Formatter passes models on both; store has existing formatter debt on both with **identical 227 edit lines**: [baseline](task9/format-baseline.json), [delta](task9/format-delta.json). Existing debt was not broadly reformatted. Final whitespace and evidence-integrity checks are recorded in [final-checks.json](final-checks.json).

## Actual app, production styles

[Launch manifest](live/launch.json) and [77-file product source hashes](live/product-source-sha256.json) pin the actual code, including the three-line count fix. The existing `roleplay-library-layout-review-2026-10-01/harness` supplied `mkprofile.sh`, `env.sh`, `mock_llm.py`, `click.sh`, `where.py`, `shot.sh` and `shot_png.py`. The profile was a mode-700 child of platform `mktemp -d`; canonical `/private/var/...` spelling satisfied the existing lexical guard. The first noncanonical spelling was refused, not bypassed.

Normal launch used the console script through `rp_app_env`'s `env -i` allowlist, with child-only HOME/XDG/config/data paths and the null keyring backend:

```sh
tmux -f /dev/null -L mixins-task9 new-session -d -x 160 -y 45 -c <private-profile>/cwd "${RP_ENV[@]}" /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/tldw-cli --focus
HARNESS_STATE=<private-root> $PY Docs/superpowers/qa/roleplay-library-layout-review-2026-10-01/harness/mock_llm.py 18765 --bodies
```

The actual production UI sent HTTP to that loopback responder. This is a deterministic local integration receipt, **not real external model behavior**. No separate `asyncio.run` seeded loop-owned services. No user profile, keychain, shell environment, permission guard or recovery state was overridden.

| Actual UI action | Evidence and observation |
| --- | --- |
| Menu → Prompts → Mix-ins | Existing Browse landing retained. [160×45](live/captures/workbench-160x45.txt), [100×30](live/captures/workbench-100x30.txt), [80×24](live/captures/workbench-80x24.txt). |
| Library creates Scene and Brief, guarded Enable in current chat | Scene contains literal `{{scene}}` and an SVG instruction; editor saves and handoff populate existing Workbench. No new renderer exercised. |
| Keyboard toggle/order | Space toggles; Tab/Enter moves Brief ahead of Scene. [Ordered controls](live/captures/keyboard-order-160x45.txt). |
| Exact next-send preview | Literal instruction visible in the actual request preview. [Preview](live/captures/preview-literal.txt). The early composer contained a test navigation string; it was cleared before provider sends. |
| Running send plus queued send | Actual queue drained two sends; first two bounded [HTTP receipts](live/receipts.json) contain Brief then Scene then the distinct clean user input. [Rendered responses](live/captures/live-queued-response.txt). Queue acceptance was observed interactively; the retained frame is after drain. |
| Failed response, Retry original while both current entries off | Stopping only the private responder caused a real connection refusal; original Retry label and close-confirmation state are preserved in [failed frame](live/captures/failed-original-label-confirmed.txt). After restarting responder, request three still contained both original definitions. [Successful Retry](live/captures/retry-original-while-off.txt). |
| Regenerate with current off | Request four's current user is exactly `Retry original scene.` and contains neither current mix-in; Assistant 2/2 also appears in the later [source-loss send frame](live/captures/source-unavailable-send.txt). |
| Close/reopen and new chat | [Fresh chat](live/captures/new-chat-empty-selections.txt) has no active selections. [Reopened chat](live/captures/reopened-order-off.txt) retains Brief→Scene with count zero. Plain terminal checkbox glyphs alone do not encode the color/state; enabled count and owner receipts establish state. |
| Invalid save | Blank Name + Save displays “Library changes were not saved. Check Name and Instructions, or reopen after a source/version change. Your edit is kept.” [Full frame](live/captures/invalid-name-save.txt). This is validation failure, not simulated disk failure. |
| Selected source removed | A controlled soft-delete changed only the private Scene Library record, preserving its version/timestamp triggers. Initial missing-version update was refused and rolled back. [Injection receipt](live/source-failure-injection.json). Reopening Workbench displays Scene unavailable and “Enabled mix-ins unavailable; disable, remove or recover them to send.” [Full frame](live/captures/source-unavailable-reference.txt). |
| Source-loss Preview/Send | Preview returns the generic “Preview unavailable due to an internal error.” [Exact frame](live/captures/source-unavailable-preview.txt); controller catches and logs `MixinSnapshotUnavailableError` with traceback, rather than source-specific preview copy (`Chat/console_chat_controller.py:20448` logging, `:20489` fallback; capture raises at `Chat/console_prompt_mixin_service.py:414`). Send is refused and retains `Preserve this unsent source-loss draft.` [Full frame](live/captures/source-unavailable-send.txt), four HTTP receipts unchanged, five accepted operational attempts total, zero pending dispatch checkpoints. |

All live `.txt` and `.ansi` files are captures of the real running Textual PTY. Matching PNGs come from the existing ANSI→Rich→browser renderer; these are **not OS terminal-paint screenshots and not Pilot SVGs**. Font/border fidelity and wide PNG horizontal presentation are limited; plain/ANSI terminal cells carry exact layout/text evidence. Prior slice Pilot captures remain separately under `task7-captures/` and `slices/task-8-evidence/`; they do not substitute for the actions above.

The initial served launch `python -m tldw_chatbook.app --serve --host 127.0.0.1 --port 18879` opened the actual browser but its child exited with `AttributeError: module '__main__' has no attribute '_CONSOLE_SKILL_APP_SOURCE'`. [Server log](live/server.log) and [browser capture](live/served-child-ended.png) preserve the failure. The normal console-script launch succeeded. No unrelated entry-point repair was attempted or claimed to be baseline-qualified.

Normal Ctrl+Q completed at 18:35:21. [App log](live/app.log) records chat/library unmount, MCP sessions disconnected, cached client closed and timers stopped. App PID and isolated tmux server disappeared; own served process and responder were then terminated. [Teardown](live/teardown.json). Detached tmux did not retain an exit code; no claim of exit code zero or per-client instrumented closure is made. A retiring browser-read worker logged caught `console_browser_source_changed` during quit. It is retained as a limitation, not hidden behind the process-exit check.

## Provider/cache limits

Only presence (never values) of `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY` in the inherited shell was checked: all absent in [presence record](live/credential-presence.json). The real user's config/keychain was not inspected, so this does not prove all other credential sources absent. The private fixture's placeholder API key is synthetic. No eligible external-model request, model-specific minimum stable-prefix experiment, cache TTL/kill-switch measurement, or live cache-read/write counter qualification was performed. Deterministic payload/provider-boundary tests pass; **no cache-hit rate, latency, token or monetary savings are claimed**. Existing provider policy remains unchanged.

## Failure attribution and retained historical evidence

Earlier failures are not all called inherited simply because a slice BASE failed. The controller's original-product audit used the exact 26 previously failed nodes, original `d83ca84cdbd8cb934ba747bcb7a2179b934904ed` versus pinned Task8 `3c77bbbdd701052bb778eaf1bc09fea7b838c8ae`, identical admitted bootstrap-profile plugin and three frozen native harness files on both. Original: **20 failed, 6 passed (27.86s)**. Task8: **19 failed, 7 passed (31.77s)**. There were no current-only failures. The original-only workspace-chip assertion was nondeterministic across prior runs. This audit was preserved, not rerun on Task9.

Exact node lists, commands and hashes: [original launch](slices/original-attribution-launch.json), [original result](slices/original-attribution-result.json), [current launch](slices/current-attribution-launch.json), [current result](slices/current-attribution-result.json); complete logs/plugin/frozen fixtures in `original-attribution/`. Common failures include trace availability, generation projection, stale fake signatures, attachment gates, tuple/StopIteration, Full output labels, maintenance calls and missing left rail. Current traces additionally contain fail-closed `mixin_persistence_unavailable` from old fake persistence; same outer assertion does not prove the real path. The new native owner journey supplies that proof separately.

Six Canvas/Mermaid cases passed on both products with the same legitimate profile fixture. Earlier disabled-Canvas slice failures were fixture qualification, not product parity. Task1's already proven original UI/decorative/lazy-export failures were not rerun. The Task6 diagnostic-inventory fixed normalization assertion is original debt by identical test/fixture/checked-inventory blobs at original base, Task6 BASE and Task8: [blob attribution](slices/original-manifest-blob-attribution.json). The assertion precedes generated inventory comparison; this does **not** claim the generated whole-repository inventory passes.

Historical slice reports/reviews and exact logs are preserved under `slices/`; private paths inside immutable reports resolve through [private-log-locations.json](private-log-locations.json), same-name `private-logs/`, and the separately named capture/audit directories. Final provenance audit identified 46 shared-temporary filename collisions from unrelated fleet/mobile tasks. Only those exact hash-verified newly copied files were reversibly quarantined in ignored workspace; originals remain untouched and none are committed. Retained feature iteration logs are historical material, not additional green Task9 results. No database, private source archive, credential value or user-profile copy is part of this evidence package. Failed development iterations remain: fixture API/id corrections, real queue terminal-state prerequisite, regenerated branch identity, live/durable CAS revision distinction, owner-loop Stop, Resend replacement identity and Continue synthetic current input. The final passing commands above supersede them; counts are not aggregated across repeated/overlapping runs.

## Integrity and review boundary

`evidence-manifest.json` records SHA-256 and Git blob identity for every preserved evidence artifact except itself; `product-source-manifest.json` records final feature source and test bytes. `verify_evidence.py` checks filesystem bytes against the manifest and the Git index or a requested committed revision. Historical capture source hashes retain their own slice qualification; the current live product source manifest is checked against final product bytes. Scratch `.superpowers` and platform temporary paths are provenance only; review does not require them to survive.

Self-review checked source/store/runtime ownership, exact receipt anchors, clean stored users, selected-generation projection, no count-based Retry authority, and the narrowly scoped diff. Independent spec/quality review remains required. Remaining concerns are the served module launcher failure, generic preview failure copy/logging, shutdown browser-read log and unavailable external provider/cache proof described above. These are not represented as passing cloud/UI evidence.


## Files and TDD

Product: `Chat/console_chat_models.py` and `Chat/console_chat_store.py` (three added lines). Test: `Tests/Chat/test_console_prompt_mixin_journey.py` (seven cases). Documentation: five subsystem architecture documents, Console and Library prompt guides, ADR-286 implementation state, plan evidence state and TASK-38544 notes. QA: preserved historical and final logs/captures/manifests with no source archives or databases.

RED: `$PY -m pytest Tests/Chat/test_console_prompt_mixin_journey.py -q -p no:cacheprovider` demonstrated manual replacement count `1` instead of `None` in `task9/task9-red.log`. The model/store capture/apply fix is the minimal corresponding change. GREEN: the final249 command above includes this regression and all integrated cases. No product edits followed that final passing run.

Self-review found no new authority or serialization boundary: the count field is repr-hidden, transient, copied alongside existing generation metadata and never queried for replay. The native tests retain production source/service/store/controller/gateway ownership and operational SQLite receipts. The loopback live UI reached production HTTP and native persisted receipts; it does not assert model quality. Actual source loss is a version-guarded mutation of a private fixture only, not a Console state bypass. The save failure is Name validation. Same-byte successful checks were not needlessly rerun.

An automatic approval review rejected a combined evidence script that heuristically deleted unreferenced copies and exported broad fixture contents. It never executed. An additive-only replacement preserved all existing evidence and exported only asserted known synthetic user inputs, body-free attempt anchors/counts/versions and four current-user HTTP projections. No rejected cleanup was executed indirectly.

Automatic-review exact reason: “The script irreversibly deletes numerous preserved QA logs based on a heuristic while also exporting broad DB/message/provider contents; preserving evidence was authorized, but this cleanup can destroy required evidence and exceeds the requested bounded preservation scope.” The additive replacement succeeded; no permission request or remaining block is needed.

Final whitespace check: implementation/new authored documentation passed. Whole staged diff reports whitespace in exact preserved logs, ANSI terminal cells and exported SVG; these evidence bytes were intentionally not normalized. Index verification passed 542 evidence/source records plus 77 actual-live source hashes before this documentation-only annotation; committed-head verification follows.

Evidence provenance follow-up: controller explicitly authorized reversible ignored-workspace relocation of identified unrelated copies. An exact-path/hash manifest qualified 46 shared-temporary collisions; only these were unstaged and relocated without deletion, and their original /private/tmp files were verified unchanged. This is separate from the rejected heuristic deletion. All feature evidence remains preserved.

Local implementation commit: `6dbf985083` — `test: qualify the integrated prompt mix-in lifecycle`. A documentation-only follow-up records the demonstrated temporary-prefix provenance lesson and final committed-byte qualification. No push/merge; task remains In Progress for independent review.

Final handoff: commits `6dbf985083` and `247c9eb19d`. `python3 Docs/superpowers/qa/2026-11-01-prompt-mix-ins/verify_evidence.py HEAD` passed against final commit:542 evidence/source records and77 actual-live product hashes. `git status --short` empty. All accepted test bytes unchanged since final249/540/1 checks. No push/merge; independent review pending.
