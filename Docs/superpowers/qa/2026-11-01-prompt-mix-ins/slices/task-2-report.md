# Task 2 / TASK-38537 — portability and source contracts

Status: DONE_WITH_CONCERNS; task remains In Progress for independent review.
BASE: `72ec16000cb953c349ee2bf9a56b8ac7295b0ab1`.
Worktree: `/Users/macbook-dev/.codex/worktrees/prompt-mix-ins/tldw_chatbook` only.

## Result and scope

- Portable codec dispatches Mix-ins to record v2 and leaves Prompt/Recipe v1 and legacy meanings intact. Exact stored definition text, literal body, metadata and placement survive real Chatbook archive export/import. V2 requires a valid schema-3 text_mixin and matching compatibility projection; bad shapes, unknown versions, oversized exact definition text, and nested malformed JSON refuse before record mutation with bounded errors.
- Mix-in Markdown uses exactly `# Chatbook mix-in`, a blank line, and one fenced canonical JSON record line. All embedded newlines/headings remain escaped. Parser detects it before classic section parsing and invokes the same strict portable codec; malformed/unknown envelopes never fall back. Explicit mixin metadata in classic sections also refuses rather than flattening. Frozen pre-feature section-reader fixture sees no Prompt sections.
- Known server brief/detail/create/preview/version schemas preserve mixin type. Capability normalization accepts advertised mixin type and keeps only the actual advertised kind/version pairs. No endpoint or automatic activation was added.
- New read-only `PromptSourceCapabilities.mixin_snapshot_reads` is True locally and False for every current server source; the scope entry point enforces it. Actual client `get_prompt` is an unversioned identifier read, and `list_prompt_versions` is a version list with no advertised coherent UUID/revision-qualified capture contract. Server capture therefore always refuses explicitly, including source replacement and denied-policy scenarios, without a name read or local fallback. No live server support is claimed.
- Required `expected_version` updates refuse in PromptScopeService and both public server adapters because the existing shared update API cannot honor the fence; nested dict `request_data` is included. Ordinary unfenced Prompt/Recipe writes remain covered. Local UUID snapshots now include current name in the same SELECT/transaction; real rename coverage proves stable UUID, current version, current name and exact definition.
- The real new Mix-in archive regression exposed body text reused as accepted keyword metadata leaking from the shared KeywordStore creation/reuse logs. Shared add/undelete/reuse and write/membership-failure sinks now log fixed messages/categories; storage, validation and raised exceptions are unchanged. The existing keyword failure test now injects at the live shared owner rather than its obsolete host wrapper.
- Updated user export docs and recorded the concrete downstream-logging lesson. Implements existing ADR-286 and ADR-057; no new ADR required. AC 1–3 checked; status intentionally remains In Progress.

Files: `Prompt_Management/{prompt_chatbook_record,prompt_markdown_export,Prompts_Interop,prompt_source_capabilities,prompt_scope_service,server_prompt_service}.py`, `tldw_api/prompt_chatbook_schemas.py`, `DB/{Prompts_DB,keyword_store}.py`; the six named portability/source/schema test files; user guides `library/{prompts,import-and-export}.md`; task file; `backlog/docs/lessons-testing-evidence.md`. Existing Chatbook collection/import owners continue to use the shared codec without a second path. No server adapter conversion rewrite was needed; its existing mixin identity handling was verified by targeted tests.

## RED evidence

All commands used `/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python`, managed-worktree cwd, `-p no:cacheprovider`, and private platform-temp basetemps.

1. `-m pytest Tests/Prompt_Management/test_prompt_chatbook_record.py::test_mixin_chatbook_v2_preserves_exact_definition Tests/Library/test_prompt_export_roundtrip.py::test_mixin_markdown_exact_roundtrip_and_frozen_legacy_reader_refusal Tests/tldw_api/test_prompt_chatbook_schemas.py::test_all_server_prompt_schemas_preserve_mixin_identity Tests/Prompt_Management/test_prompt_scope_service.py::test_mixin_server_advertises_only_actual_kinds_but_cannot_capture_revisions Tests/Prompt_Management/test_prompt_scope_service.py::test_server_expected_version_never_silently_drops_write_precondition Tests/Prompt_Management/test_server_prompt_service.py::test_server_adapter_refuses_conditional_update_without_client_contract -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-red`: six failures. Codec/type, legacy Markdown rendering, Pydantic type rejection, capability downgrade and dropped adapter fence were expected; one scope fixture omitted required local_service and was corrected before implementation. Rerun of `...::test_server_expected_version_never_silently_drops_write_precondition` with `--basetemp=/private/tmp/mixin-task2-red-condition` then failed correctly with DID NOT RAISE.
2. `-m pytest Tests/Library/test_prompt_export_roundtrip.py::test_mixin_markdown_old_section_metadata_cannot_flatten_to_prompt Tests/Library/test_prompt_export_roundtrip.py::test_mixin_markdown_import_failure_does_not_log_body_or_exception Tests/Prompt_Management/test_prompt_chatbook_record.py::test_mixin_chatbook_rejects_oversized_exact_definition_text -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-red-privacy`: three expected failures: classic metadata fell back, generic import errors leaked the sentinel, and oversized exact JSON text was accepted after normalization.
3. Scope server-adapter test first used a non-awaitable Mock (fixture error). Corrected to AsyncMock; `/private/tmp/task2-adapter-red.py` temporarily removed only the new adapter guard, ran `...test_scope_server_adapter_refuses_conditional_update_without_client_contract --basetemp=/private/tmp/mixin-task2-red-scope-adapter-fixed`, and restored it in finally. Correct RED was DID NOT RAISE; GREEN counterpart passed.
4. `...::test_mixin_capture_uses_uuid_after_local_rename --basetemp=/private/tmp/mixin-task2-red-snapshot-name --tb=short`: after correcting required fixture arguments, failed with KeyError(name), proving the coherent snapshot lacked current name. Added only name to the snapshot SELECT; GREEN counterpart passed.
5. `...::test_mixin_archive_roundtrip_never_logs_literal_body_even_when_reused_as_keyword --basetemp=/private/tmp/mixin-task2-red-keyword-privacy --tb=short`: real archive export/import succeeded but body appeared in KeywordStore add/reuse logs. After sink sanitation, it and the shared keyword-store matrix pass.
6. `...::test_mixin_snapshot_read_capability_is_enforced --basetemp=/private/tmp/mixin-task2-red-read-capability --tb=short`: the disabled capability was ignored and execution reached the unavailable fake read method. Capability gate added; GREEN passed without reading.
7. `-m pytest Tests/Library/test_prompt_export_roundtrip.py::test_mixin_markdown_deeply_nested_json_refuses_with_bounded_error Tests/Prompt_Management/test_prompt_chatbook_record.py::test_mixin_chatbook_deeply_nested_definition_refuses_with_bounded_error -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-red-nesting2 --tb=short`: two RecursionErrors escaped. The initial depth 1100 did not exceed pytest's recursion limit and passed; depth 10000 gave correct RED. Wrappers now turn those malformed JSON failures into bounded record errors; GREEN counterpart passed.
8. `...::test_server_adapter_preserves_nested_request_version_precondition --basetemp=/private/tmp/mixin-task2-red-nested-fence --tb=short`: nested request_data fence was silently discarded (DID NOT RAISE). Guard added before schema conversion; final matrix covers it.

## Final GREEN evidence

```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_chatbook_record.py Tests/Prompt_Management/test_prompt_scope_service.py Tests/Prompt_Management/test_server_prompt_service.py Tests/Prompt_Management/test_prompt_mixins.py Tests/Prompt_Management/test_server_prompt_adapter.py Tests/Prompt_Management/test_local_prompt_service.py Tests/Library/test_prompt_export_roundtrip.py Tests/tldw_api/test_prompt_chatbook_schemas.py Tests/Chatbooks/test_chatbook_prompt_round_trip.py Tests/Prompts_DB/test_prompts_db_chatbook_export.py Tests/DB/test_keyword_store.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-final-targeted --deselect=Tests/Chatbooks/test_chatbook_prompt_round_trip.py::test_prompt_export_failure_is_atomic_bounded_and_private --tb=short
```

Result: **423 passed, 5 deselected in 8.76s**. No full sweep. Real archive/storage owners are exercised, not replaced by fakes. Separate focused GREEN for snapshot/source/privacy/shared-keyword boundaries: 12 passed. No product code changed after final GREEN; subsequent changes were task/docs only.

## Exact-BASE attribution and static checks

Prepared a read-only source copy using `git archive 72ec16000cb953c349ee2bf9a56b8ac7295b0ab1 tldw_chatbook Tests pyproject.toml packages | tar -x -C /private/tmp/mixin-task2-exact-base`.

Plain Chatbook suite initially failed all 19 constructor entries with `RecoveryRequired(raw_source_selection_changed)`: root fixture redirected config after source-bound imports. Exact BASE node `test_empty_prompt_selection_remains_a_noop_without_a_prompt_source` under `/private/tmp/mixin-task2-base-constructor` reproduced that failure. Added the already-supported `pytest.mark.bootstrap_profile` to this real-config-consuming suite, retaining the private collection-selected profile; product recovery behavior is untouched.

Ran exact BASE's complete Chatbook file with an external collection plugin applying the same existing marker (`PYTHONPATH=/private/tmp ... -m pytest Tests/Chatbooks/test_chatbook_prompt_round_trip.py -q -p no:cacheprovider -p task2_bootstrap --basetemp=/private/tmp/mixin-task2-base-chatbooks --tb=no`). Result: **12 passed, 7 failed**, matching the initial marked working-tree result. Five are `test_prompt_export_failure_is_atomic_bounded_and_private[{missing,database_init,database,encode,write}]`: asserting source ID string `1` is absent anywhere hits ordinary temp-profile paths. Those five remain unchanged and are the final five deselections. The other two were shared keyword value logging and a stale failure-injection target; because that owner affected new mix-ins, the real new-path regression and narrow log fixes above address them rather than skipping them.

Scoped Ruff check over all 15 changed Python files, with per-file `(diagnostic code, message)` multiplicities compared against exact BASE: **515 inherited findings on each; zero introduced**. Diagnostic JSONs are `/private/tmp/mixin-task2-ruff-{current,base}.json`.

`ruff format --no-cache --check` passes all 13 changed Python files except inherited formatter-debt owners `DB/Prompts_DB.py` and `Prompt_Management/prompt_scope_service.py`. Full formatter diff added/removed lines for each of those two are **identical to exact BASE** (not just counts); changed ranges are formatted. `git diff --check` passed. No unrelated lint/formatter debt was repaired.

## Self-review and concerns

Reviewed every changed hunk and direct codec/renderer/parser/server/source caller. Verified: explicit record type/version dispatch; v1 fixture unchanged; complete portable shape before mutation; exact definition not canonicalized on portable round trip; stale projections rejected; literal headings cannot form legacy sections; bounded import errors and downstream shared logging; no chat-selection field or side effect in imports; no name-based server read/local fallback; policy precedes snapshot access; advertised CRUD does not imply capture; both public adapters retain their interfaces and refuse version fences; exact UUID rename reads include name/version/definition in one transaction.

Concerns: five exact-BASE digit-sentinel tests remain red if run without deselection; 515 lint findings and two formatter-debt files are inherited unchanged. Server capture intentionally stays unavailable until the client has a truthful revision-qualified read contract. No live server, broad UI matrix or full-suite run was performed. No push/merge. No subagents or reviewers were dispatched.

Local commit: `4f33145297` (`feat: preserve mix-in portability and source capabilities`). Managed worktree clean after commit. Git repeated the pre-existing auto-gc/unreachable-loose-object warning; no cleanup was attempted. This report is an ignored local handoff artifact, matching the preceding slice report.

## Independent review round 1 — I1 BOM envelope fix

Review I1 was verified against the real parser and its `encoding="utf-8"` file reader. A leading U+FEFF survives file reading and is not whitespace to `str.lstrip()`, so the new envelope previously entered legacy raw-snippet logging (or imported appended legacy sections as an ordinary Prompt).

The parser now creates `envelope_content = content.removeprefix("\ufeff")` and uses that same envelope-only view for both discriminator and full framing validation. It removes only one optional leading transport BOM. The JSON record, including BOM characters inside the body/definition, is untouched; legacy parsing still receives its original input. No file-reader/error-handler change is needed because the parser now routes BOM envelopes through the existing strict codec and bounded record-error path.

Added eight cases across two parametrized tests in `Tests/Library/test_prompt_export_roundtrip.py`: valid, bad JSON, missing closing fence and appended classic legacy sections, each through the actual parser and actual UTF-8 file-import/SQLite path. The file test verifies EF BB BF bytes, valid stored exact body/definition/type (including the body-leading literal BOM), malformed bounded failure with zero active destination rows, no body-sentinel/raw-snippet log and no legacy parser log. No mock replaces parser, file reading, import mutation owner or storage.

RED:
```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_prompt_export_roundtrip.py::test_bom_mixin_markdown_never_falls_back_or_logs_literal_data Tests/Library/test_prompt_export_roundtrip.py::test_bom_mixin_utf8_file_import_preserves_or_refuses_before_mutation -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-review1-red --tb=short
```
Result: **8 failed**. Valid envelopes returned no record; malformed envelopes failed to raise bounded errors; bad JSON logged the literal sentinel in `Content snippet`; appended legacy sections were parsed/imported as an ordinary Prompt. Full relevant output retained in `/private/tmp/mixin-task2-review1-red.txt`.

An initial exact-text replacement asserted before editing product code because its escaped-regex context did not match. The covering command consequently remained RED (**8 failed, 106 passed**, basetemp `/private/tmp/mixin-task2-review1-green`). The edit was reapplied against the exact discriminator and framing lines; no unrelated code was changed.

GREEN:
```sh
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Library/test_prompt_export_roundtrip.py Tests/Prompt_Management/test_prompt_chatbook_record.py -q -p no:cacheprovider --basetemp=/private/tmp/mixin-task2-review1-green-fixed --tb=short
```
Result: **114 passed in 2.14s**. This covers the requested parser, real file import and strict codec without rerunning the 423-test matrix or inherited-failure nodes.

Scoped checks: `ruff format --no-cache --check Tests/Library/test_prompt_export_roundtrip.py tldw_chatbook/Prompt_Management/Prompts_Interop.py` passed (**2 files already formatted**); `git diff --check` passed. Ruff diagnostic code/message multiplicities across those two files are identical to exact review BASE `4f33145297d61846ba5c3f8ee03dd926f4e3e479`: **162 inherited findings on each, zero introduced/removed**. Exact-base source/layout was archived to `/private/tmp/mixin-task2-review1-base`; diagnostic artifacts are `/private/tmp/mixin-task2-review1-ruff-{current,base}.json`. A preliminary incomplete-layout static comparison produced package/import-classification noise; the full exact-base layout removed that noise. No inherited lint debt was repaired.

Self-review: checked that optional transport BOM normalization occurs before detection and is reused during framing validation; literal BOM/body bytes survive the JSON codec/storage; malformed BOM input cannot reach classic section parsing; current bounded ValueError file-import handling reports fixed record-error text without a traceback/raw body; ordinary legacy parsing remains unchanged. Existing ADR-286/057 govern this privacy-boundary bug fix; no new ADR. AC remain checked and task remains In Progress for scoped re-review.

No broad/full-suite, inherited-failure, live-server or UI runs. Earlier report qualifications (five inherited Chatbook assertions and broader static debt) remain in force; this scoped 114-pass result does not claim those were repaired. No agent/reviewer, push, merge, original-checkout product edit or recovery-fence bypass.

Review-fix local commit: `ffb0326c50` (`fix: recognize BOM-prefixed mix-in Markdown envelopes`). Managed worktree clean after commit. Git repeated its pre-existing auto-gc/unreachable-object warnings; no cleanup was attempted.
