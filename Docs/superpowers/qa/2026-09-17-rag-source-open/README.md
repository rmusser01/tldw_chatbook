# Search/RAG source opening — TASK-4111

2026-09-17 UTC (2026-09-16 Pacific), on `feat/component-pattern-library`, based on
`aff8681508`. Recognized Media/Prompt result IDs now open the exact local record;
unresolvable identities give a visible warning without leaving the result card.

## Repair and scope

The result-open boundary accepts positive integer IDs, matching `media_N` /
`prompt_N`, matching hyphen forms, and matching `local:type:N` envelopes. It
passes canonical `local:media:N` or bare Prompt `N` to the existing reader route.
Server authority, mismatched source wrappers, malformed IDs and IDs changed by
display sanitization are refused. Opaque Notes/Conversation identities remain
unchanged. Display/citation IDs are retained.

Button and focused-card `o` share this validation. The shared deep-link dispatcher,
reader save vetoes, generation fences and existing load-error behavior remain in
place. A missing Media record shows its unavailable state. The real Prompt service
currently turns a missing record into its generic load-error/Retry surface; the
regression checks that visible recovery rather than claiming a not-found-specific
message. No CSS or design-token values changed.

ADR required: no new ADR. Existing ADRs
[003](../../../../backlog/decisions/003-settings-library-rag-defaults.md),
[084](../../../../backlog/decisions/084-library-media-reader-ia.md) and
[150](../../../../backlog/decisions/150-design-token-system-and-design-language.md)
govern the existing boundary and local reader ownership.

## Automated evidence

- [Route red](route-red.txt): before the production fix, six of eight button
  ID-shape cases failed; two already-supported paths passed. The `setup-attempt`
  log is an earlier fixture setup failure, not regression evidence.
- [Review red](review-red.txt): all 13 additional cases failed before the identity
  sanitation guard. `media_javascript:17`, `media_onclick=17` and `media_1\x007`
  had all become valid record 17 after display cleanup.
- [Final targeted run](final-targeted-tests.txt): **466 passed, 1 deselected**.
  Includes 66 identity cases, 24 mounted result-open cases, source-open routes for
  all four types, Search/RAG state and keyboard behavior, query-gate races, Media
  reader flows, Prompt dirty vetoes, design-token governance and CSS bundle sync.
- The deselected evidence-heading test remains TASK-15390. The earlier broad
  `-k open` probe also picked up two unrelated failures; both reproduce with the
  unchanged production modules from `aff8681508` in [baseline evidence](baseline-failures.txt):
  the Starter collection fixture lacks `active_authority`, and the Console handoff
  fixture lacks `get_conversation_metadata`. They are not included in this slice's
  final selection. The former is already catalogued in TASK-31249.
- The old focused-card Open test used the retired Media fixture contract. It now
  uses integer backing IDs and checks the canonical loaded reader and exact title.
  The neighboring button test checks the same contract.
- New test/runner files pass Ruff lint and format checks. Changed production
  ranges pass the formatter. [Lint comparison](lint-comparison.json) shows no new
  diagnostics in the four existing modified Python files; their baseline debt
  remains. `git diff --check` passes.

Final command:

```sh
.venv/bin/python -m pytest \
  Tests/Library/test_library_rag_open_identity.py \
  Tests/Library/test_library_rag_state.py \
  Tests/UI/test_library_rag_open_source_ids.py \
  Tests/UI/test_library_rag_query_gate_race.py \
  Tests/UI/test_library_rag_keystroke.py \
  Tests/UI/test_product_maturity_gate16_library_search_rag.py \
  Tests/UI/test_library_prompt_dirty_vetoes.py \
  Tests/UI/test_library_prompt_dirty_escape.py \
  Tests/UI/test_library_media_reader_flow.py \
  Tests/UI/test_design_token_governance.py \
  Tests/UI/test_css_bundle_sync_guard.py \
  Tests/UI/test_library_shell.py::test_library_shell_search_result_open_media_switches_to_viewer \
  Tests/UI/test_library_shell.py::test_library_shell_search_result_open_prompt_lands_in_editor \
  Tests/UI/test_library_shell.py::test_library_shell_search_result_open_note_lands_in_editor \
  Tests/UI/test_library_shell.py::test_library_shell_search_result_open_conversation_locates_owning_page \
  -q -k 'not test_evidence_heading_and_coverage_note_are_mode_aware_and_conditional'
```

Independent review found the sanitation collision, then reported no remaining
actionable findings after the fix and independently passing all 66 identity tests.

## Native evidence

[Runner](native_check.py), [results](result.json), [attempts](native-attempts.json)
and [lifecycle checks](lifecycle.json). The final private profile was
`/private/tmp/tldw-4111-run-004`; the app used the real `LinuxDriver`, owned tmux
TTY and exclusive instance lock. Both output streams stayed attached to the TTY.

Each cell performs real local keyword retrieval over real private Media and
Prompt databases. An explicit adapter changes only the returned ID spelling to
`media_N` / `prompt_N` and adds two malformed result rows. It asserts the real
retrieved IDs before changing them. The Media Open button is focused then
activated with Enter; the Prompt card uses `o`. Invalid-card `o` keeps query and
focus while painting a warning. Reader models and fields match exact stored IDs,
titles and bodies. All eight valid opens and eight refusals pass.

| Theme and size | Media | Prompt | Prompt body | Refusal |
| --- | --- | --- | --- | --- |
| Dark 170×48 | [View](auditdarkwide-media.svg) | [View](auditdarkwide-prompt.svg) | [View](auditdarkwide-prompt-body.svg) | [View](auditdarkwide-refusal.svg) |
| Dark 80×24 | [View](auditdarkcompact-media.svg) | [View](auditdarkcompact-prompt.svg) | [View](auditdarkcompact-prompt-body.svg) | [View](auditdarkcompact-refusal.svg) |
| Light 170×48 | [View](auditlightwide-media.svg) | [View](auditlightwide-prompt.svg) | [View](auditlightwide-prompt-body.svg) | [View](auditlightwide-refusal.svg) |
| Light 80×24 | [View](auditlightcompact-media.svg) | [View](auditlightcompact-prompt.svg) | [View](auditlightcompact-prompt-body.svg) | [View](auditlightcompact-refusal.svg) |

All 16 SVGs were rendered and visually inspected; [hashes](capture-hashes.json)
pin raw and stored evidence (trailing line whitespace normalized for git). Final captures wait for natural toast expiry before showing
readers. Exact Prompt body paint is captured separately after focusing its field.
The wide Prompt Items pane is still loading in the capture; this check qualifies
the loaded detail and return to Search, not the entire Prompt browse lifecycle.

All source snapshots match before/after. Ten private databases pass read-only
SQLite `quick_check` with the actual Canvas payload validator registered. The
default config/UI/runtime hashes match; zero conversation messages were created.
No error/critical/traceback log entries or faulthandler output. `app.run` returned,
exit code was 0, the exact PID was absent, and the owned terminal was closed.

Limits: programmatic field entry/focus followed by actual key activation, not a
full Tab-only or pointer journey. This uses keyword retrieval plus an ID adapter;
it does not qualify semantic retrieval, remote opening, provider calls, or every
Search/RAG action. No full test suite, push or merge was run.

Follow-up, 2026-09-17: [TASK-15390](../2026-09-17-rag-evidence-heading/README.md)
is closed. Its original child-order failure had already been corrected; the
exclusion above came from the open historical task, not a fresh reproduction on
this branch. A separate saved-dev profile-storage dependency is now removed from
the rendering test. The complete gate16 file passes without exclusions in that
follow-up's targeted run. The 466-test record above remains unchanged.
