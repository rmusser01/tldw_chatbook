# Console Prompt Improvement Workbench — final QA evidence

Date: 2026-08-02

Client SHA: `b856795415cb8f8f6abf9eafeb2f73a7a6bae908`

Server compatibility SHA: `a6e289031ddbe531fab4983d7e5671a6a85292ed`

ADR: [ADR-040](../../../../backlog/decisions/040-versioned-prompt-artifacts-and-safe-improvement-transactions.md)

## Outcome

The prescribed client and server matrices, static smoke gates, isolated
real-app Textual scenarios, rendered-capture review, and content-log canary
audit passed. The repaired editor footer renders stacked System/User choices,
one primary Apply action, and one contextual Save menu at all three required
terminal sizes. Live assertions prove the checkbox glyphs and full labels are
painted, contained by the editor, and free of overlap with each other, the
action row, and the modal's Back/Close footer.

The reusable-prompt path now teaches its three starting points before opening
the editor. Outcome-first shows the four essential blocks first and keeps five
optional guidance blocks one keyboard action away. A successful Recipe save
names Library > Prompts and offers a direct Open Library handoff to the saved
artifact.

`Improve current draft…` is the first normal item in the existing Console
composer hamburger when a draft is available; `Browse Prompt Library…` follows
it and remains available with an empty composer. Neither is present in the
top/tab-bar actions or as an always-visible composer button. The idle composer
retains its existing Collapse / hamburger / Send / Mic controls.

## Environment and method

The deterministic runner [capture_qa.py](capture_qa.py) mounts the real
`TldwCli`, bundled stylesheet, real Console and Library screens, a real
isolated `PromptsDatabase`, and the production Prompt scope/improvement
services. Only the final provider/network boundary is replaced through the
supported dependency-injection seam. No external request or real credential
is used.

The run creates private temporary config and data roots matching
`<temporary-root>/console-prompt-*/`, then removes them in `finally`. It seeds
13 isolated records: legacy, foreign v1, block-v2 Prompt, block-v2 Recipe,
malformed, future-version, type/kind-mismatched, and six pagination fixtures.
The run used terminal sizes 140x40, 100x30, and 80x24.

The server compatibility worktree remained pinned to the SHA above. It was
not clean: these two unrelated pre-existing untracked templates were present
before and after verification and were not read, modified, staged, or
committed:

- `tldw_Server_API/Config_Files/templates/watchlists/cti_osint_report_markdown.md`
- `tldw_Server_API/Config_Files/templates/watchlists/news_briefing_markdown.md`

## Commands and automated evidence

### Client

```bash
.venv/bin/python -m pytest \
  Tests/Prompts_DB \
  Tests/Prompt_Management \
  Tests/Library/test_library_prompts_state.py \
  Tests/Library/test_prompt_export_roundtrip.py \
  Tests/UI/test_prompt_block_editor.py \
  Tests/UI/test_console_prompts_modal.py \
  Tests/UI/test_console_composer_improvement_transaction.py \
  Tests/UI/test_console_composer_menu.py \
  Tests/UI/test_console_workbench_contract.py \
  Tests/UI/test_console_workbench_parity_matrix.py \
  Tests/UI/test_library_prompts_canvas.py \
  Tests/Chat/test_console_provider_gateway.py \
  Tests/Chat/test_sensitive_llm_logging.py -q
```

Result: `943 passed, 2 warnings in 359.28s`; exit 0. The warnings were the
environment's existing Requests dependency-version warning and Python 3.12
`audioop` deprecation. The known conversation-browser stale-search timing
flake did not occur, so no node rerun was needed.

```bash
.venv/bin/python tldw_chatbook/css/build_css.py
.venv/bin/python -c "import tldw_chatbook.app"
.venv/bin/python -m compileall -q \
  tldw_chatbook/Prompt_Management \
  tldw_chatbook/Widgets/Prompts \
  tldw_chatbook/Widgets/Console
git diff --check
TLDW_QA_CAPTURE_STAGE=all \
  .venv/bin/python \
  Docs/superpowers/qa/console-prompt-improvement-2026-08/capture_qa.py
```

All five commands exited 0. The CSS build generated 457,922 characters; only
its timestamp changed, so the generated file was restored and not included in
the evidence commit. The final isolated full-app run seeded 13 records,
exercised every scenario below in one coherent pass, regenerated 25 SVGs and
the observation manifest, and exited 0.

The 2026-08-24 HCI UAT follow-up ran the changed editor/modal/controller tests
as a targeted matrix: `153 passed, 1 environment warning in 78.37s`. Its
current-dev `TLDW_QA_CAPTURE_STAGE=all` pass then regenerated 43 SVGs plus the
observation manifest, including the guided chooser and complete Recipe >
Library > Console round trip, and exited 0.

The configured Ruff and mypy tools were run across all 97 Python files touched
between feature base `2166a677562869796244a744346e213a75474ae6` and the client
SHA. They reproduce the repository baseline and are recorded honestly rather
than called green:

- Ruff check: 7 existing findings.
- Ruff format: 32 files would reformat; 65 already formatted.
- mypy 2.3.0: 213 existing errors in 30 files.

The QA runner itself passes Ruff check, Ruff format check, and `py_compile`.

### Server

Run from the pinned server compatibility worktree with the repository-level
virtual environment:

```bash
../../.venv/bin/python -m pytest \
  tldw_Server_API/tests/Prompt_Management \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_prompts_structured_api.py \
  tldw_Server_API/tests/Prompt_Management_NEW/integration/test_structured_prompt_search.py -q
```

Result: `160 passed, 2 warnings in 65.43s`; exit 0. The warnings cover the
legacy single-user API-key format and an isolated user-DB fallback used by the
test environment.

The plan's exact non-recursive Bandit command exited 0 with zero findings, but
warned that `structured_prompts` was skipped because it is a directory. The
same command with `-r` scanned 4,696 lines with zero findings at every severity
and confidence. Sixteen deliberate `# nosec` sites and 23 B608 no-failed-test
notices were reported; zero files were skipped. Server `git diff --check`
exited 0.

## Real-app observations

- Browse opens from `Prompts`, the first normal item in the composer hamburger,
  shows `Improve My Prompt`, uses one source at a time, and paginates 13 seeded
  rows as page 1 of 2. At every required size, the top/tab control row has no
  Prompts action and the idle composer retains its existing Collapse /
  hamburger / Send / Mic controls without a standalone Prompts button.
- The real Server source without an external service shows the explicit
  unavailable message and Retry action. Modern/old-server behavior is paired
  with the green adapter and server compatibility matrices; no live external
  Server Library was claimed.
- A legacy Prompt opens through its real normalized database row as conservative
  editable System/User blocks with the exact stored lane content and zero model
  calls. Foreign v1 remains read-only with explicit conversion.
- Malformed, future-version, and type/kind-mismatched structured records open
  through their real normalized rows in read-only compatibility state. Their
  stored records remain unchanged, `Convert and save as new` stays available,
  Update Original is blocked, and converted copies cannot Apply before save.
- The shared block editor supports content edits and reordering. Introducing an
  invalid XML tag disables Apply/Save; resolving it restores validity without
  replacing the unaffected widget or losing its cursor/focus. Narrow surfaces
  scroll while the modal footer remains reachable. At 140x40, 100x30, and
  80x24, the footer paints checkbox glyphs and the full
  `Replace this session's System prompt` / `Apply User` labels plus exactly one
  Apply action and one Save menu without clipping or overlap. Apply precedes
  Save in keyboard order, and `Ctrl+S` opens the native menu.
- `Let the improver read the current System prompt` is a request-only permission
  with explicit non-mutation disclosure. Included, excluded, and absent-System
  cases were checked across Auto, Review, and Recipe: request context changes as
  chosen, but the live System value does not. With no System prompt, the option
  is disabled and the improver receives only the unsent message.
- Auto no-change leaves the composer untouched. Auto success exposes the
  conditional hamburger Undo and restores the exact draft snapshot.
- The provider-unavailable Improve path performs resolution only, makes no
  auxiliary or send call, disables improvement actions, exposes actionable
  recovery copy, and opens the real Console provider/model settings. It is
  distinct from the Server Prompt-source unavailable state.
- A real inline-file composer segment is projected as an opaque token. When the
  provider candidate drops that token, the modal shows generic review-required
  copy, blocks Apply, and preserves the exact protected body/label and composer
  snapshot without exposing them to the provider.
- A delayed result made stale by live draft and System edits mounts an editable
  Review candidate with actionable stale copy. It retains both live edits,
  transcript, and attachments, performs no partial apply, and makes exactly one
  auxiliary call with no hidden retry. Cancellation still ignores a detached
  late result.
- Applying a reviewed structured Prompt leaves System unchecked by default and
  User checked. Explicitly opting into System applies the compiled System and
  User values atomically to the live session/composer, with no send/model call,
  transcript mutation, attachment mutation, or persistence requirement in the
  isolated non-persistent session.
- Recipe Fill uses one auxiliary call and remains mandatory review. At every
  size a valid non-empty `additional_context` becomes one mapped User-lane
  `Additional context` block; the composer remains byte-equivalent. Duplicate
  is disabled and Save as Recipe is omitted from the Save menu for that mapped
  block. Deleting it restores Recipe-save eligibility. System Apply remains an
  independent unchecked review choice.
- The Recipe chooser explains Outcome-first, Saved Recipe, and Blank before
  selection. Outcome-first initially shows Goal, Context and evidence,
  Constraints, and Output; Role, Personality, Collaboration style, Success
  criteria, and Stop rules remain keyboard-discoverable through one optional-
  block reveal action at 140x40, 100x30, and 80x24.
- A real UI round trip saves a Recipe from the direct Improve entry, confirms
  its Library > Prompts destination, deep-links to the new local Recipe, edits
  and version-bumps it with starter content, finds and fills it from Console,
  reviews the generated Prompt, and applies User without changing System or
  using the normal send stream.
- Library > Prompts labels Prompt and Recipe distinctly and shows the tested
  save-name conflict recovery.

## Privacy canary audit

The isolated run planted distinct random values in System content, User
content, Recipe block content, inline-file body, inline-file label, provider
response, and the opaque placeholder produced for protected inline content.
It attached a DEBUG sink only around the provider-boundary canary flow, read
the isolated log, and then deleted the profile.

| Category | Log count |
|---|---:|
| System | 0 |
| User | 0 |
| Block | 0 |
| Inline-file body | 0 |
| Inline-file label | 0 |
| Opaque placeholder | 0 |
| Provider response | 0 |

Permitted metadata is limited to provider, model, mode, duration, input/output
sizes, and typed outcome. The run made exactly one provider-boundary call in
the canary flow. A separate scan found no real home path, temporary profile
path, API key, authorization header, secret, password, traceback, internal
error, or dynamic content-canary value in any capture. The only path-shaped
metadata is the intentionally redacted temporary-root pattern in
[qa-observations.json](captures/qa-observations.json).

## Capture index

All captures are deterministic Textual SVG exports from the real app and were
visually inspected after rendering. There are 43 SVGs plus the observation
record.

| Surface | 140x40 | 100x30 | 80x24 |
|---|---|---|---|
| Composer hamburger | [capture](captures/140x40-composer-menu.svg) | [capture](captures/100x30-composer-menu.svg) | [capture](captures/80x24-composer-menu.svg) |
| Browse page 1 | [capture](captures/140x40-browse-page-1.svg) | [capture](captures/100x30-browse-page-1.svg) | [capture](captures/80x24-browse-page-1.svg) |
| Automatic replacement recovery | [capture](captures/140x40-auto-success-recovery.svg) | [capture](captures/100x30-auto-success-recovery.svg) | [capture](captures/80x24-auto-success-recovery.svg) |
| Before/after comparison | [capture](captures/140x40-auto-review-changes.svg) | [capture](captures/100x30-auto-review-changes.svg) | [capture](captures/80x24-auto-review-changes.svg) |
| System analysis choice | [capture](captures/140x40-system-analysis-choice.svg) | [capture](captures/100x30-system-analysis-choice.svg) | [capture](captures/80x24-system-analysis-choice.svg) |
| Recipe starting points | [capture](captures/140x40-recipe-chooser.svg) | [capture](captures/100x30-recipe-chooser.svg) | [capture](captures/80x24-recipe-chooser.svg) |
| Editable Recipe | [capture](captures/140x40-recipe-editor.svg) | [capture](captures/100x30-recipe-editor.svg) | [capture](captures/80x24-recipe-editor.svg) |
| Filled Prompt review | [capture](captures/140x40-filled-prompt-review.svg) | [capture](captures/100x30-filled-prompt-review.svg) | [capture](captures/80x24-filled-prompt-review.svg) |

Additional 140x40 states:

- [Server unavailable and Retry](captures/140x40-server-unavailable.svg)
- [Auto success with hamburger Undo](captures/140x40-auto-success-undo.svg)
- [Legacy Prompt as editable conservative blocks](captures/140x40-legacy-editable-blocks.svg)
- [Malformed structured compatibility guard](captures/140x40-malformed-compatibility.svg)
- [Block validation introduced](captures/140x40-block-validation.svg)
- [Optional System + User apply ready](captures/140x40-system-user-apply-ready.svg)
- [Optional System + User applied](captures/140x40-system-user-applied.svg)
- [Provider unavailable Improve recovery](captures/140x40-provider-unavailable-improve.svg)
- [Protected inline-file Review veto](captures/140x40-protected-inline-review-blocked.svg)
- [Stale delayed result in Review](captures/140x40-stale-result-review.svg)
- [Foreign-v1 compatibility guard](captures/140x40-foreign-v1-guard.svg)
- [Library Prompt/Recipe labels](captures/140x40-library-prompt-recipe-labels.svg)
- [Library save-name conflict](captures/140x40-library-save-conflict.svg)
- [Recipe saved with Open Library confirmation](captures/140x40-recipe-saved-confirmation.svg)
- [Saved Recipe reopened and edited in Library](captures/140x40-library-saved-recipe.svg)
- [Versioned Recipe filled, reviewed, and applied to Console](captures/140x40-recipe-roundtrip-applied.svg)

The one-shot Impeccable detector returned `[]` for the three inspected Python
UI targets. Two guessed component-CSS paths did not exist and were reported as
unavailable; no second detector run was substituted. Visual inspection found
no non-scrollable clipping, overlap, unreachable modal footer, missing
System/User control content, source-label ambiguity, or placement regression
in the published captures.

## Design sections 1-15 traceability

| Design section | Implementation evidence | Test/eval evidence |
|---|---|---|
| 1. Console entry point | `console_composer_bar.py`, `console_composer_menu_modal.py`, `chat_screen.py` | `test_console_composer_menu.py`, `test_console_workbench_contract.py`; menu captures at all sizes |
| 2. Unified modal navigation | `console_prompts_modal.py`, `console_prompts_state.py` | `test_console_prompts_modal.py`; real Close/Discard/Back lifecycle in `capture_qa.py` |
| 3. Browse and source search | `console_prompts_browse.py`, `prompt_scope_service.py`, `prompt_normalizers.py` | Prompt-management and modal suites; pagination, unavailable, and foreign-guard captures |
| 4. Structured artifact contract | `prompt_artifact_models.py`, `prompt_artifact_codec.py`, `prompt_block_compiler.py`, local/server migrations | codec/compiler/DB matrices and server structured API tests; v1 and `single_text_recipe` coexistence coverage |
| 5. Built-in and saved Recipes | `outcome_first_recipe()`, `library_prompts_state.py`, saved-Recipe modal path | Recipe service/modal/Library tests; guided chooser, save confirmation, Library reopen, editable, and Filled Prompt captures |
| 6. Block editor | `prompt_block_editor_state.py`, `prompt_block_editor.py` | `test_prompt_block_editor_state.py`, `test_prompt_block_editor.py`; live edit/reorder/validation/cursor-focus observations and all-size editor inspection |
| 7. Improvement modes | `console_prompt_improve_view.py`, `console_prompts_modal.py`, `prompt_improvement_service.py` | improvement-service/modal tests; live Auto, Review, Recipe, and provider-recovery observations |
| 8. Request boundary | `console_provider_gateway.py`, `prompt_improvement_prompts.py`, request-scoped sensitive policy | gateway and sensitive-logging tests; one-call live canary |
| 9. Preservation/context guards | `prompt_projection.py`, `prompt_preservation.py` | preservation and composer-transaction tests; real inline-file projection, token-veto, and exact segment-retention audit |
| 10. Concurrency/application | immutable request models, modal worker tokens, composer transaction coordinator | cancellation/stale/modal/composer tests; live delayed-result Review, cancel-late-discard, atomic optional-System apply, and exact Undo observations |
| 11. Saving/version/authority | `Prompts_DB.py`, `prompt_scope_service.py`, server adapter, Library state | DB/property/server/Library matrices; normalized identity and optimistic-conflict regressions |
| 12. Typed outcomes/errors | `prompt_improvement_models.py`, fail-closed service outcomes, modal status/retry states | malformed/no-change/provider/preservation/context-limit tests; live compatibility, provider-unavailable, stale, and Review captures |
| 13. Privacy/observability | `sensitive_logging.py`, gateway `ContextVar` propagation, metadata-only telemetry | registry-parity and provider canary tests plus zero-count isolated log audit |
| 14. Testing/quality gates | trusted optimizer fixture, prompting eval cases, deterministic `capture_qa.py` | 943 client tests, 160 server tests, static smoke gates, 43 inspected captures and explicit observation manifest |
| 15. Delivery boundaries | composer menu ownership, ADR-040, stage Backlog tasks and SDD reports | workbench parity tests; this evidence archive and TASK-1777 closeout |

## Explicit completion checks and limitations

- Schema-v1 and server `single_text_recipe` v2 remain independently dispatched.
- Old servers remain browsable as Prompt-only; exact v2 save capability is
  required before structured persistence.
- Undo invalidates on the specified composer/session/provider and transaction
  drift events; exact snapshots are restored only while the guard remains live.
- Improvement performs at most one provider call and has no hidden repair,
  fallback, or retry completion.
- Context budgets fail closed without silent truncation.
- System persistence failure keeps the live applied System value and offers a
  persistence-only Retry with the documented honest status.
- No real external Server Browse session was available; only its real local
  unavailable/Retry UX and automated compatibility paths were verified.
- The repository-wide Ruff/format/mypy debt above predates this feature and was
  not mechanically rewritten during the scoped closeout.

ADR required for Task 13: no. ADR-040 remains the governing record; this task
adds verification evidence and makes no storage, authority, provider-boundary,
security, dependency, or cross-module architecture decision.
