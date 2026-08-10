# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-199 as PR 4 of 6: parse `{name}` placeholders with literal-brace escaping, collect shared values once across System/User lanes, and apply them safely through exact slash, picker, and Library Console flows.

**Architecture:** Put lexing/rendering and the typed application value in one Textual-independent module. Use one shared modal for all entry points. Extend the memory-only handoff store from bare text to a detached expiring application request, while keeping the Console composer/session as the only mutation authority. Never persist or log raw values.

**Tech Stack:** Python 3.11+ dataclasses/state machine, Textual modal/UI, monotonic time, pytest/property tests, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after TASK-198 is merged into latest `origin/dev`; create a fresh worktree/branch.
- ADR required: yes.
- ADR path: [ADR-053: Prompt Variable Grammar and Guarded Insertion](../../backlog/decisions/053-prompt-variable-grammar-and-guarded-insertion.md).
- Reason: this defines a durable grammar plus a cross-module memory-only Console application contract.

## File Responsibility Map

- Create `tldw_chatbook/Prompt_Management/prompt_variables.py`: lexer, specs, render plan, fingerprints, `PromptVariableApplication` validation.
- Create `tldw_chatbook/Widgets/Prompts/prompt_variables_dialog.py`: shared System/User dialog and typed decision.
- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: typed detached expiring `CONSOLE_PROMPT_INSERT` value.
- Modify `tldw_chatbook/UI/Navigation/screen_state_store.py`: owner-thread sanitized Console target projection, detached restore, and runtime-coupled invalidation.
- Modify `tldw_chatbook/app.py`: publish/capture the sanitized target and stage typed applications instead of strings.
- Modify `tldw_chatbook/UI/Console_Modules/prompts.py`: own slash/picker replacement, Library append consumption, expiry/session/composer/System guards, and transaction outcomes.
- Modify `tldw_chatbook/UI/Console_Modules/wiring.py`: wire the current Console owner without duplicating application logic in the screen.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py` only for its existing composer DOM helper/delegation and sanitized live-target projection seam.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: launch shared dialog and stage guarded Library application.
- Modify Prompt picker/slash host modules only where the exact current path lives after TASK-198 merges.
- Add `Tests/Prompt_Management/test_prompt_variables.py`, `Tests/State/test_screen_state_store.py`, `Tests/UI/test_prompt_variables_dialog.py`, and extend pending-handoff/Console/Library tests.
- Modify `Docs/User_Guide/library/prompts.md`, relevant Console guide, TASK-199, and the new ADR.

## Task 1: Refresh Baseline, Allocate ADR, and Expand the Task

- [x] Confirm TASK-198 merge on `origin/dev`; create `codex/task-199-shared-prompt-variables` from it.
- [x] Allocate the ADR number on this baseline and document the complete grammar truth table, limits, one-pass rendering, shared-lane values, optional System authorization, application destinations, expiry, fingerprint guards, no raw-value retention, and Recipe refusal.
- [x] Mark TASK-199 In Progress and replace its collapsed criteria with the approved grammar/dialog/application/security outcomes. Link the exact ADR in its plan.
- [x] Commit ADR + Backlog plan before code.

## Task 2: Implement the Pure Lexer with Red-Green Property Tests

- [x] Write table tests for `{customer}`, duplicates, case sensitivity, adjacent variables, `{{`/`}}`, triple braces, nested-looking braces, unmatched braces, JSON/XML text, invalid names, empty values, and braces introduced by values.
- [x] Add property tests over arbitrary brace/text input: extraction is deterministic; rendering never mutates input; escape decoding cannot reveal an inner placeholder; a rendered value is never reparsed.
- [x] Add limit tests: at most 64 unique valid variables and name length at most 64; the first syntactically valid excess placeholder yields an explicit validation issue rather than truncation/literal fallback.
- [x] Run red tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_variables.py -q
```

- [x] Implement a single-pass lexer/state machine in `prompt_variables.py` that emits literal and variable tokens and preserves first occurrence across active System then User lanes.
- [x] Implement rendering from a tokenized plan, not repeated regex substitution. Keep raw variable maps ephemeral to the caller.
- [x] Run pure tests green.

## Task 3: Define and Validate the Typed Application Request

- [x] Add tests for `PromptVariableApplication`: rendered lanes, lane flags, destination (`replace_snapshot` or `append_active`), target session, optional composer/System fingerprints, monotonic creation/expiry, nonempty applicable lane, and safe `repr`.
- [x] Add tests for `ConsolePromptTargetProjection`: session ID plus one-way System fingerprint only, safe `repr`, owner-thread publication/restoration, runtime-coupled invalidation, and the bounded no-prior-target Library refusal.
- [x] Ensure raw values, a separate source-body copy, and separate value maps are absent from the dataclass. Allow exactly the selected final original-or-rendered lane payload, hide it from `repr`, and expose no feature-owned serialization or persistence API.
- [x] Implement exact 120-second semantics: expired when elapsed is `>= 120`, with an injectable monotonic clock for deterministic tests.
- [x] Add log-capture tests asserting secret values/rendered bodies never appear during validation, expiry, refusal, or failure.

## Task 4: Build One Shared Dialog

- [x] Add UI tests for variables shown once with lane-use labels, blank defaults/blank allowed, scrollability up to 64, shared value across lanes, and markup-looking labels/content rendered literally.
- [x] Add System checkbox tests: visible only when System exists, off by default, exact approved copy, active variable list recomputes, and ephemeral values for temporarily hidden variables reappear during the mounted dialog.
- [x] Add destination-copy tests for exact `/prompt` replace, picker replace over ordinary text, and Library append. Do not label ordinary picker draft text as a slash command.
- [x] Add Use original/Cancel/limit/no-active-lane tests. With System off and no User lane, both application actions are disabled and Cancel remains enabled.
- [x] Implement `PromptVariablesDialog` as a view/controller over pure parser results. It emits rendered or use-original lanes and never owns Console mutation.
- [x] Use existing Neon Workbench tokens and inspect narrow/large-variable layouts.

## Task 5: Harden the Pending Handoff Store

- [x] Update `Tests/State/test_pending_handoff_store.py` red tests: bare strings rejected; typed application copied/detached; latest wins; one-shot claim; `ready`/`expired` claim status; release retry only while valid; expiry at boundary; wrong type; safe repr; no value leakage.
- [x] Extend `pending_handoff_store.py` to validate/copy `PromptVariableApplication`, inject/use monotonic time, and surface claim-time expiry without weakening owner-thread or revision settlement rules.
- [x] An expired claim remains consumer-visible for one bounded warning, acknowledges once, and can never be released back to pending. Preserve newer revisions when an older ready in-flight claim is released.
- [x] Update `app.py::stage_console_prompt_insert` signature/docstring to accept the typed request and navigate only after successful staging.
- [x] Update application-ownership and retirement tests that currently stage bare strings.

## Task 6: Integrate Exact Slash and Picker Replacement

- [x] Add red `Tests/UI/test_console_command_composer.py` and `Tests/UI/test_console_prompt_picker.py` cases that capture the entire segment-aware composer snapshot at `/prompt` dispatch or picker opening, before any awaited resolution/selection, and replace exactly all snapshot draft segments on Apply/Use original.
- [x] Cancel must leave the command/draft intact. A changed composer fingerprint, session change, or expired request must apply neither lane and warn.
- [x] When a resolved Prompt has no variables and no System lane, retain the direct replacement fast path through the same safe application helper. A System-only Prompt still opens authorization dialog.
- [x] Ensure selecting a Recipe still enters the existing unsaved Prompt-copy path before variables can be applied.
- [x] Refactor `ConsolePromptsController`'s two hosts to call one dialog-launch/helper and one Console application transaction rather than duplicate parsing/rendering or adding a second ChatScreen-owned path.

## Task 7: Integrate Library Append and Optional System Replacement

- [x] Add red `Tests/UI/test_library_prompts_canvas.py` and `Tests/UI/test_console_native_chat_flow.py` cases for Library append to the settled active draft, optional System replacement, shared values, use-original, expiry, latest wins, wrong session, stale System fingerprint, and transient missing composer.
- [x] Library authorization captures target session and System fingerprint, but Console captures the active composer snapshot when consuming the append handoff.
- [x] The target comes only from the app-owned sanitized projection published with the Console screen snapshot. A runtime/source change invalidates both; no prior target refuses with `Open Console once, then retry Use in Console.` before the dialog opens.
- [x] On transient missing composer, release only if still valid; expired/wrong-session/stale requests are acknowledged/discarded with a warning.
- [x] Apply composer/System in-memory changes coordinately and reversibly. If durable System persistence fails after live update, report that separate failure honestly; do not claim an atomic disk rollback.
- [x] Keep System checkbox default off and prevent a no-op application when no authorized lane exists.

## Task 8: Documentation and Verification

- [x] Document grammar/escaping examples, limits, blank values, System authorization, Use original, destination behaviors, expiry, and non-persistence in Library and Console guides.
- [x] Run focused/affected suites.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_variables.py Tests/State/test_pending_handoff_store.py Tests/State/test_screen_state_store.py Tests/UI/test_prompt_variables_dialog.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_prompt_picker.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_library_prompts_canvas.py Tests/test_application_state_ownership.py Tests/ProductionApp/test_chat_composition_retirement.py -q
git diff --check
```

- [x] Render and inspect shared dialog at narrow/normal sizes, with 0/1/64 variables, System toggled, limit error, and all three destination disclosures.
- [x] Run the full suite, self-review for secret logging/stale mutation/parser ambiguity, request independent review, and address all valid findings.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [x] Complete TASK-199 criteria/notes with ADR link and verification; mark Done only after DoD.
- [ ] Open one ready PR against `dev`, resolve CI/review, merge, and verify on `origin/dev`. Do not begin TASK-197 implementation before confirmation.
