# Feature Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver TASK-199 as PR 4 of 6: parse `{name}` placeholders with literal-brace escaping, collect shared values once across System/User lanes, and apply them safely through exact slash, picker, and Library Console flows.

**Architecture:** Put lexing/rendering and the typed application value in one Textual-independent module. Use one shared modal for all entry points. Extend the memory-only handoff store from bare text to a detached expiring application request, while keeping the Console composer/session as the only mutation authority. Never persist or log raw values.

**Tech Stack:** Python 3.11+ dataclasses/state machine, Textual modal/UI, monotonic time, pytest/property tests, Backlog.md CLI.

---

## Merge Gate and ADR

- Begin only after TASK-198 is merged into latest `origin/dev`; create a fresh worktree/branch.
- ADR required: yes.
- ADR path: allocate the next free number on latest `dev` as `backlog/decisions/NNN-prompt-variable-grammar-and-guarded-insertion.md`.
- Reason: this defines a durable grammar plus a cross-module memory-only Console application contract.

## File Responsibility Map

- Create `tldw_chatbook/Prompt_Management/prompt_variables.py`: lexer, specs, render plan, fingerprints, `PromptVariableApplication` validation.
- Create `tldw_chatbook/Widgets/Prompts/prompt_variables_dialog.py`: shared System/User dialog and typed decision.
- Modify `tldw_chatbook/UI/Navigation/pending_handoff_store.py`: typed detached expiring `CONSOLE_PROMPT_INSERT` value.
- Modify `tldw_chatbook/app.py`: stage typed applications instead of strings.
- Modify `tldw_chatbook/UI/Screens/chat_screen.py`: exact slash/picker replacement, Library append consumption, expiry/session/composer/System guards and transaction outcomes.
- Modify `tldw_chatbook/UI/Screens/library_screen.py`: launch shared dialog and stage guarded Library application.
- Modify Prompt picker/slash host modules only where the exact current path lives after TASK-198 merges.
- Add `Tests/Prompt_Management/test_prompt_variables.py`, `Tests/UI/test_prompt_variables_dialog.py`, and extend pending-handoff/Console/Library tests.
- Modify `Docs/User_Guide/library/prompts.md`, relevant Console guide, TASK-199, and the new ADR.

## Task 1: Refresh Baseline, Allocate ADR, and Expand the Task

- [ ] Confirm TASK-198 merge on `origin/dev`; create `codex/task-199-shared-prompt-variables` from it.
- [ ] Allocate the ADR number on this baseline and document the complete grammar truth table, limits, one-pass rendering, shared-lane values, optional System authorization, application destinations, expiry, fingerprint guards, no raw-value retention, and Recipe refusal.
- [ ] Mark TASK-199 In Progress and replace its collapsed criteria with the approved grammar/dialog/application/security outcomes. Link the exact ADR in its plan.
- [ ] Commit ADR + Backlog plan before code.

## Task 2: Implement the Pure Lexer with Red-Green Property Tests

- [ ] Write table tests for `{customer}`, duplicates, case sensitivity, adjacent variables, `{{`/`}}`, triple braces, nested-looking braces, unmatched braces, JSON/XML text, invalid names, empty values, and braces introduced by values.
- [ ] Add property tests over arbitrary brace/text input: extraction is deterministic; rendering never mutates input; escape decoding cannot reveal an inner placeholder; a rendered value is never reparsed.
- [ ] Add limit tests: at most 64 unique valid variables and name length at most 64; the first syntactically valid excess placeholder yields an explicit validation issue rather than truncation/literal fallback.
- [ ] Run red tests.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_variables.py -q
```

- [ ] Implement a single-pass lexer/state machine in `prompt_variables.py` that emits literal and variable tokens and preserves first occurrence across active System then User lanes.
- [ ] Implement rendering from a tokenized plan, not repeated regex substitution. Keep raw variable maps ephemeral to the caller.
- [ ] Run pure tests green.

## Task 3: Define and Validate the Typed Application Request

- [ ] Add tests for `PromptVariableApplication`: rendered lanes, lane flags, destination (`replace_snapshot` or `append_active`), target session, optional composer/System fingerprints, monotonic creation/expiry, nonempty applicable lane, and safe `repr`.
- [ ] Ensure raw values, source Prompt body, and separate value maps are absent from the dataclass and serialization/repr surface.
- [ ] Implement exact 120-second semantics: expired when elapsed is `>= 120`, with an injectable monotonic clock for deterministic tests.
- [ ] Add log-capture tests asserting secret values/rendered bodies never appear during validation, expiry, refusal, or failure.

## Task 4: Build One Shared Dialog

- [ ] Add UI tests for variables shown once with lane-use labels, blank defaults/blank allowed, scrollability up to 64, shared value across lanes, and markup-looking labels/content rendered literally.
- [ ] Add System checkbox tests: visible only when System exists, off by default, exact approved copy, active variable list recomputes, and ephemeral values for temporarily hidden variables reappear during the mounted dialog.
- [ ] Add destination-copy tests for exact `/prompt` replace, picker replace over ordinary text, and Library append. Do not label ordinary picker draft text as a slash command.
- [ ] Add Use original/Cancel/limit/no-active-lane tests. With System off and no User lane, both application actions are disabled and Cancel remains enabled.
- [ ] Implement `PromptVariablesDialog` as a view/controller over pure parser results. It emits rendered or use-original lanes and never owns Console mutation.
- [ ] Use existing Neon Workbench tokens and inspect narrow/large-variable layouts.

## Task 5: Harden the Pending Handoff Store

- [ ] Update `Tests/State/test_pending_handoff_store.py` red tests: bare strings rejected; typed application copied/detached; latest wins; one-shot claim; release retry only while valid; expiry at boundary; wrong type; safe repr; no value leakage.
- [ ] Extend `pending_handoff_store.py` to validate/copy `PromptVariableApplication` and inject/use monotonic time without weakening owner-thread or revision settlement rules.
- [ ] Check expiry when claiming. Preserve newer revisions when an older in-flight claim is released.
- [ ] Update `app.py::stage_console_prompt_insert` signature/docstring to accept the typed request and navigate only after successful staging.
- [ ] Update application-ownership and retirement tests that currently stage bare strings.

## Task 6: Integrate Exact Slash and Picker Replacement

- [ ] Add red `Tests/UI/test_console_command_composer.py` and `Tests/UI/test_console_prompt_picker.py` cases that capture the entire composer snapshot before opening the dialog and replace exactly that snapshot on Apply/Use original.
- [ ] Cancel must leave the command/draft intact. A changed composer fingerprint, session change, or expired request must apply neither lane and warn.
- [ ] When a resolved Prompt has no variables and no System lane, retain the direct replacement fast path through the same safe application helper. A System-only Prompt still opens authorization dialog.
- [ ] Ensure selecting a Recipe still enters the existing unsaved Prompt-copy path before variables can be applied.
- [ ] Refactor the two hosts to call one dialog-launch/helper and one Console application transaction rather than duplicate parsing/rendering.

## Task 7: Integrate Library Append and Optional System Replacement

- [ ] Add red `Tests/UI/test_library_prompts_canvas.py` and `Tests/UI/test_console_native_chat_flow.py` cases for Library append to the settled active draft, optional System replacement, shared values, use-original, expiry, latest wins, wrong session, stale System fingerprint, and transient missing composer.
- [ ] Library authorization captures target session and System fingerprint, but Console captures the active composer snapshot when consuming the append handoff.
- [ ] On transient missing composer, release only if still valid; expired/wrong-session/stale requests are acknowledged/discarded with a warning.
- [ ] Apply composer/System in-memory changes coordinately and reversibly. If durable System persistence fails after live update, report that separate failure honestly; do not claim an atomic disk rollback.
- [ ] Keep System checkbox default off and prevent a no-op application when no authorized lane exists.

## Task 8: Documentation and Verification

- [ ] Document grammar/escaping examples, limits, blank values, System authorization, Use original, destination behaviors, expiry, and non-persistence in Library and Console guides.
- [ ] Run focused/affected suites.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest Tests/Prompt_Management/test_prompt_variables.py Tests/State/test_pending_handoff_store.py Tests/UI/test_prompt_variables_dialog.py Tests/UI/test_console_command_composer.py Tests/UI/test_console_prompt_picker.py Tests/UI/test_console_native_chat_flow.py Tests/UI/test_library_prompts_canvas.py Tests/test_application_state_ownership.py Tests/ProductionApp/test_chat_composition_retirement.py -q
git diff --check
```

- [ ] Render and inspect shared dialog at narrow/normal sizes, with 0/1/64 variables, System toggled, limit error, and all three destination disclosures.
- [ ] Run the full suite, self-review for secret logging/stale mutation/parser ambiguity, request independent review, and address all valid findings.

```bash
/Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest
```

- [ ] Complete TASK-199 criteria/notes with ADR link and verification; mark Done only after DoD.
- [ ] Open one ready PR against `dev`, resolve CI/review, merge, and verify on `origin/dev`. Do not begin TASK-197 implementation before confirmation.
