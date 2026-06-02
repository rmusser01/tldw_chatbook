# ADR Workflow Design

Date: 2026-06-02
Status: Approved by user
Primary Repo: `tldw_chatbook`
Scope: Project workflow for Architecture Decision Records, Backlog.md task hygiene, AGENTS.md guidance, and future Superpowers generalization

## Summary

The project should revive ADR usage with a small canonical workflow that fits the existing Backlog.md source-of-truth model. New canonical ADRs live in `backlog/decisions/`. ADRs are required only for significant architectural decisions, not for every implementation task.

The first rollout adds lightweight documentation artifacts and project-specific agent instructions. It does not add automated validation or a new `CONTEXT.md` documentation layer. Existing ADR-like material stays where it is and is referenced from a historical index so contributors can find past reasoning without treating it as canonical under the new immutability rules.

## Spec Review

Local spec review passed on 2026-06-02 using the required completeness, consistency, clarity, scope, and YAGNI checklist. No implementation-planning blockers were found. External subagent review was not run because the available subagent tool is restricted to cases where the user explicitly asks for delegation.

## Current Context

The repository already has several decision-documentation patterns:

- `AGENTS.md` says project decisions live in `backlog/decisions/`, but that directory currently has no canonical ADR process or files.
- Several older design and development docs contain embedded ADR sections.
- Higgs TTS has separate ADR-like files in `docs/Development/TTS/`.
- Backlog tasks already require implementation plans, implementation notes, DoD checklists, and relevant documentation updates.
- Superpowers specs and plans already provide a design-to-plan-to-implementation workflow, but they do not consistently force an architectural decision check.

This means the project does not need a brand-new documentation system. It needs one canonical ADR home, a clear trigger rule, and a repeatable checklist that agents and humans can follow.

## Goals

- Establish `backlog/decisions/` as the canonical ADR location.
- Keep ADRs meaningful by requiring them only for significant architectural choices.
- Make ADR checks part of Backlog and Superpowers planning without adding brittle automation.
- Preserve discoverability of historical ADR-like docs without migrating them into the new canonical format.
- Keep the workflow easy enough that agents will actually follow it.
- Leave room to generalize the workflow into global Superpowers skills after it proves useful in this repo.

## Non-Goals

- Do not add an automated ADR validator or pre-commit hook in the first pass.
- Do not require ADRs for every Backlog task.
- Do not rewrite historical embedded ADRs as canonical ADRs.
- Do not add a new `CONTEXT.md` layer in the first pass.
- Do not modify global Superpowers skills until the project-specific workflow has been exercised on real tasks.

## Canonical ADR Location

Canonical ADRs live in `backlog/decisions/`.

Use numeric filenames:

```text
backlog/decisions/
  000-template.md
  001-console-native-chat-contract.md
  002-sync-v2-conflict-policy.md
```

`backlog/decisions/README.md` is the index and workflow guide. It lists accepted ADRs, superseded ADRs, and historical ADR-like documents outside the canonical ADR directory.

Existing design docs, Superpowers specs and plans, and Backlog tasks can continue to describe how work is done. Canonical ADRs explain why significant architectural choices were made.

## ADR Trigger Rule

An ADR is required when a task makes or changes a significant architectural choice.

Require an ADR for decisions involving:

- Storage, schema, migrations, sync, conflict policy, or data ownership.
- Provider/runtime boundaries, adapters, service contracts, or cross-module interfaces.
- Security, privacy, encryption, authentication, permissions, or data exposure.
- Dependency, framework, tooling, packaging, or runtime policy choices.
- Long-lived UX/application structure choices, such as navigation model or screen ownership.
- Decisions that reject a plausible alternative future contributors may ask about again.

Do not require an ADR for:

- Routine bug fixes.
- Small UI polish.
- Copy-only changes.
- Mechanical refactors that preserve existing boundaries.
- Test-only changes.
- Direct implementation of a decision already covered by an existing ADR.

Every implementation plan should include an ADR check:

```text
ADR required: yes/no
ADR path: backlog/decisions/NNN-short-title.md or N/A
Reason: brief explanation
```

If an existing ADR already covers the decision, link it instead of creating a duplicate.

## ADR Immutability And Supersession

Accepted ADRs are immutable except for typo fixes, link repairs, or metadata that does not alter the decision. If the decision changes, create a new ADR that supersedes the old one.

ADR status values:

- `Proposed`: drafted before or during design review.
- `Accepted`: active decision.
- `Superseded by ADR-NNN`: no longer active; retained for history.

When an ADR is superseded:

- The new ADR links the old ADR in a `Supersedes` field.
- The old ADR status is changed to `Superseded by ADR-NNN`.
- The README/index is updated.
- Relevant Backlog tasks and Superpowers plans link the new ADR.

## ADR Template

The canonical template should be stored as `backlog/decisions/000-template.md` and use this shape:

```markdown
# ADR-NNN: Short title

Status: Proposed | Accepted | Superseded by ADR-NNN
Date: YYYY-MM-DD
Related Task: backlog/tasks/task-N - Title.md
Supersedes: ADR-NNN or N/A

## Decision

One sentence stating what was decided.

## Context

Why this decision needed to be made, what problem it solves, and what constraints matter.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Alternative A | Reason |

## Consequences

What this decision means going forward, including accepted tradeoffs.

## Links

- Related task/spec/plan links.
```

The template is intentionally short. Long implementation plans belong in Backlog tasks or Superpowers plans, not ADRs.

## Backlog.md Integration

`AGENTS.md` should update the Backlog workflow with ADR-specific guidance:

- Before implementation planning, read relevant ADRs in `backlog/decisions/`.
- During implementation planning, decide whether the task needs a new ADR.
- If an ADR is required, create it before implementation begins and link it from the task Implementation Plan.
- If an existing ADR applies, link it instead of creating a duplicate.
- At task closeout, Implementation Notes list any ADRs created, superseded, or followed.
- DoD includes `ADR check completed`.

Backlog tasks should not embed architectural decisions that meet the ADR threshold without also linking the canonical ADR.

## Superpowers Integration

Project-specific Superpowers guidance should be added through repo docs and `AGENTS.md` first:

- Brainstorming specs include an `ADR Impact` section for significant design choices.
- Writing-plans output includes the ADR gate before implementation tasks.
- Plans reference relevant ADRs in their `Source Documents` section.
- Implementation plans do not proceed past planning when `ADR required: yes` has no ADR path.

The global Superpowers skills under `~/.codex/superpowers/skills/` should not be changed in the first implementation. After this workflow is used on several real tasks, the same language can be generalized into the global brainstorming and writing-plans skills if it proves useful.

## Historical ADR-Like Material

Existing ADR-like material should remain in place. The first rollout should add `backlog/decisions/historical-index.md` that links notable historical decision material, including:

- Separate Higgs TTS ADR files under `docs/Development/TTS/`.
- Embedded RAG ADR sections.
- Embedded subscriptions ADR sections.
- Embedded Chatbook, Chat, Worldbooks, diarization, and TTS decision sections.

The historical index must label these documents as pre-canonical. They are useful context, but they are not immutable canonical ADRs unless a future canonical ADR explicitly imports or supersedes a historical decision.

## Verification

The first implementation should verify:

- `AGENTS.md` clearly names `backlog/decisions/` as canonical.
- `000-template.md` can represent proposed, accepted, and superseded decisions.
- `README.md` explains when an ADR is required and when it is not.
- `historical-index.md` links existing ADR-like docs without rewriting them.
- Backlog task guidance includes an ADR check in planning and closeout.
- The design spec and implementation plan explain that no automation is included in the first pass.

No automated tests are required for the documentation-only first pass unless tooling is added later.

## Rollout

1. Add the ADR workflow docs/templates and update `AGENTS.md`.
2. Use the workflow on real future tasks.
3. After several tasks, decide whether to generalize the wording into global Superpowers skills.
4. Add automation only if checklist enforcement fails in practice.

## ADR Impact

This design creates a meta-decision about where canonical project ADRs live and when they are required. The implementation should create the first canonical ADR for this workflow itself, because it establishes a durable documentation and governance convention for the project.
