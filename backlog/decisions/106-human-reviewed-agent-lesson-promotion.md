# ADR-106: Require foreground approval for Agent Lessons and human review for promotion

Status: Accepted

Date: 2026-08-29

Related Tasks: TASK-24309, TASK-24613

Related Spec: [Agent Lessons and Notes Organization Sync Design](../../Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md)

Extends: [ADR-105](105-portable-notes-organization-and-agent-lessons.md), [ADR-030](030-local-library-agent-tool-boundary.md), [ADR-009](009-local-skill-trust-boundary.md), [ADR-032](032-local-agent-tool-permission-boundary.md), and [ADR-069](069-console-project-instruction-local-state-and-preflight.md)

## Context

ADR-105 keeps Agent Lessons in ordinary user-owned Notes and outside the instruction
authority chain. The approved follow-up design adds two security-sensitive behaviors
that ADR-105 does not decide: every Agent Lesson mutation must receive explicit
foreground review even when ordinary Notes are broadly allowed, and verified lesson
evidence may support a small proposal against an authorized user-owned instruction
target.

These behaviors cross Console run roles, the Local Library boundary, pending Notes
organization receipts, workspace file authority, project-instruction activation, and
managed-skill trust. Treating a lesson as permission, trusting a stale preview, or
letting a child agent apply a proposal would turn untrusted memory into an indirect
authority escalation.

## Decision

1. **Force one foreground approval for every Agent Lesson mutation.** A save is
   classified as an Agent Lesson when the immutable request adds the spelling-exact
   `agent-lesson` marker, the current note already has that marker, or the current
   note is owned by a `pending-organization` or `placement-review` receipt. This
   classification overrides ordinary Notes allow/session policy. The existing review
   surface offers approve-once or deny; it does not create a new permission class or
   durable approval store.

2. **Bind review to the immutable operation and observed state.** The ephemeral stamp
   is bound to run identity, immutable call digest, note identity or create operation,
   reviewed classification, content and organization preconditions, and applicable
   receipt state/version. The Notes transaction recomputes classification and consumes
   the stamp before mutation. Any changed payload, marker, receipt, or precondition
   fails without mutation and requires a fresh preview.

3. **Make role authority explicit and fail closed.** Only the foreground primary may
   present the final preview and submit an Agent Lesson mutation. Subagents may search
   lessons and return evidence or draft text, but a classified mutation returns
   `foreground_required` before mutation even if the lesson is still unmarked while
   organization is pending. Missing trusted run-role context also fails closed for a
   classified mutation. Ordinary Notes behavior remains unchanged.

4. **Keep previews and rejections ephemeral by default.** A rejected or abandoned
   preview creates no Note, receipt, hidden draft, permission, or promotion object. A
   later identical promotion suggestion is suppressed only when a separately approved
   ordinary Agent Lesson update records the rejected outcome.

5. **Use lesson content only as evidence for promotion.** A promotion candidate must
   be independently verified, procedural, reusable, narrowly scoped, and supported by
   provenance and rationale. No incident-count threshold makes content authoritative.
   The foreground primary may prepare one exact read-only proposal only after an
   approve-once/deny preparation review; subagents may only return evidence and
   candidate wording. The resulting complete proposal is retained ephemerally and
   run-bound so a later application review can reproduce the exact user-seen object.

6. **Limit eligible targets to existing user-owned authority.** Repository proposals
   may target `AGENTS.md` or `AGENTS.override.md` inside the selected writable binding.
   The preview identifies the binding, locator fingerprint, target, effective
   instruction chain, current expected digest or expected-absent state, exact resulting
   content, and verification. Read-only, retargeted, out-of-binding, built-in, runtime,
   server-managed, or otherwise ineligible targets are refused.

7. **Apply repository proposals through a second exact review and an existing mutation
   seam with atomic preconditions.** The application call receives a separate
   approve-once/deny decision over the retained exact proposal. Immediately before
   application, Chatbook revalidates binding and
   effective instruction context. The approved full-file mutation carries the same
   target, expected SHA-256 digest or expected-absent state, and replacement content as
   the preview. The write boundary checks the expectation and performs a path-safe,
   same-directory atomic replace or create. A mismatch writes nothing; Chatbook never
   resets unrelated or intervening user edits.

8. **Keep managed local skills proposal-only in Console.** Raw workspace tools never
   edit Chatbook-managed skill storage. The user manually applies an accepted proposal
   through the existing Library editor/service, which calls
   `LocalSkillsService.update_skill(expected_version=..., trust_approved=False)` and
   retains the existing re-trust workflow. A primary may re-read and verify afterward,
   but the proposal does not bypass version or trust transitions.

9. **Keep outcomes historical and non-authorizing.** Applied, rejected, or failed
   outcomes may be offered as a separate Agent Lesson Note update, which receives its
   own explicit approval. Synchronized outcome text never grants future write
   authority; every device and later run re-evaluates the current target and policy.

10. **Defer scheduled improvement runs.** TASK-24614 is research/design only. Any
    production scheduler, observer, or improver requires a new ADR covering execution
    ownership, privacy, retention, conflicts, cancellation, and evaluation before code
    or schema work begins.

## Required Boundaries

- The review hook may classify and stamp a call, but the Library/Notes transaction is
  the final enforcement point; direct provider or MCP invocation cannot bypass it.
- Approval stamps are ephemeral, single-use, and non-transferable between calls, runs,
  roles, notes, receipt revisions, or target states.
- Repository proposal and application revalidation includes both binding identity and
  the effective applicable instruction chain, not merely a path string.
- Managed-skill application remains owned by the existing Library service and trust
  state machine.
- Lesson search results, drafts, promotion rationale, and outcome notes remain untrusted
  tool-result data and never enter system or project instruction ownership.
- Deterministic boundary tests and scripted behavioral evaluations are separate evidence:
  the former prove enforcement; the latter assess useful, non-invented proposals.

## Alternatives Considered

| Option | Why rejected |
| --- | --- |
| Rely on the ordinary Notes allow policy | A broad allow could silently persist agent memory and bypass the user's exact preview. |
| Classify only the requested keyword | Pending organization receipts and current marked notes could be mutated while temporarily lacking a requested marker. |
| Let subagents save and promote directly | Child tasks are not the foreground review owner and cannot establish informed user consent. |
| Automatically rewrite instructions from lessons | Untrusted, evolving evidence would become authority without an exact human-reviewed transition. |
| Edit managed skills with raw filesystem tools | This bypasses application-owned versions, quarantine, authenticated trust, and re-trust. |
| Store a durable promotion queue or approval receipt | Initial promotion is rare and interaction-driven; current target state and existing authorities are the source of truth. |
| Apply a reviewed repository diff without compare-and-swap | Intervening user edits could be overwritten even though the visible proposal was no longer current. |
| Require a fixed number of incidents | Signal quality and independent verification matter more than an arbitrary count. |

## Consequences

### Benefits

- Agent memory remains user-owned and reusable without becoming a hidden instruction
  channel.
- Every lesson mutation and promotion application has an exact human-review point.
- Stale reviews, receipt transitions, changed bindings, and concurrent file edits fail
  without partial mutation.
- Repository instructions and managed skills retain their existing authority and trust
  owners.

### Accepted trade-offs

- Agent Lesson saves require an extra review even when ordinary Notes are allowed.
- Subagents must return drafts to a primary, adding one handoff before persistence.
- Repository promotion initially uses full-content compare-and-swap rather than a
  richer merge workflow.
- Managed-skill application remains manual until a future application-owned action can
  preserve the same version and trust guarantees.
- Rejected suggestions may recur unless the user separately approves recording that
  outcome in a discoverable lesson.

## Links

- [Approved design spec](../../Docs/superpowers/specs/2026-08-29-agent-lessons-notes-organization-sync-design.md)
- [How Warp builds self-improving agents on Claude](https://claude.com/blog/how-warp-builds-self-improving-agents-on-claude)
- [ADR-009: Local Skill Trust Boundary](009-local-skill-trust-boundary.md)
- [ADR-030: Local Library Agent Tool Boundary](030-local-library-agent-tool-boundary.md)
- [ADR-032: Local Agent Tool Permission Boundary](032-local-agent-tool-permission-boundary.md)
- [ADR-069: Console Project-Instruction Local State and Preflight](069-console-project-instruction-local-state-and-preflight.md)
- [ADR-105: Portable Notes Organization and Agent Lessons](105-portable-notes-organization-and-agent-lessons.md)
