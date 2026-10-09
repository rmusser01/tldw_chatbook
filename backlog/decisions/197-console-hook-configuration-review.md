# ADR-197: Console hook configuration review and local consent

Date: 2026-09-27
Status: Accepted following written-spec review and requested audit (2026-09-27)
Amended by: [ADR-210](210-console-region-ownership.md) (accepted 2026-10-01)
Task: [TASK-33151](../tasks/task-33151%20-%20Design-Console-hook-settings-and-persistent-review.md)
Spec: [Console hook settings and review](../../Docs/superpowers/specs/2026-09-27-console-hook-settings-and-review-design.md)
Implementation: [TASK-33163](../tasks/task-33163%20-%20Add-Console-hook-settings-and-consent-review.md), [plan](../../Docs/superpowers/plans/2026-09-27-console-hook-settings-and-review.md)
Extends: [ADR-148](148-console-run-hooks.md)
Follows: [ADR-033](033-settings-commit-models-three-honestly-labeled.md),
[ADR-029](029-local-private-data-boundary.md),
[ADR-069](069-console-project-instruction-local-state-and-preflight.md), and
[ADR-150](150-design-token-system-and-design-language.md)

## Context

The existing user-scope hooks execute validated external argv commands through one
ConsoleRuntime engine, but have no consent store, guided Settings editor, or visible
permissions control. The user approved a native Console toolbar icon, a review
modal, canonical Settings editing, one-time review of existing hooks, and review of
new/changed hooks on the next Send. Approvals must survive restarts.

The existing Approvals chip belongs to tool requests, the composer has bounded
width, and Settings already owns category/detail/impact panes and config writes.
An editor-only or screen-only gate would leave shared/background execution open.

## Decision

1. Add Hooks beside Settings in the visible ConsoleControlBar, with an enabled
   pending count and access to current permissions. Use one native review modal
   with expandable exact definitions and a canonical Settings Hooks deep-link.
   Hook configuration uses Settings' staged Save/Revert contract.
2. Require explicit consent for every enabled user hook, including existing
   definitions. Automatic review happens before next manual Send/queue acceptance,
   preserving drafts on cancellation. No immediate discovery popup or grandfathered
   execution. Background submissions require the same consent without opening UI.
3. Scope each grant to the effective user config file, hook identity, and a
   versioned fingerprint of event, ordered argv, matcher, and timeout. Add optional
   stable IDs and per-hook enable flags while preserving legacy config parsing.
   IDs and enable switches never confer authority. Legacy identity is definition
   plus duplicate occurrence, never raw array position. Changes to a legacy
   duplicate group's size invalidate that group's grants rather than guessing
   which occurrence survived. Guided stable IDs remove this ambiguity.
4. Store observed metadata and current grants in a small private, atomic local
   JSON file owned by the shared runtime, separate from user TOML and MCP tool
   profiles. Protect its canonical live path with the existing sensitive-path
   exclusions. Missing/corrupt/unsupported state or persistence failure cannot
   permit hook execution. Settings retains the existing config writer, stale
   snapshot checks, and truthful post-replacement outcomes. Cross-process decisions
   re-read state under its lock; stale caches cannot restore revoked grants.
5. Enforce consent at shared admission and serialize the final definition/consent
   check with actual process creation against consent changes. Read an authoritative
   lossless config inventory; neither a stale app dictionary nor a parser's omitted
   invalid entries establishes clearance. Pending/invalid guards remain restrictive,
   and consent failure cannot inherit UserPromptSubmit's execution-error fail-open.
   Revocation/disable seal current-runtime admission before persistence; a write
   failure cannot reopen that seal or claim durable cross-process success. Pin
   notification targets at event admission, recheck them at launch, and preserve
   existing bounded queues. Already launched processes retain timeout/shutdown
   ownership. Cancelled modal generations cannot resume a Send or overwrite a
   newer revocation.
6. Preserve ADR-148's six-event, argv-only, user-scope, deny-only protocol and its
   execution-error semantics. Consent is an earlier admission requirement, not a
   tool permission grant or executable-content signature. Project instructions
   cannot approve hooks. Managed plugins and v2 effects remain governed by
   [ADR-162](162-managed-agent-plugins.md) and
   [ADR-163](163-expanded-console-hook-runtime.md), outside this implementation.

The exact local JSON/lock owner participates in ADR-126 admission and drain.
Nested consent work retains the outer config lease without widening its paths.
Backup discovery recognizes the permission file and lock as intentionally excluded;
this installed owner rejects restored payloads. Portable configuration therefore
requires fresh hook review and cannot import grant authority.

## Alternatives

| Alternative | Reason not chosen |
| --- | --- |
| Settings editor alone | Leaves configured commands running without the requested review. |
| Approval in the screen only | Direct, queued, recovered, and background execution could bypass it. |
| Popup immediately when definitions change | Interrupts work; the user explicitly chose next Send. |
| Approval flags in TOML or array-index grants | Configuration edits/reordering could manufacture or misapply authority. |
| Store hooks as synthetic MCP tools | Hooks execute at lifecycle events and have distinct scope and protocol. |
| Reuse the full managed-plugin trust system | Adds package authentication/lifecycle machinery to standalone user commands. |
| Sign script contents or kill active hooks on revoke | Expands this configuration-consent feature into executable integrity and process cancellation. |

## Consequences

Existing enabled hooks pause until their one-time review; reviewed unchanged
definitions remain approved across restarts. Invalid rows stay visible for repair.
Changes discovered during a run are checked again before launch, with guards
remaining restrictive and observers reporting skipped execution.

The grant store and config writer have separate atomic commits. Interruption can
require review again but cannot broaden permission. No database migration or new
dependency is required. This feature consents to exact command definitions;
same-path script/environment changes remain outside its detection boundary. Older
Chatbook versions and arbitrary same-user processes are outside this runtime's
enforcement boundary; this store follows the existing local permission-store model.

The linked spec defines modal actions, legacy identity migration, draft custody,
revocation limits, concurrency, and verification. The user reviewed the written spec
positively on 2026-09-27 and requested an audit before continuing; the resulting
clarifications preserve the approved layout and feature scope. ADR-148 remains
authoritative
for hook execution after consent; its accepted text is not rewritten.


## Current-dev integration: standalone v2 consent (TASK-32679)

The Console next-Send review and canonical F9 Hooks settings include standalone
`hooks.handler` v2 definitions as well as legacy `hooks.hook` entries. V2 consent
uses the existing app-owned HookPermissions store and grant epochs; its fingerprint
covers the complete normalized closed-schema definition, including event, effects,
arguments, environment, matcher, required policy and timeout. Legacy fingerprints
and grants remain unchanged. A rejected v2 batch stays visible and cannot be
approved or partially activated. V2 definitions have the schema's master switch,
without inventing a per-handler enable field.

Runtime admission reads the canonical saved configuration, rather than an app's
possibly stale dictionary. Configured v2 engines capture exact grants. The existing
permission owner serializes actual subprocess creation against config/consent
changes; revocation and changed definitions fence staged effects and queued work.
Host-injected engines retain their explicit authority resolver and never become
standalone config grants. Lifecycle session replacement remains idle-only.


### TASK-34406: admission must preserve observed consent history

Every stock admission observation retains the existing fresh saved-config and
durable permission reconciliation. Observing master disable, an individually
disabled hook or removal must retire old queued launch epochs; removal must
also retire the old durable grant before the same definition is restored.
An unpublished or newly constructed owner does not prove an empty permission
store: earlier owners and processes may have persisted consent.

The proposed admission-only no-store shortcut was rejected before commit or
push. Six actual native history controls paired the unchanged complete snapshot
with the proposed controller admission route: the original three passed and
all three shortcut routes retained old launch authority. Restoring the exact
original five production files made all six pass with source and native
retirement checked. The fresh zero-history work-count controls alone had not
covered this durable-history requirement.

Future cost reductions must independently preserve these observations, grants,
queued epochs, current source/owner boundaries and cancellation cleanup. Keep
the original full/custom/v2/launch/revoke routes and responsiveness limits.
ADR-126 custody, ADR-148 execution and ADR-162/163 v2 behavior remain unchanged.

The [TASK-33648 bounded disposal policy](163-expanded-console-hook-runtime.md#task-33648-saved-standalone-sessionend-during-host-disposal)
retains only an authentic host-issued, effect-free SessionEnd command notification
after ordinary admission closes. Its original grant is re-read and serialized
with actual process creation by this same permission owner; changed or revoked
authority still refuses, and no ordinary target or model authority reopens.
