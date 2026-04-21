# Chatbook Server Capability Parity Audit Design

Date: 2026-04-21
Status: Approved for spec review
Primary Repo: `tldw_chatbook`
Reference Repo: `tldw_server`
Supersedes: `Docs/superpowers/specs/2026-04-19-chatbook-server-parity-audit-design.md`

## Summary

`tldw_chatbook` should become a credible standalone client for `tldw_server` without losing its offline-first identity. That requires a capability-first audit that distinguishes:

- what must work locally without any server
- what must be operable against a remote server
- what is intentionally remote-only
- what server surfaces are backend-internal and should not drive client parity work

The earlier coarse audit model was directionally correct but too broad. It would blur distinct domains such as scheduler workflows vs chat workflows, local MCP runtime vs remote MCP governance, and read-it-later vs outputs/artifacts. This revised design splits those boundaries explicitly so parity can be scored and planned without ambiguity.

## Problem Statement

The current parity question is not "does Chatbook implement every server endpoint." The real question is:

1. Which `tldw_server` capabilities are client-relevant for a standalone single-user Chatbook that may connect to a multi-user server.
2. For each relevant capability, should Chatbook support it locally, remotely, or both.
3. For dual-surface domains, what is the intended operating model before sync exists.
4. Where does Chatbook already have usable dual-backend seams, and where is it still thin or absent.

If those distinctions are not made up front, the audit will overcount backend internals, undercount workflow gaps, and mix compatibility work with future sync design.

## Product Constraints

- `tldw_chatbook` remains a standalone application first.
- Local operations must continue to function without the server present.
- When a capability supports both local and remote operation, the primary UX should keep `Local` and `Server` explicit rather than mixing them by default.
- Mixed views may exist later as convenience overlays, but they are not the primary audit target and should not be assumed as the default UX.
- Sync, dual-write, and mirroring are future concerns. This spec only establishes which capabilities need compatible local and remote representations.
- `tldw_chatbook` is single-user, but it may connect to a multi-user `tldw_server`.
- Billing, admin, org, tenant, and ops surfaces are out of scope unless they directly affect client-visible compatibility.

## Goals

- Produce a server-wide, client-relevant capability map for `tldw_server`.
- Score current Chatbook coverage in a way that distinguishes local parity from remote operability.
- Identify missing or weak dual-backend seams in Chatbook.
- Define target operating policy for each domain:
  - local-first
  - remote-capable
  - remote-only acceptable
  - excluded
- Generate a rollout order that starts with the highest-value compatibility surfaces.

## Non-Goals

- Implement sync, replication, or dual-write.
- Force every server-internal service into the Chatbook UI.
- Rework stable UI surfaces only for style.
- Introduce mixed-source views as the default UX.
- Solve multi-user server governance beyond what a single-user client must discover, configure, trigger, or observe.

## Chosen Approach

The audit uses a capability-first model with an API-contract appendix and a runtime-policy overlay.

That means:

- The primary row unit is a client-relevant capability, not an endpoint and not a file.
- Each row records the server surface, Chatbook surface, action coverage, and the intended local/remote target.
- The audit remains code-first. Endpoints and service code outrank README quality when evidence conflicts.
- Runtime policy is first-class. Every row answers not only "is it present" but also "where should it live" and "how should it appear in Chatbook before sync exists."

## Audit Principles

### Client-Relevance Filter

Every server domain must first be classified as one of:

- `Local parity required`
- `Remote parity required`
- `Remote-only acceptable`
- `Local-only`
- `Excluded`

This prevents backend-internal server features from inflating the parity scope.

### Code-First Evidence Order

Evidence should be gathered in this order:

1. `tldw_server` endpoints
2. `tldw_server` core/service layer
3. `tldw_chatbook` API client and service adapters
4. `tldw_chatbook` UI surfaces
5. tests and docs

This matters because some server areas, especially writing and research, are more reliable in code than in prose documentation.

### Action-Level Coverage

Domain-level labels are not enough. Each capability row must separately score:

- `Browse`
- `Detail`
- `Create`
- `Update`
- `Delete`
- `Execute/Launch`
- `Observe/Status`

A surface that can only list and view data is not full parity.

### Runtime Policy Before Sync

For every dual-surface capability, the audit must record:

- write authority now
- read authority now
- whether Local and Server should stay separate in the UI
- whether mixed views are optional later
- whether the domain is a likely future mirror candidate

### Client Obligation Clarity

Requirement class alone is not precise enough for server-backed capabilities. Every row must also declare the Chatbook client obligation for that capability:

- `Full CRUD`
- `Discover / Configure / Trigger / Observe`
- `Discover / Trigger / Observe`
- `Observe-only`
- `Unavailable offline with explicit fallback`

This prevents remote-only or server-native capabilities from being treated as if they require full local implementation.

### Confidence Tracking

Every row should carry a confidence label:

- `High`
- `Medium`
- `Low`

This keeps doc-light domains from being treated as equally validated.

## Required Runtime Policies

These policies are already decided and should drive the audit.

### Primary UI Mode

- Local and Server should be explicitly separated for dual-surface domains.
- Mixed views are optional follow-on conveniences, not the baseline.
- Source badges and backend labels should be mandatory where both surfaces can appear.

### Authority Rules

- Local changes default to local persistence.
- Remote/server mode should operate against the server surface.
- Sync and mirrored writes are deferred to a later design.
- Remote-only capabilities must still be discoverable in Chatbook and must present explicit offline/unavailable UX when no server is configured.

### Special Domain Policies

- Notifications: Chatbook must retain local notification backing for local-only operation.
- MCP: Chatbook must retain local MCP/runtime backing for local-only operation.
- Workflows: remote-only is acceptable for this audit.
- Sharing: remote-only.
- Web Clipper: remote-only.
- Collections / Read-it-later: local parity required.
- Watchlists: local parity required.
- Writing Suite: local parity required.
- Research Sessions: local parity required.

## Domain Inventory For The Audit

The coarse domain list should be replaced with the following client-relevant split.

| Domain | Requirement Class | Notes |
| --- | --- | --- |
| Chat | Local parity required + Remote parity required | Core standalone and server-backed surface |
| Characters / Personas / CCP | Local parity required + Remote parity required | Includes launch, identity, discovery, runtime metadata |
| Notes / Workspaces | Local parity required + Remote parity required | Workspace boundaries remain explicit |
| Media / Reading / Ingestion Sources | Local parity required + Remote parity required | Includes reading progress and ingestion source CRUD |
| Prompts / Chatbooks | Local parity required + Remote parity required | Client should remain usable without server |
| Study Core | Local parity required + Remote parity required | Flashcards, quizzes, study guides |
| Study Packs | Remote parity required, local parity assessed explicitly | Contract maturity must be verified before scoring as a standalone parity target |
| Study Suggestions | Remote parity required, local parity assessed explicitly | Contract maturity must be verified before assigning implementation priority |
| Collections: Reading List / Read-it-later | Local parity required + Remote parity required | Separate from outputs/templates |
| Collections: Outputs / Templates / Artifacts | Remote parity required, local parity optional | Distinct from reading-list scope |
| Watchlists | Local parity required + Remote parity required | Must be cross-walked against existing local `Subscriptions` capability before severity is assigned |
| Writing Suite | Local parity required + Remote parity required | Must work standalone, server path audited separately |
| Research Sessions / Runs | Local parity required + Remote parity required | Session/run UX, not just providers |
| Research Search / Provider Surfaces | Remote parity required, local parity assessed explicitly | Separate from sessions/runs; contract maturity and client relevance must be verified first |
| Client Notifications | Local parity required | Chatbook-owned local eventing/status |
| Server Reminders / Notification Feeds | Remote parity required | Separate from local notifications |
| Workflows | Remote-only acceptable | General workflows |
| Scheduler Workflows | Remote-only acceptable | Scheduled orchestration surface |
| Chat Workflows | Remote-only acceptable | Chat-specific orchestration |
| Local MCP Runtime | Local parity required | Must work without server |
| Remote MCP Control Plane / Governance | Remote parity required | Discovery, config, status, governance |
| Sharing | Remote-only acceptable | No local parity target |
| Web Clipper | Remote-only acceptable | No local parity target |
| Evaluations | Local parity required + Remote parity required | Already has dual-surface signs |
| RAG / Embeddings / Chunking Admin | Local parity required + Remote parity required | Includes admin and config surfaces that affect retrieval |
| Cross-cutting Runtime Policy | Local parity required + Remote parity required | Mode switching, auth/session, source labels, future mirror policies |

## Matrix Row Schema

Each audit row should contain:

- `Domain`
- `Capability`
- `Requirement class`
- `Client obligation`
- `Local target state`
- `Remote target state`
- `Primary UI mode`
- `Browse coverage`
- `Detail coverage`
- `Create coverage`
- `Update coverage`
- `Delete coverage`
- `Execute/Launch coverage`
- `Observe/Status coverage`
- `Server evidence`
- `Chatbook evidence`
- `Verification evidence`
- `User-scope / tenancy fit`
- `Current gap summary`
- `Authority policy now`
- `Mirror/sync relevance later`
- `Confidence`
- `Recommended tranche`

## Scoring Model

Use weighted prioritization so the implementation order is defensible.

- User value: `0-5`, weight `5`
- Interoperability value: `0-5`, weight `5`
- Standalone value: `0-5`, weight `4`
- Architectural leverage / unblocker value: `0-5`, weight `4`
- UX urgency: `0-5`, weight `2`
- Risk / complexity: `0-5`, subtract weight `3`

Formula:

```text
priority = user*5 + interop*5 + standalone*4 + leverage*4 + ux*2 - risk*3
```

Interpretation:

- A high-value standalone + remote-compatible surface should rise quickly.
- Remote-only surfaces should not outrank core local-first domains unless they unlock multiple capabilities.
- Cosmetic UX work should not outrank missing CRUD or missing runtime control.

## Expected High-Confidence Findings

The audit should begin with these working hypotheses, then confirm or revise them with code evidence.

### Stronger Existing Dual-Backend Seams

Chatbook already appears to have real local/server structure in:

- notes and workspaces
- media / reading / ingestion sources
- prompts / chatbooks
- study core
- evaluations
- some RAG / chunking / embeddings administration

These domains are likely the best early candidates for tightening to full parity.

### Likely Thin Or Missing Client Surfaces

Chatbook appears thinner in:

- workflows
- scheduler workflows
- chat workflows
- server reminders / notification feeds
- sharing
- web clipper
- writing suite
- research sessions / runs
- remote MCP control plane / governance

These should be treated as likely gap-heavy rows from the start, not neutral unknowns.

### Areas Likely To Be Partially Covered Under Different Local Names

The audit should explicitly test whether existing Chatbook surfaces map onto newer server domains before scoring them as missing. The clearest current example is:

- local `Subscriptions` vs server `Watchlists`

This crosswalk matters because partial local coverage should change both the severity score and the recommended rollout tranche.

### Areas Requiring Code-First Audit

The audit should not rely heavily on docs alone for:

- writing
- research
- MCP unified

Those domains have broader or more detailed code surfaces than their README quality suggests.

## Required Deliverables

The audit should produce four concrete artifacts.

1. `Parity matrix`
   The master domain/capability crosswalk using the row schema above.

2. `Gap ledger`
   A flat list of missing or weak capabilities with evidence links, severity, and proposed target state.

3. `Target-state design`
   A domain-by-domain statement of where each capability should live:
   - local
   - remote
   - both
   - remote-only

4. `Execution roadmap`
   A tranche-based implementation order with clear prerequisites and verification expectations.

## Audit Passes

### Pass 1: Inventory `tldw_server`

- enumerate client-relevant domains from server endpoints
- split broad domains where server surfaces are materially distinct
- exclude billing, admin, org, and ops-only areas

### Pass 2: Inventory `tldw_chatbook`

- map API client coverage
- map local service coverage
- map UI surfaces
- identify duplicate or legacy entry points that can confuse parity scoring

### Pass 3: Build The Capability Matrix

- create one row per client-relevant capability
- record action coverage explicitly
- classify each row by requirement class and runtime policy

### Pass 4: Produce The Gap Ledger

- explain the concrete parity gap
- distinguish absent surface from partial CRUD from missing runtime policy
- mark confidence level

### Pass 5: Define Target State

- decide whether each row should be:
  - local
  - remote
  - both
  - remote-only
- specify primary Local/Server UI separation rules where applicable

## Post-Audit Planning Outputs

After the audit passes are complete, the implementation plan should convert the matrix and gap ledger into explicit tranches:

- tranche 0: cross-cutting runtime policy and client capability map
- tranche 1: strengthen already-partial dual-backend domains
- tranche 2: add missing high-value local/remote surfaces
- tranche 3: remote-only surfaces and follow-on conveniences

## Post-Audit Execution Rules

When implementation planning begins, use these rules:

- compatibility before sync
- explicit Local/Server operation before mixed views
- action completeness before polish
- foundational runtime policy before UX embellishment
- one capability vertical at a time unless two slices are truly independent

## Success Criteria

This design succeeds if it yields:

- a parity matrix that does not confuse local parity with remote operability
- a backlog that reflects the user-decided local vs remote policies
- a clear separation between client-relevant and backend-internal server features
- a roadmap that can move Chatbook toward standalone-client parity without prematurely solving sync

## Next Step

After user review of this written spec:

- write the implementation plan in `Docs/superpowers/plans/2026-04-21-chatbook-server-capability-parity-audit.md`
- execute the audit itself before any direct feature work
