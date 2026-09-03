# llama.cpp Lab-to-Console Connection Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Establish and accept the architectural contract that lets Lab verify a llama.cpp destination and hand it to the active Console session without conflating process liveness, API readiness, session adoption, or durable defaults.

**Architecture:** Create ADR-114 as a documentation-only decision. It extends the existing endpoint normalization, managed/external GGUF authority, and Console settings ownership decisions; it selects `http://127.0.0.1:8080` as the absent-value llama.cpp default, reserves `chatbook-llamacpp` as the safe API model alias for Lab-owned launches, defines a sanitized cross-surface connection descriptor, and assigns each lifecycle and persistence transition to one existing owner. Existing servers with path-identifying model IDs fail closed before handoff. No production behavior changes in TASK-31200.

**Tech Stack:** Markdown ADRs, Backlog.md, Python 3.11 application contracts, Textual 8.x UI terminology, OpenAI-compatible llama.cpp HTTP endpoints.

**Spec:** `Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md`

## Global Constraints

- TASK-31200 is documentation-only; do not edit Python, TCSS, TOML defaults, schemas, or tests.
- ADR required: yes.
- ADR path: `backlog/decisions/114-llamacpp-lab-console-connection-authority.md`.
- Reason: this work changes the provider/runtime boundary, cross-screen state ownership, persistence scope, privacy boundary, and long-lived setup flow.
- Preserve ADR-025 managed/external GGUF ownership and exact process-lease lifetime.
- Preserve ADR-095: ordinary Console session Apply does not persist endpoints; only the explicit full Settings default action may persist an explicitly edited checked endpoint.
- Preserve TASK-16476 behavior: adopting a local server never silently overwrites a different configured endpoint.
- Credentials, executable paths, GGUF paths, query strings, fragments, and raw logs never enter the cross-surface descriptor or conversation metadata.
- Lab-owned launch construction emits exactly one `--alias chatbook-llamacpp`; raw/expert arguments cannot replace or duplicate it.
- Existing-server model IDs with unambiguous filesystem markers fail closed without display, logging, or copy; an interior `/` alone remains valid for namespace-style IDs.
- The narrow cross-surface display allowlist does not prevent Lab from owning and rendering its executable/GGUF selection, PID, expert configuration, redacted command/arguments, or bounded sanitized diagnostics.
- Existing explicit user endpoints remain authoritative; the canonical `8080` choice applies only when no explicit or configured value exists.
- Keep this contract llama.cpp-specific until TASK-31201 proves it; do not introduce a universal local-runtime framework.

---

### Task 1: Author ADR-114 and freeze the connection vocabulary

**Files:**

- Create: `backlog/decisions/114-llamacpp-lab-console-connection-authority.md`
- Read: `backlog/decisions/002-openai-compatible-model-discovery.md`
- Read: `backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md`
- Read: `backlog/decisions/095-conversation-owned-console-generation-settings.md`
- Read: `tldw_chatbook/Chat/provider_endpoint_contract.py:14-194`
- Read: `tldw_chatbook/Event_Handlers/LLM_Management_Events/server_lifecycle.py:26-424`

**Interfaces:**

- Consumes: `resolve_provider_endpoint(provider: str, value: object) -> ProviderEndpointResolution`, `canonical_connection_identity(provider: str, value: object) -> tuple[str, str] | None`, ADR-025 GGUF authority, and ADR-095 Console session/default ownership.
- Produces: the normative `LlamaCppConnectionTarget`, readiness vocabulary, ownership table, default-resolution order, handoff actions, and privacy contract consumed by TASK-31201 through TASK-31206.

- [ ] **Step 1: Reconfirm that ADR number 114 is unclaimed**

Run:

```bash
git for-each-ref --format='%(refname)' refs/remotes/ | while IFS= read -r ref_name; do
  git ls-tree -r --name-only "$ref_name" backlog/decisions/
done | rg '/114-' || true

git worktree list --porcelain | sed -n 's/^worktree //p' | while IFS= read -r worktree_path; do
  find "$worktree_path/backlog/decisions" -maxdepth 1 -type f -name '114-*.md' -print 2>/dev/null
done
```

Expected: no matching ADR path. If another ADR-114 appears, rescan all refs and worktrees, select the next unused number, and update every path in this plan and TASK-31200 before writing.

- [ ] **Step 2: Create the ADR metadata and context**

Create `backlog/decisions/114-llamacpp-lab-console-connection-authority.md` with this opening structure:

```markdown
# ADR-114: Own llama.cpp process, readiness, Console adoption, and defaults separately

Status: Proposed
Date: 2026-09-03
Related Tasks: TASK-31200 through TASK-31206
Extends: ADR-002, ADR-025, ADR-095

## Decision

## Context

## Ownership and state transitions

## Default and compatibility policy

## Privacy and observability boundary

## Alternatives Considered

## Consequences

## Rollback plan

## Verification obligations

## Links
```

In Context, cite the exact current divergence:

- Lab launch fallback: `8001` in `llm_management_events.py`.
- configured provider and local discovery default: `8080` in `config.py` and `local_server_discovery.py`.
- Console direct-path fallback: `9099` in `console_session_settings.py`.
- subprocess liveness currently drives running presentation while stdout and stderr are discarded.
- the Lab command builder supplies `--model` without a reserved `--alias`, so llama.cpp's default `/v1/models` ID is the selected filesystem path.
- Console detected-server adoption already has safe session application and configured-endpoint preservation, but its default-persistence behavior is not the Lab handoff contract.
- TASK-16473 already requires a warning when an active Console endpoint will not survive restart; Lab handoff must retain that session-only truth rather than imply persistence.
- TASK-26837 records a provider-setup path that can report success without a durable `api_settings` entry; Make default must not inherit or normalize that false-success behavior.

- [ ] **Step 3: Define the sanitized connection descriptor**

Put this normative shape in the Decision section:

```text
LlamaCppConnectionTarget
  provider_key: canonical llama_cpp identity
  base_url: canonical credential-free persisted endpoint root
  model_id: exact non-path-identifying model ID returned by the verified endpoint
  runtime_owner: lab_process | external_server
  verification_generation: process-local opaque generation
```

State explicitly:

- `base_url` is produced by `resolve_provider_endpoint`; the descriptor never carries a chat-completions suffix.
- `model_id` is endpoint-reported identity, not a GGUF path, managed artifact path, or filename-derived global identity.
- every Lab-owned launch emits the stable reserved `--alias chatbook-llamacpp`, and readiness requires the endpoint to report that exact alias.
- raw arguments reject every separated or equals-attached `-a` and `--alias` form so they cannot replace or duplicate the reserved alias.
- existing-server IDs are rejected as path-identifying only for unambiguous filesystem markers: file URIs; absolute, explicit-relative, home-relative, drive-root, or UNC prefixes; backslashes; dot path segments; or a final `.gguf` component. An interior forward slash alone remains valid for `owner/model` identities.
- a rejected existing-server ID fails before selector, descriptor, display, copy, log, or adoption and produces only generic recovery to configure `llama-server --alias`.
- `runtime_owner` is informational lifecycle provenance, not authority for Console to stop a process.
- `verification_generation` exists only to reject stale probes and is never persisted.
- executable path, external GGUF path, and managed store path remain Lab-owned; credentials, raw command, and raw log output are excluded, with Lab retention/rendering governed by the narrow observability exception.

- [ ] **Step 4: Define distinct lifecycle and product states**

Document two layers rather than one overloaded running flag:

```text
Runtime truth: unclaimed -> reserved -> process_alive -> process_dead
Connection truth: unchecked -> checking -> api_healthy -> model_available -> stale_or_failed
Product state: not_configured | checking | starting | loading_model | api_ready | console_connected | needs_attention
```

Assign owners:

- `server_lifecycle.py` remains the exact claim/process owner.
- a TASK-31201 app-scoped llama.cpp connection owner will own the target plus generation-fenced HTTP/model evidence.
- Lab projects those two owners into the product state.
- Console owns whether the active session adopted the target.
- Settings/config owns durable provider defaults.

Specify that `process_alive` alone can render Starting or Loading model, never API ready.

- [ ] **Step 5: Choose the default-resolution and readiness contract**

Record this precedence:

```text
explicit user-entered or launch endpoint
  -> exact current-session target
  -> configured provider endpoint
  -> canonical llama.cpp absent-value default http://127.0.0.1:8080
```

Also record:

- existing explicit `8001`, `9099`, LAN, HTTPS, and reverse-proxy-prefix endpoints remain valid and are not migrated or rewritten.
- new Lab launch defaults to loopback `127.0.0.1:8080` only when the user has not supplied a value.
- API ready requires a successful health-compatible probe and successful `/v1/models` response containing the selected exact admissible model ID: the reserved alias for a Lab-owned launch or an accepted non-path-identifying ID for an existing server.
- a user-entered existing-server check is an explicit, exact-endpoint exception to ADR-002's configured-provider-only discovery rule; it does not enable ambient LAN scanning or background remote discovery.
- a port collision may offer Connect to it only after that exact endpoint passes the llama.cpp-compatible health and model checks.

- [ ] **Step 6: Define adoption and persistence actions**

Put this action table in the ADR:

| Action | Owner and effect | Persistence |
|---|---|---|
| Start on this computer | Lab reserves and owns one exact process claim | None |
| Connect to existing server | Lab verifies one user-entered endpoint | None |
| Use in Console | Console applies provider, exact model, and base URL to the active session | Process-local session only |
| Make default | Full Settings commit path applies its existing checked-endpoint rules | Explicit config mutation |
| Stop server | Lab stops only its exact owned process claim | Does not alter Console or defaults |

Require that Use in Console:

- never calls the detected-server path that auto-fills missing config.
- never persists `base_url` in conversation metadata, consistent with ADR-095.
- preserves a different configured endpoint and labels the adopted target Session only.
- refreshes Console readiness in the same application process.

Require that Make default:

- opens or delegates to the full Settings commit path rather than treating Lab verification as permission to write configuration.
- reports success only after the normalized provider endpoint is durably present in the configuration layer that restart resolution reads.
- retains TASK-16473's distinction between session-only and restart-safe endpoints and TASK-16476's protection against silently replacing a different configured endpoint.
- treats TASK-26837's missing-`api_settings` success state as an unresolved defect to prevent or surface, never as accepted behavior.

- [ ] **Step 7: Define settlement, privacy, and rollback**

State that every probe and handoff carries the exact verification generation. Cancellation, process death, target edit, model change, screen recomposition, or newer probe invalidates older evidence. A stale result cannot expose Use in Console or modify Console.

Define observability:

- the narrow display allowlist applies to cross-surface UI, Console, app-global metadata, and application/unrestricted logs: canonical provider identity and credential-free endpoint, accepted endpoint model ID, coarse lifecycle state, and bounded failure category.
- Lab may retain/render its own executable/GGUF selections, PID, expert configuration, redacted command/arguments, and bounded sanitized runtime diagnostics; user-entered arguments appear only in their owning editor and derived presentations are redacted.
- app-global state and logs must not retain raw executable/model paths, credentials, raw command arguments, or unbounded process output.
- TASK-31206 diagnostics and copy remain bounded, sanitized, and Lab-local; they do not enter the handoff descriptor, Console, conversation metadata, app-global metadata, or application logs.
- rejected path-identifying endpoint IDs are suppressed before every render, copy, or log projection.

Rollback rule: disable the new Lab handoff and fall back to existing Console Settings/discovery; retain existing explicit config and do not synthesize migrations.

- [ ] **Step 8: Record alternatives and consequences**

Include and reject at least these alternatives:

- treat process liveness as readiness.
- auto-write the Lab endpoint into provider configuration.
- reuse `_apply_detected_local_server` unchanged for Lab handoff.
- persist the complete Lab launch command in Console conversation metadata.
- retain three context-specific defaults.
- generalize all local runtimes before the llama.cpp path is proven.

Consequences must name the follow-on work: TASK-31201 implements the app-scoped connection owner and adoption; TASK-31202/31203 project it into onboarding; TASK-31204/31205 retain launch configuration separately; TASK-31206 adds Lab-local diagnostics.

- [ ] **Step 9: Run the ADR content check**

Run:

```bash
rg -n -- 'LlamaCppConnectionTarget|127\.0\.0\.1:8080|--alias|path-identifying|Use in Console|Make default|process_alive|model_available|verification_generation|ADR-002|ADR-025|ADR-095|TASK-16473|TASK-16476|TASK-26837' backlog/decisions/114-llamacpp-lab-console-connection-authority.md
```

Expected: every required term appears in its normative section; no requirement relies only on the Context narrative.

- [ ] **Step 10: Commit the ADR draft**

```bash
git add backlog/decisions/114-llamacpp-lab-console-connection-authority.md
git commit -m "docs: define llama.cpp Lab Console authority"
```

### Task 2: Index the ADR and connect every planning artifact

**Files:**

- Modify: `backlog/decisions/README.md`
- Modify: `Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md`
- Modify: `backlog/tasks/task-31200 - Define-the-llama.cpp-Lab-to-Console-connection-and-readiness-contract.md`

**Interfaces:**

- Consumes: proposed ADR-114 path and the existing TASK-31200 acceptance criteria.
- Produces: discoverable canonical decision links for later task plans and reviewers.

- [ ] **Step 1: Add ADR-114 to the canonical index**

Add one row to `backlog/decisions/README.md`:

```markdown
| [ADR-114](114-llamacpp-lab-console-connection-authority.md) | Proposed | Keep llama.cpp process ownership, HTTP/model readiness, Console session adoption, and durable provider defaults separate behind one sanitized connection target. |
```

Place it in numeric order after ADR-113. Do not renumber or repair historical duplicate ADR IDs in this task.

- [ ] **Step 2: Link the ADR from the wireframe brief**

Add an `## Architecture authority` section before Backlog mapping:

```markdown
## Architecture authority

- [ADR-114](../../../backlog/decisions/114-llamacpp-lab-console-connection-authority.md) owns endpoint defaults, readiness truth, cross-surface handoff, persistence scope, and privacy boundaries.
- [ADR-025](../../../backlog/decisions/025-shared-stt-artifacts-and-runtime-routing.md) continues to own Managed versus External GGUF authority and exact process-lifetime artifact leases.
- [ADR-095](../../../backlog/decisions/095-conversation-owned-console-generation-settings.md) continues to own Console session settings and explicit default persistence.
```

- [ ] **Step 3: Update TASK-31200 documentation metadata**

Use Backlog.md rather than hand-editing frontmatter:

```bash
backlog task edit 31200 \
  --doc backlog/decisions/114-llamacpp-lab-console-connection-authority.md \
  --doc Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md \
  --doc Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md \
  --plain
```

Expected: all three documentation paths render and the existing description, acceptance criteria, status, assignee, labels, and dependency list remain unchanged.

- [ ] **Step 4: Verify links and commit the index/brief update**

Run:

```bash
test -f backlog/decisions/114-llamacpp-lab-console-connection-authority.md
test -f Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md
test -f Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md
backlog task 31200 --plain
git diff --check
```

Expected: every command succeeds; TASK-31200 prints all three documentation paths.

```bash
git add backlog/decisions/README.md \
  Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md \
  'backlog/tasks/task-31200 - Define-the-llama.cpp-Lab-to-Console-connection-and-readiness-contract.md'
git commit -m "docs: link llama.cpp handoff contract"
```

### Task 3: Prove contract coverage and close TASK-31200

**Files:**

- Modify: `backlog/decisions/114-llamacpp-lab-console-connection-authority.md`
- Modify: `backlog/decisions/README.md`
- Modify: `backlog/tasks/task-31200 - Define-the-llama.cpp-Lab-to-Console-connection-and-readiness-contract.md`
- Read: `backlog/tasks/task-31201 - Bridge-a-verified-Lab-llama.cpp-runtime-into-the-active-Console-session.md`
- Read: `backlog/tasks/task-31202 - Guide-first-time-llama.cpp-setup-from-prerequisites-to-a-verified-Chatbook-response.md`
- Read: `backlog/tasks/task-31203 - Make-the-llama.cpp-setup-flow-visible-and-keyboard-efficient-at-narrow-widths.md`
- Read: `backlog/tasks/task-31204 - Separate-current-and-next-llama.cpp-launch-configuration-and-add-Restart-last.md`
- Read: `backlog/tasks/task-31205 - Add-durable-named-llama.cpp-launch-profiles.md`
- Read: `backlog/tasks/task-31206 - Add-bounded-llama.cpp-diagnostics-command-preview-and-recovery-actions.md`

**Interfaces:**

- Consumes: ADR-114 and all follow-on task acceptance criteria.
- Produces: an accepted, implementation-ready contract with explicit verification obligations and a fully closed TASK-31200 record.

- [ ] **Step 1: Add the follow-on verification matrix to ADR-114**

Before the matrix, require future launch-construction tests for exact reserved-alias
emission and raw-argument override prevention. Require existing-server tests that
reject path-identifying IDs before selector, descriptor, display, copy, log, or
adoption; prove the rejected sentinel is absent from every projection; and accept an
ordinary namespace ID containing one forward slash.

The `## Verification obligations` table must contain these rows:

| Contract | Required future evidence |
|---|---|
| Canonical endpoint | Pure normalization/default-precedence tests in `Tests/Chat/test_provider_endpoint_contract.py` and Console settings tests |
| Process versus readiness | Lifecycle plus real loopback HTTP tests proving live-process/not-ready and model-ready transitions |
| Stale-result fencing | Generation replacement tests for process exit, model edit, endpoint edit, cancellation, and recomposition |
| Console adoption | Mounted Lab-to-Console test proving exact provider/base URL/model apply without restart |
| Persistence boundary | Regression test proving Use in Console does not write config and Make default preserves unrelated or newer fields |
| Managed model privacy | Tests proving no filesystem path enters the descriptor, rendered authority text, app-global metadata, or Console settings |
| Compact UX | Production-stylesheet 80x24, 100x30, and 120x40 compositor/focus tests |
| Live qualification | Scratch-profile run against a real llama-server, with default-profile fingerprints checked before cleanup |

- [ ] **Step 2: Audit every TASK-31200 acceptance criterion against the ADR**

Run:

```bash
backlog task 31200 --plain
rg -n '^## |LlamaCppConnectionTarget|Runtime truth|Connection truth|Product state|Default and compatibility|Privacy and observability|Verification obligations' backlog/decisions/114-llamacpp-lab-console-connection-authority.md
```

Expected mapping:

1. Descriptor and default policy: Decision plus Default and compatibility policy.
2. Six distinct truth states: Ownership and state transitions.
3. Four explicit actions and no silent overwrite: action table.
4. Cross-surface privacy and the narrow Lab-owned diagnostics exception: Privacy and observability boundary.
5. Existing ADR/task reconciliation: Context, Decision, and Links.
6. Sequence, settlement, observability, verification: Ownership, Privacy, Rollback, and Verification obligations.

- [ ] **Step 3: Run the documentation integrity checks**

Run:

```bash
blocked_terms='T''BD|TO''DO|FIX''ME|implement lat''er|similar to Ta''sk'
rg -n "$blocked_terms" \
  backlog/decisions/114-llamacpp-lab-console-connection-authority.md \
  Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md \
  Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md

rg -n '[[:blank:]]+$' \
  backlog/decisions/114-llamacpp-lab-console-connection-authority.md \
  Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md \
  Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md

git diff --check
```

Expected: both `rg` commands return no matches and `git diff --check` succeeds. These are documentation checks; do not run the application test suite for TASK-31200.

- [ ] **Step 4: Promote the reviewed ADR from Proposed to Accepted**

After the coverage and integrity checks pass, change the ADR status:

```markdown
Status: Accepted
```

and change ADR-114's index status from `Proposed` to `Accepted`. Do not alter either decision summary.

Do not mark TASK-31200 Done yet.

- [ ] **Step 5: Check all TASK-31200 acceptance criteria and add Implementation Notes**

Use Backlog.md:

```bash
backlog task edit 31200 \
  --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-ac 4 --check-ac 5 --check-ac 6 \
  --notes "Accepted ADR-114 to separate llama.cpp process ownership, HTTP and model readiness, active Console adoption, and explicit provider-default persistence. Selected the absent-value loopback default at port 8080, reserved a stable path-independent alias for Lab launches, made path-identifying existing-server model IDs fail closed, and scoped the cross-surface privacy allowlist while preserving bounded Lab-local diagnostics. Preserved ADR-025 GGUF authority and ADR-095 settings ownership and linked the verification obligations for TASK-31201 through TASK-31206. Modified the canonical ADR index, handoff wireframe, plan, and TASK-31200 metadata; no production code or tests changed." \
  --plain
```

Expected: all six acceptance criteria are checked and the notes name the ADR decision, compatibility boundaries, modified artifacts, and documentation-only verification.

- [ ] **Step 6: Mark TASK-31200 Done and verify the rendered record**

```bash
backlog task edit 31200 -s Done --plain
backlog task 31200 --plain
```

Expected: status Done, six checked criteria, ADR-114 and both design/plan documents linked, and Implementation Notes present. If any field is missing, restore it before committing.

- [ ] **Step 7: Commit closeout metadata**

```bash
git add backlog/decisions/114-llamacpp-lab-console-connection-authority.md \
  backlog/decisions/README.md \
  Docs/superpowers/specs/2026-09-03-llamacpp-lab-console-handoff-wireframe.md \
  Docs/superpowers/plans/2026-09-03-task-31200-llamacpp-lab-console-contract.md \
  'backlog/tasks/task-31200 - Define-the-llama.cpp-Lab-to-Console-connection-and-readiness-contract.md'
git commit -m "docs: accept llama.cpp connection contract"
```

- [ ] **Step 8: Final scope verification**

Run:

```bash
base_sha=$(git merge-base origin/dev HEAD)
git diff --check "$base_sha..HEAD"
git diff --name-only "$base_sha..HEAD"
```

Expected changed paths are limited to ADR-114, the decisions index, the handoff wireframe, this plan, and TASK-31200. No Python, TCSS, TOML, schema, migration, test, or generated CSS file may appear.
