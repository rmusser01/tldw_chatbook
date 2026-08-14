---
id: TASK-15676
title: Harden Moonshot Kimi and Z.ai GLM as first-class hosted providers
status: Done
assignee: []
created_date: '2026-08-12 20:45'
updated_date: '2026-08-14 00:45'
labels: []
dependencies:
  - TASK-15675
references:
  - Docs/superpowers/specs/2026-08-12-kimi-zai-hosted-chat-completions-design.md
  - >-
    Docs/superpowers/plans/2026-08-12-kimi-zai-hosted-chat-completions-implementation.md
  - backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Bring the existing Moonshot AI and Z.ai integrations up to the same first-class reliability, security, streaming, tooling, Settings, discovery, and resumability standard as the newest hosted providers, while establishing a reusable hosted Chat-Completions wire boundary.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Moonshot and Z.ai preserve their stable provider identities and public handler compatibility while using the neutral hosted Chat-Completions boundary; unrelated providers do not change behavior.
- [x] #2 Fresh configuration defaults use `kimi-k3` for Moonshot and `glm-5.2` for Z.ai while explicit historical selections remain usable.
- [x] #3 Explicit arguments, canonical configuration, environment credentials, defaults, and structural endpoint validation follow one documented fail-closed contract without mutating source configuration or disclosing secrets.
- [x] #4 Streaming and non-streaming calls strictly validate payloads, errors, finish states, usage, retries, cancellation, ownership, and size/depth bounds.
- [x] #5 Existing Chatbook function tools complete joined Console continuation for both providers; private reasoning uses TASK-15675 checkpoints and vendor built-in tools remain excluded.
- [x] #6 Kimi K3 preserves and budgets every retained assistant reasoning turn required by its always-on Preserved Thinking contract; other curated Kimi/GLM models follow their exact policies, with GLM using `clear_thinking=false` only for active/restored tool runs.
- [x] #7 Moonshot and best-effort Z.ai model discovery use the same normalized endpoint and credential resolution as chat, preserve prior cache on failure, and never log sensitive payloads.
- [x] #8 Canonical Settings exposes actionable readiness, save, search/focus, endpoint, credential, model, and reasoning guidance without an API-mode selector.
- [x] #9 QwenCloud Chat behavior remains unchanged after any shared parser extraction, proven by its complete contract suite and mutation checks.
- [x] #10 Documentation and optional doubly-gated isolated live tests cover the current defaults, endpoints, controls, tools, recovery, and no-paid-default contract.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
1. Establish focused baselines for the existing Moonshot, Z.ai, QwenCloud,
   Console native-tool, Settings, readiness, and model-discovery contracts.
2. Add the minimal provider-neutral hosted Chat-Completions URL, HTTP ownership,
   SSE framing, and OpenAI-shaped normalization boundary, preserving QwenCloud
   behavior through focused parity tests.
3. Migrate Moonshot/Kimi and Z.ai/GLM behind provider-local resolution, payload,
   finish, usage, and durable-reasoning policies while retaining public handler
   compatibility.
4. Carry frozen provider resolution, terminal metadata, private continuation,
   and usage through the existing Console/AgentService path, then prove joined
   function-tool continuation and cancellation through loopback HTTP.
5. Refresh defaults, readiness, canonical Settings, and model discovery without
   changing explicit historical selections or unrelated providers.
6. Update user documentation, add default-skipped isolated live-test gates, run
   only the focused provider-related verification authorized for this task,
   self-review each acceptance criterion, and record observed evidence.
7. Post-merge review correction: translate the hosted transport's private base
   URL validation exception into its documented redacted public error contract,
   align Z.ai's adapter-only retry-delay fallback with canonical Console/config
   policy, and prove both through focused regressions before re-closing the task.
8. Late compliance correction: give the Z.ai retry fallback one provider-local
   named owner and document the added regression callable, then resolve both
   PR #1614 rule threads through a final focused follow-up.

Detailed plan:
`Docs/superpowers/plans/2026-08-12-kimi-zai-hosted-chat-completions-implementation.md`

ADR required: yes

ADR path:
`backlog/decisions/063-hosted-provider-wire-and-durable-tool-continuation.md`

Reason: this task implements ADR-063's reusable hosted Chat wire boundary and
provider-specific durable continuation policies; no additional decision is
needed.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
- Added a provider-neutral hosted Chat-Completions HTTP/SSE boundary with
  bounded parsing, strict response normalization, retry/resource ownership, and
  redacted typed failures. Moonshot/Kimi and Z.ai/GLM now use provider-local
  resolution, request allowlists, model/reasoning policy, continuation
  translation, and compatibility wrappers; QwenCloud reuses only the proven
  Chat framing/normalization seam.
- Joined both providers to Console/AgentService function tools, terminal usage,
  cancellation, private TASK-15675 checkpoints, context budgeting, and recovery.
  Kimi K3 retains required historical reasoning; GLM retains thinking only for
  active/restored tool runs. Vendor built-in tools, Responses mode, provider
  session state, and schema changes remain out of scope.
- Refreshed fresh defaults, canonical Settings/readiness, normalized discovery,
  cache behavior, and user guidance. Added an isolated paid-live harness whose
  provider case requires both its exact opt-in flag and nonblank API key; default
  collection made no paid request. ADR-063 remains the governing decision and no
  additional ADR was required.
- Verification evidence: the focused hosted-provider/Qwen matrix passed 467
  tests; joined Console/native-tool/live-harness coverage passed 48 with the two
  paid cases skipped by their explicit gates; selected AgentService/gateway
  continuation seams passed 5 with 483 unrelated cases deselected. The Task 6
  Settings/readiness/catalog matrix passed 609 tests, with its focused post-audit
  matrix passing 349. The four provider modules pass MyPy, LLM modules compile,
  and cumulative diff checks pass. Ruff lint/format passes for all changed files
  except exact pre-feature debt reproduced in two legacy lint files (8 findings)
  and eight legacy formatter files; those unrelated files were not churned.
- Test-scope deviation: on 2026-08-13 the user explicitly stopped broad/full-suite
  execution and required only tests related to touched files or functionality.
  No full-repository or broad-directory result is claimed. The exact focused
  commands and this deviation are retained in the implementation plan/evidence.
- Post-merge review correction: Qodo's two late PR #1612 findings were reproduced
  and fixed in follow-up PR #1614. The shared hosted transport now converts its
  private base-URL validation exception into a context-free, redacted
  `ChatProviderError`, and Z.ai's absent-setting retry fallback is `5.0` seconds,
  matching canonical Console/config policy without changing explicit precedence.
  Both original review threads were replied to with the exact correction commit
  and resolved.
- Follow-up evidence: the two regressions failed before production edits and then
  passed; the complete hosted Chat and Z.ai modules passed 124 tests outside the
  localhost socket sandbox; four exact hosted catalog checks passed. Ruff lint
  and format, MyPy for both production modules, compileall, diff checks, and an
  independent code review were green. No broad suite was run under the user's
  related-test-only restriction. ADR-063 remains the governing decision; no new
  ADR was required.
- Late compliance correction: PR #1614's post-merge rule review now has one
  provider-local owner for the Z.ai retry fallback and a concise docstring on
  the added transport regression. The same two exact regressions and complete
  touched modules passed (`2` and `124` tests); focused Ruff lint/format, Z.ai
  MyPy/compileall, diff checks, and independent review were green. PR #1615's
  delayed review caught and closed one circular-test assertion: the test now
  independently pins the numeric `5.0` policy and the resolver's use of its
  named owner. Mutating the owner to `1.0` failed the exact test before it was
  restored; the restored test and static checks passed, and the review thread
  was replied to and resolved.
<!-- SECTION:NOTES:END -->
