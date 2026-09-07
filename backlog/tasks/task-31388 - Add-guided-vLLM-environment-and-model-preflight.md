---
id: TASK-31388
title: Add guided vLLM environment and model preflight
status: Done
assignee:
  - '@codex'
created_date: '2026-09-03 22:32'
updated_date: '2026-09-04 15:25'
labels:
  - vllm
  - lab
  - onboarding
dependencies:
  - TASK-31387
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Prevent predictable first-run vLLM launch failures by making environment, model-source, network, and managed-argument readiness visible before Start.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [x] #1 Users can select either a Hugging Face repository ID or a local model directory through source-appropriate controls.
- [x] #2 Preflight reports interpreter resolution, vLLM module availability, model-source validity, port availability, and bind-address exposure without starting a server.
- [x] #3 Start is disabled with a visible field-adjacent reason until required checks pass.
- [x] #4 Managed host, port, and model flags cannot be duplicated or overridden through raw arguments.
- [x] #5 Focused unit and mounted Textual tests cover success, failure, preservation, and recovery states.
- [x] #6 Guided readiness remains a persistent four-row Environment, vLLM installation, Model, and Network checklist with row-specific status, bounded adjacent recovery, resolved versions on success, Python-environment Browse, and a Cancel check action bound to the current worker generation.
- [x] #7 Advanced exposes editable typed dtype, tensor parallel size, maximum model length, GPU memory utilization, and trust-remote-code controls with adjacent validation/consequence copy; raw arguments remain under a separate nested Advanced arguments disclosure.
- [x] #8 Existing-server setup shows credential-source status without values, discovers only bounded admissible model IDs after an explicit Check connection, requires explicit model selection, and does not publish readiness until the selected exact model is re-probed for the current generation.
- [x] #9 Selecting a saved local profile immediately reports missing Python or local-model prerequisites beside the visible source-appropriate field without running version, port, launch, or network probes; existing-server mode clearly disables local-profile mutations.
- [x] #10 Existing-server mode disables the saved-profile selector, removes it from keyboard traversal, ignores forged/programmatic selection and mutation events without repository changes, and restores profile selection after an explicit return to Local mode.
<!-- AC:END -->

## Implementation Plan

<!-- SECTION:PLAN:BEGIN -->
Source: .superpowers/sdd/2026-09-03-vllm-lab-console-complete-redesign/task-1-brief.md
ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This task directly implements the accepted runtime and UX boundaries.

1. Add pure vLLM launch/preflight contract tests first, then record their expected RED result.
2. Implement immutable contracts, validation, bounded preflight, and public CLI command construction.
3. Replace the inline pane with VllmSetupView and add mounted workflow tests.
4. Run the specified focused GREEN suites and incumbent deferred-view checks.
5. Check acceptance criteria, record evidence and no-ADR rationale, mark the task Done, and commit Task 1 files.

Task 6 Fix Round 2:
6. Add RED boundary tests covering canonical underscore/equals/abbreviated long options, credential/config options, every structured managed option, negative boolean aliases, and malformed direct `VllmLaunchDraft` construction.
7. Canonicalize and reject conflicting raw options at the command-construction boundary, then enforce exact enum/type/range validation for every draft field without weakening the existing public CLI contract.
8. Run each focused RED/GREEN node sequentially, then the complete vLLM setup/core matrix and static/diff gates; append exact evidence before restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This review round enforces ADR-117's existing launch-input and privacy boundary without changing ownership or architecture.

Task 6 Fix Round 3:
9. Confirm the reported short option against the official vLLM serve contract
   and the local launch boundary, then add RED tests for spaced/equals `-tp`, a
   forged successful preflight, and an allowed non-managed raw option.
10. Add the smallest short-alias-to-managed-option mapping in the shared raw
    validator; retain immediate command-builder revalidation and prove the
    allowed control is not overblocked.
11. Run focused setup RED/GREEN, the touched core suite, static/inventory/diff
    gates, and append exact evidence before restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: This extends ADR-117's existing managed-argument boundary to one
official spelling of an already-owned structured option.

Final UX fix round:
12. Add sequential RED mounted tests for the persistent four-row checklist, field-adjacent recovery/version copy, established Python-environment picker, current-generation Cancel check, typed Advanced controls, nested raw arguments, and profile-repair adjacency.
13. Add sequential RED pure/mounted tests for bounded external discovery, credential-source status, explicit model selection, empty/missing/changed lists, and exact selected-model reprobe before readiness publication.
14. Implement the smallest view/controller and connection-result changes that satisfy those contracts without changing the established launch, persistence, credential, or handoff security boundaries.
15. Run each focused node RED then GREEN, the complete vLLM primary and compatibility matrices, responsive geometry, generated-CSS, inventory, static, privacy, and diff gates before checking the new ACs and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already requires the persistent guided setup, structured expert fields, bounded explicit discovery, user-selected existing-server model, and exact current-generation reprobe; this round implements those accepted outcomes.

UX Fix Round 2/5:
16. Add RED pure and mounted tests for immediate profile repair validation, source-aware focus/copy, and truthful existing-server profile actions without invoking runtime or network probes.
17. Add the smallest synchronous repair-only validator and project its bounded issue through the existing checklist/adjacent-control seams; disable local-profile mutation controls outside local mode.
18. Run each focused RED/GREEN node, then the primary, geometry, compatibility, CSS, inventory, privacy, static, and diff gates before checking the new AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already requires selected missing environment/local paths to remain selected as Needs setup with repair, and defines profiles as local-launch convenience.

UX Fix Round 3/5:
19. Add a RED mounted regression proving existing-server mode removes saved-profile selection from the reachable action set and that forged selection/mutation messages preserve repository bytes and revision; prove Local mode restores the selector.
20. Guard the view and controller at the authoritative mode boundary and project the disabled selector through the existing truthful profile-action copy.
21. Run focused workflow and geometry GREEN checks, then the complete requested primary, compatibility, CSS, static, privacy, inventory, and diff gates before checking the AC and restoring Done.

ADR required: no
ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md
Reason: ADR-117 already defines saved profiles as local-launch convenience and existing-server setup as non-persistent connection state; this round enforces that accepted boundary.
<!-- SECTION:PLAN:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Implemented the immutable vLLM launch/preflight contracts and focused deferred setup view. Local starts resolve the matching public vllm CLI, reject managed/secret raw argument overrides, use a source-specific local-directory picker, reserve the existing server lifecycle claim, and never perform readiness or Console adoption.

Tests: /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Management/test_vllm_setup.py Tests/UI/test_vllm_lab_workflow.py -k "preflight or initial or mode or command or source" (19 passed, 10 deselected); /Users/macbook-dev/Documents/GitHub/tldw_chatbook/.venv/bin/python -m pytest -q Tests/LLM_Management/test_gguf_server_sources.py Tests/UI/test_llm_deferred_views.py (78 passed); full focused vLLM files (29 passed); compileall and git diff --check passed.

ADR required: no. Existing ADR path: backlog/decisions/117-vllm-lab-console-readiness-and-profiles.md. Reason: directly implements accepted runtime and UX boundaries.

Modified: tldw_chatbook/UI/LLM_Management/{__init__.py,vllm_setup.py,vllm_setup_view.py}, tldw_chatbook/UI/LLM_Management_Window.py, tldw_chatbook/Event_Handlers/LLM_Management_Events/llm_management_events_vllm.py, and focused tests.

Task 6 integration fix round: preflight issues now project bounded human recovery
copy beside the owning visible control instead of exposing internal field keys in
the aggregate blocker. Raw-argument failures open Advanced options so their help
is actually visible; Start remains disabled and Retry remains the focused recovery
action. IPv6 wildcard availability now binds the requested `::` address and lets
the platform apply its dual-stack policy instead of substituting `::1`; client URLs
still normalize wildcard binds to loopback. Focused TDD evidence: the ordinary
field-help node went RED for a missing adjacent label then GREEN (`1 passed`); its
collapsed Advanced assertion went RED then GREEN (`1 passed`); the deterministic
socket double went RED on `('::1', 8000)` then the setup/URL nodes went GREEN
(`3 passed`). A macOS real-bind comparison with an IPv4 listener held open observed
the same result from a direct `::` bind and `is_port_available("::", port)`.

Task 6 Fix Round 2 closes the runtime's direct-caller boundary. Raw long
options are normalized across underscore/hyphen and `--name=value` spellings,
then fail closed for config/credential flags, every structured launch owner,
the negative trust alias, and protected abbreviations. Command construction
repeats this check, so forged successful preflight state cannot place a
credential or config path in argv. Preflight and command construction also
require exact mode/source enums, exact string/bool/integer/float types, a
finite bounded GPU fraction, supported dtype, and the existing field ranges;
bool-as-int and malformed non-UI drafts settle without invoking probes,
resolvers, or sockets. Raw-boundary RED was `24 failed, 6 passed, 29
deselected`; GREEN was `30 passed, 29 deselected`, plus `3 passed, 79
deselected` for the final builder canaries. Structured-validation RED was `21
failed, 59 deselected`; GREEN was `28 passed, 52 deselected`. No new ADR:
ADR-117 already owns the launch-input boundary.

Task 6 Fix Round 3 verifies the reviewer claim against the
[official vLLM `serve` reference](https://docs.vllm.ai/en/stable/cli/serve/),
which documents `--tensor-parallel-size, -tp`; this
checkout's shared environment has no installed `vllm` executable, so no local
version-specific help contract was available. The shared raw validator now
maps exact short option `-tp` (including `-tp=value`) to the already-protected
canonical tensor-parallel option. The command builder's existing immediate
revalidation therefore closes the same forged-preflight path without another
policy list. RED was `3 failed, 1 passed, 82 deselected`: both short spellings
and forged command construction escaped, while `--enable-prefix-caching`
remained allowed. GREEN was `4 passed, 82 deselected`. Final setup was `86
passed`; setup/profile/lifecycle was `173 passed`; focused Ruff, `py_compile`,
diagnostic/profile inventories, and `git diff --check` passed. No new ADR or
lesson: this is one official spelling of the structured option already owned
by ADR-117.

The final UX fix round makes readiness continuously legible instead of
collapsing it into one global result. Environment, vLLM installation, Model,
and Network remain visible as four independently projected rows; successful
local checks include bounded Python and vLLM versions, failures recover beside
their owning field, and the established picker updates the Python-environment
field. Cancel check carries the rendered generation and can cancel only that
generation's live worker. Advanced now owns editable typed dtype, tensor
parallel size, maximum model length, GPU memory utilization, and
trust-remote-code controls, with consequence/validation copy beside each
field; raw arguments are a separate nested disclosure. Existing-server Check
connection publishes only a bounded admissible candidate list, never an
implicit READY target; the user must choose a returned ID and complete a fresh
exact-generation probe. The focused new nodes were first observed RED for the
missing checklist projection, typed controls, picker/cancel behavior, and
discovery-selection contract, then GREEN after the view/controller and bounded
probe-result changes. No new ADR: this is the guided setup and explicit
selection flow already accepted by ADR-117.

Final shared qualification for this round: the setup/connection/profile and
mounted workflow/geometry primary passed `308` tests in `428.25s` with no
descriptor-growth warning; the production CSS build/sync/staleness gate passed
`39`; format, critical Ruff, `py_compile`, both profile/diagnostic inventories,
and `git diff --check` passed. The host still has neither a `vllm` executable
nor an importable `vllm` package, so no live server was downloaded or launched.

UX Fix Round 2/5 adds a synchronous repair-only profile check at selection. It
validates draft shape, source/model, local-directory shape, and Python
resolution without invoking version, port, launch, or network probes; the
four-row checklist leaves the deferred installation and network rows honestly
`not checked`. Missing Python and local-directory mounted cases project Needs
attention beside the visible owning input. Existing-server mode disables the
five local-profile mutation actions and explains how to return to local mode.
Semantically unchanged profile refreshes preserve stronger completed preflight
evidence instead of replacing it with repair-only proof; the mounted regression
was RED before that retention guard and GREEN afterward. The pure repair pair
and focused mounted profile cases passed, and the shared final primary passed
`325` tests under the normal descriptor threshold. ADR-117
already owns this repair and local-profile boundary; no new ADR or lesson was
required.

UX Fix Round 3/5 makes the saved-profile selector follow the same Local-only
ownership rule as the mutation actions: Existing server disables it and removes
it from Tab traversal, while an explicit return to Local restores it. Both the
view event and every screen mutation/select handler recheck the authoritative
screen draft, so forged programmatic messages schedule no repository work or
dialog and preserve exact document bytes and revision. The view/controller
regressions were observed RED before the guards and GREEN afterward.
Final Round 3 qualification passed the `60`-case workflow, `71`-case geometry,
and `329`-case five-file primary gates under the normal descriptor limit.
Final combined-review hardening reuses ADR-117 and adds one shared host-only bind
rule (IP literals, including IPv6, or case-insensitive localhost) across draft
preflight and profile validation. Existing-server URLs are bounded at 2,048
code points and malformed URL/parser inputs settle as field-local issues before
any resolver, socket, or HTTP work. Focused bind/URL and mounted settlement
regressions pass; no new ADR is required because this directly enforces the
accepted preflight boundary.

Post-PR review hardening gives the default-version probe one owner-enforced
monotonic loop over platform-safe output transports: a nonblocking descriptor
on POSIX and the repository's local-only `PIPE_NOWAIT` named pipe on Windows.
There is no reader thread or synchronous anonymous-pipe readiness call for an
inherited writer to strand. Every oversize, timeout, read, process-wait, spawn,
and transport failure terminates, kills when needed, reaps, and closes without
retaining output or exception text. Ten focused cases cover the real
cross-platform subprocess path, the exact 256/257-byte boundary, quiet timeout,
read/wait failure, simulated Windows transport ownership, inherited writers,
and thread/handle cleanup. The same round adds a shared strict lexical boundary
for every vLLM draft control and raw arguments, preserving exact partial edits
while rejecting bounded invalid events without echo. ADR-117 already governs
these preflight and privacy boundaries, so no new ADR or contract change was
required.
<!-- SECTION:NOTES:END -->

## Renumbering provenance

This task previously held id TASK-31214. During the branch integration sweep,
current `origin/dev` already shipped `task-31214 -
Prevent-main-PyPI-workflow-from-publishing-stale-versions.md` at add commit
`3d2d5403e2994a717674f6f6e0217cc41c1c6e26`. The unmerged vLLM task therefore
moved to collision-free TASK-31264, carrying every dependency and documentation
reference with it. The vLLM record was originally added by
`ffc4f9d8f8343169097dcac40d3ba4ed0a2177c0`.

A second merge-time sweep found that `origin/dev` had advanced to
`1a1b5c19e0bb3243effb1ae9671158b6670ad6da` and now canonically claimed the
intermediate TASK-31263 and TASK-31264 IDs for unrelated theme follow-up work.
The complete vLLM sequence therefore moved together from TASK-31263..31268 to
the next contiguous block proven free across every fetched non-vLLM ref,
TASK-31282..31287. This preflight task maps TASK-31264 -> TASK-31283; ADR-117
remained collision-free.

A third merge-time sweep found that `origin/dev`
`24d931d0a4f6beec3e0fd7e94d24850ca196e86c` had made the unrelated theme
TASK-31282..31284 claims canonical. Across every fetched non-vLLM local and
remote ref, TASK-31386 was the numeric maximum and TASK-31387..31392 were the
first six contiguous IDs strictly above it. The complete vLLM chain therefore
moved together from TASK-31282..31287 to TASK-31387..31392; this preflight task
maps TASK-31283 -> TASK-31388. ADR-117 remained collision-free across the same
refs.
