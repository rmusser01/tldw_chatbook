---
id: TASK-34100
title: First-run setup wizard burn-down of the 2026-10-02 UX review
status: In Progress
assignee:
  - '@claude'
created_date: '2026-10-03 16:00'
labels:
  - first-run-wizard
  - ux-review-2026-10-02
dependencies: []
priority: high
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The 2026-10-02 senior-design/HCI review of the first-run setup wizard (Docs/superpowers/qa/first-run-wizard-ux-review-2026-10-02/README.md) found that setup means "saved", not "works". Setup reports ✓ while 4 of the 5 cloud providers walked live with real keys cannot send a first message; only Anthropic replied. OpenAI's recommended and default models are refused before sending, OpenRouter accepts any key, Gemini ships only retired models, and Moonshot replies are thrown away. Setting a key-encryption password can lock the user out at the next launch. The heuristic score is 16/40 (Poor). The register holds 151 verified issues (5 P0, 23 P1, 71 P2, 52 P3). Before this filing, 122 of them had no open task, including 4 of the 5 P0s.

Every fix in this programme enforces the four rules in §5 of the report:
(1) Done means a reply, not a write: setup is complete when a first turn can be sent to the chosen model.
(2) Write nothing the user didn't touch, and never drop anything the user typed without saying so.
(3) Every mark (✓, ✗, !, "Ready") is computed from persisted, verified state, through the same resolver the runtime uses.
(4) One owner per fact: provider facts, step names, "change it later" destinations, download jobs and encryption state each live in one place.

The work runs in waves:
- Wave A, in parallel: .1 step extraction, .2 model catalog and readiness, .3 Moonshot continuation, .4 encryption core, .5 Console handoff, .16 portability, .17 design spec.
- Wave B, after .1 merges: .6 provider step, .7 model step, .8 voice step, .13 full-track steps.
- Wave C: .9 honest status, .10 setup session, .11 input policy, .12 terminal frame, .14 downloads.
- Wave D: .15 setup registry and docs.
- After that, the approved .17 spec is implemented as follow-up tasks.

The owner has made three decisions. Merge each group when it is green. Verify against a real llama.cpp server and real cloud providers, never mock servers. Any redesign that changes the wizard's shape is written as a spec first and needs the owner's approval. group-assignment.json in the report folder maps every issue id to its subtask, and §5.6 lists the existing tasks that each subtask absorbs or updates.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 TASK-34100.1 is Done: the wizard's steps are extracted into their own modules with room under the size ratchet, and every Next shows a busy cue
- [ ] #2 TASK-34100.2 is Done: the first chat works on every shipped default, with a repaired model catalog and context budget, an honest readiness verdict and real key checks
- [ ] #3 TASK-34100.3 is Done: Moonshot replies survive provider-continuation persistence
- [ ] #4 TASK-34100.4 is Done: the key-encryption lifecycle has one unlock path, safe re-encryption, recovery and Settings controls
- [ ] #5 TASK-34100.5 is Done: the Console handoff and first reply have honest toasts, expose no internal tools and give actionable first-send errors
- [ ] #6 TASK-34100.6 is Done: the Provider step has a real skip, sticky local detection, endpoint fields, search and honest key checks
- [ ] #7 TASK-34100.7 is Done: the Model step has curated chat-only ranking and a searchable list, and skipping it never discards the key
- [ ] #8 TASK-34100.8 is Done: an untouched Voice step writes nothing, and each voice service says whether it will work
- [ ] #9 TASK-34100.9 is Done: one outcome record drives the tracker, the Summary and what Next means
- [ ] #10 TASK-34100.10 is Done: re-runs, resume, exits and entry points respect existing state
- [ ] #11 TASK-34100.11 is Done: one input policy, where highlight browses and Enter, Space or a click selects
- [ ] #12 TASK-34100.12 is Done: the terminal-native frame works at 80x24 and at large sizes, shows state as text and has readable contrast
- [ ] #13 TASK-34100.13 is Done: the Full-track steps (RAG, Tools, Notes, Speech, Appearance) do what they say
- [ ] #14 TASK-34100.14 is Done: model downloads in setup can be cancelled, clean up after themselves and finish what they start
- [ ] #15 TASK-34100.15 is Done: one setup registry gives consistent names and real 'change it later' homes, and the User Guide matches the wizard
- [ ] #16 TASK-34100.16 is Done: portable setup, with a documented second-machine path, a restore that understands config files, and launch flags
- [ ] #17 TASK-34100.17 is Done: the owner has approved a design spec for the setup flow (Quick track, tldw server, re-run dashboard, Say hello)
- [ ] #18 On a fresh profile, following the wizard's own recommendations reaches a first reply for OpenAI, Anthropic, Gemini and a real llama.cpp server
- [ ] #19 Setting a key-encryption password and relaunching through tldw-cli and python -m both open the app
- [ ] #20 No Next in the wizard is a dead end
- [ ] #21 Every review issue id in group-assignment.json is resolved or explicitly re-scoped with the owner's agreement
<!-- AC:END -->
