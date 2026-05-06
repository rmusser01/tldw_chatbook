# Post-UX Product Roadmap Design

Date: 2026-05-06
Status: User-approved design; pending written-spec review
Primary Repo: `tldw_chatbook`
Source Branch: `dev`
Related Roadmaps:

- `Docs/superpowers/trackers/unified-shell-maturity-roadmap.md`
- `Docs/superpowers/trackers/product-maturity-roadmap.md`
- `Docs/superpowers/specs/2026-05-05-product-maturity-phased-roadmap-design.md`
- `Docs/superpowers/specs/2026-05-06-destination-layout-ia-contracts-design.md`

## Summary

This roadmap begins after the current in-progress UX/UI and destination layout-contract work is complete. It does not replace that workstream. It becomes the next product center once the shell, destination contracts, and visual/interaction baseline are stable.

The roadmap has one canonical spine with three aligned views:

- **Strategy view**: the product promise, audience, value loops, and sequencing logic.
- **Execution view**: internal phases, gates, owners, QA evidence, and backlog-ready slices.
- **Public view**: directional roadmap themes without dates, task IDs, or delivery promises.

The roadmap sequencing is:

1. Reliability and product confidence.
2. Complete workflow value.
3. Server parity and live integrations.
4. Release hardening and distribution.

## North Star

Chatbook turns local knowledge into controlled agentic work and durable artifacts: ingest sources, reason over them in Console, configure agents/tools/workflows, monitor execution, and preserve outputs as reusable Chatbooks, reports, study sets, or other artifacts.

## Primary Audience

The primary wedge is solo builder/operators using Chatbook as a local-first agentic knowledge console. They use the product to ingest material, reason over it, control task agents, coordinate tools and workflows, monitor work, and keep outputs locally durable.

Researcher and student workflows are expansion paths. They share the same source-to-output loop, but their durable outputs are more often flashcards, quizzes, notes, summaries, and study artifacts.

## Product Positioning

Chatbook is not primarily a study app, generic chatbot, or file manager. It is a local-first agentic knowledge console for users who want to keep sources, agent behavior, tools, workflows, execution state, and generated artifacts under inspectable control.

The product promise is not "chat with documents." It is "turn a personal knowledge base into controlled, repeatable agentic work and durable outputs."

## Canonical Spine

### 1. Reliability Baseline

Prove the completed UX/UI is dependable.

The user should be able to launch from a clean state, navigate top-level surfaces, understand setup requirements, recover from blocked states, use keyboard-first flows, and trust that layout and authority labels remain stable across supported terminal sizes.

### 2. Workflow Value

Complete the highest-value loops end to end.

The product becomes compelling when users can finish loops without manual state reconstruction: ask grounded questions, save outputs, reopen artifacts, configure agent work, monitor progress, generate study outputs, and recover from missing dependencies or runtimes.

### 3. Server Parity And Live Integrations

Connect local Chatbook workflows to high-value `tldw_server2` capabilities only where they improve local-first use.

Parity should be measured by workflow value, not raw endpoint coverage. Users must be able to see whether an action is local, server-backed, synced, remote-only, workspace-scoped, or dry-run.

### 4. Release Hardening And Distribution

Convert the matured product into a release-candidate experience.

The product should be installable, configurable, recoverable, documented, and explainable without requiring developer knowledge.

## Strategy View

### Value Loops

The roadmap should preserve six core value loops.

| Loop | User Outcome | Primary Surfaces |
| --- | --- | --- |
| Grounded Answer Loop | Select or retrieve sources, stage evidence in Console, ask or command, inspect authority, and save output. | Library, Search/RAG, Console, Artifacts |
| Source-to-Artifact Loop | Ingest material, reason over it, produce a durable output, and reopen or export it later. | Library, Console, Artifacts, Home |
| Agent Run Loop | Configure behavior/tools/runtime, launch controlled work, approve risky actions, monitor, recover, and save results. | Personas, Skills, MCP, ACP, Schedules, Workflows, Console, Home |
| Monitoring Loop | Watch sources or collections, surface changes, follow work, and convert findings into outputs or next actions. | W+C, Schedules, Home, Console, Artifacts |
| Study Loop | Turn source sets or Console outputs into flashcards, quizzes, and other study artifacts that remain reusable. | Library, Workspaces, Flashcards, Quizzes, Console, Artifacts |
| Recovery Loop | Understand missing provider/runtime/server/dependency states and take a clear configure, retry, pause, or local-mode action. | Home, Console, Settings, affected destination |

### Sequencing Logic

Reliability comes first because broken setup, layout, focus, authority, or recovery states destroy trust in a terminal product faster than missing advanced features.

Workflow value comes second because Chatbook becomes useful when users can complete loops without stitching state together manually.

Parity and live integrations come third because server-backed capabilities should strengthen the local-first product rather than define it prematurely.

### Strategic Non-Goals

- Do not make every destination a live agent console.
- Do not let study workflows become the product center.
- Do not collapse MCP, ACP, Skills, Personas, Schedules, and Workflows into one vague "agents" bucket.
- Do not chase `tldw_server2` parity unless the capability improves a real local Chatbook workflow.
- Do not publish roadmap dates or delivery commitments before execution evidence supports them.

## Execution View

Execution should be staged as PR-sized gates under the canonical spine. Each gate must prove a running-app workflow, not merely add UI.

### Stage 1: Reliability Baseline

Goal: make the completed UX/UI work trustworthy.

Primary gates:

- Clean first-run replay after the UX/UI work lands.
- Compact, default, and large terminal layout verification.
- Keyboard/focus sweep for top-level destinations and major subflows.
- Empty/error/setup-state coverage for providers, local runtimes, server connection, optional dependencies, and missing sources.
- Recovery taxonomy pass proving cause, impact, owner, and next action are visible.
- Regression harness update so future product work reuses the same QA evidence pattern.

Exit criteria:

- A fresh user can launch, navigate, understand blocked states, and complete at least one local source-to-Console proof.
- P0/P1 usability defects are fixed or explicitly accepted with owner and follow-up.
- QA evidence is tracked in the repo and connected to the affected roadmap gate.

### Stage 2: Workflow Value

Goal: complete the product's core loops.

Primary gates:

- Grounded Answer Loop: Library/Search/RAG source selection stages evidence into Console and preserves authority through answer/save.
- Source-to-Artifact Loop: Console output saves as a Chatbook or artifact, reopens from Artifacts, and resumes from Home.
- Knowledge Organization Loop: Workspaces and Collections clearly separate broad user context from reusable source sets.
- Study Loop: flashcards and quizzes can be generated from selected sources or Console outputs and reused later.
- Agent Run Loop: Personas, Skills, MCP/ACP readiness, Schedules, and Workflows can configure and launch controlled work into Console.
- Monitoring Loop: W+C and Schedules surface changed, running, or failed work in Home and Console.
- Recovery Loop: missing provider/runtime/server/dependency states remain understandable and recoverable across all loops.

Exit criteria:

- Each loop has automated regression coverage, running-app QA evidence, and documented residual-risk decisions.
- Console remains the live execution surface.
- Other destinations prepare, inspect, organize, resume, or hand off work instead of becoming alternate live-run consoles.

### Stage 3: Server Parity And Live Integrations

Goal: add server-backed value where it strengthens local-first use.

Primary gates:

- `tldw_server2` parity inventory by workflow family.
- Server-backed Library/Search/RAG where it improves source coverage, scale, or reuse.
- Import/sync paths for sources, personas, skills, artifacts, and collections where feasible.
- Live event producers for Home and Console where services support real running work.
- W+C parity for watchlists, collection feeds, alert rules, retries, run history, and WebSub-style monitoring.
- Explicit local/server/workspace/remote-only/dry-run authority labels.

Exit criteria:

- Users can tell what is local, server-backed, synced, remote-only, or unavailable.
- Remaining parity gaps are documented as roadmap items or honest unavailable states.
- Server-backed paths preserve local-first control and recovery behavior.

### Stage 4: Release Hardening And Distribution

Goal: make the product shippable and explainable.

Primary gates:

- Full first-time user replay.
- Power-user replay across the major workflow loops.
- Docs and onboarding updates matching current behavior.
- Packaging and configuration validation.
- Migration and data-safety checks.
- Optional dependency and model/provider setup documentation.
- Public roadmap refresh from the same canonical spine.

Exit criteria:

- The product can be installed, configured, used, recovered, and understood without relying on developer knowledge.
- Public-facing docs match the actual product behavior.
- P0/P1 release blockers are closed or explicitly accepted with owner and follow-up.

## Public Roadmap View

The public roadmap should be directional. It should describe priorities and likely future areas without dates, task IDs, internal gates, or delivery promises.

### Roadmap Intro

Chatbook is evolving into a local-first agentic knowledge console: a place to ingest sources, reason over them, configure controlled agentic work, monitor progress, and preserve useful outputs as durable artifacts.

### Now: Reliability And Product Confidence

Current focus:

- first-run setup and configuration clarity.
- stable layouts across terminal sizes.
- keyboard-first navigation and focus behavior.
- understandable empty, error, and blocked states.
- clear local/server/runtime authority labels.
- repeatable QA coverage for core workflows.

### Next: Complete Workflow Loops

Next focus:

- ask grounded questions over selected sources.
- turn answers into reusable Chatbooks and artifacts.
- organize knowledge with Workspaces and Collections.
- generate and reuse flashcards, quizzes, reports, and study outputs.
- configure personas, skills, tools, schedules, and workflows.
- launch and monitor controlled agent work through Console.
- recover cleanly when providers, runtimes, or optional capabilities are missing.

### Later: Server-Backed And Live Capabilities

Longer-term focus:

- richer source and RAG integration.
- server-assisted watchlists and collections.
- live status updates for running work.
- import/sync paths for sources, personas, skills, and artifacts.
- clearer collaboration between local and remote runtimes.
- documented residual gaps where local mode remains the better default.

### Always: Local-First Control

Across every phase, Chatbook should keep source authority, runtime readiness, approvals, recovery paths, and generated outputs visible. Console remains the live work surface; other areas prepare, inspect, organize, resume, or preserve work.

## Alignment Rules

The three roadmap views must stay aligned through these rules:

- Strategy owns the why, not the task list.
- Execution owns phase gates, backlog slicing, QA evidence, and acceptance criteria.
- Public roadmap owns directional communication, not commitments.
- Every public theme must map to at least one strategy loop and one execution stage.
- Every execution gate must identify which user loop it improves.
- Server parity work must be justified by user workflow value.
- Study features remain explicit but do not become the product center.
- Workspaces and Collections remain distinct: Workspaces are broad user scope; Collections are reusable source sets.
- Console remains the live agentic work surface.

## Immediate Next Artifacts

After the current UX/UI work completes, the next roadmap work should produce:

1. A reconciled post-UX baseline note confirming which UX/UI and layout-contract gates are complete.
2. An updated execution tracker that starts Stage 1 Reliability Baseline from that post-UX baseline.
3. Backlog parent/child tasks for the first reliability gates only.
4. A public roadmap page or README section derived from this spec's public roadmap view.

## Risks And Mitigations

| Risk | Mitigation |
| --- | --- |
| Roadmap becomes too broad to execute. | Keep execution gates PR-sized and require each gate to prove one workflow outcome. |
| Public roadmap overpromises. | Use directional language and avoid dates or delivery guarantees. |
| Server parity distorts the local-first product. | Prioritize parity by workflow value and show local/server authority explicitly. |
| Study workflows take over positioning. | Keep study as an expansion loop built on the same source-to-output model. |
| Agent surfaces blur together. | Keep Personas, Skills, MCP, ACP, Schedules, Workflows, and Console roles explicit. |
| Reliability work becomes documentation-only. | Require running-app QA evidence and regression coverage for every gate. |

## Transition Criteria

Move from this design into implementation planning only after:

- the current UX/UI work is confirmed complete.
- this spec is reviewed and accepted.
- Stage 1 Reliability Baseline is narrowed to the first PR-sized task set.
- the public roadmap text is accepted as directional and non-committal.
