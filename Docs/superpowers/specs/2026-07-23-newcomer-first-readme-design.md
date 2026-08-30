# Newcomer-First README Design

Date: 2026-07-23
Status: User-approved design; repository review refreshed 2026-08-23
Related Task: `TASK-2803` (renumbered from `TASK-403` by the duplicate-ID sweep)

## Summary

Rewrite the project README as an end-user-first landing page. A newcomer should be able to understand what `tldw_chatbook` is, set realistic expectations about its Alpha state, install the latest source checkout, connect either a hosted or local model, and send a first Console message without first reading the advanced feature catalog.

The README will use progressive disclosure: the essential newcomer path comes first, concise capability and project-direction context follows, and specialist setup material is summarized or linked.

## Audience

The primary audience is an end user evaluating or installing the application. Contributors remain a secondary audience served by a short contribution section and links to project documentation.

The primary installation path is the latest source checkout. A packaged install may be mentioned only if its current availability and behavior can be verified; it will not displace the source-based quick start while the project is Alpha and changing quickly.

## Goals

- Explain the product and its local-first purpose in plain language.
- State the current project maturity honestly and distinguish dependable core behavior from evolving or optional workflows.
- Get a source-checkout user from clone to launch with one short command sequence.
- Give hosted-provider and local-model users equally visible paths to a first conversation.
- Preserve discoverability of optional capabilities without making them prerequisites for basic use.
- Remove stale, duplicated, overly detailed, or opinion-based material that obscures the newcomer path.

## Non-Goals

- Change application behavior, packaging, configuration, or runtime defaults.
- Document every provider, model, optional dependency, configuration key, database, or internal module in the landing page.
- Complete the broader contributor-documentation corrections tracked by `TASK-333`.
- Create a new documentation hierarchy when existing specialist guides can be linked.
- Promise uniform maturity or full local/server parity across every visible destination.

## Information Architecture

The README will use this order:

1. **Introduction** — a short explanation of the product, intended users, and local-first positioning.
2. **Project status** — current Alpha version, active-development expectations, dependable baseline, evolving areas, and project goal.
3. **Quick start** — requirements, clone, virtual environment, core editable install, and launch.
4. **First conversation** — parallel hosted-provider and local-model setup paths that converge on sending a message in Console.
5. **What users can do** — a concise capability overview organized around user outcomes and the current application destinations. Use current public vocabulary, including Console, Library, Artifacts, Roleplay, Watchlists, Lab, and Settings; do not restore retired names such as Personas as the primary label.
6. **Project direction** — the durable product goal and near-term maturity priorities without turning the README into a commitment-heavy roadmap.
7. **Optional capabilities** — a compact extras table and a few representative source-checkout install commands.
8. **Configuration and local data** — the settings and data locations, API-key choices, and links to deeper guides.
9. **Troubleshooting and documentation** — common recovery pointers and links to maintained project documents.
10. **Contributing, license, and contact** — concise project participation and legal information.

The target is approximately 250–350 lines. Advanced details may use a compact table or a disclosure block when retaining them in the README materially helps users.

## Project-Status Framing

The status section will use explicit categories:

- **Available now:** the local-first Textual application; Console conversations; local conversation, note, Library, Roleplay, Artifact/Chatbook, provider, and settings workflows; and connections to hosted or supported local model servers.
- **Still evolving:** pre-1.0 APIs and UX; advanced optional capabilities; ACP/runtime integration; write synchronization; and complete local/server parity. Some workflows require an external service, model, runtime, or optional dependency group.
- **Goal:** a modular, local-first terminal workbench for LLM conversations, personal knowledge, and controllable agent-assisted workflows while keeping the core installation reasonably lightweight.

Copy must be confident about verified behavior without implying that every destination or integration has equal depth. Current code, package metadata, accepted ADRs, and maturity trackers take precedence over older README claims.

## Quick-Start and First-Conversation Flow

The primary setup sequence is:

1. Clone the repository.
2. Create and activate a Python 3.11-or-newer virtual environment, with Unix/macOS and Windows activation shown clearly.
3. Install the core package from the checkout.
4. Launch the application using a verified package entry point and follow the first-run setup wizard.

The first-run wizard presents the two model-connection paths. The README will explain them with equal prominence:

- **Hosted provider:** choose a cloud provider and model in the wizard, supply the API key through the supported wizard/Settings flow or a documented environment variable, complete setup, open Console, and send a message.
- **Local model:** start a supported local server such as Ollama, llama.cpp, or another OpenAI-compatible endpoint, let the wizard detect it or configure its endpoint/model, complete setup, open Console, and send a message.

The README will also give the recovery path for skipped or incomplete setup: run the wizard again from **Settings › Diagnostics › Run setup wizard**, or use **Settings › Providers & Models** for direct configuration.

Both paths end at the same visible success condition: a user can send a first message from Console. RAG, ingestion, media processing, web search, transcription, MCP, and browser access appear only after this baseline path.

## Content Boundaries

The rewrite will:

- Replace repetitive feature-by-feature prose with a short outcome-oriented overview.
- Replace the exhaustive extras installation block with representative commands and a concise extras table grounded in `pyproject.toml`.
- Remove duplicated headings, obsolete implementation states, old navigation descriptions, and the opinionated local/commercial model recommendation section.
- Avoid copying large configuration examples already covered by specialist documentation.
- Correct stale README claims that overlap `TASK-333`, while leaving that task open for its separate contributor-documentation scope.
- Use the current destination information architecture, including accepted route consolidation decisions.
- Link to the maintained `Docs/User_Guide/` entry points for task-level guidance instead of recreating the user guide inside the README.

## Screenshot Treatment

The obsolete PoC screenshot will not remain as the primary visual. It will be replaced only by a current, reader-facing screenshot that accurately depicts the present product and is suitable as a landing-page image. QA captures with diagnostic state, excess canvas, temporary data, or setup-specific context are not automatically suitable merely because they are recent. If no existing image meets that bar, the README will omit the screenshot rather than present a misleading or unpolished interface.

## Validation

Before completion:

- Cross-check the version, Python requirement, exact Textual 8.x pin, dependency groups, and command entry points against `pyproject.toml` and package files.
- Exercise safe CLI help or launch-adjacent checks locally and verify the source install command where practical without modifying user configuration.
- Validate README-relative links and image targets.
- Check Markdown heading hierarchy and fenced-code balance.
- Cross-check project-state, first-run wizard, provider configuration, and navigation claims against current code, accepted ADRs, canonical maturity trackers, and the maintained user guide.
- Review the final diff for newcomer readability, stale claims, unnecessary repetition, and unrelated changes.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a documentation-only rewrite that describes existing package behavior and follows accepted product and navigation decisions. It does not change architecture, runtime policy, storage, security, dependencies, or long-lived UX ownership.
