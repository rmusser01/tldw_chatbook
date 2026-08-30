# Newcomer-First README Design

Date: 2026-07-23
Status: User-approved corrective revision, 2026-08-30
Related Task: `TASK-2803` (renumbered from `TASK-403` by the duplicate-ID sweep)

## Corrective Revision

The first implementation of this design overcorrected toward brevity. It
replaced the original README's useful layered reference material with a short,
abstract landing page that read more like an internal architecture memo than a
welcoming project README. The approved correction restores the original README
as the source document and repairs it in place.

The README may remain long. Its usability comes from progressive disclosure:
the first two screenfuls must explain and launch the product, while detailed
feature, installation, configuration, and development material remains
available farther down the page.

## Summary

Restore and repair the original project README as a layered landing page and
technical reference. A newcomer should quickly understand what
`tldw_chatbook` is, see the application, set realistic expectations about its
Alpha state, install it, connect either a hosted or local model, and send a
first Console message. Returning users and contributors should still find the
original README's detailed feature, extras, configuration, troubleshooting,
and development material below that quick path.

## Audience

The primary audience is an end user evaluating or installing the application. Contributors remain a secondary audience served by a short contribution section and links to project documentation.

The primary installation path is the latest source checkout. A packaged install may be mentioned only if its current availability and behavior can be verified; it will not displace the source-based quick start while the project is Alpha and changing quickly.

## Goals

- Explain the product and its local-first purpose in plain language.
- State the current project maturity honestly and distinguish dependable core behavior from evolving or optional workflows.
- Get a source-checkout user from clone to launch with one short command sequence.
- Give hosted-provider and local-model users equally visible paths to a first conversation.
- Preserve useful detail about optional capabilities, configuration, and
  development without making it prerequisite reading.
- Remove only stale claims, duplication, obsolete navigation, speculative
  features, and unhelpful giant install commands.
- Preserve the original README's voice, screenshot-led opening, and layered
  reference structure.

## Non-Goals

- Change application behavior, packaging, configuration, or runtime defaults.
- Turn the README into an exhaustive API or configuration reference.
- Complete the broader contributor-documentation corrections tracked by `TASK-333`.
- Create a new documentation hierarchy when existing specialist guides can be linked.
- Promise uniform maturity or full local/server parity across every visible destination.

## Information Architecture

The README will use this order:

1. **Immediate orientation** — project name, plain-English description, current
   screenshot, direct Alpha notice, concrete reasons to use the application,
   and links to the User Guide and troubleshooting.
2. **Five-minute quick start** — requirements, clone, virtual environment,
   core editable install, launch, first-run setup, and first Console message.
3. **What users can do** — capability groups organized around real workflows,
   not a flat inventory of screens.
4. **Local and hosted models** — equal setup paths and an explicit explanation
   that local-first storage does not imply an embedded inference runtime.
5. **Installation options** — core install first, followed by accurate optional
   extras, common combinations, platform notes, and development installation.
6. **Configuration and data** — wizard-first configuration, settings, secrets,
   storage, profiles, backups, and external trust boundaries.
7. **Detailed feature reference** — retain the original README's useful depth,
   corrected to current names and verified behavior.
8. **Troubleshooting and documentation** — practical recovery steps and links
   to maintained task-level guides.
9. **Development, contributing, license, and contact** — contributor setup,
   tests, project status, legal information, and support routes.

There is no arbitrary line-count target. The top path must stay short; the
reference material below it may be comprehensive when it remains accurate and
well structured.

## Project-Status Framing

The status section will use explicit categories:

- **Available now:** the local-first Textual application; Console conversations; local conversation, note, Library, Roleplay, Artifact/Chatbook, provider, and settings workflows; and connections to hosted or supported local model servers.
- **Still evolving:** pre-1.0 APIs and UX; advanced optional capabilities; ACP/runtime integration; write synchronization; and complete local/server parity. Some workflows require an external service, model, runtime, or optional dependency group.
- **Goal:** a modular, local-first terminal workbench for LLM conversations, personal knowledge, and controllable agent-assisted workflows while keeping the core installation reasonably lightweight.

Copy must be confident about verified behavior without implying that every
destination or integration has equal depth. Status should be a compact,
plain-spoken warning, not the dominant voice of the README. Current code,
package metadata, accepted ADRs, and maturity trackers take precedence over
older README claims.

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

- Start from the README immediately before PR #2045 rather than expanding the
  replacement copy.
- Retain detailed feature and installation material where it helps users make
  or troubleshoot a real choice.
- Add a short outcome-oriented overview before the detailed feature reference.
- Replace only the unmaintainable all-extras command with practical,
  representative combinations grounded in `pyproject.toml`.
- Remove duplicated headings, obsolete implementation states, old navigation descriptions, and the opinionated local/commercial model recommendation section.
- Avoid copying large configuration examples already covered by specialist documentation.
- Correct stale README claims that overlap `TASK-333`, while leaving that task open for its separate contributor-documentation scope.
- Use the current destination information architecture, including accepted route consolidation decisions.
- Link to the maintained `Docs/User_Guide/` entry points for task-level guidance instead of recreating the user guide inside the README.

## Screenshot Treatment

The opening will include a current reader-facing screenshot. The historical PoC
screenshot may be retained temporarily only if repository review confirms that
it is still more representative than available QA captures. A diagnostic or
temporary QA image must not become the landing-page visual merely because it is
newer.

## Validation

Before completion:

- Cross-check the version, Python requirement, exact Textual 8.x pin, dependency groups, and command entry points against `pyproject.toml` and package files.
- Exercise safe CLI help or launch-adjacent checks locally and verify the source install command where practical without modifying user configuration.
- Validate README-relative links and image targets.
- Check Markdown heading hierarchy and fenced-code balance.
- Cross-check project-state, first-run wizard, provider configuration, and navigation claims against current code, accepted ADRs, canonical maturity trackers, and the maintained user guide.
- Review the final diff for newcomer readability, stale claims, unnecessary repetition, and unrelated changes.
- Verify that the corrective PR contains only the README, its task/design/plan
  records, and any deliberately selected landing-page asset.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is a documentation-only rewrite that describes existing package behavior and follows accepted product and navigation decisions. It does not change architecture, runtime policy, storage, security, dependencies, or long-lived UX ownership.
