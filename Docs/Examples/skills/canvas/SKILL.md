---
name: canvas
description: Create or revise a requested Canvas using Chatbook's supported HTML, interactive controls, and offline diagrams.
argument_hint: requested Canvas or change
context: inline
user_invocable: true
disable_model_invocation: true
---
# Canvas

## User request

{{args}}

## Consent

Offer Canvas only when a substantial visual or interaction materially helps.
For proactive use, describe the proposed artifact and benefit in one short
sentence and wait for acceptance before loading detailed guides, delegating,
or generating artifact source. A decline or no answer does not authorize
creation; do not repeat the same declined offer. Explicit Canvas requests and
requested edits already authorize that work. Consent covers that artifact and
bounded corrections; unrelated artifacts or unrequested redesigns need a new
offer. If context does not establish consent, clarify.

Use the user request above and surrounding conversation to identify the task.
Bare activation without a concrete request requires clarification before authoring;
an empty argument section alone does not negate an explicit surrounding request.
Work inline in the owning Console conversation. Do not invoke this skill as a
model tool or spawn a child to author the Canvas.

## Discover and author

Discover available Canvas tools with `find_tools` and disclose needed schemas
with `load_tools` when available; use already disclosed tools directly. Never
assume unavailable tools are authorized. Read only relevant topics after consent
through `canvas_guide`: `basics` for document shape, `controls` for interaction,
`mermaid` for offline diagrams, and `repair` for diagnostics or revision recovery.
Reuse guidance already in context; do not fetch all topics by default.

Use the supported complete HTML document format. Exact-profile guidance and
runtime checks take precedence over generic examples or this skill. Loading a
guide does not prove a profile is available. Preserve source if its profile is
unavailable; adapting it requires an explicitly requested new Canvas, never a
silent migration.

For edits, list if needed to identify the intended Canvas, then read its current
complete source and revision ID. Submit the complete replacement with that ID
as `expected_parent_revision_id`. On conflict, reread and adapt the requested
change; do not overwrite newer work or retry the same parent. Report repeated
conflicts instead of looping.

## Report and recover

Use only available evidence to distinguish staged source, committed source,
preview pending, preview ready, and preview failed/source-only. Source acceptance
does not prove browser success; the assistant turn must settle for staged changes
to persist. Do not claim to see a preview or diagnostic you have not received.
Repair from concrete diagnostics or the user's request; after one failed repair
attempt, stop, report the limitation, and let the user decide on further repair.
A report that the repair still fails is not permission for a second attempt:
explain that the problem remains and ask whether to attempt another repair.
Do not claim to create, update, or repair without a successful matching tool result.
Browser failures do not authorize automatic model submissions.

Missing, locked, or untrusted skills follow normal Library review/trust refusal.
Never silently replace a refused skill invocation with ordinary Canvas work.
If Canvas tools are unavailable, explain briefly and continue useful chat work;
do not enable settings, fetch libraries, or generate external runnable HTML as
an automatic workaround. If only the guide is unavailable, use existing
authoritative runtime guidance when sufficient, otherwise explain the limitation.
Submit and download remain requests for user-confirmed host actions.
