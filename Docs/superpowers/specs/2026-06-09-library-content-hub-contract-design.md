# Library Content Hub Contract

Date: 2026-06-09
Backlog task: TASK-89.1
Status: corrected implementation contract

## Purpose

Library is the landing page and center hub for media and ingested content in Chatbook. It is not the primary surface for starting conversations about media, not a second Console, and not a replacement for the older owner screens.

The default Library route should help users answer:

- What content do I have?
- Which module owns the deeper workflow?
- Where do I go to add, edit, browse, retrieve from, organize, or study that content?
- What can be handed off to Console, Search/RAG, Study, or other modules, and what is blocked?

## ADR Check

ADR required: no
ADR path: N/A
Reason: this contract changes UI/UX expectations and mounted regressions only. It does not change storage/schema, sync/conflict policy, provider/runtime boundaries, service ownership, security posture, or persistence contracts.

## Product Contract

Library must summarize all user-owned content across Notes, Media, Conversations, Collections, Search/RAG, Import/Export, Study, Flashcards, and Quizzes. It should route to owner surfaces for deep work rather than recreating those tools inline.

Workspace switching must not hide notes, media, conversations, or collections from global browsing/search. Active workspace gates whether content can be staged, manipulated, or used in the active Console/RAG/agent context. The user can still view and search all their content across workspaces.

Console remains the live agentic control surface. Library may prepare context, run Library-native Search/RAG, and hand off eligible content or evidence, but Console handoff is secondary to the hub’s inventory and routing role.

Unavailable or incomplete actions must be honest. Disabled controls need visible reason and recovery copy in the detail pane, inspector, tooltip, or command palette. WIP routes must identify themselves as placeholders rather than looking like broken production controls.

## Owner Model

| Area | Library responsibility | Owner for deeper work |
| --- | --- | --- |
| Notes | Count, recent titles, route, eligibility summary | Notes screen handles create/edit/sync/templates/export/delete. |
| Media | Count, recent titles, route, ingestion status summary | Media screen handles browse, ingest review, analysis, read-it-later, and metadata. |
| Conversations | Count, route, placeholder detection until browser is functional | Conversations/Chat surfaces handle saved conversation browsing and resume. |
| Search/RAG | Library-native query entry, readiness, evidence inspection, handoff | Library Search/RAG panel owns retrieval flow. |
| Collections | Library-owned grouping and membership surface | Library Collections panel owns content grouping. |
| Import/Export | Entry point and return framing | Ingest/import/export routes perform acquisition and movement. |
| Study / Flashcards / Quizzes | Source/context preparation and route entry | Study, Flashcards, and Quizzes own generation/review sessions. |
| Console handoff | Secondary eligible-content handoff | Console owns live agentic execution. |

## Layout Contract

Library uses the same terminal-native grammar as the other redesigned destinations:

- Top global navigation remains visible.
- Destination header states purpose, local/server/workspace status, and active mode.
- Mode strip remains compact and keyboard reachable.
- Main content fills the terminal/web canvas.
- Three columns remain visually delineated: Library Modules, Content Hub, Hub Inspector.
- Default center pane shows content-hub cards, not selected-source detail.
- Inspector explains ownership, workspace boundary, handoff status, and recovery.
- Focus state is visible and does not obscure labels.
- Empty, loading, blocked, and error states use explicit text, not color alone.
- Horizontal overflow is treated as a bug at supported wide and compact sizes.

## Default Hub Requirements

The default Library mode must:

- Render a `Library Content Hub` center pane.
- Show real counts and recent titles for Notes, Media, and Conversations when services are available.
- Show Search/RAG, Import/Export, Collections, and Study as module/workflow cards.
- Keep owner-route actions keyboard reachable with stable IDs.
- Preserve workspace-gated `Use in Console` behavior, but explain that Console handoff is secondary.
- Teach empty state recovery: import media, create notes, or open Library Search/RAG after indexing.
- Avoid selected-source detail or inspector rows until a deeper mode actually supports item selection.

## Mode Contracts

### Sources / Hub

The default mode is a content hub. It summarizes local content and routes to owners. It must not imply that source selection is required before the user can understand Library.

### Search/RAG

Search/RAG is Library-native. A user can enter a query in Library, retrieve results, inspect evidence, and launch Console with grounded context. Readiness must be real: dependencies, index state, and provider state should not be aspirational.

Citation and snippet support remains a later-stage requirement, but this mode must leave room for source title, snippet, citation metadata, and authority.

### Import/Export

Import/Export is a Library-owned workflow entry even when implementation routes to ingest/import screens. The Library surface must make that handoff explicit: what will happen, where the user is going, and how imported material returns as Library content.

### Conversations

Conversations must become a functional Library-owned browser, not a placeholder. It should list saved conversations, allow inspection, and support eligible handoff without exiting into legacy IA unexpectedly.

### Collections

Collections live inside Library. Collections must allow local grouping, membership inspection, workspace eligibility, and eventual collection-scoped Search/RAG, Study, and Console handoff where policy permits.

### Workspaces

Workspaces are global operating context, not a Library-only filter. Library must show all content across workspaces for browse/search, with labels for workspace ownership. Active workspace gates what can be staged, manipulated, or handed to Console.

### Study, Flashcards, Quizzes

Library owns source preparation and handoff. Study owns deeper sessions, card generation/review, and quizzes. Library actions should carry selected or scoped content context into Study and clearly state when no content is eligible.

## Staged Implementation Plan

1. TASK-89.2: Make the default Library route a content hub landing page.
2. TASK-89.3: Replace the placeholder Library Conversations route with a functional browser.
3. TASK-89.4: Clarify Library Import/Export as a Library-owned content acquisition workflow while preserving existing route handoff.
4. TASK-89.5: Deepen Collections and workspace handoff actions while preserving global visibility.
5. TASK-89.6: Harden Library Search/RAG query and evidence workflow, including citation/snippet readiness.
6. TASK-89.7: Clarify Study, Flashcards, and Quizzes handoffs from Library content context.
7. TASK-89.8: Run actual-use QA across the complete Library hub and cross-screen handoffs before the epic is considered complete.

## Actual-Use Closeout Status

TASK-89.8 adds CDP/Textual-web actual-use evidence for Content Hub, Search/RAG, Import/Export, Workspaces, Collections, Conversations, Study, Flashcards, and Quizzes. The closeout evidence is tracked in `Docs/superpowers/qa/library-content-hub-closeout/2026-06-22-library-content-hub-actual-use-closeout.md`.

Current status: implementation and focused regression coverage are complete for the accepted content-hub scope; final visual acceptance was approved from the rendered screenshots.

## Non-Goals

- Do not rebuild Notes, Media, Study, Flashcards, Quizzes, or Import/Export inside Library.
- Do not implement sync engine behavior in this Library epic.
- Do not make Collections a top-level destination again.
- Do not hide cross-workspace content from Library browsing/search.
- Do not implement full citation/snippet persistence in the first hub slice.
- Do not treat generated concept pages or ASCII sketches as approval substitutes. Future UI work still requires actual rendered screenshots.

## Open Risks

- Search/RAG readiness may overstate availability because current panel state still sets some readiness flags optimistically.
- Conversations currently looks reachable but is not usable, which can mislead first-time users into thinking their conversation library is broken.
- Import/Export and Study handoffs can look like mode switches unless the destination and return path are explicit.
- The first hub slice still needs actual CDP/Textual-web screenshot approval before it can be considered complete.
