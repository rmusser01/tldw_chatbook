# Buddy management UI implementation plan

**Goal:** Expose independently selected Buddy artwork and exact conversation/workspace
scope through one native management modal without leaving the current destination.
**Architecture:** A staged form returns a validated choice; an app-owned coordinator
applies it through BuddyLibrary, controller preference persistence, existing Persona
services, and scoped lifecycle projection. Modal owns no long-running execution.
**Tech stack:** Textual native widgets, existing Rich pixel renderer, local config.
**Spec:** Docs/superpowers/specs/2026-09-08-console-buddy-management-design.md

ADR required: yes
ADR path: backlog/decisions/139-independent-buddy-conversation-and-workspace-bindings.md
Reason: independent artwork, explicit scope and cross-screen management.

## Global constraints

Reuse established modal theme tokens; vertical scroll with fixed action footer;
keyboard and normal/compact terminal coverage. Apply is the only mutation boundary.
Temporary conversation bindings stay in memory. Local IDs never match remote IDs.
No ambient active-session fallback for a binding. No new long-lived ChatScreen fields
for the feature; app-owned coordinator is the shared entry from both surfaces.

## TASK-32081

1. Add `Persona_Buddy/interaction.py`: immutable scope/preferences contracts,
   exact session resolution against live ID, binding revision and durable local
   conversation ID; profile-safe config encoding/decoding; reject invalid targets.
   Test retargeting, session replacement, restart matching and temporary exclusion.
   Promote a first-persisted durable conversation ID under the preference Apply
   lock; retry failed storage without changing the target. Canonicalize restored
   live IDs in the staged form through the exact resolver, never ambient selection.
2. Add `Widgets/Persona_Widgets/buddy_management_modal.py`: injected named choices,
   staged selection/import path, scope, Persona unchanged/None/pick, animation,
   size and speech preferences. Preview is read-only. Form validates before Apply;
   Escape/Cancel returns None. No hidden install or Persona mutation on preview.
3. Add `UI/Navigation/buddy_management.py`: shared opener/read model/apply worker,
   uses completed BuddyLibrary API and existing Persona services. Apply publication
   before choosing its committed ID. Preserve old usable selection if validation or
   publication fails. Config errors remain visible for retry rather than false success.
4. Add stable composer Buddy action and floating settings control. Integrate through
   small local imports; preserve existing lazy disabled-Buddy boot behavior. Plain
   click will open management until the subsequent interaction task is connected.
5. Filter Buddy adapter events by exact binding and reconcile current scoped runtime
   snapshot when switching bindings; no unrelated-session state leakage. Add Static
   option to rendering's existing motion stop boundary; retain current expression.
6. Integrate explicit target Persona changes through store/service authority with
   source validation and future-turn application. Preserve accepted turn settings.
7. Targeted native pilots: menu action, staged Cancel, valid Apply, missing targets,
   preview, normal and compact scroll/focus, only one visible Buddy, Persona=None.

## Interfaces

Foundation library: list_buddies/get_buddy/get_graph, review_archive then publish_review,
copy_persona, ensure_builtin. Controller owns actual enabled/open/geometry/selection.
The `[buddy_interaction]` section owns only scope/animation/notifications, keeping
content ownership and app presentation separate. Root coordinates application of both
and rollback/failure reporting; no duplicate caches or hidden Persona selection.
