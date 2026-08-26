---
target: Research Workspace design brief
total_score: 20
max_score: 40
na_heuristics: ''
p0_count: 1
p1_count: 5
timestamp: 2026-08-24T05-03-31Z
slug: research-workspace-design-brief
---
## Design Health Score

| # | Heuristic | Score | Key issue |
|---|---|---:|---|
| 1 | Visibility of system status | 2/4 | Authority is visible, but stale context, indexing, long-running operations, and partial failures need explicit states. |
| 2 | Match with the real world | 2/4 | Sources, questions, and outputs are natural; Data/Inference/RAG/Studio/ACP/MCP need clearer task language. |
| 3 | User control and freedom | 2/4 | Copy preflight helps, but dirty drafts, active streams, cancellation, switch interruption, and partial transfer are unresolved. |
| 4 | Consistency and standards | 1/4 | Two new top-level destinations conflict with the pinned 13-destination shell and Settings' workspace-lifecycle ownership. |
| 5 | Error prevention | 3/4 | Fail-closed authority and explicit transfer are strong; remote inference over local sources still needs egress consent and fencing. |
| 6 | Recognition rather than recall | 2/4 | Users may have to remember which of Library, Research, Runs, Console, Notes, Study, Artifacts, or Settings owns the result. |
| 7 | Flexibility and efficiency | 3/4 | Batch actions and pane controls are promising, but exact keyboard and persistence contracts are absent. |
| 8 | Aesthetic and minimalist design | 1/4 | Full server-control parity plus fifteen output choices would overload a three-pane terminal screen without stronger hierarchy. |
| 9 | Error recovery | 2/4 | Receipts and remediation are promising; disconnect, stale result, partial Copy, and interrupted generation recovery are incomplete. |
| 10 | Help and documentation | 2/4 | Tour and shortcut help exist, but the first-value path and contextual explanations are not defined. |
| **Total** | | **20/40** | **Acceptable foundation; major contract improvements required before implementation.** |

## Design Specificity Verdict

The design is highly specific to Chatbook's source-to-evidence-to-artifact loop and its local-first trust model. Its weakness is interaction specificity: it currently reads as a complete capability inventory more than a prioritized workbench. The deterministic scan returned no findings because the proposed screen does not exist and the incumbent target is Python/Textual; that result provides no visual clearance.

## Overall Impression

The authority model is the strongest part and should remain. The largest opportunity is to turn exhaustive parity into a single obvious loop: select sources, ask with citations, save or generate a durable result. Everything else should be contextual, progressively disclosed, or routed to its owning screen.

## What's Working

1. Local and Server are modeled as separate authorities with no silent fallback or blending.
2. Existing canonical owners are reused instead of duplicating Notes, conversations, Study records, or specialist artifacts.
3. Research Workspace and durable Research Runs remain distinct responsibilities connected by an explicit bundle handoff.

## Priority Issues

### [P0] Local data authority can be mistaken for local processing

`Data: Local` remains capable of sending source content to a cloud provider. Replace the two independent labels with an effective route such as `Sources: This device -> Inference: Anthropic cloud`, require first-use egress confirmation per workspace/provider, fail closed if the destination cannot be determined, and attach the route/redaction result to generation receipts.

### [P1] The proposed navigation conflicts with the shell contract

The shell is fixed and tested at 13 destinations. Adding two top-level destinations produces 15, shifts positional shortcuts, leaves later destinations without direct keys, and conflicts with ADR-015. Prefer one shell destination, `Research`, with two separately routed screen tabs, `Workspace` and `Runs`, preserving the real `research` route for saved links. If two top-level entries remain required, ADR-015, palette cardinality, overflow, and stable non-positional shortcuts must all be redesigned.

### [P1] Parity inventory lacks action hierarchy and owner boundaries

Ten Studio outputs, five work products, full workspace lifecycle, sharing, source administration, chat modes, notes, and ACP/MCP remediation cannot share equal prominence. Keep one primary action and at most two secondary actions per pane state. Group output discovery as Learn, Analyze, and Present; show the primary five first and put the rest behind `More outputs`. Keep local lifecycle quick actions contextual while routing full local management to Settings. Route ACP/MCP/sandbox actions to their owners.

### [P1] Authority switching needs a transactional contract

Use a screen-local, fail-closed `WorkspaceDataSource`, not app-wide `ActiveSource`, whose bootstrap can silently fall back from Server to Local. Capture immutable request context containing authority, endpoint/profile, principal, workspace, and capability revision. Fence late results. Define behavior for dirty drafts, active streams, imports, generations, partial Copy, server-profile changes, and source deletion. Label server folders/annotations `Device-only overlay - not uploaded or shared`, and key them by server identity plus workspace ID.

### [P1] Local output ownership is not yet implementable as written

Local managed Outputs are unavailable, the Artifacts screen is not a generic output catalog, local Study generation is not workspace-scoped, and TTS only provides an ephemeral result seam. Avoid a speculative universal artifact database. First define a ten-row ownership matrix: canonical store, workspace association, progress/cancel/retry, versioning, reopen destination, provenance, export/share, and unsupported state for Local and Server. Use existing canonical records plus `WorkspaceMembership`; add storage only for output types with no real owner, under a dedicated ADR.

### [P1] Responsive behavior is underspecified for Textual

Equal `1fr` panes are known to produce off-screen or empty layouts. Define wide, medium, narrow, and short-height states from measured pane minimums; use three panes only when budgets fit, two panes with Chat plus the chosen companion at medium widths, and one mounted pane with a persistent labeled mode strip at narrow widths. Preserve drafts, selection, scroll, and semantic focus across reflow. F6/Shift+F6 must cycle only visible panes, and footer hints must remain truthful. Verify production hierarchy and CSS at 160/120/100/84/80/60 columns.

## Persona Red Flags

- **Alex, power user:** two new shell entries break memorized shortcuts; full parity without searchable/grouped actions slows the common loop.
- **Jordan, first timer:** Library, Research Workspace, Research Runs, Deep Research, and Console sound like competing homes; the first screen needs `1 Add sources -> 2 Ask -> 3 Save` guidance.
- **Sam, accessibility-dependent:** pane focus, reorder, resizing, mind maps, slides, audio, tables, disabled reasons, and async state announcements all require explicit keyboard/text alternatives.
- **Riley, stress tester:** switching authority during import/chat/generation, changing server profiles, retrying partial Copy, deleting selected sources, and sharing device-only overlays currently have no defined outcome.

## Minor Observations

- Preserve `research` as the existing run-screen route; do not silently repoint it to the workbench.
- Use a new workspace-local data-source type; `WorkspaceAuthority` already means materialization/sync state and should not be overloaded.
- Key per-authority recents and overlays by endpoint/profile and principal, not just Local/Server.
- Source folders in ADR-028 are filesystem tool roots, not research-source organization folders; use different terminology and storage.
- `Studio/Notes` is ambiguous. Quick Notes should be a subordinate inspector inside Studio, not a peer owner.
- `General` chat risks duplicating Console. Prefer `Grounded` and `Sources off` only if the latter has a clear workspace-Q&A purpose.
- Deep Research needs a durable launch-context link and one canonical return action. Research Runs retains lifecycle ownership; Workspace imports the normalized bundle as a draft artifact/reference.
- Server capabilities are coarse; Chatbook needs a per-output availability projection instead of assuming one text-generation capability means eight output types work.

## Questions to Consider

1. Should the shell expose one Research destination with separately routed Workspace and Runs tabs, or pay the full IA/shortcut cost for two top-level destinations?
2. Should the first release display only the five active outputs plus `More outputs`, or all ten grouped with explicit capability states?
