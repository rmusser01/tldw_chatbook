# Citation And Snippet Carry-Through Epic Design

Date: 2026-05-23
Owner: `TASK-60.4.4`
Target branch model: stacked PRs

## Purpose

Chatbook needs full evidence carry-through, not just a Library-to-Console handoff. A user should be able to run a Library/Search-RAG query, send grounded evidence into Console, get an answer with explicit source citations, save the result as a Chatbook artifact, and export that Chatbook without losing the source identity, snippets, authority labels, or stale/missing evidence state.

This epic intentionally uses a stacked PR model because full answer-level citation injection crosses several seams: retrieval, handoff, prompt construction, response validation, message persistence, artifact persistence, export, and QA.

## Evidence Source

This work follows the `TASK-60.3` actual-use audit and the `TASK-60.4` deferred tranche plan:

- Library/Search-RAG can stage evidence into Console, but downstream citation/snippet durability remains future work.
- `TASK-60.4.3` clarified workspace/source authority, which is required before source evidence can be trusted downstream.
- The app must not imply grounded answers when no source evidence is attached, stale, blocked by workspace authority, or missing from persistence.

## Epic Branch And PR Model

Create one long-lived epic branch:

```text
codex/citations-snippets-epic -> dev
```

Open one epic PR against `dev` for tracking and review. Each implementation PR branches from the epic branch and targets the epic branch:

```text
codex/citations-contract -> codex/citations-snippets-epic
codex/library-rag-evidence-bundle -> codex/citations-snippets-epic
codex/console-evidence-staging -> codex/citations-snippets-epic
codex/answer-level-citation-injection -> codex/citations-snippets-epic
codex/message-citation-persistence -> codex/citations-snippets-epic
codex/chatbook-citation-export -> codex/citations-snippets-epic
codex/citations-qa-closeout -> codex/citations-snippets-epic
```

The epic PR should merge to `dev` only after all sub-PRs merge into the epic branch and the final actual-use QA pass is approved.

## Sub-PR Plan

### 1. Citation And Evidence Contract

Goal: define durable, sanitized data shapes before touching UI behavior.

Scope:

- Add evidence/citation model types such as `EvidenceReference`, `EvidenceBundle`, `CitationRef`, and evidence status values.
- Include source id, source type, title, snippet text, authority label, workspace id, source owner, content ref, ranking score if available, and stale/missing state.
- Add JSON-safe serialization that strips secrets and rejects unsupported payload types.
- Add pure tests for round-trip, sanitization, truncation, missing evidence, and stale evidence status.

Exit gate:

- Contract tests fail before implementation and pass after.
- No UI, prompt, persistence, or export behavior changes in this slice.

### 2. Library/Search-RAG Evidence Bundle

Goal: convert retrieved Library/Search-RAG results into the evidence contract.

Scope:

- Build evidence bundles from local Library source snapshots and Search/RAG panel state.
- Preserve source identity, snippet text, workspace/source authority labels, and global-browse versus active-context eligibility.
- Attach evidence bundles to the existing Console handoff payload without changing answer generation yet.
- Add mounted tests proving the handoff payload includes evidence snippets and authority metadata.

Exit gate:

- Empty/no-source states remain recoverable and do not claim grounded evidence.
- Cross-workspace blocked sources are labeled as ineligible evidence for active Console context.

### 3. Console Evidence Staging

Goal: make staged evidence visible and actionable before send.

Scope:

- Extend Console staged context and run inspector copy to show evidence count, source authority, and stale/missing evidence warnings.
- Add explicit blocked-state feedback when the user tries to send a grounded/RAG question with no valid evidence bundle.
- Keep source evidence in `ChatHandoffPayload.model_context_block()` in a structured, readable section.
- Add mounted tests for visible evidence state and blocked/no-evidence recovery.

Exit gate:

- Console cannot silently claim RAG/source grounding when evidence is absent.
- Keyboard and button send paths report the same blocked-state feedback.

### 4. Answer-Level Citation Injection

Goal: move from metadata carry-through to actual answer-level citations.

Scope:

- Inject numbered evidence snippets into model prompts using stable citation labels such as `[S1]`, `[S2]`.
- Instruct the model to cite claims using those labels and to say when evidence is insufficient.
- Parse assistant responses for citation markers and validate them against the evidence bundle.
- Mark uncited, unknown, stale, or missing citation references in message metadata and visible UI state.
- Add deterministic tests around prompt construction, citation parsing, unknown citation rejection, and insufficient-evidence recovery.

Exit gate:

- Generated answers can carry validated citation refs.
- Unknown citation markers do not become trusted citations.
- Missing evidence produces honest recovery copy, not a grounded answer claim.

### 5. Message Citation Persistence

Goal: keep answer citations after reload.

Scope:

- Persist evidence bundles and validated citation refs through the existing local chat RAG context/citation storage.
- Reload conversations with assistant message citation metadata intact.
- Render message-level citations or evidence summaries in the Console transcript.
- Add tests for save, reload, and conversation-scope citation retrieval.

Exit gate:

- Closing/reopening a conversation does not drop citation/snippet metadata.
- Existing conversations without citation metadata remain readable.

### 6. Chatbook Artifact And Export

Goal: preserve evidence into saved work and exported Chatbooks.

Scope:

- Include citation refs and evidence manifests when saving a Chatbook artifact from Console.
- Export a user-readable citation section and a machine-checkable evidence manifest.
- Preserve source ids, snippets, authority labels, workspace ids, and stale/missing states.
- Add tests proving Console answer to saved Chatbook to export retains attribution.

Exit gate:

- Exported Chatbooks are understandable to users and parseable by future tooling.
- Missing/stale evidence is preserved as status, not silently dropped.

### 7. Actual-Use QA Closeout

Goal: prove the whole workflow in the actual app.

Scope:

- Run a CDP/textual-web workflow: Library/Search-RAG query to Console answer to saved Chatbook to export.
- Capture actual screenshots for Library/Search-RAG evidence, Console evidence staging, cited answer, Chatbook artifact, and export state.
- Update `TASK-60.4.4`, roadmap, and QA evidence.

Exit gate:

- User approves actual rendered screenshots.
- Focused automated tests pass.
- `TASK-60.4.4` Definition of Done is complete.

## Data Contract Sketch

```python
EvidenceReference(
    evidence_id="S1",
    source_id="note-1",
    source_type="note",
    title="Release notes",
    snippet="The feature is enabled by default...",
    authority_label="Workspace: Research",
    workspace_id="workspace-a",
    source_owner="local",
    content_ref="local:note:note-1",
    status="available",
)

CitationRef(
    evidence_id="S1",
    source_id="note-1",
    quote="The feature is enabled by default...",
    status="validated",
)
```

The contract accepts `text` as an input alias for both `EvidenceReference.snippet` and
`CitationRef.quote` so existing RAG citation models can be adapted without lossy field
renames. Serialized payloads keep `snippet` for source evidence and `quote` for answer
citations because those two fields have different user-facing meanings downstream.

Status values should include at least:

- `available`
- `blocked`
- `missing`
- `stale`
- `unknown`
- `validated`
- `uncited`

## UX Rules

- Evidence count is visible before send.
- Source authority is visible before evidence is used.
- The UI must distinguish "source is visible in Library" from "source is eligible for active Console context".
- Answers without valid evidence must not be labeled as grounded.
- Unknown citation markers are shown as invalid or unverified, not silently trusted.
- Saved/exported Chatbooks must include both readable citations and structured evidence metadata.

## Verification Strategy

Each implementation sub-PR must follow TDD:

- Write the smallest failing test first.
- Verify the failure is for the intended missing behavior.
- Implement the smallest safe fix.
- Rerun focused tests and `git diff --check`.
- For UI-affecting slices, capture an actual CDP/textual-web screenshot and get user approval before merging that sub-PR into the epic branch.

The epic closeout must run the full focused workflow suite for Library, Console, Chatbook artifact save/resume, and export.
