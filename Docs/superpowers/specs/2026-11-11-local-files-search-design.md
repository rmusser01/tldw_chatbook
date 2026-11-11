# Local Files search and cited answers

Status: approved for implementation on 2026-11-11

## Intent

Give people a place in Chatbook to find a file they remember by name, exact words, or meaning, and to ask a question grounded in the matching files. Open **Local Files** from the command palette and use it independently of a Console chat. The source files stay on disk and Chatbook keeps a derived, device-local index for folders the user has explicitly selected.

## Approaches

- **Dedicated Local Files view.** The user selected this approach. It gives folder scope, search status, file locations, and cited answers a single home. The view reuses the Workspace folder index and needs a user-facing adapter for that index's existing authority and evidence contracts.
- **Source in Library Search & RAG.** This puts retrieval in a familiar place. It would expand Library source selection, saved scopes, and evidence dispatch to represent file bindings and their different lifecycle.
- **Standalone CLI command.** This would make searches scriptable. It would also need a way to configure and admit folders, report index status, and present citations outside the TUI.

## First release

Start with one indexed local Workspace folder binding per query, in the native TUI. Accept existing eligible UTF-8 files, including text, Markdown, and code. Deliver filename, path, and content keyword search, preview, and keyword-grounded Ask from `dev`. Add semantic and hybrid retrieval to both Search and Ask when PR #3237's backend is available and the selected binding has a compatible semantic index. Each delivery is usable on its own. Defer cross-folder ranking and PDF, office document, image, audio, and video extraction. [DigUp](https://github.com/ARahim3/DigUp) is the reference for broader file and media support.

The view shows a folder selector and its readiness, a query field, Search and Ask actions, and results with a relative path, line range, and excerpt. Offer a handoff to the exact folder in **Settings → Workspaces** to enable, refresh, disable, or clear its index. A new user can reach folder setup from the empty view. Provide a read-only preview at the cited lines and an explicit **Reveal in file manager** action on supported native hosts. Both use fresh resolution against the selected binding and a live-file check. Use existing form, status, and list components under ADR-150 and ADR-161.

Keyword search works without embeddings, an answer model, or network access. Semantic search defaults to an installed local embedding model during setup. Existing per-folder consent governs any user-selected hosted embedding destination. Show semantic readiness separately and give an actionable keyword fallback when semantic work is unavailable. Report changed files, partial coverage, and a completed search with zero matches distinctly. Indexing and queries run in workers. Keep keyword search available while semantic model loading or search is slow, and drop results when their query or selected folder is superseded.

**Ask** uses the same selected folder and retrieval mode as Search. The user selects an available Chatbook answer provider and sees the destination before sending. A hosted answer requires an explicit confirmation for that request, covering the question and bounded file excerpts. At dispatch, revalidate the exact evidence and effective provider, model, and destination. A change to the query, selected folder, evidence, or destination invalidates the pending approval. With zero citable evidence, display that state without a generation call. Cite relative paths and line ranges. The native view is a separate surface from Library search, content sync, and automatic Console context. A person can bring material into Library through the existing explicit import flow.

## Reuse and authority

- Build on the keyword index in `dev` and the scoped embedding, vector search, and fusion implementation in draft PR #3237. GitHub reports conflicts with `dev`. The draft head passed the Project index contracts job and failed the derived-artifacts check. Reconcile and verify the semantic backend separately from the keyword delivery. Keep its existing per-binding cache, opt-ins, limits, refresh and cleanup semantics. The historical 20,000-item benchmark reported 62–74 seconds for semantic queries. Measure query phases on the integrated build, report its latency, and retain a responsive, independent keyword path. The initial design sets no semantic latency target. [Draft PR #3237](https://github.com/rmusser01/tldw_chatbook/pull/3237) · [Derived-artifacts check](https://github.com/rmusser01/tldw_chatbook/actions/runs/40066100567/job/120805445300)
- Add an application-owned entry point that issues a finite search from the exact selected live binding. Reuse project-index admission, the binding's current exclusions, bounded native Capture, and identity/hash checks for both ranking disclosure and result preview. Treat changed roots, detached bindings, and exclusions as revocation. The view never issues or borrows an agent permission grant. The existing Console tool retains its run-specific read checks.
- Reuse the RAG evidence bundle, answer dispatch, and citation validation. The existing Library answer entry point builds evidence from Library result rows, so share the evidence-consuming portion and preserve the Library adapter. Define a file-evidence adapter with an opaque source identity and a relative path and line range for display. Validate file citations against the exact evidence submitted, and resolve a selected citation through the current binding. A file's content is evidence text and carries no instruction or execution authority.
- Keep a view request tied to the selected binding and query. Search and Answer publish only against that request and its current authority. Avoid persisting questions, excerpts, or answers in conversations, logs, or Library. Existing project cache storage, exclusion, and lifecycle rules apply. Folder and query content stays on device when using keyword search and local providers.

## Acceptance and verification

- From the command palette, reach Local Files, select an indexed folder, and find an exact filename or a phrase inside a file with no embedding dependency. Verify the selected result opens an in-app excerpt at its cited line and can reveal the current original on a supported host.
- With an installed local model, a paraphrase retrieves matching file content and hybrid search preserves an exact-name match. Report keyword and semantic timings separately and retain a mounted test that the UI stays responsive while model work is held.
- Changing the selection, excluding a file, refreshing, or detaching while work is in flight prevents the old result or citation from appearing. After source edits or deletion, cached text cannot be displayed or sent to an answer provider.
- An answer cites the submitted file evidence. Zero evidence and declined external disclosure cause zero generation requests. A selected local provider can answer while external network access is blocked. A hosted-provider control records the exact approved destination and request evidence.
- Compare Library and Console behavior before and after enabling or clearing a folder index. Existing search scopes and tool permission checks still apply. Use targeted service, mounted Textual, and provider-boundary tests plus a real offline model smoke test.

ADR required: yes. Amend `backlog/decisions/241-workspace-project-context-and-folder-bound-index.md` for user-initiated search authority and `backlog/decisions/024-rag-citation-provenance-and-source-resolution.md` for file citations. The existing ADRs own those contracts. The design introduces a new application search entry point and a new source of citable evidence.
