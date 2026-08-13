# Library Conversations Pagination Design

**Task:** [TASK-15703](../../../backlog/tasks/task-15703%20-%20Make-Library-conversations-list-scrollable-and-paginated.md)

**Date:** 2026-08-12

**Status:** Approved

## Purpose

Library currently requests only the first 50 conversations and renders the
rows in a plain `Vertical` inside a fixed-height canvas. Rows beyond the
terminal's visible height have no dedicated scroll surface, while conversations
beyond the first fetch are absent entirely. The in-canvas filter also searches
only that loaded snapshot.

The Library conversation browser will provide a scrollable, server-backed,
20-row paged display. A user can reach every saved conversation and can filter
the entire collection rather than only the current page.

## User Experience

The existing Export, Select, status, and filter controls remain above the
conversation rows. The rows move into a `VerticalScroll` that owns the flexible
height in the canvas, making every mounted row reachable by wheel, scrollbar,
Tab focus, and Textual's normal focus-following scroll behavior.

A pager remains visible below the list:

```text
Previous    21-40 of 137 · Page 2 of 7    Next
```

Previous is disabled on page 1. Next is disabled on the final page. Empty
results show page 1 of 1 with both controls disabled and retain the existing
empty-result copy. The selected-conversation preview remains below the pager.

Each page contains at most 20 conversations. Changing page resets the row
selection to the first row on the destination page and exits multi-select mode.
Submitting a new filter clears multi-selection, resets to page 1, and searches
the complete conversation collection.

## Architecture and State

`LibraryScreen` will own a dedicated conversation-page state instead of using
the general Library source snapshot as mutable paging storage. It includes:

- the current page records;
- the submitted query;
- the one-based current page;
- the total matching record count and whether that total is known;
- loading and recoverable-error state; and
- a monotonically increasing request generation used to discard stale results.

The general source snapshot remains responsible for Library-wide counts and
samples. Its background refresh must not replace an actively browsed
conversation page or send the user back to page 1. The first successful source
snapshot may seed the unfiltered first page before any dedicated page request
has completed.

`library_conversations_state.py` remains a pure display-state builder. It will
stop applying its own row cap and will accept page metadata from the screen,
deriving the range label, page count, and Previous/Next disabled states. Page
normalization treats an empty result as page 1 of 1 and never permits a page
below 1 or beyond the available final page.

`LibraryConversationsCanvas` renders the scroll viewport and pager controls.
It contains no database logic. Existing row buttons, selection markers,
tooltips, preview, and Console handoff behavior remain unchanged.

## Data Flow

The screen loads a page through the existing
`ChatConversationScopeService.list_conversations()` seam using:

```python
mode="local"
scope_type="all"
query=submitted_query or None
limit=20
offset=(page - 1) * 20
```

The service already returns `items` and `pagination.total`. The screen converts
that response into the dedicated page state and recomposes only the
conversation canvas where practical.

Previous and Next calculate the requested page from the last successful state.
Filter submission sanitizes the input using the existing 200-character
boundary and requests page 1. Paging and filter requests run through a Textual
worker so database work does not block the UI. A newer request generation
invalidates any older response that finishes later.

## Loading and Errors

During a page request, the last successful rows stay visible, paging controls
are disabled, and the status announces loading. This avoids an empty-state
flash and prevents duplicate navigation.

On failure, the last successful page remains visible. The status shows a
recoverable load error and the controls become usable again so the user can
retry by paging or resubmitting the filter. If the initial page fails before
any successful data exists, the canvas shows the error instead of claiming
there are no conversations.

Malformed or incomplete pagination metadata degrades safely: displayed records
remain usable, the known range is shown when possible, and Next is disabled
unless the response proves that another page exists.

## Testing

Pure state tests will cover first, middle, final, single, and empty pages;
range/page labels; disabled controls; and the removal of the old display cap.

Screen/service tests will prove the exact query, limit, and offset arguments;
full-dataset filtering with a page-1 reset; background snapshot isolation;
loading and error preservation; and stale-response rejection.

Textual Pilot tests will mount a constrained-height Library screen and verify
that the conversation list has real overflow, later rows become reachable by
scrolling or focus, pager buttons navigate pages, and filtering can find a
conversation outside the initial page.

## Compatibility and Scope

This change is local-mode Library UI work. It does not alter the conversation
database schema, the conversation service contract, Console conversation
browsing, export formats, or selection semantics beyond clearing selection on
page/filter changes.

## ADR Decision

ADR required: no

ADR path: N/A

Reason: the design consumes the existing paginated conversation-service
contract and changes only bounded Library view state and presentation. It does
not introduce a new storage, ownership, provider, security, or cross-module
interface decision.
