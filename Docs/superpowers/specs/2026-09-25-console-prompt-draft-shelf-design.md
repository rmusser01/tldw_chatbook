# Console Prompt Draft Shelf Design

## Goal

Let a Console user preserve an unfinished message without sending it, return
to it later, edit or insert it safely, and promote it into the existing Prompt
Library when it becomes reusable.

Success means a first-time user can save without losing the current composer,
while an experienced user can use the command palette, search a bounded shelf,
and move between shelf drafts and Library Prompts without leaving the Prompt
Workbench.

## Product Decisions

- The existing Prompt Workbench gains a **Draft Shelf** source beside Local and
  Server. There is no second browser.
- The composer Menu and command palette expose **Save draft to shelf…**.
- Saving asks the user to choose **Save and keep** or **Save and clear**.
  Clear happens only after persistence succeeds.
- The shelf is local-only and holds 100 entries. At capacity, saving is blocked
  until the user deletes an entry. Nothing is evicted automatically.
- Draft content is plain text. The saved value is the composer's canonical
  draft, so collapsed paste tokens contribute their full underlying text.
- Promotion creates a local Library Prompt and can assign one existing local
  collection. The shelf entry remains after promotion.
- Insert places the draft at the current caret using paste semantics. It never
  overwrites the rest of the current message.
- No Ctrl+S binding is added; the command palette is the keyboard-first route.

## Primary Flows

### Save from the composer

1. The user opens Menu or the command palette and chooses **Save draft to
   shelf…**.
2. A compact decision dialog states that nothing will be sent and offers Save
   and keep, Save and clear, and Cancel.
3. The app captures the canonical composer draft and creates one shelf entry.
4. On success, the app either preserves the composer exactly or clears it and
   writes the cleared session draft. On failure, the composer is unchanged.
5. At 100 entries, the dialog explains that the shelf is full and offers a
   clear recovery instruction to open Prompt Workbench > Draft Shelf.

### Browse and edit

1. The user opens Prompt Workbench and selects Draft Shelf.
2. Search filters draft content; paging remains bounded and stale completions
   cannot replace a newer source/query result.
3. Each row shows a derived first-line title, a short preview, and updated time.
4. Opening a row shows one editable plain-text area plus Update, Insert at
   caret, Save to Library, and Delete actions.
5. Update and delete use the reviewed version. A stale result stays open and
   tells the user to reload rather than overwriting newer content.

### Promote to Library

1. The editor asks for a required Prompt name and an optional existing local
   collection.
2. Save creates a legacy local Prompt with the draft as its User lane.
3. If a collection is selected, the existing local membership API assigns it.
4. The UI reports Prompt creation and collection assignment separately. The
   draft remains on the shelf in every outcome.

## Information Architecture and Interaction

- Prompt Workbench retains its current modal shell, header, Back/Close
  behavior, and composer-focus restoration.
- The source selector reads Draft Shelf, Local, Server. Draft Shelf copy never
  calls itself a Library or implies server availability.
- Search stays focused while results update. Arrow keys change a synthetic row
  highlight; Enter opens it. Pointer activation follows the same selected-row
  path.
- Escape from a clean editor returns to the shelf; Escape from the root closes
  the workbench and restores the composer. Dirty text receives the existing
  Keep editing / Discard guard.
- Delete is two-press: the first press changes the action to **Press again to
  delete** and describes the permanence; any intervening edit/navigation
  cancels arming.

## Truthful States

- Loading: name the source and operation.
- Empty: explain how to save the first draft from the composer.
- No matches: preserve the source and invite a query change.
- Full: state **100 of 100** and direct the user to delete an entry.
- Save failure: keep all composer/editor text and offer Retry.
- Stale edit/delete: keep the user's text, refuse the mutation, and offer
  Reload.
- Promotion membership failure: say the Prompt was saved but remains unfiled.

## Storage and Service Contract

ADR-184 governs ownership. `PromptScopeService` exposes local-only draft
create, page/search, detail, update, and delete operations backed by
`LocalPromptService`. The service rejects server mode, validates content and
bounds, and normalizes results for the Console. The storage table is excluded
from Prompt sync, FTS, history, and export code paths.

## Non-goals

- Cross-device draft sync.
- Automatic eviction or pinning.
- Draft folders, tags, or multiple collection assignments during promotion.
- Restoring collapsed-token visual metadata after insertion.
- Model-assisted editing inside the Draft Shelf editor; Improve remains a
  separate existing workbench path.

## Validation

- Service tests cover exact content, pagination/search, version conflicts,
  hard delete, local-only routing, and the atomic 100-entry refusal.
- Widget/controller tests cover Save and keep/clear, failure preservation,
  full-shelf recovery, stale result rejection, search focus, keyboard row
  movement, insertion at the caret, two-press delete, and promotion with and
  without collection assignment.
- Live UAT covers wide and 80x24 terminals, first-time save/browse, source
  switching, keyboard row opening, caret insertion, two-press delete, and
  focus-driven compact scrolling. Targeted automated coverage verifies
  collapsed-paste preservation and the 100-entry full state.
