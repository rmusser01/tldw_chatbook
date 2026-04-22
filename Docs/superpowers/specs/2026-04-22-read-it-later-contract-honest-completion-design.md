# tldw_chatbook Read-it-Later Contract-Honest Completion Design

**Date:** 2026-04-22  
**Primary Repo:** `tldw_chatbook`  
**Reference Repo:** `tldw_server2`

## Goal

Finish the currently open `Collections / Read-it-later` follow-up in a way that is honest to the active server contract.

This slice does **not** attempt to invent true per-media-type server saved views inside Chatbook. Instead, it completes the existing local/server `Read-it-later` behavior so that:

- local mode remains fully usable across media types
- server mode remains aggregate-only for saved browsing
- the UI makes that boundary explicit
- invalid state combinations are corrected consistently
- docs and capability language stop implying that Chatbook alone can finish per-type server saved browsing

The desired outcome is a completed, stable, source-honest `Read-it-later` surface that can be followed by the larger `Writing Suite` vertical without leaving an ambiguous parity claim behind.

## Context

The earlier `Media / Read-it-later` vertical already landed substantial real functionality:

- local persisted save/remove behavior
- source-aware scope-service routing
- server save/remove compatibility mapping through `status="saved"` and `status="archived"`
- aggregate `All Media` server saved browsing
- invalid-context guardrails in the UI

Relevant Chatbook seams already exist:

- `tldw_chatbook/Media/media_reading_scope_service.py`
- `tldw_chatbook/Media/server_media_reading_service.py`
- `tldw_chatbook/UI/MediaWindow_v2.py`
- `tldw_chatbook/Widgets/Media/media_search_panel.py`
- `tldw_chatbook/UI/Screens/media_runtime_state.py`

The unresolved question was whether Chatbook could now finish the remaining gap by exposing per-media-type server saved views.

The current server answer is no.

The active server reading list contract in `tldw_server2/tldw_Server_API/app/api/v1/endpoints/reading.py` exposes list filtering by:

- `status`
- `tags`
- `favorite`
- `q`
- `domain`
- `date_from`
- `date_to`
- `sort`

It does **not** expose a media-type filter for reading items. Chatbook therefore cannot honestly implement per-media-type server saved views without either:

- fetching aggregate saved items and locally filtering them into pseudo-type buckets, or
- extending the server contract first

The first option would create silent drift between what the server actually returns and what Chatbook claims the server supports. This design rejects that approach.

## Fixed Decisions

The following decisions are fixed for this slice:

- `Read-it-later` remains inside the existing `Media` destination.
- Local and server browsing remain source-scoped.
- No mixed local/server saved view is introduced.
- Local mode keeps full saved-reading behavior across media types.
- Server mode keeps saved browsing as an aggregate `All Media` view only.
- Chatbook must not fetch aggregate server saved items and locally filter them into fake per-media-type server saved views.
- Invalid server saved-view contexts must be reset explicitly rather than silently degraded.
- User-facing copy should explain that the limit is a current server capability boundary, not a random UI restriction.
- Parity documentation must record that true per-type server saved browsing depends on a server contract change.
- This slice is a cleanup-and-clarification vertical, not a broad `Media` redesign.

## In Scope

- Tighten the current server aggregate-only `Read-it-later` behavior so it is explicit and stable.
- Review and harden the runtime-state rules around:
  - backend switching
  - media-type switching
  - state restore
  - saved-view selector enablement
- Improve user-facing wording wherever server `Read-it-later` is constrained to aggregate `All Media`.
- Keep the scope service as the single source of truth for server saved browsing behavior.
- Ensure tests cover the no-fake-filtering rule and the invalid-context reset rules.
- Update parity docs so they reflect the true dependency on a server media-type filter.

## Out Of Scope

- A new `Read-it-later` destination
- A generic collections system
- True per-media-type server saved browsing without server support
- Client-side pseudo-type filtering of aggregate server saved items
- Sync, dual-write, or mirror semantics
- New server API work in this slice
- Broader media search redesign
- Writing suite implementation

## Approaches Considered

### Option A: Contract-honest completion

Keep server saved browsing aggregate-only, harden the boundary, clarify the UI, and document the remaining server dependency.

Why chosen:

- preserves truthful parity claims
- requires no server API invention
- keeps the existing media seam coherent
- resolves the current ambiguity quickly
- lets execution move cleanly to `Writing Suite`

### Option B: Client-shaped pseudo per-type server views

Fetch aggregate server saved items and locally bucket them by type.

Why not chosen:

- misrepresents server capability
- risks partial and misleading result sets
- creates hidden coupling to local normalization logic
- violates the earlier design rule against client-side fake parity

### Option C: Block on a server contract extension

Do no Chatbook completion work until the server adds a type-qualified reading list filter.

Why not chosen for this slice:

- leaves the current UI boundary ambiguous longer than necessary
- delays the cleanup needed before moving to larger parity rows
- converts a small completion task into a cross-repo dependency immediately

This remains the correct path for future **true** per-type server saved browsing, but not for this Chatbook-only completion pass.

## Chosen Model

`Read-it-later` is treated as two source-specific products that share a UI concept but not a falsely unified browse contract:

- in `local` mode:
  - `Read-it-later` is a real local saved-state collection surface over local media
  - users can browse saved items under any media type context supported by the local store
- in `server` mode:
  - save/remove continues to map to server reading item status
  - saved browsing is only valid in aggregate `All Media`
  - media-type-specific saved browsing is unavailable until the server supports it directly

The UI must never blur these two truths into a fake "same feature everywhere" claim.

## Architecture

### 1. Scope service remains authoritative

`tldw_chatbook/Media/media_reading_scope_service.py` already centralizes local versus server `Read-it-later` routing:

- local browse => local saved-state-aware search
- server browse => server `status=["saved"]`
- local save/remove => local persistence
- server save/remove => server metadata update

This slice keeps that service as the behavioral authority. UI code should not re-interpret the server contract or try to synthesize unsupported server browse shapes on top of it.

### 2. UI state must be capability-aware

`tldw_chatbook/UI/MediaWindow_v2.py` already contains the key guardrails:

- `_saved_view_available_for_context()`
- `_sync_saved_view_controls()`
- `_reset_invalid_saved_view_for_context()`

This slice should tighten those rules so the UI is resilient under:

- switching from local to server mode while a type-specific saved view is active
- restoring saved runtime state after app navigation or screen remount
- changing media type while server `Read-it-later` is selected
- any future code path that attempts to rehydrate an invalid `server + read-it-later + non-all-media` combination

### 3. Search-panel affordances must explain the truth

`tldw_chatbook/Widgets/Media/media_search_panel.py` already exposes the saved-view selector and enable/disable behavior.

This slice should ensure the panel copy and control state communicate:

- local saved browsing is fully available
- server saved browsing is aggregate-only
- the disabled state is a capability limit, not a broken UI

The user should understand why the selector is unavailable in server mode outside `All Media` without needing to infer it from behavior alone.

## User Experience Rules

The intended UX rules are:

1. In local mode, `Read-it-later` behaves like a normal saved filter across supported media types.
2. In server mode, `Read-it-later` is available only from `All Media`.
3. If the user enters an invalid server context while `Read-it-later` is active:
   - the subview resets to `all`
   - the controls update immediately
   - the user receives one clear warning
4. The UI must not silently display a subset of aggregate server saved items and label it as a type-specific server saved view.
5. Restored state must respect the same validity rules as live interaction.

## Capability And Parity Language

The parity docs should explicitly say:

- local `Read-it-later` parity is strong and remains first-class
- server `Read-it-later` parity is currently aggregate-only
- true per-media-type server saved browsing is blocked on a server list-contract extension

This slice should remove any wording that sounds like Chatbook merely has "one more UI pass" left before per-type server saved views exist.

## Verification

Verification should prove boundaries, not just nominal behavior.

Required coverage:

- local `Read-it-later` browsing remains available by media type
- server `Read-it-later` is only available from `All Media`
- switching to a non-aggregate media type in server mode resets the saved subview
- runtime-state restore does not resurrect invalid server saved-view combinations
- saved-view controls stay synchronized with the valid context
- no service or UI code path performs aggregate server saved fetch plus client-side media-type bucketing and then presents it as a true server saved view

## Follow-On Dependency

If future parity work requires true per-media-type server saved views, the next step is **not** more Chatbook shaping. The next step is a server contract addition, most likely a type-qualified reading list filter or an equivalent endpoint that can truthfully return media-type-scoped saved items.

That future work should be tracked as:

- server contract change first
- then Chatbook client wiring
- then parity docs upgrade

until then, Chatbook should remain explicit about the aggregate-only server behavior.

## Success Criteria

This slice is successful when:

- the current `Read-it-later` boundary is fully truthful and user-comprehensible
- local behavior stays fully intact
- server aggregate-only behavior is enforced consistently
- invalid states are corrected deterministically
- parity docs clearly distinguish "landed now" from "blocked on server contract"
- the team can move to `Writing Suite` without leaving a misleading `Read-it-later` parity claim behind
