# Watchlists backend-aware source types

**Task:** TASK-2510
**Status:** Approved design
**Date:** 2026-08-24

## Purpose

Make the Watchlists New source form offer only source types that its active
backend can create. Remove the current dead-end choices and give stale or
programmatic invalid values an actionable, backend-specific recovery message.

## Current problem

The create form and source filter share one option list. The form therefore
offers `Feed`, `Playlist`, and `Channel`, although the local service rejects all
three. The resulting toast says only `Failed to create source.`

Local and Server also have different contracts:

- the Local form can configure RSS, Atom, and Web page sources;
- the Server API accepts RSS, Site, and Forum sources;
- Local accepts additional programmatic/import payload types that the form
  cannot configure and must not advertise;
- the current form sends Local-only cadence and noise fields to the Server
  method, whose signature rejects them.

## Goals

- Local New source choices are RSS, Atom, and Web page.
- Server New source choices are RSS, Site, and Forum.
- The source-list filter keeps its broader existing vocabulary.
- Switching backends preserves the user's free-text draft while normalizing an
  incompatible type to RSS.
- Local-only cadence and noise controls are absent from the Server form and are
  not submitted to the Server backend.
- Invalid stale or programmatic values fail before persistence or network I/O
  and produce backend-specific recovery copy.

## Non-goals

- Do not make the form configure Local sitemap, API, URL-list, podcast, or JSON
  Feed payloads.
- Do not reinterpret Feed, Playlist, or Channel as aliases.
- Do not change stored source types, database schema, remote API schemas, the
  source filter vocabulary, or imported-source behavior.
- Do not redesign the create form or introduce a shared application-wide form
  framework.

## Design

### Backend-owned form contracts

Each backend service publishes an ordered tuple of machine-readable source
types supported by this form:

- Local: `rss`, `atom`, `url`
- Server: `rss`, `site`, `forum`

This is deliberately distinct from the Local service's complete accepted
payload vocabulary. The latter remains broader for programmatic and imported
sources.

`WatchlistScopeService` returns the active backend's form contract. It does not
own display labels. `SourcesPane` maps the five form-supported machine values
to the existing plain-language labels: RSS, Atom, Web page, Site, and Forum.
The pane also retains the broader existing source-type label registry used by
the filter, including known legacy labels such as Playlist.

The existing source filter options remain separate and unchanged so users can
filter existing rows whose types are not creatable by this form.

### Pane state and backend changes

`WatchlistsCollectionsScreen` seeds the active backend and its form-supported
types into each rebuilt `SourcesPane`. A backend change also pushes the new
backend and contract into an already-mounted pane immediately; it must not wait
for an unrelated region rebuild. The pane recomposes its open form from that
new contract, so the visible options and fields always match the backend shown
in the screen header.

If the saved draft type is not in the active contract, only that field falls
back to `rss`. Name, URL, Active state, tags, watchlist destination, and the
Local cadence and noise drafts survive the backend change. Active state and
cadence therefore join the existing screen-mirrored draft contract rather than
being read only from controls that a recompose can destroy. Switching back to
Local restores the Local-only cadence and noise values; it does not silently
clear them while their controls are hidden. The create form remains open across
the backend change.

The mounted Select is the submission source of truth, preserving the existing
form contract. Normal interaction therefore cannot submit an unsupported type.

### Backend-specific fields and payloads

The Local form keeps cadence and, for URL-family types, noise-selector controls.
The Server form omits both because the shipped Server create contract does not
define those Local fields.

Local submissions retain `check_frequency` and `ignore_selectors`. Server
submissions contain only the fields accepted by `ServerWatchlistsService`:
name, URL, source type, active state, and tags. Existing watchlist-destination
gating remains unchanged.

The submission captures the pane's backend together with its payload. The
screen passes that captured backend through the controller and scope service
and uses it when filing the created source into its chosen watchlist and
building the confirmation. An immediately following backend-selector change
therefore cannot reroute an already submitted source or make a Local source
lose its Local destination. Post-completion list refreshes still target the
backend currently visible to the user.

### Validation and recovery

Immediately before posting `CreateSourceRequested`, `SourcesPane` compares the
mounted Select's submitted type with the contract currently rendered by that
pane. A mismatch stops event dispatch, keeps the populated form open, and shows
the recovery message below so the user can choose a supported type in place.
This form-boundary guard is required even for values such as Local `sitemap`
that the persistence service accepts for programmatic or import callers but
this form cannot configure.

Backend validators remain defense in depth at the persistence/network boundary.
They continue validating against each service's complete accepted payload
vocabulary, rejecting values outside that broader contract before database or
API I/O and retaining precise machine-level errors.

When the submitted type is outside the active form contract, the pane stops
creation and shows an error toast with `markup=False`:

- `Local sources don't support 'Playlist'. Choose RSS, Atom, or Web page.`
- `Server sources don't support '<type>'. Choose RSS, Site, or Forum.`

The pane derives this message from its rendered form contract and UI labels;
backend services do not depend on presentation labels. For the rejected value,
it first uses the broader existing source-type label registry, so `playlist`
renders as `Playlist`. If no label is registered, the fallback is computed
exactly as follows: coerce the submitted value to text, remove terminal control
characters with the existing `strip_control_characters` helper, collapse every
whitespace run to one ASCII space, and trim the result. If it exceeds 40
characters, retain the first 39 and append `…`; if it is empty, display
`Unknown`. Toast markup remains disabled, so arbitrary stale values are
rendered as inert text. Other validation or unexpected failures retain the
existing generic creation-failure copy rather than being misclassified as
source-type errors.

## Testing

Focused tests will prove:

1. Local and Server create-form options match their backend-owned form
   contracts and the filter options remain unchanged.
2. A mounted, open form updates immediately on Local-to-Server-to-Local
   switching, normalizes only an incompatible type, stays open, and preserves
   name, URL, Active state, tags, destination, cadence, and noise drafts.
3. Local submissions retain cadence and noise fields.
4. Server RSS, Site, and Forum submissions omit Local-only fields and route
   successfully through the real scope/service signature.
5. The form-boundary guard rejects form-unsupported values, including values
   that the broader Local persistence contract accepts, before event,
   controller, Local DB, or Server client dispatch; the populated form remains
   open for correction.
6. Backend validators still reject values outside their complete service
   contracts before Local DB or Server client dispatch.
7. Exact backend-specific recovery copy is emitted with markup disabled for a
   known legacy label and for unregistered machine values containing controls,
   whitespace, excessive length, or no displayable text.
8. A submitted request, destination filing, and confirmation remain bound to
   the backend shown when Create was pressed even if the selector changes
   before the worker executes; the visible backend is the one refreshed.
9. Existing form focus order and supported-width geometry remain valid for both
   backend variants.

Verification remains scoped to the changed Watchlists service, pane, screen,
and CSS/geometry contracts. No repository-wide suite is required unless the
user requests it.

## Architecture decision

ADR required: no.

This is a bounded correction within the existing Watchlists local/server
routing boundary. It changes neither storage nor the service/API ownership
recorded by existing ADRs.
