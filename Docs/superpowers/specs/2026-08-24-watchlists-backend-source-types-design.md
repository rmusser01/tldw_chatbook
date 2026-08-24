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
own display labels. `SourcesPane` maps the five machine values to the existing
plain-language labels: RSS, Atom, Web page, Site, and Forum.

The existing source filter options remain separate and unchanged so users can
filter existing rows whose types are not creatable by this form.

### Pane state and backend changes

`WatchlistsCollectionsScreen` seeds the active backend and its form-supported
types into each rebuilt `SourcesPane`.

If the saved draft type is not in the active contract, only that field falls
back to `rss`. Name, URL, tags, watchlist destination, and the Local cadence and
noise drafts survive the backend change. Switching back to Local restores those
Local-only draft values; it does not silently clear them while their controls
are hidden.

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

### Validation and recovery

Backend services remain the persistence/network boundary. Their validators
reject unsupported payload types before database or API I/O and retain precise
machine-level errors.

If creation fails and the submitted type is outside the active form contract,
the screen shows an error toast with `markup=False`:

- `Local sources don't support 'Playlist'. Choose RSS, Atom, or Web page.`
- `Server sources don't support '<type>'. Choose RSS, Site, or Forum.`

The screen derives this message from the active form contract and its UI label
map; backend services do not depend on presentation labels. Other validation or
unexpected failures retain the existing generic creation-failure copy rather
than being misclassified as source-type errors.

## Testing

Focused tests will prove:

1. Local and Server create-form options match their backend-owned form
   contracts and the filter options remain unchanged.
2. Local-to-Server-to-Local switching normalizes only an incompatible type and
   preserves every other draft value.
3. Local submissions retain cadence and noise fields.
4. Server RSS, Site, and Forum submissions omit Local-only fields and route
   successfully through the real scope/service signature.
5. Unsupported values are rejected before Local DB or Server client dispatch.
6. The exact backend-specific recovery copy is emitted with markup disabled.
7. Existing form focus order and supported-width geometry remain valid for both
   backend variants.

Verification remains scoped to the changed Watchlists service, pane, screen,
and CSS/geometry contracts. No repository-wide suite is required unless the
user requests it.

## Architecture decision

ADR required: no.

This is a bounded correction within the existing Watchlists local/server
routing boundary. It changes neither storage nor the service/API ownership
recorded by existing ADRs.
