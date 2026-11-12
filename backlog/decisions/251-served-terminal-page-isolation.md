# ADR-251: Served terminal isolation within an authenticated browser login

Date: 2026-10-27
Status: Accepted
Task: TASK-37143

## Context

A browser login cookie is shared by tabs, while every terminal WebSocket starts
an independent AppService. Binding that child directly to the login ID rejects
the second tab before startup. Rebinding would instead transfer the first tab's
Canvas authority to its sibling.

## Decision

Every served HTML render supplies a fresh random 32-hex terminal page ID. The
owned browser scripts carry it on the terminal WebSocket and Canvas session
poll. The server validates the ID and composes it with the authenticated login
ID before using the existing exactly-one-child binding. A page ID labels a
terminal beneath its login; it grants no authentication or cross-login authority.
No page-render reservation state is stored. Active terminal bindings are bounded
and their exact child incarnation releases its slot during teardown. Canvas
disabling revokes broker, gateway and launches while retaining active terminal
bindings until their WebSockets retire; disabling Canvas does not stop terminals.
Every awaited snapshot revalidates the captured child, broker and enabled state
before publishing a Canvas launch. A response from a retired page cannot recreate
its revoked shell or consume a new gateway slot.
Exact-child terminal teardown fully revokes its Canvas session, pending shell and
capabilities. Retaining an unavailable event lane would strand the gateway's
finite slots after repeated fresh-page refreshes; a retired terminal has no
mounted authority and no further use for that lane.

The login cookie, same-origin and CSRF checks, expiry, device revocation and
logout remain login-wide. Mounted Canvas URLs must match an active terminal of
the authenticated login as well as the gateway's existing shell binding. The
login ID alone remains a compatibility scope for clients omitting the page ID;
the shipped page always supplies its fresh ID. Changed owned scripts receive
content-fingerprinted URLs so cached scripts cannot erase the page identity.

## Alternatives

Allowing a second child to overwrite a login binding would hijack the first
terminal's Canvas. Sharing one child among tabs changes terminal input and
application ownership. A separate cookie per tab cannot work because browser
cookies are shared. Pending-page registries or signed page capabilities add
state and lifecycle without extending authority: authentication already scopes
every accepted page label to its owning login.

## Consequences

Tabs and refreshes own independent terminals and Canvas scopes. A slow departing
child cannot block a fresh page or release its binding. Logout and expiry close
all sibling terminals. Existing direct fixtures and old clients retain their
single-terminal compatibility scope. The protocol does not offer reconnecting
a new WebSocket to an existing child.
