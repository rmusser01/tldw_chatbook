# ADR-251: Served terminal isolation within an authenticated browser login

Date: 2026-10-27
Status: Accepted

Amended 2026-11-01 by ADR244’s “Current-dev integration: terminal page authority over a retained Console.” The original independent-child-per-page and no-reattachment execution contract is superseded for shared served Console operation. Page-ID validation, login-scoped Canvas authority, bounded bindings, asset fingerprinting, late-publication refusal and login-wide channel revocation remain applicable as refined below.
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

### Shared Console compatibility (2026-11-01)

Terminal page identity separates browser/Canvas authority, not execution ownership. Under ADR244, pages for one canonical storage identity attach successively to one retained child; only the current attachment may send terminal input or own the viewport. A replacement revokes the predecessor's terminal and Canvas authority. Binding teardown and awaited Canvas publication must compare the exact attachment generation as well as the retained child and authenticated page scope. Child equality alone cannot identify a retiring page.

The supplied `terminal_id` contract is unchanged: exactly one lowercase 32-hex value, with malformed, empty and repeated values refused; absence retains the legacy login-only scope. The shipped HTML, WebSocket shim and Canvas poll continue to carry the fresh page ID and fingerprinted scripts. Page labels confer no authentication or storage authority. Mounted Canvas requests retain the existing login, shell, cookie and capability checks against a current terminal binding. Expiry/logout/device revocation close login channels without terminating retained execution. Canvas disablement revokes Canvas facilities without closing Console control or the terminal host. The ADR244 amendment defines the resulting retention and physical shutdown boundaries.
