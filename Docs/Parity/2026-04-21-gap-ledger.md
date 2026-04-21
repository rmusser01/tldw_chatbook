# Chatbook Server Parity Gap Ledger

## Critical Gaps

## High-Value Partial Crosswalks

## Local Name Crosswalks

- `Subscriptions` -> likely local precursor for server `Watchlists`

## Remote-Only Client Obligations

## Contract-Maturity Holds

- `Client Notifications`: client-local notification state is not directly mirrored by a dedicated server contract; the adjacent remote notifications/reminders/feed surface is covered separately, so keep the client-local contract on hold until a direct server analog appears.
- `Research Search / Provider Surfaces`: the visible route family is legacy and deprecated in `research.py`, while provider behavior is split across core third-party modules; keep the client contract on hold as present but low-confidence.
- `Remote MCP Control Plane / Governance`: the surface is admin-heavy and the current evidence only confirms backend governance plumbing, not a clearly client-scoped governance contract; keep it on hold until the client-facing boundary is clearer.

## Deferred / Explicitly Out Of Scope
