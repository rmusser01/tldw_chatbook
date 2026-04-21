# Chatbook Server Parity Gap Ledger

## Critical Gaps

## High-Value Partial Crosswalks

## Remote-Only Client Obligations

## Contract-Maturity Holds

- `Study Core study guides`: the study-guide surface is visible in `core/Chat/document_generator.py` and wired through `chat_documents.py`, but no distinct endpoint family surfaced, so keep the client contract on hold as core-only / immature.
- `Study Packs`: the server surface is still schema-backed and nested in `core/StudyPacks` plus `ChaChaNotes_DB`; there is no dedicated endpoint family yet, so keep the client contract on hold until a direct API surface appears.
- `Research Search / Provider Surfaces`: the visible route family is legacy and deprecated in `research.py`, while provider behavior is split across core third-party modules; keep the client contract on hold as present but low-confidence.
- `Client Notifications`: only adjacent notification/feed infrastructure surfaced in `notifications.py`; keep the client contract on hold because no dedicated client-notification server contract was identified.
- `Remote MCP Control Plane / Governance`: the surface is admin-heavy and the current evidence only confirms backend governance plumbing, not a clearly client-scoped governance contract; keep it on hold until the client-facing boundary is clearer.

## Deferred / Explicitly Out Of Scope
