# Workflows — Reusable procedures, recipes, dry-runs, and outputs

## Threat-intelligence news briefing in Console

This workflow stays inside the core Watchlists and briefing product:

1. Give the Console agent the RSS or Atom URLs and a Watchlist name.
2. Approve the specific local Watchlists tools the agent needs.
3. Have the agent create the sources and Watchlist, then follow every source-check receipt.
4. Have it generate a briefing, follow the briefing receipt, and save an every-24-hours schedule.
5. Read the completed briefing yourself, or ask the agent to open and summarize it for you.
6. Verify the resulting Watchlist and scheduled job in their dedicated views when you need cross-surface confirmation.

The Console agent can consume full briefing content; external MCP clients are
restricted to metadata and receipt status. “Existing model” means the persisted
collection preset, then persisted `chat_defaults`, then the saved model for that
same persisted provider—not the active conversation model.

Threat-hunt hypothesis or document creation is deliberately outside this workflow. Export or hand off content only as a separate, user-directed activity.

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only. See the [guide index](index.md).

## What this screen is for

Workflows manages reusable procedures: runs, dry-runs, and approvals
(on-screen header: "Workflows | Procedures, runs, dry-runs, approvals |
Local | Console handoff"). It's organized into modes: Recipes, Inputs,
Steps, Dry Run, Approvals, and Outputs.

## Getting there

- Press **Ctrl+8**, click **⌃8 Workflows** in the nav bar, or press
  **Ctrl+P** → "Tab Navigation: Switch to Workflows".
