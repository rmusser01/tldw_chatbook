# MCP — MCP servers, tools, permissions, auth, and audit

> 🚧 **This page is a stub.** The full write-up is planned; the sections
> below cover orientation only, except where a section says otherwise. See
> the [guide index](index.md).

## What this screen is for

MCP manages MCP servers, scoped tools, permissions, and audit readiness
(on-screen subtitle: "MCP (Model Context Protocol) lets chatbook use
external tools — most people never need to change anything here."). It's
organized into four modes: Servers, Tools, Permissions, and Audit.

## Getting there

- Click **MCP** in the nav bar, or press **Ctrl+P** → "Switch to MCP".
  (Or press **Ctrl+9** from anywhere.)

## Testing a tool (Tools mode)

Tools mode lists every tool the hub knows about — the app's own built-in
tools plus anything discovered from a connected server — with a Schema
column reading "form" or "raw" so you know before selecting whether a tool
gets a typed form. Selecting a row opens its detail in the inspector on
the right; when the tool is executable, a **Test Tool** button opens a
panel to run it with arguments you choose.

Selecting a tool hides the readiness badge that normally sits at the top
of the inspector (the "Pick a server, tool, or entry…" placeholder, or a
selected server's readiness state) — that badge belongs to server
selection, not tool detail, and reappears once you clear the tool
selection.

### Typed forms

Every one of the app's built-in tools, and any server tool with a
straightforward JSON-Schema, renders as a real form instead of a raw JSON
textarea: text/number inputs with defaults already filled in, checkboxes
for booleans, dropdowns for enums, and a comma-separated text input for a
simple list parameter (e.g. `a, b, c`). A field marked `*` is required.

A schema the form can't represent faithfully — a nested object, a real
mixed-type union, an array of non-simple items — falls back to a raw JSON
textarea for the whole tool instead of silently dropping a parameter it
can't render: "This tool's parameters can't be rendered as a form — edit
raw JSON." The tool can still be tested either way.

### Running it and reading the result

Press **Run**. If the tool is set to **Ask** in Permissions, the first
press arms the button into **Confirm run** ("Ask is set for this tool —
press again to run once.") instead of dispatching — press it again to run,
or do anything else to cancel. A tool set to **Off** never runs at all:
the result reads "Blocked · not run", with "Blocked — this tool is set to
Off in Permissions." underneath.

A completed run shows:

- **A summary line** — e.g. `OK · local · 981ms · 3 results` (outcome,
  where it ran, how long it took, and how many results came back), or
  `Failed · 1.2s` when the call itself failed.
- **A quiet note** underneath, when there's something worth adding to the
  summary: "The tool ran and returned no results." for an empty result,
  the tool's own error text when it reported one, and — alongside either,
  or on its own — a line naming *why* the run was allowed to happen, e.g.
  "Ran because you approved this run (the tool is set to Ask)." or "Ran
  because this tool is set to Allow. Inherited from the global default."
- **A collapsed "Raw response" section** with the full result as JSON —
  secrets redacted, capped at 20,000 characters — for whenever the summary
  isn't enough.

### Permission continuity for built-in tools

An **Allow**/**Ask** choice you've made for one of the app's own built-in
tools survives app updates. A server tool's allow is re-checked against a
stored fingerprint of its description/schema (so a server that quietly
changes what a tool does drops back to Ask) — built-in tools skip that
check entirely, since an ordinary app update that only edits a docstring
must never silently turn your "Allow" back into "Ask".

In Audit mode, a run you confirmed under an Ask gate is recorded with the
decision **approved**, distinct from **allowed** (a tool already set to
Allow) — so the log shows not just that a call reached the tool, but
whether it needed your confirmation first.

## Running more than one copy of the app

Launching a second copy of the app against the same profile never blocks
either one — both keep working. The second instance gets a one-time
warning toast, "Profile already open": whichever instance last changes a
setting or a permission wins, and a restart sweep may mark the other
instance's still-running jobs as interrupted. Detection is an advisory
lock file (`.instance.lock`, inside the profile's data directory) that's
never deleted once created — safe to ignore if you notice it.

---

*Verified against a953e4c1e — 2026-08-04 (PR-5 live check).*
