# I2 independently authored interpretation fixtures

The original synthetic fixture bytes and expected inventories are original test data under this
repository's AGPL-3.0-or-later license. In those synthetic families, no upstream prose, executable, schema,
plugin asset or skill body is vendored or executed. `expected.json` was written
independently of the adapter. The controlled rejection scripts must never run.

| Fixture | Primary format reference / upstream revision | Source licensing context |
| --- | --- | --- |
| portable-openai | [Agent Plugins 1.0.0](https://agent-plugins.org/specification), [OpenAI packaging](https://developers.openai.com/plugins/build/plugins); OpenAI repository pin below | Agent Plugins docs CC BY 4.0; fixture AGPL |
| codex | [OpenAI plugin repository](https://github.com/openai/plugins/tree/5fd93af4cd0c623e020d0cc7e9ce178b4ac1f70f/plugins/figma), commit `5fd93af4cd0c623e020d0cc7e9ce178b4ac1f70f` | Referenced manifest declares LicenseRef-Figma-Developer-Terms; no Figma material copied; fixture AGPL |
| cursor, cursor-catalog | [Cursor template](https://github.com/cursor/plugin-template/tree/46216072ac5750f782f95bb325b4d12b7c3ae9c9/plugins/starter-advanced), commit `46216072ac5750f782f95bb325b4d12b7c3ae9c9` | Source manifest declares MIT; no license file in the inspected template tree; no source copied; fixture AGPL |
| cursor-guard, cursor-observer | Same Cursor template revision; [Cursor hooks schema 1](https://cursor.com/docs/hooks) observed 2026-10-01 | No hook code/prose copied; fixture AGPL |

Packaging contracts were checked against official documentation on 2026-10-01:
[Cursor plugin reference](https://cursor.com/docs/reference/plugins),
[Codex hook contract](https://learn.chatgpt.com/docs/hooks), and the versioned
Agent Plugins schema. Unversioned web docs are dated observations, not immutable
upstream releases or original-host certification. That historical interpretation remains pinned to `2026-10-01.1`. Fresh inspections
use `/2`; qualifying source-hook codecs requires a new review and current host gates.

Parser evidence: deterministic path/overlay precedence, explicit exclusions,
normalized manual metadata, tool/model constraints and visible unsupported nodes.
Native behavior evidence: separately exercises actual reviewed Console instruction
injection and current readiness. Under the retained dated dev interpretation, foreign hooks remain unsupported: native envelopes
cannot supply the complete vendor payload/cwd/permission/transcript contract, and
native output/timeout/success-only semantics cannot be borrowed by event-name match.
A simple POSIX argv or a passing parser is not executable qualification. Required or
unknown-scope guards fence activation; unsupported optional observers stay visible.
No source script is launched; no original Codex/Cursor host comparison is claimed.

Executable catalog inputs are retained as a bounded regular
`.chatbook-plugin/catalog.json` package member. Manifest fields take precedence
wholesale over its fields. Normal snapshot materialization, review and recovery
capture these bytes. For the historical dev recipe an external overlay without identical retained bytes
stays unavailable. Fresh `/2` reviews bind an explicit external overlay through a
bounded canonical carrier; a conflicting captured catalog refuses. No new store is added.


# Inert interoperability captures

These selected upstream files are parser evidence. They are never executed,
installed, or connected by the tests. Runnable behavioral controls are separately
authored temporary packages in the three adapter test files.

`upstream/provenance.json` records immutable commits, original file URLs, exact
SHA-256/size/blob evidence and independent selected inventories:

- Cursor teaching: two skills, no hooks/MCP/agents/rules/commands. Its manifest,
  both complete skill files and MIT notice match their same-commit Git blobs.
- Codex Linear: one HTTP MCP declaration and one separate app mapping, zero
  skills. All three JSON bodies match their blobs. MIT is a **manifest
  declaration only**; the same-commit package listing has no LICENSE. Assets and
  prose are omitted, with no claim about their rights. OAuth/app metadata grants
  no permission and does not establish a connection.
- Portable arch: zero core inventory. The manifest matches its blob. Its absent
  extension skill is never followed outside the package. The same-commit root
  MIT notice was retrieved from its pinned raw URL but **was not independently
  matched against the repository root tree**.

API tree/catalog JSON is provenance data, not an upstream file-blob match.
Omitted assets and README files mean these are selected captures, not complete
installed packages. No original Cursor/Codex execution or runtime-equivalence
comparison has been performed. Keep copied bytes and supplied notices intact.
