# Personas Markdown Character Import Design

## Task

TASK-140 - Support Markdown character card imports.

## Goal

Let users import Markdown files from the ds-native Personas character import flow when the Markdown embeds the same character-card data formats already supported by the lower-level importer.

## Current Context

The ds-native Personas import picker currently advertises character card imports as `.json` and `.png`. The lower-level character import path already reads text files and parses character-card data from direct JSON, YAML frontmatter, or fenced JSON blocks before saving through the existing character-card import helper.

The gap is mostly the Personas picker affordance and regression coverage. A user with a valid `.md` character card can bypass the picker only indirectly; the normal import flow does not make Markdown visible as a supported card source.

## Scope

In scope:

- Add `.md` / `.markdown` to the ds-native Personas character import picker filters.
- Keep JSON and PNG import behavior unchanged.
- Add tests for Markdown import paths that already match the importer contract.
- Add tests for invalid Markdown failing through the existing import failure path.

Out of scope:

- A new heading-based or prose-based Markdown character schema.
- Markdown-to-character inference.
- Bulk imports.
- Schema or DB changes.
- Changes to avatar upload behavior.

## Supported Markdown Forms

Markdown import is valid only when the file contains existing character-card data in one of these forms:

1. YAML frontmatter containing the character card object.
2. A fenced JSON block containing the character card object.

The object inside either wrapper must still satisfy the existing V1/V2 character card parser. Invalid Markdown should not create or select a character.

## Proposed Flow

1. User chooses Import from the Personas character mode.
2. `EnhancedFileOpen` offers `.json`, `.md` / `.markdown`, and `.png`.
3. The selected path continues through `_import_character_from_path()`.
4. `_import_character_from_path()` continues delegating to `ccp_character_handler.import_character_card()`.
5. The lower-level importer reads text content and extracts/parses the embedded character-card object.
6. Success refreshes, selects, and reveals the imported character exactly like JSON/PNG import.
7. Failure uses the existing import failure notification.

## Alternatives Considered

### Recommended: Picker + Regression Coverage

Expose Markdown in the ds-native picker and add parser/import tests around the existing text-card contract.

Pros: smallest change, reuses existing parser, does not create a second format, low risk to the open PR.

Cons: Markdown must embed structured card data; prose-only Markdown remains unsupported.

### New Human-Readable Markdown Schema

Parse headings like `# Name`, `## Personality`, and `## Scenario`.

Pros: easier for humans to write from scratch.

Cons: ambiguous, larger parser surface, likely to diverge from V1/V2 card semantics, and out of scope for this follow-up.

## Tests

Add focused coverage for:

- Personas import picker filters include Markdown alongside JSON and PNG.
- `_import_character_from_path()` accepts `.md` paths and routes them through the existing import helper.
- `load_character_card_from_string_content()` parses Markdown fenced JSON with valid V2 card data.
- `load_character_card_from_string_content()` parses Markdown YAML frontmatter with valid card data.
- Invalid Markdown returns no parsed card and the screen import flow shows the existing failure notification without changing the current selection.

## ADR Check

ADR required: no

ADR path: N/A

Reason: this is an import picker/parser affordance and regression coverage using existing import and storage boundaries. It introduces no new storage schema, persistence policy, sync policy, provider/runtime boundary, or long-lived Markdown schema.
