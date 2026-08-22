# Chunking Template System Migration Guide

This guide helps you migrate from plain chunking options to the template
system. It was rewritten for the template-parity work (2026-08-21): the
system this guide used to describe — a JSON **file store** under
`~/.config/tldw_cli/chunking_templates/`, a `ChunkingTemplateManager`, and
bare-name `Chunker(template="words")` calls — was **deleted**. Templates are
now database rows resolved at the service layer, and `Chunker` accepts only
pre-resolved template **dicts**.

## What exists now

- **Templates are DB rows** in the media database's `ChunkingTemplates`
  table (schema v7): soft-deleted, validate-on-write, with the server's six
  built-ins seeded. Manage them through RAG Admin (Library → RAG controls)
  or the interop service (`Chunking.chunking_interop_library`).
- **One resolution path**: `tldw_chatbook.Chunking.template_runtime.resolve_template(db, name)`
  turns a name into the flat template dict. Nothing else resolves names.
- **Ingest picks a template in the Library import form** ("Chunking
  template", defaulting to "None (manual settings)") or via config
  (`[chunking] default_template`); the choice is stored per imported item in
  `Media.chunking_config`.

## Migration examples

### Before (plain options — still works, unchanged)

```python
from tldw_chatbook.Chunking.Chunk_Lib import Chunker

chunker = Chunker(options={
    'method': 'words',
    'max_size': 500,
    'overlap': 100,
})
chunks = chunker.chunk_text(text)
```

### After (template)

```python
from tldw_chatbook.Chunking.Chunk_Lib import Chunker
from tldw_chatbook.Chunking.template_runtime import resolve_template

template = resolve_template(media_db, "academic_paper")  # -> dict | None
if template is None:
    raise SystemExit("no such live template")

chunker = Chunker(template=template)
chunks = chunker.chunk_text(text)

# Or override template options (explicit options win):
chunker = Chunker(
    template=template,
    options={'max_size': 500, 'overlap': 100},
)
```

A bare name string **raises** (`TemplateError`): the file store that used to
resolve names is gone, and guessing it back would silently chunk differently
than the DB row says.

### Pipelines

`Chunker(template=...)` applies the template's chunk-stage options only.
Executing a template's full pre/chunk/post pipeline — with the synthesized
flat chunk contract (offsets, indices, word counts, `metadata.offset_basis`)
— is `template_runtime.apply_template`'s job:

```python
from tldw_chatbook.Chunking.template_runtime import apply_template

chunks = apply_template(template, text, options={"overlap": 0})
```

## What was deleted (do not carry forward)

| Old spelling | Replacement |
|---|---|
| `Chunker(template="words")` (bare name) | `Chunker(template=resolve_template(db, "words"))` |
| `from tldw_chatbook.Chunking import ChunkingTemplateManager` | RAG Admin / `chunking_interop_library` CRUD |
| `~/.config/tldw_cli/chunking_templates/*.json` | `ChunkingTemplates` DB rows |
| `[chunking_config] template = "..."` in config.toml | `[chunking] default_template = "..."` |
| `templates/example_usage.py` | the examples in this guide |

## Precedence

`Chunker` merges `defaults <- template <- explicit options`: a template's
options beat the built-in defaults, and only an explicitly passed option
beats the template. On the Library import form the same ruling applies —
a picked template beats the form's untouched defaults, and only a value you
changed in the form overrides the template.

## FAQ

**Q: Do I have to use templates?**
A: No. Plain options behave exactly as before; the import form's default is
"None (manual settings)".

**Q: What happens if a template name stops resolving (deleted/renamed)?**
A: The import fails that item with a named error (`TemplateResolutionError`)
instead of silently falling back to different chunking. Re-chunk (PR E)
skips and counts such items.

**Q: Are templates slower?**
A: No — resolution is one indexed DB read, cached by the caller if needed.

**Q: Where is the per-item choice stored?**
A: `Media.chunking_config` (`{"template": "<name>", ...}`), plus the
`chunking_template` / `chunking_params` columns on the stored chunk rows.
