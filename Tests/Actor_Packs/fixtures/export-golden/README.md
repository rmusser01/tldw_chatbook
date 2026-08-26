# Actor Pack export goldens

These two archives pin the complete deterministic ZIP bytes for the smallest
Character and Persona exports. The tests inspect their ZIP metadata, manifest,
inventory, content digest, and member digests independently of the writer.

Regenerate only after an intentional Actor Pack V1 byte-contract change:

```bash
PYTHONPATH=. python Tests/Actor_Packs/fixtures/export-golden/generate.py
```
