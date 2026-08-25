# Vendored tiktoken encodings

These files are a tiktoken cache directory, pointed at by `TIKTOKEN_CACHE_DIR` from
`Tests/conftest.py`. They are committed so that no test needs the network to tokenize.

## Why the filenames are opaque

They are not names anyone chose. tiktoken derives each cache entry's filename as
`sha1(<download url>)`, so the mapping is:

| encoding | source URL | filename |
|---|---|---|
| `gpt2` (vocab) | `…/gpt-2/encodings/main/vocab.bpe` | `6d1cbeee0f20b3d9449abfede4726ed8212e3aee` |
| `gpt2` (encoder) | `…/gpt-2/encodings/main/encoder.json` | `6c7ea1a7e38e3a7f062df639a5b80947f075ffe6` |
| `cl100k_base` | `…/encodings/cl100k_base.tiktoken` | `9b5ad71b2ce5302211f9c61530b329a4922fc6a4` |
| `o200k_base` | `…/encodings/o200k_base.tiktoken` | `fb374d419588a4632f3f557e76b4b70aebbca790` |

(base URL `https://openaipublic.blob.core.windows.net`)

`Tests/test_tiktoken_vendored_cache.py` recomputes those hashes, so a renamed or corrupted
file fails loudly rather than silently falling back to a download.

## Why these three and not others

Only what the suite actually asks for. `gpt2` is the chunking engine's default tokenizer;
`cl100k_base` is the fallback every `encoding_for_model` miss lands on; `o200k_base` is what
`encoding_for_model` returns for gpt-4o-class model names, which the Console UI suites use.

`o200k_base` was dropped once and put back. The first scope check sampled Chunking,
Subscriptions, RAG_Search and Utils — none of which name a gpt-4o-class model — and concluded
nothing asked for it. A 95-file Console/Library run then requested it 53 times. If you are
deciding whether to add an encoding here, sample widely: the requester is whichever suite
happens to name a model, not whichever suite looks like it tokenizes.

`p50k_base` and `r50k_base` remain absent; nothing has requested them across either sample.

If a future change asks for another encoding, the guard test will not catch it (it only
checks what is here). The symptom is the one this directory exists to prevent: an egress
attempt to `openaipublic.blob.core.windows.net` recorded by the network guard. Add the
encoding here rather than granting the test network access.

## Refreshing

From outside pytest, where the network is available:

```
TIKTOKEN_CACHE_DIR=Tests/fixtures/tiktoken_cache python -c \
  "import tiktoken; tiktoken.get_encoding('gpt2'); tiktoken.get_encoding('cl100k_base')"
```

These are stable published artifacts; they should not need refreshing unless tiktoken
changes where it fetches from, which would change the hashes and fail the guard.
