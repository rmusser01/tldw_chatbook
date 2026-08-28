# Packaging

Chatbook builds a source distribution and wheel with setuptools. Runtime data
is declared explicitly in `pyproject.toml` and `MANIFEST.in`, then checked from
the built archives by `Packaging/check_manifest.py`. Tests must inspect both a
source-built wheel and a wheel rebuilt from the source distribution; reading
the declarations alone does not prove what users receive.

## Offline tiktoken runtime assets

Standard installations depend on exactly `tiktoken==0.14.0` and ship its
reviewed encoding data in `tldw_chatbook/assets/tiktoken_cache/`. The directory
is an immutable, closed cache with exactly nine files: six SHA-1-named blobs for
GPT-2, `r50k_base`, `p50k_base`, `cl100k_base`, and `o200k_base`, plus
`manifest.json`, `LICENSE.txt`, and `NOTICE.txt`. The manifest and notice record
each source URL, native `sha1(source_url UTF-8 bytes)` cache key, expected
SHA-256 digest, model coverage, and the reviewed tiktoken loader seam.

When neither supported override is present before importing Chatbook, package
initialization points `TIKTOKEN_CACHE_DIR` at that directory and installs a
manifest-checked reader. It reads only the packaged files and never fetches,
deletes, creates directories, or writes. Missing, corrupt, hash-mismatched, or
unmanifested data fails closed. The installed package remains a read-only owner
under ADR-032.

A caller-provided `TIKTOKEN_CACHE_DIR` or legacy `DATA_GYM_CACHE_DIR` is
authoritative only when it exists before the first Chatbook import. In that
case Chatbook does not import or patch tiktoken during bootstrap and leaves the
value byte-for-byte unchanged; upstream tiktoken cache and download behavior,
including writes and possible network access, applies.

The default bundle does not automatically support encodings introduced after
tiktoken 0.14.0. Adding or upgrading an encoding is one reviewed change:

1. Pin and install the proposed tiktoken version.
2. Review `tiktoken_ext/openai_public.py` and `tiktoken/load.py` for constructor
   URLs and hashes, model mappings, `read_file_cached` parameters, and the
   SHA-1 URL-key algorithm.
3. Download each reviewed URL, name it with the SHA-1 of the URL's UTF-8 bytes,
   and verify its SHA-256 against the constructor's `expected_hash`.
4. Update the exact package-data and source-distribution lists, manifest,
   notice, and license. The MIT license is copied from
   `tiktoken-0.14.0.dist-info/licenses/LICENSE`; the repository owner accepts
   the [OpenAI collaborator clarification](https://github.com/openai/tiktoken/issues/92#issuecomment-1497875652)
   that it covers encoding files as the redistribution basis.
5. Run source tests and source-built plus sdist-rebuilt installed-wheel probes
   offline, read-only, and with missing/corrupt/unexpected-entry mutations
   before changing the exact dependency pin.

## Portable archive policy

The canonical release checker rejects archive members that could extract to a
different path on a supported platform. Both wheel and sdist paths must be
canonical relative POSIX paths: no absolute or drive-qualified names,
backslashes, dot or parent segments, repeated separators, duplicate names,
trailing-slash aliases, or case-insensitive extraction collisions. Components
may not contain control characters or Windows-invalid characters, end in a dot
or space, or use reserved Windows device stems (including extensions and the
superscript-digit COM/LPT aliases). Sdist entries must be regular files or
directories; wheel cache assets must be regular files. The checker also
requires exactly one canonical metadata record, exactly
`Requires-Dist: tiktoken==0.14.0`, and exact equality for every member under
the tiktoken cache prefix.

References:

- [Using uv on Fedora](https://fedoramagazine.org/enhancing-your-python-workflow-with-uv-on-fedora/)
- [Briefcase](https://github.com/beeware/briefcase)
- [Packaging a Python application](https://ahgamut.github.io/2021/07/13/ape-python/)
- [Publishing a Python package to PyPI](https://medium.com/@blackary/publishing-a-python-package-from-github-to-pypi-in-2024-a6fb8635d45d)



