# Exact source and text-span digests

The shared-core primitive calculates two exact text identities. It does not
resolve a source or establish a claim's truth, support, approval or authority.
[TASK-25907.12](../tasks/task-25907.12%20-%20Add-exact-text-span-digests-to-shared-profile-core.md)
implements the convention in
[ADR-185](../decisions/185-versioned-profile-evidence-and-temporal-claims.md);
its [native plan](../../Docs/superpowers/plans/2026-09-26-personal-context-exact-span-digests.md)
records the scope and targeted checks.

## API

```python
from tldw_profile_core.evidence import exact_text_span_digests

result = exact_text_span_digests("Hi 👋 — café", 7, 11)
assert result.representation_sha256 == (
    "bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef"
)
assert result.span_sha256 == (
    "850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e"
)
```

The function returns frozen `ExactTextSpanDigests`, with named
`representation_sha256` and `span_sha256` fields containing lowercase
64-character SHA-256 hex values. These fields cover the entire supplied string
and the selected substring, respectively. The return value contains no source
text, locator, authority or access decision. It is a Python data value, not a
new canonical profile model or a serialized evidence binding.

Inputs must be a built-in `str` and built-in `int` offsets. Subclasses,
booleans, strings used as numbers and all floats (including `1.0`) reject with
`TypeError`. Exact types prevent overridden Python methods from changing
encoding or range checks. This native API does not change the existing
canonical JSON integer rules.

## Exact representation and offsets

The range is zero-based and half-open: `0 <= start <= end <= len(text)`.
Python string positions count Unicode codepoints; one astral emoji counts
once. They do not count UTF-8 bytes, UTF-16 code units or visual graphemes.
Combining marks count separately. The implementation performs strict UTF-8
encoding without Unicode normalization, line-ending conversion, trimming,
JSON quoting or field joining.

Invalid ranges raise `ValueError`; Python's negative-index and clipped-slice
behavior is not admitted. Valid empty spans, including an empty representation
at `[0, 0)`, produce the standard empty UTF-8 span hash
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
An empty calculation does not establish supporting evidence. Future binding
admission remains responsible for evidence requirements.

An unpaired surrogate anywhere in the full representation raises
`UnicodeEncodeError`, a `ValueError` subclass, even if it is outside the
selected span. That Python exception carries the failing input; future callers
must keep its source-bearing object out of diagnostics disclosed to another
owner. The helper performs no logging, source lookup, decryption or persistence.

Editing text outside a span may leave `span_sha256` unchanged while changing
`representation_sha256`. A complete future binding needs both values and its
exact owner version/capture token, authority and scope. A valid but unintended
range returns its own digests; the helper cannot decide what its caller meant
to cite or compare against an absent expected binding.

## Ownership and remaining work

Callers choose and authorize the source representation before supplying text.
They also own resource limits and scheduling: this synchronous calculation
encodes the whole representation and span, so time and temporary memory grow
with supplied text. This primitive imposes no new source-size or access policy.

These SHA-256 identities are different from future RFC 8785 canonical
binding/claim digests and existing keyed V1 integrity tags. A digest is governed
source metadata, not anonymization, authentication or permission to disclose.
The existing V1 `source_references` and `source_hashes` remain inert legacy
metadata. No existing profile-tool consumer is rewired to use this helper.

V2 schemas/records/proposals/manifests, source capture and current-version
resolution, semantic support, temporal operations, encrypted retirement,
forgetting, destination/purpose disclosure and client/server qualification
remain unimplemented by this slice. SERIALIZED_SCHEMA_VERSION remains 1;
the package root does not advertise this helper or V2 capabilities. The
explicit submodule API is independently testable, with no runtime caller
activation.

## Targeted verification

From the isolated worktree, use its native interpreter and shared-core source:

```bash
PYTHONPATH=.:packages/tldw_profile_core/src .venv/bin/python -m pytest packages/tldw_profile_core/tests/test_evidence.py packages/tldw_profile_core/tests/test_canonical.py packages/tldw_profile_core/tests/test_models.py packages/tldw_profile_core/tests/test_schema_fixtures.py packages/tldw_profile_core/tests/test_interview.py -q
```

The new tests use fixed expected digests for the accepted Unicode example,
decomposed accents, line endings, astral characters, shifted ranges and
outside-span edits. Rejected-input and successful empty-span controls call the
actual API. Existing V1 canonical/HMAC, models, schema/fixture parity and
interview checks remain separate regression evidence; none qualifies a source
resolver or a server runtime.
