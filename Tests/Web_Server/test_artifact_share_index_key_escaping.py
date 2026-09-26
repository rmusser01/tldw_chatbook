"""The index page escapes EVERY manifest field, `key` included.

Tier-2 review S14, P2 [D1]: `_render_index` escapes `display_name`,
`description`, `kind` and `share_name` through `_escape_fragment` -- and
interpolated `item.key` raw into `href="/artifact/{item.key}"`. The
module's whole design is "escape every field" (`_escape_fragment`'s
docstring states it as policy), so the one field outside it is the bug.

`key` is `secrets.token_urlsafe(16)` when *this* app stages the share, but
the server is a separate process (`python -m ...artifact_share_server
<manifest>`) that trusts whatever manifest path it is handed, and
`load_manifest` validated only `schema_version` and non-emptiness. The
CSP (`default-src 'none'; style-src 'unsafe-inline'`) is defence in
depth, not the escape.

Two layers, because either alone leaves a hole: the model now refuses a
key that is not the opaque token shape (so `load_manifest` fails closed on
a hand-edited manifest), and the render escapes it anyway.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from tldw_chatbook.Web_Server.artifact_share_manifest import (
    ArtifactShareManifest,
    SharedArtifact,
)
from tldw_chatbook.Web_Server.artifact_share_server import ArtifactShareServer

_HOSTILE_KEY = '" onmouseover="alert(1)'


def _artifact(key: str) -> SharedArtifact:
    return SharedArtifact(
        key=key,
        display_name="Alpha",
        description="desc",
        size_bytes=10,
        sha256="0" * 64,
        source_chatbook_id=1,
        staged_name="alpha.zip",
    )


def test_a_hostile_key_is_refused_by_the_model():
    with pytest.raises(ValidationError):
        _artifact(_HOSTILE_KEY)


def test_the_real_token_shape_is_still_accepted():
    """Fails closed, so prove it does not reject what staging produces."""
    from tldw_chatbook.Web_Server.artifact_share_manifest import new_artifact_key

    assert _artifact(new_artifact_key()).key


def test_the_rendered_href_never_carries_a_raw_quote(tmp_path, monkeypatch):
    """Belt and braces: even past the model, the render escapes it."""
    server = ArtifactShareServer.__new__(ArtifactShareServer)
    item = _artifact("abcDEF0123456789")
    object.__setattr__(item, "key", _HOSTILE_KEY)  # bypass validation
    server.manifest = ArtifactShareManifest(
        schema_version=1,
        share_id="s1",
        share_name="Test",
        created_at="2026-09-21T00:00:00Z",
        artifacts=[item],
        auth=None,
    )

    html = server._render_index()

    assert 'onmouseover="alert(1)"' not in html
    assert "&quot; onmouseover=&quot;alert(1)" in html
