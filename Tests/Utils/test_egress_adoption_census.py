"""TASK-19556 (+ Qodo follow-up on PR #1967): a module that opens URLs must
also consult the egress policy.

Three seams in this codebase reached the network without ever calling
`Utils/egress.py`, and each was found by a human reading code rather than by
anything that would fail. The sharpest -- `Library/ingest_preflight.py`'s
`urlopen` HEAD on a typing debounce -- had been in place across several
reviews. This census is the "next seam cannot silently skip it" guard the
task's last AC asks for.

**The rule.** A module under `tldw_chatbook/` that uses a URL-opening
primitive from the two families this task touched must also use at least
one egress-policy symbol. Referencing the policy is not proof of using it
correctly -- no static check can be -- but a module that never uses it
demonstrably is not using it, which is exactly the state all three seams
were in.

**Why this is AST-based, not regex-over-text.** The original version of this
census (`re.Pattern.search` over the raw source) was reviewed on PR #1967
and correctly called bypassable in both directions: a docstring or comment
*mentioning* `check_url_or_raise` made an unguarded module look guarded
(false green -- the exact shape this programme keeps finding elsewhere), and
a docstring or comment merely *naming* `urlopen` in prose made a module that
never touches the network look like it needed an exemption (false red). Both
directions are proven closed below
(`test_comment_mentioning_the_real_egress_symbol_does_not_launder_an_unguarded_opener`,
`test_docstring_or_comment_opener_mention_is_not_flagged`), and the original
demonstration -- that the census independently rediscovers the two real
seams this task fixed -- is re-proven against the rewrite
(`test_census_rediscovers_the_original_two_seams_when_their_egress_calls_are_removed`).

This version resolves real `import`/`from ... import` bindings (including
`as` aliases and `from pkg import module` + `module.symbol(...)` attribute
calls) and matches actual `ast.Call` targets against the opener/egress
symbol sets. It does not execute anything -- comments and docstrings never
produce `Name`/`Attribute` nodes, so a mention inside either is structurally
invisible to it, which is what makes both the false-green and false-red
bypasses close.

**What this still cannot express (stated, not left as a silent gap).** This
is a single-file, syntactic scan -- it resolves names bound by `import`
statements in the *same* file, not whatever an imported name actually does
in its own module (a thin same-file wrapper around `urlopen` is invisible;
that wrapper's own file is separately censused when the scan reaches it).
It also cannot follow dynamic dispatch: `getattr(urllib.request,
"urlopen")(u)` or a call built through `functools.partial`/`exec` is not a
`Name`/`Attribute` chain a static resolver can trace, so the AST version
will not flag it -- notably, the OLD regex *would* have (the string literal
`"urlopen"` is still text in the file), so this is a real, if obscure,
regression traded for closing the comment/docstring bypass. Nobody in this
codebase uses `getattr`-based dispatch to reach `urlopen`/`YoutubeDL`
today (checked at rewrite time), so it costs nothing currently, but a
reflection-based seam would need a human read, same as it always did.

**The scope, stated so it is not overclaimed.** This covers the
`urllib.request` family (the primitive that produced defect (a): it
auto-follows redirects and consults nothing) and `yt_dlp.YoutubeDL` (defect
(b)). It deliberately does NOT census bare `requests`/`httpx` calls: this
app has dozens of legitimate ones to fixed, non-user-supplied API endpoints
(every LLM provider in `LLM_Calls/`), so a census of those would be noise
that gets suppressed rather than a guard that gets read. Behavioural
adoption for the individual seams is pinned by
`Tests/Library/test_ingest_preflight_egress.py`,
`Tests/Local_Ingestion/test_video_egress_guard.py` and
`Tests/Web_Scraping/test_sitemap_crawl_trusted_origins.py`.
"""

from __future__ import annotations

import ast
from pathlib import Path

import tldw_chatbook

PACKAGE_ROOT = Path(tldw_chatbook.__file__).parent

#: (module suffix, attribute name) pairs whose CALL means "this module opens
#: a URL". Matched after resolving the callee through this file's own
#: import bindings -- see `_resolve_call_target`.
_OPENER_TARGETS = frozenset(
    {
        "urllib.request.urlopen",
        "urllib.request.urlretrieve",
        "urllib.request.build_opener",
        "urllib.request.OpenerDirector",
        "yt_dlp.YoutubeDL",
    }
)

#: Egress-policy symbol names. An exact match, or (for the guarded_fetch_*
#: family, which has several sync/async/backend variants) a prefix match.
_EGRESS_SYMBOLS = frozenset(
    {
        "check_url_or_raise",
        "check_url_or_raise_async",
        "evaluate_url_policy",
        "evaluate_url_policy_async",
        "is_public_http_url",
    }
)


def _is_egress_symbol(name: str) -> bool:
    return name in _EGRESS_SYMBOLS or name.startswith("guarded_fetch_")


def _is_egress_module(module: str) -> bool:
    """True for `Utils.egress`, `tldw_chatbook.Utils.egress`, `.egress`, etc.

    Matched on the last dotted segment so it is agnostic to relative-import
    depth (`node.module` never carries the leading dots for a relative
    import; `node.level` does, and we don't need it here).
    """
    return module.split(".")[-1] == "egress"


def _dotted_name(node: ast.AST) -> str | None:
    """Collapse a `Name`/`Attribute` chain (e.g. `egress.check_url_or_raise`)
    into a dotted string, or return None if it isn't one (e.g. the callee is
    itself a call, a subscript, or anything else not statically resolvable).
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        parts.reverse()
        return ".".join(parts)
    return None


def _resolve_call_target(dotted: str, bindings: dict[str, str]) -> str:
    """Rewrite the leftmost segment of `dotted` through this file's import
    bindings, so `req.urlopen` resolves to `urllib.request.urlopen` when the
    file has `import urllib.request as req`, and a bare `urlopen` resolves
    the same way when the file has `from urllib.request import urlopen`.
    """
    head, *rest = dotted.split(".")
    resolved_head = bindings.get(head, head)
    return ".".join([resolved_head, *rest]) if rest else resolved_head


def _scan(tree: ast.Module) -> tuple[frozenset[str], frozenset[str]]:
    """Return (opener_hits, egress_hits): the resolved dotted names of every
    real opener call and every real egress-policy import/call in `tree`.

    Two passes: bindings first (so a call is resolved against every import
    in the file regardless of source order), then calls.
    """
    bindings: dict[str, str] = {}
    egress_hits: set[str] = set()
    opener_hits: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                # `import a.b.c` binds only `a`; `import a.b.c as x` binds
                # `x` to the full dotted path.
                local = alias.asname or alias.name.split(".")[0]
                qualified = alias.name if alias.asname else local
                bindings.setdefault(local, qualified)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                qualified = f"{module}.{alias.name}" if module else alias.name
                bindings[local] = qualified
                # `from ...Utils.egress import check_url_or_raise` is a real
                # AST import node naming the specific symbol -- that is
                # itself a use of the policy, not a substring in prose.
                if _is_egress_module(module) and _is_egress_symbol(alias.name):
                    egress_hits.add(qualified)

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        dotted = _dotted_name(node.func)
        if dotted is None:
            continue
        resolved = _resolve_call_target(dotted, bindings)
        if resolved in _OPENER_TARGETS:
            opener_hits.add(resolved)
            continue
        parts = resolved.split(".")
        if len(parts) >= 2 and parts[-2] == "egress" and _is_egress_symbol(parts[-1]):
            egress_hits.add(resolved)

    return frozenset(opener_hits), frozenset(egress_hits)


def _analyze_source(source: str) -> tuple[frozenset[str], frozenset[str]]:
    return _scan(ast.parse(source))


#: Modules exempt from the rule, each with the reason it is exempt.
#: An entry here is a claim someone has to defend at review time -- which is
#: the point; the alternative is a seam nobody ever looks at.
_EXEMPT: dict[str, str] = {
    "Local_Ingestion/parakeet_v2_installer.py": (
        "The URL is a module constant (`_source_url` interpolates two "
        "hardcoded constants into a huggingface.co path); no caller-supplied "
        "component reaches it, so there is no SSRF input to guard. The "
        "download is additionally size- and SHA-256-pinned per file."
    ),
}


def _module_key(path: Path) -> str:
    return path.relative_to(PACKAGE_ROOT).as_posix()


def _offenders() -> dict[str, list[str]]:
    """Modules that open URLs without using the egress policy."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        openers, egress = _analyze_source(source)
        if not openers:
            continue
        if egress:
            continue
        key = _module_key(path)
        if key in _EXEMPT:
            continue
        offenders[key] = sorted(openers)
    return offenders


def test_every_url_opening_module_consults_the_egress_policy() -> None:
    offenders = _offenders()
    assert offenders == {}, (
        "these modules open URLs without consulting Utils/egress.py: "
        + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(offenders.items()))
        + " -- route the fetch through the policy, or add an entry to _EXEMPT "
        "with the reason it needs none."
    )


def test_the_census_detector_actually_detects() -> None:
    """Methodology check: a detector that matched nothing would be silent.

    The census scans real files, so this proves the *predicate* bites by
    running it over synthetic modules rather than by trusting the empty
    result above.
    """
    guarded = (
        "from tldw_chatbook.Utils.egress import check_url_or_raise\n"
        "from urllib.request import urlopen\n"
        "def f(u):\n"
        "    check_url_or_raise(u)\n"
        "    return urlopen(u)\n"
    )
    unguarded = "from urllib.request import urlopen\ndef f(u):\n    return urlopen(u)\n"
    inert = "x = 1\n"

    inert_openers, inert_egress = _analyze_source(inert)
    assert not inert_openers and not inert_egress

    unguarded_openers, unguarded_egress = _analyze_source(unguarded)
    assert unguarded_openers and not unguarded_egress

    guarded_openers, guarded_egress = _analyze_source(guarded)
    assert guarded_openers and guarded_egress


def test_scan_resolves_aliased_and_attribute_style_imports() -> None:
    """Pins the two resolver paths a bare-word regex cannot distinguish:
    an `as`-aliased opener import, and a `from pkg import module` +
    `module.symbol(...)` attribute call -- the exact shape
    `Image_Generation/http_client.py` and `Media_Playback/stream_resolve.py`
    use for the egress side (`from tldw_chatbook.Utils import egress` then
    `egress.check_url_or_raise(...)`).
    """
    aliased_opener = (
        "import urllib.request as req\n"
        "def f(u):\n"
        "    return req.urlopen(u)\n"
    )
    openers, egress = _analyze_source(aliased_opener)
    assert openers == frozenset({"urllib.request.urlopen"})
    assert not egress

    module_attr_egress = (
        "from tldw_chatbook.Utils import egress\n"
        "from urllib.request import urlopen\n"
        "def f(u):\n"
        "    egress.check_url_or_raise(u)\n"
        "    return urlopen(u)\n"
    )
    openers, egress = _analyze_source(module_attr_egress)
    assert openers == frozenset({"urllib.request.urlopen"})
    assert egress

    yt_dlp_attr = "import yt_dlp\ndef f(opts):\n    return yt_dlp.YoutubeDL(opts)\n"
    openers, egress = _analyze_source(yt_dlp_attr)
    assert openers == frozenset({"yt_dlp.YoutubeDL"})
    assert not egress


def test_docstring_or_comment_opener_mention_is_not_flagged() -> None:
    """No false red: Qodo's finding on PR #1967 was that an opener word
    appearing only in a comment/docstring could trip the OLD regex census
    into treating a module that never touches the network as one that needs
    an `_EXEMPT` entry. Neither a docstring nor a comment produces a
    `Name`/`Attribute` AST node, so the rewritten census must see zero
    openers here -- the module is not even classified as opening URLs, so
    whether it "consults the policy" never comes up.
    """
    prose_only = (
        '"""This module used to call urlopen() directly; see YoutubeDL for '
        'the video case too."""\n'
        "# TODO: consider urlretrieve for large downloads via build_opener\n"
        "def helper(x):\n"
        "    return x + 1\n"
    )
    openers, egress = _analyze_source(prose_only)
    assert openers == frozenset(), (
        "a comment/docstring mention of an opener must not be classified as "
        f"opening a URL, got: {sorted(openers)}"
    )
    assert egress == frozenset()


def test_comment_mentioning_the_real_egress_symbol_does_not_launder_an_unguarded_opener() -> (
    None
):
    """No false green: this is Qodo's exact bypass -- a comment naming the
    real egress function next to a genuine, unguarded opener call used to
    read as "consults the policy" under the old raw-text regex. It must not
    under the AST version: comments never produce AST nodes, so the only
    thing that can satisfy `egress_hits` is a real import or a real call.
    """
    laundered = (
        "from urllib.request import urlopen\n"
        "def fetch(u):\n"
        "    # Should route through check_url_or_raise from Utils.egress, "
        "but doesn't yet.\n"
        "    return urlopen(u)\n"
    )
    openers, egress = _analyze_source(laundered)
    assert openers, "the module genuinely opens a URL and must be detected as such"
    assert egress == frozenset(), (
        "a comment merely naming check_url_or_raise must not count as "
        f"consulting the policy, got: {sorted(egress)}"
    )
    # And the top-level census function classifies it as an offender.
    key = "synthetic/laundered_opener.py"
    offenders: dict[str, list[str]] = {}
    if openers and not egress and key not in _EXEMPT:
        offenders[key] = sorted(openers)
    assert offenders == {key: ["urllib.request.urlopen"]}


def test_census_rediscovers_the_original_two_seams_when_their_egress_calls_are_removed() -> (
    None
):
    """The census's original demonstration, re-proven against the rewrite.

    TASK-19556's own implementation notes record that, run at the
    pre-fix base, the census "independently rediscovered exactly the two
    seams this task names: `Library/ingest_preflight.py` and
    `Local_Ingestion/video_processing.py`". This test does not need the old
    base commit to re-prove that: it takes the CURRENT (fixed) source of
    both modules and mechanically un-fixes each -- deletes the
    `check_url_or_raise` import and blanks the call site -- which is exactly
    the code shape the task describes as the original defect ("contains
    zero references to the egress helpers"). If the census can no longer
    catch that shape, the rewrite regressed the one thing this file exists
    to guarantee.
    """
    ingest_preflight = PACKAGE_ROOT / "Library" / "ingest_preflight.py"
    video_processing = PACKAGE_ROOT / "Local_Ingestion" / "video_processing.py"

    ip_source = ingest_preflight.read_text(encoding="utf-8")
    ip_mutated = ip_source.replace(
        "from tldw_chatbook.Utils.egress import EgressBlockedError, check_url_or_raise",
        "from tldw_chatbook.Utils.egress import EgressBlockedError",
    ).replace("check_url_or_raise(url)", "pass")
    assert "check_url_or_raise" not in ip_mutated, (
        "the fixture substitution did not match current source -- "
        "Library/ingest_preflight.py's egress call shape moved; update the "
        "literal strings above to match"
    )
    ip_openers, ip_egress = _analyze_source(ip_mutated)
    assert ip_openers, "the un-fixed module must still be detected as opening a URL"
    assert ip_egress == frozenset(), (
        "the un-fixed module must be rediscovered as NOT consulting the "
        f"policy, but the census still found: {sorted(ip_egress)}"
    )

    vp_source = video_processing.read_text(encoding="utf-8")
    vp_mutated = vp_source.replace(
        "from ..Utils.egress import EgressBlockedError, check_url_or_raise, origin_set",
        "from ..Utils.egress import EgressBlockedError, origin_set",
    ).replace("check_url_or_raise(url, trusted_origins=origin_set(url))", "pass")
    assert "check_url_or_raise" not in vp_mutated, (
        "the fixture substitution did not match current source -- "
        "Local_Ingestion/video_processing.py's egress call shape moved; "
        "update the literal strings above to match"
    )
    vp_openers, vp_egress = _analyze_source(vp_mutated)
    assert vp_openers, "the un-fixed module must still be detected as opening a URL (yt_dlp.YoutubeDL)"
    assert vp_egress == frozenset(), (
        "the un-fixed module must be rediscovered as NOT consulting the "
        f"policy, but the census still found: {sorted(vp_egress)}"
    )

    # And each still parses as valid Python -- a mutation that broke syntax
    # would make this a test of ast.parse's error handling, not the census.
    ast.parse(ip_mutated)
    ast.parse(vp_mutated)


def test_exempt_entries_still_exist_and_still_need_the_exemption() -> None:
    """An exemption for a module that no longer opens URLs is stale."""
    for key in _EXEMPT:
        path = PACKAGE_ROOT / key
        assert path.exists(), f"stale exemption for a deleted module: {key}"
        source = path.read_text(encoding="utf-8", errors="replace")
        openers, _egress = _analyze_source(source)
        assert openers, f"stale exemption: {key} no longer opens URLs, drop the entry"
