"""TASK-19556: a module that opens URLs must also consult the egress policy.

Three seams in this codebase reached the network without ever calling
`Utils/egress.py`, and each was found by a human reading code rather than by
anything that would fail. The sharpest -- `Library/ingest_preflight.py`'s
`urlopen` HEAD on a typing debounce -- had been in place across several
reviews. This census is the "next seam cannot silently skip it" guard the
task's last AC asks for.

**The rule.** A module under `tldw_chatbook/` that uses a URL-opening
primitive from the two families this task touched must also name at least
one egress-policy symbol. Referencing the policy is not proof of using it
correctly -- no static check can be -- but a module that never mentions it
demonstrably is not using it, which is exactly the state all three seams
were in.

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

import re
from pathlib import Path

import tldw_chatbook

PACKAGE_ROOT = Path(tldw_chatbook.__file__).parent

#: URL-opening primitives whose users must consult the egress policy.
_OPENER_PATTERNS = (
    re.compile(r"\burlopen\b"),
    re.compile(r"\burlretrieve\b"),
    re.compile(r"\bbuild_opener\b"),
    re.compile(r"\bOpenerDirector\b"),
    re.compile(r"\bYoutubeDL\b"),
)

#: Any of these names means the module consults the policy.
_EGRESS_PATTERNS = (
    re.compile(r"\bcheck_url_or_raise\b"),
    re.compile(r"\bcheck_url_or_raise_async\b"),
    re.compile(r"\bevaluate_url_policy\b"),
    re.compile(r"\bevaluate_url_policy_async\b"),
    re.compile(r"\bis_public_http_url\b"),
    re.compile(r"\bguarded_fetch_"),
)

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
    """Modules that open URLs without naming the egress policy."""
    offenders: dict[str, list[str]] = {}
    for path in sorted(PACKAGE_ROOT.rglob("*.py")):
        source = path.read_text(encoding="utf-8", errors="replace")
        hits = [p.pattern for p in _OPENER_PATTERNS if p.search(source)]
        if not hits:
            continue
        if any(p.search(source) for p in _EGRESS_PATTERNS):
            continue
        key = _module_key(path)
        if key in _EXEMPT:
            continue
        offenders[key] = hits
    return offenders


def test_every_url_opening_module_consults_the_egress_policy() -> None:
    offenders = _offenders()
    assert offenders == {}, (
        "these modules open URLs without consulting Utils/egress.py: "
        + "; ".join(f"{k} ({', '.join(v)})" for k, v in sorted(offenders.items()))
        + " -- route the fetch through the policy, or add an entry to _EXEMPT "
        "with the reason it needs none."
    )


def test_the_census_detector_actually_detects(tmp_path: Path) -> None:
    """Methodology check: a detector that matched nothing would be silent.

    The census scans real files, so this proves the *predicate* bites by
    running it over a synthetic module rather than by trusting the empty
    result above.
    """
    guarded = "from x import check_url_or_raise\nurlopen(u)\n"
    unguarded = "from urllib.request import urlopen\nurlopen(u)\n"
    inert = "x = 1\n"
    assert not any(p.search(inert) for p in _OPENER_PATTERNS)
    assert any(p.search(unguarded) for p in _OPENER_PATTERNS)
    assert not any(p.search(unguarded) for p in _EGRESS_PATTERNS)
    assert any(p.search(guarded) for p in _EGRESS_PATTERNS)


def test_exempt_entries_still_exist_and_still_need_the_exemption() -> None:
    """An exemption for a module that no longer opens URLs is stale."""
    for key in _EXEMPT:
        path = PACKAGE_ROOT / key
        assert path.exists(), f"stale exemption for a deleted module: {key}"
        source = path.read_text(encoding="utf-8", errors="replace")
        assert any(p.search(source) for p in _OPENER_PATTERNS), (
            f"stale exemption: {key} no longer opens URLs, drop the entry"
        )
