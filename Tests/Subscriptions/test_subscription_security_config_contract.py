"""Source-level ownership contracts for subscription URL security policy."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
import tomllib
import warnings

import pytest

from tldw_chatbook.config import CONFIG_TOML_CONTENT
from tldw_chatbook.Subscriptions.security import SecurityValidator


PACKAGE_ROOT = Path(__file__).resolve().parents[2] / "tldw_chatbook"
CANONICAL_METADATA_ENDPOINTS = frozenset(
    {
        "169.254.169.254",
        "100.100.100.200",
        "fd00:ec2::254",
        "metadata.google.internal",
        "metadata.azure.com",
    }
)
EGRESS_POLICY_PATH = Path("Utils/egress.py")
DISALLOWED_SCHEMES = frozenset({"file", "ftp", "gopher", "javascript", "data"})


@dataclass(frozen=True)
class _PolicyInventory:
    """Immutable source-policy findings collected during one package scan."""

    metadata_owners: tuple[tuple[str, frozenset[Path]], ...]
    scheme_violations: tuple[tuple[Path, int, frozenset[str]], ...]


def _production_python_files() -> list[Path]:
    """Return Python sources belonging to the shipped application package."""
    return sorted(PACKAGE_ROOT.rglob("*.py"))


def _source_tree(source_path: Path) -> ast.Module:
    """Parse one production source file without reading comments as policy."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        return ast.parse(
            source_path.read_text(encoding="utf-8"), filename=str(source_path)
        )


def _literal_collection_values(node: ast.AST) -> set[str] | None:
    """Return literal string elements for a supported source collection."""
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        if not all(isinstance(element, ast.Constant) for element in node.elts):
            return None
        return {
            element.value for element in node.elts if isinstance(element.value, str)
        }
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "frozenset"
        and len(node.args) == 1
        and not node.keywords
    ):
        return _literal_collection_values(node.args[0])
    return None


def _egress_metadata_endpoints(source_tree: ast.Module) -> frozenset[str]:
    """Return endpoint literals assigned to the egress metadata policy."""
    endpoints: set[str] = set()
    for node in source_tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name)
            and target.id in {"_METADATA_IPS", "METADATA_HOSTNAMES"}
            for target in node.targets
        ):
            continue
        endpoints.update(
            child.value
            for child in ast.walk(node.value)
            if isinstance(child, ast.Constant) and isinstance(child.value, str)
        )
    return frozenset(endpoints)


@pytest.fixture(scope="module")
def _policy_inventory() -> _PolicyInventory:
    """Scan production Python sources once for all shared-policy duplicates."""
    source_paths = _production_python_files()
    egress_source_path = PACKAGE_ROOT / EGRESS_POLICY_PATH
    egress_tree = _source_tree(egress_source_path)
    metadata_endpoints = CANONICAL_METADATA_ENDPOINTS | _egress_metadata_endpoints(
        egress_tree
    )
    endpoint_paths = {endpoint: set() for endpoint in metadata_endpoints}
    violations: dict[tuple[Path, int], set[str]] = {}

    for source_path in source_paths:
        relative_path = source_path.relative_to(PACKAGE_ROOT)
        source_tree = (
            egress_tree
            if source_path == egress_source_path
            else _source_tree(source_path)
        )
        for node in ast.walk(source_tree):
            if isinstance(node, ast.Constant) and node.value in endpoint_paths:
                endpoint_paths[node.value].add(relative_path)

            values = _literal_collection_values(node)
            if values is None:
                continue
            blocked_schemes = values & DISALLOWED_SCHEMES
            if len(blocked_schemes) >= 3:
                violations.setdefault((relative_path, node.lineno), set()).update(
                    blocked_schemes
                )

    return _PolicyInventory(
        metadata_owners=tuple(
            (endpoint, frozenset(paths))
            for endpoint, paths in sorted(endpoint_paths.items())
        ),
        scheme_violations=tuple(
            (path, line, frozenset(schemes))
            for (path, line), schemes in sorted(
                violations.items(), key=lambda item: (item[0][0].as_posix(), item[0][1])
            )
        ),
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ('["file", "ftp", "gopher", "file"]', {"file", "ftp", "gopher"}),
        ('("file", 1, "ftp", None, "gopher")', {"file", "ftp", "gopher"}),
        (
            'frozenset({"file", "ftp", "gopher", "file"})',
            {"file", "ftp", "gopher"},
        ),
        ('["file", scheme, "ftp", "gopher"]', None),
    ],
)
def test_literal_collection_values_handles_static_mixed_collections(
    source: str, expected: set[str] | None
) -> None:
    """Static collections retain string members; dynamic ones are rejected."""
    node = ast.parse(source, mode="eval").body

    assert _literal_collection_values(node) == expected


def test_egress_metadata_endpoints_include_future_assigned_literals() -> None:
    """The ownership scan follows new endpoints declared by egress policy."""
    source_tree = ast.parse(
        """
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),
        ipaddress.ip_address("192.0.2.99"),
    }
)
METADATA_HOSTNAMES = frozenset(
    {"metadata.google.internal", "metadata.future.invalid"}
)
"""
    )

    assert _egress_metadata_endpoints(source_tree) == {
        "169.254.169.254",
        "192.0.2.99",
        "metadata.google.internal",
        "metadata.future.invalid",
    }


def test_shipped_subscription_config_has_no_security_child_table() -> None:
    """Subscription security policy is not duplicated in generated config."""
    config_defaults = tomllib.loads(CONFIG_TOML_CONTENT)

    assert "subscriptions" in config_defaults
    assert "security" not in config_defaults["subscriptions"]


def test_canonical_metadata_endpoints_are_owned_only_by_egress_policy(
    _policy_inventory: _PolicyInventory,
) -> None:
    """Metadata endpoint literals have one canonical production policy owner."""
    mismatches = tuple(
        (endpoint, paths)
        for endpoint, paths in _policy_inventory.metadata_owners
        if paths != {EGRESS_POLICY_PATH}
    )

    assert not mismatches, "\n".join(
        f"{endpoint} must be declared only in {EGRESS_POLICY_PATH}; found "
        f"{tuple(sorted(path.as_posix() for path in paths))}"
        for endpoint, paths in mismatches
    )


def test_disallowed_url_scheme_collections_are_not_duplicated(
    _policy_inventory: _PolicyInventory,
) -> None:
    """The egress allowlist, rather than blocked-scheme tables, owns scheme policy."""
    assert not _policy_inventory.scheme_violations, (
        f"duplicate disallowed URL-scheme policy: {_policy_inventory.scheme_violations}"
    )


def test_subscription_validator_retains_only_its_http_scheme_boundary() -> None:
    """Subscription validation keeps its boundary but delegates shared policy."""
    assert "BLOCKED_SCHEMES" not in SecurityValidator.__dict__
    assert "METADATA_ENDPOINTS" not in SecurityValidator.__dict__
    assert SecurityValidator.ALLOWED_SCHEMES == {"http", "https"}
