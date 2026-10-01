import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

from tldw_profile_core.canonical import canonical_bytes, integrity_tag

ROOT = Path(__file__).parents[1]
FIXTURE = Path("fixtures/claim_meaning/v2/01-standing-preference.json")
RESOURCE = "tldw_profile_core/" + FIXTURE.as_posix()
CANONICAL = b'{"claim_basis":"direct_user_assertion","kind":"preference","payload":{"kind":"preference","polarity":"like","schema_version":1,"subject":"replies","value":"concise"},"profile_id":"p1","projection":"profile-claim-v2","record_id":"r1","relations":[],"scope_id":"s1","temporal_validity":{"basis":{"kind":"user_reviewed"},"kind":"standing"}}'
FIXED_SHA = "4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593"
CHANGED_SCOPE_SHA = "de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382"
FIXED_TAG = (
    "hmac-sha256-v1:adfefb01c7d615f8b0f1d2540da7cf3b0e5de1f0862eb573bc0e32120c900210"
)


def api():
    assert importlib.util.find_spec("tldw_profile_core.v2_meaning") is not None, (
        "meaning API missing"
    )
    return importlib.import_module("tldw_profile_core.v2_meaning")


def test_fixed_resource_and_distribution_parity():
    assert (ROOT / FIXTURE).exists(), "meaning fixture missing"
    raw = (ROOT / FIXTURE).read_bytes()
    fixture = json.loads(raw)
    assert set(fixture) == {
        "fixture_version",
        "data",
        "canonical_utf8",
        "claim_sha256",
        "changed_scope_sha256",
        "integrity_tag",
    }
    assert fixture["fixture_version"] == 1
    assert raw.endswith(b"\n")
    module = api()
    model = module.ClaimMeaningV2(**fixture["data"])
    assert fixture["canonical_utf8"].encode() == canonical_bytes(model) == CANONICAL
    assert len(CANONICAL) == 337
    assert fixture["claim_sha256"] == module.claim_meaning_digest(model) == FIXED_SHA
    assert (
        fixture["changed_scope_sha256"]
        == module.claim_meaning_digest(
            module.ClaimMeaningV2(**(fixture["data"] | {"scope_id": "s2"}))
        )
        == CHANGED_SCOPE_SHA
    )
    assert (
        fixture["integrity_tag"] == integrity_tag(model, bytes(range(32))) == FIXED_TAG
    )
    assert raw == (ROOT / "src/tldw_profile_core" / FIXTURE).read_bytes()


def test_offline_wheel_install_contains_meaning_and_original_resources(tmp_path):
    project = tmp_path / "project"
    shutil.copytree(
        ROOT,
        project,
        ignore=shutil.ignore_patterns(
            "build", "dist", "*.egg-info", "__pycache__", ".pytest_cache", ".ruff_cache"
        ),
    )
    wheels = tmp_path / "wheels"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--no-index",
            "--no-cache-dir",
            "--wheel-dir",
            str(wheels),
            str(project),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert built.returncode == 0, built.stdout + built.stderr
    wheel = next(wheels.glob("tldw_profile_core-*.whl"))
    originals = [
        path.relative_to(ROOT)
        for folder in (
            ROOT / "fixtures/v1",
            ROOT / "fixtures/evidence_binding/v1",
            ROOT / "schemas",
        )
        for path in sorted(folder.glob("*.json"))
    ]
    all_resources = [FIXTURE, *originals]
    with zipfile.ZipFile(wheel) as archive:
        metadata_paths = [
            name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
        ]
        assert len(metadata_paths) == 1
        assert (
            "Requires-Python: >=3.12"
            in archive.read(metadata_paths[0]).decode("utf-8").splitlines()
        ), "wheel Python floor must be 3.12"
        assert RESOURCE in archive.namelist(), "meaning wheel resource missing"
        for relative in all_resources:
            package_resource = "tldw_profile_core/" + relative.as_posix()
            assert archive.read(package_resource) == (ROOT / relative).read_bytes()
            distribution_resource = [
                name
                for name in archive.namelist()
                if name.endswith(".data/data/" + relative.as_posix())
            ]
            assert len(distribution_resource) == 1
            assert (
                archive.read(distribution_resource[0]) == (ROOT / relative).read_bytes()
            )
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--isolated",
            "install",
            "--no-deps",
            "--no-index",
            "--no-cache-dir",
            "--target",
            str(installed),
            str(wheel),
        ],
        text=True,
        capture_output=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    for relative in all_resources:
        assert (installed / relative).read_bytes() == (ROOT / relative).read_bytes()
        assert (installed / "tldw_profile_core" / relative).read_bytes() == (
            ROOT / relative
        ).read_bytes()
    script = """import importlib.resources, json, pathlib, sys
sys.path.insert(0, sys.argv[1])
import tldw_profile_core
import tldw_profile_core.v2_meaning as component
from tldw_profile_core.canonical import canonical_bytes, integrity_tag
installed = pathlib.Path(sys.argv[1]).resolve()
assert pathlib.Path(tldw_profile_core.__file__).resolve().is_relative_to(installed)
assert pathlib.Path(component.__file__).resolve().is_relative_to(installed)
assert tldw_profile_core.SERIALIZED_SCHEMA_VERSION == 1
assert 'ClaimMeaningV2' not in tldw_profile_core.__all__
assert not hasattr(tldw_profile_core, 'ClaimMeaningV2')
resource = importlib.resources.files('tldw_profile_core').joinpath('fixtures/claim_meaning/v2/01-standing-preference.json')
fixture = json.loads(resource.read_text(encoding='utf-8'))
value = component.ClaimMeaningV2(**fixture['data'])
assert component.claim_meaning_digest(value) == '4d91768974a491f84e8ef67d9b8577975e8615c47065cd97a16260a7f4c28593'
assert canonical_bytes(value).decode() == fixture['canonical_utf8']
assert integrity_tag(value, bytes(range(32))) == 'hmac-sha256-v1:adfefb01c7d615f8b0f1d2540da7cf3b0e5de1f0862eb573bc0e32120c900210'
assert component.claim_meaning_digest(component.ClaimMeaningV2(**(fixture['data'] | {'scope_id': 's2'}))) == 'de7ef8e9bffe5f5854390682357e8f8cbc08c4706dfa3e2c288910a08f7b1382'
"""
    loaded = subprocess.run(
        [sys.executable, "-I", "-c", script, str(installed)],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
    )
    assert loaded.returncode == 0, loaded.stdout + loaded.stderr
