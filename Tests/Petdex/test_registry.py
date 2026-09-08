"""Public manifest parsing and bounded exact-pet acquisition."""

import json

import pytest

from tldw_chatbook.Petdex import registry
from tldw_chatbook.Petdex.network import PetdexNetworkError

FIELDS = [
    "slug",
    "displayName",
    "kind",
    "submittedBy",
    "spritesheet",
    "petJson",
    "zip",
    "spriteVersionNumber",
]


def compact():
    return {
        "v": 2,
        "assetBase": "https://assets.petdex.dev",
        "fields": FIELDS,
        "total": 1,
        "pets": [
            [
                "boba",
                "Boba",
                "cat",
                "Creator",
                "/pets/boba/sprite.webp",
                "/pets/boba/pet.json",
                None,
                1,
            ]
        ],
    }


def legacy():
    return {
        "total": 1,
        "pets": [
            {
                "slug": "boba",
                "displayName": "Boba",
                "kind": "cat",
                "submittedBy": "Creator",
                "spritesheetUrl": "https://assets.petdex.dev/pets/boba/sprite.webp",
                "petJsonUrl": "https://assets.petdex.dev/pets/boba/pet.json",
                "zipUrl": None,
                "spriteVersionNumber": 1,
            }
        ],
    }


def test_compact_and_object_normalize_to_same_pinned_entry():
    assert registry.parse_manifest(
        json.dumps(compact()).encode()
    ) == registry.parse_manifest(json.dumps(legacy()).encode())
    entry = registry.parse_manifest(json.dumps(compact()).encode())[0]
    assert entry["sourceUrl"] == "https://petdex.dev/pets/boba"
    assert entry["spriteVersionNumber"] == 1
    assert entry["spritesheetUrl"] == "https://assets.petdex.dev/pets/boba/sprite.webp"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda data: data.update(v=3),
        lambda data: data.update(v=True),
        lambda data: data.update(total=2),
        lambda data: data["fields"].reverse(),
        lambda data: data["pets"][0].__setitem__(7, 3),
        lambda data: data["pets"][0].__setitem__(7, True),
        lambda data: data["pets"][0].__setitem__(0, "../boba"),
        lambda data: data["pets"][0].__setitem__(4, "https://petdex.dev/image.png"),
        lambda data: data["pets"][0].__setitem__(4, "//attacker.test/image.png"),
        lambda data: data["pets"][0].__setitem__(
            5, "https://user:secret@assets.petdex.dev/pet.json"
        ),
        lambda data: data["pets"][0].__setitem__(
            6, "https://assets.petdex.dev:444/pet.zip"
        ),
        lambda data: data.update(assetBase="http://assets.petdex.dev"),
    ],
)
def test_malformed_manifest_rejected(mutate):
    data = compact()
    data["fields"] = FIELDS.copy()
    mutate(data)
    with pytest.raises(ValueError):
        registry.parse_manifest(json.dumps(data).encode())


def test_duplicate_slug_rejected_even_when_other_pet_selected():
    data = legacy()
    data["pets"] *= 2
    data["total"] = 2
    with pytest.raises(ValueError, match="duplicate"):
        registry.parse_manifest(json.dumps(data).encode())


def test_duplicate_json_keys_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        registry.parse_manifest(b'{"pets": [], "pets": []}')


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.test/pets/boba",
        "https://petdex.dev/pets/boba?token=secret",
        "https://petdex.dev/pets/boba%2fmore",
        "boba extra",
        "BOBA",
        "https://petdex.dev/pets/boba/more",
    ],
)
def test_invalid_user_selection_does_not_fetch(monkeypatch, value):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid selection reached network")

    monkeypatch.setattr(registry, "fetch_bytes", unexpected)
    with pytest.raises(ValueError):
        registry.fetch_petdex_source(value)


@pytest.fixture
def capture(monkeypatch):
    calls = []

    def source(metadata, image, image_name, **kwargs):
        return {
            "metadata": metadata,
            "image": image,
            "image_name": image_name,
            **kwargs,
        }

    monkeypatch.setattr(registry, "source_from_bytes", source)

    def fetch(url, **kwargs):
        calls.append((url, kwargs))
        if url.endswith("/api/manifest/v2"):
            return json.dumps(compact()).encode()
        if url.endswith("/pet.json"):
            return b'{"name":"Boba"}'
        if url.endswith("/sprite.webp"):
            return b"image bytes"
        pytest.fail("unexpected URL")

    monkeypatch.setattr(registry, "fetch_bytes", fetch)
    return calls


@pytest.mark.parametrize(
    "value", ["boba", "https://petdex.dev/pets/boba", "https://petdex.dev/pets/boba/"]
)
def test_exact_pet_fetches_only_declared_metadata_and_sprite(capture, value):
    result = registry.fetch_petdex_source(value)
    assert [call[1]["max_bytes"] for call in capture] == [
        10 * 1024 * 1024,
        2 * 1024 * 1024,
        25 * 1024 * 1024,
    ]
    assert result["image"] == b"image bytes"
    assert result["image_name"] == "sprite.webp"
    assert result["registry_entry"]["submittedBy"] == "Creator"
    assert result["guard"]()


def test_no_prefix_matching(capture):
    with pytest.raises(ValueError, match="not found"):
        registry.fetch_petdex_source("bob")
    assert len(capture) == 1


@pytest.mark.parametrize("status", [404, 405, 410, 501])
def test_absent_compact_endpoint_uses_public_legacy(monkeypatch, capture, status):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        if url.endswith("/v2"):
            raise PetdexNetworkError("HTTP error", status_code=status)
        if url.endswith("/api/manifest"):
            return json.dumps(legacy()).encode()
        return b"{}"

    monkeypatch.setattr(registry, "fetch_bytes", fetch)
    assert registry.fetch_petdex_source("boba")["registry_entry"]["slug"] == "boba"
    assert calls[:2] == [
        "https://petdex.dev/api/manifest/v2",
        "https://petdex.dev/api/manifest",
    ]


@pytest.mark.parametrize(
    "failure",
    [
        PetdexNetworkError("rate limited", status_code=429),
        PetdexNetworkError("access denied", status_code=403),
        ValueError("bad JSON"),
    ],
)
def test_security_or_rate_limit_failure_never_falls_back(monkeypatch, capture, failure):
    calls = []

    def fetch(url, **kwargs):
        calls.append(url)
        raise failure

    monkeypatch.setattr(registry, "fetch_bytes", fetch)
    with pytest.raises(ValueError):
        registry.fetch_petdex_source("boba")
    assert calls == ["https://petdex.dev/api/manifest/v2"]


def test_cancel_between_fetches_prevents_next_request(monkeypatch, capture):
    with pytest.raises(ValueError, match="cancel"):
        registry.fetch_petdex_source("boba", cancel_requested=lambda: bool(capture))
    assert len(capture) == 1


def test_real_source_captures_creator_and_immutable_image(monkeypatch):
    import io

    from PIL import Image

    image = io.BytesIO()
    Image.new("RGBA", (96, 117), (255, 0, 0, 255)).save(image, format="PNG")
    original = image.getvalue()
    data = compact()
    data["pets"][0][4] = "/pets/boba/sprite.png"

    def fetch(url, **kwargs):
        if url.endswith("/v2"):
            return json.dumps(data).encode()
        if url.endswith("/pet.json"):
            return b'{"name":"Boba", "spritesheetPath":"spritesheet.png", "spriteVersionNumber":1, "license":"CC-BY-4.0", "notices":"Original creator notice"}'
        return original

    monkeypatch.setattr(registry, "fetch_bytes", fetch)
    source = registry.fetch_petdex_source("boba")
    assert source.title == "Boba"
    assert source.image_bytes == original
    assert source.image_name == "spritesheet.png"
    assert source.is_current()
    artwork = source.artwork
    assert artwork["creator"] == "Creator"
    assert artwork["license"] == "CC-BY-4.0"
    assert artwork["notices"] == "Original creator notice"
    assert artwork["source_url"] == "https://petdex.dev/pets/boba"


def test_registry_preserves_explicit_artwork_notices_and_terms():
    data = legacy()
    data["pets"][0].update(
        license="CC-BY-4.0",
        terms="Do credit the original artist.",
        notices="Original artwork notice.",
    )
    entry = registry.parse_manifest(json.dumps(data).encode())[0]
    assert entry["license"] == "CC-BY-4.0"
    assert entry["terms"] == "Do credit the original artist."
    assert entry["notices"] == "Original artwork notice."
