from dataclasses import FrozenInstanceError

import pytest
from tldw_profile_core.evidence import exact_text_span_digests


class CustomText(str):
    def encode(self, *_args, **_kwargs):
        raise AssertionError("custom encoding must not be called")


class CustomOffset(int):
    pass


SOURCE = "Hi 👋 — café"
SOURCE_SHA = "bdd355f021e0095bbd9ce9da729cfd0b9556b01bc01c5b59461ebc4e12e55bef"
SPAN_SHA = "850f7dc43910ff890f8879c0ed26fe697c93a067ad93a7d50f466a7028a9bf4e"
EMPTY_SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_accepted_unicode_vector_and_immutable_result():
    result = exact_text_span_digests(SOURCE, 7, 11)
    assert result.representation_sha256 == SOURCE_SHA
    assert result.span_sha256 == SPAN_SHA
    assert exact_text_span_digests(SOURCE, 7, 11) == result
    with pytest.raises(FrozenInstanceError):
        result.span_sha256 = EMPTY_SHA


@pytest.mark.parametrize(
    "text,start,end,representation_sha,span_sha",
    [
        (
            "Hi 👋 — cafe\u0301",
            7,
            12,
            "b433fe833253863d292ee2dbf15f9ea928a33bc28eb61bca9aeaba23bb2d030c",
            "81ef060bcd98adc7824eb5c1ada83c32491b16018e11e79f00ab9d09e04b015a",
        ),
        (
            "A\r\nB",
            1,
            3,
            "255e24970eef1cf6a0503f246be4b2ecd25d69bbbeb4073f4abb26d62886f64b",
            "7eb70257593da06f682a3ddda54a9d260d4fc514f645237f5ca74b08f8da61a6",
        ),
        (
            "A\nB",
            1,
            2,
            "23519a43c66b4c342f25b32e09797ec5f3fc0be388cd8243fb3449afbdce4013",
            "01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b",
        ),
        (
            "👋",
            0,
            1,
            "1d0452e3d194cc7950909b578c611d5ad4cd15105c6aeefc38ce213240ffc457",
            "1d0452e3d194cc7950909b578c611d5ad4cd15105c6aeefc38ce213240ffc457",
        ),
    ],
)
def test_exact_text_vectors(text, start, end, representation_sha, span_sha):
    result = exact_text_span_digests(text, start, end)
    assert result.representation_sha256 == representation_sha
    assert result.span_sha256 == span_sha


def test_outside_span_edit_changes_source_identity_only():
    original = exact_text_span_digests(SOURCE, 7, 11)
    edited = exact_text_span_digests(SOURCE + "!", 7, 11)
    assert edited.span_sha256 == original.span_sha256 == SPAN_SHA
    assert edited.representation_sha256 == (
        "acb2a8f3ad9c66ce276c5db2c069c559e25c3d55c779f3607f0c92b0af7905c9"
    )
    assert edited.representation_sha256 != original.representation_sha256


def test_valid_but_shifted_offset_does_not_match_accepted_span():
    shifted = exact_text_span_digests(SOURCE, 8, 11)
    assert shifted.representation_sha256 == SOURCE_SHA
    assert shifted.span_sha256 == (
        "6a43309ff7bbc6b3fd79d434521fcc2d1a77e71cfbdc894a11de20ede7ee929b"
    )
    assert shifted.span_sha256 != SPAN_SHA


@pytest.mark.parametrize("text,index", [("", 0), (SOURCE, 0), (SOURCE, 11)])
def test_empty_span_at_valid_boundary(text, index):
    result = exact_text_span_digests(text, index, index)
    assert result.span_sha256 == EMPTY_SHA
    assert result.representation_sha256 == (EMPTY_SHA if not text else SOURCE_SHA)


@pytest.mark.parametrize(
    "text,start,end",
    [
        (None, 0, 0),
        (b"abc", 0, 1),
        (42, 0, 0),
        ("abc", True, 2),
        ("abc", 0, False),
        ("abc", 1.0, 2),
        ("abc", 0, 2.0),
        ("abc", 0, 1.5),
        ("abc", "0", 2),
        ("abc", 0, None),
        pytest.param(CustomText("abc"), 0, 1, id="string-subclass"),
        pytest.param("abc", CustomOffset(0), 1, id="start-subclass"),
        pytest.param("abc", 0, CustomOffset(1), id="end-subclass"),
    ],
)
def test_wrong_scalar_types_are_not_coerced(text, start, end):
    with pytest.raises(TypeError):
        exact_text_span_digests(text, start, end)


@pytest.mark.parametrize("start,end", [(-1, 2), (0, -1), (2, 1), (0, 4), (4, 4)])
def test_invalid_bounds_are_not_silently_sliced(start, end):
    with pytest.raises(ValueError):
        exact_text_span_digests("abc", start, end)


@pytest.mark.parametrize("text,start,end", [("\ud800", 0, 1), ("a\udfff", 0, 1)])
def test_unencodable_whole_representation_is_rejected(text, start, end):
    with pytest.raises(UnicodeEncodeError):
        exact_text_span_digests(text, start, end)
