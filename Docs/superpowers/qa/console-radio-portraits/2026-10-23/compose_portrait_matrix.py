"""Build exact paired artwork only after explicit local-compositing authorization."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


def main() -> None:
    """Preserve one original base; vary only shared eyelid and mouth patches."""
    repo = Path(__file__).resolve().parents[5]
    previous = repo / "output/imagegen/console-radio-samira-2026-10-23"
    output = previous / "paired-v2"
    assert not output.exists(), "Refuse to overwrite an existing correction"
    source = repo / "tldw_chatbook/assets/characters/samira/expressions/neutral.webp"
    inputs = {
        "source": source,
        "eyes_closed": previous / "ready/blink_rest.png",
        "mouth_small": previous / "ready/small.png",
        "mouth_open": previous / "ready/open.png",
    }
    hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in inputs.items()
    }
    with Image.open(source) as image:
        base = image.convert("RGB")
    assert base.size == (1024, 1024)
    eye_mask = Image.new("L", base.size)
    draw = ImageDraw.Draw(eye_mask)
    draw.polygon(
        [
            (531, 344),
            (551, 337),
            (574, 337),
            (593, 345),
            (606, 357),
            (600, 373),
            (576, 378),
            (549, 370),
            (532, 360),
        ],
        fill=255,
    )
    draw.polygon(
        [
            (639, 356),
            (652, 349),
            (674, 350),
            (691, 358),
            (696, 375),
            (682, 389),
            (657, 386),
            (640, 374),
        ],
        fill=255,
    )
    eye_mask = eye_mask.filter(ImageFilter.GaussianBlur(2))
    mouth_mask = Image.new("L", base.size)
    ImageDraw.Draw(mouth_mask).polygon(
        [
            (572, 458),
            (591, 450),
            (615, 448),
            (644, 457),
            (657, 471),
            (650, 496),
            (634, 509),
            (600, 504),
            (580, 488),
            (570, 474),
        ],
        fill=255,
    )
    mouth_mask = mouth_mask.filter(ImageFilter.GaussianBlur(2))
    eyes = np.asarray(eye_mask) > 0
    mouth = np.asarray(mouth_mask) > 0
    assert not np.any(eyes & mouth)
    fixed = ~(eyes | mouth)
    base_pixels = np.asarray(base)
    frames = {}
    mouth_bases = {"rest": base.copy()}
    for role, key in (("small", "mouth_small"), ("open", "mouth_open")):
        with Image.open(inputs[key]) as image:
            assert image.size == base.size
            mouth_bases[role] = Image.composite(image.convert("RGB"), base, mouth_mask)
    with Image.open(inputs["eyes_closed"]) as image:
        closed = image.convert("RGB")
    for role, image in mouth_bases.items():
        frames[role] = image
        frames["blink_" + role] = Image.composite(closed, image, eye_mask)
    arrays = {role: np.asarray(image) for role, image in frames.items()}
    for role, pixels in arrays.items():
        assert np.array_equal(pixels[fixed], base_pixels[fixed]), role
    for role in ("rest", "small", "open"):
        assert np.array_equal(arrays[role][mouth], arrays["blink_" + role][mouth]), role
        assert np.array_equal(arrays[role][~eyes], arrays["blink_" + role][~eyes]), role
    for role in ("small", "open"):
        assert np.array_equal(arrays["rest"][eyes], arrays[role][eyes])
        assert np.array_equal(arrays["blink_rest"][eyes], arrays["blink_" + role][eyes])
    assert np.array_equal(arrays["rest"], base_pixels)
    assert len({hashlib.sha256(p.tobytes()).hexdigest() for p in arrays.values()}) == 6
    (output / "selected").mkdir(parents=True)
    manifest = {
        "method": "One unchanged original base, shared generated eyelid/mouth patches; local compositing explicitly authorized by the user.",
        "input_sha256": hashes,
        "frames": {},
        "checks": {
            "outside_edit_pixels_equal_original": True,
            "mouth_equal_across_eye_pair": True,
            "eyes_equal_across_mouth_states": True,
            "rest_equals_original_pixels": True,
        },
    }
    for role, image in frames.items():
        path = output / "selected" / (role + ".png")
        image.info.clear()
        image.save(path)
        manifest["frames"][role] = {
            "file": "selected/" + path.name,
            "eyes": "closed" if role.startswith("blink_") else "open",
            "mouth": role.removeprefix("blink_"),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "width": 1024,
            "height": 1024,
        }
    eye_mask.save(output / "eyelid-mask.png")
    mouth_mask.save(output / "mouth-mask.png")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Diagnostic face crops keep the six eye/mouth combinations easy to compare.
    sheet = Image.new("RGB", (1080, 860), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
    small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 16)
    draw.text(
        (15, 12),
        "Same base portrait · 3 mouth states × 2 eye states",
        fill="black",
        font=font,
    )
    crop = (455, 285, 725, 565)
    for row, prefix in enumerate(("", "blink_")):
        for col, role in enumerate(("rest", "small", "open")):
            x, y = col * 360 + 15, row * 390 + 65
            label = (
                ("Eyes closed" if row else "Eyes open")
                + " · Mouth "
                + ("closed" if role == "rest" else role)
            )
            draw.text((x, y), label, fill="black", font=small_font)
            detail = (
                frames[prefix + role]
                .crop(crop)
                .resize((324, 336), Image.Resampling.LANCZOS)
            )
            sheet.paste(detail, (x, y + 28))
    draw.text(
        (15, 840),
        "Face detail for review; every import file retains the full 1024 × 1024 canvas.",
        fill="black",
        font=small_font,
    )
    sheet.save(output / "paired-review.png")
    for name, path in inputs.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[name]
    print(
        "PASS: six exact eye/mouth combinations; invariant base pixels; source files unchanged"
    )


if __name__ == "__main__":
    main()
