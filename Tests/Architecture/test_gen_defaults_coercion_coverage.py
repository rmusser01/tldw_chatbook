"""Every `FieldSpec.kind` a gen-defaults schema declares must be coercible.

Tier-2 review S19 [D4] found `settings_image_gen_defaults._coerce_value` and
`settings_video_gen_defaults._coerce_value` are parallel re-rolls that have
drifted: image handles `int`/`float`/`origin`, video handles `int`/`bool`.
Today nothing breaks -- neither schema declares a kind its own coercer is
missing -- but `_coerce_value`'s output is what `diff_to_sections` writes
into `config.toml`. Adding, say, a `bool` field to the image schema would
silently persist the raw editor string `"False"` (truthy) where a real
boolean is expected, and `validate_draft` would not catch it.

The review's recommended fix is one shared `settings_gen_defaults_common.py`
holding `FieldSpec`, `_spec_for`, `canonical_backend_order`, `_coerce_value`
and the `diff_to_sections` skeleton. That is an M-sized refactor of two live
settings surfaces with zero observable behaviour change -- and it is not what
makes the defect impossible. THIS is: the day someone adds a field of a kind
their module cannot coerce, CI says so, whether or not the modules were ever
merged.

Kinds that pass through unchanged as strings (`text`, `url`, `path`,
`secret`, `region`, ...) are correct to leave alone; only the kinds a
coercer is expected to TYPE are listed below.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.UI.Screens import (
    settings_image_gen_defaults as image_defaults,
    settings_video_gen_defaults as video_defaults,
)

#: Kinds that must come back as something other than the raw string. A kind
#: absent here is a deliberate pass-through (free text, URLs, paths, secrets).
_TYPED_KINDS: dict[str, tuple[str, type | tuple[type, ...]]] = {
    "int": ("7", int),
    "float": ("1.5", float),
    "bool": ("false", bool),
}


@pytest.mark.unit
@pytest.mark.parametrize(
    "module",
    (image_defaults, video_defaults),
    ids=("image_gen", "video_gen"),
)
def test_every_declared_typed_kind_is_coerced_by_its_own_module(module) -> None:
    """A schema may not declare a typed kind its `_coerce_value` ignores."""
    uncoerced: list[tuple[str, str, str]] = []
    for backend_id, specs in module.FIELD_SCHEMA.items():
        for spec in specs:
            probe = _TYPED_KINDS.get(spec.kind)
            if probe is None:
                continue
            raw, expected_type = probe
            coerced = module._coerce_value(spec, raw)
            if not isinstance(coerced, expected_type):
                uncoerced.append((backend_id, spec.toml_key, spec.kind))

    assert not uncoerced, (
        f"{module.__name__}._coerce_value has no branch for these declared "
        f"kinds, so diff_to_sections would write the raw editor string into "
        f"config.toml: {uncoerced}. Add the branch (the sibling gen-defaults "
        f"module already has it) -- do not remove the field."
    )
