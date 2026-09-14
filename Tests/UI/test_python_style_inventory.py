"""Regression coverage for syntax bypasses of the visual-style hard floor."""

import pytest

from Tests.UI.python_style_inventory import inventory_styles


@pytest.mark.parametrize(
    ("source", "properties"),
    [
        ("w.styles.width = 3", ["width"]),
        ("w.styles.width: int = 3", ["width"]),
        ("w.styles.width, w.styles.height = 3, 4", ["width", "height"]),
        ("w.styles.width = w.styles.height = 3", ["width", "height"]),
        ("w.set_styles(width=3, margin_top=1)", ["width", "margin_top"]),
        ('w.set_styles("width: 3; color: red; display: block;")', ["width", "color"]),
        ('setattr(w.styles, "border_right", ("solid", "red"))', ["border_right"]),
        ('w.styles.set_styles("height: 3")', ["height"]),
    ],
)
def test_inventory_catches_each_write_form(source, properties):
    writes = inventory_styles(source)
    assert [write.property for write in writes] == properties
    assert all(write.violation for write in writes)


def test_inventory_ignores_comments_strings_and_original_scope_exclusions():
    assert not inventory_styles("""
# w.styles.width = 3
text = "w.set_styles(height=3)"
w.styles.min_width = 3
w.set_styles(max_height=9, display="none")
setattr(w.styles, "min_height", 3)
""")


@pytest.mark.parametrize(
    "expression",
    [
        "3",
        '"auto"',
        "(1, 2)",
        "3 if active else 4",
        "SIZE",
        "self.BORDER_COLOR",
        "(0, SIZE)",
    ],
)
def test_runtime_annotation_cannot_exempt_static_values(expression):
    source = f"""SIZE = 3
w.set_styles(width={expression})  # ds-runtime: measured from current content
"""
    writes = inventory_styles(source)
    assert len(writes) == 1
    assert writes[0].violation
    assert writes[0].value_kind == "static"


@pytest.mark.parametrize(
    "source",
    [
        "# ds-runtime: measured from current content\nw.styles.height = measured_height",
        "w.set_styles(width=event.value)  # ds-runtime: user-selected preview width",
        "w.set_styles(  # ds-runtime: measured terminal region\n    width=self.size.width,\n)",
        "w.set_styles(\n    # ds-runtime: measured terminal region\n    width=self.size.width,\n)",
    ],
)
def test_adjacent_specific_annotation_accepts_runtime_value(source):
    writes = inventory_styles(source)
    assert len(writes) == 1
    assert not writes[0].violation
    assert writes[0].runtime_reason


@pytest.mark.parametrize(
    "marker",
    [
        "",
        "# ds-runtime:",
        "# ds-runtime: dynamic",
        "# ds-runtime: runtime",
        "# ds-runtime: TODO",
        "# ds-runtime: measured width\n\n",
    ],
)
def test_unmarked_or_nonspecific_runtime_expressions_remain_visible(marker):
    writes = inventory_styles(f"{marker}\nw.set_styles(width=measured_width)")
    assert len(writes) == 1
    assert writes[0].violation
    assert writes[0].value_kind == "runtime"


def test_none_reset_is_legal_without_runtime_marker():
    writes = inventory_styles("w.styles.border = None\nw.set_styles(width=None)")
    assert len(writes) == 2
    assert all(not write.violation and write.value_kind == "reset" for write in writes)


def test_inventory_reports_unknown_css_and_expanded_kwargs_instead_of_skipping():
    writes = inventory_styles("w.set_styles(css_text)\nw.set_styles(**options)")
    assert [(write.property, write.violation) for write in writes] == [
        ("*", True),
        ("*", True),
    ]


def test_marker_inside_a_string_is_not_an_annotation():
    writes = inventory_styles(
        'note = "# ds-runtime: measured terminal region"; w.set_styles(width=measured)'
    )
    assert len(writes) == 1
    assert writes[0].violation


def test_static_name_alias_and_finite_conditional_do_not_hide_literal_values():
    writes = inventory_styles("""
SIZE = 3
alias = SIZE
size = alias if active else 4
# ds-runtime: measured terminal region
w.styles.width = size
""")
    assert len(writes) == 1
    assert writes[0].violation
    assert writes[0].value_kind == "static"


def test_pin_refuses_ast_violations_without_rewriting_baseline(tmp_path):
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    script_dir = tmp_path / "Tests/UI"
    script_dir.mkdir(parents=True)
    for filename in ("pin_pattern_ratchets.py", "python_style_inventory.py"):
        shutil.copyfile(Path(__file__).parent / filename, script_dir / filename)
    package = tmp_path / "tldw_chatbook"
    package.mkdir()
    (package / "__init__.py").write_text("")
    css = package / "css"
    css.mkdir()
    (css / "__init__.py").write_text("")
    (css / "build_css.py").write_text("CSS_MODULES = []\n")
    (package / "widget.py").write_text("w.set_styles(height=3)\n")
    baseline = script_dir / "pattern_ratchet_baseline.json"
    original = '{"python_styles": {"widget.py": 99}}\n'
    baseline.write_text(original)
    result = subprocess.run(
        [sys.executable, str(script_dir / "pin_pattern_ratchets.py")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "widget.py:1" in result.stderr
    assert baseline.read_text() == original


def test_function_parameter_is_not_confused_with_a_module_constant():
    writes = inventory_styles("""
width = 3
def resize(width):
    w.set_styles(width=width)  # ds-runtime: measured terminal width
""")
    assert len(writes) == 1
    assert not writes[0].violation


@pytest.mark.parametrize(
    "expression",
    [
        'Color.parse("red")',
        'Scalar.parse("100%")',
        "Spacing(1, 2, 1, 2)",
        "COLORS[state]",
    ],
)
def test_literal_constructors_and_finite_lookup_remain_static(expression):
    writes = inventory_styles(f"""COLORS = {{"good": "green", "bad": "red"}}
w.set_styles(color={expression})  # ds-runtime: user-selected theme preview
""")
    assert len(writes) == 1
    assert writes[0].violation
    assert writes[0].value_kind == "static"


def test_css_keyword_cannot_hide_string_or_dynamic_declarations():
    writes = inventory_styles(
        'w.set_styles(css="height: 3; color: red;")\nw.set_styles(css=theme_css)'
    )
    assert [(write.property, write.violation) for write in writes] == [
        ("height", True),
        ("color", True),
        ("*", True),
    ]
