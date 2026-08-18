import os
from pathlib import Path

from tldw_chatbook.Skills_Interop.project_skills_discovery import (
    MAX_DISCOVERED_ENTRIES,
    discover_project_skills,
    find_project_dir_with_skills,
    find_project_skills_dir,
)


def _skill_dir(root, name, description="Does a thing."):
    d = root / ".SKILLS" / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {description}\n---\nBody\n",
        encoding="utf-8",
    )
    return d


def test_no_skills_dir_returns_none(tmp_path):
    assert discover_project_skills(tmp_path) is None


def test_discovers_directory_and_loose_file_skills(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    (tmp_path / ".SKILLS" / "beta-skill.md").write_text(
        "---\ndescription: Loose one.\n---\nBody\n", encoding="utf-8"
    )
    discovery = discover_project_skills(tmp_path)
    kinds = {(e.name, e.kind, e.status) for e in discovery.entries}
    assert ("alpha-skill", "directory", "ok") in kinds
    assert ("beta-skill", "file", "ok") in kinds
    assert discovery.truncated == 0


def test_subdir_without_skill_md_is_skipped_with_reason(tmp_path):
    (tmp_path / ".SKILLS" / "not-a-skill").mkdir(parents=True)
    discovery = discover_project_skills(tmp_path)
    assert ("not-a-skill", "no SKILL.md") in discovery.skipped


def test_invalid_name_flagged_not_failed(tmp_path):
    _skill_dir(tmp_path, "My_Skill")
    discovery = discover_project_skills(tmp_path)
    entry = discovery.entries[0]
    assert entry.status == "invalid"
    assert "lowercase" in entry.reason


def test_symlinked_skills_dir_refused(tmp_path):
    real = tmp_path / "elsewhere"
    real.mkdir()
    os.symlink(real, tmp_path / ".SKILLS")
    assert find_project_skills_dir(tmp_path) is None


def test_symlinked_entry_skipped(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    outside = tmp_path / "outside"
    outside.mkdir()
    os.symlink(outside, tmp_path / ".SKILLS" / "sneaky")
    discovery = discover_project_skills(tmp_path)
    assert ("sneaky", "symlink") in discovery.skipped
    assert [e.name for e in discovery.entries] == ["alpha-skill"]


def test_entry_cap_reports_truncation(tmp_path):
    for i in range(MAX_DISCOVERED_ENTRIES + 3):
        _skill_dir(tmp_path, f"skill-{i:03d}")
    discovery = discover_project_skills(tmp_path)
    assert len(discovery.entries) == MAX_DISCOVERED_ENTRIES
    assert discovery.truncated == 3


def test_fingerprint_changes_when_a_skill_is_added(tmp_path):
    _skill_dir(tmp_path, "alpha-skill")
    first = discover_project_skills(tmp_path).fingerprint
    _skill_dir(tmp_path, "gamma-skill")
    assert discover_project_skills(tmp_path).fingerprint != first


def test_fingerprint_changes_when_skill_md_edited_in_place(tmp_path):
    d = _skill_dir(tmp_path, "alpha-skill")
    first = discover_project_skills(tmp_path).fingerprint
    skill_md = d / "SKILL.md"
    skill_md.write_text(
        "---\nname: alpha-skill\ndescription: Changed content.\n---\nNew body\n",
        encoding="utf-8",
    )
    # Bump mtime_ns explicitly (rather than sleeping) so the assertion doesn't
    # depend on filesystem mtime resolution catching a fast in-process edit.
    st = skill_md.stat()
    os.utime(skill_md, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    second = discover_project_skills(tmp_path).fingerprint
    assert second != first


def test_hostile_description_survives_as_plain_data(tmp_path):
    _skill_dir(tmp_path, "alpha-skill", description='"[red]evil[/red]"')
    discovery = discover_project_skills(tmp_path)
    assert discovery.entries[0].description == "[red]evil[/red]"  # escaping is UI-side


def test_unparseable_frontmatter_degrades_to_empty_description(tmp_path):
    # Unquoted brackets break the front matter's YAML grammar; the importer's
    # own _parse_front_matter degrades this to empty metadata (not an error),
    # and discovery mirrors that -- still "ok", still importable, just no
    # preview text.
    _skill_dir(tmp_path, "alpha-skill", description="[red]evil[/red]")
    discovery = discover_project_skills(tmp_path)
    entry = discovery.entries[0]
    assert entry.status == "ok"
    assert entry.description == ""


def test_ancestor_walk_finds_project_root(tmp_path):
    _skill_dir(tmp_path / "repo", "alpha-skill")
    (tmp_path / "repo" / ".git").mkdir()
    sub = tmp_path / "repo" / "src" / "deep"
    sub.mkdir(parents=True)
    assert find_project_dir_with_skills(sub) == tmp_path / "repo"


def test_ancestor_walk_stops_at_git_root_without_skills(tmp_path):
    (tmp_path / "repo" / ".git").mkdir(parents=True)
    sub = tmp_path / "repo" / "src"
    sub.mkdir()
    _skill_dir(tmp_path, "above-the-repo")  # must NOT be found past the .git root
    assert find_project_dir_with_skills(sub) is None


def test_ancestor_walk_stops_at_home(monkeypatch, tmp_path):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    start = tmp_path / "sub"
    start.mkdir()
    _skill_dir(tmp_path, "in-home-itself")
    assert find_project_dir_with_skills(start) is None


def test_ancestor_walk_stops_at_home_reached_via_symlinked_ancestor(monkeypatch, tmp_path):
    # The walk start is reached through a symlinked path component that
    # resolves onto $HOME. The unresolved `current` never string-equals the
    # resolved `home`, so a stop-check that compares unresolved `current`
    # against resolved `home` never fires -- the walk sails past the $HOME
    # boundary and can discover skills planted at (or beyond) home.
    real_home = tmp_path / "real_home"
    real_home.mkdir()
    _skill_dir(real_home, "in-home-itself")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: real_home))

    link_root = tmp_path / "link_root"
    link_root.mkdir()
    home_link = link_root / "home_link"
    os.symlink(real_home, home_link)

    start = home_link / "sub"
    start.mkdir()

    assert find_project_dir_with_skills(start) is None
