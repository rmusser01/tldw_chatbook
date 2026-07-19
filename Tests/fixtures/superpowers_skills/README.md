# superpowers skill fixtures

Provenance: `executing-plans/`, `requesting-code-review/` (incl. its
`code-reviewer.md` supporting file), `using-superpowers/`,
`verification-before-completion/`, and `writing-plans/` are unmodified
copies of `SKILL.md` (and, for `requesting-code-review`, its one sibling
file) from the [obra/superpowers](https://github.com/obra/superpowers)
plugin, version 6.1.1 (MIT license, Copyright (c) 2025 Jesse Vincent), as
cached locally at
`~/.claude/plugins/cache/claude-plugins-official/superpowers/6.1.1/skills/`.
`using-superpowers`'s real `references/` subfolder is deliberately NOT
copied here (see `Tests/Skills/test_skills_import.py` for why).

`executing-plans-with-metadata/SKILL.md` is NOT a raw upstream copy: it
reuses `executing-plans`'s real description/body verbatim under a distinct
skill name, with extra frontmatter fields (`argument_hint`,
`allowed-tools`, `license`, `compatibility`, `model`, `context`,
`metadata`, plus two unrecognized fields, `priority` and `tags`) added on
top to exercise fields the real superpowers skillset never uses on its
own.
