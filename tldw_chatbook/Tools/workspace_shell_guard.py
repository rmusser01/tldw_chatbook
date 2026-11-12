"""Recognize common raw-shell deletes without evaluating shell code."""

from collections.abc import Iterator
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
from typing import Literal, TypeAlias


ShellDialect: TypeAlias = Literal["bash", "powershell", "cmd"]
WorkspaceDeletionRule: TypeAlias = Literal[
    "workspace-folder-delete", "workspace-folder-wipe"
]
_ItemKind: TypeAlias = Literal["item", "directory", "file", "cmd-file"]

_PS_REMOVE = frozenset({"remove-item", "ri", "rm", "rmdir", "rd", "del", "erase"})
_PS_LIST = frozenset({"get-childitem", "gci", "dir", "ls"})
_PS_LOCATION = frozenset({"set-location", "sl", "cd", "chdir"})
_PS_VALUE_OPTIONS = frozenset({"-erroraction", "-ea", "-warningaction", "-wa"})
_GLOB_CHARS = "*?["


@dataclass(frozen=True, slots=True)
class _Word:
    text: str
    wildcards: tuple[int, ...] = ()
    dynamic: bool = False
    starts_quoted: bool = False


@dataclass(frozen=True, slots=True)
class _Segment:
    words: list[_Word]
    separator: str
    redirected: bool


@dataclass(frozen=True, slots=True)
class _Delete:
    path: str
    kind: _ItemKind = "item"
    recursive: bool = False
    children: bool = False
    prune_parents: bool = False


def _segments(command: str, dialect: ShellDialect) -> list[_Segment]:
    """Split simple commands while preserving literal and wildcard operands."""
    segments: list[_Segment] = []
    words: list[_Word] = []
    chars: list[str] = []
    wildcards: list[int] = []
    quote = ""
    nesting = 0
    dynamic = starts_quoted = quoted_word = opaque = started = redirecting = False
    redirected = literal_call = False
    escape = {"bash": "\\", "powershell": chr(96), "cmd": "^"}[dialect]
    quotes = '"' if dialect == "cmd" else "\"'"
    separators = "&|\n" if dialect == "cmd" else ";&|\n"

    def finish_word() -> None:
        nonlocal dynamic, starts_quoted, quoted_word, started, redirecting
        if started:
            if not redirecting:
                words.append(
                    _Word(
                        "".join(chars),
                        tuple(wildcards),
                        dynamic,
                        starts_quoted and not (literal_call and not words),
                    )
                )
            redirecting = False
        chars.clear()
        wildcards.clear()
        dynamic = starts_quoted = quoted_word = started = False

    i = 0
    while i < len(command):
        char = command[i]
        following = command[i + 1 : i + 2]
        i += 1
        # Bash here-doc bodies and PowerShell here-strings are script data.
        # Leave the rest opaque rather than scanning them as shell statements.
        if not quote and (
            (dialect == "bash" and char == "<" and following == "<")
            or (
                dialect == "powershell"
                and char == "@"
                and following in quotes
                and following
                and command[i + 1 : i + 2] in {"\r", "\n"}
            )
        ):
            return segments
        if dialect == "powershell" and not quote and char == "<" and following == "#":
            finish_word()
            end = command.find("#>", i + 1)
            if end < 0:
                return segments
            i = end + 2
            continue
        if (
            not quote
            and not nesting
            and char == "#"
            and not started
            and dialect != "cmd"
        ):
            end = command.find("\n", i)
            i = len(command) if end < 0 else end
            continue
        if (
            char == escape
            and following
            and (
                not quote
                or (
                    dialect == "bash"
                    and quote == '"'
                    and following in '\\"$\n' + chr(96)
                )
                or (dialect == "powershell" and quote == '"')
            )
        ):
            i += 1
            if following != "\n":
                chars.append(following)
                quoted_word = started = True
            continue
        if char == quote:
            if dialect == "powershell" and following == quote:
                chars.append(char)
                i += 1
            else:
                quote = ""
            continue
        if not quote and char in quotes:
            quote = char
            starts_quoted = starts_quoted or not started
            quoted_word = started = True
            continue
        if dialect == "bash" and char == chr(96) and quote != "'":
            opaque = True
            end = command.find(chr(96), i)
            if end < 0:
                return segments
            i = end + 1
            continue
        if not quote:
            if not nesting and (
                char in "<>" or (dialect == "bash" and char == "&" and following == ">")
            ):
                # An adjacent descriptor (or PowerShell's all-streams *)
                # and the following redirect target are not command operands.
                if (
                    not redirecting
                    and not quoted_word
                    and (
                        "".join(chars).isdecimal()
                        or (dialect == "powershell" and chars == ["*"])
                    )
                ):
                    started = False
                finish_word()
                opaque |= redirecting
                redirecting = True
                redirected = True
                if char == "&":
                    char = ">"
                    i += 1
                suffix = command[i : i + 1]
                if suffix == char or (
                    dialect == "bash" and char + suffix in {">|", "<>"}
                ):
                    i += 1
                if command[i : i + 1] == "&":
                    i += 1
                continue
            if char in "({":
                nesting += 1
                opaque = True
            elif char in ")}":
                nesting = max(nesting - 1, 0)
                opaque = True
            if not nesting and char in separators:
                if (
                    char == "\n"
                    and dialect != "cmd"
                    and not (words or started or opaque or redirecting or literal_call)
                    and segments
                    and segments[-1].separator in {"&&", "||", "|"}
                ):
                    continue
                if (
                    dialect == "powershell"
                    and char == "&"
                    and following != "&"
                    and not (words or started or redirecting or literal_call)
                ):
                    literal_call = True
                    continue
                finish_word()
                if char in "&|" and following == char:
                    char += following
                    i += 1
                segments.append(
                    _Segment([] if opaque or redirecting else words, char, redirected)
                )
                words = []
                opaque = redirecting = redirected = literal_call = False
                continue
            if char.isspace() or (dialect == "powershell" and char == ","):
                finish_word()
                continue
        if char in _GLOB_CHARS and (not quote or dialect != "bash"):
            wildcards.append(len(chars))
        if (
            (dialect == "cmd" and char in "%!")
            or (dialect != "cmd" and char == "$" and quote != "'")
            or (dialect == "bash" and char == "~" and not started)
        ):
            dynamic = True
        chars.append(char)
        started = True
    finish_word()
    segments.append(
        _Segment(
            [] if quote or nesting or opaque or redirecting else words, "", redirected
        )
    )
    return segments


def _name(words: list[_Word], dialect: ShellDialect) -> str:
    if not words or words[0].dynamic:
        return ""
    first = words[0]
    if dialect == "powershell" and first.starts_quoted:
        return ""
    if dialect == "bash":
        return first.text.rsplit("/", 1)[-1]
    return first.text.replace("\\", "/").rsplit("/", 1)[-1].lstrip("@").casefold()


def _literal_builtin(segment: _Segment, dialect: ShellDialect) -> str:
    words = segment.words
    if not words or segment.redirected or any(word.dynamic for word in words):
        return ""
    if dialect == "powershell" and words[0].starts_quoted:
        return ""
    if dialect == "bash":
        return words[0].text
    return words[0].text.lstrip("@").casefold()


def _is_location(name: str, dialect: ShellDialect) -> bool:
    return name in (_PS_LOCATION if dialect == "powershell" else {"cd", "chdir"})


def _operand(
    word: _Word,
    dialect: ShellDialect,
    kind: _ItemKind = "item",
    recursive: bool = False,
    *,
    literal: bool = False,
    prune_parents: bool = False,
) -> _Delete | None:
    if not word.text or word.dynamic:
        return None
    if literal or not word.wildcards:
        return _Delete(word.text, kind, recursive, prune_parents=prune_parents)
    separators = "/\\" if os.name == "nt" else "/"
    text = word.text.rstrip(separators)
    start = max(text.rfind(sep) for sep in separators) + 1
    leaf = text[start:]
    all_stars = leaf == "*" or (dialect == "bash" and leaf and not leaf.strip("*"))
    if all_stars and word.wildcards == tuple(range(start, len(text))):
        return _Delete(text[:start] or ".", kind, recursive, children=True)
    if dialect == "cmd" and leaf == "*.*" and word.wildcards == (start, start + 2):
        return _Delete(text[:start] or ".", kind, recursive, children=True)
    return None


def _bash_deletes(name: str, args: list[_Word]) -> list[_Delete | None]:
    if name == "find":
        # Only a single literal starting point and unfiltered -delete. Depth
        # zero may remove the starting entry, while depth one preserves it.
        start = args.pop(0) if args and not args[0].text.startswith("-") else _Word(".")
        expression = [word.text for word in args]
        if not expression or expression[-1] != "-delete":
            return []
        options = tuple(word for word in expression[:-1] if word != "-depth")
        if options not in {(), ("-mindepth", "0"), ("-mindepth", "1")}:
            return []
        target = _operand(start, "bash", recursive=True)
        if target is None or target.children:
            return []
        return [
            _Delete(target.path, recursive=True, children=options == ("-mindepth", "1"))
        ]
    if name not in {"rm", "rmdir"}:
        return []

    flags: set[str] = set()
    operands = []
    options_enabled = True
    for word in args:
        text = word.text
        if options_enabled and text == "--":
            options_enabled = False
        elif options_enabled and text.startswith("--"):
            flags.add(text.partition("=")[0])
        elif options_enabled and text.startswith("-") and len(text) > 1:
            flags.update(text[1:])
        else:
            operands.append(word)
    recursive = name == "rm" and bool(flags & {"r", "R", "--recursive"})
    kind: _ItemKind = "file"
    if name == "rmdir":
        kind = "directory"
    elif recursive or flags & {"d", "--dir"}:
        kind = "item"
    prune_parents = name == "rmdir" and bool(flags & {"p", "--parents"})
    return [
        _operand(word, "bash", kind, recursive, prune_parents=prune_parents)
        for word in operands
    ]


def _ps_arguments(
    args: list[_Word],
) -> tuple[list[_Word], bool, bool] | None:
    paths: list[_Word] = []
    literal = recursive = False
    args_iter = iter(args)
    for word in args_iter:
        flag = word.text.casefold()
        switch, colon, value = flag.partition(":")
        if switch in {"-recurse", "-r", "-force", "-verbose", "-confirm", "-whatif"}:
            if colon and value not in {"$true", "$false"}:
                return None
            enabled = value != "$false"
            if switch in {"-recurse", "-r"}:
                recursive = enabled
            elif switch == "-whatif" and enabled:
                return None
        elif flag in {"-literalpath", "-pspath", "-lp"}:
            literal = True
        elif flag in _PS_VALUE_OPTIONS:
            next(args_iter, None)
        elif flag == "-path":
            continue
        elif flag.startswith("-"):
            # Filters, type/depth restrictions, previews and unsupported
            # parameters do not describe an unconditional deletion.
            return None
        else:
            paths.append(word)
    return paths, literal, recursive


def _powershell_deletes(
    name: str, args: list[_Word], pipe_target: list[_Word] | None
) -> list[_Delete | None]:
    if name not in _PS_REMOVE and not (name in _PS_LIST and pipe_target):
        return []
    parsed = _ps_arguments(args)
    if parsed is None:
        return []
    paths, literal, recursive = parsed
    if name in _PS_REMOVE:
        return [
            _operand(word, "powershell", recursive=recursive, literal=literal)
            for word in paths
        ]
    if pipe_target is None or _name(pipe_target, "powershell") not in _PS_REMOVE:
        return []
    downstream = _ps_arguments(pipe_target[1:])
    if downstream is None or downstream[0]:
        return []
    targets = []
    for word in paths or [_Word(".")]:
        target = _operand(word, "powershell", literal=literal)
        if target is not None:
            targets.append(
                _Delete(
                    target.path, recursive=recursive or downstream[2], children=True
                )
            )
    return targets


def _cmd_deletes(name: str, args: list[_Word]) -> list[_Delete | None]:
    if name not in {"del", "erase", "rd", "rmdir"}:
        return []
    recursive = False
    operands = []
    for word in args:
        if word.text.casefold().startswith("/a:"):
            return []
        flag = word.text.casefold()
        if flag == "/s":
            recursive = True
        elif flag not in {"/q", "/p", "/f", "/a"}:
            operands.append(word)
    kind: _ItemKind = "cmd-file" if name in {"del", "erase"} else "directory"
    return [_operand(word, "cmd", kind, recursive) for word in operands]


def _path(value: str, cwd: Path | None, dialect: ShellDialect) -> Path | None:
    if os.name == "nt" and dialect == "bash" and re.match(r"^/[a-zA-Z](?:/|$)", value):
        # Git Bash's drive-root spelling.
        value = value[1] + ":/" + value[3:]
    path = Path(value)
    if path.is_absolute():
        return path
    if cwd is None or (path.drive and path.drive.casefold() != cwd.drive.casefold()):
        return None
    return cwd / path


def _changed_directory(
    args: list[_Word], cwd: Path | None, dialect: ShellDialect
) -> tuple[Path | None, bool | None]:
    """Return the new directory and the known status of a literal location change."""
    literal = False
    switch_drive = False
    if dialect == "powershell":
        parsed = _ps_arguments(args)
        if parsed is None:
            return None, None
        args, literal, _ = parsed
    elif args and args[0].text.casefold() == ("/d" if dialect == "cmd" else "--"):
        switch_drive = dialect == "cmd"
        args = args[1:]
    if len(args) != 1 or args[0].dynamic or (args[0].wildcards and not literal):
        return None, None
    path = _path(args[0].text, cwd, dialect)
    if path is None:
        return None, None
    if not path.is_dir():
        return cwd, False
    if (
        dialect == "cmd"
        and not switch_drive
        and cwd is not None
        and path.drive.casefold() != cwd.drive.casefold()
    ):
        return cwd, True
    return path.resolve(), True


def _directory_options(
    segments: list[_Segment], dialect: ShellDialect, cwd: Path
) -> Iterator[set[Path | None]]:
    """Track literal working directories through simple success and fallback chains."""
    active: set[Path | None] = {cwd}
    skipped: set[Path | None] = set()
    preceding = ""
    in_pipeline = False
    for segment in segments:
        yield active
        words, separator = segment.words, segment.separator
        if separator == "|":
            in_pipeline = True
            continue
        if (
            _is_location(_name(words, dialect), dialect)
            and not in_pipeline
            and (separator != "&" or dialect == "cmd")
        ):
            succeeded: set[Path | None] = set()
            failed: set[Path | None] = set()
            for directory in active:
                target, success = _changed_directory(words[1:], directory, dialect)
                if success is not False:
                    succeeded.add(target)
                if success is not True or segment.redirected:
                    failed.add(directory)
        else:
            builtin = (
                _literal_builtin(segment, dialect)
                if dialect == "bash" and not in_pipeline
                else ""
            )
            succeeded = set(active) if builtin != "false" else set()
            failed = set(active) if builtin != "true" else set()
        if preceding == "||":
            succeeded.update(skipped)
        elif preceding == "&&":
            failed.update(skipped)
        if separator == "&&":
            active, skipped = succeeded, failed
        elif separator == "||":
            active, skipped = failed, succeeded
        else:
            active, skipped = succeeded | failed, set()
        # More than two possible locations indicates a compound branch.
        if len(active) > 2:
            active = {None}
        if len(skipped) > 2:
            skipped = {None}
        preceding = separator
        in_pipeline = False


def _entry_path(path: Path) -> Path:
    # Resolve traversal through links but leave a final link as the entry
    # that rm/Remove-Item/rd can remove without deleting its destination.
    if not path.name or path.name == "..":
        return path.resolve()
    # Let Windows canonicalize alternate spellings of ordinary entries,
    # including short names and trailing dots. Preserve a final link.
    if os.name == "nt" and not path.is_symlink() and not path.is_junction():
        return path.resolve()
    return path.parent.resolve() / path.name


def _same_entry(path: Path, other: Path) -> bool:
    if path == other:
        return True
    if str(path).casefold() != str(other).casefold():
        return False
    # Path equality is case-sensitive on POSIX even on a case-insensitive
    # volume. Compare native identities without following a final link.
    try:
        return os.path.samestat(path.lstat(), other.lstat())
    except (FileNotFoundError, NotADirectoryError):
        return False


def _removes_child(path: Path, deletion: _Delete) -> bool:
    try:
        directory = stat.S_ISDIR(path.lstat().st_mode) and not path.is_junction()
    except (FileNotFoundError, NotADirectoryError):
        return False
    except OSError as exc:
        # ERROR_INVALID_NAME, for example a quoted literal "*" on Windows.
        if getattr(exc, "winerror", None) == 123:
            return False
        raise
    if deletion.kind in {"file", "cmd-file"}:
        return not directory
    if deletion.kind == "directory" and not directory:
        return os.name == "nt" and path.is_dir()
    if deletion.recursive or not directory:
        return True
    with os.scandir(path) as entries:
        return next(entries, None) is None


def _empties_root(root: Path, covered: set[Path]) -> bool:
    try:
        with os.scandir(root) as entries:
            for entry in entries:
                child = root / entry.name
                if child not in covered and not any(
                    _same_entry(child, path) for path in covered
                ):
                    return False
            return bool(covered)
    except (FileNotFoundError, NotADirectoryError):
        return False


def _removed_child(
    root: Path, entry: Path, deletion: _Delete, named_parents: tuple[Path, ...]
) -> Path | None:
    if _same_entry(entry.parent, root):
        return entry if _removes_child(entry, deletion) else None
    if not any(_same_entry(parent.parent, root) for parent in named_parents):
        return None
    if not _removes_child(entry, deletion):
        return None
    for parent in named_parents:
        if not _same_entry(entry.parent, parent) or not _empties_root(parent, {entry}):
            return None
        if _same_entry(parent.parent, root):
            return parent
        entry = parent
    return None


def _check_deletions(
    deletes: list[_Delete | None],
    directory: Path | None,
    dialect: ShellDialect,
    protected_roots: tuple[Path, ...],
    roots: dict[Path, set[Path]],
    covered: dict[Path, set[Path]],
) -> WorkspaceDeletionRule | None:
    for deletion in deletes:
        if (
            deletion is None
            or (path := _path(deletion.path, directory, dialect)) is None
        ):
            continue
        if not roots:
            for bound in protected_roots:
                root = bound.resolve()
                roots.setdefault(root, {root}).add(_entry_path(bound))
        entry = _entry_path(path)
        children = deletion.children or (
            deletion.kind == "cmd-file"
            and (
                path.is_dir()
                or any(
                    _same_entry(entry, alias)
                    for names in roots.values()
                    for alias in names
                )
            )
        )
        if children:
            entry = path.resolve()
        # rmdir -p names the parents in its operand, stopping before the
        # implicit cwd of a relative path. It does not prune every
        # ancestor of the resolved path.
        named_parents = (
            tuple(
                _entry_path(parent_path)
                for parent in Path(deletion.path).parents
                if parent.name
                and (parent_path := _path(str(parent), directory, dialect)) is not None
            )
            if deletion.prune_parents
            else ()
        )
        for root, aliases in roots.items():
            if any(_same_entry(entry, alias) for alias in aliases):
                return (
                    "workspace-folder-wipe" if children else "workspace-folder-delete"
                )
            contains_root = any(
                _same_entry(entry, parent)
                for alias in aliases
                for parent in alias.parents
            )
            if contains_root and deletion.recursive:
                return (
                    "workspace-folder-wipe"
                    if deletion.kind == "cmd-file"
                    else "workspace-folder-delete"
                )
            if children and deletion.kind != "file" and deletion.kind != "cmd-file":
                if any(_same_entry(entry, alias.parent) for alias in aliases):
                    return "workspace-folder-delete"
            if children:
                continue
            if any(
                _same_entry(parent, alias)
                for parent in named_parents
                for alias in aliases
            ):
                return "workspace-folder-delete"
            removed = _removed_child(root, entry, deletion, named_parents)
            if removed is not None:
                covered.setdefault(root, set()).add(removed)
    return None


def workspace_deletion_violation(
    command: str,
    *,
    dialect: ShellDialect,
    cwd: Path,
    protected_roots: tuple[Path, ...],
) -> WorkspaceDeletionRule | None:
    """Return the rule hit by a recognizable deletion, or None.

    Args:
        command: Raw shell text. It is inspected and never executed here.
        dialect: Dialect of the shell selected for launch.
        cwd: Validated starting directory.
        protected_roots: Trusted local workspace binding locators.

    Raises:
        OSError: A filesystem inspection failed.
    """
    if not protected_roots:
        return None
    segments = _segments(command, dialect)
    roots: dict[Path, set[Path]] = {}
    covered: dict[Path, set[Path]] = {}
    for i, (segment, directories) in enumerate(
        zip(segments, _directory_options(segments, dialect, cwd), strict=True)
    ):
        if not directories:
            continue
        words, separator = segment.words, segment.separator
        name = _name(words, dialect)
        args = words[1:]
        if dialect == "powershell":
            piped = segments[i + 1].words if separator == "|" else None
            deletes = _powershell_deletes(name, args, piped)
        elif dialect == "cmd":
            deletes = _cmd_deletes(name, args)
        else:
            deletes = _bash_deletes(name, args)

        for directory in directories:
            # A compound working-directory choice does not share child
            # coverage across its alternatives.
            affected = covered if len(directories) == 1 else {}
            violation = _check_deletions(
                deletes, directory, dialect, protected_roots, roots, affected
            )
            if violation is not None:
                return violation
            if affected is not covered and any(
                _empties_root(root, entries) for root, entries in affected.items()
            ):
                return "workspace-folder-wipe"

        builtin = _literal_builtin(segment, dialect)
        literal_output = (
            builtin == "echo"
            or (dialect == "powershell" and builtin == "write-output")
            or (
                dialect == "bash"
                and builtin == "printf"
                and args
                and not args[0].text.startswith("-")
            )
        )
        preserves_children = not segment.redirected and (
            _is_location(name, dialect) or literal_output
        )
        if (
            len(directories) != 1
            or (not deletes and not preserves_children)
            or separator in {"", "||", "|"}
            or (separator == "&" and dialect != "cmd")
        ):
            if any(_empties_root(root, entries) for root, entries in covered.items()):
                return "workspace-folder-wipe"
            covered.clear()
    return None
