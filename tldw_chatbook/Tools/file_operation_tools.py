"""
File Operation Tools for LLM function calling.

These tools allow LLMs to perform safe file operations with proper validation.
"""

from pathlib import Path
from typing import Dict, Any
from loguru import logger

from . import Tool
from ..Utils.path_validation import validate_path_multi
from .workspace_file_roots import allowed_file_roots


def _resolve_sandbox_config() -> str:
    """Return the configured sandbox root string (indirection for tests)."""
    from ..config import get_cli_setting, get_user_data_dir

    default_root = str(get_user_data_dir() / "tool_sandbox")
    return get_cli_setting("tools", "file_sandbox_root", default_root) or default_root


def _is_within(candidate: Path, root: Path) -> bool:
    """Return whether ``candidate`` resolves inside ``root``.

    Args:
        candidate: Path to test.
        root: The sandbox root it must stay under.

    Returns:
        True only when the fully-resolved candidate is the root or below it.
    """
    try:
        resolved = candidate.resolve()
        root_resolved = root.resolve()
    except OSError:
        return False
    return resolved == root_resolved or root_resolved in resolved.parents


def _tool_sandbox_root() -> Path:
    """Resolve + create the file-tool sandbox root.

    The file tools confine all reads/writes/listings under this directory.
    Defaults to ``<user data dir>/tool_sandbox``; override with
    ``[tools] file_sandbox_root`` in config.toml.
    """
    root = Path(_resolve_sandbox_config()).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root


class ReadFileTool(Tool):
    """Tool for reading file contents."""

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return "Read the contents of a file. Returns the file content as text."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to read",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
            },
            "required": ["file_path"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Reading arbitrary sandbox files is a disclosure risk."""
        return ("reads",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Read a file's contents.

        Args:
            file_path: Path to the file
            encoding: File encoding (default: utf-8)

        Returns:
            Dictionary with file content or error
        """
        file_path = kwargs.get("file_path")
        if not file_path:
            return {"error": "No file path provided"}

        encoding = kwargs.get("encoding", "utf-8")

        try:
            # Validate the path against the sandbox plus any read-eligible
            # workspace folder roots bound to the run.
            validated_path = validate_path_multi(
                file_path,
                allowed_file_roots(write=False, sandbox_root=_tool_sandbox_root()),
            )
            path = Path(validated_path)

            # Check if file exists
            if not path.exists():
                return {
                    "error": f"File not found: {file_path}",
                    "absolute_path": str(path.absolute()),
                }

            # Check if it's a file
            if not path.is_file():
                return {
                    "error": f"Path is not a file: {file_path}",
                    "path_type": "directory" if path.is_dir() else "other",
                }

            # Read the file
            content = path.read_text(encoding=encoding)

            # Get file info
            stat = path.stat()

            return {
                "file_path": str(path),
                "content": content,
                "size_bytes": stat.st_size,
                "encoding": encoding,
                "lines": len(content.splitlines()),
            }

        except UnicodeDecodeError as e:
            return {
                "file_path": file_path,
                "error": f"Unable to decode file with {encoding} encoding: {e}",
                "suggestion": "Try a different encoding like 'latin-1' or 'cp1252'",
            }
        except PermissionError:
            return {"file_path": file_path, "error": "Permission denied to read file"}
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return {"file_path": file_path, "error": f"Failed to read file: {str(e)}"}


class ListDirectoryTool(Tool):
    """Tool for listing directory contents."""

    @property
    def name(self) -> str:
        return "list_directory"

    @property
    def description(self) -> str:
        return "List the contents of a directory. Returns file names, types, and basic info."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "directory_path": {
                    "type": "string",
                    "description": "The path to the directory to list",
                },
                "include_hidden": {
                    "type": "boolean",
                    "description": "Include hidden files (starting with .)",
                    "default": False,
                },
                "recursive": {
                    "type": "boolean",
                    "description": "List subdirectories recursively",
                    "default": False,
                },
                "max_depth": {
                    "type": "integer",
                    "description": "Maximum depth for recursive listing",
                    "default": 2,
                    "minimum": 1,
                    "maximum": 5,
                },
            },
            "required": ["directory_path"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Enumerating the sandbox discloses its structure."""
        return ("reads",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        List directory contents.

        Args:
            directory_path: Path to the directory
            include_hidden: Include hidden files
            recursive: List recursively
            max_depth: Max depth for recursion

        Returns:
            Dictionary with directory contents or error
        """
        directory_path = kwargs.get("directory_path", ".")
        include_hidden = kwargs.get("include_hidden", False)
        recursive = kwargs.get("recursive", False)
        max_depth = kwargs.get("max_depth", 2)

        try:
            # Validate the path against the sandbox plus any read-eligible
            # workspace folder roots bound to the run.
            sandbox_root = _tool_sandbox_root()
            read_roots = allowed_file_roots(write=False, sandbox_root=sandbox_root)
            validated_path = validate_path_multi(directory_path, read_roots)
            path = Path(validated_path)

            # The recursive-descent symlink guard below must compare against
            # whichever allowed root actually contains ``path`` — not always
            # the sandbox — otherwise a legitimately bound workspace folder
            # would silently refuse to recurse past its top level.
            containment_root = next(
                (root for root in read_roots if _is_within(path, root)),
                sandbox_root,
            )

            # Check if directory exists
            if not path.exists():
                return {
                    "error": f"Directory not found: {directory_path}",
                    "absolute_path": str(path.absolute()),
                }

            # Check if it's a directory
            if not path.is_dir():
                return {
                    "error": f"Path is not a directory: {directory_path}",
                    "path_type": "file" if path.is_file() else "other",
                }

            entries = []

            def list_dir_contents(dir_path: Path, current_depth: int = 0):
                """Recursively list directory contents."""
                if current_depth > max_depth:
                    return

                try:
                    for item in sorted(dir_path.iterdir()):
                        # Skip hidden files if not requested
                        if not include_hidden and item.name.startswith("."):
                            continue

                        # Get item info
                        try:
                            stat = item.stat()
                            entry = {
                                "name": item.name,
                                "path": str(item.relative_to(path)),
                                "type": "directory" if item.is_dir() else "file",
                                "size_bytes": stat.st_size if item.is_file() else None,
                                "depth": current_depth,
                            }
                            entries.append(entry)

                            # Recursively list subdirectories, but NEVER
                            # follow a symlink: a link planted inside an
                            # allowed root would otherwise let the walk
                            # enumerate files outside it, breaking the
                            # containment every other path here relies on.
                            # Belt-and-braces, the resolved child must still
                            # sit under the root that contains this listing.
                            if (
                                recursive
                                and item.is_dir()
                                and not item.is_symlink()
                                and current_depth < max_depth
                                and _is_within(item, containment_root)
                            ):
                                list_dir_contents(item, current_depth + 1)

                        except PermissionError:
                            entries.append(
                                {
                                    "name": item.name,
                                    "path": str(item.relative_to(path)),
                                    "type": "inaccessible",
                                    "error": "Permission denied",
                                    "depth": current_depth,
                                }
                            )
                        except Exception as e:
                            logger.warning(f"Error accessing {item}: {e}")

                except PermissionError:
                    return {"error": f"Permission denied to list directory: {dir_path}"}

            # List the directory
            list_dir_contents(path)

            # Count types
            file_count = sum(1 for e in entries if e.get("type") == "file")
            dir_count = sum(1 for e in entries if e.get("type") == "directory")

            return {
                "directory_path": str(path),
                "total_entries": len(entries),
                "file_count": file_count,
                "directory_count": dir_count,
                "entries": entries[:100],  # Limit to first 100 entries
            }

        except PermissionError:
            return {
                "directory_path": directory_path,
                "error": "Permission denied to access directory",
            }
        except Exception as e:
            logger.error(f"Error listing directory {directory_path}: {e}")
            return {
                "directory_path": directory_path,
                "error": f"Failed to list directory: {str(e)}",
            }


class WriteFileTool(Tool):
    """Tool for writing content to files."""

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return "Write content to a file. Creates the file if it doesn't exist. Use with caution."

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "The path to the file to write",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: 'overwrite' or 'append'",
                    "enum": ["overwrite", "append"],
                    "default": "overwrite",
                },
                "encoding": {
                    "type": "string",
                    "description": "File encoding (default: utf-8)",
                    "default": "utf-8",
                },
                "create_directories": {
                    "type": "boolean",
                    "description": "Create parent directories if they don't exist",
                    "default": False,
                },
            },
            "required": ["file_path", "content"],
        }

    @property
    def risk_tags(self) -> tuple[str, ...]:
        """Creates, overwrites, or appends to files."""
        return ("mutates",)

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """
        Write content to a file.

        Args:
            file_path: Path to the file
            content: Content to write
            mode: Write mode (overwrite or append)
            encoding: File encoding
            create_directories: Create parent dirs if needed

        Returns:
            Dictionary with success status or error
        """
        file_path = kwargs.get("file_path")
        content = kwargs.get("content")

        if not file_path:
            return {"error": "No file path provided"}
        if content is None:
            return {"error": "No content provided"}

        mode = kwargs.get("mode", "overwrite")
        encoding = kwargs.get("encoding", "utf-8")
        create_directories = kwargs.get("create_directories", False)

        try:
            # Validate the path against the sandbox plus any write-eligible
            # (rw) workspace folder roots bound to the run.
            validated_path = validate_path_multi(
                file_path,
                allowed_file_roots(write=True, sandbox_root=_tool_sandbox_root()),
            )
            path = Path(validated_path)

            # Check if we're overwriting an existing file
            file_exists = path.exists()

            # Create parent directories if requested
            if create_directories and not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created directories: {path.parent}")

            # Check if parent directory exists
            if not path.parent.exists():
                return {
                    "error": f"Parent directory does not exist: {path.parent}",
                    "suggestion": "Set create_directories=true to create it",
                }

            # Write the file
            if mode == "append" and file_exists:
                # Append mode
                with open(path, "a", encoding=encoding) as f:
                    f.write(content)
                action = "appended to"
            else:
                # Overwrite mode
                with open(path, "w", encoding=encoding) as f:
                    f.write(content)
                action = "created" if not file_exists else "overwritten"

            # Get file info after writing
            stat = path.stat()

            return {
                "file_path": str(path),
                "action": action,
                "size_bytes": stat.st_size,
                "encoding": encoding,
                "lines_written": len(content.splitlines()),
            }

        except PermissionError:
            return {"file_path": file_path, "error": "Permission denied to write file"}
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return {"file_path": file_path, "error": f"Failed to write file: {str(e)}"}
