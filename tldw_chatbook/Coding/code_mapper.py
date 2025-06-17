# tldw_chatbook/Coding/code_mapper.py
# Description: This module provides a context manager for handling code files in a TUI application.
#
# Imports
import os
import time
from pathlib import Path
from collections import defaultdict
#
# Third-Party Imports
#
# Local Imports
from tldw_chatbook.Third_Party.aider.repomap import RepoMap
#
########################################################################################################################
#
# You might need to provide stubs or mock objects for RepoMap's dependencies
# if you're not running the full Aider environment, e.g., for `io` and `main_model`.
class SimpleIO:
    def tool_output(self, message):
        print(f"[INFO] {message}")

    def tool_warning(self, message):
        print(f"[WARNING] {message}")

    def tool_error(self, message):
        print(f"[ERROR] {message}")

    def read_text(self, fpath):
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception as e:
            self.tool_error(f"Error reading {fpath}: {e}")
            return None


class MockModel:
    def token_count(self, text):
        # A simple approximation for token counting.
        # For more accuracy, integrate with a real tokenizer (e.g., tiktoken).
        return len(text.split())


class CodeContextManager:
    def __init__(self, repo_root, aider_map_tokens=1024, verbose=False):
        self.repo_root = os.path.abspath(repo_root)
        self.verbose = verbose

        # --- Initialize Aider's RepoMap ---
        # You'll need to provide implementations or stubs for `io` and `main_model`
        # if they are strictly required by the parts of RepoMap you use.
        self.io = SimpleIO()  # Replace with your TUI's IO if it has one
        self.main_model = MockModel()  # Replace with a proper model tokenizer

        self.aider_repo_map = RepoMap(
            map_tokens=aider_map_tokens,
            root=self.repo_root,
            main_model=self.main_model,
            io=self.io,
            verbose=self.verbose,
            # You might want to configure other RepoMap params as needed
        )
        # To store data for "indexing for display"
        self.file_index = {}  # {rel_fpath: {"abs_fpath": str, "tags": list[Tag], "mtime": float, "error": str}}
        self.last_index_time = 0

    # --- Goal 1: Indexing for Display and Review ---
    def get_file_list_for_display(self, force_rescan=False):
        """
        Scans the repository (or uses cached data) to get a list of all files
        and their top-level symbols/tags for display in a TUI.

        Args:
            force_rescan (bool): If True, forces a re-scan of all files, ignoring mtime checks.

        Returns:
            dict: {rel_fpath: {"abs_fpath": str, "tags_summary": list[str], "error": str or None}}
                  tags_summary might be like ["class MyClass", "def my_func"]
        """
        print("Building file index for display...")
        current_scan_time = time.time()
        updated_files = 0
        processed_files = 0

        # Discover all potential source files in the repository
        # You might want to use a more sophisticated discovery mechanism
        # like git ls-files, or respect .gitignore. Aider's RepoMap
        # often gets file lists from git, so it might not have its own
        # comprehensive discover_files respecting .gitignore.
        # For now, let's use a simple walk.
        all_repo_files = []
        for root, _, files in os.walk(self.repo_root):
            if ".git" in root.split(os.sep):  # Basic .git ignore
                continue
            for file in files:
                abs_fpath = os.path.join(root, file)
                # Filter out some common non-code files (can be improved)
                if not any(abs_fpath.endswith(ext) for ext in
                           [".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".hpp", ".rs", ".go", ".md"]):
                    if self.aider_repo_map.get_rel_fname(abs_fpath).startswith('.'):  # hidden files
                        continue
                    # Check if language can be determined by Aider, crude filter for now
                    if not self.aider_repo_map.filename_to_lang(abs_fpath):
                        continue

                all_repo_files.append(abs_fpath)

        display_index = {}

        for abs_fpath in all_repo_files:
            processed_files += 1
            if processed_files % 100 == 0 and self.verbose:
                self.io.tool_output(f"Scanned {processed_files}/{len(all_repo_files)} files for index...")

            rel_fpath = self.aider_repo_map.get_rel_fname(abs_fpath)
            try:
                current_mtime = os.path.getmtime(abs_fpath)
            except FileNotFoundError:
                if rel_fpath in self.file_index:
                    del self.file_index[rel_fpath]  # Remove if deleted
                continue

            # Check cache
            if not force_rescan and rel_fpath in self.file_index and self.file_index[rel_fpath][
                "mtime"] == current_mtime:
                # Use cached tags if mtime hasn't changed
                tags = self.file_index[rel_fpath]["tags"]
                error_msg = self.file_index[rel_fpath]["error"]
            else:
                # Get fresh tags using Aider's method
                # get_tags returns a list of Tag namedtuples
                try:
                    tags = list(self.aider_repo_map.get_tags(abs_fpath, rel_fpath))
                    error_msg = None
                    self.file_index[rel_fpath] = {
                        "abs_fpath": abs_fpath,
                        "tags": tags,
                        "mtime": current_mtime,
                        "error": None
                    }
                    updated_files += 1
                except Exception as e:
                    tags = []
                    error_msg = f"Error processing {rel_fpath}: {e}"
                    self.file_index[rel_fpath] = {
                        "abs_fpath": abs_fpath,
                        "tags": [],
                        "mtime": current_mtime,
                        "error": str(e)
                    }
                    if self.verbose: self.io.tool_warning(error_msg)

            # Prepare a summary for display (e.g., class and function definitions)
            tags_summary = []
            if tags:
                for tag_obj in tags:
                    if tag_obj.kind == "def":  # We are interested in definitions for tree view
                        # Tag(rel_fname, fname, line, name, kind)
                        tags_summary.append(f"{tag_obj.kind}: {tag_obj.name} (L{tag_obj.line + 1})")

            display_index[rel_fpath] = {
                "abs_fpath": abs_fpath,
                "tags_summary": sorted(list(set(tags_summary))),  # Unique, sorted
                "error": error_msg
            }

        self.last_index_time = current_scan_time
        self.io.tool_output(
            f"File index refreshed. {updated_files} files updated/added. Total {len(display_index)} files.")
        return display_index

    # --- Goal 2: Generating Context in Aider's Way ---
    def get_aider_context(self, chat_files, other_files, mentioned_fnames=None, mentioned_idents=None):
        """
        Generates a context string using Aider's RepoMap logic.

        Args:
            chat_files (list[str]): List of absolute file paths currently in "chat" or focus.
            other_files (list[str]): List of other absolute file paths in the repo to consider.
            mentioned_fnames (set[str], optional): Set of relative filenames explicitly mentioned.
            mentioned_idents (set[str], optional): Set of identifiers explicitly mentioned.

        Returns:
            str: The context string generated by Aider's RepoMap, or None.
        """
        if self.verbose:
            self.io.tool_output(f"Generating Aider-style context for {len(chat_files)} chat files"
                                f" and {len(other_files)} other files.")

        # Aider's RepoMap methods generally expect absolute paths for chat_fnames and other_fnames
        # and it handles the rel_path conversion internally.

        # Ensure RepoMap's internal cache is primed or updated if necessary.
        # RepoMap.get_ranked_tags_map handles its own caching and refreshing logic
        # based on its `refresh` setting. We might need to call `get_tags` for all files
        # beforehand if RepoMap relies on that being up-to-date from an external call,
        # but typically its `get_ranked_tags` will call `get_tags` as needed.
        # For safety, let's ensure tags are reasonably fresh for `other_files` if not done by get_file_list_for_display
        # (This is a bit redundant if get_file_list_for_display was just called, but good for standalone use)
        # for f_path in chat_files + other_files:
        #     rel_f_path = self.aider_repo_map.get_rel_fname(f_path)
        #     _ = self.aider_repo_map.get_tags(f_path, rel_f_path) # Primes cache

        return self.aider_repo_map.get_repo_map(
            chat_files=chat_files,
            other_files=other_files,
            mentioned_fnames=mentioned_fnames,
            mentioned_idents=mentioned_idents,
            # force_refresh=False # Control this based on TUI actions
        )

    # --- Goal 3: Generating Context via Simple Concatenation ---
    def get_simple_concatenated_context(self, selected_abs_fpaths, include_headers=True, max_total_size_mb=None):
        """
        Concatenates the full content of selected files with demarcations.

        Args:
            selected_abs_fpaths (list[str]): List of absolute file paths to concatenate.
            include_headers (bool): Whether to include a header for each file.
            max_total_size_mb (float, optional): Maximum total size of concatenated output in MB.

        Returns:
            str: The concatenated content.
        """
        if self.verbose:
            self.io.tool_output(f"Generating simple concatenated context for {len(selected_abs_fpaths)} files.")

        output_parts = []
        current_size_bytes = 0
        limit_bytes = (max_total_size_mb * 1024 * 1024) if max_total_size_mb else float('inf')
        files_included_count = 0

        for abs_fpath in selected_abs_fpaths:
            rel_fpath = self.aider_repo_map.get_rel_fname(abs_fpath)
            try:
                file_size = os.path.getsize(abs_fpath)
                if current_size_bytes + file_size > limit_bytes and max_total_size_mb is not None:
                    self.io.tool_warning(
                        f"Warning: Reached size limit of {max_total_size_mb}MB. Skipping remaining files.")
                    break

                content = self.io.read_text(abs_fpath)
                if content is None:
                    output_parts.append(f"--- ERROR READING FILE: {rel_fpath} ---\n[Content not available]\n\n")
                    continue

                if include_headers:
                    header = f"--- BEGIN FILE: {rel_fpath} ---\n"
                    # Optionally, add some basic info from our index
                    if rel_fpath in self.file_index and self.file_index[rel_fpath].get("tags"):
                        defs = [
                            tag.name
                            for tag in self.file_index[rel_fpath]["tags"]
                            if tag.kind == "def"
                        ]
                        if defs:
                            header += f"Definitions: {', '.join(defs[:5])}{'...' if len(defs) > 5 else ''}\n"
                    header += "---\n"  # Simple separator
                    output_parts.append(header)

                output_parts.append(content)
                # Ensure a newline after content if it doesn't have one, before the end marker
                if not content.endswith('\n'):
                    output_parts.append("\n")

                if include_headers:
                    output_parts.append(f"--- END FILE: {rel_fpath} ---\n\n")
                else:
                    output_parts.append("\n\n")  # Just add some separation

                current_size_bytes += len(content.encode('utf-8'))  # More accurate size based on content read
                files_included_count += 1

            except FileNotFoundError:
                output_parts.append(f"--- FILE NOT FOUND: {rel_fpath} ---\n\n")
            except Exception as e:
                output_parts.append(f"--- ERROR PROCESSING FILE {rel_fpath}: {e} ---\n\n")

        if self.verbose:
            self.io.tool_output(
                f"Concatenated {files_included_count} files. Total size: {current_size_bytes / (1024 * 1024):.2f} MB")
        return "".join(output_parts)

    # --- Utility related to Aider's RepoMap if needed for TUI ---
    def get_repo_root(self):
        return self.repo_root

    def get_aider_map_tokens(self):
        return self.aider_repo_map.max_map_tokens

    def set_aider_map_tokens(self, tokens):
        self.aider_repo_map.max_map_tokens = tokens
        # Aider's RepoMap also has map_mul_no_files which influences effective tokens
        # when no chat files are present. You might want to expose/control that too.
        # self.aider_repo_map.map_mul_no_files = new_val

    def clear_aider_map_cache(self):
        """ Clears the cache used by Aider's get_ranked_tags_map. """
        self.aider_repo_map.map_cache = {}
        self.aider_repo_map.last_map = None
        self.io.tool_output("Aider RepoMap internal cache cleared.")

#
# End of code_mapper.py
########################################################################################################################
