"""Legacy Prompt CSV/Markdown export, keeping its defining DB dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .Prompts_DB import PromptsDatabase


def export_prompts_formatted(
    _source: Any,
    db_instance: PromptsDatabase,
    export_format: str = "csv",  # 'csv' or 'markdown'
    filter_keywords: list[str] | None = None,
    include_system: bool = True,
    include_user: bool = True,
    include_details: bool = True,
    include_author: bool = True,
    include_associated_keywords: bool = True,  # Renamed for clarity
    markdown_template_name: str | None = "Basic Template",  # Name of template
) -> tuple[str, str]:
    import csv
    import os
    import tempfile
    import zipfile  # For markdown if multiple files
    from datetime import datetime

    if not isinstance(db_instance, _source.PromptsDatabase):
        raise TypeError("db_instance must be a PromptsDatabase object.")

    _source.logging.debug(
        f"export_prompts_formatted (format: {export_format}) from DB: {db_instance.db_path_str}"
    )

    # --- Fetch Prompts Data ---
    # Build base query parts
    select_fields = ["p.id", "p.name", "p.uuid"]  # Always include id, name, uuid
    if include_author:
        select_fields.append("p.author")
    if include_details:
        select_fields.append("p.details")
    if include_system:
        select_fields.append("p.system_prompt")
    if include_user:
        select_fields.append("p.user_prompt")

    query_sql = f"SELECT DISTINCT {', '.join(select_fields)} FROM Prompts p"
    query_params = []

    # Keyword filtering
    if filter_keywords and len(filter_keywords) > 0:
        normalized_filter_keywords = [
            db_instance._normalize_keyword(k)
            for k in filter_keywords
            if k and k.strip()
        ]
        if normalized_filter_keywords:
            placeholders = ",".join(["?"] * len(normalized_filter_keywords))
            query_sql += f"""
                JOIN PromptKeywordLinks pkl ON p.id = pkl.prompt_id
                JOIN PromptKeywordsTable pkw ON pkl.keyword_id = pkw.id
                WHERE p.deleted = 0 AND pkw.deleted = 0 AND pkw.keyword IN ({placeholders})
            """
            query_params.extend(normalized_filter_keywords)
        else:  # No valid filter keywords, so just filter active prompts
            query_sql += " WHERE p.deleted = 0"
    else:  # No keyword filter, just active prompts
        query_sql += " WHERE p.deleted = 0"

    query_sql += " ORDER BY p.name COLLATE NOCASE"

    try:
        cursor = db_instance.execute_query(query_sql, tuple(query_params))
        prompts_data = [dict(row) for row in cursor.fetchall()]

        if not prompts_data:
            return "No prompts found matching the criteria for export.", "None"

        # Fetch associated keywords for each prompt if needed
        if include_associated_keywords:
            for prompt_dict in prompts_data:
                prompt_dict["keywords_list"] = db_instance.fetch_keywords_for_prompt(
                    prompt_dict["id"], include_deleted=False
                )

        # --- Perform Export ---
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_path = "None"

        if export_format == "csv":
            # TASK-32806.7: unique, O_EXCL, 0600 -- not a guessable
            # world-readable name in the shared temp dir.
            _fd, temp_csv_file = tempfile.mkstemp(
                prefix="prompts_export_", suffix=".csv"
            )
            os.close(_fd)
            header_row = ["Name", "UUID"]  # Start with common fields
            if include_author:
                header_row.append("Author")
            if include_details:
                header_row.append("Details")
            if include_system:
                header_row.append("System Prompt")
            if include_user:
                header_row.append("User Prompt")
            if include_associated_keywords:
                header_row.append("Keywords")

            with open(temp_csv_file, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header_row)
                for p_data in prompts_data:
                    row_to_write = [p_data["name"], p_data["uuid"]]
                    if include_author:
                        row_to_write.append(p_data.get("author", ""))
                    if include_details:
                        row_to_write.append(p_data.get("details", ""))
                    if include_system:
                        row_to_write.append(p_data.get("system_prompt", ""))
                    if include_user:
                        row_to_write.append(p_data.get("user_prompt", ""))
                    if include_associated_keywords:
                        row_to_write.append(", ".join(p_data.get("keywords_list", [])))
                    writer.writerow(row_to_write)
            output_file_path = temp_csv_file
            status_msg = f"Successfully exported {len(prompts_data)} prompts to CSV."

        elif export_format == "markdown":
            # TASK-32806.7: the zip previously went to a guessable world-readable
            # temp name; mkstemp makes it unique and 0600.
            _zip_fd, zip_file_path = tempfile.mkstemp(
                prefix="prompts_export_markdown_", suffix=".zip"
            )
            os.close(_zip_fd)

            templates = {
                "Basic Template": """# {name} ({uuid})
{author_section}
{details_section}
{system_section}
{user_section}
{keywords_section}
""",
                "Detailed Template": """# {name}
**UUID**: {uuid}

## Author
{author_section}

## Description
{details_section}

## System Prompt
```
{system_prompt_content}
```

## User Prompt
```
{user_prompt_content}
```

## Keywords
{keywords_section}
""",
            }
            chosen_template_str = templates.get(
                markdown_template_name, templates["Basic Template"]
            )

            with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                used_arcname_keys: set[str] = set()
                for p_data in prompts_data:
                    author_sec = (
                        f"**Author**: {p_data['author']}"
                        if include_author and p_data.get("author")
                        else ""
                    )
                    details_sec = (
                        f"**Details**: {p_data['details']}"
                        if include_details and p_data.get("details")
                        else ""
                    )
                    system_sec = (
                        f"**System Prompt**:\n```\n{p_data['system_prompt']}\n```"
                        if include_system and p_data.get("system_prompt")
                        else ""
                    )
                    user_sec = (
                        f"**User Prompt**:\n```\n{p_data['user_prompt']}\n```"
                        if include_user and p_data.get("user_prompt")
                        else ""
                    )
                    keywords_sec = (
                        f"**Keywords**: {', '.join(p_data['keywords_list'])}"
                        if include_associated_keywords and p_data.get("keywords_list")
                        else ""
                    )

                    md_content = chosen_template_str.format(
                        name=p_data["name"],
                        uuid=p_data["uuid"],
                        author_section=author_sec,
                        details_section=details_sec,
                        system_section=system_sec,  # For Basic Template direct injection
                        system_prompt_content=p_data.get(
                            "system_prompt", ""
                        ),  # For Detailed Template
                        user_section=user_sec,  # For Basic Template direct injection
                        user_prompt_content=p_data.get(
                            "user_prompt", ""
                        ),  # For Detailed Template
                        keywords_section=keywords_sec,
                    ).strip()  # Clean up extra newlines if sections are empty

                    # Unique, collision-safe filename; writestr straight into
                    # the archive (no leaking on-disk staging copy). See
                    # _markdown_export_arcname for the disambiguation contract.
                    arcname = _source._markdown_export_arcname(
                        p_data["name"], p_data["id"], used_arcname_keys
                    )
                    zipf.writestr(arcname, md_content)

            output_file_path = zip_file_path
            status_msg = f"Successfully exported {len(prompts_data)} prompts to Markdown in a ZIP file."
        else:
            raise ValueError(
                f"Unsupported export_format: {export_format}. Must be 'csv' or 'markdown'."
            )

        _source.logging.info(status_msg)
        return status_msg, output_file_path

    except (_source.DatabaseError, _source.sqlite3.Error, ValueError) as e:
        error_msg = f"Error exporting prompts: {e}"
        _source.logging.opt(exception=True).error(error_msg)
        return error_msg, "None"
    except Exception as e:  # Catch any other unexpected error
        error_msg = f"Unexpected error exporting prompts: {e}"
        _source.logging.opt(exception=True).error(error_msg)
        return error_msg, "None"
