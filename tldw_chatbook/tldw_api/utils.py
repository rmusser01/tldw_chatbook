# tldw_chatbook/tldw_api/utils.py
#
#
# Imports
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, IO, Tuple
import mimetypes
#
# 3rd-party Libraries
from pydantic import BaseModel
#
#######################################################################################################################
#
# Functions:

def model_to_form_data(model_instance: BaseModel) -> Dict[str, Any]:
    """
    Converts a Pydantic model instance into a dictionary suitable for
    FastAPI Form data submission (handles None, bool, list to string).
    """
    form_data = {}
    for field_name, field_value in model_instance.model_dump(exclude_none=True).items():
        if field_name == "keywords" and isinstance(field_value, list):
            form_data[field_name] = ",".join(field_value) # Server expects comma-separated string for "keywords"
        elif isinstance(field_value, bool):
            form_data[field_name] = str(field_value).lower() # FastAPI Form booleans
        elif isinstance(field_value, list):
            # For lists other than keywords (e.g., urls), httpx handles them correctly
            # if the server endpoint expects multiple values for the same form field name.
            form_data[field_name] = field_value
        elif field_value is not None:
            form_data[field_name] = str(field_value) # Most other fields as strings
    return form_data

def prepare_files_for_httpx(
    file_paths: Optional[List[str]],
    upload_field_name: str = "files"
) -> Optional[List[Tuple[str, Tuple[str, IO[bytes], Optional[str]]]]]:
    """
    Prepares a list of file paths for httpx multipart upload.

    Args:
        file_paths: A list of string paths to local files.
        upload_field_name: The name of the field for file uploads (FastAPI often uses 'files').

    Returns:
        A list of tuples formatted for httpx's `files` argument, or None.
        Example: [('files', ('filename.mp4', <file_obj>, 'video/mp4')), ...]
        
    Note:
        Callers are responsible for calling cleanup_file_objects() after using
        the returned file objects to prevent resource leaks.
    """
    if not file_paths:
        return None

    httpx_files_list = []
    for file_path_str in file_paths:
        file_obj = None
        try:
            file_path_obj = Path(file_path_str)
            if not file_path_obj.is_file():
                logging.warning(f"Warning: File not found or not a file: {file_path_str}")
                continue

            file_obj = open(file_path_obj, "rb")

            mime_type, _ = mimetypes.guess_type(file_path_obj.name)

            if mime_type is None:
                mime_type = 'application/octet-stream'
                logging.warning(f"Could not guess MIME type for {file_path_obj.name}. Defaulting to {mime_type}.")

            httpx_files_list.append(
                (upload_field_name, (file_path_obj.name, file_obj, mime_type))
            )
        except Exception as e:
            logging.error(f"Error preparing file {file_path_str} for upload: {e}")
            # Close the file if it was opened but failed to be added to the list
            if file_obj:
                try:
                    file_obj.close()
                except Exception:
                    pass  # Ignore close errors
            # Continue to next file instead of breaking the loop
            continue
    return httpx_files_list if httpx_files_list else None


def cleanup_file_objects(httpx_files: Optional[List[Tuple[str, Tuple[str, IO[bytes], Optional[str]]]]]) -> None:
    """
    Closes all file objects in an httpx files list to prevent resource leaks.
    
    Args:
        httpx_files: The list returned by prepare_files_for_httpx()
    """
    if not httpx_files:
        return
        
    for field_name, (filename, file_obj, mime_type) in httpx_files:
        try:
            if hasattr(file_obj, 'close'):
                file_obj.close()
        except Exception as e:
            logging.warning(f"Failed to close file object for {filename}: {e}")

#
# End of utils.py
#######################################################################################################################
