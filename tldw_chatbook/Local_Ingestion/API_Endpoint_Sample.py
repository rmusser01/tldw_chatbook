# =============================================================================
# Video Processing Endpoint
# =============================================================================
@router.post(
    "/process-videos",
    # status_code=status.HTTP_200_OK, # Status determined dynamically
    summary="Transcribe / chunk / analyse videos and return the full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
)
async def process_videos_endpoint(
    # --- Dependencies ---
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # Add check here for granular permissions if needed
    # 2. DB Dependency
    db: MediaDatabase = Depends(get_media_db_for_user),
    # 3. Form Data Dependency: Parses form fields into the Pydantic model.
    form_data: ProcessVideosForm = Depends(get_process_videos_form),
    # 4. File Uploads
    files: Optional[List[UploadFile]] = File(None, description="Video file uploads"),
    # user_info: dict = Depends(verify_token), # Optional Auth
):
    """
    **Process Videos Endpoint (Fixed)**

    Transcribes, chunks, and analyses videos from URLs or uploaded files.
    Returns processing artifacts without saving to the database.
    Corrected the run_in_executor call and input_ref mapping.
    """
    # --- Validation and Logging ---
    logger.info("Request received for /process-videos. Form data validated via dependency.")

    if form_data.urls and form_data.urls == ['']:
        logger.info("Received urls=[''], treating as no URLs provided for video processing.")
        form_data.urls = None # Or []

    _validate_inputs("video", form_data.urls, files) # Keep basic input check

    # --- Setup ---
    loop = asyncio.get_running_loop()
    batch_result: Dict[str, Any] = {"processed_count": 0, "errors_count": 0, "errors": [], "results": [], "confabulation_results": None}
    file_handling_errors_structured: List[Dict[str, Any]] = []
    # --- Map to store temporary path -> original filename ---
    temp_path_to_original_name: Dict[str, str] = {}

    # --- Use TempDirManager for reliable cleanup ---
    with TempDirManager(cleanup=True, prefix="process_video_") as temp_dir:
        logger.info(f"Using temporary directory for /process-videos: {temp_dir}")

        # --- Save Uploads ---
        saved_files_info, file_handling_errors_raw = await _save_uploaded_files(files or [], temp_dir, validator=file_validator_instance,)

        # --- Populate the temp path to original name map ---
        for sf in saved_files_info:
            if sf.get("path") and sf.get("original_filename"):
                # Convert Path object to string for consistent dictionary keys
                temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
            else:
                logger.warning(f"Missing path or original_filename in saved_files_info item: {sf}")


        # --- Process File Handling Errors ---
        if file_handling_errors_raw:
            batch_result["errors_count"] += len(file_handling_errors_raw)
            batch_result["errors"].extend([err.get("error", "Unknown file save error") for err in file_handling_errors_raw])
            # Adapt raw file errors to the MediaItemProcessResponse structure
            for err in file_handling_errors_raw:
                 # *** Use original filename for input_ref here ***
                 original_filename = err.get("input", "Unknown Filename") # Assume 'input' holds original name from _save_uploaded_files error
                 file_handling_errors_structured.append({
                     "status": "Error",
                     "input_ref": err.get("input", "Unknown Filename"),
                     "processing_source": "N/A - File Save Failed",
                     "media_type": "video",
                     "metadata": {}, "content": "", "segments": None, "chunks": None,
                     "analysis": None, "analysis_details": {},
                     "error": err.get("error", "Failed to save uploaded file."), "warnings": None,
                     "db_id": None, "db_message": "Processing only endpoint.", "message": None,
                 })
            batch_result["results"].extend(file_handling_errors_structured) # Add structured errors

        # --- Prepare Inputs for Processing ---
        url_list = form_data.urls or []
        # Get the temporary paths (as strings) from saved_files_info
        uploaded_paths = [str(sf["path"]) for sf in saved_files_info if sf.get("path")]
        all_inputs_to_process = url_list + uploaded_paths

        # Check if there's anything left to process
        if not all_inputs_to_process:
            if file_handling_errors_raw: # Only file errors occurred
                logger.warning("No valid video sources to process after file saving errors.")
                # Return 207 with the structured file errors
                return JSONResponse(status_code=status.HTTP_207_MULTI_STATUS, content=batch_result)
            else: # No inputs provided at all
                logger.warning("No video sources provided.")
                raise HTTPException(status.HTTP_400_BAD_REQUEST, "No valid video sources supplied.")

        # --- Call process_videos ---
        video_args = {
            "inputs": all_inputs_to_process,
            # Use form_data directly
            "start_time": form_data.start_time,
            "end_time": form_data.end_time,
            "diarize": form_data.diarize,
            "vad_use": form_data.vad_use,
            "transcription_model": form_data.transcription_model,
            "transcription_language": form_data.transcription_language, # Add language if process_videos needs it
            "perform_analysis": form_data.perform_analysis,
            "custom_prompt": form_data.custom_prompt,
            "system_prompt": form_data.system_prompt,
            "perform_chunking": form_data.perform_chunking,
            "chunk_method": form_data.chunk_method,
            "max_chunk_size": form_data.chunk_size,
            "chunk_overlap": form_data.chunk_overlap,
            "use_adaptive_chunking": form_data.use_adaptive_chunking,
            "use_multi_level_chunking": form_data.use_multi_level_chunking,
            "chunk_language": form_data.chunk_language,
            "summarize_recursively": form_data.summarize_recursively,
            "api_name": form_data.api_name if form_data.perform_analysis else None,
            "api_key": form_data.api_key,
            "use_cookies": form_data.use_cookies,
            "cookies": form_data.cookies,
            "timestamp_option": form_data.timestamp_option,
            "perform_confabulation_check": form_data.perform_confabulation_check_of_analysis,
            "temp_dir": str(temp_dir),  # Pass the managed temporary directory path
            # 'keep_original' might be relevant if library needs it, default is False
            # 'perform_diarization' seems redundant if 'diarize' is passed, check library usage
            # If perform_diarization is truly needed separately:
            # "perform_diarization": form_data.diarize, # Or map if different logic
        }

        try:
            logger.debug(f"Calling process_videos for /process-videos endpoint with {len(all_inputs_to_process)} inputs.")
            batch_func = functools.partial(process_videos, **video_args)

            processing_output = await loop.run_in_executor(None, batch_func)

            # Debug logging
            try:
                print(f"!!! DEBUG PRINT !!! My debug message: {json.dumps(processing_output, indent=2, default=str)}")
            except Exception as log_err:
                print(f"!!! DEBUG PRINT !!! My debug message: {log_err}")

            # --- Combine Processing Results ---
            # Reset results list if we only had file errors before, or append otherwise
            # Clear the specific counters before processing the library output
            batch_result["processed_count"] = 0
            batch_result["errors_count"] = 0
            batch_result["errors"] = []

            # Start with any structured file errors we recorded earlier
            final_results_list = list(file_handling_errors_structured)
            final_errors_list = [err.get("error", "File handling error") for err in file_handling_errors_structured]

            if isinstance(processing_output, dict):
                # Add results from the library processing
                processed_results_from_lib = processing_output.get("results", [])
                for res in processed_results_from_lib:
                    # *** Map input_ref back to original filename if applicable ***
                    current_input_ref = res.get("input_ref") # This is likely the temp path or URL
                    # If the current_input_ref is a key in our map, use the original name
                    # Otherwise, keep the current_input_ref (it's likely a URL)
                    res["input_ref"] = temp_path_to_original_name.get(current_input_ref, current_input_ref)

                    # Add endpoint-specific fields
                    res["db_id"] = None
                    res["db_message"] = "Processing only endpoint."
                    final_results_list.append(res) # Add the modified result

                # Add specific errors reported by the library
                final_errors_list.extend(processing_output.get("errors", []))

                # Handle confabulation results if present
                if "confabulation_results" in processing_output:
                    batch_result["confabulation_results"] = processing_output["confabulation_results"]

            else:
                # Handle unexpected output from process_videos library function
                logger.error(f"process_videos function returned unexpected type: {type(processing_output)}")
                general_error_msg = "Video processing library returned invalid data."
                final_errors_list.append(general_error_msg)
                # Create error entries for all inputs attempted in *this specific* processing call
                for input_src in all_inputs_to_process:
                    # *** Use original name for error input_ref if possible ***
                    original_ref_for_error = temp_path_to_original_name.get(input_src, input_src)
                    final_results_list.append({
                        "status": "Error",
                        "input_ref": original_ref_for_error, # Use original name/URL
                        "processing_source": input_src, # Show what was actually processed (temp path/URL)
                        "media_type": "video", "metadata": {}, "content": "", "segments": None,
                        "chunks": None, "analysis": None, "analysis_details": {},
                        "error": general_error_msg, "warnings": None, "db_id": None,
                        "db_message": "Processing only endpoint.", "message": None
                    })

            # --- Recalculate final counts based on the merged list ---
            batch_result["results"] = final_results_list
            batch_result["processed_count"] = sum(1 for r in final_results_list if r.get("status") == "Success")
            batch_result["errors_count"] = sum(1 for r in final_results_list if r.get("status") == "Error")
            # Remove duplicates from error messages list if desired
            # Make sure errors are strings before adding to set
            unique_errors = set(str(e) for e in final_errors_list if e is not None)
            batch_result["errors"] = list(unique_errors)


        except Exception as exec_err:
            # Catch errors during the library execution call itself
            logger.error(f"Error executing process_videos: {exec_err}", exc_info=True)
            error_msg = f"Error during video processing execution: {type(exec_err).__name__}"

            # Start with existing file errors
            final_results_list = list(file_handling_errors_structured)
            final_errors_list = [err.get("error", "File handling error") for err in file_handling_errors_structured]
            final_errors_list.append(error_msg)  # Add the execution error

            # Create error entries for all inputs attempted in this batch
            for input_src in all_inputs_to_process:
                 # *** Use original name for error input_ref if possible ***
                 original_ref_for_error = temp_path_to_original_name.get(input_src, input_src)
                 final_results_list.append({
                    "status": "Error",
                    "input_ref": original_ref_for_error, # Use original name/URL
                    "processing_source": input_src, # Show what was actually processed (temp path/URL)
                    "media_type": "video", "metadata": {}, "content": "", "segments": None,
                    "chunks": None, "analysis": None, "analysis_details": {},
                    "error": error_msg, "warnings": None, "db_id": None,
                    "db_message": "Processing only endpoint.", "message": None
                })

            # --- Update batch_result with merged errors ---
            batch_result["results"] = final_results_list
            batch_result["processed_count"] = 0 # Assume all failed if execution failed
            batch_result["errors_count"] = len(final_results_list) # Count all items as errors now
            unique_errors = set(str(e) for e in final_errors_list if e is not None)
            batch_result["errors"] = list(unique_errors)

        # --- Determine Final Status Code & Return ---
        # Base the status code *solely* on the final calculated errors_count
        final_error_count = batch_result.get("errors_count", 0)
        # Check if there are only warnings and no errors
        final_success_count = batch_result.get("processed_count", 0)
        total_items = len(batch_result.get("results", []))
        has_warnings = any(r.get("status") == "Warning" for r in batch_result.get("results", []))

        if total_items == 0: # Should not happen if validation passed, but handle defensively
            final_status_code = status.HTTP_400_BAD_REQUEST # Or 500?
            logger.error("No results generated despite processing attempt.")
        elif final_error_count == 0:
             final_status_code = status.HTTP_200_OK
        elif final_error_count == total_items:
             final_status_code = status.HTTP_207_MULTI_STATUS # All errors, could also be 4xx/5xx depending on cause
        else: # Mix of success/warnings/errors
             final_status_code = status.HTTP_207_MULTI_STATUS

        log_level = "INFO" if final_status_code == status.HTTP_200_OK else "WARNING"
        logger.log(log_level,
                   f"/process-videos request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Errors: {final_error_count}")

        # --- TEMPORARY DEBUG ---
        try:
            logger.debug("Final batch_result before JSONResponse:")
            # Log only a subset if the full result is too large
            logged_result = batch_result.copy()
            if len(logged_result.get('results', [])) > 5: # Log details for first 5 results only
                 logged_result['results'] = logged_result['results'][:5] + [{"message": "... remaining results truncated for logging ..."}]
            logger.debug(json.dumps(logged_result, indent=2, default=str)) # Use default=str for non-serializable items

            success_item_debug = next((r for r in batch_result.get("results", []) if r.get("status") == "Success"), None)
            if success_item_debug:
                logger.debug(f"Value of input_ref for success item before return: {success_item_debug.get('input_ref')}")
            else:
                logger.debug("No success item found in final results before return.")
        except Exception as debug_err:
            logger.error(f"Error during debug logging: {debug_err}")
        # --- END TEMPORARY DEBUG ---

        return JSONResponse(status_code=final_status_code, content=batch_result)

#
# End of Video Processing
####################################################################################


######################## Audio Processing Endpoint ###################################
# Endpoints:
#   /process-audio

# =============================================================================
# Dependency Function for Audio Form Processing
# =============================================================================
def get_process_audios_form(
    # Replicate relevant Form(...) definitions for audio
    urls: Optional[List[str]] = Form(None, description="List of URLs of the audio items"),
    title: Optional[str] = Form(None, description="Optional title (applied if only one item processed)"),
    author: Optional[str] = Form(None, description="Optional author (applied similarly to title)"),
    keywords: str = Form("", alias="keywords", description="Comma-separated keywords"),
    custom_prompt: Optional[str] = Form(None, description="Optional custom prompt"),
    system_prompt: Optional[str] = Form(None, description="Optional system prompt"),
    overwrite_existing: bool = Form(False, description="Overwrite existing media (Not used in this endpoint, but needed for model)"),
    perform_analysis: bool = Form(True, description="Perform analysis"),
    api_name: Optional[str] = Form(None, description="Optional API name"),
    api_key: Optional[str] = Form(None, description="Optional API key"),
    use_cookies: bool = Form(False, description="Use cookies for URL download requests"),
    cookies: Optional[str] = Form(None, description="Cookie string if `use_cookies` is True"),
    transcription_model: str = Form("deepdml/faster-distil-whisper-large-v3.5", description="Transcription model"),
    transcription_language: str = Form("en", description="Transcription language"),
    diarize: bool = Form(False, description="Enable speaker diarization"),
    timestamp_option: bool = Form(True, description="Include timestamps in transcription"),
    vad_use: bool = Form(False, description="Enable VAD filter"),
    perform_confabulation_check_of_analysis: bool = Form(False, description="Enable confabulation check"),
    # Chunking options
    perform_chunking: bool = Form(True, description="Enable chunking"),
    chunk_method: Optional[ChunkMethod] = Form(None, description="Chunking method"),
    use_adaptive_chunking: bool = Form(False, description="Enable adaptive chunking"),
    use_multi_level_chunking: bool = Form(False, description="Enable multi-level chunking"),
    chunk_language: Optional[str] = Form(None, description="Chunking language override"),
    chunk_size: int = Form(500, description="Target chunk size"),
    chunk_overlap: int = Form(200, description="Chunk overlap size"),
    # Summarization options
    perform_rolling_summarization: bool = Form(False, description="Perform rolling summarization"), # Keep if AddMediaForm has it
    summarize_recursively: bool = Form(False, description="Perform recursive summarization"),
    # PDF options (Needed for AddMediaForm compatibility, ignored for audio)
    pdf_parsing_engine: Optional[PdfEngine] = Form("pymupdf4llm", description="PDF parsing engine (for model compatibility)"),
    custom_chapter_pattern: Optional[str] = Form(None, description="Regex pattern for custom chapter splitting (for model compatibility)"),
    # Audio/Video specific timing (Not applicable to audio-only usually, but keep for model compatibility if needed)
    start_time: Optional[str] = Form(None, description="Optional start time (HH:MM:SS or seconds)"),
    end_time: Optional[str] = Form(None, description="Optional end time (HH:MM:SS or seconds)"),

) -> ProcessAudiosForm:
    """
    Dependency function to parse form data and validate it
    against the ProcessAudiosForm model.
    """
    try:
        # Map form fields to ProcessAudiosForm fields
        form_instance = ProcessAudiosForm(
            urls=urls,
            title=title,
            author=author,
            keywords=keywords,
            custom_prompt=custom_prompt,
            system_prompt=system_prompt,
            overwrite_existing=overwrite_existing,
            keep_original_file=False,
            perform_analysis=perform_analysis,
            api_name=api_name,
            api_key=api_key,
            use_cookies=use_cookies,
            cookies=cookies,
            transcription_model=transcription_model,
            transcription_language=transcription_language,
            diarize=diarize,
            timestamp_option=timestamp_option,
            vad_use=vad_use,
            perform_confabulation_check_of_analysis=perform_confabulation_check_of_analysis,
            perform_chunking=perform_chunking,
            chunk_method=chunk_method,
            use_adaptive_chunking=use_adaptive_chunking,
            use_multi_level_chunking=use_multi_level_chunking,
            chunk_language=chunk_language,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            summarize_recursively=summarize_recursively,
            # Include fields inherited from AddMediaForm even if not directly used for audio
            perform_rolling_summarization=perform_rolling_summarization,
            pdf_parsing_engine=pdf_parsing_engine,
            custom_chapter_pattern=custom_chapter_pattern,
            start_time=start_time,
            end_time=end_time,
        )
        return form_instance
    except ValidationError as e:
        # Log the validation error details for debugging
        logger.warning(f"Form validation failed for /process-audios: {e.errors()}")
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=e.errors(),
        ) from e
    except Exception as e:
        logger.error(f"Unexpected error creating ProcessAudiosForm: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Internal server error during form processing: {type(e).__name__}"
        )


# =============================================================================
# Audio Processing Endpoint (REFACTORED)
# =============================================================================
@router.post(
    "/process-audios",
    # status_code=status.HTTP_200_OK, # Status determined dynamically
    summary="Transcribe / chunk / analyse audio and return full artefacts (no DB write)",
    tags=["Media Processing (No DB)"],
    # Consider adding response models for better documentation and validation
    # response_model=YourBatchResponseModel,
    # responses={ # Example explicit responses
    #     200: {"description": "All items processed successfully."},
    #     207: {"description": "Partial success with some errors."},
    #     400: {"description": "Bad request (e.g., no input)."},
    #     422: {"description": "Validation error in form data."},
    #     500: {"description": "Internal server error."},
    # }
)
async def process_audios_endpoint(
    background_tasks: BackgroundTasks,
    # 1. Auth + UserID Determined through `get_db_by_user`
    # token: str = Header(None), # Use Header(None) for optional
    # 2. DB Dependency
    db: MediaDatabase = Depends(get_media_db_for_user),
    # 3. Use Dependency Injection for Form Data
    form_data: ProcessAudiosForm = Depends(get_process_audios_form),
    # 4. File uploads remain separate
    files: Optional[List[UploadFile]] = File(None, description="Audio file uploads"),
):
    """
    **Process Audio Endpoint (Refactored)**

    Transcribes, chunks, and analyses audio from URLs or uploaded files.
    Returns processing artifacts without saving to the database. Uses dependency
    injection for form handling, consistent with the video endpoint.
    """
    # --- 0) Validation and Logging ---
    # Validation happened in the dependency. Log success or handle HTTPException.
    logger.info(f"Request received for /process-audios. Form data validated via dependency.")

    if form_data.urls and form_data.urls == ['']:
        logger.info("Received urls=[''], treating as no URLs provided for audio processing.")
        form_data.urls = None # Or []

    # Use the helper function from media_endpoints_utils
    try:
        _validate_inputs("audio", form_data.urls, files)
    except HTTPException as e:
         logger.warning(f"Input validation failed: {e.detail}")
         # Re-raise the HTTPException from _validate_inputs
         raise e

    # --- Rest of the logic using form_data ---
    loop = asyncio.get_running_loop()
    file_errors: List[Dict[str, Any]] = []
    # Initialize batch result structure
    batch_result: Dict[str, Any] = {"processed_count": 0, "errors_count": 0, "errors": [], "results": []}
    temp_path_to_original_name: Dict[str, str] = {}

    # ── 1) temp dir + uploads ────────────────────────────────────────────────
    with TempDirManager(cleanup=True, prefix="process_audio_") as temp_dir:
        temp_dir_path = Path(temp_dir)
        ALLOWED_AUDIO_EXTENSIONS = ['.mp3', '.aac', '.flac', '.wav', '.ogg', '.m4a'] # Define allowed extensions
        saved_files, file_errors_raw = await _save_uploaded_files(
            files or [],
            temp_dir_path,
            validator=file_validator_instance,
            allowed_extensions=ALLOWED_AUDIO_EXTENSIONS # Pass allowed extensions
        )

        for sf in saved_files:
            if sf.get("path") and sf.get("original_filename"):
                temp_path_to_original_name[str(sf["path"])] = sf["original_filename"]
            else:
                logger.warning(f"Missing path or original_filename in saved_files_info item for audio: {sf}")

        # --- Adapt File Errors to Response Structure ---
        if file_errors:
            adapted_file_errors = []
            for err in file_errors:
                 # Ensure all necessary keys are present for consistency
                 original_filename = err.get("original_filename") or err.get("input", "Unknown Upload")
                 adapted_file_errors.append({
                     "status": "Error",
                     "input_ref": original_filename,
                     "processing_source": err.get("input", "Unknown Filename"),
                     "media_type": "audio",
                     "metadata": {},
                     "content": "",
                     "segments": None,
                     "chunks": None,      # Add chunks field
                     "analysis": None,    # Add analysis field
                     "analysis_details": {},
                     "error": err.get("error", "Failed to save uploaded file."),
                     "warnings": None,
                     "db_id": None,       # Explicitly None
                     "db_message": "Processing only endpoint.", # Explicit message
                     "message": "File saving failed.", # Optional general message
                 })
            batch_result["results"].extend(adapted_file_errors)
            batch_result["errors_count"] = len(file_errors)
            batch_result["errors"].extend([err["error"] for err in adapted_file_errors])

        url_list = form_data.urls or []
        uploaded_paths = [str(f["path"]) for f in saved_files]
        all_inputs = url_list + uploaded_paths

        # Check if there are any valid inputs *after* attempting saves
        if not all_inputs:
            # If only file errors occurred, return 207, otherwise 400
            status_code = status.HTTP_207_MULTI_STATUS if file_errors_raw else status.HTTP_400_BAD_REQUEST
            detail = "No valid audio sources supplied (or all uploads failed)."
            logger.warning(f"Request processing stopped: {detail}")
            if status_code == status.HTTP_400_BAD_REQUEST:
                 raise HTTPException(status_code=status_code, detail=detail)
            else:
                 return JSONResponse(status_code=status_code, content=batch_result)


        # ── 2) invoke library batch processor ────────────────────────────────
        # Use validated form_data directly
        audio_args = {
            "inputs": all_inputs,
            "transcription_model": form_data.transcription_model,
            "transcription_language": form_data.transcription_language,
            "perform_chunking": form_data.perform_chunking,
            "chunk_method": form_data.chunk_method if form_data.chunk_method else None, # Pass enum value
            "max_chunk_size": form_data.chunk_size, # Correct mapping
            "chunk_overlap": form_data.chunk_overlap,
            "use_adaptive_chunking": form_data.use_adaptive_chunking,
            "use_multi_level_chunking": form_data.use_multi_level_chunking,
            "chunk_language": form_data.chunk_language,
            "diarize": form_data.diarize,
            "vad_use": form_data.vad_use,
            "timestamp_option": form_data.timestamp_option,
            "perform_analysis": form_data.perform_analysis,
            "api_name": form_data.api_name if form_data.perform_analysis else None,
            "api_key": form_data.api_key,
            "custom_prompt_input": form_data.custom_prompt,
            "system_prompt_input": form_data.system_prompt,
            "summarize_recursively": form_data.summarize_recursively,
            "use_cookies": form_data.use_cookies,
            "cookies": form_data.cookies,
            "keep_original": False, # Explicitly false for this endpoint
            "custom_title": form_data.title,
            "author": form_data.author,
            "temp_dir": str(temp_dir_path), # Pass the managed temp dir path
        }

        processing_output = None
        try:
            logger.debug(f"Calling process_audio_files for /process-audios with {len(all_inputs)} inputs.")
            # Use functools.partial to pass arguments cleanly
            batch_func = functools.partial(process_audio_files, **audio_args)
            # Run the synchronous library function in an executor thread
            processing_output = await loop.run_in_executor(None, batch_func)

        except Exception as exec_err:
            # Catch errors during the execution setup or within the library if it raises unexpectedly
            logging.error(f"Error executing process_audio_files: {exec_err}", exc_info=True)
            error_msg = f"Error during audio processing execution: {type(exec_err).__name__}"
            # Calculate errors based on *attempted* inputs for this batch
            num_attempted = len(all_inputs)
            batch_result["errors_count"] += num_attempted # Assume all failed if executor errored
            batch_result["errors"].append(error_msg)
            # Create error entries for all inputs attempted in this batch
            error_results = []
            for input_src in all_inputs:
                original_ref = temp_path_to_original_name.get(str(input_src), str(input_src))
                if input_src in uploaded_paths:
                    for sf in saved_files:
                         if str(sf["path"]) == input_src:
                              original_ref = sf.get("original_filename", input_src)
                              break
                error_results.append({
                    "status": "Error",
                    "input_ref": original_ref,
                    "processing_source": input_src,
                    "media_type": "audio",
                    "error": error_msg,
                    "db_id": None,
                    "db_message": "Processing only endpoint.",
                    "metadata": {},
                    "content": "",
                    "segments": None,
                    "chunks": None,
                    "analysis": None,
                    "analysis_details": {},
                    "warnings": None,
                    "message": "Processing execution failed."
                })
            # Combine these errors with any previous file errors
            batch_result["results"].extend(error_results)
            # Fall through to return section

        # --- Merge Processing Results ---
        if processing_output and isinstance(processing_output, dict) and "results" in processing_output:
            # Update counts based on library's report
            batch_result["processed_count"] += processing_output.get("processed_count", 0)
            new_errors_count = processing_output.get("errors_count", 0)
            batch_result["errors_count"] += new_errors_count
            batch_result["errors"].extend(processing_output.get("errors", []))

            processed_items = processing_output.get("results", [])
            adapted_processed_items = []
            for item in processed_items:

                 identifier_from_lib = item.get("input_ref") or item.get("processing_source")
                 original_ref = temp_path_to_original_name.get(str(identifier_from_lib), str(identifier_from_lib))
                 item["input_ref"] = original_ref
                 # Keep processing_source as what library used
                 item["processing_source"] = identifier_from_lib or original_ref

                 # Ensure DB fields are set correctly and all expected fields exist
                 item["db_id"] = None
                 item["db_message"] = "Processing only endpoint."
                 item.setdefault("status", "Error") # Default status if missing
                 item.setdefault("input_ref", "Unknown")
                 item.setdefault("processing_source", "Unknown")
                 item.setdefault("media_type", "audio") # Ensure media type
                 item.setdefault("metadata", {})
                 item.setdefault("content", None) # Default content to None
                 item.setdefault("segments", None)
                 item.setdefault("chunks", None) # Add default for chunks
                 item.setdefault("analysis", None) # Add default for analysis
                 item.setdefault("analysis_details", {})
                 item.setdefault("error", None)
                 item.setdefault("warnings", None)
                 item.setdefault("message", None) # Optional message from library
                 adapted_processed_items.append(item)

            # Combine processing results with any previous file errors
            batch_result["results"].extend(adapted_processed_items)

        elif processing_output is None and not batch_result["results"]: # Handle case where executor failed AND no file errors
             # This case is now handled by the try/except around run_in_executor
             pass
        elif processing_output is not None:
            # Handle unexpected output format from the library function more gracefully
            logging.error(f"process_audio_files returned unexpected format: Type={type(processing_output)}")
            error_msg = "Audio processing library returned invalid data."
            num_attempted = len(all_inputs)
            batch_result["errors_count"] += num_attempted
            batch_result["errors"].append(error_msg)
            # Create error results for inputs if not already present
            existing_refs = {res.get("input_ref") for res in batch_result["results"]}
            error_results = []
            for input_src in all_inputs:
                original_ref = temp_path_to_original_name.get(str(input_src), str(input_src))
                if input_src in uploaded_paths:
                    for sf in saved_files:
                         if str(sf["path"]) == input_src:
                              original_ref = sf.get("original_filename", input_src)
                              break
                if original_ref not in existing_refs: # Only add errors for inputs not already covered (e.g., by file errors)
                    error_results.append({
                        "status": "Error",
                        "input_ref": original_ref,
                        "processing_source": input_src,
                        "media_type": "audio",
                        "error": error_msg,
                        "db_id": None,
                        "db_message": "Processing only endpoint.",
                        "metadata": {}, "content": "",
                        "segments": None,
                        "chunks": None,
                        "analysis": None,
                        "analysis_details": {},
                        "warnings": None,
                        "message": "Invalid processing result."
                    })
            batch_result["results"].extend(error_results)

    # TempDirManager cleans up the directory automatically here (unless keep_original=True passed to it)
    # ── 4) Determine Final Status Code ───────────────────────────────────────
    # Base final status on whether *any* errors occurred (file saving or processing)
    final_processed_count = sum(1 for r in batch_result["results"] if r.get("status") == "Success")
    final_error_count = sum(1 for r in batch_result["results"] if r.get("status") == "Error")
    batch_result["processed_count"] = final_processed_count
    batch_result["errors_count"] = final_error_count
    # Update errors list to avoid duplicates (optional)
    unique_errors = list(set(str(e) for e in batch_result["errors"] if e))
    batch_result["errors"] = unique_errors

    final_status_code = (
        status.HTTP_200_OK if batch_result.get("errors_count", 0) == 0 and batch_result.get("processed_count", 0) > 0
        else status.HTTP_207_MULTI_STATUS if batch_result.get("results") # Return 207 if there are *any* results (success, warning, or error)
        else status.HTTP_400_BAD_REQUEST # Only 400 if no inputs were ever processed (e.g., invalid initial request)
    )

    # --- Return Combined Results ---
    if final_status_code == status.HTTP_200_OK:
        logging.info("Congrats, all successful!")
        logger.info(
            f"/process-audios request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Total Errors: {batch_result.get('errors_count', 0)}")
    else:
        logging.warning("Not all submissions were processed succesfully! Please Try Again!")
        logger.warning(f"/process-audios request finished with status {final_status_code}. Results count: {len(batch_result.get('results', []))}, Total Errors: {batch_result.get('errors_count', 0)}")

    return JSONResponse(status_code=final_status_code, content=batch_result)
