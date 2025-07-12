# XML_Ingestion.py
# Description: This file contains functions for reading and writing XML files.
# Imports
import xml.etree.ElementTree as ET
import time
#
# External Imports
#
# Local Imports
from tldw_chatbook.LLM_Calls.Summarization_General_Lib import analyze
from tldw_chatbook.Chunking.Chunk_Lib import chunk_xml
from tldw_chatbook.DB.Client_Media_DB_v2 import ingest_article_to_db_new, add_media_to_database
from tldw_chatbook.Utils.Utils import logging
from tldw_chatbook.Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

def xml_to_text(xml_file):
    start_time = time.time()
    log_counter("xml_ingestion_to_text_attempt")
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        # Extract text content recursively
        text_content = []
        element_count = 0
        
        for elem in root.iter():
            element_count += 1
            if elem.text and elem.text.strip():
                text_content.append(elem.text.strip())
        
        # Log success metrics
        duration = time.time() - start_time
        result_text = '\n'.join(text_content)
        
        log_histogram("xml_ingestion_to_text_duration", duration, labels={"status": "success"})
        log_histogram("xml_ingestion_element_count", element_count)
        log_histogram("xml_ingestion_text_length", len(result_text))
        log_counter("xml_ingestion_to_text_success")
        
        return result_text
    except ET.ParseError as e:
        # Log parse error
        duration = time.time() - start_time
        log_histogram("xml_ingestion_to_text_duration", duration, labels={"status": "error"})
        log_counter("xml_ingestion_to_text_error", labels={"error_type": "parse_error"})
        logging.error(f"Error parsing XML file: {str(e)}")
        return None
    except Exception as e:
        # Log unexpected error
        duration = time.time() - start_time
        log_histogram("xml_ingestion_to_text_duration", duration, labels={"status": "error"})
        log_counter("xml_ingestion_to_text_error", labels={"error_type": type(e).__name__})
        logging.error(f"Unexpected error in xml_to_text: {str(e)}")
        return None


def import_xml_handler(import_file, title, author, keywords, system_prompt,
                       custom_prompt, auto_summarize, api_name, api_key):
    if not import_file:
        log_counter("xml_ingestion_import_error", labels={"error_type": "no_file"})
        return "Please upload an XML file"

    start_time = time.time()
    log_counter("xml_ingestion_import_attempt", labels={
        "has_title": str(bool(title)),
        "has_author": str(bool(author)),
        "auto_summarize": str(auto_summarize)
    })
    
    try:
        # Parse XML and extract text with structure
        parse_start = time.time()
        tree = ET.parse(import_file.name)
        root = tree.getroot()
        parse_duration = time.time() - parse_start
        log_histogram("xml_ingestion_parse_duration", parse_duration)

        # Create chunk options
        chunk_options = {
            'method': 'xml',
            'max_size': 1000,  # Adjust as needed
            'overlap': 200,  # Adjust as needed
            'language': 'english'  # Add language detection if needed
        }

        # Use the chunk_xml function to get structured chunks
        chunk_start = time.time()
        chunks = chunk_xml(ET.tostring(root, encoding='unicode'), chunk_options)
        chunk_duration = time.time() - chunk_start
        
        log_histogram("xml_ingestion_chunking_duration", chunk_duration)
        log_histogram("xml_ingestion_chunk_count", len(chunks))

        # Convert chunks to segments format expected by add_media_to_database
        segments = []
        for chunk in chunks:
            segment = {
                'Text': chunk['text'],
                'metadata': chunk['metadata']  # Preserve XML structure metadata
            }
            segments.append(segment)

        # Create info_dict
        info_dict = {
            'title': title or 'Untitled XML Document',
            'uploader': author or 'Unknown',
            'file_type': 'xml',
            'structure': root.tag  # Save root element type
        }

        # Process keywords
        keyword_list = [kw.strip() for kw in keywords.split(',') if kw.strip()] if keywords else []

        # Handle summarization
        if auto_summarize and api_name and api_key:
            # Combine all chunks for summarization
            summarize_start = time.time()
            full_text = '\n'.join(chunk['text'] for chunk in chunks)
            log_histogram("xml_ingestion_full_text_length", len(full_text))
            
            summary = analyze(api_name, full_text, custom_prompt, api_key)
            summarize_duration = time.time() - summarize_start
            
            log_histogram("xml_ingestion_summarization_duration", summarize_duration)
            log_counter("xml_ingestion_summarization_performed")
        else:
            summary = "No summary provided"
            log_counter("xml_ingestion_summarization_skipped")

        # Add to database
        result = add_media_to_database(
            url=import_file.name,  # Using filename as URL
            info_dict=info_dict,
            segments=segments,
            summary=summary,
            keywords=keyword_list,
            custom_prompt_input=custom_prompt,
            whisper_model="XML Import",
            media_type="xml_document",
            overwrite=False
        )

        # Log successful import
        duration = time.time() - start_time
        log_histogram("xml_ingestion_import_duration", duration, labels={"status": "success"})
        log_counter("xml_ingestion_import_success", labels={
            "root_tag": root.tag,
            "chunk_count": str(len(chunks)),
            "summarized": str(auto_summarize and api_name and api_key)
        })
        
        return f"XML file '{import_file.name}' import complete. Database result: {result}"

    except ET.ParseError as e:
        # Log parse error
        duration = time.time() - start_time
        log_histogram("xml_ingestion_import_duration", duration, labels={"status": "error"})
        log_counter("xml_ingestion_import_error", labels={"error_type": "parse_error"})
        logging.error(f"XML parsing error: {str(e)}")
        return f"Error parsing XML file: {str(e)}"
    except Exception as e:
        # Log unexpected error
        duration = time.time() - start_time
        log_histogram("xml_ingestion_import_duration", duration, labels={"status": "error"})
        log_counter("xml_ingestion_import_error", labels={"error_type": type(e).__name__})
        logging.error(f"Error processing XML file: {str(e)}")
        return f"Error processing XML file: {str(e)}"

#
# End of XML_Ingestion_Lib.py
#######################################################################################################################
