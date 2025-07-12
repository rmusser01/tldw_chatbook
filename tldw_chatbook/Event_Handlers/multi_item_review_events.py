# tldw_chatbook/Event_Handlers/multi_item_review_events.py
"""
Event handlers for the Multi-Item Review functionality.
Handles batch analysis generation and result management.
"""

from typing import TYPE_CHECKING, List, Dict, Any, Optional
from textual.message import Message
from loguru import logger
from datetime import datetime
import asyncio

if TYPE_CHECKING:
    from ..app import TldwCli


class BatchAnalysisStartEvent(Message):
    """Event to start batch analysis generation."""
    
    def __init__(self, items: List[Dict[str, Any]], prompt: str, save_permanently: bool) -> None:
        super().__init__()
        self.items = items
        self.prompt = prompt
        self.save_permanently = save_permanently


class BatchAnalysisProgressEvent(Message):
    """Event for batch analysis progress updates."""
    
    def __init__(self, current: int, total: int, current_item: str) -> None:
        super().__init__()
        self.current = current
        self.total = total
        self.current_item = current_item


class BatchAnalysisCompleteEvent(Message):
    """Event when batch analysis is complete."""
    
    def __init__(self, successful: int, failed: int, results: Dict[int, str]) -> None:
        super().__init__()
        self.successful = successful
        self.failed = failed
        self.results = results  # media_id -> analysis content


async def handle_batch_analysis_start(app: 'TldwCli', event: BatchAnalysisStartEvent) -> None:
    """
    Handle the start of batch analysis generation.
    
    Args:
        app: The application instance
        event: The start event containing items, prompt, and save preference
    """
    logger.info(f"Starting batch analysis for {len(event.items)} items")
    
    try:
        if not app.media_db:
            raise RuntimeError("Media DB service not available")
            
        # Check if LLM is available
        if not hasattr(app, 'llm_api_client'):
            app.notify("LLM service not available for analysis generation", severity="error")
            return
            
        # Generate analyses for each item
        results = {}
        successful = 0
        failed = 0
        
        for index, item in enumerate(event.items):
            media_id = item['id']
            title = item.get('title', 'Untitled')
            
            # Post progress update
            app.post_message(BatchAnalysisProgressEvent(
                current=index + 1,
                total=len(event.items),
                current_item=title
            ))
            
            try:
                # Generate analysis
                analysis = await generate_single_analysis(app, item, event.prompt)
                
                if analysis:
                    results[media_id] = analysis
                    successful += 1
                    
                    # Save to database if requested
                    if event.save_permanently:
                        await save_analysis_to_db(app, media_id, analysis)
                else:
                    failed += 1
                    
            except Exception as e:
                logger.error(f"Error generating analysis for media {media_id}: {e}")
                failed += 1
                
            # Small delay to avoid rate limiting
            await asyncio.sleep(0.5)
            
        # Post completion event
        app.post_message(BatchAnalysisCompleteEvent(
            successful=successful,
            failed=failed,
            results=results
        ))
        
    except Exception as e:
        logger.error(f"Error in batch analysis: {e}", exc_info=True)
        app.notify(f"Error during batch analysis: {str(e)[:100]}", severity="error")


async def generate_single_analysis(app: 'TldwCli', item: Dict[str, Any], prompt: str) -> Optional[str]:
    """
    Generate analysis for a single media item using the LLM.
    
    Args:
        app: The application instance
        item: Media item dictionary
        prompt: Analysis prompt from user
        
    Returns:
        Generated analysis text or None if failed
    """
    try:
        # Get the content
        content = item.get('content', '')
        if not content:
            return "No content available for analysis."
            
        # Prepare the full prompt with content context
        full_prompt = f"""
{prompt}

Title: {item.get('title', 'Untitled')}
Type: {item.get('type', 'Unknown')}
Author: {item.get('author', 'Unknown')}
Date: {item.get('ingestion_date', 'Unknown')}

Content:
{content[:8000]}  # Limit content to avoid token limits
"""
        
        # Use the LLM to generate analysis
        if hasattr(app, 'llm_api_client') and app.llm_api_client:
            try:
                # Create messages for the LLM
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that provides detailed content analysis based on the given prompt."},
                    {"role": "user", "content": full_prompt}
                ]
                
                # Get current LLM settings
                model = app.llm_model_var
                temperature = app.llm_temperature_var
                max_tokens = app.llm_context_size_var
                
                # Generate response
                response = await app.run_in_thread(
                    app.llm_api_client.chat_with_model,
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=min(2000, max_tokens),  # Limit response size
                    stream=False
                )
                
                if response and isinstance(response, str):
                    # Add metadata to the analysis
                    analysis = f"{response}\n\n---\n*Analysis generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
                    return analysis
                else:
                    logger.error(f"Invalid response from LLM: {response}")
                    return None
                    
            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                return f"Error generating analysis: {str(e)}"
        else:
            # Fallback if no LLM available
            return generate_placeholder_analysis(item, prompt)
            
    except Exception as e:
        logger.error(f"Error in single analysis generation: {e}")
        return f"Error generating analysis: {str(e)}"


def generate_placeholder_analysis(item: Dict[str, Any], prompt: str) -> str:
    """
    Generate a placeholder analysis when LLM is not available.
    
    Args:
        item: Media item dictionary
        prompt: Analysis prompt
        
    Returns:
        Placeholder analysis text
    """
    return f"""## Analysis of "{item.get('title', 'Untitled')}"

### Summary
[Placeholder: In a real implementation, this would contain an AI-generated analysis based on the prompt: "{prompt[:100]}..."]

### Key Information
- **Type**: {item.get('type', 'Unknown')}
- **Author**: {item.get('author', 'Unknown')}
- **Date**: {item.get('ingestion_date', 'Unknown')}
- **Content Length**: {len(item.get('content', ''))} characters

### Analysis Note
This is a placeholder analysis. To generate real analyses, ensure:
1. An LLM provider is configured in settings
2. The LLM service is accessible
3. Valid API credentials are provided

*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""


async def save_analysis_to_db(app: 'TldwCli', media_id: int, analysis: str) -> bool:
    """
    Save analysis to the database.
    
    Args:
        app: The application instance
        media_id: ID of the media item
        analysis: Analysis content to save
        
    Returns:
        True if successful, False otherwise
    """
    try:
        if not app.media_db:
            return False
            
        # Update the analysis_content field
        query = """
            UPDATE Media 
            SET analysis_content = ?, last_modified = ?
            WHERE id = ?
        """
        
        await app.run_in_thread(
            app.media_db.execute_query,
            query,
            (analysis, datetime.now().isoformat(), media_id)
        )
        
        await app.run_in_thread(app.media_db.commit)
        
        logger.debug(f"Saved analysis for media ID {media_id}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving analysis to DB: {e}")
        return False


async def load_existing_analyses(app: 'TldwCli', media_ids: List[int]) -> Dict[int, Optional[str]]:
    """
    Load existing analyses for given media IDs.
    
    Args:
        app: The application instance
        media_ids: List of media IDs to check
        
    Returns:
        Dictionary mapping media_id to analysis_content (or None if not exists)
    """
    try:
        if not app.media_db or not media_ids:
            return {}
            
        placeholders = ','.join('?' * len(media_ids))
        query = f"""
            SELECT id, analysis_content 
            FROM Media 
            WHERE id IN ({placeholders})
        """
        
        cursor = await app.run_in_thread(
            app.media_db.execute_query,
            query,
            media_ids
        )
        
        results = {}
        for row in cursor.fetchall():
            results[row['id']] = row['analysis_content']
            
        return results
        
    except Exception as e:
        logger.error(f"Error loading existing analyses: {e}")
        return {}