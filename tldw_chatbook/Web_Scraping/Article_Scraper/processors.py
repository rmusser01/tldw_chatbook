"""
article_scraper/processors.py
=============================

High-level processing pipeline for scraped content.

This module provides the glue between scraping and downstream
processing like summarization and database storage. It supports
dependency injection for custom processing functions.

Key Features:
-------------
- Concurrent processing of multiple articles
- Pluggable summarization function
- Optional database logging
- Progress tracking with tqdm

Types:
------
- Summarizer: Async function for content summarization
- DBLogger: Async function for database storage

Functions:
----------
- scrape_and_process_urls(): Main processing pipeline
- default_summarizer(): Placeholder summarizer
- default_db_logger(): Placeholder DB logger

Example:
--------
    async with Scraper(config=scraper_config) as scraper:
        results = await scrape_and_process_urls(
            urls=urls_list,
            proc_config=processor_config,
            scraper=scraper,
            summarizer=my_llm_summarizer,
            db_logger=my_db_function
        )
"""
import logging
import time
from typing import List, Dict, Any, Callable, Optional, Coroutine
from tqdm.asyncio import tqdm_asyncio
#
# Third-Party Libraries
import asyncio
#
# Local Imports
from .scraper import Scraper
from .config import ProcessorConfig
from ...Metrics.metrics_logger import log_counter, log_histogram
#
#######################################################################################################################
#
# Functions:

# Define types for our injectable functions
Summarizer = Callable[[str, ProcessorConfig], Coroutine[Any, Any, str]]
DBLogger = Callable[[Dict[str, Any]], Coroutine[Any, Any, None]]


async def default_summarizer(content: str, config: ProcessorConfig) -> str:
    # This is where you would put your actual LLM call.
    # We are simulating it for decoupling.
    # e.g., from tldw_chatbook.LLM_Calls import analyze
    start_time = time.time()
    logging.info(f"Simulating summarization for content of length {len(content)}...")
    log_counter("processor_default_summarizer_call", labels={"content_length": str(len(content) // 1000) + "k"})
    
    # summary = await analyze(...)
    await asyncio.sleep(0.1)  # Simulate network latency
    
    duration = time.time() - start_time
    log_histogram("processor_default_summarizer_duration", duration)
    
    return f"This is a simulated summary based on prompt: '{config.custom_prompt}'"


async def default_db_logger(article_data: Dict[str, Any]):
    # This is where you would put your DB ingestion logic.
    # e.g., from tldw_chatbook.DB import ingest_article_to_db
    start_time = time.time()
    logging.info(f"Simulating DB ingestion for article: '{article_data.get('title')}'")
    log_counter("processor_default_db_logger_call")
    
    # await ingest_article_to_db(...)
    await asyncio.sleep(0.05)  # Simulate DB latency
    
    duration = time.time() - start_time
    log_histogram("processor_default_db_logger_duration", duration)


async def scrape_and_process_urls(
        urls: List[str],
        proc_config: ProcessorConfig,
        scraper: Scraper,
        summarizer: Summarizer = default_summarizer,
        db_logger: Optional[DBLogger] = None
) -> List[Dict[str, Any]]:
    """
    High-level pipeline to scrape, process, and optionally store articles.
    
    This function orchestrates the complete workflow:
    1. Scrapes all URLs concurrently using the provided scraper
    2. Filters successful extractions
    3. Optionally summarizes content using the injected summarizer
    4. Optionally logs to database using the injected logger
    5. Returns all results including failures
    
    Args:
        urls (List[str]): List of URLs to process
        proc_config (ProcessorConfig): Configuration for processing:
            - api_name: LLM API to use
            - api_key: API authentication
            - summarize: Whether to generate summaries
            - custom_prompt: Prompt for summarization
        scraper (Scraper): Initialized Scraper instance
        summarizer (Summarizer): Async function for summarization
            Signature: async (content: str, config: ProcessorConfig) -> str
        db_logger (Optional[DBLogger]): Async function for DB storage
            Signature: async (article_data: Dict[str, Any]) -> None
            
    Returns:
        List[Dict[str, Any]]: All articles including:
            - Successful extractions with optional summaries
            - Failed extractions with error details
            
    Example:
        >>> async def my_summarizer(content: str, config: ProcessorConfig) -> str:
        ...     # Call your LLM API here
        ...     return await llm_summarize(content, config.custom_prompt)
        ...
        >>> async def my_db_logger(article: Dict[str, Any]):
        ...     await db.insert_article(article)
        ...
        >>> results = await scrape_and_process_urls(
        ...     urls=["https://example.com/1", "https://example.com/2"],
        ...     proc_config=ProcessorConfig(api_name="openai", summarize=True),
        ...     scraper=scraper,
        ...     summarizer=my_summarizer,
        ...     db_logger=my_db_logger
        ... )
    start_time = time.time()
    url_count = len(urls)
    
    # Log processing start
    log_counter("processor_pipeline_start", labels={
        "url_count": str(url_count),
        "summarize": str(proc_config.summarize),
        "has_db_logger": str(db_logger is not None)
    })
    
    results = []

    # Scrape all URLs concurrently
    scrape_start = time.time()
    scraped_articles = await scraper.scrape_many(urls)
    scrape_duration = time.time() - scrape_start
    
    log_histogram("processor_scraping_batch_duration", scrape_duration, labels={"url_count": str(url_count)})

    successful_articles = [art for art in scraped_articles if art.get('extraction_successful')]
    failed_count = len(scraped_articles) - len(successful_articles)
    
    log_counter("processor_scraping_results", labels={
        "successful": str(len(successful_articles)),
        "failed": str(failed_count)
    })

    async def process_one(article: Dict[str, Any]):
        process_start = time.time()
        
        try:
            # Summarization
            if proc_config.summarize and article.get('content'):
                summarize_start = time.time()
                article['summary'] = await summarizer(article['content'], proc_config)
                summarize_duration = time.time() - summarize_start
                log_histogram("processor_article_summarize_duration", summarize_duration)
                log_counter("processor_article_summarized")
            else:
                article['summary'] = None
                log_counter("processor_article_skipped_summary")

            # Database logging
            if db_logger:
                db_start = time.time()
                await db_logger(article)
                db_duration = time.time() - db_start
                log_histogram("processor_article_db_log_duration", db_duration)
                log_counter("processor_article_db_logged")
            
            # Log successful processing
            process_duration = time.time() - process_start
            log_histogram("processor_article_process_duration", process_duration, labels={"status": "success"})
            
            return article
        except Exception as e:
            # Log processing error
            process_duration = time.time() - process_start
            log_histogram("processor_article_process_duration", process_duration, labels={"status": "error"})
            log_counter("processor_article_process_error", labels={"error_type": type(e).__name__})
            logging.error(f"Error processing article: {e}")
            article['processing_error'] = str(e)
            return article

    # Process all successful articles concurrently
    process_start = time.time()
    tasks = [process_one(art) for art in successful_articles]
    processed_results = await tqdm_asyncio.gather(*tasks, desc="Summarizing and Processing")
    process_duration = time.time() - process_start
    
    log_histogram("processor_batch_process_duration", process_duration, labels={
        "article_count": str(len(successful_articles))
    })

    # Add failed articles back in for a complete report
    failed_articles = [art for art in scraped_articles if not art.get('extraction_successful')]
    
    # Log final pipeline results
    total_duration = time.time() - start_time
    log_histogram("processor_pipeline_duration", total_duration, labels={
        "url_count": str(url_count),
        "successful_count": str(len(successful_articles)),
        "failed_count": str(failed_count)
    })
    log_counter("processor_pipeline_complete", labels={
        "total_urls": str(url_count),
        "processed": str(len(processed_results)),
        "failed": str(len(failed_articles))
    })

    return processed_results + failed_articles

#
# End of article_scraper/processors.py
#######################################################################################################################
