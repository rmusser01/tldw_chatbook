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
from typing import List, Dict, Any, Callable, Optional, Coroutine
from tqdm.asyncio import tqdm_asyncio
#
# Third-Party Libraries
import asyncio
#
# Local Imports
from .scraper import Scraper
from .config import ProcessorConfig
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
    logging.info(f"Simulating summarization for content of length {len(content)}...")
    # summary = await analyze(...)
    await asyncio.sleep(0.1)  # Simulate network latency
    return f"This is a simulated summary based on prompt: '{config.custom_prompt}'"


async def default_db_logger(article_data: Dict[str, Any]):
    # This is where you would put your DB ingestion logic.
    # e.g., from tldw_chatbook.DB import ingest_article_to_db
    logging.info(f"Simulating DB ingestion for article: '{article_data.get('title')}'")
    # await ingest_article_to_db(...)
    await asyncio.sleep(0.05)  # Simulate DB latency


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
    results = []

    # Scrape all URLs concurrently
    scraped_articles = await scraper.scrape_many(urls)

    successful_articles = [art for art in scraped_articles if art.get('extraction_successful')]

    async def process_one(article: Dict[str, Any]):
        if proc_config.summarize and article.get('content'):
            article['summary'] = await summarizer(article['content'], proc_config)
        else:
            article['summary'] = None

        if db_logger:
            await db_logger(article)

        return article

    # Process all successful articles concurrently
    tasks = [process_one(art) for art in successful_articles]
    processed_results = await tqdm_asyncio.gather(*tasks, desc="Summarizing and Processing")

    # Add failed articles back in for a complete report
    failed_articles = [art for art in scraped_articles if not art.get('extraction_successful')]

    return processed_results + failed_articles

#
# End of article_scraper/processors.py
#######################################################################################################################
