# main.py
#
#
# Imports
#
# Third-Party Imports
import asyncio

#
# Local Imports
from .scraper import Scraper, ScraperConfig
from .processors import ProcessorConfig, scrape_and_process_urls
#
#######################################################################################################################
#
# Functions:


async def main():
    urls_to_scrape = [
        "https://www.theverge.com/2024/5/14/24156134/google-io-2024-ai-search-gemini-astea-project",
        "https://arstechnica.com/gadgets/2024/05/android-15s-second-beta-is-here/",
        "https://www.invalid-url-that-will-fail.com/article",
    ]

    # 1. Configure the scraper and processor
    scraper_config = ScraperConfig(stealth=True, retries=2)
    processor_config = ProcessorConfig(
        api_name="openai",
        api_key="your_key_here",
        summarize=True,
        custom_prompt="Provide three key takeaways from this tech article.",
    )

    # 2. Use the Scraper in a context manager for efficient browser handling
    async with Scraper(config=scraper_config) as scraper:
        # 3. Run the high-level processing pipeline
        results = await scrape_and_process_urls(
            urls=urls_to_scrape,
            proc_config=processor_config,
            scraper=scraper,
            # You could inject your own custom summarizer or DB logger here
            # summarizer=my_custom_llm_call,
            # db_logger=my_db_ingestion_function
        )

    # 4. Print the results
    for result in results:
        if result.get("extraction_successful"):
            print("-" * 50)
            print(f"URL: {result['url']}")
            print(f"Title: {result['title']}")
            print(f"Summary: {result.get('summary', 'N/A')}")
            print("-" * 50, "\n")
        else:
            print(
                f"Failed to process URL: {result['url']}. Error: {result.get('error')}\n"
            )


if __name__ == "__main__":
    asyncio.run(main())
