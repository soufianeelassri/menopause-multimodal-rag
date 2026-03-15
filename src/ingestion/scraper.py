"""Selenium-based PLOS ONE article scraper for menopause corpus.

Automates article discovery, PDF download, and validation from PLOS ONE.
Implements rate limiting to respect robots.txt.
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from src.config.settings import Settings, get_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ArticleScraper:
    """Scrapes menopause research articles from PLOS ONE.

    Downloads PDFs to the configured raw PDF directory with deduplication
    and PDF integrity validation.

    Args:
        settings: Application settings. Uses defaults if not provided.
        max_articles: Maximum number of articles to scrape.
    """

    BASE_URL = (
        "https://journals.plos.org/plosone/search?"
        "filterJournals=PLoSONE&q=menopause&sortOrder=RELEVANCE"
    )

    def __init__(
        self,
        settings: Settings | None = None,
        max_articles: int = 100,
    ) -> None:
        self._settings = settings or get_settings()
        self._max_articles = max_articles
        self._output_dir = self._settings.raw_pdf_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._driver: webdriver.Chrome | None = None

    def _initialize_driver(self) -> webdriver.Chrome:
        """Create a headless Chrome WebDriver instance.

        Returns:
            Configured Chrome WebDriver.
        """
        options = Options()
        options.add_argument("--headless")
        options.add_argument("--disable-gpu")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(
            "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

        driver = webdriver.Chrome(options=options)
        driver.set_page_load_timeout(30)
        logger.info("chrome_driver_initialized")
        return driver

    def get_article_urls(
        self,
        start_page: int = 1,
    ) -> list[str]:
        """Scrape article URLs from PLOS ONE search results.

        Args:
            start_page: Page number to start scraping from.

        Returns:
            List of article URLs found.
        """
        if self._driver is None:
            self._driver = self._initialize_driver()

        article_urls: list[str] = []
        page = start_page

        while len(article_urls) < self._max_articles:
            url = f"{self.BASE_URL}&page={page}"
            logger.info("scraping_search_page", page=page, url=url)

            try:
                self._driver.get(url)
                WebDriverWait(self._driver, 10).until(
                    EC.presence_of_element_located(
                        (By.CSS_SELECTOR, "dt.search-results-title a")
                    )
                )

                links = self._driver.find_elements(
                    By.CSS_SELECTOR, "dt.search-results-title a"
                )
                if not links:
                    logger.info("no_more_results", page=page)
                    break

                for link in links:
                    href = link.get_attribute("href")
                    if href and href not in article_urls:
                        article_urls.append(href)
                        if len(article_urls) >= self._max_articles:
                            break

                page += 1
                time.sleep(2)  # Rate limiting

            except Exception as e:
                logger.error("search_page_error", page=page, error=str(e))
                break

        logger.info("articles_found", count=len(article_urls))
        return article_urls

    def download_pdf(self, article_url: str) -> bytes | None:
        """Download PDF from a PLOS ONE article page.

        Args:
            article_url: URL of the article page.

        Returns:
            PDF content as bytes, or None on failure.
        """
        try:
            doi = article_url.split("article?id=")[-1]
            pdf_url = (
                f"https://journals.plos.org/plosone/article/file?"
                f"id={doi}&type=printable"
            )

            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()

            if response.content[:4] != b"%PDF":
                logger.warning("invalid_pdf_header", url=pdf_url)
                return None

            return response.content

        except Exception as e:
            logger.error("pdf_download_error", url=article_url, error=str(e))
            return None

    @staticmethod
    def sanitize_filename(title: str) -> str:
        """Create a filesystem-safe filename from an article title.

        Args:
            title: Raw article title.

        Returns:
            Sanitized filename string (max 100 chars).
        """
        safe = re.sub(r'[<>:"/\\|?*]', "_", title)
        safe = re.sub(r"\s+", " ", safe).strip()
        return safe[:100]

    def save_pdf(self, title: str, pdf_data: bytes) -> Path | None:
        """Save PDF data to the raw PDFs directory.

        Skips if a file with the same name already exists (deduplication).

        Args:
            title: Article title for filename generation.
            pdf_data: Raw PDF bytes.

        Returns:
            Path to saved file, or None if skipped/failed.
        """
        filename = self.sanitize_filename(title) + ".pdf"
        filepath = self._output_dir / filename

        if filepath.exists():
            logger.info("pdf_already_exists", filename=filename)
            return None

        filepath.write_bytes(pdf_data)
        logger.info("pdf_saved", filename=filename, size_kb=len(pdf_data) // 1024)
        return filepath

    def scrape(self) -> list[Path]:
        """Run the full scraping pipeline.

        Returns:
            List of paths to newly downloaded PDFs.
        """
        logger.info(
            "scraping_started",
            max_articles=self._max_articles,
            output_dir=str(self._output_dir),
        )

        try:
            self._driver = self._initialize_driver()
            article_urls = self.get_article_urls()
            saved_paths: list[Path] = []

            for i, url in enumerate(article_urls):
                logger.info(
                    "processing_article",
                    index=i + 1,
                    total=len(article_urls),
                )

                if self._driver is None:
                    break

                try:
                    self._driver.get(url)
                    title_element = WebDriverWait(self._driver, 10).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "h1#artTitle")
                        )
                    )
                    title = title_element.text
                except Exception:
                    title = f"article_{i + 1}"

                pdf_data = self.download_pdf(url)
                if pdf_data:
                    path = self.save_pdf(title, pdf_data)
                    if path:
                        saved_paths.append(path)

                time.sleep(2)  # Rate limiting

            logger.info("scraping_completed", new_pdfs=len(saved_paths))
            return saved_paths

        finally:
            if self._driver:
                self._driver.quit()
                self._driver = None
