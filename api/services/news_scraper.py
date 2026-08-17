"""
News Scraper Service using BeautifulSoup.
Scrapes financial news headlines for sentiment analysis.
"""
import logging
import re
import time
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests
from bs4 import BeautifulSoup

from api.models import NewsArticle

logger = logging.getLogger(__name__)

# Common headers to avoid being blocked
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    ),
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
}


class NewsScraper:
    """Service for scraping financial news using BeautifulSoup."""

    @staticmethod
    def scrape_finviz_news(ticker: str, max_articles: int = 50) -> list:
        """
        Scrape news headlines from Finviz for a given ticker.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL')
            max_articles: Maximum number of articles to scrape

        Returns:
            List of dicts with title, source, url, published_date
        """
        ticker = ticker.upper().strip()
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        logger.info(f"Scraping Finviz news for {ticker}")

        articles = []
        try:
            response = requests.get(url, headers=HEADERS, timeout=4)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, 'lxml')
            news_table = soup.find(id='news-table')

            if not news_table:
                logger.warning(f"No news table found on Finviz for {ticker}")
                return articles

            rows = news_table.find_all('tr')
            current_date = datetime.now()

            for row in rows[:max_articles]:
                try:
                    # Extract link and title
                    link_tag = row.find('a')
                    if not link_tag:
                        continue

                    title = link_tag.get_text(strip=True)
                    article_url = link_tag.get('href', '')

                    # Extract date/time
                    td_tag = row.find('td', {'align': 'right'})
                    date_text = td_tag.get_text(strip=True) if td_tag else ''

                    # Parse date — Finviz shows "Aug-12-24 09:30AM" or just "09:30AM"
                    published_date = NewsScraper._parse_finviz_date(date_text, current_date)

                    # Extract source
                    source_span = row.find('span', class_='news-link-right')
                    source = ''
                    if source_span:
                        source_text = source_span.get_text(strip=True)
                        source = source_text.strip('()')

                    articles.append({
                        'title': title,
                        'source': source,
                        'url': article_url,
                        'published_date': published_date,
                    })
                except Exception as row_err:
                    logger.debug(f"Error parsing news row: {row_err}")
                    continue

            logger.info(f"Scraped {len(articles)} articles from Finviz for {ticker}")

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error scraping Finviz for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error scraping Finviz for {ticker}: {e}", exc_info=True)

        return articles

    @staticmethod
    def scrape_yahoo_news(ticker: str, max_articles: int = 30) -> list:
        """
        Scrape news from Yahoo Finance RSS feed for a given ticker.

        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles

        Returns:
            List of article dicts
        """
        ticker = ticker.upper().strip()
        url = f'https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US'
        logger.info(f"Scraping Yahoo Finance RSS for {ticker}")

        articles = []
        try:
            response = requests.get(url, headers=HEADERS, timeout=4)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml-xml')
            items = soup.find_all('item')

            for item in items[:max_articles]:
                try:
                    title = item.find('title')
                    link = item.find('link')
                    pub_date = item.find('pubDate')
                    source = item.find('source')

                    published_date = None
                    if pub_date and pub_date.string:
                        try:
                            published_date = datetime.strptime(
                                pub_date.string.strip(),
                                '%a, %d %b %Y %H:%M:%S %z'
                            )
                        except ValueError:
                            published_date = datetime.now()

                    articles.append({
                        'title': title.string.strip() if title and title.string else '',
                        'source': source.string.strip() if source and source.string else 'Yahoo Finance',
                        'url': link.string.strip() if link and link.string else '',
                        'published_date': published_date,
                    })
                except Exception as row_err:
                    logger.debug(f"Error parsing Yahoo news item: {row_err}")
                    continue

            logger.info(f"Scraped {len(articles)} articles from Yahoo Finance for {ticker}")

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error scraping Yahoo Finance for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error scraping Yahoo Finance for {ticker}: {e}", exc_info=True)

        return articles

    @staticmethod
    def scrape_google_news(ticker: str, max_articles: int = 30) -> list:
        """
        Scrape news from Google News RSS for a given ticker.

        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum number of articles

        Returns:
            List of article dicts
        """
        ticker = ticker.upper().strip()
        query = quote_plus(f'{ticker} stock')
        url = f'https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en'
        logger.info(f"Scraping Google News RSS for {ticker}")

        articles = []
        try:
            response = requests.get(url, headers=HEADERS, timeout=4)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'lxml-xml')
            items = soup.find_all('item')

            for item in items[:max_articles]:
                try:
                    title = item.find('title')
                    link = item.find('link')
                    pub_date = item.find('pubDate')
                    source = item.find('source')

                    published_date = None
                    if pub_date and pub_date.string:
                        try:
                            published_date = datetime.strptime(
                                pub_date.string.strip(),
                                '%a, %d %b %Y %H:%M:%S %Z'
                            )
                        except ValueError:
                            try:
                                published_date = datetime.strptime(
                                    pub_date.string.strip(),
                                    '%a, %d %b %Y %H:%M:%S %z'
                                )
                            except ValueError:
                                published_date = datetime.now()

                    articles.append({
                        'title': title.string.strip() if title and title.string else '',
                        'source': source.get_text(strip=True) if source else 'Google News',
                        'url': link.string.strip() if link and link.string else '',
                        'published_date': published_date,
                    })
                except Exception as row_err:
                    logger.debug(f"Error parsing Google News item: {row_err}")
                    continue

            logger.info(f"Scraped {len(articles)} articles from Google News for {ticker}")

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP error scraping Google News for {ticker}: {e}")
        except Exception as e:
            logger.error(f"Error scraping Google News for {ticker}: {e}", exc_info=True)

        return articles

    @staticmethod
    def scrape_all_sources(ticker: str, max_articles: int = 50) -> dict:
        """
        Scrape news from all available sources and store in database.

        Args:
            ticker: Stock ticker symbol
            max_articles: Maximum articles per source

        Returns:
            dict with status and article counts
        """
        ticker = ticker.upper().strip()
        all_articles = []

        # Scrape from multiple sources with delays
        sources = [
            ('Finviz', NewsScraper.scrape_finviz_news),
            ('Yahoo Finance', NewsScraper.scrape_yahoo_news),
            ('Google News', NewsScraper.scrape_google_news),
        ]

        for source_name, scrape_func in sources:
            try:
                articles = scrape_func(ticker, max_articles=max_articles)
                all_articles.extend(articles)
                time.sleep(1)  # Respectful delay between sources
            except Exception as e:
                logger.error(f"Failed to scrape {source_name} for {ticker}: {e}")
                continue

        # Deduplicate by title similarity
        seen_titles = set()
        unique_articles = []
        for article in all_articles:
            # Normalize title for dedup
            normalized = re.sub(r'[^a-zA-Z0-9\s]', '', article['title'].lower()).strip()
            if normalized and normalized not in seen_titles:
                seen_titles.add(normalized)
                unique_articles.append(article)

        # Perform sentiment analysis and store
        from api.services.sentiment_engine import SentimentEngine

        saved_count = 0
        for article in unique_articles:
            try:
                if not article['title']:
                    continue

                # Analyze sentiment
                sentiment = SentimentEngine.get_combined_sentiment(article['title'])

                # Save to database (avoid duplicates)
                obj, created = NewsArticle.objects.get_or_create(
                    ticker=ticker,
                    title=article['title'][:500],
                    defaults={
                        'source': article.get('source', '')[:200],
                        'url': article.get('url', '')[:1000],
                        'published_date': article.get('published_date'),
                        'sentiment_score': sentiment['combined_score'],
                        'sentiment_label': sentiment['label'],
                        'vader_score': sentiment['vader_score'],
                        'textblob_score': sentiment['textblob_score'],
                    }
                )
                if created:
                    saved_count += 1
            except Exception as save_err:
                logger.error(f"Error saving article: {save_err}")
                continue

        return {
            'status': 'success',
            'ticker': ticker,
            'total_scraped': len(all_articles),
            'unique_articles': len(unique_articles),
            'saved_to_db': saved_count,
            'sources_scraped': len(sources),
        }

    @staticmethod
    def _parse_finviz_date(date_text: str, current_date: datetime) -> datetime:
        """Parse Finviz-style date strings."""
        if not date_text:
            return current_date

        date_text = date_text.strip()

        # Try full date format: "Aug-12-24 09:30AM"
        try:
            parts = date_text.split(' ')
            if len(parts) == 2 and '-' in parts[0]:
                return datetime.strptime(date_text, '%b-%d-%y %I:%M%p')
        except ValueError:
            pass

        # Try time-only format: "09:30AM" (same day)
        try:
            time_part = datetime.strptime(date_text, '%I:%M%p')
            return current_date.replace(
                hour=time_part.hour,
                minute=time_part.minute,
                second=0,
                microsecond=0
            )
        except ValueError:
            pass

        # Try "Today HH:MM" format
        try:
            if date_text.lower().startswith('today'):
                time_str = date_text.split(' ', 1)[1] if ' ' in date_text else ''
                if time_str:
                    time_part = datetime.strptime(time_str, '%I:%M%p')
                    return current_date.replace(
                        hour=time_part.hour,
                        minute=time_part.minute
                    )
                return current_date
        except (ValueError, IndexError):
            pass

        return current_date
