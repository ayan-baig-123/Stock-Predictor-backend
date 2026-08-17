"""
Stock Data Scraper Service using yfinance.
Fetches OHLCV data for given stock tickers.
"""
import logging
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

from api.models import StockData

logger = logging.getLogger(__name__)


class StockScraper:
    """Service for scraping stock price data using yfinance."""

    VALID_PERIODS = ['1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'max']
    VALID_INTERVALS = ['1d', '5d', '1wk', '1mo']

    @staticmethod
    def scrape_stock_data(ticker: str, period: str = '2y', interval: str = '1d') -> dict:
        """
        Scrape stock data for a given ticker using yfinance.

        Args:
            ticker: Stock ticker symbol (e.g., 'AAPL', 'GOOGL')
            period: Data period ('1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')
            interval: Data interval ('1d', '5d', '1wk', '1mo')

        Returns:
            dict with status, count of records, and data summary
        """
        logger.info(f"Scraping stock data for {ticker} | period={period} | interval={interval}")

        try:
            # Validate inputs
            ticker = ticker.upper().strip()
            if period not in StockScraper.VALID_PERIODS:
                period = '2y'
            if interval not in StockScraper.VALID_INTERVALS:
                interval = '1d'

            # Fetch data from Yahoo Finance
            stock = yf.Ticker(ticker)
            df = stock.history(period=period, interval=interval)

            if df.empty:
                logger.warning(f"No data returned for ticker: {ticker}")
                return {
                    'status': 'error',
                    'message': f'No data found for ticker: {ticker}. Verify the symbol is correct.',
                    'count': 0
                }

            # Clean and validate data
            df = df.dropna()
            df = df.reset_index()

            # Rename columns to match our model
            column_map = {
                'Date': 'date',
                'Open': 'open_price',
                'High': 'high_price',
                'Low': 'low_price',
                'Close': 'close_price',
                'Volume': 'volume',
            }
            df = df.rename(columns=column_map)

            # Handle Adj Close — may or may not exist
            if 'Adj Close' in df.columns:
                df = df.rename(columns={'Adj Close': 'adj_close'})
            else:
                df['adj_close'] = df['close_price']

            # Ensure date column is date type (not datetime with timezone)
            if hasattr(df['date'].dtype, 'tz') and df['date'].dtype.tz is not None:
                df['date'] = df['date'].dt.tz_localize(None)
            df['date'] = pd.to_datetime(df['date']).dt.date

            # Store to database — bulk create with conflict handling
            records_created = 0
            records_updated = 0
            batch = []

            for _, row in df.iterrows():
                try:
                    obj, created = StockData.objects.update_or_create(
                        ticker=ticker,
                        date=row['date'],
                        defaults={
                            'open_price': float(row['open_price']),
                            'high_price': float(row['high_price']),
                            'low_price': float(row['low_price']),
                            'close_price': float(row['close_price']),
                            'adj_close': float(row['adj_close']),
                            'volume': int(row['volume']),
                        }
                    )
                    if created:
                        records_created += 1
                    else:
                        records_updated += 1
                except Exception as row_err:
                    logger.error(f"Error saving row for {ticker} on {row['date']}: {row_err}")
                    continue

            # Get stock info for metadata
            try:
                info = stock.info
                company_name = info.get('longName', info.get('shortName', ticker))
                current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
                market_cap = info.get('marketCap', 0)
            except Exception:
                company_name = ticker
                current_price = float(df['close_price'].iloc[-1]) if not df.empty else 0
                market_cap = 0

            result = {
                'status': 'success',
                'ticker': ticker,
                'company_name': company_name,
                'current_price': current_price,
                'market_cap': market_cap,
                'records_created': records_created,
                'records_updated': records_updated,
                'total_records': records_created + records_updated,
                'date_range': {
                    'start': str(df['date'].min()),
                    'end': str(df['date'].max()),
                },
                'price_range': {
                    'min': float(df['close_price'].min()),
                    'max': float(df['close_price'].max()),
                    'mean': float(df['close_price'].mean()),
                },
            }

            logger.info(
                f"Successfully scraped {result['total_records']} records for {ticker} "
                f"({records_created} new, {records_updated} updated)"
            )
            return result

        except Exception as e:
            logger.error(f"Error scraping data for {ticker}: {str(e)}", exc_info=True)
            return {
                'status': 'error',
                'message': f'Failed to scrape data for {ticker}: {str(e)}',
                'count': 0
            }

    @staticmethod
    def get_stock_dataframe(ticker: str, limit: int = None) -> pd.DataFrame:
        """
        Get stored stock data as a pandas DataFrame.

        Args:
            ticker: Stock ticker symbol
            limit: Optional limit on number of records

        Returns:
            pandas DataFrame with stock data sorted by date ascending
        """
        queryset = StockData.objects.filter(ticker=ticker.upper()).order_by('date')
        if limit:
            queryset = queryset[:limit]

        data = list(queryset.values(
            'date', 'open_price', 'high_price', 'low_price',
            'close_price', 'adj_close', 'volume'
        ))

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df

    @staticmethod
    def scrape_multiple_stocks(tickers: list, period: str = '2y', interval: str = '1d') -> list:
        """
        Scrape data for multiple stock tickers.

        Args:
            tickers: List of ticker symbols
            period: Data period
            interval: Data interval

        Returns:
            List of scrape results for each ticker
        """
        results = []
        for ticker in tickers:
            result = StockScraper.scrape_stock_data(ticker, period, interval)
            results.append(result)
        return results
