"""
Combined Predictor Service.
Combines stock price forecasting with news sentiment analysis.
Optimized for high performance and dynamic sequence window length.
"""
import logging
from typing import Dict, Any

from api.services.scraper import StockScraper
from api.services.news_scraper import NewsScraper
from api.services.sentiment_engine import SentimentEngine
from api.services.lstm_model import LSTMModelService
from api.models import StockData, NewsArticle, TrainedModel

logger = logging.getLogger(__name__)


class CombinedPredictorService:
    """
    Integrates stock price scraping, news scraping, sentiment evaluation,
    and sequence modeling into unified workflow functions.
    """

    @staticmethod
    def get_full_dashboard_data(ticker: str) -> Dict[str, Any]:
        ticker = ticker.upper().strip()

        # 1. Scrape if stock has no data
        stock_count = StockData.objects.filter(ticker=ticker).count()
        if stock_count == 0:
            logger.info(f"Fetching real market dataset for new ticker: {ticker}")
            scrape_res = StockScraper.scrape_stock_data(ticker, period='2y')
            if scrape_res.get('status') == 'error':
                return {
                    'status': 'error',
                    'message': scrape_res.get('message', f'Could not fetch market data for symbol: {ticker}. Check ticker name.')
                }

        # Fetch recent historical stock records
        stock_df = StockScraper.get_stock_dataframe(ticker, limit=180)
        stock_history = []
        if not stock_df.empty:
            for _, row in stock_df.iterrows():
                stock_history.append({
                    'date': row['date'].strftime('%Y-%m-%d'),
                    'open': float(row['open_price']),
                    'high': float(row['high_price']),
                    'low': float(row['low_price']),
                    'close': float(row['close_price']),
                    'volume': int(row['volume']),
                })

        # 2. Scrape news articles if not already scraped
        news_count = NewsArticle.objects.filter(ticker=ticker).count()
        if news_count == 0:
            logger.info(f"Scraping news dataset for ticker: {ticker}")
            try:
                NewsScraper.scrape_all_sources(ticker, max_articles=20)
            except Exception as news_err:
                logger.warning(f"News scraping warning for {ticker}: {news_err}")

        sentiment_data = SentimentEngine.get_ticker_sentiment(ticker)

        # 3. Check for active model or train auto baseline
        active_model = TrainedModel.objects.filter(ticker=ticker, is_active=True).first()
        if not active_model:
            seq_len = min(30, max(5, len(stock_history) - 5))
            logger.info(f"Auto-training initial sequence model for {ticker} (seq_len={seq_len})...")
            LSTMModelService.train_model(
                ticker=ticker,
                epochs=15,
                sequence_length=seq_len,
                model_type='lstm',
                include_sentiment=True
            )

        # 4. Generate predictions
        pred_res = LSTMModelService.predict_future(
            ticker=ticker,
            days=30,
            include_sentiment=True
        )

        metrics_res = LSTMModelService.get_model_metrics(ticker)

        return {
            'status': 'success',
            'ticker': ticker,
            'historical_prices': stock_history,
            'predictions': pred_res.get('predictions', []) if pred_res.get('status') == 'success' else [],
            'sentiment': sentiment_data,
            'metrics': metrics_res.get('latest_metrics', {}) if metrics_res.get('status') == 'success' else {},
            'model_info': metrics_res if metrics_res.get('status') == 'success' else {},
        }
