"""
API Views for Stock Price Prediction + News & Comment Sentiment Analysis.
"""
import logging
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from api.models import StockData, NewsArticle, PredictionResult, TrainedModel
from api.serializers import (
    StockDataSerializer, NewsArticleSerializer,
    ScrapeRequestSerializer, NewsScrapeRequestSerializer,
    TrainingRequestSerializer, PredictionRequestSerializer
)
from api.services.scraper import StockScraper
from api.services.news_scraper import NewsScraper
from api.services.sentiment_engine import SentimentEngine
from api.services.lstm_model import LSTMModelService
from api.services.combined_predictor import CombinedPredictorService

logger = logging.getLogger(__name__)


class StockDataView(APIView):
    """
    Fetch historical prices or trigger scraper for any stock ticker symbol.
    """

    def get(self, request, ticker=None):
        if not ticker:
            return Response({'error': 'Ticker parameter required'}, status=status.HTTP_400_BAD_REQUEST)
        
        ticker = ticker.upper().strip()
        period = request.query_params.get('period', '2y')
        
        # Check database records
        data = StockData.objects.filter(ticker=ticker).order_by('date')
        if not data.exists():
            logger.info(f"No existing data for {ticker}, fetching live market dataset...")
            scrape_res = StockScraper.scrape_stock_data(ticker, period=period)
            if scrape_res.get('status') == 'error':
                return Response(scrape_res, status=status.HTTP_400_BAD_REQUEST)
            data = StockData.objects.filter(ticker=ticker).order_by('date')

        serializer = StockDataSerializer(data, many=True)
        return Response({
            'status': 'success',
            'ticker': ticker,
            'count': data.count(),
            'data': serializer.data
        })

    def post(self, request):
        serializer = ScrapeRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        ticker = serializer.validated_data['ticker']
        period = serializer.validated_data['period']
        interval = serializer.validated_data['interval']
        
        result = StockScraper.scrape_stock_data(ticker, period, interval)
        return Response(result, status=status.HTTP_200_OK if result.get('status') == 'success' else status.HTTP_400_BAD_REQUEST)


class NewsView(APIView):
    """
    Fetch news or trigger news scraping for stock symbol.
    """

    def get(self, request, ticker=None):
        if not ticker:
            return Response({'error': 'Ticker parameter required'}, status=status.HTTP_400_BAD_REQUEST)

        ticker = ticker.upper().strip()
        sentiment_summary = SentimentEngine.get_ticker_sentiment(ticker)
        return Response(sentiment_summary)

    def post(self, request):
        serializer = NewsScrapeRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ticker = serializer.validated_data['ticker']
        max_articles = serializer.validated_data['max_articles']
        
        result = NewsScraper.scrape_all_sources(ticker, max_articles)
        return Response(result)


class AnalyzeCommentView(APIView):
    """
    MANUAL COMMENT / TEXT SENTIMENT ANALYZER
    Allows testing any custom comment/text to classify positive, negative, or neutral!
    """

    def post(self, request):
        text = request.data.get('text', '').strip()
        if not text:
            return Response({'error': 'Text/Comment is required for sentiment analysis.'}, status=status.HTTP_400_BAD_REQUEST)

        analysis = SentimentEngine.get_combined_sentiment(text)
        return Response({
            'status': 'success',
            'text': text,
            'sentiment_label': analysis['label'],
            'score': analysis['combined_score'],
            'vader_score': analysis['vader_score'],
            'textblob_score': analysis['textblob_score'],
            'details': analysis
        })


class TrainModelView(APIView):
    """
    Train dynamic LSTM/GRU model on real scraped dataset.
    """

    def post(self, request):
        serializer = TrainingRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ticker = serializer.validated_data['ticker'].upper().strip()
        
        # Ensure data exists first
        if StockData.objects.filter(ticker=ticker).count() < 50:
            StockScraper.scrape_stock_data(ticker, period='2y')

        result = LSTMModelService.train_model(
            ticker=ticker,
            epochs=serializer.validated_data['epochs'],
            batch_size=serializer.validated_data['batch_size'],
            sequence_length=serializer.validated_data['sequence_length'],
            train_split=serializer.validated_data['train_split'],
            model_type=serializer.validated_data['model_type'],
            include_sentiment=serializer.validated_data['include_sentiment']
        )
        return Response(result)


class PredictView(APIView):
    """
    Generate stock price forecast using trained LSTM model.
    """

    def post(self, request):
        serializer = PredictionRequestSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        ticker = serializer.validated_data['ticker'].upper().strip()
        days = serializer.validated_data['days']
        include_sentiment = serializer.validated_data['include_sentiment']

        result = LSTMModelService.predict_future(
            ticker=ticker,
            days=days,
            include_sentiment=include_sentiment
        )
        return Response(result)


class DashboardView(APIView):
    """
    Full aggregated dashboard view for any stock symbol.
    """

    def get(self, request, ticker='AAPL'):
        data = CombinedPredictorService.get_full_dashboard_data(ticker)
        return Response(data)
