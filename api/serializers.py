"""
DRF Serializers for Stock Prediction API.
"""
from rest_framework import serializers
from .models import StockData, NewsArticle, PredictionResult, TrainedModel


class StockDataSerializer(serializers.ModelSerializer):
    """Serializer for stock OHLCV data."""

    class Meta:
        model = StockData
        fields = '__all__'
        read_only_fields = ('created_at',)


class NewsArticleSerializer(serializers.ModelSerializer):
    """Serializer for news articles with sentiment."""

    class Meta:
        model = NewsArticle
        fields = '__all__'
        read_only_fields = ('created_at', 'sentiment_score', 'sentiment_label',
                            'vader_score', 'textblob_score')


class PredictionResultSerializer(serializers.ModelSerializer):
    """Serializer for prediction results."""

    class Meta:
        model = PredictionResult
        fields = '__all__'
        read_only_fields = ('prediction_date',)


class TrainedModelSerializer(serializers.ModelSerializer):
    """Serializer for trained model registry."""

    class Meta:
        model = TrainedModel
        fields = '__all__'
        read_only_fields = ('created_at', 'updated_at')


# --- Request/Response Serializers ---

class ScrapeRequestSerializer(serializers.Serializer):
    """Request serializer for stock data scraping."""

    ticker = serializers.CharField(max_length=10)
    period = serializers.CharField(max_length=10, default='2y')
    interval = serializers.CharField(max_length=5, default='1d')


class NewsScrapeRequestSerializer(serializers.Serializer):
    """Request serializer for news scraping."""

    ticker = serializers.CharField(max_length=10)
    max_articles = serializers.IntegerField(default=50, min_value=1, max_value=200)


class TrainingRequestSerializer(serializers.Serializer):
    """Request serializer for model training."""

    ticker = serializers.CharField(max_length=10)
    epochs = serializers.IntegerField(default=100, min_value=10, max_value=500)
    sequence_length = serializers.IntegerField(default=60, min_value=10, max_value=120)
    batch_size = serializers.IntegerField(default=32, min_value=8, max_value=128)
    train_split = serializers.FloatField(default=0.8, min_value=0.5, max_value=0.95)
    model_type = serializers.ChoiceField(
        choices=['lstm', 'gru', 'bilstm'],
        default='lstm'
    )
    include_sentiment = serializers.BooleanField(default=False)


class PredictionRequestSerializer(serializers.Serializer):
    """Request serializer for price prediction."""

    ticker = serializers.CharField(max_length=10)
    days = serializers.IntegerField(default=30, min_value=1, max_value=90)
    include_sentiment = serializers.BooleanField(default=False)


class SentimentResponseSerializer(serializers.Serializer):
    """Response serializer for sentiment analysis."""

    ticker = serializers.CharField()
    overall_sentiment = serializers.FloatField()
    sentiment_label = serializers.CharField()
    total_articles = serializers.IntegerField()
    bullish_count = serializers.IntegerField()
    neutral_count = serializers.IntegerField()
    bearish_count = serializers.IntegerField()
    articles = NewsArticleSerializer(many=True)
    sentiment_trend = serializers.ListField(child=serializers.DictField())


class DashboardResponseSerializer(serializers.Serializer):
    """Response serializer for combined dashboard data."""

    ticker = serializers.CharField()
    stock_data = StockDataSerializer(many=True)
    predictions = serializers.DictField()
    sentiment = serializers.DictField()
    metrics = serializers.DictField()
    model_info = serializers.DictField()
