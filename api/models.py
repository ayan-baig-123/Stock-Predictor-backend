"""
Database models for Stock Price Prediction + Sentiment Analysis.
"""
from django.db import models


class StockData(models.Model):
    """Stores scraped OHLCV stock price data."""

    ticker = models.CharField(max_length=10, db_index=True)
    date = models.DateField(db_index=True)
    open_price = models.FloatField()
    high_price = models.FloatField()
    low_price = models.FloatField()
    close_price = models.FloatField()
    adj_close = models.FloatField()
    volume = models.BigIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-date']
        unique_together = ['ticker', 'date']
        verbose_name = 'Stock Data'
        verbose_name_plural = 'Stock Data'
        indexes = [
            models.Index(fields=['ticker', 'date']),
        ]

    def __str__(self):
        return f"{self.ticker} - {self.date} - Close: {self.close_price}"


class NewsArticle(models.Model):
    """Stores scraped news articles with sentiment scores."""

    SENTIMENT_CHOICES = [
        ('bullish', 'Bullish'),
        ('neutral', 'Neutral'),
        ('bearish', 'Bearish'),
    ]

    ticker = models.CharField(max_length=10, db_index=True)
    title = models.CharField(max_length=500)
    source = models.CharField(max_length=200, blank=True, default='')
    url = models.URLField(max_length=1000, blank=True, default='')
    published_date = models.DateTimeField(null=True, blank=True)
    content = models.TextField(blank=True, default='')
    sentiment_score = models.FloatField(default=0.0)
    sentiment_label = models.CharField(
        max_length=10,
        choices=SENTIMENT_CHOICES,
        default='neutral'
    )
    vader_score = models.FloatField(default=0.0)
    textblob_score = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-published_date']
        verbose_name = 'News Article'
        verbose_name_plural = 'News Articles'

    def __str__(self):
        return f"[{self.sentiment_label}] {self.ticker}: {self.title[:80]}"


class PredictionResult(models.Model):
    """Stores LSTM prediction results and metrics."""

    ticker = models.CharField(max_length=10, db_index=True)
    prediction_date = models.DateTimeField(auto_now_add=True)
    predicted_prices = models.JSONField(default=list)
    actual_prices = models.JSONField(default=list)
    prediction_dates = models.JSONField(default=list)
    days_predicted = models.IntegerField(default=30)
    include_sentiment = models.BooleanField(default=False)

    # Model performance metrics
    rmse = models.FloatField(null=True, blank=True)
    mae = models.FloatField(null=True, blank=True)
    mape = models.FloatField(null=True, blank=True)
    r2_score = models.FloatField(null=True, blank=True)
    directional_accuracy = models.FloatField(null=True, blank=True)

    model_version = models.CharField(max_length=50, default='v1.0')

    class Meta:
        ordering = ['-prediction_date']
        verbose_name = 'Prediction Result'
        verbose_name_plural = 'Prediction Results'

    def __str__(self):
        return f"{self.ticker} - {self.prediction_date} - RMSE: {self.rmse}"


class TrainedModel(models.Model):
    """Registry for trained LSTM models."""

    MODEL_TYPE_CHOICES = [
        ('lstm', 'LSTM'),
        ('gru', 'GRU'),
        ('bilstm', 'Bidirectional LSTM'),
    ]

    ticker = models.CharField(max_length=10, db_index=True)
    model_type = models.CharField(
        max_length=10,
        choices=MODEL_TYPE_CHOICES,
        default='lstm'
    )
    model_path = models.CharField(max_length=500)
    scaler_path = models.CharField(max_length=500)
    sequence_length = models.IntegerField(default=60)
    epochs_trained = models.IntegerField(default=100)
    batch_size = models.IntegerField(default=32)
    features_used = models.JSONField(default=list)
    loss_history = models.JSONField(default=list)
    val_loss_history = models.JSONField(default=list)
    final_train_loss = models.FloatField(null=True, blank=True)
    final_val_loss = models.FloatField(null=True, blank=True)
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-created_at']
        verbose_name = 'Trained Model'
        verbose_name_plural = 'Trained Models'

    def __str__(self):
        return f"{self.ticker} - {self.model_type} - {self.created_at.strftime('%Y-%m-%d')}"
