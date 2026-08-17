from django.contrib import admin
from .models import StockData, NewsArticle, PredictionResult, TrainedModel


@admin.register(StockData)
class StockDataAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'date', 'close_price', 'volume')
    list_filter = ('ticker',)
    search_fields = ('ticker',)
    ordering = ('-date',)


@admin.register(NewsArticle)
class NewsArticleAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'title', 'sentiment_label', 'sentiment_score', 'published_date')
    list_filter = ('ticker', 'sentiment_label')
    search_fields = ('ticker', 'title')
    ordering = ('-published_date',)


@admin.register(PredictionResult)
class PredictionResultAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'prediction_date', 'days_predicted', 'rmse', 'mae', 'r2_score')
    list_filter = ('ticker',)
    ordering = ('-prediction_date',)


@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'model_type', 'epochs_trained', 'is_active', 'created_at')
    list_filter = ('ticker', 'model_type', 'is_active')
    ordering = ('-created_at',)
