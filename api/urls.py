"""
API URLs routing.
"""
from django.urls import path
from api.views import (
    StockDataView, NewsView, AnalyzeCommentView,
    TrainModelView, PredictView, DashboardView
)

urlpatterns = [
    path('stocks/', StockDataView.as_view(), name='stock-list-scrape'),
    path('stocks/<str:ticker>/', StockDataView.as_view(), name='stock-detail'),
    path('news/', NewsView.as_view(), name='news-scrape'),
    path('news/<str:ticker>/', NewsView.as_view(), name='news-detail'),
    path('analyze-comment/', AnalyzeCommentView.as_view(), name='analyze-comment'),
    path('model/train/', TrainModelView.as_view(), name='model-train'),
    path('model/predict/', PredictView.as_view(), name='model-predict'),
    path('dashboard/<str:ticker>/', DashboardView.as_view(), name='dashboard'),
]
