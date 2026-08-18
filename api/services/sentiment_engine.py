"""
Sentiment Analysis Engine using NLTK VADER + TextBlob.
Supports Urdu & Roman Urdu via built-in keyword dictionary (zero extra memory).
Analyzes financial news text for market sentiment.
"""
import logging
import re
from typing import Optional

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob

from api.models import NewsArticle

logger = logging.getLogger(__name__)

# Download NLTK data on module load
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


# ──────────────────────────────────────────────────────────────
# Built-in Urdu & Roman Urdu Financial Keyword Dictionary
# No extra libraries needed — zero memory overhead
# ──────────────────────────────────────────────────────────────
URDU_POSITIVE_KEYWORDS = [
    # Roman Urdu
    'munafa', 'faida', 'izafa', 'barh', 'achha', 'behtareen', 'shandar',
    'kamyab', 'kamyabi', 'mazboot', 'buland', 'upar', 'tezi', 'zabardast',
    'umeed', 'bharosa', 'kharido', 'invest', 'khushi', 'behtari', 'kamai',
    'grow', 'growth', 'profit', 'acha', 'accha', 'best', 'great',
    'positive', 'stable', 'recover', 'recovery', 'strong', 'boom',
    'surge', 'rally', 'up', 'high', 'record', 'dividend',
    # Urdu script
    'منافع', 'فائدہ', 'اضافہ', 'بڑھ', 'اچھا', 'بہترین', 'شاندار',
    'کامیاب', 'کامیابی', 'مضبوط', 'بلند', 'اوپر', 'تیزی', 'زبردست',
    'امید', 'بھروسہ', 'خریدو', 'خوشی', 'بہتری', 'کمائی',
]

URDU_NEGATIVE_KEYWORDS = [
    # Roman Urdu
    'nuqsan', 'gir', 'gira', 'girr', 'kami', 'kharab', 'bura', 'girawat',
    'dhoka', 'khatarnak', 'band', 'tabahi', 'barbadi', 'kamzor', 'neeche',
    'mandi', 'sell', 'becho', 'fikar', 'pareshani', 'loss', 'danger',
    'crash', 'problem', 'risk', 'weak', 'low', 'down', 'fail', 'worst',
    'negative', 'decline', 'fall', 'drop', 'dump', 'recession',
    # Urdu script
    'نقصان', 'گر', 'گرا', 'کمی', 'خراب', 'برا', 'گراوٹ',
    'دھوکا', 'خطرناک', 'بند', 'تباہی', 'بربادی', 'کمزور', 'نیچے',
    'مندی', 'بیچو', 'فکر', 'پریشانی',
]


def _urdu_keyword_score(text: str) -> float:
    """
    Score text using built-in Urdu/Roman Urdu keyword dictionary.
    Returns a score between -1.0 and 1.0.
    Zero extra memory — just a simple word lookup.
    """
    if not text:
        return 0.0

    text_lower = text.lower().strip()
    words = re.split(r'[\s,.\-!?؟،۔]+', text_lower)

    pos_hits = sum(1 for w in words if w in URDU_POSITIVE_KEYWORDS)
    neg_hits = sum(1 for w in words if w in URDU_NEGATIVE_KEYWORDS)

    total_hits = pos_hits + neg_hits
    if total_hits == 0:
        return 0.0

    # Score = (positive - negative) / total, bounded [-1, 1]
    return (pos_hits - neg_hits) / total_hits


def _has_urdu_chars(text: str) -> bool:
    """Check if text contains Urdu/Arabic script characters."""
    return bool(re.search(r'[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]', text))


def _has_roman_urdu(text: str) -> bool:
    """Check if text contains common Roman Urdu financial keywords."""
    text_lower = text.lower()
    roman_urdu_markers = [
        'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'mein', 'se', 'par',
        'ho', 'nahi', 'bohot', 'bahut', 'zyada', 'kam', 'acha', 'bura',
        'karo', 'karna', 'raha', 'rahi', 'gaya', 'gayi', 'wala',
        'munafa', 'nuqsan', 'tezi', 'mandi', 'girawat', 'kharido', 'becho',
    ]
    words = set(re.split(r'[\s,.\-!?]+', text_lower))
    matches = sum(1 for m in roman_urdu_markers if m in words)
    return matches >= 2  # At least 2 markers = likely Roman Urdu


class SentimentEngine:
    """
    Sentiment analysis engine combining VADER and TextBlob
    for robust financial text analysis.
    Supports Urdu & Roman Urdu via built-in keyword dictionary.
    """

    # VADER instance (reused across calls)
    _vader = None

    # Sentiment thresholds
    BULLISH_THRESHOLD = 0.15
    BEARISH_THRESHOLD = -0.15

    # Custom financial lexicon additions for VADER
    FINANCIAL_LEXICON = {
        'bullish': 3.0,
        'bearish': -3.0,
        'upgrade': 2.5,
        'downgrade': -2.5,
        'outperform': 2.0,
        'underperform': -2.0,
        'buy': 2.0,
        'sell': -2.0,
        'overweight': 1.5,
        'underweight': -1.5,
        'rally': 2.5,
        'crash': -3.5,
        'surge': 2.5,
        'plunge': -3.0,
        'soar': 2.5,
        'tumble': -2.5,
        'boom': 2.5,
        'bust': -2.5,
        'profit': 2.0,
        'loss': -2.0,
        'growth': 1.5,
        'decline': -1.5,
        'beat': 2.0,
        'miss': -2.0,
        'exceed': 2.0,
        'disappoint': -2.0,
        'optimistic': 2.0,
        'pessimistic': -2.0,
        'strong': 1.5,
        'weak': -1.5,
        'record': 1.5,
        'bankruptcy': -4.0,
        'dividend': 1.5,
        'acquisition': 1.0,
        'layoff': -2.0,
        'lawsuit': -1.5,
        'innovation': 1.5,
        'recession': -3.0,
        'recovery': 2.0,
        'inflation': -1.0,
        'volatile': -1.0,
    }

    @classmethod
    def _get_vader(cls) -> SentimentIntensityAnalyzer:
        """Get or create VADER analyzer instance with financial lexicon."""
        if cls._vader is None:
            cls._vader = SentimentIntensityAnalyzer()
            # Add financial terms to VADER lexicon
            cls._vader.lexicon.update(cls.FINANCIAL_LEXICON)
        return cls._vader

    @staticmethod
    def analyze_sentiment_vader(text: str) -> dict:
        """
        Analyze sentiment using NLTK VADER.

        VADER is specifically designed for social media / news text
        and returns compound score in [-1, 1].

        Args:
            text: Text to analyze

        Returns:
            dict with compound, pos, neg, neu scores
        """
        if not text or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

        vader = SentimentEngine._get_vader()
        scores = vader.polarity_scores(text)

        return {
            'compound': scores['compound'],
            'pos': scores['pos'],
            'neg': scores['neg'],
            'neu': scores['neu'],
        }

    @staticmethod
    def analyze_sentiment_textblob(text: str) -> dict:
        """
        Analyze sentiment using TextBlob.

        TextBlob returns polarity [-1, 1] and subjectivity [0, 1].

        Args:
            text: Text to analyze

        Returns:
            dict with polarity and subjectivity scores
        """
        if not text or not text.strip():
            return {'polarity': 0.0, 'subjectivity': 0.0}

        blob = TextBlob(text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
        }

    @staticmethod
    def get_combined_sentiment(text: str) -> dict:
        """
        Get combined sentiment from VADER, TextBlob, and Urdu keyword dictionary.
        Supports English, Urdu script, and Roman Urdu — zero extra libraries.

        Uses weighted average: 60% VADER + 40% TextBlob for English.
        For Urdu/Roman Urdu: blends keyword score with VADER/TextBlob.

        Args:
            text: Text to analyze (English, Urdu, or Roman Urdu)

        Returns:
            dict with combined score, label, and individual scores
        """
        if not text or not text.strip():
            return {
                'combined_score': 0.0,
                'label': 'neutral',
                'vader_score': 0.0,
                'textblob_score': 0.0,
            }

        vader_result = SentimentEngine.analyze_sentiment_vader(text)
        textblob_result = SentimentEngine.analyze_sentiment_textblob(text)

        vader_score = vader_result['compound']
        textblob_score = textblob_result['polarity']

        # Check if text is Urdu script or Roman Urdu
        is_urdu = _has_urdu_chars(text) or _has_roman_urdu(text)

        if is_urdu:
            # Blend: 40% VADER + 20% TextBlob + 40% Urdu keyword dictionary
            urdu_score = _urdu_keyword_score(text)
            combined_score = (0.4 * vader_score) + (0.2 * textblob_score) + (0.4 * urdu_score)
        else:
            # English: 60% VADER + 40% TextBlob (original formula)
            combined_score = (0.6 * vader_score) + (0.4 * textblob_score)

        # Determine label
        if combined_score > SentimentEngine.BULLISH_THRESHOLD:
            label = 'bullish'
        elif combined_score < SentimentEngine.BEARISH_THRESHOLD:
            label = 'bearish'
        else:
            label = 'neutral'

        return {
            'combined_score': round(combined_score, 4),
            'label': label,
            'vader_score': round(vader_score, 4),
            'textblob_score': round(textblob_score, 4),
            'vader_detail': vader_result,
            'textblob_detail': textblob_result,
        }

    @staticmethod
    def get_ticker_sentiment(ticker: str) -> dict:
        """
        Get aggregated sentiment analysis for a stock ticker.

        Calculates overall sentiment from all stored news articles.

        Args:
            ticker: Stock ticker symbol

        Returns:
            dict with overall sentiment, counts by category, and article details
        """
        ticker = ticker.upper().strip()
        articles = NewsArticle.objects.filter(ticker=ticker).order_by('-published_date')

        if not articles.exists():
            return {
                'ticker': ticker,
                'overall_sentiment': 0.0,
                'sentiment_label': 'neutral',
                'total_articles': 0,
                'bullish_count': 0,
                'neutral_count': 0,
                'bearish_count': 0,
                'articles': [],
                'sentiment_trend': [],
            }

        # Calculate aggregate scores
        total = articles.count()
        bullish = articles.filter(sentiment_label='bullish').count()
        neutral = articles.filter(sentiment_label='neutral').count()
        bearish = articles.filter(sentiment_label='bearish').count()

        # Weighted average of sentiment scores
        scores = list(articles.values_list('sentiment_score', flat=True))
        overall = sum(scores) / len(scores) if scores else 0.0

        # Determine overall label — pure majority vote
        # Whichever category has the MOST articles wins the badge
        counts = {'bullish': bullish, 'neutral': neutral, 'bearish': bearish}
        overall_label = max(counts, key=counts.get)

        # Build sentiment trend (by date)
        from django.db.models import Avg
        from django.db.models.functions import TruncDate

        trend = (
            articles
            .annotate(day=TruncDate('published_date'))
            .values('day')
            .annotate(avg_sentiment=Avg('sentiment_score'))
            .order_by('day')
        )
        sentiment_trend = [
            {
                'date': str(entry['day']) if entry['day'] else None,
                'sentiment': round(entry['avg_sentiment'], 4) if entry['avg_sentiment'] else 0,
            }
            for entry in trend
            if entry['day'] is not None
        ]

        # Serialize articles
        article_data = []
        for article in articles[:50]:  # Limit to 50 most recent
            article_data.append({
                'id': article.id,
                'title': article.title,
                'source': article.source,
                'url': article.url,
                'published_date': article.published_date.isoformat() if article.published_date else None,
                'sentiment_score': article.sentiment_score,
                'sentiment_label': article.sentiment_label,
                'vader_score': article.vader_score,
                'textblob_score': article.textblob_score,
            })

        return {
            'ticker': ticker,
            'overall_sentiment': round(overall, 4),
            'sentiment_label': overall_label,
            'total_articles': total,
            'bullish_count': bullish,
            'neutral_count': neutral,
            'bearish_count': bearish,
            'bullish_pct': round(bullish / total * 100, 1) if total > 0 else 0,
            'neutral_pct': round(neutral / total * 100, 1) if total > 0 else 0,
            'bearish_pct': round(bearish / total * 100, 1) if total > 0 else 0,
            'articles': article_data,
            'sentiment_trend': sentiment_trend,
        }
