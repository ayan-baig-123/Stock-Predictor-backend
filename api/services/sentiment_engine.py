"""
Sentiment Analysis Engine using NLTK VADER + TextBlob.
Supports multilingual text (Urdu, Roman Urdu, English) via auto-translation.
Analyzes financial news text for market sentiment.
"""
import logging
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


def _translate_to_english(text: str) -> str:
    """
    Detect language and translate non-English text to English.
    Supports Urdu (ur), Roman Urdu, Hindi (hi), Arabic (ar), and others.
    Falls back to original text if translation fails.
    """
    if not text or not text.strip():
        return text

    try:
        from langdetect import detect, LangDetectException
        try:
            lang = detect(text)
        except LangDetectException:
            lang = 'en'

        # If already English, return as-is
        if lang == 'en':
            return text

        # Translate to English using deep-translator (Google Translate backend)
        from deep_translator import GoogleTranslator
        translated = GoogleTranslator(source='auto', target='en').translate(text)
        logger.debug(f"Translated [{lang}] → [en]: '{text[:60]}' → '{translated[:60]}'")
        return translated if translated else text

    except Exception as e:
        logger.warning(f"Translation failed, using original text. Error: {e}")
        return text


class SentimentEngine:
    """
    Sentiment analysis engine combining VADER and TextBlob
    for robust financial text analysis.
    Supports Urdu, Roman Urdu, Hindi, Arabic via auto-translation.
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
        Auto-translates non-English text to English before analysis.

        Args:
            text: Text to analyze (any language)

        Returns:
            dict with compound, pos, neg, neu scores
        """
        if not text or not text.strip():
            return {'compound': 0.0, 'pos': 0.0, 'neg': 0.0, 'neu': 1.0}

        # Translate to English if needed
        english_text = _translate_to_english(text)

        vader = SentimentEngine._get_vader()
        scores = vader.polarity_scores(english_text)

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
        Auto-translates non-English text to English before analysis.

        Args:
            text: Text to analyze (any language)

        Returns:
            dict with polarity and subjectivity scores
        """
        if not text or not text.strip():
            return {'polarity': 0.0, 'subjectivity': 0.0}

        # Translate to English if needed
        english_text = _translate_to_english(text)

        blob = TextBlob(english_text)
        return {
            'polarity': blob.sentiment.polarity,
            'subjectivity': blob.sentiment.subjectivity,
        }

    @staticmethod
    def get_combined_sentiment(text: str) -> dict:
        """
        Get combined sentiment from both VADER and TextBlob.
        Supports Urdu, Roman Urdu, Hindi, Arabic via auto-translation.

        Uses weighted average: 60% VADER + 40% TextBlob
        (VADER is better for short news headlines).

        Args:
            text: Text to analyze (any language)

        Returns:
            dict with combined score, label, and individual scores
        """
        if not text or not text.strip():
            return {
                'combined_score': 0.0,
                'label': 'neutral',
                'vader_score': 0.0,
                'textblob_score': 0.0,
                'translated_text': '',
            }

        # Translate once, reuse for both engines
        english_text = _translate_to_english(text)

        vader_result = SentimentEngine.analyze_sentiment_vader(english_text)
        textblob_result = SentimentEngine.analyze_sentiment_textblob(english_text)

        # Weighted combination (VADER weighted higher for news headlines)
        vader_score = vader_result['compound']
        textblob_score = textblob_result['polarity']
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
            'translated_text': english_text if english_text != text else None,
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
