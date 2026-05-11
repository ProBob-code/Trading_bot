"""
Sentiment Analysis Module
=========================

Provides news sentiment analysis for trading decisions.
"""

from .news_sentiment import (
    NewsSentimentAnalyzer,
    SentimentResult,
    NewsItem,
    get_sentiment_analyzer,
)

__all__ = [
    'NewsSentimentAnalyzer',
    'SentimentResult', 
    'NewsItem',
    'get_sentiment_analyzer',
]
