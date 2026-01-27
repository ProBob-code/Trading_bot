"""
News Sentiment Analysis Module
==============================

Analyzes financial news headlines for market sentiment to enhance trading decisions.
Uses keyword-based sentiment scoring and provides aggregated sentiment per symbol.
Now includes LIVE news fetching from RSS feeds.
"""

import csv
import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class NewsItem:
    """Single news item with sentiment."""
    url: str
    title: str
    timestamp: Optional[datetime]
    sentiment_score: float  # -1.0 to +1.0
    related_symbols: List[str]
    source: str


@dataclass 
class SentimentResult:
    """Aggregated sentiment result for a symbol."""
    symbol: str
    score: float  # -1.0 to +1.0  
    label: str  # 'BULLISH', 'BEARISH', 'NEUTRAL'
    confidence: float  # 0.0 to 1.0
    news_count: int
    top_headlines: List[str]


# Sentiment keywords for financial news analysis
BULLISH_KEYWORDS = [
    # Strong bullish
    'surge', 'soar', 'rally', 'skyrocket', 'breakout', 'all-time high', 'ath',
    'moon', 'bullish', 'outperform', 'upgrade', 'buy', 'strong buy', 'accumulate',
    'beat expectations', 'beats estimate', 'profit jumps', 'profit rises', 'net profit up',
    'revenue up', 'revenue grows', 'sales surge', 'margin improves', 'eps beat',
    # Moderate bullish
    'gain', 'rise', 'up', 'higher', 'positive', 'growth', 'expand', 'recover',
    'boost', 'lift', 'advance', 'climb', 'strength', 'optimism', 'upbeat',
    'dividend', 'buyback', 'investment', 'expansion', 'partnership',
    # Sector specific
    'top gainer', 'new high', '52-week high', 'market high', 'record',
]

BEARISH_KEYWORDS = [
    # Strong bearish  
    'crash', 'plunge', 'collapse', 'tank', 'sink', 'tumble', 'freefall',
    'bearish', 'downgrade', 'sell', 'underperform', 'avoid', 'cut',
    'miss expectations', 'misses estimate', 'profit falls', 'profit drops', 'net profit down',
    'revenue down', 'revenue falls', 'sales drop', 'margin pressure', 'eps miss',
    'layoff', 'layoffs', 'job cuts', 'restructuring',
    # Moderate bearish
    'fall', 'drop', 'down', 'lower', 'decline', 'slump', 'weak', 'concern',
    'risk', 'warning', 'uncertainty', 'pressure', 'headwind', 'challenge',
    'loss', 'losses', 'deficit', 'debt', 'litigation', 'lawsuit',
    # Sector specific
    'top loser', 'new low', '52-week low', 'market low', 'correction',
    'tariff', 'sanction', 'ban', 'restriction',
]

# Stock ticker to company name mapping for better matching
SYMBOL_MAPPINGS = {
    # Crypto
    'BTCUSDT': ['bitcoin', 'btc', 'crypto'],
    'ETHUSDT': ['ethereum', 'eth', 'crypto'],
    'BNBUSDT': ['binance', 'bnb', 'crypto'],
    'SOLUSDT': ['solana', 'sol', 'crypto'],
    'XRPUSDT': ['ripple', 'xrp', 'crypto'],
    # US Stocks
    'AAPL': ['apple', 'aapl', 'iphone', 'ipad', 'mac'],
    'TSLA': ['tesla', 'tsla', 'elon', 'musk', 'ev', 'electric vehicle'],
    'GOOGL': ['google', 'alphabet', 'googl', 'goog', 'android', 'youtube'],
    'MSFT': ['microsoft', 'msft', 'azure', 'windows', 'office', 'xbox'],
    'AMZN': ['amazon', 'amzn', 'aws', 'prime'],
    # Indian Stocks (from the CSV data)
    'TCS': ['tcs', 'tata consultancy', 'tata tech'],
    'INFY': ['infosys', 'infy'],
    'RELIANCE': ['reliance', 'ril', 'jio'],
    'HDFCBANK': ['hdfc bank', 'hdfc'],
    'ICICIBANK': ['icici bank', 'icici'],
    'TATASTEEL': ['tata steel', 'tatasteel'],
    'TATAMOTORS': ['tata motors', 'tatamotors'],
    'WIPRO': ['wipro'],
    'TITAN': ['titan'],
    'NTPC': ['ntpc', 'power grid'],
    'ONGC': ['ongc', 'oil india'],
    'CIPLA': ['cipla'],
    'LT': ['l&t', 'larsen', 'larsen toubro'],
    'SBIN': ['sbi', 'state bank'],
    'ASIANPAINT': ['asian paints', 'asian paint'],
    'HCLTECH': ['hcl tech', 'hcltech'],
    'TECHM': ['tech mahindra', 'techm'],
    'MARUTI': ['maruti', 'maruti suzuki'],
    'M&M': ['mahindra', 'm&m'],
    'BRITANNIA': ['britannia'],
}

# RSS Feed URLs for LIVE news
LIVE_NEWS_FEEDS = {
    'crypto': [
        'https://cointelegraph.com/rss',
        'https://www.newsbtc.com/feed/',
    ],
    'stocks': [
        'https://feeds.finance.yahoo.com/rss/2.0/headline?s=AAPL,TSLA,GOOGL,MSFT,AMZN&region=US&lang=en-US',
    ]
}


class NewsSentimentAnalyzer:
    """Analyzes news headlines for market sentiment."""
    
    def __init__(self, data_dir: str = None):
        """Initialize with path to news data files."""
        if data_dir is None:
            # Default to project root directory
            data_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        self.data_dir = data_dir
        self.news_cache: List[NewsItem] = []
        self.last_load_time: Optional[datetime] = None
        self.cache_ttl = timedelta(minutes=5)
        
        logger.info(f"📰 NewsSentimentAnalyzer initialized with data_dir: {data_dir}")
    
    def _load_news_data(self, force_reload: bool = False) -> List[NewsItem]:
        """Load news from CSV files."""
        if not force_reload and self.news_cache and self.last_load_time:
            if datetime.now() - self.last_load_time < self.cache_ttl:
                return self.news_cache
        
        news_items = []
        
        # Load from posted_headlines_perplexity.csv
        perplexity_file = os.path.join(self.data_dir, 'posted_headlines_perplexity.csv')
        if os.path.exists(perplexity_file):
            try:
                with open(perplexity_file, 'r', encoding='utf-8') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and row[0].startswith('http'):
                            url = row[0]
                            timestamp = None
                            if len(row) > 1 and row[1]:
                                try:
                                    # Try parsing various date formats
                                    timestamp = self._parse_timestamp(row[1])
                                except:
                                    pass
                            
                            # Extract title from URL
                            title = self._extract_title_from_url(url)
                            
                            # Calculate sentiment
                            sentiment = self._calculate_headline_sentiment(title)
                            
                            # Find related symbols
                            symbols = self._find_related_symbols(title)
                            
                            # Determine source
                            source = self._extract_source(url)
                            
                            news_items.append(NewsItem(
                                url=url,
                                title=title,
                                timestamp=timestamp,
                                sentiment_score=sentiment,
                                related_symbols=symbols,
                                source=source
                            ))
            except Exception as e:
                logger.error(f"Error loading news from {perplexity_file}: {e}")
        
        self.news_cache = news_items
        self.last_load_time = datetime.now()
        logger.info(f"📰 Loaded {len(news_items)} news items")
        
        return news_items
    
    def _parse_timestamp(self, ts_str: str) -> Optional[datetime]:
        """Parse timestamp from various formats."""
        formats = [
            '%Y-%m-%d %H:%M',
            '%Y-%m-%dT%H:%M:%SZ',
            '%a, %d %b %Y %H:%M:%S %z',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts_str.strip(), fmt)
            except:
                continue
        return None
    
    def _extract_title_from_url(self, url: str) -> str:
        """Extract a readable title from URL path."""
        # Get the last path segment
        path = url.split('/')[-1]
        
        # Remove query params and extensions
        path = path.split('?')[0].split('.')[0]
        
        # Replace dashes/underscores with spaces
        title = re.sub(r'[-_]', ' ', path)
        
        # Remove numeric IDs at the end
        title = re.sub(r'\d{6,}$', '', title)
        
        return title.strip() or url
    
    def _extract_source(self, url: str) -> str:
        """Extract source name from URL domain."""
        if 'moneycontrol' in url:
            return 'MoneyControl'
        elif 'economictimes' in url:
            return 'Economic Times'
        elif 'yahoo' in url:
            return 'Yahoo Finance'
        elif 'thehindubusinessline' in url:
            return 'Hindu Business Line'
        elif 'ndtvprofit' in url:
            return 'NDTV Profit'
        elif 'livemint' in url:
            return 'LiveMint'
        else:
            return 'News'
    
    def _calculate_headline_sentiment(self, text: str) -> float:
        """Calculate sentiment score from headline text."""
        text_lower = text.lower()
        
        bullish_count = 0
        bearish_count = 0
        
        for keyword in BULLISH_KEYWORDS:
            if keyword in text_lower:
                bullish_count += 1
        
        for keyword in BEARISH_KEYWORDS:
            if keyword in text_lower:
                bearish_count += 1
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        # Score from -1 (very bearish) to +1 (very bullish)
        score = (bullish_count - bearish_count) / max(total, 1)
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, score))
    
    def _find_related_symbols(self, text: str) -> List[str]:
        """Find stock/crypto symbols mentioned in text."""
        text_lower = text.lower()
        related = []
        
        for symbol, keywords in SYMBOL_MAPPINGS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    related.append(symbol)
                    break
        
        return list(set(related))
    
    def get_sentiment_for_symbol(self, symbol: str) -> SentimentResult:
        """Get aggregated sentiment for a specific symbol."""
        news_items = self._load_news_data()
        
        # Normalize symbol
        symbol_upper = symbol.upper().replace('/', '')
        
        # Filter news related to this symbol
        relevant_news = [
            n for n in news_items 
            if symbol_upper in n.related_symbols or 
               any(kw in n.title.lower() for kw in SYMBOL_MAPPINGS.get(symbol_upper, [symbol_upper.lower()]))
        ]
        
        if not relevant_news:
            # Return neutral if no relevant news
            return SentimentResult(
                symbol=symbol,
                score=0.0,
                label='NEUTRAL',
                confidence=0.0,
                news_count=0,
                top_headlines=[]
            )
        
        # Calculate weighted average sentiment
        total_score = sum(n.sentiment_score for n in relevant_news)
        avg_score = total_score / len(relevant_news)
        
        # Determine label
        if avg_score > 0.2:
            label = 'BULLISH'
        elif avg_score < -0.2:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'
        
        # Confidence based on number of news and score magnitude
        confidence = min(1.0, len(relevant_news) / 10) * abs(avg_score)
        
        # Get top headlines (sorted by recency if timestamp available, then by sentiment strength)
        sorted_news = sorted(
            relevant_news,
            key=lambda n: (n.timestamp or datetime.min, abs(n.sentiment_score)),
            reverse=True
        )[:5]
        
        return SentimentResult(
            symbol=symbol,
            score=avg_score,
            label=label,
            confidence=confidence,
            news_count=len(relevant_news),
            top_headlines=[n.title for n in sorted_news]
        )
    
    def get_market_sentiment(self) -> Dict[str, SentimentResult]:
        """Get sentiment for all tracked symbols."""
        results = {}
        
        for symbol in SYMBOL_MAPPINGS.keys():
            results[symbol] = self.get_sentiment_for_symbol(symbol)
        
        return results
    
    def get_recent_news(self, limit: int = 20, symbol: Optional[str] = None, market: Optional[str] = None) -> List[Dict]:
        """Get recent news items for display, optionally filtered by market type."""
        # First try to fetch LIVE news
        live_news = self._fetch_live_news(market or 'crypto')
        
        # Also load cached CSV news
        csv_news = self._load_news_data()
        
        # Combine live and cached, prioritizing live
        all_news = live_news + csv_news
        logger.debug(f"📰 get_recent_news: Live={len(live_news)}, CSV={len(csv_news)}, market={market}")
        
        news_items = all_news
        
        # Filter by market type
        if market:
            crypto_keywords = ['bitcoin', 'btc', 'ethereum', 'eth', 'crypto', 'binance', 'solana', 'ripple', 'xrp', 'blockchain', 'defi', 'nft']
            stock_keywords = ['stock', 'shares', 'nasdaq', 'nyse', 'earnings', 'quarterly', 'dividend', 'ipo', 'market cap', 'apple', 'tesla', 'google', 'microsoft', 'amazon']
            
            if market == 'crypto':
                news_items = [n for n in all_news if any(kw in n.title.lower() for kw in crypto_keywords) or 'crypto' in n.source.lower() or 'coin' in n.source.lower()]
            elif market == 'stocks':
                news_items = [n for n in all_news if any(kw in n.title.lower() for kw in stock_keywords) or 'yahoo' in n.source.lower() or 'economic' in n.source.lower() or 'moneycontrol' in n.source.lower()]
            
            # Fallback to all if no matches
            if not news_items:
                news_items = all_news
        
        # Filter by symbol if provided
        if symbol:
            symbol_upper = symbol.upper().replace('/', '')
            keywords = SYMBOL_MAPPINGS.get(symbol_upper, [symbol_upper.lower()])
            
            filtered = [
                n for n in news_items 
                if symbol_upper in n.related_symbols or 
                   any(kw in n.title.lower() for kw in keywords)
            ]
            
            if filtered:
                news_items = filtered
        
        # Sort by timestamp (most recent first)
        # Normalize timestamps to avoid comparing offset-naive and offset-aware datetimes
        def get_sort_key(n):
            if n.timestamp is None:
                return datetime.min
            # Make timezone-aware datetimes naive for comparison
            ts = n.timestamp
            if hasattr(ts, 'tzinfo') and ts.tzinfo is not None:
                ts = ts.replace(tzinfo=None)
            return ts
        
        sorted_news = sorted(
            news_items,
            key=get_sort_key,
            reverse=True
        )[:limit]
        
        logger.debug(f"📰 Returning {len(sorted_news)} news items")
        
        return [
            {
                'title': n.title,
                'url': n.url,
                'source': n.source,
                'sentiment': n.sentiment_score,
                'sentiment_label': 'BULLISH' if n.sentiment_score > 0.2 else ('BEARISH' if n.sentiment_score < -0.2 else 'NEUTRAL'),
                'symbols': n.related_symbols,
                'timestamp': n.timestamp.isoformat() if n.timestamp else None
            }
            for n in sorted_news
        ]
    
    def _fetch_live_news(self, market: str = 'crypto') -> List[NewsItem]:
        """Fetch LIVE news from RSS feeds."""
        news_items = []
        feeds = LIVE_NEWS_FEEDS.get(market, LIVE_NEWS_FEEDS['crypto'])
        
        for feed_url in feeds:
            try:
                # Disable SSL verification for corporate networks with proxy certificates
                import urllib3
                urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
                response = requests.get(feed_url, timeout=5, verify=False, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    
                    # Parse RSS items
                    for item in root.findall('.//item')[:10]:  # Limit to 10 per feed
                        title_el = item.find('title')
                        link_el = item.find('link')
                        pubdate_el = item.find('pubDate')
                        
                        if title_el is not None and link_el is not None:
                            title = title_el.text or ''
                            url = link_el.text or ''
                            
                            # Parse timestamp
                            timestamp = None
                            if pubdate_el is not None and pubdate_el.text:
                                timestamp = self._parse_timestamp(pubdate_el.text)
                            
                            sentiment = self._calculate_headline_sentiment(title)
                            symbols = self._find_related_symbols(title)
                            source = self._extract_source(url)
                            
                            news_items.append(NewsItem(
                                url=url,
                                title=title,
                                timestamp=timestamp,
                                sentiment_score=sentiment,
                                related_symbols=symbols,
                                source=source
                            ))
                            
            except Exception as e:
                logger.warning(f"Failed to fetch RSS from {feed_url}: {e}")
                continue
        
        logger.info(f"📰 Fetched {len(news_items)} live news items for {market}")
        return news_items
    
    def get_top_movers(self) -> Dict[str, List[str]]:
        """Get stocks mentioned most in bullish/bearish news."""
        news_items = self._load_news_data()
        
        bullish_mentions = {}
        bearish_mentions = {}
        
        for news in news_items:
            for symbol in news.related_symbols:
                if news.sentiment_score > 0.2:
                    bullish_mentions[symbol] = bullish_mentions.get(symbol, 0) + 1
                elif news.sentiment_score < -0.2:
                    bearish_mentions[symbol] = bearish_mentions.get(symbol, 0) + 1
        
        # Sort by mention count
        top_gainers = sorted(bullish_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        top_losers = sorted(bearish_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'top_gainers': [s[0] for s in top_gainers],
            'top_losers': [s[0] for s in top_losers]
        }


# Singleton instance
_sentiment_analyzer: Optional[NewsSentimentAnalyzer] = None


def get_sentiment_analyzer() -> NewsSentimentAnalyzer:
    """Get singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = NewsSentimentAnalyzer()
    return _sentiment_analyzer
