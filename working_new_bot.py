import requests
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
import datetime
import csv
import os
import time
import os
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout
import datetime

ENV_FILE = 'discord.env'

if not os.path.exists(ENV_FILE):
    raise FileNotFoundError(f"{ENV_FILE} not found. Please create it.")

load_dotenv(ENV_FILE)
# Now access the variables like:
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
CHANNEL_ID = int(os.getenv("CHANNEL_ID"))
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")                    # <-- your NewsAPI key

NIFTY50 = [
    'ADANIENT','ADANIPORTS','APOLLOHOSP','ASIANPAINT','AXISBANK','BAJAJ-AUTO',
    'BAJFINANCE','BAJAJFINSV','BPCL','BHARTIARTL','BRITANNIA','CIPLA','COALINDIA',
    'DIVISLAB','DRREDDY','EICHERMOT','GRASIM','HCLTECH','HDFCBANK','HDFCLIFE',
    'HEROMOTOCO','HINDALCO','HINDUNILVR','ICICIBANK','ITC','INDUSINDBK','INFY',
    'JSWSTEEL','KOTAKBANK','LTIM','LT','M&M','MARUTI','NTPC','NESTLEIND',
    'ONGC','POWERGRID','RELIANCE','SBILIFE','SHRIRAMFIN','SBIN','SUNPHARMA',
    'TCS','TATACONSUM','TATAMOTORS','TATASTEEL','TECHM','TITAN','ULTRACEMCO','WIPRO'
]

STORAGE_FILE = "posted_headlines_perplexity.csv"
ET_RSS_URL = "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"

def load_seen():
    seen = set()
    if os.path.exists(STORAGE_FILE):
        with open(STORAGE_FILE, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    seen.add(row[0])
    return seen

def save_seen_entry(url, pub_date):
    with open(STORAGE_FILE, "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([url, pub_date])

def fetch_moneycontrol_news(keyword):
    url = f"https://www.moneycontrol.com/news/tags/{keyword.lower()}.html"
    resp = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=10)
    print("[MC]", keyword, "status", resp.status_code)
    if resp.status_code != 200:
        return []
    soup = BeautifulSoup(resp.content,"html.parser")
    out=[]
    for li in soup.find_all("li", class_="clearfix")[:5]:
        a=li.find("a", href=True)
        if not a: continue
        link=a["href"].strip()
        if not link.startswith("https://www.moneycontrol.com/news/"):
            continue
        title_tag=li.find("h2") or a
        title=title_tag.get_text(strip=True)
        summary=(li.find("p").get_text(strip=True) if li.find("p") else "")
        out.append({"source":"Moneycontrol","title":title,"url":link,"summary":summary,"pub_date":""})
    return out

def fetch_et_rss_news(keyword):
    resp = requests.get(ET_RSS_URL, timeout=10)
    print("[ET‑RSS] status", resp.status_code)
    if resp.status_code !=200:
        return []
    root=ET.fromstring(resp.content)
    out=[]
    for item in root.findall(".//item")[:20]:
        title=item.findtext("title","").strip()
        desc=item.findtext("description","").strip()
        link=item.findtext("link","").strip()
        pub=item.findtext("pubDate","").strip()
        if keyword.lower() in title.lower() or keyword.lower() in desc.lower():
            out.append({"source":"EconomicTimes","title":title,"url":link,"summary":desc,"pub_date":pub})
    return out

def fetch_yahoo_news(symbol):
    url = f"https://query1.finance.yahoo.com/v1/finance/search?q={symbol}&newsCount=50"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)  # 10 seconds timeout
        print("[Yahoo]", symbol, "status", resp.status_code)
        if resp.status_code != 200:
            return []
        data = resp.json().get("news", [])
        out = []
        for itm in data[:5]:
            t = itm.get("title", "").strip()
            url = itm.get("link", "").strip()
            pub_ts = itm.get("providerPublishTime")
            pubstr = datetime.datetime.fromtimestamp(pub_ts).strftime("%Y-%m-%d %H:%M") if pub_ts else ""
            pubr = itm.get("publisher", "").strip()
            summary = f"{pubr} | {pubstr}"
            out.append({"source": "YahooFinance", "title": t, "url": url, "summary": summary, "pub_date": pubstr})
        return out
    except Timeout:
        print(f"[Yahoo] Request timed out after 10 seconds for symbol: {symbol}")
        return []
    except RequestException as e:
        print(f"[Yahoo] Network error: {e}")
        return []

def fetch_newsapi_news(keyword):
    # NewsAPI: q is usually best as full company name for best matches, but ticker is fine
    headers = {"User-Agent":"Mozilla/5.0"}
    params = dict(
        apiKey=NEWSAPI_KEY,
        q=keyword + " stock",
        language="en", pageSize=5, sortBy="publishedAt"
    )
    resp = requests.get("https://newsapi.org/v2/everything", params=params, headers=headers, timeout=10)
    print("[NewsAPI]", keyword, "status", resp.status_code)
    if resp.status_code != 200:
        print("    ", resp.text[:80])
        return []
    items = []
    for art in resp.json().get("articles", []):
        items.append({
            "source":"NewsAPI",
            "title":art.get("title","").strip(),
            "url":art.get("url","").strip(),
            "summary":(art.get("description","") or "").strip(),
            "pub_date":art.get("publishedAt","")
        })
    return items

def summarize_article_text(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code != 200:
            return []
        soup = BeautifulSoup(response.text, "html.parser")
        paragraphs = soup.find_all('p')
        text = " ".join(p.get_text() for p in paragraphs)
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 40]
        return [f"• {s}." for s in sentences[:5]]
    except Exception as e:
        return [f"• (Failed to summarize: {e})"]

def post_to_discord_webhook(message, webhook=WEBHOOK_URL):
    try:
        resp = requests.post(webhook, json={"content": message})
        if not (200 <= resp.status_code < 300):
            print(f"[Discord] Error: {resp.status_code}\n{resp.text[:100]}")
            return False
        return True
    except Exception as e:
        print(f"[Discord] Exception: {e}")
        return False

def aggregate_and_dedup(articles):
    dedup = {}
    for art in articles:
        url = art["url"]
        if url and url not in dedup:
            dedup[url] = art
    return list(dedup.values())

def safe_discord_message(msg):
    # Discord API allows up to 2000 chars for 'content'
    return msg[:1990] + "\n...(truncated)" if len(msg) > 2000 else msg


if __name__ == "__main__":
    SLEEP_DURATION = 300  # in seconds (5 min); you can set to 900 for 15 mins if preferred

    while True:
        seen = load_seen()
        new_seen = set(seen)
        all_to_post = []
        all_found = 0

        for ticker in NIFTY50:
            print("\n----", ticker)
            mc = fetch_moneycontrol_news(ticker)
            et = fetch_et_rss_news(ticker)
            yf = fetch_yahoo_news(ticker + ".NS")
            na = fetch_newsapi_news(ticker)
            coll = aggregate_and_dedup(mc + et + yf + na)

            for art in coll:
                key = art["url"]
                if not key or key in new_seen:
                    continue

                all_found += 1
                new_seen.add(key)
                save_seen_entry(key, art.get("pub_date", ""))

                print(f"[{art['source']}] {ticker}: {art['title']}\n  {art['url']}\n  {art['summary']}\n")
                bullet_points = summarize_article_text(art['url'])
                if bullet_points:
                    print("  🔍 Summary:")
                    for point in bullet_points:
                        print(f"   {point}")
                print()

                # Bundle for possible push
                all_to_post.append({
                    "source": art['source'],
                    "ticker": ticker,
                    "title": art["title"],
                    "url": art["url"],
                    "summary": art["summary"],
                    "bullets": bullet_points
                })

            time.sleep(1)

        print(f"Total new to post: {len(all_to_post)}")
        if not all_to_post:
            print("No new news this cycle. Waiting...")

        # Post immediately, no asking for confirmation
        for art in all_to_post:
            msg = f"**[{art['ticker']}] {art['title']}**\n<{art['url']}>\n{art['summary']}"
            if art["bullets"]:
                msg += "\n\n" + "\n".join(art["bullets"])
            print(f"Pushing: {art['title']}")
            msg = safe_discord_message(msg)
            post_to_discord_webhook(msg)
            time.sleep(1.5)

        print(f"Posted {len(all_to_post)} news items to Discord.")

        # Sleep until next round
        print(f"\nSleeping {SLEEP_DURATION//60} minutes before next check...\n{'-'*45}\n")
        time.sleep(SLEEP_DURATION)

