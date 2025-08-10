import yfinance as yf
import requests
import datetime
import pandas as pd
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os


API_KEY = "wioh62aY3LKrNZoyNSoIFTQzKfjbR5AvGtX"
API_URL = "https://cloud.utradea.com/v1/get-social"

def get_utradea_social_sentiment(ticker: str, social: str = "reddit", charts: str = "posts,comments,likes"):
    headers = {
        "Content-Type": "application/json",
        "Authorization": API_KEY
    }
    params = {
        "tickers": ticker,
        "social": social,
        "charts": charts
    }
    resp = requests.get(API_URL, headers=headers, params=params)
    if resp.status_code == 200:
        return resp.json()
    else:
        raise Exception(f"Error {resp.status_code}: {resp.text}")
    
# ---- 1. Get VIX Data ---- #
def get_current_vix():
    vix = yf.Ticker("^VIX")
    data = vix.history(period="1d", interval="1m")
    if data.empty:
        return None
    latest = data["Close"].iloc[-1]
    return latest

def scale_vix(vix_value, low=12, high=40):
    """Inverse scale: lower VIX = more bullish (closer to 1)"""
    scaled = (high - vix_value) / (high - low)
    return max(0.0, min(1.0, scaled))

def get_current_sp():
    sp = yf.Ticker("^GSPC")
    data = vix.history(period="1d", interval="1m")
    if data.empty:
        return None
    latest = data["Close"].iloc[-1]
    return latest

def scale_sp(sp_value, low=12, high=40):
    """Inverse scale: lower VIX = more bullish (closer to 1)"""
    scaled = (high - sp_value) / (high - low)
    return max(0.0, min(1.0, scaled))

# ---- 2. Get News Headlines ---- #
def get_headlines_yahoo():
    url = "https://finance.yahoo.com/"
    r = requests.get(url)
    soup = BeautifulSoup(r.text, 'html.parser')
    headlines = [item.text for item in soup.find_all('h3')][:10]
    return headlines

# ---- 3. Analyze with FinBERT ---- #
class FinBERTSentiment:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        self.model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")

    def score(self, texts):
        scores = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.nn.functional.softmax(logits, dim=1).squeeze()
                scores.append(probs[2].item())  # Positive score
        return sum(scores) / len(scores) if scores else 0.5  # Neutral fallback

# ---- 4. Mock Reddit Sentiment ---- #
def get_mock_reddit_sentiment():
    return 0.6  # Simulate an average slightly bullish sentiment

# ---- 5. Combine into Final Score ---- #
def compute_sentiment_score(news_s, reddit_s, vix_s):
    weighted = (
        0.5 * news_s +
        0.3 * reddit_s +
        0.2 * vix_s
    )
    return round(weighted * 100, 2)

# ---- 6. Save to CSV ---- #
def log_sentiment_score(score, details, filepath="sentiment_log.csv"):
    now = datetime.datetime.utcnow().replace(second=0, microsecond=0)
    row = {
        "timestamp_utc": now.isoformat(),
        "sentiment_score": score,
        **details
    }

    try:
        df = pd.read_csv(filepath)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])

    df.to_csv(filepath, index=False)
    print(f"✅ Logged sentiment score to {filepath}")

# ---- MAIN ---- #
if __name__ == "__main__":
    print(f"\n📊 Running market sentiment evaluation at {datetime.datetime.utcnow().isoformat()} UTC\n")

    vix = get_current_vix()
    if vix is None:
        print("❌ Could not retrieve VIX data. Exiting.")
        exit()

    vix_scaled = scale_vix(vix)
    headlines = get_headlines_yahoo()
    finbert = FinBERTSentiment()
    news_sentiment = finbert.score(headlines)
    reddit_sentiment = get_mock_reddit_sentiment()

    score = compute_sentiment_score(news_sentiment, reddit_sentiment, vix_scaled)

    print(f"📰 News Sentiment:   {news_sentiment:.2f}")
    print(f"🧠 Reddit Sentiment: {reddit_sentiment:.2f}")
    print(f"📉 VIX Scaled:       {vix_scaled:.2f}")
    print(f"\n🔮 Final Sentiment Score: {score}/100")

    log_sentiment_score(score, {
        "news_sentiment": round(news_sentiment, 3),
        "reddit_sentiment": round(reddit_sentiment, 3),
        "vix_scaled": round(vix_scaled, 3),
        "vix_raw": round(vix, 2)
    })
