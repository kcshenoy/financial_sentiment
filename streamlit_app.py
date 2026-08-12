"""
Portfolio Sentiment Analyzer — Streamlit Web App
Combines fine-tuned DistilBERT sentiment (AWS Lambda) with rules-based technical analysis.

Local dev:
  cp .env.example .env        # fill in your Lambda URL
  streamlit run streamlit_app.py

Deploy to Streamlit Cloud:
  Add LAMBDA_URL in the Streamlit Cloud secrets dashboard.
"""

import json
import os
import re
from datetime import datetime
from pathlib import Path

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf
from bs4 import BeautifulSoup

# LAMBDA_URL is IAM-scoped — loaded from .env locally, Streamlit secrets in prod.
LAMBDA_URL: str = os.getenv("LAMBDA_URL", "")
if not LAMBDA_URL:
    try:
        LAMBDA_URL = st.secrets.get("LAMBDA_URL", "")
    except Exception:
        pass


# ── Technical analysis ────────────────────────────────────────────────────────

def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    close = df["Close"]

    df["SMA_20"] = close.rolling(20).mean()
    df["SMA_50"] = close.rolling(50).mean()
    df["SMA_200"] = close.rolling(200).mean()

    df["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    df["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    df["MACD"] = df["EMA_12"] - df["EMA_26"]
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"]  = df["MACD"] - df["MACD_Signal"]

    # RSI — Wilder's smoothing (EWM alpha = 1/14)
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    avg_loss = (-delta).clip(lower=0).ewm(alpha=1 / 14, min_periods=14).mean()
    df["RSI"] = 100 - 100 / (1 + avg_gain / avg_loss)

    # Bollinger Bands (20-day, ±2σ)
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    df["BB_Mid"] = bb_mid
    df["BB_Upper"] = bb_mid + 2 * bb_std
    df["BB_Lower"] = bb_mid - 2 * bb_std
    df["BB_Pct"] = (close - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"])

    return df


def generate_signals(df: pd.DataFrame) -> list[dict]:
    r = df.iloc[-1]
    price = float(r["Close"])
    sigs  = []

    # RSI
    rsi = float(r["RSI"])
    if rsi > 70:
        sigs.append({"name": "RSI", "direction": "Bearish",
                     "msg": f"RSI {rsi:.1f} — overbought territory, pullback risk"})
    elif rsi < 30:
        sigs.append({"name": "RSI", "direction": "Bullish",
                     "msg": f"RSI {rsi:.1f} — oversold, potential mean-reversion bounce"})
    else:
        sigs.append({"name": "RSI", "direction": "Neutral",
                     "msg": f"RSI {rsi:.1f} — neutral band (30–70)"})

    # MACD vs Signal line
    macd, macd_sig = float(r["MACD"]), float(r["MACD_Signal"])
    if macd > macd_sig:
        sigs.append({"name": "MACD", "direction": "Bullish",
                     "msg": f"MACD ({macd:.3f}) above signal ({macd_sig:.3f}) — bullish momentum"})
    else:
        sigs.append({"name": "MACD", "direction": "Bearish",
                     "msg": f"MACD ({macd:.3f}) below signal ({macd_sig:.3f}) — bearish momentum"})

    # Golden / Death Cross (50 vs 200 SMA)
    sma50, sma200 = r["SMA_50"], r["SMA_200"]
    if not (pd.isna(sma50) or pd.isna(sma200)):
        if sma50 > sma200:
            sigs.append({"name": "SMA Cross", "direction": "Bullish",
                         "msg": f"Golden cross — 50-day (${sma50:.2f}) above 200-day (${sma200:.2f})"})
        else:
            sigs.append({"name": "SMA Cross", "direction": "Bearish",
                         "msg": f"Death cross — 50-day (${sma50:.2f}) below 200-day (${sma200:.2f})"})

    # Price vs 20-day SMA (short-term trend)
    sma20 = r["SMA_20"]
    if not pd.isna(sma20):
        if price > float(sma20):
            sigs.append({"name": "Short-term", "direction": "Bullish",
                         "msg": f"Price (${price:.2f}) above 20-day SMA (${float(sma20):.2f})"})
        else:
            sigs.append({"name": "Short-term", "direction": "Bearish",
                         "msg": f"Price (${price:.2f}) below 20-day SMA (${float(sma20):.2f})"})

    # Bollinger Band position
    bb_pct = r["BB_Pct"]
    if not pd.isna(bb_pct):
        bp = float(bb_pct)
        if bp > 1.0:
            sigs.append({"name": "Bollinger Bands", "direction": "Bearish",
                         "msg": "Price above upper band — extended, mean-reversion risk"})
        elif bp < 0.0:
            sigs.append({"name": "Bollinger Bands", "direction": "Bullish",
                         "msg": "Price below lower band — compressed, potential bounce"})
        else:
            sigs.append({"name": "Bollinger Bands", "direction": "Neutral",
                         "msg": f"Price at {bp * 100:.0f}% of band width"})

    return sigs


def score_signals(sigs: list[dict]) -> tuple[int, str]:
    score = sum(1 if s["direction"] == "Bullish" else -1 if s["direction"] == "Bearish" else 0
                  for s in sigs)
    verdict = "Bullish" if score >= 2 else "Bearish" if score <= -2 else "Neutral"
    return score, verdict


# ── News & sentiment ──────────────────────────────────────────────────────────

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
}


def fetch_news(ticker: str, n: int = 5) -> list[dict]:
    raw = yf.Ticker(ticker).news or []
    results = []
    for item in raw[:n]:
        c = item.get("content", item)  # new API nests under "content"; old API is flat
        url = (
            (c.get("canonicalUrl") or {}).get("url")
            or (c.get("clickThroughUrl") or {}).get("url")
            or c.get("link", "")
        )
        pub = c.get("pubDate") or c.get("displayTime", "")
        try:
            published = datetime.fromisoformat(pub.replace("Z", "+00:00")).replace(tzinfo=None)
        except Exception:
            published = datetime.fromtimestamp(c.get("providerPublishTime", 0))
        results.append({
            "title":     c.get("title", ""),
            "url":       url,
            "publisher": (c.get("provider") or {}).get("displayName", c.get("publisher", "")),
            "published": published,
        })
    return results


def scrape_article(url: str, timeout: int = 6) -> str:
    """Best-effort article body extraction; returns empty string on any failure."""
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        article = soup.find("article")
        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            paras = [p.get_text(strip=True) for p in soup.find_all("p")
                     if len(p.get_text(strip=True)) > 80]
            text = " ".join(paras[:12])
        return text[:2000]  # Lambda enforces 2 000-char limit
    except Exception:
        return ""


def _lambda_region() -> str:
    m = re.search(r"lambda-url\.([a-z0-9-]+)\.on\.aws", LAMBDA_URL)
    return m.group(1) if m else "us-east-2"


def call_lambda(text: str) -> dict | None:
    text = text.strip()[:2000]
    if not text:
        return None
    try:
        url = LAMBDA_URL.rstrip("/") + "/predict"
        body = json.dumps({"text": text})
        creds = boto3.Session().get_credentials().get_frozen_credentials()
        aws_req = AWSRequest(
            method="POST",
            url=url,
            data=body,
            headers={"Content-Type": "application/json"},
        )
        SigV4Auth(creds, "lambda", _lambda_region()).add_auth(aws_req)
        resp = requests.post(url, data=body, headers=dict(aws_req.headers), timeout=15)
        if not resp.ok:
            st.warning(f"Lambda {resp.status_code}: {resp.text}")
            return None
        return resp.json()
    except Exception as exc:
        st.warning(f"Lambda error: {exc}")
        return None


# ── Plotly charts ─────────────────────────────────────────────────────────────

def price_chart(df: pd.DataFrame, ticker: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    ))
    for col, color in [("SMA_20", "#1565C0"), ("SMA_50", "#F57F17"), ("SMA_200", "#AB47BC")]:
        fig.add_trace(go.Scatter(x=df.index, y=df[col], name=col,
                                  line=dict(color=color, width=1.4)))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Upper"],
        line=dict(color="rgba(160,160,160,0.5)", dash="dot"), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=df["BB_Lower"],
        line=dict(color="rgba(160,160,160,0.5)", dash="dot"),
        fill="tonexty", fillcolor="rgba(160,160,160,0.06)", showlegend=False,
    ))
    fig.update_layout(
        title=f"{ticker} — Price, Moving Averages & Bollinger Bands (1 Year)",
        xaxis_rangeslider_visible=False,
        height=480, template="plotly_dark",
        legend=dict(orientation="h", y=-0.14),
        margin=dict(l=40, r=20, t=44, b=40),
    )
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    colors = ["#26a69a" if v >= 0 else "#ef5350" for v in df["MACD_Hist"]]
    fig = go.Figure([
        go.Bar(x=df.index, y=df["MACD_Hist"], name="Histogram", marker_color=colors),
        go.Scatter(x=df.index, y=df["MACD"],        name="MACD",   line=dict(color="#1565C0", width=1.5)),
        go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#F57F17", width=1.5)),
    ])
    fig.update_layout(title="MACD (12, 26, 9)", height=280, template="plotly_dark",
                      legend=dict(orientation="h"), margin=dict(l=40, r=20, t=40, b=20))
    return fig


def rsi_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure([
        go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", line=dict(color="#1565C0", width=1.5)),
    ])
    fig.add_hline(y=70, line_dash="dash", line_color="#ef5350", annotation_text="Overbought 70")
    fig.add_hline(y=30, line_dash="dash", line_color="#26a69a", annotation_text="Oversold 30")
    fig.add_hrect(y0=30, y1=70, fillcolor="rgba(255,255,255,0.02)", line_width=0)
    fig.update_layout(title="RSI (14)", height=280, template="plotly_dark",
                      yaxis=dict(range=[0, 100]), margin=dict(l=40, r=20, t=40, b=20))
    return fig


# ── Streamlit layout ──────────────────────────────────────────────────────────

st.set_page_config(page_title="Portfolio Sentiment Analyzer", page_icon="📈", layout="wide")
st.title("📈 Portfolio Sentiment Analyzer")
st.caption("Fine-tuned DistilBERT on AWS Lambda · Rules-based technical analysis")

ICON = {"Bullish": "🟢", "Bearish": "🔴", "Neutral": "⚪",
        "Positive": "🟢", "Negative": "🔴"}

with st.sidebar:
    st.header("Settings")
    ticker_input = st.text_input("Stock ticker", value="AAPL", max_chars=10).upper().strip()
    n_articles   = st.slider("News articles to analyze", 1, 10, 5)
    run          = st.button("Analyze", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Legend:** 🟢 Bullish &nbsp; 🔴 Bearish &nbsp; ⚪ Neutral")

if not run:
    st.info("Enter a ticker in the sidebar and click **Analyze**.")
    st.stop()

ticker = ticker_input

# ── Price data ────────────────────────────────────────────────────────────────
with st.spinner(f"Fetching price data for {ticker}…"):
    yft  = yf.Ticker(ticker)
    df   = yft.history(period="1y")
    info = yft.info or {}

if df.empty:
    st.error(f"No price data for **{ticker}**. Check the ticker symbol and try again.")
    st.stop()

df             = compute_technicals(df)
signals        = generate_signals(df)
score, verdict = score_signals(signals)

# ── Header metrics ─────────────────────────────────────────────────────────────
price = float(df["Close"].iloc[-1])
prev  = float(df["Close"].iloc[-2])
chg   = price - prev
chg_p = chg / prev * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric(info.get("longName", ticker), f"${price:.2f}", f"{chg:+.2f} ({chg_p:+.2f}%)")
c2.metric("52W High",   f"${df['High'].max():.2f}")
c3.metric("52W Low",    f"${df['Low'].min():.2f}")
mktcap = info.get("marketCap")
c4.metric("Market Cap", f"${mktcap / 1e9:.1f}B" if mktcap else "—")
c5.metric("Volume",     f"{float(df['Volume'].iloc[-1]) / 1e6:.1f}M")

# ── Charts ─────────────────────────────────────────────────────────────────────
st.plotly_chart(price_chart(df, ticker), use_container_width=True)

col_macd, col_rsi = st.columns(2)
with col_macd:
    st.plotly_chart(macd_chart(df), use_container_width=True)
with col_rsi:
    st.plotly_chart(rsi_chart(df), use_container_width=True)

# ── Technical signal cards ─────────────────────────────────────────────────────
st.subheader("Technical Signals")
active = len([s for s in signals if s["direction"] != "Neutral"])
st.markdown(f"### {ICON[verdict]} **{verdict}** &nbsp; ({score:+d} / {active} directional signals)")

sig_cols = st.columns(len(signals))
for col, s in zip(sig_cols, signals):
    col.markdown(f"**{ICON[s['direction']]} {s['name']}**")
    col.caption(s["msg"])

# ── News + sentiment ───────────────────────────────────────────────────────────
st.divider()
st.subheader("Recent News & Sentiment")

if not LAMBDA_URL:
    st.warning("LAMBDA_URL not set — showing headlines only. Add it to your .env file.")

with st.spinner("Fetching news…"):
    articles = fetch_news(ticker, n=n_articles)

if not articles:
    st.info("No recent news found for this ticker.")
else:
    sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
    total_analyzed   = 0

    for art in articles:
        date_str = art["published"].strftime("%b %d, %Y")
        with st.expander(f"**{art['title']}** — {art['publisher']} · {date_str}"):
            st.markdown(f"[Read full article ↗]({art['url']})")

            if not LAMBDA_URL:
                continue

            with st.spinner("Scraping article & running sentiment…"):
                body          = scrape_article(art["url"])
                used_fallback = len(body) < 100
                if used_fallback:
                    body = art["title"]
                result = call_lambda(body)

            if not result:
                continue

            total_analyzed += 1
            lbl = result["label"]
            sentiment_counts[lbl] += 1
            sc       = result["scores"]
            p, neu, n_ = sc.get("Positive", 0), sc.get("Neutral", 0), sc.get("Negative", 0)

            fallback_note = " · *headline only — article paywalled or blocked*" if used_fallback else ""
            st.markdown(
                f"**Sentiment: {ICON[lbl]} {lbl}** &nbsp; (confidence {result['score']:.1%})"
                + fallback_note
            )
            st.caption(f"🟢 Positive {p:.1%} &nbsp; ⚪ Neutral {neu:.1%} &nbsp; 🔴 Negative {n_:.1%}")

    # ── Aggregate + composite signal ───────────────────────────────────────────
    if LAMBDA_URL and total_analyzed > 0:
        st.subheader("News Sentiment Summary")
        total = total_analyzed
        c1, c2, c3 = st.columns(3)
        c1.metric("🟢 Positive", sentiment_counts["Positive"],
                  f"{sentiment_counts['Positive'] / total:.0%} of {total} articles")
        c2.metric("⚪ Neutral",  sentiment_counts["Neutral"],
                  f"{sentiment_counts['Neutral'] / total:.0%} of {total} articles")
        c3.metric("🔴 Negative", sentiment_counts["Negative"],
                  f"{sentiment_counts['Negative'] / total:.0%} of {total} articles")

        sent_score       = sentiment_counts["Positive"] - sentiment_counts["Negative"]
        combined         = score + sent_score
        combined_verdict = "Bullish" if combined >= 2 else "Bearish" if combined <= -2 else "Neutral"

        st.divider()
        st.subheader("Composite Signal")
        cc1, cc2, cc3 = st.columns(3)
        cc1.metric("Technical", f"{score:+d}",      verdict)
        cc2.metric("Sentiment", f"{sent_score:+d}",
                   "Net positive" if sent_score > 0 else "Net negative" if sent_score < 0 else "Balanced")
        cc3.metric("Combined",  f"{combined:+d}",   f"{ICON[combined_verdict]} {combined_verdict}")
