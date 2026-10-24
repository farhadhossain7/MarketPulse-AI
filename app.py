from urllib.parse import quote

import feedparser
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from plotly.subplots import make_subplots
from transformers import pipeline

st.set_page_config(page_title="MarketPulse-AI", page_icon="📈", layout="wide")


# ---------- 1. Sentiment engine (FinBERT) ----------
@st.cache_resource(show_spinner="Loading FinBERT model (first run takes a few minutes)...")
def load_model():
    return pipeline("text-classification", model="ProsusAI/finbert", top_k=None)


def score_headlines(headlines):
    """Return a score from -1.0 (very negative) to +1.0 (very positive) per headline.
    Score = P(positive) - P(negative)."""
    model = load_model()
    results = model(list(headlines), truncation=True)
    scores = []
    for result in results:
        probs = {item["label"].lower(): item["score"] for item in result}
        scores.append(probs.get("positive", 0.0) - probs.get("negative", 0.0))
    return scores


# ---------- 2. News fetcher ----------
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def _parse_feed(url):
    feed = feedparser.parse(url, agent=USER_AGENT)
    rows = []
    for entry in feed.entries:
        rows.append(
            {
                "headline": entry.get("title", ""),
                "published": pd.to_datetime(entry.get("published"), errors="coerce", utc=True),
                "link": entry.get("link", ""),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data(ttl=600)
def fetch_news(ticker):
    """Try Yahoo Finance RSS first, then fall back to Google News RSS
    (Yahoo often blocks requests coming from cloud servers)."""
    try:
        news = _parse_feed(
            f"https://feeds.finance.yahoo.com/rss/2.0/headline?s={quote(ticker)}&region=US&lang=en-US"
        )
        if not news.empty:
            return news, "Yahoo Finance RSS"
    except Exception:
        pass
    try:
        query = quote(f"{ticker} stock")
        news = _parse_feed(f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en")
        if not news.empty:
            return news.head(30), "Google News RSS"
    except Exception:
        pass
    return pd.DataFrame(), "none"


def _stooq_symbol(ticker):
    if ticker == "^GSPC":
        return "^spx"
    if ticker.endswith("-USD"):
        return ticker[:-4].lower() + ".v"
    if "." not in ticker and not ticker.startswith("^"):
        return ticker.lower() + ".us"
    return ticker.lower()


@st.cache_data(ttl=600)
def fetch_prices(ticker):
    """Try Yahoo (yfinance) first, then fall back to Stooq daily prices."""
    try:
        prices = yf.Ticker(ticker).history(period="3mo", interval="1d")
        if not prices.empty:
            prices.index = pd.to_datetime(prices.index).tz_localize(None).normalize()
            return prices, "Yahoo Finance"
    except Exception:
        pass
    try:
        url = f"https://stooq.com/q/d/l/?s={_stooq_symbol(ticker)}&i=d"
        prices = pd.read_csv(url, parse_dates=["Date"], index_col="Date").sort_index()
        prices = prices.tail(90)
        if not prices.empty and "Close" in prices.columns:
            return prices, "Stooq"
    except Exception:
        pass
    return pd.DataFrame(), "none"


# ---------- 3. Backtest ----------
def run_backtest(prices, daily_sentiment, buy_threshold, sell_threshold):
    """Simple rule: sentiment > buy_threshold -> hold the asset,
    sentiment < sell_threshold -> move to cash, otherwise keep the last position.
    The position decided today earns TOMORROW's return (avoids look-ahead bias)."""
    df = prices[["Close"]].copy()
    df["next_day_return"] = df["Close"].pct_change().shift(-1)
    df = df.join(daily_sentiment.rename("sentiment"), how="left")

    position, positions = 0, []
    for s in df["sentiment"]:
        if pd.notna(s):
            if s > buy_threshold:
                position = 1
            elif s < sell_threshold:
                position = 0
        positions.append(position)
    df["position"] = positions

    df = df[df.index >= daily_sentiment.index.min()].dropna(subset=["next_day_return"])
    if df.empty:
        return None
    df["strategy_return"] = df["position"] * df["next_day_return"]
    df["Strategy"] = (1 + df["strategy_return"]).cumprod() - 1
    df["Buy & Hold"] = (1 + df["next_day_return"]).cumprod() - 1
    return df


# ---------- 4. User interface ----------
st.title("📈 MarketPulse-AI")
st.caption("Open-source financial news sentiment engine and backtest simulator (FinBERT)")

ticker = st.sidebar.text_input("Ticker symbol", value="AAPL").strip().upper()
st.sidebar.markdown("Examples: `AAPL`, `TSLA`, `MSFT`, `BTC-USD`, `^GSPC`")
st.sidebar.info("Educational project. Not financial advice.")

if not ticker:
    st.stop()

news, news_source = fetch_news(ticker)
prices, price_source = fetch_prices(ticker)

if news.empty:
    st.warning(
        "Could not load news headlines for this ticker. The news source may be "
        "temporarily blocking requests. Try again in a minute or try another symbol."
    )
    st.stop()
if prices.empty:
    st.warning(
        "Could not load price data for this ticker. The price source may be "
        "temporarily blocking requests. Try again in a minute or try another symbol."
    )
    st.stop()

st.caption(f"Data sources: news from {news_source}, prices from {price_source}.")

news = news.dropna(subset=["published"]).copy()
news["sentiment"] = score_headlines(news["headline"])
news["date"] = news["published"].dt.tz_localize(None).dt.normalize()
daily = news.groupby("date")["sentiment"].mean()
daily.index = pd.to_datetime(daily.index)

tab1, tab2, tab3 = st.tabs(["Dashboard", "Backtest simulator", "About & limitations"])

with tab1:
    now = pd.Timestamp.now(tz="UTC")
    last_24h = news[news["published"] >= now - pd.Timedelta(hours=24)]["sentiment"]
    earlier = news[news["published"] < now - pd.Timedelta(hours=24)]["sentiment"]
    shift = (last_24h.mean() - earlier.mean()) if len(last_24h) and len(earlier) else None
    bullish = int((news["sentiment"] > 0.2).sum())
    bearish = int((news["sentiment"] < -0.2).sum())

    c1, c2, c3 = st.columns(3)
    c1.metric("Current sentiment index", f"{news['sentiment'].mean():+.2f}")
    c2.metric("Bullish / Bearish headlines", f"{bullish} / {bearish}")
    c3.metric("24-hour sentiment shift", f"{shift:+.2f}" if shift is not None else "n/a")

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Candlestick(
            x=prices.index,
            open=prices["Open"],
            high=prices["High"],
            low=prices["Low"],
            close=prices["Close"],
            name="Price",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Bar(x=daily.index, y=daily.values, name="Avg daily sentiment", opacity=0.6),
        secondary_y=True,
    )
    fig.update_layout(height=550, xaxis_rangeslider_visible=False, title=f"{ticker}: price vs news sentiment")
    fig.update_yaxes(title_text="Price", secondary_y=False)
    fig.update_yaxes(title_text="Sentiment (-1 to +1)", range=[-1, 1], secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Scored headlines")
    st.dataframe(
        news[["published", "headline", "sentiment", "link"]].sort_values("published", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

with tab2:
    st.write(
        "Rule: if daily sentiment is above the **buy threshold**, hold the asset; "
        "if it is below the **sell threshold**, move to cash."
    )
    buy_th = st.slider("Buy threshold", 0.0, 1.0, 0.3, 0.05)
    sell_th = st.slider("Sell threshold", -1.0, 0.0, -0.3, 0.05)
    result = run_backtest(prices, daily, buy_th, sell_th)
    if result is None:
        st.info("Not enough overlapping news and price days to simulate.")
    else:
        m1, m2 = st.columns(2)
        m1.metric("Strategy return", f"{result['Strategy'].iloc[-1] * 100:+.2f}%")
        m2.metric("Buy & hold return", f"{result['Buy & Hold'].iloc[-1] * 100:+.2f}%")
        line = go.Figure()
        line.add_trace(go.Scatter(x=result.index, y=result["Strategy"] * 100, name="Strategy"))
        line.add_trace(go.Scatter(x=result.index, y=result["Buy & Hold"] * 100, name="Buy & Hold"))
        line.update_layout(yaxis_title="Cumulative return (%)", height=400)
        st.plotly_chart(line, use_container_width=True)
        st.caption(f"Simulation covers only {len(result)} trading days with available headlines.")

with tab3:
    st.markdown(
        """
**How it works:** headlines come from Yahoo Finance RSS, are scored by FinBERT
(a language model trained on financial text), and each score = P(positive) - P(negative).

**Limitations (please read):**
- RSS feeds only provide the most recent headlines, so the backtest covers a very short window
  and is **not statistically meaningful**.
- No transaction costs, slippage, or risk metrics are modeled.
- Headline sentiment alone is a weak trading signal. This is a learning tool, **not financial advice**.

**Future work:** store headlines daily to build a real historical dataset, add significance testing,
and compare against other sentiment models.
"""
    )
