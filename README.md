# MarketPulse-AI

An open-source financial news sentiment engine and backtest simulator.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/Streamlit-app-red) ![HuggingFace](https://img.shields.io/badge/HuggingFace-FinBERT-yellow) ![License](https://img.shields.io/badge/License-MIT-green)

**Live demo:** PASTE-YOUR-STREAMLIT-LINK-HERE

## What it does
1. Pulls live financial headlines for a ticker from Yahoo Finance RSS.
2. Scores each headline with **FinBERT** (`ProsusAI/finbert`) from -1.0 (negative) to +1.0 (positive).
3. Overlays daily sentiment on price candles.
4. Runs a simple rule-based sentiment backtest against buy-and-hold.

## Why I built it
I wrote a research paper on how choice architecture and dark patterns influence decisions.
This project applies the same question to markets: how does the language of news shape what participants do?

## Architecture
RSS news -> FinBERT sentiment scoring -> daily aggregation -> Streamlit dashboard + backtest

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Limitations
RSS only provides recent headlines, so the backtest window is short and not statistically meaningful.
No transaction costs are modeled. Educational use only, not financial advice.

## Roadmap
- Store headlines daily to build a real historical dataset
- Add significance tests and risk metrics
- Compare multiple sentiment models

## License
MIT
