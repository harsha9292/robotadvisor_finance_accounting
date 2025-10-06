from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import os

from risk_scoring import calculate_score, determine_risk_profile
from etf_allocations import (
    get_etf_portfolio, 
    ETF_UNIVERSE, 
    ETFMetrics,
    refresh_etf_universe
)
from portfolio_analysis import generate_etf_report, backtest_portfolio

app = FastAPI(title="RoboAdvisor MVP")
app.mount("/screens", StaticFiles(directory="screens", html=True), name="frontend")


@app.get("/")
def root():
    return FileResponse(os.path.join("screens", "index.html"))

# ---------- Risk Questionnaire Input ----------
class RiskInput(BaseModel):
    age: int
    income: float
    investment_goal: str
    #risk_tolerance: int  # 1-5
    primary_goal: str
    access_time: str
    income_stability: str
    emergency_fund: str
    investment_plan: str
    initial_investment: str
    reaction_to_loss: str
    investing_experience: str
    geographical_focus: str
    esg_preference: str
    sectors_to_avoid: Optional[List[str]] = []

# ---------- Chat Input ----------
class ChatIn(BaseModel):
    message: str

# ---------- ETF Data + Clustering ----------
TICKERS = ["VTI", "VEA", "VWO", "BND", "QQQ", "SPY", "TLT", "GLD"]

def fetch_etf_data():
    data = {}
    for t in TICKERS:
        etf = yf.Ticker(t)
        hist = etf.history(period="3y")
        if len(hist) < 200:
            continue
        ret = hist["Close"].pct_change().dropna()
        ann_return = (1 + ret.mean()) ** 252 - 1
        vol = ret.std() * np.sqrt(252)
        avg_vol = hist["Volume"].mean()
        info = etf.info
        expense = info.get("expenseRatio", 0.002)
        hist_monthly = hist["Close"].resample("ME").last()
        data[t] = {
            "ann_return": ann_return,
            "volatility": vol,
            "avg_volume": avg_vol,
            "expense_ratio": expense,
            "history": hist_monthly.to_dict()
        }
    df = pd.DataFrame(data).T
    return df.dropna()

def make_clusters(df):
    if df.empty:
        return df
    features = df[["ann_return", "volatility", "avg_volume", "expense_ratio"]]
    X = StandardScaler().fit_transform(features)
    kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
    df["cluster"] = kmeans.labels_
    return df

ETF_DF = make_clusters(fetch_etf_data())

# ---------- Risk Assessment ----------
@app.post("/assess")
def assess(inputs: RiskInput):
    input_dict = inputs.dict()
    score = calculate_score(input_dict)
    profile_data = determine_risk_profile(score)
    etf_portfolio = get_etf_portfolio(profile_data["risk_bucket"])
    return {
        "score": score,
        **profile_data,
        "etf_portfolio": etf_portfolio
    }

# ---------- ETF Recommendations ----------
@app.get("/etf-universe")
def get_etf_universe():
    """Return details about available ETFs"""
    return {
        ticker: {
            "name": etf.name,
            "expense_ratio": etf.expense_ratio,
            "avg_volume": etf.avg_volume,
            "historical_return": etf.historical_return,
            "volatility": etf.volatility
        }
        for ticker, etf in ETF_UNIVERSE.items()
    }

@app.get("/recommend/{bucket}")
def recommend(bucket: int):
    portfolio = get_etf_portfolio(bucket)
    return {
        "name": portfolio["name"],
        "description": portfolio["description"],
        "target_return": f"{portfolio['target_return']:.2%}",
        "target_volatility": f"{portfolio['target_volatility']:.2%}",
        "allocation": portfolio["allocation"],
        "sharpe_ratio": f"{portfolio['sharpe_ratio']:.2f}"
    }

@app.get("/refresh-data")
async def refresh_data():
    """Refresh ETF data and regenerate portfolios"""
    try:
        refresh_etf_universe()
        return {"message": "Data refreshed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/etf-analysis")
async def get_etf_analysis():
    """Get comprehensive ETF analysis"""
    try:
        report = generate_etf_report()
        
        # Convert DataFrames to dictionaries for JSON response
        analysis = {
            'etf_characteristics': report['etf_characteristics'].to_dict(),
            'risk_mappings': report['risk_mappings'].to_dict(),
            'performance_metrics': report['performance_metrics']
        }
        
        return JSONResponse(content=analysis)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio-backtest/{risk_level}")
async def backtest_risk_level(risk_level: int, start_date: str = "2019-01-01"):
    """Backtest portfolio performance for given risk level"""
    try:
        portfolio = get_etf_portfolio(risk_level)
        performance = backtest_portfolio(portfolio, start_date)
        
        return {
            "risk_level": risk_level,
            "portfolio_name": portfolio["name"],
            "cumulative_returns": performance.to_dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/portfolio-metrics/{risk_level}")
async def get_portfolio_metrics(risk_level: int):
    """Get detailed metrics for a specific portfolio"""
    try:
        portfolio = get_etf_portfolio(risk_level)
        report = generate_etf_report()
        metrics = report['performance_metrics'][risk_level]
        
        return {
            "portfolio_name": portfolio["name"],
            "metrics": {
                "expected_return": f"{metrics['return']:.2%}",
                "volatility": f"{metrics['volatility']:.2%}",
                "sharpe_ratio": f"{metrics['sharpe']:.2f}",
                "var_95": f"{metrics['var_95']:.2%}"
            },
            "allocation": portfolio["allocation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------- Chatbot ----------
@app.post("/chat")
def chat(input: ChatIn):
    msg = input.message.lower()
    if "risk" in msg:
        return {"reply": "We use a short questionnaire and your financial goals to assign a risk profile."}
    if "etf" in msg:
        return {"reply": "ETFs are chosen by AI clustering based on volatility, returns, and cost efficiency."}
    if "sustainable" in msg or "esg" in msg:
        return {"reply": "Yes, we can include ESG or sustainable ETFs if you prefer responsible investing."}
    if "performance" in msg:
        return {"reply": "Performance is based on 3-year historical returns and volatility metrics."}
    return {"reply": "I'm your robo-advisor assistant. Ask about risk, ETFs, ESG, or performance."}
