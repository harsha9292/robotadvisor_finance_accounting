from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from risk_scoring import calculate_score, determine_risk_profile
from etf_allocations import get_etf_portfolio

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
    risk_tolerance: int  # 1-5
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
@app.get("/recommend/{bucket}")
def recommend(bucket: int):
    if ETF_DF.empty:
        return {"cluster": None, "etfs": {}}
    # Map risk bucket to cluster (adjust as needed)
    cluster_map = {1:0, 2:1, 3:2, 4:3}
    cluster = cluster_map.get(bucket, 0)
    subset = ETF_DF[ETF_DF.cluster == cluster].to_dict(orient="index")
    return {"cluster": int(cluster), "etfs": subset}

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
