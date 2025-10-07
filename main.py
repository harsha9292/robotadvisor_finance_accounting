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
import json
import requests
import openai

#openai.api_key = "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"
#openai.api_key = os.getenv("OPENAI_API_KEY")
HF_API_TOKEN = "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"
 
import math

class ChatIn(BaseModel):
    message: str

def build_projections(mu: float, sigma: float, years: int = 5) -> dict:
    """Return projection series for 1..years using geometric Brownian motion quantiles.
    mu and sigma are annualized decimal rates (e.g., 0.02, 0.03).
    Returns factors (growth multiples) for median, 10th and 90th percentiles.
    """
    try:
        z10 = -1.28155
        z90 = 1.28155
        years_list = list(range(1, years + 1))
        median = []
        p10 = []
        p90 = []
        for t in years_list:
            m = math.exp((mu - 0.5 * sigma * sigma) * t)
            q10 = math.exp((mu - 0.5 * sigma * sigma) * t + z10 * sigma * math.sqrt(t))
            q90 = math.exp((mu - 0.5 * sigma * sigma) * t + z90 * sigma * math.sqrt(t))
            median.append(m)
            p10.append(q10)
            p90.append(q90)
        return {"years": years_list, "median": median, "p10": p10, "p90": p90}
    except Exception:
        return {"years": list(range(1, years + 1)), "median": [], "p10": [], "p90": []}

from risk_scoring import calculate_score, determine_risk_profile
from etf_allocations import (
    get_etf_portfolio, 
    ETF_UNIVERSE, 
    ETF_PORTFOLIOS,
    ETFMetrics,
    refresh_etf_universe,
    fetch_historical_data,
    initialize_portfolios,
    PortfolioAllocation
)
from portfolio_analysis import generate_etf_report, backtest_portfolio


app = FastAPI(title="RoboAdvisor MVP")
app.mount("/screens", StaticFiles(directory="screens", html=True), name="frontend")

# Initialize ETF data at startup
@app.on_event("startup")
async def startup_event():
    try:
        # Minimal startup initialization - avoid verbose terminal output
        refresh_etf_universe()
        if not ETF_UNIVERSE or not ETF_PORTFOLIOS:
            raise ValueError("ETF data not properly initialized")
        
    except Exception as e:
        # Only output error message when startup fails
        print(f"Startup error: {str(e)}")

        if not ETF_UNIVERSE:
            # initialize quietly
            ETF_UNIVERSE.update({
                'SPY': ETFMetrics(
                    ticker='SPY',
                    name='SPDR S&P 500 ETF',
                    net_assets=380.0,
                    avg_volume=70.0,
                    expense_ratio=0.0009,
                    historical_return=0.105,
                    volatility=0.14,
                    min_investment=1000
                )
            })

        if not ETF_PORTFOLIOS:
            ETF_PORTFOLIOS.update(initialize_portfolios())

    # Final minimal validation log
    print(f"Startup: ETF universe={len(ETF_UNIVERSE)}, portfolios={len(ETF_PORTFOLIOS)}")


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
async def assess(inputs: RiskInput):
    try:
        # Minimal logging: compute score and profile
        input_dict = inputs.dict()
        score = calculate_score(input_dict)
        profile_data = determine_risk_profile(score)
        # Build a baseline summary from the risk profile so frontend always has something consistent
        try:
            base_tv = profile_data.get('target_volatility_pct_range', [0, 5])
            hint = profile_data.get('typical_allocation_hint', {'equity_pct': 20, 'bond_pct': 80})
            baseline_summary = {
                'score': score,
                'risk_bucket': profile_data.get('risk_bucket'),
                'target_volatility_text': f"{base_tv[0]}–{base_tv[1]}%",
                'typical_allocation_text': f"{hint.get('equity_pct')}% Equity / {hint.get('bond_pct')}% Bond",
                'allocation': {},
                'shortlisted_etfs': []
            }
        except Exception:
            baseline_summary = {'score': score, 'risk_bucket': profile_data.get('risk_bucket')}
        # Ensure the assessment score and bucket are always logged for visibility
        try:
            print(f"Assessment: Score={score}, RiskBucket={profile_data.get('risk_bucket')}")
        except Exception:
            pass

        # Ensure ETF data is available
        if not ETF_UNIVERSE:
            refresh_etf_universe()

        if not ETF_UNIVERSE:
            raise ValueError("Failed to initialize ETF universe")

        risk_bucket = profile_data.get("risk_bucket")
        if not risk_bucket:
            raise ValueError("No risk bucket determined")

        etf_portfolio = get_etf_portfolio(risk_bucket)
        if not etf_portfolio:
            raise ValueError(f"No portfolio found for risk bucket {risk_bucket}")

    # Build a concise summary for frontend display (attempt to enrich baseline_summary)
        try:
            alloc = etf_portfolio.get('allocation', {})
            converted_alloc = {k: float(v) for k, v in alloc.items()}
            bond_tickers = {"AGG", "BND", "BNDX", "TLT"}
            equity_pct = sum(v for k, v in converted_alloc.items() if k not in bond_tickers)
            bond_pct = 1.0 - equity_pct
            typical_allocation_text = f"{equity_pct:.0%} Equity / {bond_pct:.0%} Bond"
            tv = float(etf_portfolio.get('target_volatility', 0.0))
            if tv <= 0.05 and tv > 0:
                target_vol_text = "0–5%"
            else:
                target_vol_text = f"{tv:.1%}"
            summary = {
                "score": score,
                "risk_bucket": risk_bucket,
                "target_volatility_text": target_vol_text,
                "typical_allocation_text": typical_allocation_text,
                "allocation": converted_alloc,
                "shortlisted_etfs": list(converted_alloc.keys())
            }
            # attach simple projections (5-year) to the summary using GBM approximation
            try:
                mu = float(etf_portfolio.get('target_return', 0))
                sigma = float(etf_portfolio.get('target_volatility', 0))
                summary['projections'] = build_projections(mu, sigma, years=5)
            except Exception:
                summary['projections'] = {}
            print(f"Score={score}, RiskBucket={risk_bucket}, TargetVol={target_vol_text}")
            print(f"TypicalAllocation={typical_allocation_text}")
            print(f"ShortlistedETFs={list(converted_alloc.keys())}")
        except Exception:
            # Fall back to baseline summary built earlier
            summary = baseline_summary

        return {
            "score": score,
            **profile_data,
            "etf_portfolio": etf_portfolio,
            "summary": summary,
            "status": "success"
        }
    except Exception as e:
        print(f"Error in assessment: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing risk assessment: {str(e)}"
        )

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
async def recommend(bucket: int):
    try:
        portfolio = get_etf_portfolio(bucket)
        if not portfolio:
            raise ValueError(f"No portfolio found for bucket {bucket}")

        # Convert numpy values to Python native types
        converted_allocation = {k: float(v) for k, v in portfolio["allocation"].items()}

        response = {
            "name": portfolio["name"],
            "description": portfolio["description"],
            "target_return": float(portfolio['target_return']),
            "target_volatility": float(portfolio['target_volatility']),
            "allocation": converted_allocation,
            "sharpe_ratio": float(portfolio['sharpe_ratio']),
            "risk_level": bucket
        }

        # Try to include per-ETF historical series and simple metrics for frontend charts
        try:
            hist_df = fetch_historical_data()
            etfs = {}
            for tk in converted_allocation.keys():
                if hist_df is not None and tk in hist_df.columns:
                    ser = hist_df[tk].dropna()
                    history = {str(d.date()): float(v) for d, v in ser.items()}
                    returns = ser.pct_change().dropna()
                    if len(returns) > 0:
                        ann_return = float((1 + returns.mean()) ** 252 - 1)
                        vol = float(returns.std() * (252 ** 0.5))
                    else:
                        ann_return = 0.0
                        vol = 0.0
                    etfs[tk] = {"ann_return": ann_return, "volatility": vol, "history": history}
                else:
                    etfs[tk] = {"ann_return": None, "volatility": None, "history": {}}
            response["etfs"] = etfs
        except Exception as e:
            print(f"Warning: could not attach ETF histories to response: {e}")

        # Build and attach summary for frontend
        try:
            bond_tickers = {"AGG", "BND", "BNDX", "TLT"}
            equity_pct = sum(v for k, v in converted_allocation.items() if k not in bond_tickers)
            bond_pct = 1.0 - equity_pct
            typical_allocation_text = f"{equity_pct:.0%} Equity / {bond_pct:.0%} Bond"
            tv = float(response.get('target_volatility', 0.0))
            target_vol_text = "0–5%" if (tv <= 0.05 and tv > 0) else f"{tv:.1%}"
            summary = {
                "risk_bucket": bucket,
                "target_volatility_text": target_vol_text,
                "typical_allocation_text": typical_allocation_text,
                "allocation": converted_allocation,
                "shortlisted_etfs": list(converted_allocation.keys())
            }
            # add projections based on portfolio expected return/vol
            try:
                mu = float(response.get('target_return', 0))
                sigma = float(response.get('target_volatility', 0))
                summary['projections'] = build_projections(mu, sigma, years=5)
            except Exception:
                summary['projections'] = {}
            response["summary"] = summary
            print(f"Recommend: RiskBucket={bucket}, TargetVol={target_vol_text}")
            print(f"TypicalAllocation={typical_allocation_text}")
            print(f"ShortlistedETFs={list(converted_allocation.keys())}")
        except Exception:
            pass

        return response
    except Exception as e:
        print(f"Error in recommend endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

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
# @app.post("/chat")
# def chat(input: ChatIn):
#     msg = input.message.lower()
#     if "risk" in msg:
#         return {"reply": "We use a short questionnaire and your financial goals to assign a risk profile."}
#     if "etf" in msg:
#         return {"reply": "ETFs are chosen by AI clustering based on volatility, returns, and cost efficiency."}
#     if "sustainable" in msg or "esg" in msg:
#         return {"reply": "Yes, we can include ESG or sustainable ETFs if you prefer responsible investing."}
#     if "performance" in msg:
#         return {"reply": "Performance is based on 3-year historical returns and volatility metrics."}
#     return {"reply": "I'm your robo-advisor assistant. Ask about risk, ETFs, ESG, or performance."}

HF_MODEL = "bigscience/bloom"
@app.post("/chat")
def chat(input: ChatIn):
    if not HF_API_TOKEN:
        raise HTTPException(status_code=500, detail="HF_API_TOKEN not set")

    prompt = input.message

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": 150},
        "options": {"use_cache": False, "wait_for_model": True}
    }

    try:
        response = requests.post(
            f"https://api-inference.huggingface.co/models/{HF_MODEL}",
            headers=headers,
            json=payload,
            timeout=60,
            verify=False
        )
        print(response.status_code, response.text)  # <-- Debug line
        response.raise_for_status()
        data = response.json()
        reply = data[0]["generated_text"] if isinstance(data, list) else str(data)
        return {"reply": reply}

    except requests.exceptions.RequestException as e:
        print("Request Exception:", e)  # <-- Debug line
        raise HTTPException(status_code=500, detail=f"HuggingFace API error: {str(e)}")
