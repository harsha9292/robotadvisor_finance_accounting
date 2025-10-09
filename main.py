from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import math
from fastapi.responses import FileResponse
import json 
import pandas as pd
def get_dynamic_etfs(input_dict: dict):
    """
    Get filtered ETFs based on user preferences.
    """
    preferences = {
        "region_pref": input_dict.get("regional_focus", "global"),
        "sustainable_pref": input_dict.get("esg_preference", "no"),
        "sectors_to_avoid": input_dict.get("sectors_to_avoid", [])
    }
    filtered_etfs = get_filtered_etfs(preferences)
    df = pd.DataFrame.from_records(filtered_etfs)
    return df

import numpy as np
import yfinance as yf

# -----------------------------
# Import modules
# -----------------------------
from etf_allocations import (
    ETF_DATA,
    get_filtered_etfs,
    get_etf_portfolio,
    refresh_etf_universe,
    calculate_score,
    determine_risk_profile,
    get_historical_returns
)
from api_endpoints import router as etf_router

app = FastAPI(title="RoboAdvisor MVP")
app.include_router(etf_router, prefix="/api/v1", tags=["ETF Management"])

@app.get("/")
def root():
    """Serve the frontend HTML file."""
    # Assuming 'screens/index.html' is correct based on prior context
    # Use FileResponse for serving static files
    return FileResponse(os.path.join("screens", "index.html"))

# -----------------------------
# Startup: refresh ETF data
# -----------------------------
@app.on_event("startup")
async def startup_event():
    try:
        # Load the external data/universe of ETFs
        refresh_etf_universe()
        print(f"Startup: Loaded {len(ETF_DATA)} ETF categories.")
    except Exception as e:
        print(f"Startup error: {e}")

# -----------------------------
# Models: FIXED TO MATCH INCOMING PAYLOAD FIELDS
# -----------------------------
class RiskInput(BaseModel):
    # Core Risk & Goals
    age_range: str
    annual_income: str
    investment_goal: str
    investment_horizon: str
    
    # Financial Capacity
    income_stability: str
    liabilities_ratio: str
    emergency_fund: str
    monthly_savings: str
    initial_investment: str
    
    # Behavior & Attitude
    reaction_to_loss: str
    risk_preference: str
    volatility_feeling: str
    investing_experience: str
    investment_approach: str
    
    # Tech & Preferences
    tech_usage: str
    trust_ai: str
    regional_focus: str 
    esg_preference: str 
    sectors_to_avoid: Optional[List[str]] = []

class ChatIn(BaseModel):
    message: str

# -----------------------------
# Helper: GBM projections
# -----------------------------
def build_projections(mu: float, sigma: float, years: int = 5) -> dict:
    """Calculates geometric Brownian motion projections for portfolio growth."""
    z10, z90 = -1.28155, 1.28155 # Z-scores for 10th and 90th percentile
    years_list = list(range(1, years + 1))
    median, p10, p90 = [], [], []
    try:
        for t in years_list:
            # Median/Expected Growth
            m = math.exp((mu - 0.5 * sigma**2) * t)
            # 10th Percentile (Conservative)
            q10 = math.exp((mu - 0.5 * sigma**2) * t + z10 * sigma * math.sqrt(t))
            # 90th Percentile (Optimistic)
            q90 = math.exp((mu - 0.5 * sigma**2) * t + z90 * sigma * math.sqrt(t))
            median.append(m)
            p10.append(q10)
            p90.append(q90)
    except Exception as e:
        print(f"Projection calculation error: {e}")
        return {"years": years_list, "median": [], "p10": [], "p90": []}
    return {"years": years_list, "median": median, "p10": p10, "p90": p90}

# -----------------------------
# Global: latest assessment
# -----------------------------
LATEST_ASSESSMENTS: Dict[str, dict] = {}

# -----------------------------
# NEW HELPER: Maps verbose answers to score (1, 2, or 3)
# -----------------------------
def get_score_from_value(value: str | float | int, field_name: str) -> int:
    """
    Translates verbose questionnaire answers into a numerical risk score (1, 2, or 3).
    1 = Low Risk Answer (Conservative), 3 = High Risk Answer (Aggressive/High Capacity).
    """
    
    # Handle numeric input (not used with the new Pydantic model)
    if isinstance(value, (int, float)):
        return 2

    # Handle string input (verbose answers)
    v = str(value).lower().strip()
    
    # --- Explicit Scoring for Fixed Fields ---
    
    # 1. Age Range (Younger = Higher Capacity/Risk)
    if field_name == "age_range":
        if "18–24 years" in v or "25–34 years" in v: return 3 # High Capacity
        if "55–64 years" in v or "65+ years" in v: return 1 # Low Capacity
        return 2 # Medium Capacity (35-54 years)

    # 2. Liabilities Ratio (Higher percentage -> Lower Capacity Score)
    if field_name == "liabilities_ratio":
        if "less than 20%" in v: return 3
        if "more than 40%" in v: return 1
        return 2

    # 3. Monthly Savings (Higher percentage -> Higher Capacity Score)
    if field_name == "monthly_savings":
        if "15% or more" in v or "10-15%" in v: return 3
        if "< 5% of income" in v: return 1
        return 2
        
    # --- Generic String Scoring ---
    
    # High-Risk indicators (Score 3)
    high_risk_keys = ["growth", "7+ years", "10+ years", "invest more", "very stable", "significant", "experienced", "active investor", "take the riskier", "excited", "yes", "regularly", "lump sum and recurring"]
    for key in high_risk_keys:
        if key in v:
            return 3

    # Low-Risk indicators (Score 1)
    low_risk_keys = ["preservation", "< 3 years", "sell immediately", "unstable", "none or minimal", "< €1,000", "beginner", "always the safer", "nervous", "no", "rarely", "one-time investment"]
    for key in low_risk_keys:
        if key in v:
            return 1
            
    # Default to Medium Risk (Score 2) 
    return 2


def sanitize_for_json(obj):
    """
    Recursively replace np.nan, float('nan'), inf, -inf with 0.0.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    else:
        return obj
# ETF data is now managed through ETF_DATA in etf_allocations.py

def get_selected_etfs(input_dict: dict):
    """
    Return ETF tickers filtered by user preferences (region + ESG).
    """
    # Use get_filtered_etfs from etf_allocations
    preferences = {
        "region_pref": input_dict.get("regional_focus", "global"),
        "sustainable_pref": input_dict.get("esg_preference", "no")
    }
    filtered_etfs = get_filtered_etfs(preferences)
    return [etf['Ticker'] for etf in filtered_etfs if etf.get('Ticker')]

# Function moved to etf_allocations.py


REGION_MAPPING = {
    "global": "global",
    "europe (eu)": "europe",
    "u.s. (us)": "us",
    "no preference": "global"
}


def dynamic_allocation(df: pd.DataFrame, risk_score: float):
    """Calculate dynamic allocation based on risk score and available ETFs."""
    # Risk score 1–10 → equity %: 20–80, bonds = 100 - equity
    equity_pct = min(max((risk_score / 10) * 80, 20), 80)
    bond_pct = 100 - equity_pct

    # Use correct column names from filtered DataFrame
    equity_etfs = df[df['asset_type'].str.contains("Equity|World|Emerging|US|Europe", case=False, na=False)]
    bond_etfs = df[df['asset_type'].str.contains("Bond|Treasury", case=False, na=False)]

    allocation = {}
    if not equity_etfs.empty:
        eq_weight = equity_pct / len(equity_etfs)
        for _, row in equity_etfs.iterrows():
            ticker = row['Ticker']
            if ticker:
                allocation[ticker] = round(eq_weight / 100, 4)  # store as fraction

    if not bond_etfs.empty:
        bond_weight = bond_pct / len(bond_etfs)
        for _, row in bond_etfs.iterrows():
            ticker = row['Ticker']
            if ticker:
                allocation[ticker] = round(bond_weight / 100, 4)  # store as fraction

    return allocation


def calculate_var(mu: float, sigma: float, confidence: float = 0.95, horizon_years: int = 1):
    """
    Calculate 1-year Value at Risk (VaR) at a given confidence level using a normal approximation.
    mu: expected annual return (decimal)
    sigma: annual volatility (decimal)
    confidence: confidence level (default 0.95)
    horizon_years: time horizon in years (default 1)
    """
    from scipy.stats import norm
    z = norm.ppf(1 - confidence)
    # VaR formula: μ - zσ
    var = mu + z * sigma * math.sqrt(horizon_years)
    var_pct = (1 - math.exp(var)) * 100  # convert to percent loss
    return round(abs(var_pct), 2)

# -----------------------------
# /assess endpoint (dynamic allocation)
# -----------------------------
@app.post("/assess")
async def assess(inputs: RiskInput):
    try:
        input_dict = inputs.dict()

        # -----------------------------
        # Map questionnaire answers to scores
        # -----------------------------
        score_mapping = {
            "age_range_score": ("age_range", "age_range"),
            "annual_income_score": ("annual_income", "annual_income"),
            "investment_goal_score": ("investment_goal", "investment_goal"),
            "market_loss_scenario_score": ("reaction_to_loss", "reaction_to_loss"),
            "risk_vs_reward_score": ("risk_preference", "risk_preference"),
            "feelings_about_volatility_score": ("volatility_feeling", "volatility_feeling"),
            "investment_experience_score": ("investing_experience", "investing_experience"),
            "income_stability_score": ("income_stability", "income_stability"),
            "liabilities_score": ("liabilities_ratio", "liabilities_ratio"),
            "emergency_fund_score": ("emergency_fund", "emergency_fund"),
            "monthly_savings_score": ("monthly_savings", "monthly_savings"),
            "initial_investment_score": ("initial_investment", "initial_investment"),
            "investment_horizon_score": ("investment_horizon", "investment_horizon"),
            "investment_approach_score": ("investment_approach", "investment_approach"),
            "app_experience_score": ("tech_usage", "tech_usage"),
            "ai_advisor_score": ("trust_ai", "trust_ai")
        }

        for key, (field_name, _) in score_mapping.items():
            val = input_dict.get(field_name)
            input_dict[key] = get_score_from_value(val, field_name) if val is not None else 2

        print("\n--- QUESTION SCORES ---")
        for k in score_mapping.keys():
            print(f"{k}: {input_dict[k]}")

        # -----------------------------
        # Compute total score
        # -----------------------------
        score_data = calculate_score(input_dict)
        total_score = score_data["score"]
        print(f"\nTotal risk score: {total_score}")

        # -----------------------------
        # Select ETFs dynamically based on preferences
        # -----------------------------
        df_filtered = get_dynamic_etfs(input_dict)
        print(f"\nFiltered ETFs based on region & ESG: {len(df_filtered)} ETFs")
        #print(df_filtered[['ETF_Name', 'Ticker_ISIN', 'Asset_Type']].head(10))


        tickers = df_filtered['Ticker'].tolist()
        returns_df = get_historical_returns(tickers)

        if not returns_df.empty:
            profile_data = determine_risk_profile(input_dict, returns_df.values)
        else:
            # fallback if no price data available
            profile_data = {
                "risk_bucket": 5,
                "risk_profile": "Balanced Growth",
                "portfolio_expected_return": 0.06,
                "portfolio_volatility": 0.1,
                "typical_allocation_hint": {"equity_pct": 60, "bond_pct": 40}
            }

        # -----------------------------
        # Dynamic allocation based on risk score
        # -----------------------------
        # Map total_score (6–18) linearly to equity % (20–90)
        equity_target = min(max((total_score - 6) / (18 - 6) * 0.7 + 0.2, 0.2), 0.9)
        bond_target = 1 - equity_target
        print(f"\nEquity target: {equity_target*100:.1f}%, Bond target: {bond_target*100:.1f}%")

        # Separate ETFs by type
        equity_etfs = df_filtered[df_filtered['asset_type'].str.contains("Equity|World|US|Europe|Emerging", case=False)]
        bond_etfs = df_filtered[df_filtered['asset_type'].str.contains("Bond|Treasury", case=False)]

        allocation = {}

        if not equity_etfs.empty:
            eq_weight = equity_target / len(equity_etfs)
            for t in equity_etfs['Ticker']:
                allocation[t] = eq_weight

        if not bond_etfs.empty:
            bond_weight = bond_target / len(bond_etfs)
            for t in bond_etfs['Ticker']:
                allocation[t] = bond_weight

        # Normalize allocation to sum to 1 exactly
        total_alloc = sum(allocation.values())
        if total_alloc > 0:
            for k in allocation:
                allocation[k] /= total_alloc

        print("\n--- DYNAMIC ETF ALLOCATION ---")
        for ticker, weight in allocation.items():
            print(f"{ticker}: {weight*100:.2f}%")
        if not allocation:
            print("No ETFs matched the selection criteria.")

        # -----------------------------
        # Portfolio metrics (placeholder, can replace with historical returns later)
        # -----------------------------
        portfolio_expected_return = 0.05 + equity_target * 0.05  # crude linear approx
        portfolio_volatility = 0.08 + equity_target * 0.07
        sharpe_ratio = (portfolio_expected_return - 0.01) / portfolio_volatility
        var_95 = calculate_var(
            mu=portfolio_expected_return,
            sigma=portfolio_volatility,
            confidence=0.95,
            horizon_years=1
        )
        
        portfolio_name = profile_data['risk_profile']
        etf_portfolio = {
            "name": portfolio_name,
            "description": f"{portfolio_name} built dynamically based on your risk score and investment preferences.",
            "allocation": allocation,
            "target_return": round(portfolio_expected_return, 4),
            "target_volatility": round(portfolio_volatility, 4),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "value_at_risk": var_95 

        }

        # -----------------------------
        # Projections
        # -----------------------------
        projections = build_projections(
            mu=portfolio_expected_return,
            sigma=portfolio_volatility,
            years=5
            
        )



        # -----------------------------
        # Build summary
        # -----------------------------

        
        summary = {
            "score": total_score,
            "risk_profile": profile_data['risk_profile'],
            "risk_bucket": profile_data['risk_bucket'],
            "allocation": allocation,
            "shortlisted_etfs": df_filtered['ETF_Name'].tolist(),
            "target_volatility_text": f"{portfolio_volatility*100:.1f}%",
            "typical_allocation_text": f"{round(equity_target*100)}% Equity / {round(bond_target*100)}% Fixed Income",
            "projections": projections,
            "portfolio_expected_return": portfolio_expected_return,
            "portfolio_volatility": portfolio_volatility,
            "value_at_risk": var_95 
        }

        # -----------------------------
        # Store latest assessment
        # -----------------------------
                # -----------------------------
        # Store latest assessment (auto-updates chatbot)
        # -----------------------------
        LATEST_ASSESSMENTS["latest"] = {
            "score": total_score,
            **score_data,
            **profile_data,
            "summary": summary,
            "etf_portfolio": etf_portfolio
        }

        return {"status": "success", "summary": summary, "etf_portfolio": etf_portfolio}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing assessment: {str(e)}")

# Recommendation and ETF Universe endpoints moved to api_endpoints.py

# ---------- Chatbot (Context-Aware) ----------
@app.post("/chat")
def chat(input: ChatIn, request: Request):
    """
    Context-aware educational chatbot:
    - Uses the latest assessment to personalise replies.
    - Provides risk score, risk bucket, portfolio allocations, and projections.
    - Explains per-field contribution to the total score.
    """

    msg = input.message.lower().strip()
    assess = LATEST_ASSESSMENTS.get("latest")

    # Safe accessor
    def safe_field(obj, key, default=None):
        return obj.get(key, default) if obj else default

    if assess:
        profile_data = assess
        summary = safe_field(assess, "summary", {})
        etf_portfolio = safe_field(assess, "etf_portfolio", {})
        user_risk_profile = safe_field(profile_data, "risk_profile", "Balanced")
        user_risk_bucket = safe_field(profile_data, "risk_bucket", 5)
        typical_alloc_text = safe_field(summary, "typical_allocation_text", "50% Equity / 50% Fixed Income")
        allocation = safe_field(etf_portfolio, "allocation", {})
        projections = safe_field(summary, "projections", None)
        field_scores = {k: safe_field(profile_data, k) for k in profile_data if "_score" in k}
    else:
        profile_data = {}
        summary = {}
        etf_portfolio = {}
        user_risk_profile = None
        user_risk_bucket = None
        typical_alloc_text = None
        allocation = {}
        projections = None
        field_scores = {}

    # ---------- Respond only if message matches known intents ----------
    if not msg:
        return {"reply": "Hello! I'm MoneyMentorX. Complete the questionnaire and I'll analyse your profile and allocation."}

    # ---------- Risk Score & Profile ----------
    if any(k in msg for k in ["risk", "score", "category"]):
        if profile_data:
            return {
                "reply": (
                    "Let's break down your investment profile step by step:\n\n"
                    "1. **Risk Score & Profile:**\n"
                    f"   - Based on your answers, your risk profile is **{user_risk_profile}** (bucket {user_risk_bucket}).\n"
                    "   - This score reflects your comfort with risk, your financial situation, and your investment goals.\n"
                    "   - A higher score means you can take more risk for higher potential returns.\n\n"
                    "2. **What does this mean?**\n"
                    "   - Conservative: Focus on safety, mostly bonds.\n"
                    "   - Balanced: Mix of stocks and bonds for steady growth.\n"
                    "   - Growth: More stocks, aiming for higher long-term returns.\n\n"
                    f"3. **Your recommended portfolio split:** {typical_alloc_text}\n\n"
                    "Next, ask about the graph or ETF allocation for more details!"
                )
            }
        else:
            return {"reply": "I don't have an assessment for you yet — please complete the questionnaire."}

    # ---------- Explain Score per Field ----------
    if any(k in msg for k in ["explain", "breakdown", "details"]) and ("score" in msg or "risk" in msg):
        if field_scores:
            explanation_lines = []
            for field, score in field_scores.items():
                pretty_field = field.replace("_score", "").replace("_", " ").title()
                explanation_lines.append(f"- {pretty_field}: {score} points")
            total_score = safe_field(profile_data, "score", sum(field_scores.values()))
            reply_text = (
                "**How your risk score is built:**\n\n"
                "Each answer you gave adds to your total score. For example, longer time horizons, higher savings, and more experience increase your score.\n\n"
                f"Your total risk score: **{total_score}**\n\n"
                "Breakdown by question:\n" +
                "\n".join(explanation_lines)
            )
            return {"reply": reply_text}
        else:
            return {"reply": "No detailed score breakdown is available yet — complete the assessment first."}

    # ---------- Portfolio Allocation & ETF Sectors ----------
    if any(k in msg for k in ["portfolio", "allocation", "portfolio mix", "etf"]):
        if allocation:
            top_holdings = sorted(allocation.items(), key=lambda x: -float(x[1]))[:6]
            top_text_parts = [
                f"{etf_name}: {round(weight*100,1)}%" for etf_name, weight in top_holdings
            ]
            top_text = ", ".join(top_text_parts) if top_text_parts else "N/A"
            # Try to get sector info if present in summary or etf_portfolio
            sector_info = ""
            sectors = summary.get("sectors") or etf_portfolio.get("sectors")
            if sectors:
                if isinstance(sectors, dict):
                    sector_list = [f"{k} ({v}%)" for k, v in sectors.items()]
                    sector_info = "\n- Sectors invested: " + ", ".join(sector_list)
                elif isinstance(sectors, list):
                    sector_info = "\n- Sectors invested: " + ", ".join(sectors)
                else:
                    sector_info = f"\n- Sectors invested: {sectors}"
            return {
                "reply": (
                    "**ETF Portfolio Allocation:**\n\n"
                    "- Your portfolio is made up of different ETFs (funds that track stocks or bonds).\n"
                    f"- The recommended split is: {typical_alloc_text}\n"
                    f"- Top holdings: {top_text}{sector_info}\n\n"
                    "We choose a mix of ETFs to spread your risk and match your profile.\n\n"
                    "ETFs are like baskets of investments, so you get instant diversification!"
                )
            }
        else:
            return {"reply": "Complete the assessment first — then I can show the portfolio allocation."}
    # ---------- Key Metrics Explanation ----------
    if any(k in msg for k in ["key metrics", "metrics", "annual return", "volatility", "sharpe", "value at risk", "var"]):
        return {
            "reply": (
                "**Key Portfolio Metrics Explained:**\n\n"
                "- **Annual Return:** The average amount your portfolio could grow each year, based on history. Higher is better, but not guaranteed.\n"
                "- **Volatility:** How much your portfolio value might go up and down. Higher volatility means bigger swings (more risk, but also more reward).\n"
                "- **Sharpe Ratio:** This tells you how much return you get for each unit of risk. A higher Sharpe ratio means a better risk/reward balance.\n"
                "- **Value at Risk (VaR):** The most you might lose in a bad year, with 95% confidence. For example, a VaR of 10% means there's only a 5% chance you'll lose more than 10% in a year.\n\n"
                "These numbers help you compare portfolios and understand what to expect!"
            )
        }

    # ---------- How Score is Calculated ----------
    if any(k in msg for k in ["detail", "determine"]) and ("score" in msg or "risk" in msg):
        if not profile_data:
            return {"reply": "I don’t have your assessment yet — please complete the questionnaire first."}

        regional_focus = safe_field(profile_data, "regional_focus", "Global")
        esg_pref = safe_field(profile_data, "esg_preference", "No preference")
        return {
            "reply": (
                "We compute your risk score using rule-based points from your questionnaire. "
                "Each field contributes as follows:\n\n"
                "- **Time horizon / Investment horizon:** longer horizons allow higher risk.\n"
                "- **Emergency fund / Income stability / Liabilities:** stronger financial capacity allows higher risk.\n"
                "- **Investment experience / Risk preference / Reaction to losses:** higher experience or comfort with losses increases the score.\n"
                "- **Monthly savings / Initial investment:** higher amounts allow more risk.\n"
                "- **Tech usage / Trust in AI:** can indicate comfort with modern investment tools.\n"
                f"- **Regional Focus:** your selection (currently **{regional_focus}**) determines which ETFs are included in your portfolio.\n"
                f"- **ESG Preference:** selecting sustainable ETFs filters the universe accordingly.\n\n"
                "All these factors combine to produce a numeric risk score, mapped to a risk bucket, which then informs portfolio allocation."
            )
        }

    # ---------- Projections & Graph Explanation ----------
    if any(k in msg for k in ["performance", "projection", "return", "graph", "chart", "explain graph", "explain chart"]):
        if projections and projections.get("median"):
            median_final = (projections["median"][-1] - 1) * 100
            reply = (
                "**Understanding Your Investment Chart:**\n\n"
                "- The chart shows how your money could grow over 5 years.\n"
                "- The blue lines are your investment projections: the solid line is the most likely path, and the dashed lines show possible best and worst cases.\n"
                "- The green line shows how much more your savings could grow if you invest a set percentage of your income (like 2%, 10%, or 20%) in the portfolio, compared to just saving the same amount as cash.\n\n"
                "**How to read it:**\n"
                "- If the green line is at 30% after 5 years, it means investing your savings could give you 30% more than just keeping it as cash.\n"
                "- The chart helps you see the benefit of investing regularly versus just saving, and what kind of growth you might expect.\n\n"
                f"Our model projects a median 5-year growth of about {median_final:.1f}% for your portfolio.\n\n"
                "Remember: These are estimates, not guarantees. Markets can go up and down!"
            )
            return {"reply": reply}
        else:
            return {"reply": "No projection data available yet — it will appear once the portfolio is built."}

    # ---------- ESG / Sustainable Queries ----------
    if any(k in msg for k in ["esg", "sustainable", "responsible"]):
        return {
            "reply": (
                "**Sustainable Investing:**\n\n"
                "If you choose ESG (Environmental, Social, Governance) options, your portfolio will focus on companies and funds that are better for the planet and society, while still aiming for good returns.\n\n"
                "You can invest responsibly and still be diversified!"
            )
        }

    # ---------- Fallback ----------
    return {
        "reply": (
            "Hi! I'm MoneyMentorX. Here's how I help you:\n\n"
            "1. I ask you questions to understand your goals and risk comfort.\n"
            "2. I calculate your risk score and recommend a portfolio split.\n"
            "3. I show you a chart of how your money could grow, and explain the difference between just saving and investing.\n"
            "4. I pick a mix of ETFs (funds) to match your profile.\n"
            "5. I explain key metrics like annual return, volatility, Sharpe ratio, and Value at Risk.\n\n"
            "Ask me about your risk score, the chart, your ETF allocation, or key metrics for more details!"
        )
    }
