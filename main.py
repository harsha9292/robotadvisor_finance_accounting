from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, List, Dict
import os
import math
from fastapi.responses import FileResponse
import json 

# -----------------------------
# Import modules for ETF logic
# -----------------------------
from etf_allocations import (
    ETF_DATA,
    get_etf_candidates,
    get_etf_portfolio,
    refresh_etf_universe,
    calculate_score,
    determine_risk_profile
)

app = FastAPI(title="RoboAdvisor MVP")

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


# -----------------------------
# /assess endpoint
# -----------------------------
@app.post("/assess")
async def assess(inputs: RiskInput):
    try:
        input_dict = inputs.dict()

        # -----------------------------
        # Map incoming fields to score keys
        # -----------------------------
        score_mapping = {
            "age_range_score": ("age_range", "age_range"),
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

        # Populate score keys using get_score_from_value, default to 2 if missing
        for key, (field_name, _) in score_mapping.items():
            val = input_dict.get(field_name)
            input_dict[key] = get_score_from_value(val, field_name) if val is not None else 2

        # -----------------------------
        # Compute risk score & profile
        # -----------------------------
        score = calculate_score(input_dict)
        profile_data = determine_risk_profile(input_dict)
        risk_bucket = profile_data.get("risk_bucket", 5)
        risk_profile = profile_data.get("risk_profile", "Balanced")
        alloc_hint = profile_data.get("typical_allocation_hint", {"equity_pct": 50, "bond_pct": 50})

        # -----------------------------
        # ETF selection
        # -----------------------------
        etf_candidates = get_etf_candidates(
            region_pref=input_dict.get("regional_focus", "global"),
            sustainable_pref=input_dict.get("esg_preference", "no")
        )

        shortlisted = [
            e["ETF_Name"] for e in etf_candidates
            if isinstance(e, dict) and e.get("risk_level") == risk_bucket and "ETF_Name" in e
        ]

        allocation = {t: round(1/len(shortlisted), 2) for t in shortlisted} if shortlisted else {}
        etf_portfolio = get_etf_portfolio(risk_bucket, candidate_etfs=etf_candidates)
        if not etf_portfolio or not etf_portfolio.get("allocation"):
            etf_portfolio = {
                "allocation": allocation,
                "target_return": 0.05,
                "target_volatility": 0.1,
                "sharpe_ratio": 0.5,
                "name": f"Fallback Portfolio (Bucket {risk_bucket})",
                "description": "Simple equal weight allocation due to missing data."
            }

        # -----------------------------
        # Build summary
        # -----------------------------
        equity_pct = alloc_hint["equity_pct"] / 100
        bond_pct = alloc_hint["bond_pct"] / 100
        typical_alloc_text = f"{round(equity_pct*100)}% Equity / {round(bond_pct*100)}% Fixed Income"
        tv = float(etf_portfolio.get("target_volatility", 0))
        target_vol_text = f"{tv*100:.1f}%" if tv else "N/A"

        projections = build_projections(
            mu=float(etf_portfolio.get("target_return", 0)),
            sigma=float(etf_portfolio.get("target_volatility", 0)),
            years=5
        )

        summary = {
            "score": score["score"],
            "risk_profile": risk_profile,
            "risk_bucket": risk_bucket,
            "allocation": etf_portfolio.get("allocation", allocation),
            "shortlisted_etfs": shortlisted,
            "target_volatility_text": target_vol_text,
            "typical_allocation_text": typical_alloc_text,
            "projections": projections
        }

        LATEST_ASSESSMENTS["latest"] = {"score": score, **profile_data, "summary": summary}

        return {"status": "success", "summary": summary, "etf_portfolio": etf_portfolio}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing assessment: {str(e)}")

# -----------------------------
# /recommend/{bucket} endpoint
# -----------------------------
@app.get("/recommend/{bucket}")
async def recommend(bucket: int):
    try:
        portfolio = get_etf_portfolio(bucket) 
        if not portfolio:
            raise ValueError(f"No portfolio found for bucket {bucket}")

        alloc = portfolio.get("allocation", {})
        response = {
            "name": portfolio.get("name", f"Bucket {bucket} Portfolio"),
            "allocation": alloc,
            "target_return": float(portfolio.get("target_return", 0)),
            "target_volatility": float(portfolio.get("target_volatility", 0)),
            "sharpe_ratio": float(portfolio.get("sharpe_ratio", 0)),
            "risk_level": bucket,
        }

        # Include projections
        response["projections"] = build_projections(
            mu=float(response["target_return"]),
            sigma=float(response["target_volatility"]),
            years=5
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

# -----------------------------
# /etf-universe endpoint
# -----------------------------
@app.get("/etf-universe")
def get_etf_universe():
    return ETF_DATA

# ---------- Chatbot (Context-Aware) ----------
@app.post("/chat")
def chat(input: ChatIn, request: Request):
    """
    Context-aware educational chatbot:
    - Uses the latest assessment to personalise replies.
    - Provides risk score, risk bucket, portfolio allocations, and projections.
    """
    msg = input.message.lower().strip()
    assess = LATEST_ASSESSMENTS.get("latest")

    # helper to safely access fields
    def safe_field(obj, key, default=None):
        return obj.get(key, default) if obj else default

    if assess:
        user_risk_profile = safe_field(assess, "risk_profile", "Balanced")
        user_risk_bucket = safe_field(assess, "risk_bucket", 5)
        summary = safe_field(assess, "summary", {})
        etf_portfolio = safe_field(assess, "etf_portfolio", {})
        typical_alloc_text = safe_field(summary, "typical_allocation_text", None)
        projections = safe_field(summary, "projections", None)
        allocation = safe_field(etf_portfolio, "allocation", {})

    else:
        user_risk_profile = None
        user_risk_bucket = None
        typical_alloc_text = None
        projections = None
        allocation = {}

    # ---------- Risk Score & Profile ----------
    if any(k in msg for k in ["risk", "score", "category"]):
        if assess:
            return {
                "reply": (
                    f"Your risk score is **{user_risk_profile}**, mapped to risk bucket **{user_risk_bucket}**.\n\n"
                    f"Recommended portfolio targets roughly: **{typical_alloc_text or 'N/A'}**.\n\n"
                    "Risk categories:\n"
                    "- **Conservative (low):** mostly bonds, capital preservation.\n"
                    "- **Balanced (medium):** mix of equities & bonds.\n"
                    "- **Growth/Aggressive (high):** equity-heavy for higher long-term returns.\n\n"
                    "The score comes from your questionnaire (time horizon, emergency fund, income stability, experience, reaction to losses)."
                )
            }
        else:
            return {"reply": "I don't have an assessment for you yet — please complete the questionnaire."}

    # ---------- Portfolio Allocation ----------
    if any(k in msg for k in ["portfolio", "allocation", "portfolio mix"]):
        if assess:
            top_holdings = sorted(allocation.items(), key=lambda x: -float(x[1]))[:6]
            top_text_parts = [
                f"**{etf_name}**: {round(weight*100,1)}%" for etf_name, weight in top_holdings
            ]
            top_text = ", ".join(top_text_parts) if top_text_parts else "N/A"

            return {
                "reply": (
                    f"Your recommended portfolio (summary): {typical_alloc_text or 'N/A'}.\n"
                    f"Top holdings: {top_text}.\n\n"
                    "We select ETFs to diversify across regions and asset classes, weighted to match your risk bucket."
                )
            }
        else:
            return {"reply": "Complete the assessment first — then I can show the portfolio allocation."}

    # ---------- How Score is Calculated ----------
    if any(k in msg for k in ["how", "calculate", "determine"]) and ("score" in msg or "risk" in msg):
        return {
            "reply": (
                "We compute your risk score using rule-based points from your questionnaire: "
                "time horizon, emergency fund, income stability, investment experience, and reaction to losses. "
                "These map to a numeric score and a risk bucket, which determines portfolio allocation."
            )
        }

    # ---------- Projections ----------
    if any(k in msg for k in ["performance", "projection", "return"]):
        if projections and projections.get("median"):
            median_final = (projections["median"][-1] - 1) * 100
            return {
                "reply": (
                    f"Our model projects a **median 5-year growth** of about **{median_final:.1f}%** for your portfolio. "
                    "These are projections based on historical returns and volatility assumptions, not guarantees."
                )
            }
        else:
            return {"reply": "No projection data available yet — it will appear once the portfolio is built."}

    # ---------- ESG / Sustainable Queries ----------
    if any(k in msg for k in ["esg", "sustainable", "responsible"]):
        return {
            "reply": (
                "Yes — we can build ESG/sustainable ETF portfolios. They avoid certain sectors and favour companies "
                "with higher environmental, social, and governance standards, while maintaining diversification."
            )
        }

    # ---------- Fallback / Personalized ----------
    if assess:
        return {
            "reply": (
                f"I have your latest assessment: risk score **{user_risk_profile}**, bucket **{user_risk_bucket}**, "
                f"recommended split **{typical_alloc_text or 'N/A'}**. "
                "Ask me: 'Explain my score', 'Show portfolio', or 'Projection'."
            )
        }

    return {
        "reply": (
            "I’m MoneyMentorX — I can explain risk categories, how your score is calculated, portfolio allocations, and projections. "
            "Start by completing the questionnaire so I can personalise responses."
        )
    }
